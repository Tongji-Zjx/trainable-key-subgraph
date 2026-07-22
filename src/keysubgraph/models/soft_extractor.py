"""Differentiable node/edge scoring and signed soft-graph classification."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.graph_features import GraphFeatureBuilder, GraphTimepointFeatures
from keysubgraph.theory import (
    DifferentiableGWLoss,
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
)


@dataclass(frozen=True)
class SoftExtractorConfig:
    node_feature_dim: int = 13
    node_score_hidden_dim: int = 32
    edge_score_hidden_dim: int = 32
    graph_hidden_dim: int = 64
    graph_layers: int = 2
    classifier_hidden_dim: int = 32
    dropout: float = 0.1
    epsilon: float = 1e-8
    theory_alignment_enabled: bool = False
    theory_alignment_mode: str = "strong"
    laplacian_eta: float = 1.0e-4
    heat_kernel_t: float = 0.5
    spectral_quantile_grid: Tuple[float, ...] = (
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
    )
    gw_mode: str = "gw"
    gw_entropic_reg: float = 5.0e-2
    gw_max_iter: int = 20
    gw_tolerance: float = 1.0e-7
    gw_sinkhorn_iter: int = 20
    gw_failure_strategy: str = "use_last"
    node_measure: str = "uniform"

    def __post_init__(self) -> None:
        integer_values = (
            self.node_feature_dim,
            self.node_score_hidden_dim,
            self.edge_score_hidden_dim,
            self.graph_hidden_dim,
            self.graph_layers,
            self.classifier_hidden_dim,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("model dimensions and graph_layers must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.theory_alignment_mode != "strong":
            raise ValueError("only strong theory alignment is supported")
        if self.laplacian_eta <= 0.0 or self.heat_kernel_t <= 0.0:
            raise ValueError("laplacian_eta and heat_kernel_t must be positive")
        if self.gw_mode != "gw":
            raise ValueError("the current theorem requires gw_mode='gw'")
        if self.gw_entropic_reg <= 0.0 or self.gw_tolerance <= 0.0:
            raise ValueError("GW regularization and tolerance must be positive")
        if self.gw_max_iter < 1 or self.gw_sinkhorn_iter < 1:
            raise ValueError("GW iteration counts must be positive")
        if self.gw_failure_strategy not in ("use_last", "raise"):
            raise ValueError("unsupported GW failure strategy")
        if self.node_measure != "uniform":
            raise ValueError("strong theory alignment currently requires a uniform node measure")
        grid = tuple(float(value) for value in self.spectral_quantile_grid)
        if not grid or any(value <= 0.0 or value >= 1.0 for value in grid):
            raise ValueError("spectral quantile grid must lie inside (0, 1)")
        if any(left >= right for left, right in zip(grid[:-1], grid[1:])):
            raise ValueError("spectral quantile grid must be strictly increasing")
    @property
    def edge_feature_dim(self) -> int:
        return 4


@dataclass(frozen=True)
class TimepointSelection:
    node_scores: torch.Tensor
    edge_scores: torch.Tensor
    soft_adjacency: torch.Tensor
    graph_embedding: torch.Tensor


@dataclass(frozen=True)
class BatchModelOutput:
    logits: torch.Tensor
    sample_embeddings: torch.Tensor
    node_retention_ratios: torch.Tensor
    edge_retention_ratios: torch.Tensor
    timepoint_sample_indices: torch.Tensor
    selections: Optional[Tuple[Tuple[TimepointSelection, ...], ...]]
    laplacian_fidelity: Optional[torch.Tensor] = None
    laplacian_operator_error: Optional[torch.Tensor] = None
    gw_fidelity: Optional[torch.Tensor] = None
    gw_solver_converged: Optional[torch.Tensor] = None
    gw_solver_iterations: Optional[torch.Tensor] = None
    gw_solver_residuals: Optional[torch.Tensor] = None


class _ScoreMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


class SignedGraphLayer(nn.Module):
    """Absolute-degree normalization with signed message weights."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float, epsilon: float):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.message_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.epsilon = epsilon

    def forward(self, node_features: torch.Tensor, signed_adjacency: torch.Tensor) -> torch.Tensor:
        degree = signed_adjacency.abs().sum(dim=-1).clamp_min(self.epsilon)
        inverse_sqrt = degree.rsqrt()
        normalized = (
            inverse_sqrt[:, None] * signed_adjacency * inverse_sqrt[None, :]
        )
        messages = torch.matmul(normalized, node_features)
        output = self.self_projection(node_features) + self.message_projection(messages)
        return self.dropout(self.activation(self.normalization(output)))


class SoftGraphClassifier(nn.Module):
    """Baseline soft_graph training path from the verified design."""

    training_mode = "soft_graph"
    uses_raw_community_embedding = False

    def __init__(self, config: Optional[SoftExtractorConfig] = None) -> None:
        super().__init__()
        self.config = config or SoftExtractorConfig()
        self.feature_builder = GraphFeatureBuilder(epsilon=self.config.epsilon)
        self.laplacian_builder = SignedLaplacianBuilder(self.config.laplacian_eta)
        self.heat_kernel_builder = HeatKernelMetricBuilder(self.config.heat_kernel_t)
        self.gw_loss = DifferentiableGWLoss(
            entropic_reg=self.config.gw_entropic_reg,
            max_iter=self.config.gw_max_iter,
            tolerance=self.config.gw_tolerance,
            sinkhorn_iter=self.config.gw_sinkhorn_iter,
            failure_strategy=self.config.gw_failure_strategy,
        )
        self.node_scorer = _ScoreMLP(
            self.config.node_feature_dim,
            self.config.node_score_hidden_dim,
            self.config.dropout,
        )
        self.edge_scorer = _ScoreMLP(
            self.config.edge_feature_dim,
            self.config.edge_score_hidden_dim,
            self.config.dropout,
        )
        layers = []
        input_dim = self.config.node_feature_dim
        for _ in range(self.config.graph_layers):
            layers.append(
                SignedGraphLayer(
                    input_dim,
                    self.config.graph_hidden_dim,
                    self.config.dropout,
                    self.config.epsilon,
                )
            )
            input_dim = self.config.graph_hidden_dim
        self.graph_layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.graph_hidden_dim, self.config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_dim, 2),
        )

    def _score_edges(
        self, edge_features: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        scores = self.edge_scorer(edge_features)
        scores = 0.5 * (scores + scores.transpose(0, 1))
        return scores * edge_mask.to(dtype=scores.dtype)

    def _encode_graph(
        self, node_features: torch.Tensor, soft_adjacency: torch.Tensor
    ) -> torch.Tensor:
        hidden = node_features
        for layer in self.graph_layers:
            hidden = layer(hidden, soft_adjacency)
        return hidden.mean(dim=0)

    def score_timepoint(
        self, sample, time_index: int
    ) -> Tuple[GraphTimepointFeatures, TimepointSelection]:
        """Score one complete timepoint; used by both soft training and frozen export."""

        features = self.feature_builder.build_timepoint(sample, time_index)
        if features.node_feature_dim != self.config.node_feature_dim:
            raise ValueError(
                "node feature dimension {} does not match model {}".format(
                    features.node_feature_dim, self.config.node_feature_dim
                )
            )
        node_scores = self.node_scorer(features.node_features)
        edge_scores = self._score_edges(features.edge_features, features.edge_mask)
        node_pair_scores = node_scores[:, None] * node_scores[None, :]
        adjacency = sample.adjacency[time_index]
        soft_adjacency = adjacency * node_pair_scores * edge_scores
        graph_embedding = self._encode_graph(features.node_features, soft_adjacency)
        return features, TimepointSelection(
            node_scores=node_scores,
            edge_scores=edge_scores,
            soft_adjacency=soft_adjacency,
            graph_embedding=graph_embedding,
        )

    def forward(
        self, batch: GraphSequenceBatch, return_details: bool = False
    ) -> BatchModelOutput:
        if len(batch) == 0:
            raise ValueError("model cannot process an empty batch")
        sample_embeddings = []
        node_ratios = []
        edge_ratios = []
        sample_indices = []
        all_selections = [] if return_details else None
        laplacian_errors = []
        operator_errors = []
        gw_errors = []
        gw_converged = []
        gw_iterations = []
        gw_residuals = []

        for sample_index, sample in enumerate(batch):
            time_embeddings = []
            sample_selections = []
            for time_index in range(sample.num_timepoints):
                features, selection = self.score_timepoint(sample, time_index)
                time_embeddings.append(selection.graph_embedding)
                node_ratios.append(selection.node_scores.mean())
                upper_edge_mask = torch.triu(features.edge_mask, diagonal=1)
                edge_ratios.append(
                    selection.edge_scores[upper_edge_mask].sum()
                    / upper_edge_mask.sum().clamp_min(1).to(selection.edge_scores.dtype)
                )
                sample_indices.append(sample_index)
                if self.config.theory_alignment_enabled:
                    full_laplacian = self.laplacian_builder(
                        sample.adjacency[time_index], edge_mask=features.edge_mask
                    )
                    soft_laplacian = self.laplacian_builder(
                        selection.soft_adjacency, edge_mask=features.edge_mask
                    )
                    difference = full_laplacian - soft_laplacian
                    laplacian_errors.append(torch.linalg.matrix_norm(difference, ord="fro"))
                    operator_errors.append(torch.linalg.eigvalsh(difference).abs().max())
                    full_metric = self.heat_kernel_builder(full_laplacian).distance
                    soft_metric = self.heat_kernel_builder(soft_laplacian).distance
                    gw_result = self.gw_loss(full_metric, soft_metric)
                    gw_errors.append(gw_result.distance)
                    gw_converged.append(gw_result.converged)
                    gw_iterations.append(gw_result.iterations)
                    gw_residuals.append(gw_result.residual)
                if return_details:
                    sample_selections.append(
                        selection
                    )
            sample_embeddings.append(torch.stack(time_embeddings, dim=0).mean(dim=0))
            if return_details:
                all_selections.append(tuple(sample_selections))

        embeddings = torch.stack(sample_embeddings, dim=0)
        logits = self.classifier(embeddings)
        return BatchModelOutput(
            logits=logits,
            sample_embeddings=embeddings,
            node_retention_ratios=torch.stack(node_ratios),
            edge_retention_ratios=torch.stack(edge_ratios),
            timepoint_sample_indices=torch.tensor(
                sample_indices, dtype=torch.long, device=logits.device
            ),
            selections=tuple(all_selections) if return_details else None,
            laplacian_fidelity=(
                torch.stack(laplacian_errors).mean() if laplacian_errors else None
            ),
            laplacian_operator_error=(
                torch.stack(operator_errors).mean() if operator_errors else None
            ),
            gw_fidelity=torch.stack(gw_errors).mean() if gw_errors else None,
            gw_solver_converged=(
                torch.tensor(gw_converged, dtype=torch.bool, device=logits.device)
                if gw_converged
                else None
            ),
            gw_solver_iterations=(
                torch.tensor(gw_iterations, dtype=torch.long, device=logits.device)
                if gw_iterations
                else None
            ),
            gw_solver_residuals=(
                torch.tensor(gw_residuals, dtype=logits.dtype, device=logits.device)
                if gw_residuals
                else None
            ),
        )
