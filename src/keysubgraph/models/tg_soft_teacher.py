"""Stage-A differentiable TG-SGW soft teacher."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.graph_features import GraphFeatureBuilder
from keysubgraph.theory import (
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    gw_identity_coupling_upper_bound,
    laplacian_fidelity_metrics,
)

from .masked_pooling import MaskedGraphPooling
from .masked_tcn import MaskedTCNEncoder
from .signed_graph_encoder import SignedGraphEncoder


@dataclass(frozen=True)
class TGSoftTeacherConfig:
    node_feature_dim: int = 13
    edge_feature_dim: int = 4
    node_score_hidden_dim: int = 64
    edge_score_hidden_dim: int = 32
    signed_gnn_hidden_dim: int = 64
    signed_gnn_layers: int = 3
    graph_embedding_dim: int = 96
    temporal_hidden_dim: int = 96
    temporal_kernel_size: int = 3
    temporal_dilations: Tuple[int, ...] = (1, 2, 4)
    classifier_hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    epsilon: float = 1.0e-8
    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0

    def __post_init__(self) -> None:
        dimensions = (
            self.node_feature_dim,
            self.edge_feature_dim,
            self.node_score_hidden_dim,
            self.edge_score_hidden_dim,
            self.signed_gnn_hidden_dim,
            self.signed_gnn_layers,
            self.graph_embedding_dim,
            self.temporal_hidden_dim,
        ) + tuple(self.classifier_hidden_dims)
        if any(value < 1 for value in dimensions):
            raise ValueError("TG soft-teacher dimensions must be positive")
        if self.node_feature_dim != 13 or self.edge_feature_dim != 4:
            raise ValueError("TG soft teacher requires the verified 13-D/4-D feature schema")
        if self.graph_embedding_dim != 96 or self.temporal_hidden_dim != 96:
            raise ValueError("TG soft teacher requires 96-D graph and temporal hidden states")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.epsilon <= 0.0 or self.laplacian_eta <= 0.0 or self.diffusion_time <= 0.0:
            raise ValueError("epsilon, eta and diffusion time must be positive")


@dataclass(frozen=True)
class TGSoftTimepointOutput:
    node_scores: torch.Tensor
    edge_scores: torch.Tensor
    soft_adjacency: torch.Tensor
    window_embedding: torch.Tensor


@dataclass(frozen=True)
class TGScoreStatistics:
    count: int
    total: torch.Tensor
    squared_total: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    above_half_count: torch.Tensor
    entropy_total: torch.Tensor


@dataclass(frozen=True)
class TGSoftTeacherOutput:
    logits: torch.Tensor
    representation: torch.Tensor
    encoded_windows: torch.Tensor
    time_mask: torch.Tensor
    node_retention_ratios: torch.Tensor
    edge_retention_ratios: torch.Tensor
    timepoint_sample_indices: torch.Tensor
    laplacian_normalized_frobenius: torch.Tensor
    laplacian_frobenius_norms: torch.Tensor
    laplacian_operator_norms: torch.Tensor
    gw_identity_upper_bounds_squared: torch.Tensor
    gw_identity_upper_bounds: torch.Tensor
    node_score_statistics: TGScoreStatistics
    edge_score_statistics: TGScoreStatistics
    selections: Optional[Tuple[Tuple[TGSoftTimepointOutput, ...], ...]]


class _ScoreMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


def _classifier(input_dim: int, hidden_dims: Tuple[int, ...], dropout: float) -> nn.Module:
    modules = []
    current = input_dim
    for hidden in hidden_dims:
        modules.extend((nn.Linear(current, hidden), nn.GELU(), nn.Dropout(dropout)))
        current = hidden
    modules.append(nn.Linear(current, 2))
    return nn.Sequential(*modules)


class TGSoftTeacher(nn.Module):
    training_stage = "soft_teacher"
    uses_raw_community_embedding = False

    def __init__(self, config: Optional[TGSoftTeacherConfig] = None) -> None:
        super().__init__()
        self.config = config or TGSoftTeacherConfig()
        self.feature_builder = GraphFeatureBuilder(self.config.epsilon)
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
        self.graph_encoder = SignedGraphEncoder(
            self.config.node_feature_dim,
            self.config.signed_gnn_hidden_dim,
            self.config.signed_gnn_layers,
            self.config.dropout,
            self.config.epsilon,
        )
        self.graph_pooling = MaskedGraphPooling(
            self.config.signed_gnn_hidden_dim,
            self.config.graph_embedding_dim,
            dropout=self.config.dropout,
            epsilon=self.config.epsilon,
        )
        self.temporal_encoder = MaskedTCNEncoder(
            self.config.graph_embedding_dim,
            self.config.temporal_hidden_dim,
            self.config.temporal_kernel_size,
            self.config.temporal_dilations,
            self.config.dropout,
            self.config.epsilon,
        )
        self.classifier = _classifier(
            self.temporal_encoder.output_dim,
            self.config.classifier_hidden_dims,
            self.config.dropout,
        )
        self.laplacian = SignedLaplacianBuilder(self.config.laplacian_eta)
        self.heat_kernel = HeatKernelMetricBuilder(self.config.diffusion_time)

    def _score_timepoint(self, sample, time_index: int):
        features = self.feature_builder.build_timepoint(sample, time_index)
        node_scores = self.node_scorer(features.node_features)
        edge_scores = self.edge_scorer(features.edge_features)
        edge_scores = 0.5 * (edge_scores + edge_scores.transpose(0, 1))
        edge_scores = edge_scores * features.edge_mask.to(edge_scores.dtype)
        adjacency = sample.adjacency[time_index]
        soft_adjacency = (
            adjacency
            * node_scores[:, None]
            * node_scores[None, :]
            * edge_scores
        )
        hidden = self.graph_encoder(features.node_features, soft_adjacency)
        embedding = self.graph_pooling(hidden)

        full_laplacian = self.laplacian(adjacency, edge_mask=features.edge_mask)
        soft_laplacian = self.laplacian(soft_adjacency, edge_mask=features.edge_mask)
        laplacian_result = laplacian_fidelity_metrics(full_laplacian, soft_laplacian)
        full_distance = self.heat_kernel(full_laplacian).distance
        soft_distance = self.heat_kernel(soft_laplacian).distance
        gw_result = gw_identity_coupling_upper_bound(full_distance, soft_distance)
        upper_edge_mask = torch.triu(features.edge_mask, diagonal=1)
        edge_ratio = (
            edge_scores[upper_edge_mask].sum()
            / upper_edge_mask.sum().clamp_min(1).to(edge_scores.dtype)
        )
        return (
            TGSoftTimepointOutput(node_scores, edge_scores, soft_adjacency, embedding),
            node_scores.mean(),
            edge_ratio,
            laplacian_result,
            gw_result,
        )

    @staticmethod
    def _score_statistics(values) -> TGScoreStatistics:
        nonempty = [value.reshape(-1) for value in values if value.numel() > 0]
        if not nonempty:
            raise ValueError("score statistics require at least one valid value")
        flattened = torch.cat(nonempty, dim=0)
        epsilon = torch.finfo(flattened.dtype).eps
        bounded = flattened.clamp(epsilon, 1.0 - epsilon)
        entropy = -(
            bounded * bounded.log()
            + (1.0 - bounded) * (1.0 - bounded).log()
        )
        return TGScoreStatistics(
            count=int(flattened.numel()),
            total=flattened.sum(),
            squared_total=flattened.square().sum(),
            minimum=flattened.min(),
            maximum=flattened.max(),
            above_half_count=(flattened >= 0.5).sum(),
            entropy_total=entropy.sum(),
        )

    def score_timepoint(self, sample, time_index: int):
        """Frozen-export interface shared with the hard candidate generator."""

        features = self.feature_builder.build_timepoint(sample, time_index)
        output = self._score_timepoint(sample, time_index)[0]
        return features, output

    def forward(
        self, batch: GraphSequenceBatch, return_details: bool = False
    ) -> TGSoftTeacherOutput:
        if len(batch) == 0:
            raise ValueError("TG soft teacher cannot process an empty batch")
        sequences = []
        nested = [] if return_details else None
        node_ratios, edge_ratios, sample_indices = [], [], []
        lap_train, lap_fro, lap_operator = [], [], []
        gw_squared, gw_distance = [], []
        node_score_values, edge_score_values = [], []
        for sample_index, sample in enumerate(batch):
            window_embeddings = []
            sample_outputs = []
            for time_index in range(sample.num_timepoints):
                output, node_ratio, edge_ratio, lap_result, gw_result = self._score_timepoint(
                    sample, time_index
                )
                window_embeddings.append(output.window_embedding)
                node_ratios.append(node_ratio)
                edge_ratios.append(edge_ratio)
                sample_indices.append(sample_index)
                lap_train.append(lap_result.normalized_frobenius_squared)
                lap_fro.append(lap_result.frobenius_norm)
                lap_operator.append(lap_result.operator_norm)
                gw_squared.append(gw_result.squared_upper_bound)
                gw_distance.append(gw_result.distance_upper_bound)
                node_score_values.append(output.node_scores)
                upper_edge_mask = torch.triu(
                    sample.edge_mask[time_index], diagonal=1
                )
                if bool(upper_edge_mask.any()):
                    edge_score_values.append(output.edge_scores[upper_edge_mask])
                if return_details:
                    sample_outputs.append(output)
            sequences.append(torch.stack(window_embeddings, dim=0))
            if return_details:
                nested.append(tuple(sample_outputs))
        representation, encoded_windows, time_mask = self.temporal_encoder.forward_list(sequences)
        logits = self.classifier(representation)
        return TGSoftTeacherOutput(
            logits=logits,
            representation=representation,
            encoded_windows=encoded_windows,
            time_mask=time_mask,
            node_retention_ratios=torch.stack(node_ratios),
            edge_retention_ratios=torch.stack(edge_ratios),
            timepoint_sample_indices=torch.tensor(sample_indices, dtype=torch.long, device=logits.device),
            laplacian_normalized_frobenius=torch.stack(lap_train),
            laplacian_frobenius_norms=torch.stack(lap_fro),
            laplacian_operator_norms=torch.stack(lap_operator),
            gw_identity_upper_bounds_squared=torch.stack(gw_squared),
            gw_identity_upper_bounds=torch.stack(gw_distance),
            node_score_statistics=self._score_statistics(node_score_values),
            edge_score_statistics=self._score_statistics(edge_score_values),
            selections=tuple(nested) if return_details else None,
        )
