"""Differentiable node/edge scoring and signed soft-graph classification."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.graph_features import GraphFeatureBuilder, GraphTimepointFeatures


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
        )
