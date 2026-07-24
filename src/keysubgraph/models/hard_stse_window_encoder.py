"""Enhanced Hard-STSE node/edge/statistic window encoder."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from keysubgraph.features.hard_stse_classification_features import (
    HardSTSEClassificationFeatures,
)
from .hard_stse_types import HardSTSEConfig


class ResidualFFNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values + self.network(values))


class ResidualSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        block_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualFFNBlock(hidden_dim, dropout) for _ in range(block_count)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.input_normalization(values))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class MeanStdMaxAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("pooling epsilon must be positive")
        self.epsilon = float(epsilon)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_dim = 4 * int(hidden_dim)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or tuple(mask.shape) != (values.shape[0],):
            raise ValueError("set values and mask are not aligned")
        valid = mask.to(device=values.device, dtype=torch.bool)
        if not bool(valid.any()):
            raise ValueError("cannot pool an empty hard set")
        selected = values[valid]
        mean = selected.mean(dim=0)
        variance = (selected - mean).square().mean(dim=0)
        std = torch.sqrt(variance + self.epsilon)
        maximum = selected.max(dim=0).values
        logits = self.attention(selected).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        attended = (weights[:, None] * selected).sum(dim=0)
        full_attention = values.new_zeros(values.shape[0])
        full_attention[valid] = weights
        return torch.cat((mean, std, maximum, attended), dim=-1), full_attention


@dataclass(frozen=True)
class HardSTSEWindowEncoding:
    node_hidden: torch.Tensor
    edge_hidden: torch.Tensor
    node_pooled: torch.Tensor
    edge_pooled: torch.Tensor
    raw_statistics: torch.Tensor
    graph_statistic_mask: torch.Tensor
    standardized_statistics: torch.Tensor
    raw_representation: torch.Tensor
    embedding: torch.Tensor
    node_attention: torch.Tensor
    edge_attention: torch.Tensor


class HardSTSEWindowEncoder(nn.Module):
    def __init__(self, config: Optional[HardSTSEConfig] = None) -> None:
        super().__init__()
        self.config = config or HardSTSEConfig()
        self.node_encoder = ResidualSetEncoder(
            self.config.classifier_node_feature_dim,
            self.config.node_hidden_dim,
            block_count=2,
            dropout=self.config.dropout,
        )
        self.edge_encoder = ResidualSetEncoder(
            self.config.classifier_edge_feature_dim,
            self.config.edge_hidden_dim,
            block_count=1,
            dropout=self.config.dropout,
        )
        self.node_pooling = MeanStdMaxAttentionPooling(
            self.config.node_hidden_dim, self.config.epsilon
        )
        self.edge_pooling = MeanStdMaxAttentionPooling(
            self.config.edge_hidden_dim, self.config.epsilon
        )
        self.register_buffer(
            "graph_statistic_mean",
            torch.zeros(self.config.graph_statistic_dim),
        )
        self.register_buffer(
            "graph_statistic_scale",
            torch.ones(self.config.graph_statistic_dim),
        )
        self.register_buffer(
            "graph_statistic_fitted",
            torch.tensor(False, dtype=torch.bool),
        )
        raw_dim = (
            self.node_pooling.output_dim
            + self.edge_pooling.output_dim
            + self.config.graph_statistic_dim
        )
        self.window_mlp = nn.Sequential(
            nn.Linear(raw_dim, self.config.window_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.window_hidden_dim,
                self.config.window_output_dim,
            ),
            nn.LayerNorm(self.config.window_output_dim),
        )

    def set_graph_statistic_transform(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        expected = (self.config.graph_statistic_dim,)
        if tuple(mean.shape) != expected or tuple(scale.shape) != expected:
            raise ValueError("graph statistic transform has the wrong dimension")
        if bool((scale <= 0.0).any()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("graph statistic scales must be finite and positive")
        self.graph_statistic_mean.copy_(mean.detach().to(
            self.graph_statistic_mean
        ))
        self.graph_statistic_scale.copy_(scale.detach().to(
            self.graph_statistic_scale
        ))
        self.graph_statistic_fitted.fill_(True)

    def forward(
        self, features: HardSTSEClassificationFeatures
    ) -> HardSTSEWindowEncoding:
        node_hidden = self.node_encoder(features.node_features)
        node_pooled, node_attention = self.node_pooling(
            node_hidden, features.node_mask
        )
        edge_hidden_matrix = self.edge_encoder(features.edge_features)
        upper = torch.triu(features.edge_mask, diagonal=1)
        edge_hidden = edge_hidden_matrix[upper]
        edge_pooled, selected_edge_attention = self.edge_pooling(
            edge_hidden,
            torch.ones(
                edge_hidden.shape[0], dtype=torch.bool, device=edge_hidden.device
            ),
        )
        edge_attention = features.edge_features.new_zeros(
            features.edge_mask.shape
        )
        edge_attention[upper] = selected_edge_attention
        edge_attention = edge_attention + edge_attention.transpose(0, 1)
        statistics = (
            features.graph_statistics - self.graph_statistic_mean
        ) / self.graph_statistic_scale
        statistics = statistics.masked_fill(
            ~features.graph_statistic_mask, 0.0
        )
        raw = torch.cat((node_pooled, edge_pooled, statistics), dim=-1)
        embedding = self.window_mlp(raw)
        return HardSTSEWindowEncoding(
            node_hidden=node_hidden,
            edge_hidden=edge_hidden_matrix,
            node_pooled=node_pooled,
            edge_pooled=edge_pooled,
            raw_statistics=features.graph_statistics,
            graph_statistic_mask=features.graph_statistic_mask,
            standardized_statistics=statistics,
            raw_representation=raw,
            embedding=embedding,
            node_attention=node_attention,
            edge_attention=edge_attention,
        )
