"""Permutation-invariant node-only subgraph encoder for edge-message ablations."""

from __future__ import absolute_import, division, print_function

import torch
from torch import nn


class NodeOnlyLayer(nn.Module):
    """Transform nodes independently without reading adjacency information."""

    def __init__(self, input_dim, output_dim, dropout=0.0, residual=True):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual and input_dim == output_dim)

    def forward(self, node_features, node_mask):
        if node_features.dim() != 3 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("invalid node-only encoder inputs")
        hidden = self.projection(node_features)
        hidden = self.activation(self.normalization(hidden))
        hidden = self.dropout(hidden)
        if self.residual:
            hidden = hidden + node_features
        return hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class NodeOnlySubgraphEncoder(nn.Module):
    """Encode a subgraph from node features only, then apply mean/max pooling."""

    def __init__(
        self, node_feature_dim=12, hidden_dim=64, layers=2, dropout=0.1,
        residual=True
    ):
        super().__init__()
        if min(node_feature_dim, hidden_dim, layers) < 1:
            raise ValueError("node-only encoder dimensions must be positive")
        modules = []
        input_dim = node_feature_dim
        for _ in range(layers):
            modules.append(
                NodeOnlyLayer(input_dim, hidden_dim, dropout, residual)
            )
            input_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.output_dim = hidden_dim * 2

    def forward(self, node_features, node_mask):
        if not bool(node_mask.any(dim=1).all()):
            raise ValueError("each subgraph needs at least one valid node")
        hidden = node_features
        for layer in self.layers:
            hidden = layer(hidden, node_mask)
        mask = node_mask.unsqueeze(-1)
        denominator = node_mask.sum(dim=1).clamp_min(1).to(hidden.dtype).unsqueeze(-1)
        mean_pool = hidden.sum(dim=1) / denominator
        max_pool = hidden.masked_fill(~mask, float("-inf")).max(dim=1).values
        return torch.cat((mean_pool, max_pool), dim=-1)
