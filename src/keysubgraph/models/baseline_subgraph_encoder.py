"""Signed hard-subgraph encoding and mask-aware hierarchical pooling."""

from __future__ import absolute_import, division, print_function

from typing import Sequence

import torch
from torch import nn

from .signed_message_passing import SignedMessagePassingLayer


class SignedSubgraphEncoder(nn.Module):
    """Encode padded signed subgraphs and return [mean; max] embeddings."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("signed encoder must contain at least one layer")
        dimensions = [int(node_feature_dim)] + [int(hidden_dim)] * int(layers)
        self.layers = nn.ModuleList(
            SignedMessagePassingLayer(
                dimensions[index],
                dimensions[index + 1],
                dropout=dropout,
                residual=residual,
            )
            for index in range(layers)
        )
        self.node_feature_dim = int(node_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = 2 * int(hidden_dim)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        if node_features.shape[-1] != self.node_feature_dim:
            raise ValueError("unexpected node feature dimension")
        if node_mask.dim() != 2 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("node_mask does not align with node features")
        if bool((node_mask.sum(dim=1) == 0).any()):
            raise ValueError("cannot encode an empty subgraph")
        state = node_features * node_mask.unsqueeze(-1).to(node_features.dtype)
        for layer in self.layers:
            state = layer(state, adjacency, node_mask)
        denominator = node_mask.sum(dim=1, keepdim=True).to(state.dtype)
        mean_pool = state.sum(dim=1) / denominator
        minimum = torch.finfo(state.dtype).min
        masked_for_max = state.masked_fill(~node_mask.unsqueeze(-1), minimum)
        max_pool = masked_for_max.max(dim=1).values
        return torch.cat((mean_pool, max_pool), dim=-1)


class WindowMeanPooling(nn.Module):
    """Average flattened subgraph embeddings into their effective windows."""

    def forward(
        self,
        subgraph_embeddings: torch.Tensor,
        subgraph_to_window: torch.Tensor,
        window_count: int,
        expected_counts: torch.Tensor,
    ) -> torch.Tensor:
        if subgraph_embeddings.dim() != 2:
            raise ValueError("subgraph embeddings must have shape [Q,D]")
        if subgraph_to_window.dim() != 1 or subgraph_to_window.numel() != subgraph_embeddings.shape[0]:
            raise ValueError("subgraph_to_window does not align with embeddings")
        if window_count < 1 or expected_counts.shape != (window_count,):
            raise ValueError("invalid window count metadata")
        if int(subgraph_to_window.min()) < 0 or int(subgraph_to_window.max()) >= window_count:
            raise ValueError("subgraph_to_window contains an invalid index")
        sums = subgraph_embeddings.new_zeros(window_count, subgraph_embeddings.shape[-1])
        sums.index_add_(0, subgraph_to_window, subgraph_embeddings)
        counts = torch.bincount(
            subgraph_to_window, minlength=window_count
        ).to(device=subgraph_embeddings.device)
        if not torch.equal(counts.to(expected_counts.device), expected_counts):
            raise ValueError("window subgraph counts do not match mapping")
        if bool((counts == 0).any()):
            raise ValueError("cannot pool an empty effective window")
        return sums / counts.to(sums.dtype).unsqueeze(-1)
