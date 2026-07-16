"""Positive/negative separated dense message passing for padded subgraphs."""

from __future__ import absolute_import, division, print_function

from typing import Tuple

import torch
from torch import nn


class SignedMessagePassingLayer(nn.Module):
    """One signed layer over tensors shaped [Q, N, F] and [Q, N, N]."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        residual: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("signed layer dimensions must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.epsilon = float(epsilon)
        self.positive_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.negative_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.message_update = nn.Linear(
            input_dim + 2 * output_dim, output_dim
        )
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = bool(residual)
        if self.use_residual:
            self.residual_projection = (
                nn.Identity()
                if input_dim == output_dim
                else nn.Linear(input_dim, output_dim, bias=False)
            )
        else:
            self.residual_projection = None

    def signed_messages(
        self, node_state: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_state.dim() != 3:
            raise ValueError("node_state must have shape [Q,N,F]")
        if adjacency.dim() != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape [Q,N,N]")
        if adjacency.shape[0] != node_state.shape[0] or adjacency.shape[1] != node_state.shape[1]:
            raise ValueError("adjacency and node_state shapes do not align")
        if node_mask.shape != node_state.shape[:2]:
            raise ValueError("node_mask must have shape [Q,N]")
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        positive = adjacency.clamp_min(0.0) * pair_mask.to(adjacency.dtype)
        negative = (-adjacency.clamp_max(0.0)) * pair_mask.to(adjacency.dtype)
        positive_norm = positive / (
            positive.sum(dim=-1, keepdim=True) + self.epsilon
        )
        negative_norm = negative / (
            negative.sum(dim=-1, keepdim=True) + self.epsilon
        )
        masked_state = node_state * node_mask.unsqueeze(-1).to(node_state.dtype)
        positive_message = torch.bmm(
            positive_norm, self.positive_projection(masked_state)
        )
        negative_message = torch.bmm(
            negative_norm, self.negative_projection(masked_state)
        )
        output_mask = node_mask.unsqueeze(-1).to(node_state.dtype)
        return positive_message * output_mask, negative_message * output_mask

    def forward(
        self, node_state: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        positive_message, negative_message = self.signed_messages(
            node_state, adjacency, node_mask
        )
        updated = self.message_update(
            torch.cat((node_state, positive_message, negative_message), dim=-1)
        )
        updated = self.activation(self.normalization(updated))
        updated = self.dropout(updated)
        if self.residual_projection is not None:
            updated = updated + self.residual_projection(node_state)
        return updated * node_mask.unsqueeze(-1).to(updated.dtype)
