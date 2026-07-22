"""Positive/negative-channel message passing for signed TG-SGW graphs."""

from __future__ import absolute_import, division, print_function

from typing import Optional

import torch
from torch import nn


def _node_mask(features: torch.Tensor, node_mask: Optional[torch.Tensor]) -> torch.Tensor:
    count = features.shape[0]
    if node_mask is None:
        return torch.ones(count, dtype=torch.bool, device=features.device)
    if tuple(node_mask.shape) != (count,):
        raise ValueError("node_mask must have shape [N]")
    result = node_mask.to(device=features.device, dtype=torch.bool)
    if not bool(result.any()):
        raise ValueError("signed graph has no valid nodes")
    return result


def normalized_unsigned_channel(
    adjacency: torch.Tensor,
    node_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    pair_mask = node_mask[:, None] & node_mask[None, :]
    weights = adjacency * pair_mask.to(adjacency.dtype)
    weights = 0.5 * (weights + weights.transpose(0, 1))
    weights = weights.clone()
    weights.fill_diagonal_(0.0)
    degree = weights.sum(dim=-1)
    inverse_sqrt = torch.zeros_like(degree)
    positive = degree > 0.0
    inverse_sqrt[positive] = (degree[positive] + epsilon).rsqrt()
    return inverse_sqrt[:, None] * weights * inverse_sqrt[None, :]


class SignedMessageLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("signed message dimensions must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.epsilon = float(epsilon)
        self.update = nn.Sequential(
            nn.Linear(3 * input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.normalization = nn.LayerNorm(output_dim)

    def forward(
        self,
        features: torch.Tensor,
        signed_adjacency: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("node features must have shape [N, F]")
        if tuple(signed_adjacency.shape) != (features.shape[0], features.shape[0]):
            raise ValueError("signed adjacency must have shape [N, N]")
        valid = _node_mask(features, node_mask)
        positive = signed_adjacency.clamp_min(0.0)
        negative = (-signed_adjacency.clamp_max(0.0))
        normalized_positive = normalized_unsigned_channel(positive, valid, self.epsilon)
        normalized_negative = normalized_unsigned_channel(negative, valid, self.epsilon)
        positive_message = normalized_positive.matmul(features)
        negative_message = normalized_negative.matmul(features)
        update = self.update(torch.cat((features, positive_message, negative_message), dim=-1))
        output = self.normalization(self.residual(features) + update)
        return output * valid[:, None].to(output.dtype)


class SignedGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("SignedGraphEncoder requires at least one layer")
        layers = []
        current = input_dim
        for _ in range(num_layers):
            layers.append(SignedMessageLayer(current, hidden_dim, dropout, epsilon))
            current = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output_dim = int(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        signed_adjacency: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = node_features
        for layer in self.layers:
            hidden = layer(hidden, signed_adjacency, node_mask=node_mask)
        return hidden

