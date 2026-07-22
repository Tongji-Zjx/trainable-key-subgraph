"""Permutation-invariant mean, max and gated pooling for variable-size graphs."""

from __future__ import absolute_import, division, print_function

from typing import Optional, Tuple

import torch
from torch import nn


class MaskedGraphPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 96,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("pooling dimensions must be positive")
        hidden = int(hidden_dim or output_dim)
        self.epsilon = float(epsilon)
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.output = nn.Sequential(
            nn.Linear(3 * input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.output_dim = int(output_dim)

    @staticmethod
    def _mask(features: torch.Tensor, node_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("pooling features must have shape [N, F]")
        if node_mask is None:
            result = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        else:
            if tuple(node_mask.shape) != (features.shape[0],):
                raise ValueError("pooling node_mask must have shape [N]")
            result = node_mask.to(device=features.device, dtype=torch.bool)
        if not bool(result.any()):
            raise ValueError("cannot pool a graph without valid nodes")
        return result

    def components(
        self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = self._mask(features, node_mask)
        weights = valid.to(features.dtype)
        mean = (features * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(self.epsilon)
        masked = features.masked_fill(~valid[:, None], -torch.inf)
        maximum = masked.max(dim=0).values
        gate_logits = self.gate(features).squeeze(-1).masked_fill(~valid, -torch.inf)
        attention = torch.softmax(gate_logits, dim=0)
        gated = (attention[:, None] * features).sum(dim=0)
        return mean, maximum, gated

    def forward(
        self, features: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mean, maximum, gated = self.components(features, node_mask=node_mask)
        return self.output(torch.cat((mean, maximum, gated), dim=-1))

