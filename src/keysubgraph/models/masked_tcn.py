"""Mask-aware temporal convolution for variable-length graph sequences."""

from __future__ import absolute_import, division, print_function

from typing import Sequence, Tuple

import torch
from torch import nn


def pad_temporal_sequences(
    sequences: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locally pad one list batch and return an explicit time mask."""

    if not sequences:
        raise ValueError("temporal sequence list cannot be empty")
    feature_dim = int(sequences[0].shape[-1]) if sequences[0].ndim == 2 else -1
    if feature_dim < 1:
        raise ValueError("temporal sequences must have shape [M, F]")
    if any(item.ndim != 2 or item.shape[0] < 1 or item.shape[1] != feature_dim for item in sequences):
        raise ValueError("all temporal sequences must be non-empty and share feature dimension")
    max_length = max(int(item.shape[0]) for item in sequences)
    reference = sequences[0]
    padded = reference.new_zeros((len(sequences), max_length, feature_dim))
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=reference.device)
    for index, item in enumerate(sequences):
        if item.device != reference.device or item.dtype != reference.dtype:
            item = item.to(device=reference.device, dtype=reference.dtype)
        length = int(item.shape[0])
        padded[index, :length] = item
        mask[index, :length] = True
    return padded, mask


class MaskedTemporalConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("masked TCN kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("masked TCN dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.convolution = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Conv1d(input_dim, output_dim, 1)
        )
        self.normalization = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("TCN values must have shape [B, C, M]")
        if tuple(time_mask.shape) != (values.shape[0], values.shape[2]):
            raise ValueError("time_mask must have shape [B, M]")
        mask = time_mask[:, None, :].to(device=values.device, dtype=values.dtype)
        masked = values * mask
        update = self.dropout(self.activation(self.convolution(masked)))
        output = self.residual(masked) + update
        output = self.normalization(output.transpose(1, 2)).transpose(1, 2)
        return output * mask


class MaskedTCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 96,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("masked TCN requires at least one dilation")
        blocks = []
        current = input_dim
        for dilation in dilations:
            blocks.append(
                MaskedTemporalConvBlock(
                    current, hidden_dim, kernel_size, int(dilation), dropout
                )
            )
            current = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = 2 * int(hidden_dim)
        self.epsilon = float(epsilon)

    def forward(
        self, padded_sequence: torch.Tensor, time_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padded_sequence.ndim != 3:
            raise ValueError("padded temporal input must have shape [B, M, F]")
        if tuple(time_mask.shape) != tuple(padded_sequence.shape[:2]):
            raise ValueError("time_mask must match [B, M]")
        mask = time_mask.to(device=padded_sequence.device, dtype=torch.bool)
        if bool((mask.sum(dim=1) == 0).any()):
            raise ValueError("each temporal sample requires at least one valid window")
        hidden = padded_sequence.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden_time = hidden.transpose(1, 2)
        weights = mask.to(hidden_time.dtype)
        mean = (hidden_time * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        maximum = hidden_time.masked_fill(~mask[:, :, None], -torch.inf).max(dim=1).values
        representation = torch.cat((mean, maximum), dim=-1)
        return representation, hidden_time * weights[:, :, None]

    def forward_list(
        self, sequences: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded, mask = pad_temporal_sequences(sequences)
        representation, encoded = self.forward(padded, mask)
        return representation, encoded, mask

