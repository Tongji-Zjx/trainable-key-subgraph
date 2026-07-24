"""Mask-aware temporal encoder for variable-length hard-graph sequences."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from .hard_stse_types import HardSTSEConfig


@dataclass(frozen=True)
class HardSTSETemporalOutput:
    augmented_sequences: Tuple[torch.Tensor, ...]
    padded_states: torch.Tensor
    time_mask: torch.Tensor
    attention: torch.Tensor
    pooled_representation: torch.Tensor
    representation: torch.Tensor
    sequence_lengths: torch.Tensor


class HardSTSETemporalEncoder(nn.Module):
    def __init__(self, config: Optional[HardSTSEConfig] = None) -> None:
        super().__init__()
        self.config = config or HardSTSEConfig()
        self.delta_projection = nn.Sequential(
            nn.Linear(3 * self.config.window_output_dim, self.config.window_output_dim),
            nn.LayerNorm(self.config.window_output_dim),
        )
        self.bigru = nn.GRU(
            input_size=self.config.window_output_dim,
            hidden_size=self.config.temporal_hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        state_dim = 2 * self.config.temporal_hidden_per_direction
        self.attention = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * state_dim, self.config.neural_output_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.neural_output_dim),
        )

    def forward(
        self, sequences: Sequence[torch.Tensor]
    ) -> HardSTSETemporalOutput:
        if not sequences:
            raise ValueError("temporal encoder requires at least one sample")
        augmented = []
        lengths = []
        for sequence in sequences:
            if sequence.ndim != 2 or sequence.shape[0] < 1:
                raise ValueError("each temporal sequence must have shape [M, F]")
            if sequence.shape[1] != self.config.window_output_dim:
                raise ValueError("temporal window dimension is invalid")
            delta = torch.zeros_like(sequence)
            if sequence.shape[0] > 1:
                delta[1:] = sequence[1:] - sequence[:-1]
            values = torch.cat((sequence, delta, delta.abs()), dim=-1)
            augmented.append(self.delta_projection(values))
            lengths.append(int(sequence.shape[0]))
        padded = nn.utils.rnn.pad_sequence(augmented, batch_first=True)
        length_tensor = torch.tensor(
            lengths, dtype=torch.long, device=padded.device
        )
        packed = pack_padded_sequence(
            padded,
            length_tensor.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded_packed, _ = self.bigru(packed)
        states, _ = pad_packed_sequence(
            encoded_packed,
            batch_first=True,
            total_length=padded.shape[1],
        )
        positions = torch.arange(states.shape[1], device=states.device)[None, :]
        time_mask = positions < length_tensor[:, None]
        logits = self.attention(states).squeeze(-1)
        logits = logits.masked_fill(~time_mask, -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        attention = attention.masked_fill(~time_mask, 0.0)
        attended = (attention[:, :, None] * states).sum(dim=1)
        weights = time_mask.to(states.dtype)
        mean = (states * weights[:, :, None]).sum(dim=1)
        mean = mean / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        maximum = states.masked_fill(
            ~time_mask[:, :, None], -torch.inf
        ).max(dim=1).values
        pooled = torch.cat((attended, mean, maximum), dim=-1)
        representation = self.output(pooled)
        return HardSTSETemporalOutput(
            augmented_sequences=tuple(augmented),
            padded_states=states * weights[:, :, None],
            time_mask=time_mask,
            attention=attention,
            pooled_representation=pooled,
            representation=representation,
            sequence_lengths=length_tensor,
        )

