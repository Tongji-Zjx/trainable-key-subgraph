"""T1--T4 temporal models built on frozen D3-B base logits."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from keysubgraph.data.dual_temporal_dataset import DualTemporalBatch


DUAL_TEMPORAL_VARIANTS = (
    "T1_variation_mean_mlp",
    "T2_variation_unigru",
    "T3_variation_bigru",
    "T4_variation_bigru_residual",
)


@dataclass(frozen=True)
class DualVariationTemporalConfig:
    variant: str = "T4_variation_bigru_residual"
    input_dim: int = 16
    hidden_dim_per_direction: int = 32
    num_layers: int = 1
    projection_hidden_dim: int = 64
    temporal_output_dim: int = 32
    classifier_hidden_dim: int = 32
    dropout: float = 0.20
    alpha_initial: float = 0.10

    def __post_init__(self):
        if self.variant not in DUAL_TEMPORAL_VARIANTS:
            raise ValueError("unsupported dual temporal variant")
        dimensions = (
            self.input_dim,
            self.hidden_dim_per_direction,
            self.num_layers,
            self.projection_hidden_dim,
            self.temporal_output_dim,
            self.classifier_hidden_dim,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("dual temporal dimensions must be positive")
        if self.input_dim != 16 or self.num_layers != 1:
            raise ValueError("first temporal version is fixed to 16-D/1-layer")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dual temporal dropout must lie in [0,1)")
        if self.alpha_initial <= 0.0 or self.alpha_initial >= 1.0:
            raise ValueError("temporal alpha initial must lie in (0,1)")


@dataclass(frozen=True)
class DualVariationTemporalOutput:
    final_logits: torch.Tensor
    temporal_logits: torch.Tensor
    temporal_representation: torch.Tensor
    temporal_states: torch.Tensor
    time_mask: torch.Tensor
    base_logits: torch.Tensor
    alpha: Optional[torch.Tensor]


class DualVariationTemporalClassifier(nn.Module):
    model_name = "dual_d3b_variation_temporal_residual"

    def __init__(
        self, config: Optional[DualVariationTemporalConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or DualVariationTemporalConfig()
        self.is_mean = self.config.variant == "T1_variation_mean_mlp"
        self.is_residual = (
            self.config.variant == "T4_variation_bigru_residual"
        )
        self.bidirectional = self.config.variant in (
            "T3_variation_bigru",
            "T4_variation_bigru_residual",
        )
        if self.is_mean:
            pooled_dim = self.config.input_dim
            self.recurrent = None
        else:
            self.recurrent = nn.GRU(
                input_size=self.config.input_dim,
                hidden_size=self.config.hidden_dim_per_direction,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=0.0,
            )
            recurrent_dim = self.config.hidden_dim_per_direction * (
                2 if self.bidirectional else 1
            )
            pooled_dim = 2 * recurrent_dim
        self.temporal_projection = nn.Sequential(
            nn.Linear(pooled_dim, self.config.projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.projection_hidden_dim,
                self.config.temporal_output_dim,
            ),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(
                self.config.temporal_output_dim,
                self.config.classifier_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_dim, 2),
        )
        if self.is_residual:
            alpha_logit = math.log(
                self.config.alpha_initial
                / (1.0 - self.config.alpha_initial)
            )
            self.alpha_logit = nn.Parameter(
                torch.tensor(alpha_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("alpha_logit", None)

    def config_dict(self):
        return asdict(self.config)

    def _mean_input(self, batch):
        weights = batch.time_mask.to(batch.transition_values.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (
            batch.transition_values * weights[:, :, None]
        ).sum(dim=1) / denominator
        states = batch.transition_values
        return mean, states

    def _recurrent_input(self, batch):
        values = batch.transition_values
        lengths = batch.sequence_lengths
        batch_size, maximum, _ = values.shape
        recurrent_dim = self.config.hidden_dim_per_direction * (
            2 if self.bidirectional else 1
        )
        states = values.new_zeros((batch_size, maximum, recurrent_dim))
        positive = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if positive.numel():
            selected_values = values.index_select(0, positive)
            selected_lengths = lengths.index_select(0, positive)
            packed = pack_padded_sequence(
                selected_values,
                selected_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_states, _ = self.recurrent(packed)
            selected_states, _ = pad_packed_sequence(
                packed_states,
                batch_first=True,
                total_length=maximum,
            )
            states.index_copy_(0, positive, selected_states)
        mask = batch.time_mask[:, :, None]
        weights = mask.to(states.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        mean = (states * weights).sum(dim=1) / denominator
        maximum_values = states.masked_fill(~mask, float("-inf")).max(
            dim=1
        ).values
        nonempty = lengths > 0
        maximum_values = torch.where(
            nonempty[:, None],
            maximum_values,
            torch.zeros_like(maximum_values),
        )
        return torch.cat((mean, maximum_values), dim=-1), states

    def forward(
        self, batch: DualTemporalBatch
    ) -> DualVariationTemporalOutput:
        if (
            batch.transition_values.ndim != 3
            or batch.transition_values.shape[-1] != 16
            or tuple(batch.time_mask.shape)
            != tuple(batch.transition_values.shape[:2])
            or batch.sequence_lengths.shape != (len(batch),)
            or batch.base_logits.shape != (len(batch), 2)
        ):
            raise ValueError("invalid dual temporal batch")
        expected_lengths = batch.time_mask.sum(dim=1).to(
            batch.sequence_lengths
        )
        if not bool((expected_lengths == batch.sequence_lengths).all()):
            raise ValueError("temporal mask and lengths disagree")
        if self.is_mean:
            pooled, states = self._mean_input(batch)
        else:
            pooled, states = self._recurrent_input(batch)
        representation = self.temporal_projection(pooled)
        temporal_logits = self.temporal_head(representation)
        nonempty = batch.sequence_lengths > 0
        representation = torch.where(
            nonempty[:, None],
            representation,
            torch.zeros_like(representation),
        )
        temporal_logits = torch.where(
            nonempty[:, None],
            temporal_logits,
            torch.zeros_like(temporal_logits),
        )
        base_logits = batch.base_logits.detach()
        alpha = None
        if self.is_residual:
            alpha = torch.sigmoid(self.alpha_logit)
            final_logits = base_logits + alpha * temporal_logits
        else:
            final_logits = temporal_logits
        return DualVariationTemporalOutput(
            final_logits=final_logits,
            temporal_logits=temporal_logits,
            temporal_representation=representation,
            temporal_states=states,
            time_mask=batch.time_mask,
            base_logits=base_logits,
            alpha=alpha,
        )
