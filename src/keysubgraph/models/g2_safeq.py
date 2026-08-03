"""Conservative delta-Q residual correction for a frozen G2 classifier."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


G2_SAFEQ_MODEL_NAME = "g2_safeq"
G2_SAFEQ_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class G2SafeQConfig:
    transition_hidden_dim: int = 64
    residual_hidden_dim: int = 16
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if int(self.transition_hidden_dim) < 1:
            raise ValueError("SafeQ transition dimension must be positive")
        if int(self.residual_hidden_dim) < 1:
            raise ValueError("SafeQ residual dimension must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("SafeQ dropout must lie in [0,1)")

    @property
    def summary_dim(self) -> int:
        return 2 * int(self.transition_hidden_dim)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class G2SafeQOutput:
    logits: torch.Tensor
    residual_logits: torch.Tensor
    base_logits: torch.Tensor
    static_logits: torch.Tensor
    alpha: float
    beta: float


def aggregate_transition_hidden(
    hidden: Optional[torch.Tensor],
    sample_indices: Optional[torch.Tensor],
    batch_size: int,
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mask-aware mean/std summaries for variable transition counts."""

    if int(batch_size) < 1 or int(hidden_dim) < 1:
        raise ValueError("SafeQ aggregation dimensions must be positive")
    if hidden is None or sample_indices is None:
        return (
            torch.zeros(int(batch_size), 2 * int(hidden_dim)),
            torch.zeros(int(batch_size), dtype=torch.bool),
        )
    if hidden.ndim != 2 or int(hidden.shape[1]) != int(hidden_dim):
        raise ValueError("SafeQ transition hidden tensor has invalid shape")
    if tuple(sample_indices.shape) != (int(hidden.shape[0]),):
        raise ValueError("SafeQ transition indices do not align with hidden")
    if sample_indices.dtype != torch.long:
        raise ValueError("SafeQ transition indices must be int64")
    if hidden.device != sample_indices.device:
        raise ValueError("SafeQ transition tensors must share a device")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("SafeQ transition hidden tensor must be finite")
    if hidden.shape[0] and (
        int(sample_indices.min()) < 0
        or int(sample_indices.max()) >= int(batch_size)
    ):
        raise ValueError("SafeQ transition sample index is out of range")

    summaries = hidden.new_zeros(
        (int(batch_size), 2 * int(hidden_dim))
    )
    valid = torch.zeros(
        int(batch_size), dtype=torch.bool, device=hidden.device
    )
    for sample_index in range(int(batch_size)):
        selected = hidden[sample_indices == sample_index]
        if int(selected.shape[0]) < 1:
            continue
        valid[sample_index] = True
        summaries[sample_index, :hidden_dim] = selected.mean(dim=0)
        summaries[sample_index, hidden_dim:] = selected.std(
            dim=0, unbiased=False
        )
    return summaries, valid


class G2SafeQResidual(nn.Module):
    """Small residual head on frozen delta-Q transition summaries.

    ``alpha=beta=0`` is an exact identity path to the frozen G2 logit.  The
    final layer is zero initialized, and samples without a valid adjacent
    transition receive an exactly-zero residual even after training.
    """

    model_name = G2_SAFEQ_MODEL_NAME

    def __init__(self, config: G2SafeQConfig) -> None:
        super().__init__()
        self.config = config
        self.residual_head = nn.Sequential(
            nn.LayerNorm(config.summary_dim),
            nn.Linear(config.summary_dim, int(config.residual_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.residual_hidden_dim), 1),
        )
        with torch.no_grad():
            self.residual_head[-1].weight.zero_()
            self.residual_head[-1].bias.zero_()

    def forward(
        self,
        base_logits: torch.Tensor,
        static_logits: torch.Tensor,
        transition_summaries: torch.Tensor,
        has_valid_transition: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> G2SafeQOutput:
        count = int(base_logits.shape[0]) if base_logits.ndim == 1 else -1
        if count < 1 or tuple(static_logits.shape) != (count,):
            raise ValueError("SafeQ anchor logits must be aligned [B]")
        if tuple(transition_summaries.shape) != (
            count,
            self.config.summary_dim,
        ):
            raise ValueError("SafeQ transition summary shape mismatch")
        if tuple(has_valid_transition.shape) != (count,):
            raise ValueError("SafeQ transition mask shape mismatch")
        if has_valid_transition.dtype != torch.bool:
            raise ValueError("SafeQ transition mask must be boolean")
        values = (base_logits, static_logits, transition_summaries)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("SafeQ inputs must be finite")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("SafeQ alpha must lie in [0,1]")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("SafeQ beta must lie in [0,1]")

        frozen_base = base_logits.detach()
        frozen_static = static_logits.detach()
        frozen_summary = transition_summaries.detach()
        residual = self.residual_head(frozen_summary).squeeze(-1)
        residual = residual * has_valid_transition.to(residual)
        logits = (
            frozen_base
            + float(beta) * (frozen_static - frozen_base)
            + float(alpha) * residual
        )
        return G2SafeQOutput(
            logits=logits,
            residual_logits=residual,
            base_logits=frozen_base,
            static_logits=frozen_static,
            alpha=float(alpha),
            beta=float(beta),
        )
