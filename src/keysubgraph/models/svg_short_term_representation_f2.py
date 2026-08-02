"""Representation-level F2 fusion with a frozen SVG/G2 logit anchor."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


SVG_SHORT_TERM_REPRESENTATION_F2_MODEL_NAME = (
    "svg_short_term_representation_f2"
)
SVG_SHORT_TERM_REPRESENTATION_F2_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SVGShortTermRepresentationF2Config:
    short_term_representation_dim: int
    g2_representation_dim: int
    residual_hidden_dim: int = 64
    dropout: float = 0.20
    initial_gate: float = 0.01

    def __post_init__(self) -> None:
        if (
            int(self.short_term_representation_dim) < 1
            or int(self.g2_representation_dim) < 1
            or int(self.residual_hidden_dim) < 1
        ):
            raise ValueError("representation F2 dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("representation F2 dropout must lie in [0,1)")
        if not 0.0 < float(self.initial_gate) < 1.0:
            raise ValueError("representation F2 initial gate must lie in (0,1)")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SVGShortTermRepresentationF2Output:
    logits: torch.Tensor
    residual_logits: torch.Tensor
    residual_probability_prediction: torch.Tensor
    gate: torch.Tensor


class SVGShortTermRepresentationF2(nn.Module):
    """Predict a short-term residual while preserving frozen G2 at init.

    The final binary logit is ``g2_logit + gate * delta_short``.  The
    residual head consumes the short-term hidden representation rather than
    its final probability, which is the promoted representation-level F2
    model.  Its last layer is zero initialized and the gate starts at 0.01,
    so construction is exactly equal to the frozen G2 anchor.
    """

    model_name = SVG_SHORT_TERM_REPRESENTATION_F2_MODEL_NAME

    def __init__(self, config: SVGShortTermRepresentationF2Config) -> None:
        super().__init__()
        self.config = config
        self.short_term_normalization = nn.LayerNorm(
            int(config.short_term_representation_dim)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(
                int(config.short_term_representation_dim),
                int(config.residual_hidden_dim),
            ),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.residual_hidden_dim), 1),
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    float(config.initial_gate)
                    / (1.0 - float(config.initial_gate))
                ),
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            self.residual_head[-1].weight.zero_()
            self.residual_head[-1].bias.zero_()

    def forward(
        self,
        g2_anchor_logit: torch.Tensor,
        short_term_representation: torch.Tensor,
    ) -> SVGShortTermRepresentationF2Output:
        if g2_anchor_logit.ndim != 1:
            raise ValueError("representation F2 anchor logits must be [B]")
        if tuple(short_term_representation.shape) != (
            int(g2_anchor_logit.shape[0]),
            int(self.config.short_term_representation_dim),
        ):
            raise ValueError("representation F2 short-term shape mismatch")
        if not bool(torch.isfinite(g2_anchor_logit).all()) or not bool(
            torch.isfinite(short_term_representation).all()
        ):
            raise ValueError("representation F2 inputs must be finite")
        residual = self.residual_head(
            self.short_term_normalization(short_term_representation)
        ).squeeze(-1)
        gate = torch.sigmoid(self.gate_logit)
        logits = g2_anchor_logit.detach() + gate * residual
        return SVGShortTermRepresentationF2Output(
            logits=logits,
            residual_logits=residual,
            residual_probability_prediction=torch.tanh(residual),
            gate=gate,
        )

