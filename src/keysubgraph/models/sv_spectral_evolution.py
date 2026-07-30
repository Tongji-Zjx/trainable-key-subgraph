"""Static-spectral anchor plus a learned spectral evolution residual."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from keysubgraph.models.masked_tcn import MaskedTCNEncoder
from keysubgraph.models.sv_signed_gin import SVSignedGINClassifier


SV_SPECTRAL_EVOLUTION_VARIANT = "static_spectral_neural_evolution"


@dataclass(frozen=True)
class SVSpectralEvolutionConfig:
    transition_dim: int = 32
    transition_hidden_dim: int = 32
    temporal_hidden_dim: int = 32
    dynamic_projection_dim: int = 16
    kernel_size: int = 3
    dilations: Tuple[int, ...] = (1, 2)
    dropout: float = 0.10
    residual_gate_initial_logit: float = -2.197224577

    def __post_init__(self) -> None:
        if (
            self.transition_dim != 32
            or min(
                self.transition_hidden_dim,
                self.temporal_hidden_dim,
                self.dynamic_projection_dim,
                self.kernel_size,
            )
            < 1
            or not self.dilations
            or any(int(value) < 1 for value in self.dilations)
        ):
            raise ValueError("invalid spectral evolution dimensions")
        if self.kernel_size % 2 == 0:
            raise ValueError("spectral evolution kernel must be odd")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("spectral evolution dropout must lie in [0,1)")
        if not -20.0 <= self.residual_gate_initial_logit <= 0.0:
            raise ValueError("spectral evolution gate must start non-positive")


@dataclass(frozen=True)
class SVSpectralEvolutionSampleInput:
    sample_key: str
    label: int
    static_features: torch.Tensor
    transition_segments: Tuple[torch.Tensor, ...]

    def to(self, device) -> "SVSpectralEvolutionSampleInput":
        return SVSpectralEvolutionSampleInput(
            sample_key=self.sample_key,
            label=int(self.label),
            static_features=self.static_features.to(device),
            transition_segments=tuple(
                segment.to(device) for segment in self.transition_segments
            ),
        )


@dataclass(frozen=True)
class SVSpectralEvolutionBatch:
    samples: Tuple[SVSpectralEvolutionSampleInput, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(
            [sample.label for sample in self.samples], dtype=torch.long
        )

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(sample.sample_key for sample in self.samples)

    def to(self, device) -> "SVSpectralEvolutionBatch":
        return SVSpectralEvolutionBatch(
            tuple(sample.to(device) for sample in self.samples)
        )


@dataclass(frozen=True)
class SVSpectralEvolutionOutput:
    logits: torch.Tensor
    anchor_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    dynamic_representation: torch.Tensor
    residual_gate: torch.Tensor
    transition_counts: torch.Tensor
    diagnostics: Dict[str, Any]


class SVSpectralEvolutionClassifier(nn.Module):
    """Freeze S and learn only an order-sensitive residual expert."""

    model_name = SV_SPECTRAL_EVOLUTION_VARIANT

    def __init__(
        self,
        static_anchor: SVSignedGINClassifier,
        config: Optional[SVSpectralEvolutionConfig] = None,
    ) -> None:
        super().__init__()
        if static_anchor.config.variant != "static_spectral_only":
            raise ValueError(
                "spectral evolution requires a static_spectral_only anchor"
            )
        self.config = config or SVSpectralEvolutionConfig()
        self.static_anchor = static_anchor
        for parameter in self.static_anchor.parameters():
            parameter.requires_grad_(False)
        self.static_anchor.eval()
        self.transition_projection = nn.Sequential(
            nn.Linear(
                self.config.transition_dim,
                self.config.transition_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.config.transition_hidden_dim),
        )
        self.temporal_encoder = MaskedTCNEncoder(
            input_dim=self.config.transition_hidden_dim,
            hidden_dim=self.config.temporal_hidden_dim,
            kernel_size=self.config.kernel_size,
            dilations=self.config.dilations,
            dropout=self.config.dropout,
        )
        self.dynamic_projection = nn.Sequential(
            nn.Linear(
                2 * self.config.temporal_hidden_dim,
                self.config.dynamic_projection_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.config.dynamic_projection_dim),
        )
        self.dynamic_classifier = nn.Sequential(
            nn.Linear(
                self.config.dynamic_projection_dim,
                self.config.dynamic_projection_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dynamic_projection_dim, 2),
        )
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(
                self.config.residual_gate_initial_logit,
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            output = self.dynamic_classifier[-1]
            output.weight.zero_()
            output.bias.zero_()

    def train(self, mode: bool = True):
        super().train(mode)
        self.static_anchor.eval()
        return self

    def config_dict(self) -> Dict[str, Any]:
        values = asdict(self.config)
        values["dilations"] = list(self.config.dilations)
        return values

    def _anchor_logits(self, static: torch.Tensor) -> torch.Tensor:
        projected = self.static_anchor.static_projection(static[:, :16])
        return self.static_anchor.branch_classifiers[
            "static_spectral"
        ](projected)

    def _dynamic_representation(
        self, batch: SVSpectralEvolutionBatch, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        segments = []
        owners = []
        counts = []
        for sample_index, sample in enumerate(batch):
            count = 0
            for segment in sample.transition_segments:
                if (
                    segment.ndim != 2
                    or segment.shape[1] != self.config.transition_dim
                    or segment.shape[0] < 1
                ):
                    raise ValueError("invalid spectral transition segment")
                segments.append(self.transition_projection(segment))
                owners.append(sample_index)
                count += int(segment.shape[0])
            counts.append(count)
        pooled = reference.new_zeros(
            (
                len(batch),
                2 * self.config.temporal_hidden_dim,
            )
        )
        if segments:
            _, encoded, mask = self.temporal_encoder.forward_list(segments)
            for sample_index in range(len(batch)):
                values = [
                    encoded[index][mask[index]]
                    for index, owner in enumerate(owners)
                    if owner == sample_index
                ]
                if not values:
                    continue
                concatenated = torch.cat(values, dim=0)
                pooled[sample_index] = torch.cat(
                    (
                        concatenated.mean(dim=0),
                        concatenated.std(dim=0, unbiased=False),
                    ),
                    dim=-1,
                )
        count_tensor = torch.tensor(
            counts, dtype=torch.long, device=reference.device
        )
        projected = self.dynamic_projection(pooled)
        projected = projected * (count_tensor > 0).to(
            projected.dtype
        )[:, None]
        return projected, count_tensor

    def forward(
        self, batch: SVSpectralEvolutionBatch
    ) -> SVSpectralEvolutionOutput:
        if len(batch) < 1:
            raise ValueError("spectral evolution batch cannot be empty")
        static = torch.stack(
            [sample.static_features for sample in batch], dim=0
        )
        if tuple(static.shape[1:]) != (28,):
            raise ValueError("static anchor input must be 28-D")
        with torch.no_grad():
            anchor_logits = self._anchor_logits(static)
        representation, counts = self._dynamic_representation(
            batch, static
        )
        dynamic_logits = self.dynamic_classifier(representation)
        dynamic_logits = dynamic_logits * (counts > 0).to(
            dynamic_logits.dtype
        )[:, None]
        gate = torch.sigmoid(self.residual_gate_logit)
        logits = anchor_logits + gate * dynamic_logits
        return SVSpectralEvolutionOutput(
            logits=logits,
            anchor_logits=anchor_logits,
            dynamic_logits=dynamic_logits,
            dynamic_representation=representation,
            residual_gate=gate,
            transition_counts=counts,
            diagnostics={
                "residual_gate": float(gate.detach().cpu().item()),
                "mean_transition_count": float(
                    counts.to(torch.float32).mean().detach().cpu().item()
                ),
            },
        )
