"""Detached exact spectral--GW branch for Hard-STSE-Temporal-SGW."""

from __future__ import absolute_import, division, print_function

from typing import List, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.theory.tg_features import (
    SGWFeatureExtractor,
    SGWTheoryFeatureConfig,
)
from .hard_stse_types import (
    HardSTSEConfig,
    HardSTSETheoryOutput,
    HardWindowOutput,
)


class HardSTSESGWBranch(nn.Module):
    """Compute exact hard-graph descriptors, then learn their chronology.

    Eigendecomposition and GW optimization are intentionally performed under
    ``no_grad`` by :class:`SGWFeatureExtractor`.  Therefore exact SGW values
    never create a surrogate gradient for the discrete selector.  The GRU and
    downstream heads remain trainable.
    """

    def __init__(self, config: HardSTSEConfig) -> None:
        super().__init__()
        if not config.use_sgw:
            raise ValueError("the exact SGW branch is only valid for M3")
        self.config = config
        theory_config = SGWTheoryFeatureConfig(
            laplacian_eta=config.laplacian_eta,
            diffusion_time=config.diffusion_time,
        )
        self.extractor = SGWFeatureExtractor(theory_config=theory_config)
        self.sequence_gru = nn.GRU(
            input_size=config.spectral_core_dim,
            hidden_size=config.spectral_sequence_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.register_buffer(
            "fixed_mean", torch.zeros(config.spectral_fixed_dim)
        )
        self.register_buffer(
            "fixed_scale", torch.ones(config.spectral_fixed_dim)
        )
        self.register_buffer(
            "fixed_scaler_fitted", torch.tensor(False, dtype=torch.bool)
        )

    def set_fixed_standardizer(
        self, mean: torch.Tensor, scale: torch.Tensor
    ) -> None:
        expected = (self.config.spectral_fixed_dim,)
        if tuple(mean.shape) != expected or tuple(scale.shape) != expected:
            raise ValueError("SGW fixed-feature standardizer has invalid shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("SGW fixed-feature standardizer must be finite")
        if bool((scale <= 0.0).any()):
            raise ValueError("SGW fixed-feature scales must be positive")
        self.fixed_mean.copy_(mean.detach().to(self.fixed_mean))
        self.fixed_scale.copy_(scale.detach().to(self.fixed_scale))
        self.fixed_scaler_fitted.fill_(True)

    def fit_fixed_standardizer(self, fixed_features: torch.Tensor) -> None:
        if fixed_features.ndim != 2 or fixed_features.shape[1] != (
            self.config.spectral_fixed_dim
        ):
            raise ValueError("training SGW features must have shape [B, 34]")
        if fixed_features.shape[0] < 1:
            raise ValueError("cannot fit an empty SGW standardizer")
        mean = fixed_features.detach().mean(dim=0)
        variance = (
            fixed_features.detach() - mean.unsqueeze(0)
        ).square().mean(dim=0)
        scale = torch.sqrt(variance + self.config.epsilon)
        self.set_fixed_standardizer(mean, scale)

    def _extract_one(
        self,
        windows: Sequence[HardWindowOutput],
        time_values: Sequence[float],
    ):
        cropped = tuple(
            window.cropped_graph if window.window_valid else None
            for window in windows
        )
        return self.extractor.compute_hard_graph_sequence(
            cropped, time_values
        )

    def forward(
        self,
        hard_windows: Sequence[Sequence[HardWindowOutput]],
        time_values: Sequence[Sequence[float]],
    ) -> HardSTSETheoryOutput:
        if len(hard_windows) != len(time_values) or len(hard_windows) < 1:
            raise ValueError("hard graphs and time values must be batch-aligned")
        extracted = []
        for windows, times in zip(hard_windows, time_values):
            if len(windows) != len(times) or len(windows) < 1:
                raise ValueError("each hard sequence must align with its times")
            extracted.append(self._extract_one(windows, times))

        reference = next(self.parameters())
        core = torch.stack(
            [item.h_core.to(reference) for item in extracted], dim=0
        ).detach()
        fixed_raw = torch.stack(
            [item.h_classification.to(reference) for item in extracted],
            dim=0,
        ).detach()
        fixed = (
            fixed_raw - self.fixed_mean.unsqueeze(0)
        ) / self.fixed_scale.unsqueeze(0)

        sequence_items: List[torch.Tensor] = []
        transition_masks = []
        maximum_transitions = max(
            int(item.transition_mask.numel()) for item in extracted
        )
        padded_mask = torch.zeros(
            (len(extracted), maximum_transitions),
            dtype=torch.bool,
            device=reference.device,
        )
        for sample_index, item in enumerate(extracted):
            mask = item.transition_mask.to(
                device=reference.device, dtype=torch.bool
            )
            transition_masks.append(mask)
            if mask.numel():
                padded_mask[sample_index, : mask.numel()] = mask
            valid = item.transition_features.to(reference)[mask].detach()
            if valid.shape[0] < 1:
                sequence_items.append(
                    reference.new_zeros(
                        (self.config.spectral_sequence_dim,)
                    )
                )
                continue
            _, hidden = self.sequence_gru(valid.unsqueeze(0))
            sequence_items.append(hidden[-1, 0])
        sequence = torch.stack(sequence_items, dim=0)
        representation = torch.cat((fixed, sequence), dim=-1)
        if tuple(representation.shape[1:]) != (
            self.config.theory_output_dim,
        ):
            raise RuntimeError("M3 theory representation has invalid shape")
        convergence = tuple(
            tuple(item.gw_solver_converged) for item in extracted
        )
        return HardSTSETheoryOutput(
            core=core,
            fixed=fixed,
            sequence=sequence,
            representation=representation,
            transition_mask=padded_mask,
            exact_features_detached=(
                not core.requires_grad and not fixed_raw.requires_grad
            ),
        ), {
            "fixed_raw": fixed_raw,
            "gw_solver_converged": convergence,
            "fixed_scaler_fitted": bool(self.fixed_scaler_fitted.item()),
        }
