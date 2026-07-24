"""Dual-channel NoCoord-STSE plus learned Hard-SGW classifier."""

from __future__ import absolute_import, division, print_function

from typing import Optional

import torch
from torch import nn

from keysubgraph.data.dual_sgw_scaler import DualSGWStandardizer
from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from .dual_hard_sgw_selector import DualHardSGWSelector
from .dual_sgw_proxy import DualSGWProxy
from .dual_stse_channel import ExistingNoCoordSTSEChannel
from .dual_stse_hard_sgw_types import (
    DualSTSEHardSGWConfig,
    DualSTSEHardSGWOutput,
)
from .exact_stse import ExactSTSEClassifier


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
    )


def _sgw_classifier(config: DualSTSEHardSGWConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.sgw_output_dim, config.sgw_projection_dim),
        nn.GELU(),
        nn.Dropout(config.fusion_dropout),
        nn.Linear(config.sgw_projection_dim, 2),
    )


class DualSTSEHardSGWClassifier(nn.Module):
    model_name = "dual_stse_hard_sgw"

    def __init__(
        self,
        config: Optional[DualSTSEHardSGWConfig] = None,
        stse_model: Optional[ExactSTSEClassifier] = None,
    ) -> None:
        super().__init__()
        self.config = config or DualSTSEHardSGWConfig()
        self.stse_channel = ExistingNoCoordSTSEChannel(stse_model)
        self.selector = DualHardSGWSelector(self.config)
        self.proxy = DualSGWProxy(self.config)
        self.stse_normalization = nn.LayerNorm(
            self.config.stse_output_dim
        )
        self.stse_projection = _projection(
            self.config.stse_output_dim,
            self.config.stse_projection_dim,
        )
        self.sgw_projection = _projection(
            self.config.sgw_output_dim,
            self.config.sgw_projection_dim,
        )
        self.sgw_auxiliary_head = _sgw_classifier(self.config)
        self.selector_proxy_head = _sgw_classifier(self.config)
        self.fusion_head = nn.Sequential(
            nn.Linear(
                self.config.fusion_input_dim,
                self.config.fusion_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.fusion_dropout),
            nn.Linear(self.config.fusion_hidden_dim, 2),
        )
        self.register_buffer(
            "sgw_scaler_mean",
            torch.zeros(self.config.sgw_output_dim),
        )
        self.register_buffer(
            "sgw_scaler_scale",
            torch.ones(self.config.sgw_output_dim),
        )
        self.register_buffer(
            "sgw_scaler_fitted", torch.tensor(False, dtype=torch.bool)
        )
        self.sgw_scaler_sample_count = 0
        self.sgw_scaler_protocol_sha256 = ""
        self.sgw_scaler_selector_checkpoint_sha256 = ""
        self.sgw_scaler_selection_mode = ""
        self.sgw_scaler_selection_seed = -1

    def set_sgw_standardizer(
        self, scaler: DualSGWStandardizer
    ) -> None:
        if tuple(scaler.mean.shape) != (
            self.config.sgw_output_dim,
        ):
            raise ValueError("dual model received an invalid SGW scaler")
        self.sgw_scaler_mean.copy_(scaler.mean.to(self.sgw_scaler_mean))
        self.sgw_scaler_scale.copy_(
            scaler.scale.to(self.sgw_scaler_scale)
        )
        self.sgw_scaler_fitted.fill_(True)
        self.sgw_scaler_sample_count = scaler.sample_count
        self.sgw_scaler_protocol_sha256 = scaler.protocol_sha256
        self.sgw_scaler_selector_checkpoint_sha256 = (
            scaler.selector_checkpoint_sha256
        )
        self.sgw_scaler_selection_mode = scaler.selection_mode
        self.sgw_scaler_selection_seed = scaler.selection_seed

    def set_stage_trainability(self, stage: str) -> None:
        if stage not in ("selector_proxy", "sgw_classifier", "fusion"):
            raise ValueError("unsupported dual training stage")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "selector_proxy":
            for module in (self.selector, self.selector_proxy_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        elif stage == "sgw_classifier":
            for parameter in self.sgw_auxiliary_head.parameters():
                parameter.requires_grad_(True)
        else:
            for module in (
                self.stse_projection,
                self.sgw_projection,
                self.fusion_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _standardize_sgw(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        if not bool(self.sgw_scaler_fitted.item()):
            raise RuntimeError(
                "exact SGW features require a fitted train-only scaler"
            )
        if features.ndim != 2 or features.shape[-1] != (
            self.config.sgw_output_dim
        ):
            raise ValueError("exact SGW batch must have shape [B,34]")
        return (
            features - self.sgw_scaler_mean.to(features)
        ) / self.sgw_scaler_scale.to(features)

    def forward(
        self,
        batch: ExactSTSEBatch,
        exact_sgw_features: Optional[torch.Tensor] = None,
        compute_selector_proxy: bool = False,
        selection_mode: str = "learned",
        random_seed: int = 42,
    ) -> DualSTSEHardSGWOutput:
        stse = self.stse_channel(batch)
        hard_windows = None
        proxy_logits = None
        proxy_output = None
        selection_diagnostics = {}
        if compute_selector_proxy:
            selection = self.selector(
                batch,
                selection_mode=selection_mode,
                random_seed=random_seed,
            )
            hard_windows = selection.hard_windows
            selection_diagnostics = selection.diagnostics
            proxy_output = self.proxy(batch, hard_windows)
            proxy_logits = self.selector_proxy_head(
                proxy_output.representation
            )

        sgw_logits = None
        sgw_representation = None
        fusion_representation = None
        fusion_logits = stse.logits
        if exact_sgw_features is not None:
            exact_sgw_features = exact_sgw_features.to(
                device=stse.representation.device,
                dtype=stse.representation.dtype,
            )
            standardized = self._standardize_sgw(exact_sgw_features)
            sgw_logits = self.sgw_auxiliary_head(standardized)
            stse_projected = self.stse_projection(
                self.stse_normalization(stse.representation)
            )
            sgw_projected = self.sgw_projection(standardized)
            fusion_representation = torch.cat(
                (stse_projected, sgw_projected), dim=-1
            )
            fusion_logits = self.fusion_head(fusion_representation)
            sgw_representation = standardized

        diagnostics = {
            "uses_coordinates": False,
            "uses_learned_temporal_encoder": False,
            "stse_representation_dim": int(
                stse.representation.shape[-1]
            ),
            "sgw_representation_dim": (
                int(sgw_representation.shape[-1])
                if sgw_representation is not None
                else None
            ),
            "fusion_representation_dim": (
                int(fusion_representation.shape[-1])
                if fusion_representation is not None
                else None
            ),
            "sgw_scaler_fitted": bool(
                self.sgw_scaler_fitted.item()
            ),
            "selection": selection_diagnostics,
            "proxy": proxy_output,
        }
        return DualSTSEHardSGWOutput(
            fusion_logits=fusion_logits,
            stse_logits=stse.logits,
            sgw_logits=sgw_logits,
            selector_proxy_logits=proxy_logits,
            stse_representation=stse.representation,
            sgw_representation=sgw_representation,
            fusion_representation=fusion_representation,
            hard_windows=hard_windows,
            diagnostics=diagnostics,
        )
