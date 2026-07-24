"""Typed contracts for the Dual-STSE-HardSGW experiment."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


DUAL_TRAINING_STAGES = (
    "selector_proxy",
    "sgw_classifier",
    "fusion",
)
DUAL_EXPERIMENT_VARIANTS = ("D0", "D1", "D2", "D3", "D4")


@dataclass(frozen=True)
class DualSTSEHardSGWConfig:
    """Frozen first-version architecture and experiment dimensions."""

    stse_input_dim: int = 18
    stse_output_dim: int = 64
    selector_node_feature_dim: int = 15
    selector_edge_base_dim: int = 6
    selector_node_hidden_dim: int = 64
    selector_edge_hidden_dim: int = 32
    selector_dropout: float = 0.10
    target_node_ratio: float = 0.50
    target_edge_ratio: float = 0.30
    node_minimum: int = 2
    edge_minimum: int = 1
    spectral_quantile_dim: int = 16
    sgw_core_dim: int = 18
    sgw_variation_dim: int = 16
    sgw_output_dim: int = 34
    stse_projection_dim: int = 64
    sgw_projection_dim: int = 64
    fusion_hidden_dim: int = 64
    fusion_dropout: float = 0.20
    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0
    epsilon: float = 1.0e-8
    exact_sgw_detached: bool = True
    use_learned_temporal_encoder: bool = False

    def __post_init__(self) -> None:
        dimensions = (
            self.stse_input_dim,
            self.stse_output_dim,
            self.selector_node_feature_dim,
            self.selector_edge_base_dim,
            self.selector_node_hidden_dim,
            self.selector_edge_hidden_dim,
            self.node_minimum,
            self.edge_minimum,
            self.spectral_quantile_dim,
            self.sgw_core_dim,
            self.sgw_variation_dim,
            self.sgw_output_dim,
            self.stse_projection_dim,
            self.sgw_projection_dim,
            self.fusion_hidden_dim,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("Dual-STSE dimensions must be positive")
        if self.stse_input_dim != 18 or self.stse_output_dim != 64:
            raise ValueError("the validated NoCoord-STSE contract is 18 -> 64")
        if (
            self.selector_node_feature_dim != 15
            or self.selector_edge_base_dim != 6
        ):
            raise ValueError("the verified selector schemas are 15-D and 6-D")
        if self.spectral_quantile_dim != 16:
            raise ValueError("Dual-STSE requires 16 spectral quantiles")
        if self.sgw_core_dim != self.spectral_quantile_dim + 2:
            raise ValueError("SGW core must be 16 directional values plus 2 speeds")
        if self.sgw_output_dim != (
            self.sgw_core_dim + self.sgw_variation_dim
        ):
            raise ValueError("SGW output must concatenate core and variation")
        for ratio in (self.target_node_ratio, self.target_edge_ratio):
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError("selection ratios must lie in (0,1]")
        for dropout in (self.selector_dropout, self.fusion_dropout):
            if dropout < 0.0 or dropout >= 1.0:
                raise ValueError("dropout must lie in [0,1)")
        if (
            self.laplacian_eta <= 0.0
            or self.diffusion_time <= 0.0
            or self.epsilon <= 0.0
        ):
            raise ValueError("SGW scale parameters and epsilon must be positive")
        if not self.exact_sgw_detached:
            raise ValueError("exact SGW must remain detached from the selector")
        if self.use_learned_temporal_encoder:
            raise ValueError("Dual-STSE forbids learned temporal encoders")

    @property
    def fusion_input_dim(self) -> int:
        return self.stse_projection_dim + self.sgw_projection_dim


@dataclass(frozen=True)
class DualSTSEHardSGWOutput:
    fusion_logits: torch.Tensor
    stse_logits: torch.Tensor
    sgw_logits: Optional[torch.Tensor]
    selector_proxy_logits: Optional[torch.Tensor]
    stse_representation: torch.Tensor
    sgw_representation: Optional[torch.Tensor]
    fusion_representation: Optional[torch.Tensor]
    hard_windows: Optional[Tuple[Tuple[Any, ...], ...]]
    diagnostics: Dict[str, Any]

