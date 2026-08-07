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
DUAL_SELECTOR_ARCHITECTURES = ("legacy_mlp", "theory_multi_object")


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
    selector_architecture: str = "legacy_mlp"
    selector_graph_layers: int = 2
    selector_spectral_dim: int = 8
    selector_spectral_cache: bool = True
    selector_fast_runtime: bool = False
    selector_object_overlap_minimum: float = 0.05
    selector_object_overlap_maximum: float = 0.30
    selector_object_temporal_state: bool = False
    selector_temporal_confidence_threshold: float = 0.25
    target_node_ratio: float = 0.50
    target_edge_ratio: float = 0.30
    node_minimum: int = 2
    edge_minimum: int = 1
    critical_subgraph_count: int = 5
    critical_candidate_multiplier: int = 2
    critical_overlap_penalty: float = 0.25
    critical_diversity_enabled: bool = False
    critical_node_ratio_per_object: float = 0.10
    critical_node_reuse_decay: float = 0.25
    critical_edge_reuse_decay: float = 0.10
    critical_max_node_overlap: float = 0.40
    critical_max_edge_overlap: float = 0.25
    critical_min_unique_node_fraction: float = 0.50
    critical_quality_floor_ratio: float = 0.80
    critical_min_seed_distance: float = 0.15
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
            self.selector_graph_layers,
            self.selector_spectral_dim,
            self.node_minimum,
            self.edge_minimum,
            self.critical_subgraph_count,
            self.critical_candidate_multiplier,
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
        if self.selector_architecture not in DUAL_SELECTOR_ARCHITECTURES:
            raise ValueError("unsupported dual selector architecture")
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
        if self.critical_overlap_penalty < 0.0:
            raise ValueError("critical overlap penalty cannot be negative")
        if not (
            0.0
            <= self.selector_object_overlap_minimum
            <= self.selector_object_overlap_maximum
            <= 1.0
        ):
            raise ValueError("selector object overlap interval is invalid")
        if not 0.0 <= self.selector_temporal_confidence_threshold <= 1.0:
            raise ValueError("selector temporal confidence is invalid")
        if (
            self.selector_architecture == "theory_multi_object"
            and self.critical_subgraph_count < 2
        ):
            raise ValueError("theory selector requires at least two objects")
        if (
            self.selector_architecture == "legacy_mlp"
            and self.selector_object_temporal_state
        ):
            raise ValueError("legacy selector has no object temporal state")
        diversity_ratios = (
            self.critical_node_ratio_per_object,
            self.critical_node_reuse_decay,
            self.critical_edge_reuse_decay,
            self.critical_max_node_overlap,
            self.critical_max_edge_overlap,
            self.critical_min_unique_node_fraction,
            self.critical_quality_floor_ratio,
        )
        if any(value <= 0.0 or value > 1.0 for value in diversity_ratios):
            raise ValueError("critical diversity ratios must lie in (0,1]")
        if self.critical_min_seed_distance < 0.0:
            raise ValueError("critical minimum seed distance cannot be negative")
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
    selector_soft_proxy_logits: Optional[torch.Tensor] = None
    selector_hard_proxy_logits: Optional[torch.Tensor] = None
    hard_subgraphs: Optional[Tuple[Any, ...]] = None
    trajectory_sets: Optional[Tuple[Any, ...]] = None


@dataclass(frozen=True)
class DualSoftWindowOutput:
    """Same-node differentiable signed soft graph used during selection."""

    adjacency_soft: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    window_valid: bool
