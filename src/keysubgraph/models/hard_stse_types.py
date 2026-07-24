"""Configuration and typed outputs for Hard-STSE-Temporal-SGW."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


HARD_STSE_VARIANTS = ("M0", "M1", "M2", "M3")
HARD_SELECTION_MODES = ("full", "random", "learned")


@dataclass(frozen=True)
class HardSelectionSchedule:
    start_node_ratio: float = 0.90
    start_edge_ratio: float = 0.80
    target_node_ratio: float = 0.50
    target_edge_ratio: float = 0.30
    high_retention_epochs: int = 10
    anneal_end_epoch: int = 30

    def __post_init__(self) -> None:
        for value in (
            self.start_node_ratio,
            self.start_edge_ratio,
            self.target_node_ratio,
            self.target_edge_ratio,
        ):
            if value <= 0.0 or value > 1.0:
                raise ValueError("hard-selection ratios must lie in (0, 1]")
        if self.start_node_ratio < self.target_node_ratio:
            raise ValueError("node retention schedule must not increase")
        if self.start_edge_ratio < self.target_edge_ratio:
            raise ValueError("edge retention schedule must not increase")
        if self.high_retention_epochs < 0:
            raise ValueError("high-retention epoch count cannot be negative")
        if self.anneal_end_epoch <= self.high_retention_epochs:
            raise ValueError("anneal end must follow high-retention training")

    def ratios(self, epoch: int) -> Tuple[float, float]:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        if epoch <= self.high_retention_epochs:
            return self.start_node_ratio, self.start_edge_ratio
        if epoch >= self.anneal_end_epoch:
            return self.target_node_ratio, self.target_edge_ratio
        span = float(self.anneal_end_epoch - self.high_retention_epochs)
        progress = float(epoch - self.high_retention_epochs) / span
        node = self.start_node_ratio + progress * (
            self.target_node_ratio - self.start_node_ratio
        )
        edge = self.start_edge_ratio + progress * (
            self.target_edge_ratio - self.start_edge_ratio
        )
        return node, edge


@dataclass(frozen=True)
class HardSTSEConfig:
    variant: str = "M0"
    selection_mode: str = "full"
    use_sgw: bool = False
    node_extractor_feature_dim: int = 15
    edge_extractor_base_dim: int = 6
    classifier_node_feature_dim: int = 14
    classifier_edge_feature_dim: int = 7
    graph_statistic_dim: int = 14
    selector_node_hidden_dim: int = 64
    selector_edge_hidden_dim: int = 32
    node_hidden_dim: int = 96
    edge_hidden_dim: int = 64
    window_hidden_dim: int = 256
    window_output_dim: int = 128
    temporal_hidden_per_direction: int = 64
    neural_output_dim: int = 192
    spectral_core_dim: int = 18
    spectral_fixed_dim: int = 34
    spectral_sequence_dim: int = 32
    theory_output_dim: int = 66
    fusion_hidden_dims: Tuple[int, ...] = (128, 64)
    node_minimum: int = 2
    edge_minimum: int = 1
    dropout: float = 0.20
    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0
    epsilon: float = 1.0e-8
    exact_sgw_detach_from_selector: bool = True
    use_node_identity_embedding: bool = False
    use_raw_community_embedding: bool = False
    selection_schedule: HardSelectionSchedule = field(
        default_factory=HardSelectionSchedule
    )

    def __post_init__(self) -> None:
        if self.variant not in HARD_STSE_VARIANTS:
            raise ValueError("unsupported Hard-STSE experiment variant")
        if self.selection_mode not in HARD_SELECTION_MODES:
            raise ValueError("unsupported hard-selection mode")
        expected = {
            "M0": ("full", False),
            "M1": ("random", False),
            "M2": ("learned", False),
            "M3": ("learned", True),
        }
        if (self.selection_mode, self.use_sgw) != expected[self.variant]:
            raise ValueError("variant, selection mode and SGW flag disagree")
        dimensions = (
            self.node_extractor_feature_dim,
            self.edge_extractor_base_dim,
            self.classifier_node_feature_dim,
            self.classifier_edge_feature_dim,
            self.graph_statistic_dim,
            self.selector_node_hidden_dim,
            self.selector_edge_hidden_dim,
            self.node_hidden_dim,
            self.edge_hidden_dim,
            self.window_hidden_dim,
            self.window_output_dim,
            self.temporal_hidden_per_direction,
            self.neural_output_dim,
            self.spectral_core_dim,
            self.spectral_fixed_dim,
            self.spectral_sequence_dim,
            self.theory_output_dim,
            self.node_minimum,
            self.edge_minimum,
        ) + tuple(self.fusion_hidden_dims)
        if any(value < 1 for value in dimensions):
            raise ValueError("Hard-STSE dimensions and minimum budgets must be positive")
        if self.node_extractor_feature_dim != 15:
            raise ValueError("verified extractor node schema is 15-D")
        if self.edge_extractor_base_dim != 6:
            raise ValueError("verified extractor edge base schema is 6-D")
        if self.classifier_node_feature_dim != 14:
            raise ValueError("verified hard classifier node schema is 14-D")
        if self.classifier_edge_feature_dim != 7:
            raise ValueError("verified hard classifier edge schema is 7-D")
        if self.graph_statistic_dim != 14:
            raise ValueError("verified hard graph statistic schema is 14-D")
        if self.temporal_hidden_per_direction * 2 != self.window_output_dim:
            raise ValueError("BiGRU output must match the 128-D window state")
        if self.neural_output_dim != 3 * self.window_output_dim // 2:
            raise ValueError("neural output must be attention/mean/max projected to 192-D")
        if self.spectral_core_dim != 18 or self.spectral_fixed_dim != 34:
            raise ValueError("verified SGW core/fixed dimensions are 18/34")
        if self.theory_output_dim != (
            self.spectral_fixed_dim + self.spectral_sequence_dim
        ):
            raise ValueError("theory output must concatenate fixed and sequence SGW")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.laplacian_eta <= 0.0 or self.diffusion_time <= 0.0:
            raise ValueError("SGW scale parameters must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not self.exact_sgw_detach_from_selector:
            raise ValueError("verified first version detaches exact SGW at selection")
        if self.use_node_identity_embedding:
            raise ValueError("generic model forbids node identity embeddings")
        if self.use_raw_community_embedding:
            raise ValueError("generic model forbids raw community embeddings")


@dataclass(frozen=True)
class HardSelectionOutput:
    node_probabilities: torch.Tensor
    edge_probabilities: torch.Tensor
    hard_node_mask: torch.Tensor
    hard_edge_mask: torch.Tensor
    straight_through_node_mask: torch.Tensor
    straight_through_edge_mask: torch.Tensor
    requested_node_count: int
    original_edge_count: int
    candidate_edge_count: int
    requested_edge_count: int
    actual_node_count: int
    actual_edge_count: int
    selection_mode: str


@dataclass(frozen=True)
class HardWindowOutput:
    adjacency_st: torch.Tensor
    hard_node_mask: torch.Tensor
    hard_edge_mask: torch.Tensor
    straight_through_node_mask: torch.Tensor
    straight_through_edge_mask: torch.Tensor
    cropped_graph: Optional[Any]
    window_valid: bool
    selection: HardSelectionOutput


@dataclass(frozen=True)
class HardSTSENeuralOutput:
    window_embeddings: Tuple[torch.Tensor, ...]
    temporal_states: torch.Tensor
    temporal_mask: torch.Tensor
    representation: torch.Tensor
    attention: torch.Tensor


@dataclass(frozen=True)
class HardSTSETheoryOutput:
    core: torch.Tensor
    fixed: torch.Tensor
    sequence: torch.Tensor
    representation: torch.Tensor
    transition_mask: torch.Tensor
    exact_features_detached: bool


@dataclass(frozen=True)
class HardSTSEModelOutput:
    fusion_logits: torch.Tensor
    neural_logits: torch.Tensor
    theory_logits: Optional[torch.Tensor]
    neural_representation: torch.Tensor
    theory_representation: Optional[torch.Tensor]
    final_representation: torch.Tensor
    hard_windows: Tuple[Tuple[HardWindowOutput, ...], ...]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class HardSTSELossOutput:
    total: torch.Tensor
    classification: torch.Tensor
    fusion_ce: torch.Tensor
    neural_ce: torch.Tensor
    theory_ce: torch.Tensor
    node_budget: torch.Tensor
    edge_budget: torch.Tensor
    laplacian: torch.Tensor
    gw_proxy: torch.Tensor
    weights: Dict[str, float]
