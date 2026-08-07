"""Hard graph selection channel shared by D1--D4."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.features.hard_stse_extractor_features import (
    HardSTSEExtractorFeatureBuilder,
)
from keysubgraph.features.hard_stse_hard_graph import build_hard_stse_window
from .dual_stse_hard_sgw_types import DualSTSEHardSGWConfig
from .dual_stse_hard_sgw_types import DualSoftWindowOutput
from .hard_stse_selector import HardSTSEScorer, select_hard_stse_window
from .fixed_k_subgraph_selector import (
    select_fixed_k_subgraphs,
    select_object_conditioned_subgraphs,
)
from .dynamic_subgraph_tracking import build_dynamic_trajectories
from .exploration_medoid_selector import (
    ExplorationObjectCandidate,
    candidate_quality,
    select_exploration_medoids,
    singleton_memory,
    stack_real_candidate_memories,
)
from .theory_multi_object_selector import (
    MultiObjectTemporalMemory,
    TheoryGuidedMultiObjectScorer,
)
from .hard_stse_types import (
    HardSelectionSchedule,
    HardSTSEConfig,
    HardWindowOutput,
)


@dataclass(frozen=True)
class DualHardSelectionOutput:
    hard_windows: Tuple[Tuple[HardWindowOutput, ...], ...]
    hard_subgraphs: Tuple[Tuple[Tuple[Optional[HardWindowOutput], ...], ...], ...]
    soft_windows: Tuple[Tuple[DualSoftWindowOutput, ...], ...]
    trajectory_sets: Tuple[Optional[Any], ...]
    diagnostics: Dict[str, Any]


def _community_coverage(mask: torch.Tensor, communities: torch.Tensor) -> float:
    labels = torch.unique(communities, sorted=True)
    if labels.numel() < 1:
        return 0.0
    covered = sum(
        bool(mask[communities == label].any()) for label in labels
    )
    return float(covered) / float(labels.numel())


def _build_soft_window(
    adjacency: torch.Tensor,
    edge_presence_mask: torch.Tensor,
    selection,
) -> DualSoftWindowOutput:
    """Build A_ij p_i p_j p_ij without deleting any original node."""

    node_probabilities = selection.node_probabilities.to(adjacency)
    edge_probabilities = selection.edge_probabilities.to(adjacency)
    edge_mask = edge_presence_mask.to(
        device=adjacency.device, dtype=torch.bool
    )
    edge_mask = edge_mask & edge_mask.transpose(0, 1)
    edge_mask = edge_mask.clone()
    edge_mask.fill_diagonal_(False)
    adjacency_soft = (
        adjacency
        * node_probabilities[:, None]
        * node_probabilities[None, :]
        * edge_probabilities
        * edge_mask.to(adjacency.dtype)
    )
    adjacency_soft = 0.5 * (
        adjacency_soft + adjacency_soft.transpose(0, 1)
    )
    adjacency_soft = adjacency_soft.clone()
    adjacency_soft.fill_diagonal_(0.0)
    node_mask = torch.ones(
        adjacency.shape[0],
        dtype=torch.bool,
        device=adjacency.device,
    )
    return DualSoftWindowOutput(
        adjacency_soft=adjacency_soft,
        node_mask=node_mask,
        edge_mask=edge_mask,
        window_valid=bool(
            adjacency.shape[0] >= 2
            and torch.triu(edge_mask, diagonal=1).any()
        ),
    )


def _align_hard_history(
    memory: Optional[MultiObjectTemporalMemory],
    current_node_ids: Tuple[str, ...],
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Align previous discrete objects to current ROI order for hysteresis."""

    if memory is None or memory.hard_node_masks is None:
        return None, None, None
    current = tuple(str(value) for value in current_node_ids)
    lookup = {value: index for index, value in enumerate(current)}
    object_count = int(memory.hard_node_masks.shape[0])
    count = len(current)
    nodes = torch.zeros(
        object_count, count, dtype=torch.bool, device=device
    )
    edges = torch.zeros(
        object_count, count, count, dtype=torch.bool, device=device
    )
    mapping = []
    for previous_index, node_id in enumerate(memory.node_ids):
        current_index = lookup.get(str(node_id))
        if current_index is not None:
            mapping.append((previous_index, current_index))
    if mapping:
        previous_indices = torch.tensor(
            [item[0] for item in mapping], dtype=torch.long, device=device
        )
        current_indices = torch.tensor(
            [item[1] for item in mapping], dtype=torch.long, device=device
        )
        source_nodes = memory.hard_node_masks.to(device)
        source_edges = memory.hard_edge_masks.to(device)
        nodes[:, current_indices] = source_nodes.index_select(
            1, previous_indices
        )
        selected_edges = source_edges.index_select(
            1, previous_indices
        ).index_select(2, previous_indices)
        edges[:, current_indices[:, None], current_indices[None, :]] = (
            selected_edges
        )
    seeds = torch.full(
        (object_count,), -1, dtype=torch.long, device=device
    )
    if memory.seed_indices is not None:
        previous_seeds = memory.seed_indices.detach().cpu().tolist()
        for object_index, previous_seed in enumerate(previous_seeds):
            if 0 <= int(previous_seed) < len(memory.node_ids):
                seeds[object_index] = lookup.get(
                    str(memory.node_ids[int(previous_seed)]), -1
                )
    return nodes, edges, seeds


def _exploration_window_count(
    total: int,
    minimum: int,
    maximum: int,
    fraction: float,
) -> int:
    """Choose a short burn-in without assuming a fixed scan length."""

    if total < 1:
        return 0
    requested = int(math.ceil(float(fraction) * float(total)))
    return min(int(total), max(int(minimum), min(int(maximum), requested)))


def _cosine_history_strength(
    time_index: int,
    exploration_windows: int,
    ramp_windows: int,
    minimum_strength: float = 0.0,
) -> float:
    """Smoothly turn causal history on after retrospective exploration."""

    if time_index < exploration_windows:
        return 0.0
    progress = min(
        1.0,
        float(time_index - exploration_windows + 1)
        / float(max(1, ramp_windows)),
    )
    cosine = 0.5 - 0.5 * math.cos(math.pi * progress)
    minimum = min(1.0, max(0.0, float(minimum_strength)))
    return minimum + (1.0 - minimum) * cosine


class DualHardSGWSelector(nn.Module):
    """Legacy or theory-guided signed multi-object hard graph selector."""

    def __init__(
        self, config: Optional[DualSTSEHardSGWConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or DualSTSEHardSGWConfig()
        scorer_config = HardSTSEConfig(
            variant="M2",
            selection_mode="learned",
            use_sgw=False,
            selector_node_hidden_dim=self.config.selector_node_hidden_dim,
            selector_edge_hidden_dim=self.config.selector_edge_hidden_dim,
            dropout=self.config.selector_dropout,
            node_minimum=self.config.node_minimum,
            edge_minimum=self.config.edge_minimum,
            selection_schedule=HardSelectionSchedule(
                start_node_ratio=self.config.target_node_ratio,
                start_edge_ratio=self.config.target_edge_ratio,
                target_node_ratio=self.config.target_node_ratio,
                target_edge_ratio=self.config.target_edge_ratio,
                high_retention_epochs=0,
                anneal_end_epoch=1,
            ),
        )
        self.feature_builder = HardSTSEExtractorFeatureBuilder(
            epsilon=self.config.epsilon
        )
        if self.config.selector_architecture == "legacy_mlp":
            self.scorer = HardSTSEScorer(scorer_config)
        else:
            self.scorer = TheoryGuidedMultiObjectScorer(
                node_feature_dim=self.config.selector_node_feature_dim,
                edge_feature_dim=self.config.selector_edge_base_dim,
                hidden_dim=self.config.selector_node_hidden_dim,
                edge_hidden_dim=self.config.selector_edge_hidden_dim,
                object_count=self.config.critical_subgraph_count,
                spectral_dim=self.config.selector_spectral_dim,
                graph_layers=self.config.selector_graph_layers,
                dropout=self.config.selector_dropout,
                overlap_minimum=(
                    self.config.selector_object_overlap_minimum
                ),
                overlap_maximum=(
                    self.config.selector_object_overlap_maximum
                ),
                target_object_ratio=(
                    self.config.critical_node_ratio_per_object
                ),
                temporal_confidence_threshold=(
                    self.config.selector_temporal_confidence_threshold
                ),
                structural_memory_enabled=(
                    self.config.selector_structural_temporal_memory
                ),
                memory_diffusion=self.config.selector_memory_diffusion,
                sinkhorn_temperature=(
                    self.config.selector_sinkhorn_temperature
                ),
                sinkhorn_iterations=(
                    self.config.selector_sinkhorn_iterations
                ),
                alignment_node_weight=(
                    self.config.selector_alignment_node_weight
                ),
                alignment_signed_edge_weight=(
                    self.config.selector_alignment_signed_edge_weight
                ),
                alignment_latent_weight=(
                    self.config.selector_alignment_latent_weight
                ),
                alignment_coordinate_weight=(
                    self.config.selector_alignment_coordinate_weight
                ),
                alignment_spectral_weight=(
                    self.config.selector_alignment_spectral_weight
                ),
                confidence_gated_history=(
                    self.config.selector_confidence_gated_history
                ),
                epsilon=self.config.epsilon,
            )
        # The signed-Laplacian basis is a detached positional encoding of a
        # frozen input window.  Keeping it outside state_dict makes caching a
        # pure runtime optimization while avoiding one eigendecomposition per
        # window on every epoch.
        self._spectral_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}

    def clear_spectral_cache(self) -> None:
        self._spectral_cache.clear()

    def _cached_spectral_features(
        self,
        sample_key: str,
        time_index: int,
        adjacency: torch.Tensor,
        edge_presence_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if (
            self.config.selector_architecture != "theory_multi_object"
            or not self.config.selector_spectral_cache
        ):
            return None
        key = (
            str(sample_key),
            int(time_index),
            int(adjacency.shape[0]),
            adjacency.device.type,
            adjacency.device.index,
            str(adjacency.dtype),
            int(self.config.selector_spectral_dim),
        )
        cached = self._spectral_cache.get(key)
        if cached is None:
            cached = self.scorer.encoder.spectral_features(
                adjacency, edge_presence_mask
            ).detach()
            self._spectral_cache[key] = cached
        return cached

    def _harden_object_fields(
        self,
        *,
        global_node_probabilities: torch.Tensor,
        global_edge_probabilities: torch.Tensor,
        object_node_probabilities: torch.Tensor,
        object_edge_probabilities: torch.Tensor,
        edge_presence_mask: torch.Tensor,
        communities: torch.Tensor,
        current_node_ids: Tuple[str, ...],
        previous_memory: Optional[MultiObjectTemporalMemory],
        anchor_memory: Optional[MultiObjectTemporalMemory],
        history_strength: float,
        alignment_confidence: Optional[torch.Tensor],
        edge_ratio: float,
    ):
        """Apply confidence- and phase-scaled hard temporal controls."""

        strength = min(1.0, max(0.0, float(history_strength)))
        previous_hard = (None, None, None)
        anchor_hard = (None, None, None)
        if previous_memory is not None and strength > 0.0:
            previous_hard = _align_hard_history(
                previous_memory,
                current_node_ids,
                global_node_probabilities.device,
            )
            if (
                previous_hard[0] is not None
                and self.config.selector_confidence_gated_history
                and alignment_confidence is not None
            ):
                keep = alignment_confidence.detach().to(
                    device=global_node_probabilities.device
                ) >= float(self.config.selector_temporal_confidence_threshold)
                previous_hard = (
                    previous_hard[0] & keep[:, None],
                    previous_hard[1] & keep[:, None, None],
                    torch.where(
                        keep,
                        previous_hard[2],
                        torch.full_like(previous_hard[2], -1),
                    ),
                )
        if anchor_memory is not None:
            anchor_hard = _align_hard_history(
                anchor_memory,
                current_node_ids,
                global_node_probabilities.device,
            )
        node_retention = self.config.critical_node_entry_threshold - strength * (
            self.config.critical_node_entry_threshold
            - self.config.critical_node_retention_threshold
        )
        edge_retention = self.config.critical_edge_entry_threshold - strength * (
            self.config.critical_edge_entry_threshold
            - self.config.critical_edge_retention_threshold
        )
        return select_object_conditioned_subgraphs(
            global_node_probabilities=global_node_probabilities,
            global_edge_probabilities=global_edge_probabilities,
            object_node_probabilities=object_node_probabilities,
            object_edge_probabilities=object_edge_probabilities,
            edge_presence_mask=edge_presence_mask,
            per_object_node_ratio=self.config.critical_node_ratio_per_object,
            edge_ratio=edge_ratio,
            node_minimum=self.config.node_minimum,
            edge_minimum=self.config.edge_minimum,
            candidate_multiplier=self.config.critical_candidate_multiplier,
            overlap_penalty=self.config.critical_overlap_penalty,
            max_node_overlap=self.config.critical_max_node_overlap,
            max_edge_overlap=self.config.critical_max_edge_overlap,
            previous_node_masks=previous_hard[0],
            previous_edge_masks=previous_hard[1],
            previous_seed_indices=previous_hard[2],
            continuity_bonus=(
                self.config.critical_history_continuity_bonus * strength
            ),
            switch_margin=self.config.critical_history_switch_margin * strength,
            history_node_growth_bonus=(
                self.config.critical_history_node_growth_bonus * strength
            ),
            history_edge_growth_bonus=(
                self.config.critical_history_edge_growth_bonus * strength
            ),
            anchor_node_masks=anchor_hard[0],
            anchor_edge_masks=anchor_hard[1],
            anchor_continuity_bonus=(
                self.config.selector_exploration_anchor_continuity_bonus
                if anchor_hard[0] is not None
                else 0.0
            ),
            anchor_node_growth_bonus=(
                self.config.selector_exploration_anchor_node_growth_bonus
                if anchor_hard[0] is not None
                else 0.0
            ),
            anchor_edge_growth_bonus=(
                self.config.selector_exploration_anchor_edge_growth_bonus
                if anchor_hard[0] is not None
                else 0.0
            ),
            node_entry_threshold=(
                self.config.critical_node_entry_threshold
                if self.config.selector_structural_temporal_memory
                else None
            ),
            node_retention_threshold=(
                node_retention
                if self.config.selector_structural_temporal_memory
                else None
            ),
            edge_entry_threshold=(
                self.config.critical_edge_entry_threshold
                if self.config.selector_structural_temporal_memory
                else None
            ),
            edge_retention_threshold=(
                edge_retention
                if self.config.selector_structural_temporal_memory
                else None
            ),
            communities=(
                communities
                if self.config.selector_structural_temporal_memory
                else None
            ),
            cross_community_penalty=(
                self.config.critical_cross_community_penalty
                if self.config.selector_structural_temporal_memory
                else 0.0
            ),
            node_reuse_penalty=(
                self.config.critical_node_reuse_penalty
                if self.config.selector_structural_temporal_memory
                else 0.0
            ),
            community_reuse_penalty=(
                self.config.critical_community_reuse_penalty
                if self.config.selector_structural_temporal_memory
                else 0.0
            ),
        )

    def forward(
        self,
        batch: ExactSTSEBatch,
        selection_mode: str = "learned",
        random_seed: int = 42,
        track_subgraphs: bool = False,
        diagnostic_independent_windows: bool = False,
    ) -> DualHardSelectionOutput:
        if selection_mode not in ("full", "random", "learned"):
            raise ValueError("unsupported dual hard-selection mode")
        sample_outputs = []
        sample_subgraph_outputs = []
        soft_sample_outputs = []
        trajectory_sets = []
        candidate_coverages: List[float] = []
        final_coverages: List[float] = []
        selections = []
        node_probability_values = []
        edge_probability_values = []
        fixed_k_outputs = []
        multi_object_regularization = []
        multi_object_iou = []
        slot_alignment_values = []
        memory_gate_values = []
        alignment_component_values: Dict[str, List[torch.Tensor]] = {}
        exploration_window_counts: List[int] = []
        retrospective_window_count = 0
        history_strength_values: List[float] = []
        exploration_candidate_pool_sizes: List[int] = []
        exploration_shortlist_sizes: List[int] = []
        exploration_anchor_supports: List[float] = []
        exploration_anchor_similarities: List[float] = []
        exploration_cluster_similarities: List[float] = []
        exploration_medoid_objectives: List[float] = []
        selected_positive_edge_count = 0
        selected_negative_edge_count = 0
        fast_runtime = bool(self.config.selector_fast_runtime)
        for exact_sample in batch:
            sample = exact_sample.graph
            previous_object_states = None
            previous_object_memory = None
            exploration_features: Dict[int, Any] = {}
            exploration_candidates: List[ExplorationObjectCandidate] = []
            exploration_medoid_selection = None
            exploration_windows = 0
            use_retrospective_exploration = bool(
                selection_mode == "learned"
                and not diagnostic_independent_windows
                and self.config.selector_architecture == "theory_multi_object"
                and self.config.selector_structural_temporal_memory
                and self.config.selector_exploration_consensus_enabled
                and sample.num_timepoints > 1
            )
            if use_retrospective_exploration:
                exploration_windows = _exploration_window_count(
                    sample.num_timepoints,
                    self.config.selector_exploration_min_windows,
                    self.config.selector_exploration_max_windows,
                    self.config.selector_exploration_fraction,
                )
                for exploration_index in range(exploration_windows):
                    exploration_adjacency = sample.adjacency[exploration_index]
                    exploration_edge_mask = sample.edge_mask[
                        exploration_index
                    ].to(exploration_adjacency.device)
                    exploration_node_ids = tuple(
                        str(value)
                        for value in sample.node_names[exploration_index]
                    )
                    exploration_coordinates = exact_sample.coordinates[
                        exploration_index
                    ].to(exploration_adjacency.device)
                    exploration_feature = self.feature_builder.build_timepoint(
                        sample, exploration_index
                    )
                    exploration_spectrum = self._cached_spectral_features(
                        sample.sample_key,
                        exploration_index,
                        exploration_adjacency,
                        exploration_edge_mask,
                    )
                    exploration_features[exploration_index] = (
                        exploration_feature,
                        exploration_spectrum,
                        exploration_node_ids,
                        exploration_coordinates,
                    )
                    observation_scores = self.scorer(
                        exploration_feature.node_features,
                        exploration_feature.edge_base_features,
                        exploration_feature.edge_presence_mask,
                        exploration_adjacency,
                        spectral_features=exploration_spectrum,
                        previous_memory=None,
                        current_node_ids=exploration_node_ids,
                        current_coordinates=exploration_coordinates,
                        history_strength=0.0,
                    )
                    observation = observation_scores.next_memory
                    candidate_hardening = self._harden_object_fields(
                        global_node_probabilities=(
                            observation_scores.node_probabilities
                        ),
                        global_edge_probabilities=(
                            observation_scores.edge_probabilities
                        ),
                        object_node_probabilities=(
                            observation_scores.object_node_probabilities
                        ),
                        object_edge_probabilities=(
                            observation_scores.object_edge_probabilities
                        ),
                        edge_presence_mask=exploration_edge_mask,
                        communities=sample.communities[exploration_index].to(
                            exploration_adjacency.device
                        ),
                        current_node_ids=exploration_node_ids,
                        previous_memory=None,
                        anchor_memory=None,
                        history_strength=0.0,
                        alignment_confidence=None,
                        edge_ratio=self.config.target_edge_ratio,
                    )
                    for object_index, hard_object in enumerate(
                        candidate_hardening.subgraphs
                    ):
                        memory = singleton_memory(
                            observation,
                            object_index,
                            hard_object.hard_node_mask,
                            hard_object.hard_edge_mask,
                            candidate_hardening.seed_indices[object_index],
                        )
                        exploration_candidates.append(
                            ExplorationObjectCandidate(
                                window_index=exploration_index,
                                object_index=object_index,
                                quality=candidate_quality(memory),
                                memory=memory,
                            )
                        )
                exploration_medoid_selection = select_exploration_medoids(
                    exploration_candidates,
                    self.config.critical_subgraph_count,
                    similarity_threshold=(
                        self.config.selector_exploration_candidate_similarity_threshold
                    ),
                    shortlist_multiplier=(
                        self.config.selector_exploration_shortlist_multiplier
                    ),
                    coverage_weight=(
                        self.config.selector_exploration_coverage_weight
                    ),
                    support_weight=(
                        self.config.selector_exploration_support_weight
                    ),
                    quality_weight=(
                        self.config.selector_exploration_quality_weight
                    ),
                    diversity_weight=(
                        self.config.selector_exploration_diversity_weight
                    ),
                    similarity_weights={
                        "node": self.config.selector_alignment_node_weight,
                        "signed_edge": (
                            self.config.selector_alignment_signed_edge_weight
                        ),
                        "latent": self.config.selector_alignment_latent_weight,
                        "coordinate": (
                            self.config.selector_alignment_coordinate_weight
                        ),
                        "spectral": (
                            self.config.selector_alignment_spectral_weight
                        ),
                    },
                )
                reference_index = exploration_windows - 1
                reference_adjacency = sample.adjacency[reference_index]
                previous_object_memory = stack_real_candidate_memories(
                    exploration_candidates,
                    exploration_medoid_selection.recent_indices,
                    sample.node_names[reference_index],
                    reference_adjacency,
                    sample.edge_mask[reference_index],
                )
                exploration_candidate_pool_sizes.append(
                    len(exploration_candidates)
                )
                exploration_shortlist_sizes.append(
                    len(exploration_medoid_selection.shortlist_indices)
                )
                exploration_anchor_supports.extend(
                    value / float(max(1, exploration_windows))
                    for value in exploration_medoid_selection.support_window_counts
                )
                exploration_anchor_similarities.append(
                    exploration_medoid_selection.mean_anchor_similarity
                )
                exploration_cluster_similarities.append(
                    exploration_medoid_selection.mean_cluster_similarity
                )
                exploration_medoid_objectives.append(
                    exploration_medoid_selection.objective
                )
                retrospective_window_count += exploration_windows
            exploration_window_counts.append(exploration_windows)
            windows = []
            subgraph_windows = []
            soft_windows = []
            for time_index in range(sample.num_timepoints):
                adjacency = sample.adjacency[time_index]
                count = int(adjacency.shape[0])
                if diagnostic_independent_windows:
                    history_strength = 0.0
                elif use_retrospective_exploration:
                    if time_index < exploration_windows:
                        history_strength = float(
                            self.config.selector_exploration_retrospective_strength
                        )
                    else:
                        history_strength = _cosine_history_strength(
                            time_index,
                            exploration_windows,
                            self.config.selector_exploration_history_ramp_windows,
                            self.config.selector_exploration_retrospective_strength,
                        )
                else:
                    history_strength = 1.0
                history_strength_values.append(history_strength)
                if selection_mode == "learned":
                    cached_exploration = exploration_features.get(time_index)
                    features = (
                        cached_exploration[0]
                        if cached_exploration is not None
                        else self.feature_builder.build_timepoint(sample, time_index)
                    )
                    if self.config.selector_architecture == "legacy_mlp":
                        scores = self.scorer(
                            features.node_features,
                            features.edge_base_features,
                            features.edge_presence_mask,
                        )
                    else:
                        current_edge_mask = sample.edge_mask[time_index].to(
                            adjacency.device
                        )
                        current_node_ids = (
                            cached_exploration[2]
                            if cached_exploration is not None
                            else tuple(
                                str(value)
                                for value in sample.node_names[time_index]
                            )
                        )
                        current_coordinates = (
                            cached_exploration[3]
                            if cached_exploration is not None
                            else exact_sample.coordinates[time_index].to(
                                adjacency.device
                            )
                        )
                        current_anchor_memory = None
                        if use_retrospective_exploration:
                            current_anchor_memory = stack_real_candidate_memories(
                                exploration_candidates,
                                exploration_medoid_selection.anchor_indices,
                                current_node_ids,
                                adjacency,
                                current_edge_mask,
                            )
                        scorer_previous_memory = previous_object_memory
                        scorer_history_strength = history_strength
                        if (
                            use_retrospective_exploration
                            and time_index < exploration_windows
                        ):
                            # The real medoids establish object identity, but
                            # their node/edge fields are never averaged into
                            # the current soft graph.
                            scorer_previous_memory = current_anchor_memory
                            scorer_history_strength = 0.0
                        scores = self.scorer(
                            features.node_features,
                            features.edge_base_features,
                            features.edge_presence_mask,
                            adjacency,
                            previous_object_states=(
                                previous_object_states
                                if (
                                    self.config.selector_object_temporal_state
                                    and not self.config.selector_structural_temporal_memory
                                    and not diagnostic_independent_windows
                                )
                                else None
                            ),
                            spectral_features=(
                                cached_exploration[1]
                                if cached_exploration is not None
                                else self._cached_spectral_features(
                                    sample.sample_key,
                                    time_index,
                                    adjacency,
                                    current_edge_mask,
                                )
                            ),
                            previous_memory=(
                                scorer_previous_memory
                                if (
                                    self.config.selector_structural_temporal_memory
                                    and not diagnostic_independent_windows
                                )
                                else None
                            ),
                            current_node_ids=(
                                current_node_ids
                                if self.config.selector_structural_temporal_memory
                                else None
                            ),
                            current_coordinates=(
                                current_coordinates
                                if self.config.selector_structural_temporal_memory
                                else None
                            ),
                            history_strength=scorer_history_strength,
                        )
                        if (
                            not self.config.selector_structural_temporal_memory
                            and not diagnostic_independent_windows
                        ):
                            previous_object_states = scores.next_object_states
                        multi_object_regularization.append(
                            scores.regularization
                        )
                        multi_object_iou.append(
                            scores.regularization.pairwise_soft_iou
                        )
                        if (
                            self.config.selector_structural_temporal_memory
                            and scores.transported_node_probabilities is not None
                        ):
                            slot_alignment_values.append(
                                scores.slot_alignment
                            )
                            memory_gate_values.append(
                                scores.memory_update_gate
                            )
                            for name, value in scores.alignment_components.items():
                                alignment_component_values.setdefault(
                                    name, []
                                ).append(value)
                    node_probabilities = scores.node_probabilities
                    edge_probabilities = scores.edge_probabilities
                else:
                    node_probabilities = adjacency.new_full((count,), 0.5)
                    edge_probabilities = sample.edge_mask[time_index].to(
                        device=adjacency.device,
                        dtype=adjacency.dtype,
                    ) * 0.5
                node_ratio = (
                    1.0
                    if selection_mode == "full"
                    else self.config.target_node_ratio
                )
                edge_ratio = (
                    1.0
                    if selection_mode == "full"
                    else self.config.target_edge_ratio
                )
                fixed_k = None
                if selection_mode == "learned":
                    if self.config.selector_architecture == "theory_multi_object":
                        fixed_k = self._harden_object_fields(
                            global_node_probabilities=node_probabilities,
                            global_edge_probabilities=edge_probabilities,
                            object_node_probabilities=(
                                scores.object_node_probabilities
                            ),
                            object_edge_probabilities=(
                                scores.object_edge_probabilities
                            ),
                            edge_presence_mask=sample.edge_mask[time_index].to(
                                adjacency.device
                            ),
                            communities=sample.communities[time_index].to(
                                adjacency.device
                            ),
                            current_node_ids=current_node_ids,
                            previous_memory=(
                                None
                                if diagnostic_independent_windows
                                else (
                                    current_anchor_memory
                                    if (
                                        use_retrospective_exploration
                                        and time_index < exploration_windows
                                    )
                                    else previous_object_memory
                                )
                            ),
                            anchor_memory=(
                                current_anchor_memory
                                if (
                                    use_retrospective_exploration
                                    and time_index >= exploration_windows
                                )
                                else None
                            ),
                            history_strength=history_strength,
                            alignment_confidence=(
                                None
                                if use_retrospective_exploration
                                else scores.alignment_confidence
                            ),
                            edge_ratio=edge_ratio,
                        )
                        if (
                            self.config.selector_structural_temporal_memory
                            and not diagnostic_independent_windows
                            and (
                                not use_retrospective_exploration
                                or time_index >= exploration_windows
                            )
                        ):
                            previous_object_memory = replace(
                                scores.next_memory,
                                hard_node_masks=torch.stack(
                                    [
                                        item.hard_node_mask
                                        for item in fixed_k.subgraphs
                                    ]
                                ),
                                hard_edge_masks=torch.stack(
                                    [
                                        item.hard_edge_mask
                                        for item in fixed_k.subgraphs
                                    ]
                                ),
                                seed_indices=fixed_k.seed_indices,
                            )
                    else:
                        fixed_k = select_fixed_k_subgraphs(
                        node_probabilities=node_probabilities,
                        edge_probabilities=edge_probabilities,
                        communities=sample.communities[time_index].to(
                            adjacency.device
                        ),
                        edge_presence_mask=sample.edge_mask[time_index].to(
                            adjacency.device
                        ),
                        subgraph_count=self.config.critical_subgraph_count,
                        candidate_multiplier=self.config.critical_candidate_multiplier,
                        total_node_ratio=node_ratio,
                        edge_ratio=edge_ratio,
                        node_minimum=self.config.node_minimum,
                        edge_minimum=self.config.edge_minimum,
                        overlap_penalty=self.config.critical_overlap_penalty,
                        coordinates=exact_sample.coordinates[time_index].to(
                            adjacency.device
                        ),
                        diversity_enabled=self.config.critical_diversity_enabled,
                        per_object_node_ratio=(
                            self.config.critical_node_ratio_per_object
                        ),
                        node_reuse_decay=self.config.critical_node_reuse_decay,
                        edge_reuse_decay=self.config.critical_edge_reuse_decay,
                        max_node_overlap=self.config.critical_max_node_overlap,
                        max_edge_overlap=self.config.critical_max_edge_overlap,
                        min_unique_node_fraction=(
                            self.config.critical_min_unique_node_fraction
                        ),
                        quality_floor_ratio=(
                            self.config.critical_quality_floor_ratio
                        ),
                            min_seed_distance=(
                                self.config.critical_min_seed_distance
                            ),
                        )
                    if not fast_runtime:
                        fixed_k_outputs.append(fixed_k)
                    selection = fixed_k.union
                else:
                    selection = select_hard_stse_window(
                        node_probabilities=node_probabilities,
                        edge_probabilities=edge_probabilities,
                        communities=sample.communities[time_index].to(
                            adjacency.device
                        ),
                        edge_presence_mask=sample.edge_mask[time_index].to(
                            adjacency.device
                        ),
                        node_ratio=node_ratio,
                        edge_ratio=edge_ratio,
                        node_minimum=self.config.node_minimum,
                        edge_minimum=self.config.edge_minimum,
                        selection_mode=selection_mode,
                        sample_key=sample.sample_key,
                        time_index=time_index,
                        random_seed=random_seed,
                    )
                hard = build_hard_stse_window(
                    sample, time_index, selection
                )
                object_outputs: List[Optional[HardWindowOutput]] = []
                if fixed_k is not None:
                    for object_selection in fixed_k.subgraphs:
                        object_outputs.append(
                            build_hard_stse_window(
                                sample, time_index, object_selection
                            )
                        )
                else:
                    object_outputs.append(hard)
                while len(object_outputs) < self.config.critical_subgraph_count:
                    object_outputs.append(None)
                object_outputs = object_outputs[
                    : self.config.critical_subgraph_count
                ]
                soft = _build_soft_window(
                    adjacency,
                    sample.edge_mask[time_index],
                    selection,
                )
                communities = sample.communities[time_index].to(
                    adjacency.device
                )
                if not fast_runtime:
                    candidate_coverages.append(
                        _community_coverage(
                            selection.candidate_node_mask, communities
                        )
                    )
                    final_coverages.append(
                        _community_coverage(
                            selection.hard_node_mask, communities
                        )
                    )
                selections.append(selection)
                if not fast_runtime:
                    node_probability_values.append(
                        selection.node_probabilities.reshape(-1)
                    )
                    valid_upper = torch.triu(
                        sample.edge_mask[time_index].to(
                            device=adjacency.device, dtype=torch.bool
                        ),
                        diagonal=1,
                    )
                    if bool(valid_upper.any()):
                        edge_probability_values.append(
                            selection.edge_probabilities[valid_upper]
                        )
                    selected_upper = torch.triu(
                        selection.hard_edge_mask, diagonal=1
                    )
                    selected_weights = adjacency[selected_upper]
                    selected_positive_edge_count += int(
                        (selected_weights > 0.0).sum()
                    )
                    selected_negative_edge_count += int(
                        (selected_weights < 0.0).sum()
                    )
                windows.append(hard)
                subgraph_windows.append(tuple(object_outputs))
                soft_windows.append(soft)
            sample_outputs.append(tuple(windows))
            sample_subgraph_outputs.append(tuple(subgraph_windows))
            soft_sample_outputs.append(tuple(soft_windows))
            trajectory_sets.append(
                build_dynamic_trajectories(
                    subgraph_windows,
                    exact_sample.coordinates,
                    self.config.critical_subgraph_count,
                )
                if track_subgraphs
                else None
            )
        total_original_nodes = sum(
            int(item.node_probabilities.numel()) for item in selections
        )
        total_candidate_nodes = (
            sum(item.requested_node_count for item in selections)
            if fast_runtime
            else sum(
                int(item.candidate_node_mask.sum())
                for item in selections
            )
        )
        total_final_nodes = (
            sum(item.actual_node_count for item in selections)
            if fast_runtime
            else sum(
                int(item.hard_node_mask.sum()) for item in selections
            )
        )
        total_original_edges = sum(
            item.original_edge_count for item in selections
        )
        total_final_edges = sum(
            item.actual_edge_count for item in selections
        )
        node_probabilities = (
            torch.cat(node_probability_values)
            if node_probability_values
            else selections[-1].node_probabilities.new_zeros((0,))
        )
        edge_probabilities = (
            torch.cat(edge_probability_values)
            if edge_probability_values
            else selections[-1].node_probabilities.new_zeros((0,))
        )
        diagnostics = {
            "selector_architecture": self.config.selector_architecture,
            "uses_signed_graph_encoder": (
                self.config.selector_architecture == "theory_multi_object"
            ),
            "uses_object_temporal_state": (
                self.config.selector_architecture == "theory_multi_object"
                and self.config.selector_object_temporal_state
            ),
            "uses_structural_temporal_memory": (
                self.config.selector_architecture == "theory_multi_object"
                and self.config.selector_structural_temporal_memory
            ),
            "uses_retrospective_exploration_consensus": (
                self.config.selector_architecture == "theory_multi_object"
                and self.config.selector_structural_temporal_memory
                and self.config.selector_exploration_consensus_enabled
            ),
            "exploration_initializer": "real_candidate_medoid",
            "uses_real_candidate_medoid_initializer": (
                self.config.selector_architecture == "theory_multi_object"
                and self.config.selector_structural_temporal_memory
                and self.config.selector_exploration_consensus_enabled
            ),
            "exploration_window_count": tuple(exploration_window_counts),
            "retrospectively_refined_window_count": retrospective_window_count,
            "mean_history_strength": (
                sum(history_strength_values)
                / float(max(1, len(history_strength_values)))
            ),
            # Compatibility field: soft cross-window averaging is retired.
            "mean_exploration_consensus_confidence": (
                node_probabilities.new_zeros(())
            ),
            "mean_exploration_candidate_pool_size": (
                sum(exploration_candidate_pool_sizes)
                / float(max(1, len(exploration_candidate_pool_sizes)))
            ),
            "mean_exploration_shortlist_size": (
                sum(exploration_shortlist_sizes)
                / float(max(1, len(exploration_shortlist_sizes)))
            ),
            "mean_exploration_anchor_support": (
                sum(exploration_anchor_supports)
                / float(max(1, len(exploration_anchor_supports)))
            ),
            "mean_exploration_anchor_similarity": (
                sum(exploration_anchor_similarities)
                / float(max(1, len(exploration_anchor_similarities)))
            ),
            "mean_exploration_cluster_similarity": (
                sum(exploration_cluster_similarities)
                / float(max(1, len(exploration_cluster_similarities)))
            ),
            "mean_exploration_medoid_objective": (
                sum(exploration_medoid_objectives)
                / float(max(1, len(exploration_medoid_objectives)))
            ),
            "selection_mode": selection_mode,
            "diagnostic_independent_windows": bool(
                diagnostic_independent_windows
            ),
            "detailed_diagnostics_skipped": fast_runtime,
            "selection_count": len(selections),
            "candidate_node_ratio": total_candidate_nodes
            / float(max(1, total_original_nodes)),
            "actual_node_ratio": total_final_nodes
            / float(max(1, total_original_nodes)),
            "actual_edge_ratio": total_final_edges
            / float(max(1, total_original_edges)),
            "candidate_community_coverage": sum(candidate_coverages)
            / float(max(1, len(candidate_coverages))),
            "final_community_coverage": sum(final_coverages)
            / float(max(1, len(final_coverages))),
            "empty_hard_window_count": sum(
                not window.window_valid
                for windows in sample_outputs
                for window in windows
            ),
            "node_probability_mean": float(
                node_probabilities.detach().mean().cpu()
            ) if node_probabilities.numel() else 0.0,
            "node_probability_std": (
                float(
                    node_probabilities.detach()
                    .std(unbiased=False).cpu()
                )
                if node_probabilities.numel()
                else 0.0
            ),
            "edge_probability_mean": (
                float(edge_probabilities.detach().mean().cpu())
                if edge_probabilities.numel()
                else 0.0
            ),
            "edge_probability_std": (
                float(
                    edge_probabilities.detach()
                    .std(unbiased=False)
                    .cpu()
                )
                if edge_probabilities.numel()
                else 0.0
            ),
            "selected_positive_edge_count": (
                selected_positive_edge_count
            ),
            "selected_negative_edge_count": (
                selected_negative_edge_count
            ),
            "critical_subgraph_count": self.config.critical_subgraph_count,
            "critical_diversity_enabled": (
                self.config.critical_diversity_enabled
            ),
            "mean_fixed_k_union_efficiency": (
                sum(item.union_efficiency for item in fixed_k_outputs)
                / float(max(1, len(fixed_k_outputs)))
            ),
            "diversity_relaxed_window_count": sum(
                item.diversity_constraint_relaxed
                for item in fixed_k_outputs
            ),
            "hysteresis_relaxed_window_count": sum(
                item.hysteresis_constraint_relaxed
                for item in fixed_k_outputs
            ),
            "mean_fixed_k_node_overlap": (
                sum(
                    float(
                        item.pairwise_node_overlap[
                            torch.triu(
                                torch.ones_like(
                                    item.pairwise_node_overlap,
                                    dtype=torch.bool,
                                ),
                                diagonal=1,
                            )
                        ].mean()
                    )
                    for item in fixed_k_outputs
                )
                / float(max(1, len(fixed_k_outputs)))
                if fixed_k_outputs
                else 0.0
            ),
            "mean_fixed_k_edge_overlap": (
                sum(
                    float(
                        item.pairwise_edge_overlap[
                            torch.triu(
                                torch.ones_like(
                                    item.pairwise_edge_overlap,
                                    dtype=torch.bool,
                                ),
                                diagonal=1,
                            )
                        ].mean()
                    )
                    for item in fixed_k_outputs
                )
                / float(max(1, len(fixed_k_outputs)))
                if fixed_k_outputs
                else 0.0
            ),
            "trajectory_count": tuple(
                item.trajectory_count if item is not None else None
                for item in trajectory_sets
            ),
            "trajectory_birth_count": tuple(
                item.total_birth_count if item is not None else None
                for item in trajectory_sets
            ),
            "multi_object_regularization": (
                {
                    "overlap": torch.stack(
                        [item.overlap for item in multi_object_regularization]
                    ).mean(),
                    "reconstruction": torch.stack(
                        [item.reconstruction for item in multi_object_regularization]
                    ).mean(),
                    "coverage": torch.stack(
                        [item.coverage for item in multi_object_regularization]
                    ).mean(),
                    "temporal": torch.stack(
                        [item.temporal for item in multi_object_regularization]
                    ).mean(),
                    "node_continuity": torch.stack(
                        [
                            item.node_continuity
                            for item in multi_object_regularization
                        ]
                    ).mean(),
                    "edge_continuity": torch.stack(
                        [
                            item.edge_continuity
                            for item in multi_object_regularization
                        ]
                    ).mean(),
                }
                if multi_object_regularization
                else None
            ),
            "mean_slot_alignment_confidence": (
                torch.stack(
                    [item.max(dim=-1).values.mean() for item in slot_alignment_values]
                ).mean()
                if slot_alignment_values
                else node_probabilities.new_zeros(())
            ),
            "mean_memory_update_gate": (
                torch.cat(memory_gate_values).mean()
                if memory_gate_values
                else node_probabilities.new_zeros(())
            ),
            "mean_slot_alignment_components": {
                name: torch.stack(values).mean()
                for name, values in alignment_component_values.items()
                if values
            },
            "mean_soft_object_iou": (
                torch.stack(
                    [
                        item[
                            torch.triu(
                                torch.ones_like(item, dtype=torch.bool),
                                diagonal=1,
                            )
                        ].mean()
                        for item in multi_object_iou
                    ]
                ).mean()
                if multi_object_iou
                else node_probabilities.new_zeros(())
            ),
            "selections": tuple(selections),
        }
        return DualHardSelectionOutput(
            hard_windows=tuple(sample_outputs),
            hard_subgraphs=tuple(sample_subgraph_outputs),
            soft_windows=tuple(soft_sample_outputs),
            trajectory_sets=tuple(trajectory_sets),
            diagnostics=diagnostics,
        )
