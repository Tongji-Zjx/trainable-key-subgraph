"""Hard graph selection channel shared by D1--D4."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.features.hard_stse_extractor_features import (
    HardSTSEExtractorFeatureBuilder,
)
from keysubgraph.features.hard_stse_hard_graph import build_hard_stse_window
from .dual_stse_hard_sgw_types import DualSTSEHardSGWConfig
from .hard_stse_selector import HardSTSEScorer, select_hard_stse_window
from .hard_stse_types import (
    HardSelectionSchedule,
    HardSTSEConfig,
    HardWindowOutput,
)


@dataclass(frozen=True)
class DualHardSelectionOutput:
    hard_windows: Tuple[Tuple[HardWindowOutput, ...], ...]
    diagnostics: Dict[str, Any]


def _community_coverage(mask: torch.Tensor, communities: torch.Tensor) -> float:
    labels = torch.unique(communities, sorted=True)
    if labels.numel() < 1:
        return 0.0
    covered = sum(
        bool(mask[communities == label].any()) for label in labels
    )
    return float(covered) / float(labels.numel())


class DualHardSGWSelector(nn.Module):
    """Score and select signed hard graphs without adding graph encoders."""

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
        self.scorer = HardSTSEScorer(scorer_config)

    def forward(
        self,
        batch: ExactSTSEBatch,
        selection_mode: str = "learned",
        random_seed: int = 42,
    ) -> DualHardSelectionOutput:
        if selection_mode not in ("full", "random", "learned"):
            raise ValueError("unsupported dual hard-selection mode")
        sample_outputs = []
        candidate_coverages: List[float] = []
        final_coverages: List[float] = []
        selections = []
        for exact_sample in batch:
            sample = exact_sample.graph
            windows = []
            for time_index in range(sample.num_timepoints):
                adjacency = sample.adjacency[time_index]
                count = int(adjacency.shape[0])
                if selection_mode == "learned":
                    features = self.feature_builder.build_timepoint(
                        sample, time_index
                    )
                    scores = self.scorer(
                        features.node_features,
                        features.edge_base_features,
                        features.edge_presence_mask,
                    )
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
                communities = sample.communities[time_index].to(
                    adjacency.device
                )
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
                windows.append(hard)
            sample_outputs.append(tuple(windows))
        total_original_nodes = sum(
            int(item.node_probabilities.numel()) for item in selections
        )
        total_candidate_nodes = sum(
            int(item.candidate_node_mask.sum()) for item in selections
        )
        total_final_nodes = sum(
            int(item.hard_node_mask.sum()) for item in selections
        )
        total_original_edges = sum(
            item.original_edge_count for item in selections
        )
        total_final_edges = sum(
            item.actual_edge_count for item in selections
        )
        diagnostics = {
            "selection_mode": selection_mode,
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
            "selections": tuple(selections),
        }
        return DualHardSelectionOutput(
            hard_windows=tuple(sample_outputs),
            diagnostics=diagnostics,
        )

