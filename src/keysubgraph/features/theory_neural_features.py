"""Leakage-safe Stage-1 edge-aware features and exact SGW targets."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from keysubgraph.features.graph_features import align_current_to_previous
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.sv_hard_graph_features import (
    SV_NODE_FEATURE_DIM,
    SVHardSampleFeatureBuilder,
)
from keysubgraph.theory.sgw_core_features import (
    SGW_CORE_DIM,
    SGW_QUANTILE_DIM,
    SGWCoreConfig,
    compute_sgw_core_sequence,
)


THEORY_EDGE_FEATURE_DIM = 6


@dataclass(frozen=True)
class TheoryNeuralWindowFeatures:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    spectral_quantiles: torch.Tensor
    communities: torch.Tensor
    node_ids: Tuple[str, ...]
    time_start: float


@dataclass(frozen=True)
class TheoryNeuralSampleFeatures:
    windows: Tuple[Optional[TheoryNeuralWindowFeatures], ...]
    window_mask: torch.Tensor
    transition_features: torch.Tensor
    transition_mask: torch.Tensor
    gw_solver_converged: Tuple[bool, ...]


def _stable_ids(window: HardGraphWindow) -> Tuple[str, ...]:
    values = (
        tuple(str(value) for value in window.node_ids)
        if window.node_ids is not None
        else tuple(str(value) for value in window.node_names)
    )
    if len(values) != window.num_nodes or len(set(values)) != len(values):
        raise ValueError("Stage-1 node identities must be unique and aligned")
    return values


class TheoryNeuralFeatureBuilder(object):
    """Build frozen 15-D nodes, 6-D signed edges and exact SGW targets."""

    def __init__(self, core_config=None) -> None:
        self.core_config = core_config or SGWCoreConfig()
        self.extractor = self.core_config.build_extractor()
        self.base = SVHardSampleFeatureBuilder()

    def build(
        self,
        windows: Sequence[Optional[HardGraphWindow]],
        time_values: Optional[Sequence[float]] = None,
    ) -> TheoryNeuralSampleFeatures:
        if not windows:
            raise ValueError("Stage-1 hard sequence cannot be empty")
        base = self.base.build(windows)
        times = (
            [float(value) for value in time_values]
            if time_values is not None
            else [float(index) for index in range(len(windows))]
        )
        if len(times) != len(windows) or any(
            right <= left for left, right in zip(times[:-1], times[1:])
        ):
            raise ValueError("Stage-1 time values must align and increase")
        threshold = next(
            (
                float(window.edge_presence_threshold)
                for window in windows
                if window is not None
            ),
            None,
        )
        if threshold is None:
            raise ValueError("Stage-1 sample has no valid hard window")
        core = compute_sgw_core_sequence(
            [
                window.adjacency if window is not None else None
                for window in windows
            ],
            times,
            threshold,
            config=self.core_config,
            extractor=self.extractor,
        )

        output = []
        previous_window = None
        previous_adjacency = None
        for index, (source, current) in enumerate(
            zip(windows, base.windows)
        ):
            if source is None or current is None:
                output.append(None)
                previous_window = None
                previous_adjacency = None
                continue
            adjacency = current.adjacency
            count = int(adjacency.shape[0])
            delta = torch.zeros_like(adjacency)
            delta_mask = torch.zeros_like(adjacency, dtype=torch.bool)
            if previous_window is not None and previous_adjacency is not None:
                indices_cpu, present_cpu = align_current_to_previous(
                    _stable_ids(source), _stable_ids(previous_window)
                )
                indices = indices_cpu.to(adjacency.device)
                present = present_cpu.to(adjacency.device)
                safe = indices.clamp_min(0)
                previous_aligned = previous_adjacency.index_select(
                    0, safe
                ).index_select(1, safe)
                delta_mask = present[:, None] & present[None, :]
                delta_mask = delta_mask.clone()
                delta_mask.fill_diagonal_(False)
                delta = torch.where(
                    delta_mask,
                    adjacency - previous_aligned,
                    torch.zeros_like(adjacency),
                )
            communities = current.communities.to(torch.long)
            same_community = (
                communities[:, None] == communities[None, :]
            )
            edge_features = torch.stack(
                (
                    adjacency,
                    adjacency.abs(),
                    delta,
                    delta.abs(),
                    delta_mask.to(adjacency.dtype),
                    same_community.to(adjacency.dtype),
                ),
                dim=-1,
            )
            if tuple(current.node_features.shape) != (
                count,
                SV_NODE_FEATURE_DIM,
            ) or tuple(edge_features.shape) != (
                count,
                count,
                THEORY_EDGE_FEATURE_DIM,
            ):
                raise RuntimeError("Stage-1 window feature schema mismatch")
            output.append(
                TheoryNeuralWindowFeatures(
                    node_features=current.node_features,
                    adjacency=adjacency,
                    edge_features=edge_features,
                    spectral_quantiles=core.window_quantiles[index],
                    communities=communities,
                    node_ids=_stable_ids(source),
                    time_start=float(source.time_start),
                )
            )
            previous_window = source
            previous_adjacency = adjacency
        if tuple(core.transition_features.shape) != (
            max(0, len(windows) - 1),
            SGW_CORE_DIM,
        ) or tuple(core.window_quantiles.shape) != (
            len(windows),
            SGW_QUANTILE_DIM,
        ):
            raise RuntimeError("Stage-1 exact target schema mismatch")
        return TheoryNeuralSampleFeatures(
            windows=tuple(output),
            window_mask=core.window_mask,
            transition_features=core.transition_features,
            transition_mask=core.transition_mask,
            gw_solver_converged=core.gw_solver_converged,
        )
