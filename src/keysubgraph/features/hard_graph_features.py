"""Classification features recomputed strictly from frozen hard union graphs."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .graph_features import GraphFeatureBuilder, align_current_to_previous


@dataclass(frozen=True)
class HardGraphWindow:
    adjacency: torch.Tensor
    communities: torch.Tensor
    node_names: Tuple[str, ...]
    time_start: float
    edge_presence_threshold: float
    node_ids: Optional[Tuple[str, ...]] = None
    window_valid: bool = True

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])


@dataclass(frozen=True)
class HardGraphClassificationFeatures:
    time_index: int
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    delta_degree_mask: torch.Tensor
    delta_edge_mask: torch.Tensor
    source: str = "hard_union_recomputed"


def _stable_keys(window: HardGraphWindow) -> Tuple[str, ...]:
    if window.node_ids is not None:
        values = tuple(str(value) for value in window.node_ids)
        if len(values) != window.num_nodes or len(set(values)) != len(values):
            raise ValueError("hard graph node_ids must be unique and align with nodes")
        return values
    values = tuple(str(value) for value in window.node_names)
    if len(values) != window.num_nodes or len(set(values)) != len(values):
        raise ValueError("hard graph node_names must be unique and align with nodes")
    return values


class HardGraphFeatureBuilder:
    """Build the 13-D classifier view without reading full-graph statistics."""

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        self.base = GraphFeatureBuilder(epsilon=epsilon)

    @staticmethod
    def _validated_adjacency(window: HardGraphWindow) -> Tuple[torch.Tensor, torch.Tensor]:
        adjacency = window.adjacency
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("hard union adjacency must be square")
        if adjacency.shape[0] < 1:
            raise ValueError("an empty hard union cannot enter feature construction")
        if not bool(torch.isfinite(adjacency).all()):
            raise ValueError("hard union adjacency contains non-finite values")
        if window.edge_presence_threshold < 0.0:
            raise ValueError("edge threshold must be non-negative")
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        edge_mask = adjacency.abs() > float(window.edge_presence_threshold)
        edge_mask.fill_diagonal_(False)
        adjacency = adjacency * edge_mask.to(adjacency.dtype)
        if window.communities.ndim != 1 or window.communities.numel() != adjacency.shape[0]:
            raise ValueError("hard union communities must align with nodes")
        _stable_keys(window)
        return adjacency, edge_mask

    def build_sequence(
        self, windows: Sequence[HardGraphWindow]
    ) -> Tuple[Optional[HardGraphClassificationFeatures], ...]:
        if not windows:
            raise ValueError("hard graph sequence cannot be empty")
        for left, right in zip(windows[:-1], windows[1:]):
            if float(right.time_start) <= float(left.time_start):
                raise ValueError("hard graph times must be strictly increasing")

        results = []
        previous_window = None
        previous_adjacency = None
        for time_index, window in enumerate(windows):
            if not window.window_valid:
                results.append(None)
                previous_window = None
                previous_adjacency = None
                continue
            adjacency, edge_mask = self._validated_adjacency(window)
            static = self.base.build_static_node_features(
                adjacency,
                window.communities.to(device=adjacency.device, dtype=torch.long),
                float(window.edge_presence_threshold),
            )
            degree = adjacency.abs().sum(dim=-1)
            delta_degree = torch.zeros_like(degree)
            delta_degree_mask = torch.zeros(
                adjacency.shape[0], dtype=torch.bool, device=adjacency.device
            )
            delta_edge = torch.zeros_like(adjacency)
            delta_edge_mask = torch.zeros_like(adjacency, dtype=torch.bool)

            if previous_window is not None and previous_adjacency is not None:
                previous_indices_cpu, present_cpu = align_current_to_previous(
                    _stable_keys(window), _stable_keys(previous_window)
                )
                previous_indices = previous_indices_cpu.to(device=adjacency.device)
                present = present_cpu.to(device=adjacency.device)
                safe = previous_indices.clamp_min(0)
                previous_degree = previous_adjacency.abs().sum(dim=-1)
                delta_degree[present] = degree[present] - previous_degree[safe[present]]
                previous_aligned = previous_adjacency.index_select(0, safe).index_select(1, safe)
                delta_edge_mask = present[:, None] & present[None, :]
                delta_edge_mask.fill_diagonal_(False)
                delta_edge = torch.where(
                    delta_edge_mask,
                    adjacency - previous_aligned,
                    torch.zeros_like(adjacency),
                )
                delta_degree_mask = present

            node_features = torch.cat(
                (static[:, :5], delta_degree.unsqueeze(-1), static[:, 5:]), dim=-1
            )
            edge_features = torch.stack(
                (adjacency, adjacency.abs(), delta_edge, delta_edge.abs()), dim=-1
            )
            if tuple(node_features.shape) != (adjacency.shape[0], 13):
                raise RuntimeError("hard classifier node feature dimension must be 13")
            if tuple(edge_features.shape) != (adjacency.shape[0], adjacency.shape[0], 4):
                raise RuntimeError("hard classifier edge feature dimension must be 4")
            results.append(
                HardGraphClassificationFeatures(
                    time_index=time_index,
                    node_features=node_features,
                    edge_features=edge_features,
                    edge_mask=edge_mask,
                    delta_degree_mask=delta_degree_mask,
                    delta_edge_mask=delta_edge_mask,
                )
            )
            previous_window = window
            previous_adjacency = adjacency
        return tuple(results)

