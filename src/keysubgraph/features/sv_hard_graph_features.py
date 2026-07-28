"""Leakage-safe features recomputed from frozen signed hard graphs.

The selector uses 15-D node features built from the full graph.  A downstream
hard-graph encoder must not reuse those values because their degrees and
community strengths contain edges that were removed by hard selection.  This
module therefore rebuilds the same semantic schema strictly from each cropped
hard graph.
"""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from keysubgraph.features.graph_features import (
    GraphFeatureBuilder,
    align_current_to_previous,
)
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.theory.spectral_gw import (
    SignedLaplacianBuilder,
    SpectralStateExtractor,
)


SV_NODE_FEATURE_DIM = 15
SV_STATIC_FEATURE_DIM = 28
SV_VARIATION_DIM = 16


def sv_quantile_grid() -> Tuple[float, ...]:
    return tuple(0.05 + (0.90 / 15.0) * index for index in range(16))


def _stable_keys(window: HardGraphWindow) -> Tuple[str, ...]:
    values = (
        tuple(str(value) for value in window.node_ids)
        if window.node_ids is not None
        else tuple(str(value) for value in window.node_names)
    )
    if len(values) != window.num_nodes or len(set(values)) != len(values):
        raise ValueError("hard graph node identities must be unique and aligned")
    return values


def _validated_window(
    window: HardGraphWindow,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = window.adjacency
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("SV hard adjacency must be square")
    if adjacency.shape[0] < 1:
        raise ValueError("SV hard graph cannot be empty")
    if not bool(torch.isfinite(adjacency).all()):
        raise ValueError("SV hard adjacency contains non-finite values")
    if float(window.edge_presence_threshold) < 0.0:
        raise ValueError("SV edge-presence threshold must be non-negative")
    communities = window.communities.to(
        device=adjacency.device, dtype=torch.long
    )
    if tuple(communities.shape) != (adjacency.shape[0],):
        raise ValueError("SV communities must align with hard nodes")
    _stable_keys(window)
    adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(0.0)
    edge_mask = adjacency.abs() > float(window.edge_presence_threshold)
    edge_mask = edge_mask.clone()
    edge_mask.fill_diagonal_(False)
    adjacency = adjacency * edge_mask.to(adjacency.dtype)
    return adjacency, communities, edge_mask


def _local_clustering(
    edge_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    binary = edge_mask.to(dtype=dtype)
    binary = ((binary + binary.transpose(0, 1)) > 0.0).to(dtype)
    binary = binary.clone()
    binary.fill_diagonal_(0.0)
    degree = binary.sum(dim=-1)
    closed_walks = torch.diagonal(
        binary.matmul(binary).matmul(binary)
    )
    denominator = degree * (degree - 1.0)
    return torch.where(
        denominator > 0.0,
        closed_walks / denominator.clamp_min(1.0),
        torch.zeros_like(degree),
    )


@dataclass(frozen=True)
class SVHardWindowFeatures:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    communities: torch.Tensor
    edge_mask: torch.Tensor
    delta_degree_mask: torch.Tensor
    delta_edge_mask: torch.Tensor
    time_start: float


@dataclass(frozen=True)
class SVHardSampleFeatures:
    windows: Tuple[Optional[SVHardWindowFeatures], ...]
    static_features: torch.Tensor
    variation: torch.Tensor
    window_mask: torch.Tensor
    transition_mask: torch.Tensor


class SVHardNodeFeatureBuilder(object):
    """Recompute the selector's 15 semantic features on hard graphs only."""

    node_feature_dim = SV_NODE_FEATURE_DIM

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        if epsilon <= 0.0:
            raise ValueError("SV feature epsilon must be positive")
        self.epsilon = float(epsilon)
        self.base = GraphFeatureBuilder(epsilon=epsilon)

    def build_sequence(
        self, windows: Sequence[Optional[HardGraphWindow]]
    ) -> Tuple[Optional[SVHardWindowFeatures], ...]:
        if not windows:
            raise ValueError("SV hard graph sequence cannot be empty")
        valid_times = [
            float(window.time_start)
            for window in windows
            if window is not None and window.window_valid
        ]
        if any(
            right <= left
            for left, right in zip(valid_times[:-1], valid_times[1:])
        ):
            raise ValueError("SV hard graph times must be strictly increasing")

        output = []
        previous_window = None
        previous_adjacency = None
        for window in windows:
            if window is None or not window.window_valid:
                output.append(None)
                previous_window = None
                previous_adjacency = None
                continue
            adjacency, communities, edge_mask = _validated_window(window)
            static = self.base.build_static_node_features(
                adjacency,
                communities,
                float(window.edge_presence_threshold),
            )
            degree = adjacency.abs().sum(dim=-1)
            delta_degree = torch.zeros_like(degree)
            delta_degree_mask = torch.zeros(
                adjacency.shape[0],
                dtype=torch.bool,
                device=adjacency.device,
            )
            delta_edge = torch.zeros_like(adjacency)
            delta_edge_mask = torch.zeros_like(
                adjacency, dtype=torch.bool
            )
            if previous_window is not None and previous_adjacency is not None:
                indices_cpu, present_cpu = align_current_to_previous(
                    _stable_keys(window), _stable_keys(previous_window)
                )
                indices = indices_cpu.to(device=adjacency.device)
                present = present_cpu.to(device=adjacency.device)
                safe = indices.clamp_min(0)
                previous_degree = previous_adjacency.abs().sum(dim=-1)
                delta_degree[present] = (
                    degree[present] - previous_degree[safe[present]]
                )
                previous_aligned = previous_adjacency.index_select(
                    0, safe
                ).index_select(1, safe)
                delta_edge_mask = present[:, None] & present[None, :]
                delta_edge_mask = delta_edge_mask.clone()
                delta_edge_mask.fill_diagonal_(False)
                delta_edge = torch.where(
                    delta_edge_mask,
                    adjacency - previous_aligned,
                    torch.zeros_like(adjacency),
                )
                delta_degree_mask = present

            delta_count = delta_edge_mask.sum(dim=-1).to(adjacency.dtype)
            mean_abs_delta = (
                delta_edge.abs()
                * delta_edge_mask.to(adjacency.dtype)
            ).sum(dim=-1) / delta_count.clamp_min(1.0)
            valid_delta_ratio = delta_count / float(
                max(1, adjacency.shape[0] - 1)
            )
            clustering = _local_clustering(
                edge_mask, adjacency.dtype
            )

            # static columns: degree, positive degree, negative magnitude,
            # positive/negative ratios, then seven community features.  The
            # verified 15-D selector schema intentionally excludes the two
            # signed ratios and adds temporal validity/edge-change features.
            node_features = torch.cat(
                (
                    static[:, 0:3],
                    delta_degree[:, None],
                    delta_degree_mask.to(adjacency.dtype)[:, None],
                    mean_abs_delta[:, None],
                    valid_delta_ratio[:, None],
                    static[:, 5:12],
                    clustering[:, None],
                ),
                dim=-1,
            )
            if tuple(node_features.shape) != (
                adjacency.shape[0],
                SV_NODE_FEATURE_DIM,
            ):
                raise RuntimeError("SV hard node schema is not 15-D")
            if not bool(torch.isfinite(node_features).all()):
                raise ValueError("SV hard node features are non-finite")
            output.append(
                SVHardWindowFeatures(
                    node_features=node_features,
                    adjacency=adjacency,
                    communities=communities,
                    edge_mask=edge_mask,
                    delta_degree_mask=delta_degree_mask,
                    delta_edge_mask=delta_edge_mask,
                    time_start=float(window.time_start),
                )
            )
            previous_window = window
            previous_adjacency = adjacency
        return tuple(output)


class SVStaticVariationExtractor(object):
    """Build 28-D static state and 16-D adjacent-window variation."""

    static_feature_dim = SV_STATIC_FEATURE_DIM
    variation_dim = SV_VARIATION_DIM

    def __init__(
        self, laplacian_eta: float = 1.0e-3, epsilon: float = 1.0e-8
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("SV static epsilon must be positive")
        self.epsilon = float(epsilon)
        self.laplacian = SignedLaplacianBuilder(laplacian_eta)
        self.spectral = SpectralStateExtractor(sv_quantile_grid())

    def _window_structure(
        self,
        adjacency: torch.Tensor,
        communities: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        count = int(adjacency.shape[0])
        upper = torch.triu(
            torch.ones_like(edge_mask, dtype=torch.bool), diagonal=1
        )
        valid_upper = edge_mask & upper
        positive_mask = (adjacency > 0.0) & valid_upper
        negative_mask = (adjacency < 0.0) & valid_upper
        positive = adjacency.clamp_min(0.0)
        negative = -adjacency.clamp_max(0.0)
        possible_pairs = float(max(1, count * (count - 1) // 2))

        same = communities[:, None] == communities[None, :]
        intra_pairs = same & upper
        inter_pairs = (~same) & upper
        intra_count = intra_pairs.sum().to(adjacency.dtype)
        inter_count = inter_pairs.sum().to(adjacency.dtype)
        positive_sum = positive[upper].sum()
        negative_sum = negative[upper].sum()
        intra_positive = positive[intra_pairs].sum()
        intra_negative = negative[intra_pairs].sum()
        inter_positive = positive[inter_pairs].sum()
        inter_negative = negative[inter_pairs].sum()
        total_absolute = positive_sum + negative_sum

        labels, label_counts = torch.unique(
            communities, sorted=True, return_counts=True
        )
        community_count = int(labels.numel())
        proportions = label_counts.to(adjacency.dtype) / float(count)
        if community_count > 1:
            entropy = -(
                proportions
                * torch.log(proportions.clamp_min(self.epsilon))
            ).sum() / torch.log(
                adjacency.new_tensor(float(community_count))
            )
        else:
            entropy = adjacency.new_zeros(())

        values = torch.stack(
            (
                positive_mask.sum().to(adjacency.dtype)
                / possible_pairs,
                negative_mask.sum().to(adjacency.dtype)
                / possible_pairs,
                positive_sum / possible_pairs,
                negative_sum / possible_pairs,
                intra_positive / intra_count.clamp_min(1.0),
                intra_negative / intra_count.clamp_min(1.0),
                inter_positive / inter_count.clamp_min(1.0),
                inter_negative / inter_count.clamp_min(1.0),
                (intra_positive + intra_negative)
                / total_absolute.clamp_min(self.epsilon),
                positive_sum
                / total_absolute.clamp_min(self.epsilon),
                adjacency.new_tensor(float(community_count))
                / float(count),
                entropy,
            )
        )
        if tuple(values.shape) != (12,) or not bool(
            torch.isfinite(values).all()
        ):
            raise RuntimeError("SV structural summary is invalid")
        return values

    def build(
        self,
        windows: Sequence[Optional[SVHardWindowFeatures]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not windows:
            raise ValueError("SV static extractor requires windows")
        reference = next(
            (
                window.adjacency
                for window in windows
                if window is not None
            ),
            None,
        )
        if reference is None:
            raise ValueError("SV sample has no valid hard windows")
        spectra = []
        structures = []
        states = []
        window_mask_values = []
        for window in windows:
            if window is None:
                states.append(None)
                window_mask_values.append(False)
                continue
            laplacian = self.laplacian(
                window.adjacency, edge_mask=window.edge_mask
            )
            quantiles = self.spectral(laplacian).quantiles
            spectra.append(quantiles)
            structures.append(
                self._window_structure(
                    window.adjacency,
                    window.communities,
                    window.edge_mask,
                )
            )
            states.append(quantiles)
            window_mask_values.append(True)
        spectral_mean = torch.stack(spectra, dim=0).mean(dim=0)
        structure_mean = torch.stack(structures, dim=0).mean(dim=0)
        static_features = torch.cat(
            (spectral_mean, structure_mean), dim=0
        )

        differences = []
        transition_mask_values = []
        for left, right in zip(states[:-1], states[1:]):
            valid = left is not None and right is not None
            transition_mask_values.append(valid)
            if valid:
                differences.append((right - left).abs())
        variation = (
            torch.stack(differences, dim=0).mean(dim=0)
            if differences
            else reference.new_zeros((SV_VARIATION_DIM,))
        )
        window_mask = torch.tensor(
            window_mask_values,
            dtype=torch.bool,
            device=reference.device,
        )
        transition_mask = torch.tensor(
            transition_mask_values,
            dtype=torch.bool,
            device=reference.device,
        )
        if tuple(static_features.shape) != (
            SV_STATIC_FEATURE_DIM,
        ):
            raise RuntimeError("SV static feature vector is not 28-D")
        if tuple(variation.shape) != (SV_VARIATION_DIM,):
            raise RuntimeError("SV variation vector is not 16-D")
        if not bool(torch.isfinite(static_features).all()) or not bool(
            torch.isfinite(variation).all()
        ):
            raise ValueError("SV sample features contain non-finite values")
        return (
            static_features,
            variation,
            window_mask,
            transition_mask,
        )


class SVHardSampleFeatureBuilder(object):
    """Build every frozen feature required by the SV/Signed-GIN stages."""

    def __init__(
        self,
        laplacian_eta: float = 1.0e-3,
        epsilon: float = 1.0e-8,
    ) -> None:
        self.node = SVHardNodeFeatureBuilder(epsilon=epsilon)
        self.summary = SVStaticVariationExtractor(
            laplacian_eta=laplacian_eta, epsilon=epsilon
        )

    def build(
        self, windows: Sequence[Optional[HardGraphWindow]]
    ) -> SVHardSampleFeatures:
        node_windows = self.node.build_sequence(windows)
        static, variation, window_mask, transition_mask = (
            self.summary.build(node_windows)
        )
        return SVHardSampleFeatures(
            windows=node_windows,
            static_features=static,
            variation=variation,
            window_mask=window_mask,
            transition_mask=transition_mask,
        )
