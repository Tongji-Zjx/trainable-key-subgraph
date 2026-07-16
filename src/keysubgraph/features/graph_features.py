"""Node alignment, temporal differences, and signed graph features."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample


@dataclass(frozen=True)
class GraphTimepointFeatures:
    time_index: int
    node_features: torch.Tensor
    edge_features: torch.Tensor
    degree: torch.Tensor
    positive_degree: torch.Tensor
    negative_degree: torch.Tensor
    positive_ratio: torch.Tensor
    negative_ratio: torch.Tensor
    delta_degree: torch.Tensor
    delta_degree_mask: torch.Tensor
    community_features: torch.Tensor
    delta_edge_weight: torch.Tensor
    delta_edge_mask: torch.Tensor
    edge_mask: torch.Tensor

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features.shape[-1])

    @property
    def edge_feature_dim(self) -> int:
        return int(self.edge_features.shape[-1])


def align_current_to_previous(
    current_node_names: Sequence[str], previous_node_names: Sequence[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map each current node to its previous index, with -1 for missing nodes."""

    if len(set(current_node_names)) != len(current_node_names):
        raise ValueError("current node names must be unique")
    if len(set(previous_node_names)) != len(previous_node_names):
        raise ValueError("previous node names must be unique")
    previous_lookup = {name: index for index, name in enumerate(previous_node_names)}
    indices = torch.tensor(
        [previous_lookup.get(name, -1) for name in current_node_names],
        dtype=torch.long,
    )
    return indices, indices >= 0


class GraphFeatureBuilder:
    """Construct one dense timepoint at a time to limit peak memory usage."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def _temporal_differences(
        self, sample: GraphSequenceSample, time_index: int, degree: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = sample.adjacency[time_index]
        node_count = graph.shape[0]
        if time_index == 0:
            return (
                torch.zeros_like(degree),
                torch.zeros(node_count, dtype=torch.bool, device=graph.device),
                torch.zeros_like(graph),
                torch.zeros_like(graph, dtype=torch.bool),
            )

        previous_graph = sample.adjacency[time_index - 1]
        previous_degree = previous_graph.abs().sum(dim=-1)
        previous_indices_cpu, presence_cpu = align_current_to_previous(
            sample.node_names[time_index], sample.node_names[time_index - 1]
        )
        previous_indices = previous_indices_cpu.to(device=graph.device)
        presence = presence_cpu.to(device=graph.device)
        safe_indices = previous_indices.clamp_min(0)

        delta_degree = torch.zeros_like(degree)
        delta_degree[presence] = (
            degree[presence] - previous_degree[safe_indices[presence]]
        )

        previous_aligned = previous_graph.index_select(0, safe_indices).index_select(
            1, safe_indices
        )
        delta_edge_mask = presence[:, None] & presence[None, :]
        delta_edge_mask.fill_diagonal_(False)
        delta_edge = torch.where(
            delta_edge_mask, graph - previous_aligned, torch.zeros_like(graph)
        )
        return delta_degree, presence, delta_edge, delta_edge_mask

    def _community_features(
        self,
        adjacency: torch.Tensor,
        communities: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        node_count = adjacency.shape[0]
        dtype = adjacency.dtype
        device = adjacency.device
        same_community = communities[:, None] == communities[None, :]
        not_self = ~torch.eye(node_count, dtype=torch.bool, device=device)
        intra_mask = same_community & not_self
        inter_mask = ~same_community
        positive = torch.where(
            adjacency > threshold, adjacency, torch.zeros_like(adjacency)
        )
        negative_magnitude = torch.where(
            adjacency < -threshold, -adjacency, torch.zeros_like(adjacency)
        )

        community_sizes = same_community.sum(dim=-1).to(dtype=dtype)
        intra_denominator = (community_sizes - 1.0).clamp_min(1.0)
        inter_denominator = (float(node_count) - community_sizes).clamp_min(1.0)
        relative_size = community_sizes / (float(node_count) + self.epsilon)
        intra_positive = (positive * intra_mask).sum(dim=-1) / intra_denominator
        intra_negative = (
            negative_magnitude * intra_mask
        ).sum(dim=-1) / intra_denominator
        inter_positive = (positive * inter_mask).sum(dim=-1) / inter_denominator
        inter_negative = (
            negative_magnitude * inter_mask
        ).sum(dim=-1) / inter_denominator

        positive_density = torch.zeros(node_count, dtype=dtype, device=device)
        negative_density = torch.zeros(node_count, dtype=dtype, device=device)
        for community_id in torch.unique(communities):
            members = torch.nonzero(communities == community_id, as_tuple=False).flatten()
            size = int(members.numel())
            if size < 2:
                continue
            subgraph = adjacency.index_select(0, members).index_select(1, members)
            upper = torch.triu(torch.ones_like(subgraph, dtype=torch.bool), diagonal=1)
            denominator = float(size * (size - 1)) + self.epsilon
            positive_value = 2.0 * ((subgraph > threshold) & upper).sum().to(dtype) / denominator
            negative_value = 2.0 * ((subgraph < -threshold) & upper).sum().to(dtype) / denominator
            positive_density[members] = positive_value
            negative_density[members] = negative_value

        return torch.stack(
            (
                relative_size,
                intra_positive,
                intra_negative,
                inter_positive,
                inter_negative,
                positive_density,
                negative_density,
            ),
            dim=-1,
        )

    def build_static_node_features(
        self,
        adjacency: torch.Tensor,
        communities: torch.Tensor,
        edge_presence_threshold: float,
    ) -> torch.Tensor:
        """Build signed structural node features without temporal differences.

        The returned columns are absolute degree, positive degree, negative
        magnitude degree, positive/negative ratios, and seven community
        structural features. Community identifiers are only used for grouping.
        """

        if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if communities.dim() != 1 or communities.numel() != adjacency.shape[0]:
            raise ValueError("community labels must align with adjacency")
        if edge_presence_threshold < 0.0:
            raise ValueError("edge_presence_threshold must be non-negative")
        degree = adjacency.abs().sum(dim=-1)
        positive_degree = adjacency.clamp_min(0.0).sum(dim=-1)
        negative_degree = (-adjacency.clamp_max(0.0)).sum(dim=-1)
        denominator = positive_degree + negative_degree + self.epsilon
        positive_ratio = positive_degree / denominator
        negative_ratio = negative_degree / denominator
        community_features = self._community_features(
            adjacency, communities, edge_presence_threshold
        )
        features = torch.cat(
            (
                degree.unsqueeze(-1),
                positive_degree.unsqueeze(-1),
                negative_degree.unsqueeze(-1),
                positive_ratio.unsqueeze(-1),
                negative_ratio.unsqueeze(-1),
                community_features,
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("static node features contain non-finite values")
        return features

    def build_timepoint(
        self, sample: GraphSequenceSample, time_index: int
    ) -> GraphTimepointFeatures:
        if time_index < 0 or time_index >= sample.num_timepoints:
            raise IndexError("time_index is out of range")
        adjacency = sample.adjacency[time_index]
        communities = sample.communities[time_index]
        degree = adjacency.abs().sum(dim=-1)
        positive_degree = adjacency.clamp_min(0.0).sum(dim=-1)
        negative_degree = (-adjacency.clamp_max(0.0)).sum(dim=-1)
        signed_degree_denominator = positive_degree + negative_degree + self.epsilon
        positive_ratio = positive_degree / signed_degree_denominator
        negative_ratio = negative_degree / signed_degree_denominator
        delta_degree, delta_degree_mask, delta_edge, delta_edge_mask = (
            self._temporal_differences(sample, time_index, degree)
        )
        community_features = self._community_features(
            adjacency, communities, sample.edge_presence_threshold
        )
        node_features = torch.cat(
            (
                degree.unsqueeze(-1),
                positive_degree.unsqueeze(-1),
                negative_degree.unsqueeze(-1),
                positive_ratio.unsqueeze(-1),
                negative_ratio.unsqueeze(-1),
                delta_degree.unsqueeze(-1),
                community_features,
            ),
            dim=-1,
        )

        edge_features = torch.stack(
            (
                adjacency,
                adjacency.abs(),
                delta_edge,
                delta_edge.abs(),
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(node_features).all()):
            raise ValueError("node features contain non-finite values")
        if not bool(torch.isfinite(edge_features).all()):
            raise ValueError("edge features contain non-finite values")
        return GraphTimepointFeatures(
            time_index=time_index,
            node_features=node_features,
            edge_features=edge_features,
            degree=degree,
            positive_degree=positive_degree,
            negative_degree=negative_degree,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            delta_degree=delta_degree,
            delta_degree_mask=delta_degree_mask,
            community_features=community_features,
            delta_edge_weight=delta_edge,
            delta_edge_mask=delta_edge_mask,
            edge_mask=sample.edge_mask[time_index],
        )

    def iter_sample(
        self, sample: GraphSequenceSample
    ) -> Iterator[GraphTimepointFeatures]:
        for time_index in range(sample.num_timepoints):
            yield self.build_timepoint(sample, time_index)
