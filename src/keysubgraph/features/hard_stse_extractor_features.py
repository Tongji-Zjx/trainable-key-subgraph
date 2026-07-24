"""Extractor features for Hard-STSE with explicit temporal-validity masks."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample
from .graph_features import GraphFeatureBuilder


@dataclass(frozen=True)
class HardSTSEExtractorFeatures:
    time_index: int
    node_features: torch.Tensor
    edge_base_features: torch.Tensor
    edge_presence_mask: torch.Tensor
    delta_degree_mask: torch.Tensor
    delta_edge_mask: torch.Tensor
    communities: torch.Tensor


def _local_clustering(edge_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if edge_mask.ndim != 2 or edge_mask.shape[0] != edge_mask.shape[1]:
        raise ValueError("edge mask must be square")
    binary = edge_mask.to(dtype=dtype)
    binary = 0.5 * (binary + binary.transpose(0, 1))
    binary = (binary > 0.0).to(dtype)
    binary = binary.clone()
    binary.fill_diagonal_(0.0)
    degree = binary.sum(dim=-1)
    closed_walks = torch.diagonal(binary.matmul(binary).matmul(binary))
    denominator = degree * (degree - 1.0)
    return torch.where(
        denominator > 0.0,
        closed_walks / denominator.clamp_min(1.0),
        torch.zeros_like(degree),
    )


class HardSTSEExtractorFeatureBuilder(object):
    """Build the verified 15-D node and 6-D edge-base schemas."""

    node_feature_dim = 15
    edge_base_feature_dim = 6

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.base = GraphFeatureBuilder(epsilon=epsilon)

    def build_timepoint(
        self, sample: GraphSequenceSample, time_index: int
    ) -> HardSTSEExtractorFeatures:
        base = self.base.build_timepoint(sample, time_index)
        adjacency = sample.adjacency[time_index]
        communities = sample.communities[time_index].to(
            device=adjacency.device, dtype=torch.long
        )
        delta_mask = base.delta_edge_mask.to(device=adjacency.device)
        delta_count = delta_mask.sum(dim=-1).to(dtype=adjacency.dtype)
        possible = max(1, adjacency.shape[0] - 1)
        valid_delta_ratio = delta_count / float(possible)
        mean_abs_delta = (
            base.delta_edge_weight.abs()
            * delta_mask.to(dtype=adjacency.dtype)
        ).sum(dim=-1) / delta_count.clamp_min(1.0)
        clustering = _local_clustering(base.edge_mask, adjacency.dtype)

        # Community features are relative size, intra +/- mean strength,
        # inter +/- mean strength and intra +/- density.
        node_features = torch.cat(
            (
                base.degree[:, None],
                base.positive_degree[:, None],
                base.negative_degree[:, None],
                base.delta_degree[:, None],
                base.delta_degree_mask.to(adjacency.dtype)[:, None],
                mean_abs_delta[:, None],
                valid_delta_ratio[:, None],
                base.community_features,
                clustering[:, None],
            ),
            dim=-1,
        )
        same_community = communities[:, None] == communities[None, :]
        edge_base_features = torch.stack(
            (
                adjacency,
                adjacency.abs(),
                base.delta_edge_weight,
                base.delta_edge_weight.abs(),
                delta_mask.to(adjacency.dtype),
                same_community.to(adjacency.dtype),
            ),
            dim=-1,
        )
        expected_nodes = (adjacency.shape[0], self.node_feature_dim)
        expected_edges = (
            adjacency.shape[0],
            adjacency.shape[0],
            self.edge_base_feature_dim,
        )
        if tuple(node_features.shape) != expected_nodes:
            raise RuntimeError("Hard-STSE extractor node schema is not 15-D")
        if tuple(edge_base_features.shape) != expected_edges:
            raise RuntimeError("Hard-STSE extractor edge schema is not 6-D")
        if not bool(torch.isfinite(node_features).all()):
            raise ValueError("Hard-STSE extractor node features are non-finite")
        if not bool(torch.isfinite(edge_base_features).all()):
            raise ValueError("Hard-STSE extractor edge features are non-finite")
        return HardSTSEExtractorFeatures(
            time_index=time_index,
            node_features=node_features,
            edge_base_features=edge_base_features,
            edge_presence_mask=base.edge_mask,
            delta_degree_mask=base.delta_degree_mask,
            delta_edge_mask=base.delta_edge_mask,
            communities=communities,
        )

