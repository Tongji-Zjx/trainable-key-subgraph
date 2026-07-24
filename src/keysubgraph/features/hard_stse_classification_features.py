"""Classification features recomputed from Hard-STSE hard union graphs."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.models.hard_stse_types import HardWindowOutput
from .graph_features import align_current_to_previous
from .hard_stse_extractor_features import _local_clustering


@dataclass(frozen=True)
class HardSTSEClassificationFeatures:
    time_index: int
    node_features: torch.Tensor
    edge_features: torch.Tensor
    graph_statistics: torch.Tensor
    graph_statistic_mask: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    delta_degree_mask: torch.Tensor
    delta_edge_mask: torch.Tensor
    source: str = "hard_stse_recomputed"


def _safe_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum()
    valid = count > 0
    mean = (
        (values * mask.to(values.dtype)).sum()
        / count.to(values.dtype).clamp_min(1.0)
    )
    return mean, valid


def _component_count(edge_mask: torch.Tensor, node_mask: torch.Tensor) -> int:
    remaining = set(torch.nonzero(node_mask, as_tuple=False).flatten().tolist())
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbors = torch.nonzero(
                edge_mask[current], as_tuple=False
            ).flatten().tolist()
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _aligned_previous(
    sample: GraphSequenceSample,
    time_index: int,
    previous: HardWindowOutput,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_names = sample.node_names[time_index]
    previous_names = sample.node_names[time_index - 1]
    indices_cpu, present_cpu = align_current_to_previous(
        current_names, previous_names
    )
    device = previous.adjacency_st.device
    indices = indices_cpu.to(device=device)
    present = present_cpu.to(device=device)
    safe = indices.clamp_min(0)
    adjacency = previous.adjacency_st.index_select(0, safe).index_select(1, safe)
    nodes = previous.hard_node_mask.index_select(0, safe)
    edges = previous.hard_edge_mask.index_select(0, safe).index_select(1, safe)
    return adjacency, nodes, edges, present


class HardSTSEClassificationFeatureBuilder(object):
    node_feature_dim = 14
    edge_feature_dim = 7
    graph_statistic_dim = 14

    def __init__(self, epsilon: float = 1.0e-8) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def build_timepoint(
        self,
        sample: GraphSequenceSample,
        time_index: int,
        current: HardWindowOutput,
        previous: Optional[HardWindowOutput],
    ) -> HardSTSEClassificationFeatures:
        if time_index < 0 or time_index >= sample.num_timepoints:
            raise IndexError("hard classification time index is invalid")
        if not current.window_valid:
            raise ValueError("invalid hard windows cannot enter classification features")
        adjacency = current.adjacency_st
        node_mask = current.hard_node_mask.to(device=adjacency.device)
        edge_mask = current.hard_edge_mask.to(device=adjacency.device)
        communities = sample.communities[time_index].to(
            device=adjacency.device, dtype=torch.long
        )
        node_count = adjacency.shape[0]
        if tuple(node_mask.shape) != (node_count,):
            raise ValueError("hard node mask does not align with adjacency")

        degree = adjacency.abs().sum(dim=-1)
        positive_degree = adjacency.clamp_min(0.0).sum(dim=-1)
        negative_degree = (-adjacency.clamp_max(0.0)).sum(dim=-1)
        delta_degree = torch.zeros_like(degree)
        delta_degree_mask = torch.zeros_like(node_mask)
        delta_edge = torch.zeros_like(adjacency)
        delta_edge_mask = torch.zeros_like(edge_mask)
        previous_node_aligned = torch.zeros_like(node_mask)
        previous_edge_aligned = torch.zeros_like(edge_mask)
        node_birth = torch.zeros_like(degree)
        edge_birth = torch.zeros_like(adjacency)
        node_death_ratio = adjacency.new_zeros(())
        edge_death_ratio = adjacency.new_zeros(())
        transition_valid = False

        if time_index > 0 and previous is not None and previous.window_valid:
            previous_adjacency, previous_nodes, previous_edges, present = (
                _aligned_previous(sample, time_index, previous)
            )
            present = present.to(device=adjacency.device)
            previous_adjacency = previous_adjacency.to(device=adjacency.device)
            previous_nodes = previous_nodes.to(device=adjacency.device)
            previous_edges = previous_edges.to(device=adjacency.device)
            previous_node_aligned = previous_nodes & present
            previous_edge_aligned = (
                previous_edges
                & present[:, None]
                & present[None, :]
            )
            delta_degree_mask = node_mask & previous_node_aligned
            delta_degree = torch.where(
                delta_degree_mask,
                degree - previous_adjacency.abs().sum(dim=-1),
                torch.zeros_like(degree),
            )
            aligned_pairs = (
                node_mask[:, None]
                & node_mask[None, :]
                & previous_node_aligned[:, None]
                & previous_node_aligned[None, :]
            )
            union_edges = edge_mask | previous_edge_aligned
            delta_edge_mask = aligned_pairs & union_edges
            delta_edge_mask = delta_edge_mask.clone()
            delta_edge_mask.fill_diagonal_(False)
            delta_edge = torch.where(
                delta_edge_mask,
                adjacency - previous_adjacency,
                torch.zeros_like(adjacency),
            )
            node_birth = (node_mask & ~previous_node_aligned).to(adjacency.dtype)
            edge_birth = (edge_mask & ~previous_edge_aligned).to(adjacency.dtype)
            previous_node_total = previous.hard_node_mask.sum().clamp_min(1)
            previous_edge_total = torch.triu(
                previous.hard_edge_mask, diagonal=1
            ).sum().clamp_min(1)
            current_name_set = set(sample.node_names[time_index])
            previous_selected_names = {
                sample.node_names[time_index - 1][index]
                for index in torch.nonzero(
                    previous.hard_node_mask, as_tuple=False
                ).flatten().tolist()
            }
            current_selected_names = {
                sample.node_names[time_index][index]
                for index in torch.nonzero(node_mask, as_tuple=False).flatten().tolist()
            }
            disappeared_nodes = previous_selected_names - current_selected_names
            node_death_ratio = adjacency.new_tensor(
                len(disappeared_nodes) / float(int(previous_node_total))
            )
            # Edge deaths are evaluated on stable endpoint-name pairs.
            previous_names = sample.node_names[time_index - 1]
            previous_pairs = {
                tuple(sorted((previous_names[left], previous_names[right])))
                for left, right in torch.nonzero(
                    torch.triu(previous.hard_edge_mask, diagonal=1),
                    as_tuple=False,
                ).tolist()
            }
            current_names = sample.node_names[time_index]
            current_pairs = {
                tuple(sorted((current_names[left], current_names[right])))
                for left, right in torch.nonzero(
                    torch.triu(edge_mask, diagonal=1), as_tuple=False
                ).tolist()
            }
            edge_death_ratio = adjacency.new_tensor(
                len(previous_pairs - current_pairs)
                / float(int(previous_edge_total))
            )
            del current_name_set
            transition_valid = True

        delta_count = delta_edge_mask.sum(dim=-1).to(adjacency.dtype)
        mean_abs_delta = (
            delta_edge.abs() * delta_edge_mask.to(adjacency.dtype)
        ).sum(dim=-1) / delta_count.clamp_min(1.0)
        valid_delta_ratio = delta_count / float(max(1, node_count - 1))

        same = communities[:, None] == communities[None, :]
        selected_pair = node_mask[:, None] & node_mask[None, :]
        not_self = ~torch.eye(node_count, dtype=torch.bool, device=adjacency.device)
        same_selected = same & selected_pair & not_self
        inter_selected = ~same & selected_pair
        selected_community_size = same_selected.sum(dim=-1).to(adjacency.dtype) + 1.0
        selected_total = node_mask.sum().to(adjacency.dtype).clamp_min(1.0)
        relative_size = selected_community_size / selected_total
        intra_denominator = (selected_community_size - 1.0).clamp_min(1.0)
        inter_denominator = (
            selected_total - selected_community_size
        ).clamp_min(1.0)
        positive = adjacency.clamp_min(0.0)
        negative = -adjacency.clamp_max(0.0)
        intra_positive = (positive * same_selected).sum(dim=-1) / intra_denominator
        intra_negative = (negative * same_selected).sum(dim=-1) / intra_denominator
        inter_positive = (positive * inter_selected).sum(dim=-1) / inter_denominator
        inter_negative = (negative * inter_selected).sum(dim=-1) / inter_denominator
        clustering = _local_clustering(edge_mask, adjacency.dtype)

        node_features = torch.cat(
            (
                degree[:, None],
                positive_degree[:, None],
                negative_degree[:, None],
                delta_degree[:, None],
                delta_degree_mask.to(adjacency.dtype)[:, None],
                mean_abs_delta[:, None],
                valid_delta_ratio[:, None],
                relative_size[:, None],
                intra_positive[:, None],
                intra_negative[:, None],
                inter_positive[:, None],
                inter_negative[:, None],
                clustering[:, None],
                node_birth[:, None],
            ),
            dim=-1,
        )
        edge_features = torch.stack(
            (
                adjacency,
                adjacency.abs(),
                delta_edge,
                delta_edge.abs(),
                delta_edge_mask.to(adjacency.dtype),
                same.to(adjacency.dtype),
                edge_birth,
            ),
            dim=-1,
        )

        upper = torch.triu(edge_mask, diagonal=1)
        edge_count = upper.sum().to(adjacency.dtype)
        selected_count = node_mask.sum().to(adjacency.dtype)
        density_denominator = (
            selected_count * (selected_count - 1.0)
        ).clamp_min(1.0)
        density = 2.0 * edge_count / density_denominator
        mean_clustering, clustering_valid = _safe_mean(clustering, node_mask)
        absolute = adjacency.abs()
        total_absolute = absolute.sum()
        degree_abs = absolute.sum(dim=-1)
        same_float = same.to(adjacency.dtype)
        modularity = adjacency.new_zeros(())
        if float(total_absolute.detach().cpu()) > 0.0:
            expected = degree_abs[:, None] * degree_abs[None, :] / total_absolute
            modularity = (
                ((absolute - expected) * same_float * selected_pair).sum()
                / total_absolute
            )
        components = adjacency.new_tensor(
            float(_component_count(edge_mask, node_mask))
        )
        upper_positive = upper & (adjacency > 0.0)
        upper_negative = upper & (adjacency < 0.0)
        positive_mean, positive_valid = _safe_mean(adjacency, upper_positive)
        negative_mean, negative_valid = _safe_mean(absolute, upper_negative)
        absolute_mean, edge_valid = _safe_mean(absolute, upper)
        absolute_variance = (
            ((absolute - absolute_mean).square() * upper.to(adjacency.dtype)).sum()
            / upper.sum().to(adjacency.dtype).clamp_min(1.0)
        )
        # A selected hard graph may contain only one retained edge.  Its
        # variance is exactly zero, where the derivative of ``sqrt`` is
        # singular and can turn an otherwise finite STE gradient into NaN.
        # The same numerical convention is used by the feature pooler.
        absolute_std = torch.sqrt(
            absolute_variance.clamp_min(0.0) + self.epsilon
        )
        upper_delta = torch.triu(delta_edge_mask, diagonal=1)
        mean_graph_delta, delta_valid = _safe_mean(delta_edge.abs(), upper_delta)
        node_birth_ratio = node_birth.sum() / selected_count.clamp_min(1.0)
        edge_birth_ratio = (
            torch.triu(edge_birth > 0.0, diagonal=1).sum().to(adjacency.dtype)
            / edge_count.clamp_min(1.0)
        )
        graph_statistics = torch.stack(
            (
                selected_count,
                edge_count,
                density,
                mean_clustering,
                modularity,
                components,
                positive_mean,
                negative_mean,
                absolute_std,
                mean_graph_delta,
                node_birth_ratio,
                node_death_ratio,
                edge_birth_ratio,
                edge_death_ratio,
            )
        )
        transition_tensor = torch.tensor(
            transition_valid, dtype=torch.bool, device=adjacency.device
        )
        graph_statistic_mask = torch.stack(
            (
                torch.tensor(True, device=adjacency.device),
                torch.tensor(True, device=adjacency.device),
                torch.tensor(True, device=adjacency.device),
                clustering_valid,
                edge_valid,
                torch.tensor(True, device=adjacency.device),
                positive_valid,
                negative_valid,
                edge_valid,
                delta_valid,
                transition_tensor,
                transition_tensor,
                transition_tensor,
                transition_tensor,
            )
        ).to(torch.bool)

        if tuple(node_features.shape) != (node_count, self.node_feature_dim):
            raise RuntimeError("hard classifier node schema is not 14-D")
        if tuple(edge_features.shape) != (
            node_count, node_count, self.edge_feature_dim
        ):
            raise RuntimeError("hard classifier edge schema is not 7-D")
        if tuple(graph_statistics.shape) != (self.graph_statistic_dim,):
            raise RuntimeError("hard graph statistic schema is not 14-D")
        for value in (node_features, edge_features, graph_statistics):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("hard classification features are non-finite")
        return HardSTSEClassificationFeatures(
            time_index=time_index,
            node_features=node_features,
            edge_features=edge_features,
            graph_statistics=graph_statistics,
            graph_statistic_mask=graph_statistic_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
            delta_degree_mask=delta_degree_mask,
            delta_edge_mask=delta_edge_mask,
        )
