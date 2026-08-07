"""Real-candidate exploration initialization for the multi-object selector.

No node or edge field is averaged across windows.  Every selected anchor is a
subgraph that was actually hardened in one exploration window.  The discrete
medoid indices are treated like top-k indices: forward routing is hard while
the selected candidate tensors retain their original differentiable paths.
"""

from __future__ import absolute_import, division, print_function

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .theory_multi_object_selector import MultiObjectTemporalMemory


@dataclass(frozen=True)
class ExplorationObjectCandidate:
    window_index: int
    object_index: int
    quality: float
    memory: MultiObjectTemporalMemory


@dataclass(frozen=True)
class ExplorationMedoidSelection:
    anchor_indices: Tuple[int, ...]
    recent_indices: Tuple[int, ...]
    assignments: Tuple[int, ...]
    support_window_counts: Tuple[int, ...]
    shortlist_indices: Tuple[int, ...]
    similarity: torch.Tensor
    objective: float
    mean_anchor_similarity: float
    mean_cluster_similarity: float


def stack_real_candidate_memories(
    candidates: Sequence[ExplorationObjectCandidate],
    indices: Sequence[int],
    current_node_ids: Sequence[str],
    adjacency: torch.Tensor,
    edge_presence_mask: torch.Tensor,
) -> MultiObjectTemporalMemory:
    """Transport K singleton real candidates into one current ROI order.

    Transport only reindexes stable ROI identities and removes edges absent in
    the current graph.  It never averages two candidates or invents an edge.
    """

    current = tuple(str(value) for value in current_node_ids)
    if len(current) != len(set(current)):
        raise ValueError("current candidate reference identities must be unique")
    valid = edge_presence_mask.to(device=adjacency.device, dtype=torch.bool)
    valid = valid & valid.transpose(0, 1)
    valid = valid.clone()
    valid.fill_diagonal_(False)
    nodes = []
    edges = []
    signed = []
    hard_nodes = []
    hard_edges = []
    seeds = []
    states = []
    representations = []
    centroids = []
    spectra = []
    for candidate_index in indices:
        memory = candidates[int(candidate_index)].memory
        if int(memory.node_probabilities.shape[0]) != 1:
            raise ValueError("candidate memory must contain exactly one object")
        previous = tuple(str(value) for value in memory.node_ids)
        lookup = {value: index for index, value in enumerate(previous)}
        pairs = [
            (new_index, lookup[value])
            for new_index, value in enumerate(current)
            if value in lookup
        ]
        new_indices = torch.tensor(
            [item[0] for item in pairs],
            dtype=torch.long,
            device=adjacency.device,
        )
        old_indices = torch.tensor(
            [item[1] for item in pairs],
            dtype=torch.long,
            device=adjacency.device,
        )
        node = adjacency.new_zeros((len(current),))
        edge = adjacency.new_zeros((len(current), len(current)))
        signed_edge = adjacency.new_zeros((len(current), len(current)))
        hard_node = torch.zeros(
            len(current), dtype=torch.bool, device=adjacency.device
        )
        hard_edge = torch.zeros_like(valid)
        if pairs:
            node[new_indices] = memory.node_probabilities.to(adjacency)[
                0
            ].index_select(0, old_indices)
            selected_edge = memory.edge_probabilities.to(adjacency)[0].index_select(
                0, old_indices
            ).index_select(1, old_indices)
            edge[new_indices[:, None], new_indices[None, :]] = selected_edge
            if memory.signed_edge_values is not None:
                selected_signed = memory.signed_edge_values.to(adjacency)[
                    0
                ].index_select(0, old_indices).index_select(1, old_indices)
                signed_edge[
                    new_indices[:, None], new_indices[None, :]
                ] = selected_signed
            if memory.hard_node_masks is not None:
                hard_node[new_indices] = memory.hard_node_masks.to(adjacency.device)[
                    0
                ].index_select(0, old_indices)
            if memory.hard_edge_masks is not None:
                selected_hard = memory.hard_edge_masks.to(adjacency.device)[
                    0
                ].index_select(0, old_indices).index_select(1, old_indices)
                hard_edge[new_indices[:, None], new_indices[None, :]] = selected_hard
        edge = edge * valid.to(edge.dtype)
        signed_edge = signed_edge * valid.to(signed_edge.dtype)
        hard_edge = hard_edge & valid
        seed = -1
        if memory.seed_indices is not None:
            previous_seed = int(memory.seed_indices[0])
            if 0 <= previous_seed < len(previous):
                seed_name = previous[previous_seed]
                seed = current.index(seed_name) if seed_name in current else -1
        nodes.append(node)
        edges.append(edge)
        signed.append(signed_edge)
        hard_nodes.append(hard_node)
        hard_edges.append(hard_edge)
        seeds.append(seed)
        states.append(memory.object_states.to(adjacency)[0])
        if memory.object_representations is not None:
            representations.append(memory.object_representations.to(adjacency)[0])
        if memory.coordinate_centroids is not None:
            centroids.append(memory.coordinate_centroids.to(adjacency)[0])
        if memory.spectral_descriptors is not None:
            spectra.append(memory.spectral_descriptors.to(adjacency)[0])

    def optional_stack(values):
        return torch.stack(values) if len(values) == len(indices) else None

    return MultiObjectTemporalMemory(
        object_states=torch.stack(states),
        node_probabilities=torch.stack(nodes),
        edge_probabilities=torch.stack(edges),
        node_ids=current,
        signed_edge_values=torch.stack(signed),
        object_representations=optional_stack(representations),
        coordinate_centroids=optional_stack(centroids),
        spectral_descriptors=optional_stack(spectra),
        hard_node_masks=torch.stack(hard_nodes),
        hard_edge_masks=torch.stack(hard_edges),
        seed_indices=torch.tensor(seeds, dtype=torch.long, device=adjacency.device),
        alignment_confidence=adjacency.new_ones((len(indices),)),
        consensus_weight=1.0,
        consensus_object_weights=None,
        alignment_observation_count=0,
    )


def singleton_memory(
    memory: MultiObjectTemporalMemory,
    object_index: int,
    hard_node_mask: torch.Tensor,
    hard_edge_mask: torch.Tensor,
    seed_index: torch.Tensor,
) -> MultiObjectTemporalMemory:
    """Extract one actual object without averaging any structural field."""

    index = int(object_index)

    def row(value):
        return None if value is None else value[index : index + 1]

    return MultiObjectTemporalMemory(
        object_states=row(memory.object_states),
        node_probabilities=row(memory.node_probabilities),
        edge_probabilities=row(memory.edge_probabilities),
        node_ids=memory.node_ids,
        signed_edge_values=row(memory.signed_edge_values),
        object_representations=row(memory.object_representations),
        coordinate_centroids=row(memory.coordinate_centroids),
        spectral_descriptors=row(memory.spectral_descriptors),
        hard_node_masks=hard_node_mask[None, :],
        hard_edge_masks=hard_edge_mask[None, :, :],
        seed_indices=seed_index.reshape(1),
        alignment_confidence=None,
        consensus_weight=1.0,
        consensus_object_weights=None,
        alignment_observation_count=0,
    )


def candidate_quality(
    memory: MultiObjectTemporalMemory,
    epsilon: float = 1.0e-8,
) -> float:
    """Local selector confidence on the actual selected nodes and edges."""

    if memory.hard_node_masks is None or memory.hard_edge_masks is None:
        raise ValueError("exploration candidate requires hard masks")
    nodes = memory.hard_node_masks[0].to(torch.bool)
    edges = torch.triu(memory.hard_edge_masks[0].to(torch.bool), diagonal=1)
    node_value = (
        memory.node_probabilities[0][nodes].mean()
        if bool(nodes.any())
        else memory.node_probabilities.new_zeros(())
    )
    edge_value = (
        memory.edge_probabilities[0][edges].mean()
        if bool(edges.any())
        else memory.edge_probabilities.new_zeros(())
    )
    return float((0.60 * node_value + 0.40 * edge_value).detach().cpu())


def _node_ids(candidate: ExplorationObjectCandidate) -> set:
    memory = candidate.memory
    mask = memory.hard_node_masks[0].detach().cpu().to(torch.bool)
    return {
        str(memory.node_ids[index])
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()
    }


def _signed_edges(candidate: ExplorationObjectCandidate) -> Dict[Tuple[str, str], float]:
    memory = candidate.memory
    mask = memory.hard_edge_masks[0].detach().cpu().to(torch.bool)
    values = (
        memory.signed_edge_values[0].detach().cpu()
        if memory.signed_edge_values is not None
        else torch.zeros_like(mask, dtype=torch.float32)
    )
    result = {}
    for left, right in torch.nonzero(
        torch.triu(mask, diagonal=1), as_tuple=False
    ).tolist():
        names = sorted((str(memory.node_ids[left]), str(memory.node_ids[right])))
        result[(names[0], names[1])] = float(values[left, right])
    return result


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / float(len(union)) if union else 1.0


def _signed_edge_similarity(
    left: Mapping[Tuple[str, str], float],
    right: Mapping[Tuple[str, str], float],
    epsilon: float,
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        first = float(left.get(key, 0.0))
        second = float(right.get(key, 0.0))
        denominator += max(abs(first), abs(second))
        if first * second > 0.0:
            numerator += min(abs(first), abs(second))
    return numerator / max(float(epsilon), denominator)


def _cosine(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> float:
    if left is None or right is None:
        return 0.5
    first = left.detach().cpu().reshape(1, -1).to(torch.float64)
    second = right.detach().cpu().reshape(1, -1).to(torch.float64)
    if float(first.norm()) <= 1.0e-12 or float(second.norm()) <= 1.0e-12:
        return 0.5
    return float((0.5 * (F.cosine_similarity(first, second) + 1.0))[0])


def exploration_candidate_similarity(
    candidates: Sequence[ExplorationObjectCandidate],
    weights: Optional[Mapping[str, float]] = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Pairwise real-subgraph similarity using only correspondence metadata."""

    count = len(candidates)
    if count < 1:
        raise ValueError("exploration candidate pool cannot be empty")
    configured = {
        "node": 0.40,
        "signed_edge": 0.25,
        "latent": 0.15,
        "coordinate": 0.10,
        "spectral": 0.10,
    }
    if weights is not None:
        configured.update({key: float(value) for key, value in weights.items()})
    if any(value < 0.0 for value in configured.values()) or sum(
        configured.values()
    ) <= 0:
        raise ValueError("exploration similarity weights are invalid")
    node_sets = [_node_ids(item) for item in candidates]
    edge_maps = [_signed_edges(item) for item in candidates]
    centroids = [
        item.memory.coordinate_centroids[0].detach().cpu().to(torch.float64)
        if item.memory.coordinate_centroids is not None
        else None
        for item in candidates
    ]
    usable = [
        item
        for item in centroids
        if item is not None and bool(torch.isfinite(item).all())
    ]
    coordinate_scale = 1.0
    coordinate_informative = False
    if len(usable) > 1:
        distances = torch.pdist(torch.stack(usable))
        positive = distances[distances > float(epsilon)]
        if positive.numel():
            coordinate_informative = True
            coordinate_scale = max(float(epsilon), float(positive.median()))
    result = torch.eye(count, dtype=torch.float64)
    denominator = float(sum(configured.values()))
    for left in range(count):
        for right in range(left + 1, count):
            coordinate = 0.5
            if (
                coordinate_informative
                and centroids[left] is not None
                and centroids[right] is not None
            ):
                coordinate = float(
                    torch.exp(
                        -(centroids[left] - centroids[right]).norm()
                        / coordinate_scale
                    )
                )
            components = {
                "node": _jaccard(node_sets[left], node_sets[right]),
                "signed_edge": _signed_edge_similarity(
                    edge_maps[left], edge_maps[right], epsilon
                ),
                "latent": _cosine(
                    candidates[left].memory.object_representations,
                    candidates[right].memory.object_representations,
                ),
                "coordinate": coordinate,
                "spectral": _cosine(
                    candidates[left].memory.spectral_descriptors,
                    candidates[right].memory.spectral_descriptors,
                ),
            }
            value = sum(configured[key] * components[key] for key in configured) / denominator
            result[left, right] = value
            result[right, left] = value
    return result.to(torch.float32)


def _window_normalized_quality(
    candidates: Sequence[ExplorationObjectCandidate],
) -> torch.Tensor:
    result = torch.zeros(len(candidates), dtype=torch.float32)
    windows = sorted({int(item.window_index) for item in candidates})
    for window in windows:
        indices = [
            index for index, item in enumerate(candidates)
            if int(item.window_index) == window
        ]
        values = torch.tensor([candidates[index].quality for index in indices])
        scale = float(values.max() - values.min())
        normalized = (
            (values - values.min()) / scale
            if scale > 1.0e-8
            else torch.full_like(values, 0.5)
        )
        result[torch.tensor(indices, dtype=torch.long)] = normalized
    return result


def select_exploration_medoids(
    candidates: Sequence[ExplorationObjectCandidate],
    object_count: int,
    similarity_threshold: float = 0.45,
    shortlist_multiplier: int = 3,
    coverage_weight: float = 0.45,
    support_weight: float = 0.25,
    quality_weight: float = 0.15,
    diversity_weight: float = 0.15,
    similarity_weights: Optional[Mapping[str, float]] = None,
) -> ExplorationMedoidSelection:
    """Select K diverse real medoids and each cluster's most recent member."""

    count = len(candidates)
    object_count = int(object_count)
    if object_count < 1 or count < object_count:
        raise ValueError("candidate pool must contain at least K objects")
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("candidate similarity threshold must lie in [0,1]")
    if shortlist_multiplier < 1:
        raise ValueError("candidate shortlist multiplier must be positive")
    objective_weights = (
        coverage_weight, support_weight, quality_weight, diversity_weight
    )
    if any(float(value) < 0.0 for value in objective_weights) or sum(
        objective_weights
    ) <= 0:
        raise ValueError("medoid objective weights are invalid")
    similarity = exploration_candidate_similarity(
        candidates, weights=similarity_weights
    )
    quality = _window_normalized_quality(candidates)
    windows = sorted({int(item.window_index) for item in candidates})
    window_count = max(1, len(windows))
    support_counts = []
    representativeness = []
    for index in range(count):
        supported = {
            int(candidates[other].window_index)
            for other in range(count)
            if float(similarity[index, other]) >= float(similarity_threshold)
        }
        support_counts.append(len(supported))
        per_window = []
        for window in windows:
            indices = [
                other for other, candidate in enumerate(candidates)
                if int(candidate.window_index) == window
            ]
            per_window.append(max(float(similarity[index, other]) for other in indices))
        representativeness.append(sum(per_window) / float(window_count))
    base = torch.tensor(representativeness) * 0.55
    base = base + torch.tensor(support_counts, dtype=torch.float32) / float(window_count) * 0.30
    base = base + quality * 0.15
    shortlist_size = min(count, max(object_count, object_count * int(shortlist_multiplier)))
    # N is itself a representative pool, rather than the N largest local
    # scores.  MMR keeps a dominant structure from occupying every shortlist
    # position before the exact K-medoid search begins.
    selected_shortlist: List[int] = []
    remaining = set(range(count))
    while len(selected_shortlist) < shortlist_size:
        def shortlist_key(index):
            diversity = (
                1.0
                if not selected_shortlist
                else 1.0
                - max(
                    float(similarity[index, chosen])
                    for chosen in selected_shortlist
                )
            )
            score = 0.75 * float(base[index]) + 0.25 * diversity
            return (
                score,
                float(base[index]),
                -int(candidates[index].window_index),
                -index,
            )

        chosen = max(remaining, key=shortlist_key)
        selected_shortlist.append(chosen)
        remaining.remove(chosen)
    shortlist = tuple(selected_shortlist)
    best = None
    weight_sum = float(sum(objective_weights))
    for combination in itertools.combinations(shortlist, object_count):
        coverage = float(similarity[:, list(combination)].max(dim=1).values.mean())
        support = sum(
            support_counts[index] / float(window_count)
            for index in combination
        ) / object_count
        selected_quality = sum(
            float(quality[index]) for index in combination
        ) / object_count
        pair_values = [
            float(similarity[left, right])
            for left, right in itertools.combinations(combination, 2)
        ]
        diversity = 1.0 - (sum(pair_values) / len(pair_values) if pair_values else 0.0)
        objective = (
            coverage_weight * coverage
            + support_weight * support
            + quality_weight * selected_quality
            + diversity_weight * diversity
        ) / weight_sum
        distinct_windows = len(
            {candidates[index].window_index for index in combination}
        )
        candidate_key = (objective, distinct_windows, tuple(-index for index in combination))
        if best is None or candidate_key > best[0]:
            best = (candidate_key, tuple(combination))
    anchors = best[1]
    assignments: List[int] = []
    for index in range(count):
        values = [float(similarity[index, anchor]) for anchor in anchors]
        selected = max(range(object_count), key=lambda slot: (values[slot], -slot))
        assignments.append(
            selected if values[selected] >= float(similarity_threshold) else -1
        )
    # Every medoid is, by definition, a confirmed member of its own cluster.
    for slot, anchor in enumerate(anchors):
        assignments[anchor] = slot
    recent = []
    selected_support = []
    cluster_similarities = []
    for slot, anchor in enumerate(anchors):
        members = [index for index, value in enumerate(assignments) if value == slot]
        members.sort(
            key=lambda index: (
                int(candidates[index].window_index),
                float(candidates[index].quality),
                -index,
            ),
            reverse=True,
        )
        recent.append(members[0])
        selected_support.append(
            len({int(candidates[index].window_index) for index in members})
        )
        cluster_similarities.extend(float(similarity[index, anchor]) for index in members)
    pair_values = [
        float(similarity[left, right])
        for left, right in itertools.combinations(anchors, 2)
    ]
    return ExplorationMedoidSelection(
        anchor_indices=tuple(int(value) for value in anchors),
        recent_indices=tuple(int(value) for value in recent),
        assignments=tuple(int(value) for value in assignments),
        support_window_counts=tuple(int(value) for value in selected_support),
        shortlist_indices=shortlist,
        similarity=similarity,
        objective=float(best[0][0]),
        mean_anchor_similarity=(
            sum(pair_values) / len(pair_values) if pair_values else 0.0
        ),
        mean_cluster_similarity=(
            sum(cluster_similarities) / len(cluster_similarities)
            if cluster_similarities else 0.0
        ),
    )
