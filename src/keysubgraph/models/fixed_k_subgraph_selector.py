"""Deterministic fixed-K hard objects built from differentiable selector scores.

The scorer remains unchanged.  This module only replaces the single-union
hardening rule with community-aware seed selection and connected multi-source
growth.  Every selected edge is an original signed edge; no edge is created
and no sign is changed.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .hard_stse_types import HardSelectionOutput


@dataclass(frozen=True)
class FixedKSelectionOutput:
    union: HardSelectionOutput
    subgraphs: Tuple[HardSelectionOutput, ...]
    subgraph_mask: torch.Tensor
    seed_indices: torch.Tensor
    pairwise_node_overlap: torch.Tensor
    pairwise_edge_overlap: torch.Tensor
    unique_node_fractions: torch.Tensor
    seed_distance_matrix: torch.Tensor
    union_efficiency: float
    diversity_constraint_relaxed: bool


def _valid_edges(mask: torch.Tensor) -> torch.Tensor:
    result = mask.to(dtype=torch.bool)
    result = result & result.transpose(0, 1)
    result = result.clone()
    result.fill_diagonal_(False)
    return result


def _edge_scores(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
) -> torch.Tensor:
    pair = torch.sqrt(
        (
            node_probabilities[:, None]
            * node_probabilities[None, :]
        ).clamp_min(0.0)
    )
    return edge_probabilities * pair


def _seed_order(
    node_probabilities: torch.Tensor,
    communities: torch.Tensor,
) -> Tuple[int, ...]:
    """Prefer one high-score seed per community, then fill globally."""

    representatives = []
    for label in torch.unique(communities, sorted=True):
        members = torch.nonzero(
            communities == label, as_tuple=False
        ).flatten()
        ranked = members.index_select(
            0,
            torch.argsort(
                node_probabilities.index_select(0, members),
                descending=True,
            ),
        )
        representatives.append(int(ranked[0]))
    representatives.sort(
        key=lambda index: (-float(node_probabilities[index]), index)
    )
    selected = list(representatives)
    seen = set(selected)
    remaining = sorted(
        (index for index in range(node_probabilities.numel()) if index not in seen),
        key=lambda index: (-float(node_probabilities[index]), index),
    )
    selected.extend(remaining)
    return tuple(selected)


def _selection_from_masks(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    valid_edges: torch.Tensor,
    hard_node_mask: torch.Tensor,
    hard_edge_mask: torch.Tensor,
    candidate_node_mask: torch.Tensor,
    requested_node_count: int,
    requested_edge_count: int,
    candidate_edge_count: int,
    selection_mode: str,
    original_edge_count: int = None,
    actual_node_count: int = None,
    actual_edge_count: int = None,
) -> HardSelectionOutput:
    hard_node_float = hard_node_mask.to(node_probabilities.dtype)
    hard_edge_float = hard_edge_mask.to(edge_probabilities.dtype)
    straight_node = node_probabilities + (
        hard_node_float - node_probabilities
    ).detach()
    straight_edge = edge_probabilities + (
        hard_edge_float - edge_probabilities
    ).detach()
    straight_edge = straight_edge * valid_edges.to(straight_edge.dtype)
    return HardSelectionOutput(
        node_probabilities=node_probabilities,
        edge_probabilities=edge_probabilities,
        hard_node_mask=hard_node_mask,
        hard_edge_mask=hard_edge_mask,
        candidate_node_mask=candidate_node_mask,
        straight_through_node_mask=straight_node,
        straight_through_edge_mask=straight_edge,
        requested_node_count=int(requested_node_count),
        original_edge_count=(
            int(torch.triu(valid_edges, diagonal=1).sum())
            if original_edge_count is None
            else int(original_edge_count)
        ),
        candidate_edge_count=int(candidate_edge_count),
        requested_edge_count=int(requested_edge_count),
        actual_node_count=(
            int(hard_node_mask.sum())
            if actual_node_count is None
            else int(actual_node_count)
        ),
        actual_edge_count=(
            int(torch.triu(hard_edge_mask, diagonal=1).sum())
            if actual_edge_count is None
            else int(actual_edge_count)
        ),
        selection_mode=str(selection_mode),
    )


def _rebind_selection(
    decision: HardSelectionOutput,
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    valid_edges: torch.Tensor,
) -> HardSelectionOutput:
    """Attach CPU hard decisions to the differentiable scorer tensors.

    Ranking and graph growth are discrete.  Computing those decisions from one
    detached CPU snapshot avoids hundreds of CUDA scalar synchronizations per
    window, while the straight-through masks below retain the exact gradient
    path to the original node and edge probabilities.
    """

    device = node_probabilities.device
    return _selection_from_masks(
        node_probabilities=node_probabilities,
        edge_probabilities=edge_probabilities,
        valid_edges=valid_edges,
        hard_node_mask=decision.hard_node_mask.to(device=device),
        hard_edge_mask=decision.hard_edge_mask.to(device=device),
        candidate_node_mask=decision.candidate_node_mask.to(device=device),
        requested_node_count=decision.requested_node_count,
        requested_edge_count=decision.requested_edge_count,
        candidate_edge_count=decision.candidate_edge_count,
        selection_mode=decision.selection_mode,
        original_edge_count=decision.original_edge_count,
        actual_node_count=decision.actual_node_count,
        actual_edge_count=decision.actual_edge_count,
    )


def _grow_one(
    seed: int,
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    valid_edges: torch.Tensor,
    target_nodes: int,
    edge_ratio: float,
    edge_minimum: int,
) -> HardSelectionOutput:
    count = int(node_probabilities.numel())
    scores = _edge_scores(node_probabilities, edge_probabilities)
    selected_mask = torch.zeros(
        count, dtype=torch.bool, device=node_probabilities.device
    )
    selected_mask[int(seed)] = True
    selected_count = 1
    tree_edges = []
    growth_scores = scores * (
        0.5 + 0.5 * node_probabilities[None, :]
    )
    while selected_count < int(target_nodes):
        frontier = (
            valid_edges
            & selected_mask[:, None]
            & (~selected_mask)[None, :]
        )
        if not bool(frontier.any()):
            break
        # Row-major argmax reproduces the previous tie-break rule: highest
        # score, then the smallest source node and smallest target node.
        flat_index = int(
            torch.argmax(
                growth_scores.masked_fill(frontier.logical_not(), -float("inf"))
            )
        )
        left, right = divmod(flat_index, count)
        selected_mask[right] = True
        selected_count += 1
        tree_edges.append((min(left, right), max(left, right)))

    candidate_node_mask = selected_mask.clone()
    candidate = (
        valid_edges
        & candidate_node_mask[:, None]
        & candidate_node_mask[None, :]
    )
    edge_indices = torch.nonzero(
        torch.triu(candidate, diagonal=1), as_tuple=False
    )
    candidate_count = int(edge_indices.shape[0])
    requested_edges = 0
    hard_edge_mask = torch.zeros_like(valid_edges)
    if candidate_count and selected_count >= 2:
        requested_edges = min(
            candidate_count,
            max(
                int(edge_minimum),
                len(tree_edges),
                int(math.ceil(float(edge_ratio) * candidate_count)),
            ),
        )
        chosen_pairs = set(tree_edges)
        ranked = sorted(
            (
                (float(scores[int(pair[0]), int(pair[1])]), int(pair[0]), int(pair[1]))
                for pair in edge_indices.tolist()
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        for _, left, right in ranked:
            if len(chosen_pairs) >= requested_edges:
                break
            chosen_pairs.add((left, right))
        for left, right in chosen_pairs:
            hard_edge_mask[left, right] = True
            hard_edge_mask[right, left] = True
    hard_node_mask = hard_edge_mask.any(dim=-1)
    return _selection_from_masks(
        node_probabilities,
        edge_probabilities,
        valid_edges,
        hard_node_mask,
        hard_edge_mask,
        candidate_node_mask,
        selected_count,
        requested_edges,
        candidate_count,
        "learned_k_object",
    )


def _mask_overlap(left: HardSelectionOutput, right: HardSelectionOutput) -> float:
    node_union = left.hard_node_mask | right.hard_node_mask
    edge_union = left.hard_edge_mask | right.hard_edge_mask
    node_value = float(
        (left.hard_node_mask & right.hard_node_mask).sum()
    ) / float(max(1, int(node_union.sum())))
    edge_value = float(
        torch.triu(
            left.hard_edge_mask & right.hard_edge_mask, diagonal=1
        ).sum()
    ) / float(
        max(1, int(torch.triu(edge_union, diagonal=1).sum()))
    )
    return 0.5 * (node_value + edge_value)


def _candidate_score(selection: HardSelectionOutput) -> float:
    nodes = selection.hard_node_mask
    edges = torch.triu(selection.hard_edge_mask, diagonal=1)
    node_score = (
        float(selection.node_probabilities[nodes].mean())
        if bool(nodes.any())
        else 0.0
    )
    edge_score = (
        float(selection.edge_probabilities[edges].mean())
        if bool(edges.any())
        else 0.0
    )
    return 0.5 * (node_score + edge_score)


def _candidate_quality(
    selection: HardSelectionOutput,
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
) -> float:
    nodes = selection.hard_node_mask
    edges = torch.triu(selection.hard_edge_mask, diagonal=1)
    node_score = (
        float(node_probabilities[nodes].mean()) if bool(nodes.any()) else 0.0
    )
    edge_score = (
        float(edge_probabilities[edges].mean()) if bool(edges.any()) else 0.0
    )
    return 0.5 * (node_score + edge_score)


def _overlap_coefficient(
    left: torch.Tensor, right: torch.Tensor
) -> float:
    denominator = min(int(left.sum()), int(right.sum()))
    if denominator < 1:
        return 0.0
    return float((left & right).sum()) / float(denominator)


def _selection_overlaps(
    left: HardSelectionOutput, right: HardSelectionOutput
) -> Tuple[float, float]:
    return (
        _overlap_coefficient(left.hard_node_mask, right.hard_node_mask),
        _overlap_coefficient(
            torch.triu(left.hard_edge_mask, diagonal=1),
            torch.triu(right.hard_edge_mask, diagonal=1),
        ),
    )


def _coordinate_scale(coordinates: torch.Tensor) -> float:
    if coordinates.numel() < 3:
        return 1.0
    finite = torch.isfinite(coordinates).all(dim=-1)
    nonzero = coordinates.abs().sum(dim=-1) > 0.0
    valid = coordinates[finite & nonzero]
    if valid.shape[0] < 2:
        return 1.0
    return max(1.0e-8, float((valid.max(dim=0).values - valid.min(dim=0).values).norm()))


def _normalized_seed_distance(
    coordinates: Optional[torch.Tensor],
    left: int,
    right: int,
    scale: float,
) -> float:
    if coordinates is None:
        return 0.0
    first = coordinates[int(left)]
    second = coordinates[int(right)]
    if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
        return 0.0
    if not bool(first.abs().sum() > 0.0 and second.abs().sum() > 0.0):
        return 0.0
    return float((first - second).norm()) / max(1.0e-8, float(scale))


def _selection_diagnostics(
    selected: Sequence[Tuple[int, HardSelectionOutput]],
    coordinates: Optional[torch.Tensor],
    relaxed: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    count = len(selected)
    node_overlap = torch.zeros((count, count), dtype=torch.float32, device=device)
    edge_overlap = torch.zeros_like(node_overlap)
    seed_distance = torch.zeros_like(node_overlap)
    coordinate_cpu = coordinates.detach().cpu() if coordinates is not None else None
    scale = _coordinate_scale(coordinate_cpu) if coordinate_cpu is not None else 1.0
    union = torch.zeros_like(selected[0][1].hard_node_mask) if selected else None
    unique = []
    total_membership = 0
    for index, (_, selection) in enumerate(selected):
        nodes = selection.hard_node_mask
        unique.append(
            1.0
            if union is None or not bool(union.any())
            else float((nodes & ~union).sum()) / float(max(1, int(nodes.sum())))
        )
        if union is not None:
            union |= nodes
        total_membership += int(nodes.sum())
        for other in range(index):
            node_value, edge_value = _selection_overlaps(
                selection, selected[other][1]
            )
            node_overlap[index, other] = node_overlap[other, index] = node_value
            edge_overlap[index, other] = edge_overlap[other, index] = edge_value
            distance = _normalized_seed_distance(
                coordinate_cpu,
                selected[index][0],
                selected[other][0],
                scale,
            )
            seed_distance[index, other] = seed_distance[other, index] = distance
    union_efficiency = (
        float(union.sum()) / float(max(1, total_membership))
        if union is not None
        else 0.0
    )
    return (
        node_overlap,
        edge_overlap,
        torch.tensor(unique, dtype=torch.float32, device=device),
        seed_distance,
        union_efficiency,
        bool(relaxed),
    )


def _diverse_seed_order(
    node_probabilities: torch.Tensor,
    communities: torch.Tensor,
    coordinates: torch.Tensor,
    selected: Sequence[Tuple[int, HardSelectionOutput]],
    minimum_distance: float,
) -> Tuple[Tuple[int, ...], bool]:
    base = _seed_order(node_probabilities, communities)
    if not selected:
        return base, False
    used_seeds = {item[0] for item in selected}
    used_communities = {int(communities[item[0]]) for item in selected}
    union = torch.zeros_like(selected[0][1].hard_node_mask)
    for _, item in selected:
        union |= item.hard_node_mask
    scale = _coordinate_scale(coordinates)
    remaining = [
        index for index in base
        if index not in used_seeds and not bool(union[index])
    ]
    for factor in (1.0, 0.8, 0.6, 0.0):
        threshold = float(minimum_distance) * factor
        eligible = [
            index for index in remaining
            if all(
                _normalized_seed_distance(
                    coordinates, index, prior[0], scale
                ) >= threshold
                for prior in selected
            )
        ]
        if eligible:
            eligible.sort(
                key=lambda index: (
                    int(int(communities[index]) in used_communities),
                    -float(node_probabilities[index]),
                    index,
                )
            )
            return tuple(eligible), factor < 1.0
    return (), True


def _select_diverse_fixed_k(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    communities: torch.Tensor,
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    subgraph_count: int,
    candidate_multiplier: int,
    per_object_node_ratio: float,
    edge_ratio: float,
    node_minimum: int,
    edge_minimum: int,
    node_reuse_decay: float,
    edge_reuse_decay: float,
    max_node_overlap: float,
    max_edge_overlap: float,
    min_unique_node_fraction: float,
    quality_floor_ratio: float,
    min_seed_distance: float,
) -> Tuple[List[Tuple[int, HardSelectionOutput]], bool]:
    count = int(node_probabilities.numel())
    target_nodes = min(
        count,
        max(int(node_minimum), int(math.ceil(float(per_object_node_ratio) * count))),
    )
    node_use = torch.zeros(count, dtype=node_probabilities.dtype)
    edge_use = torch.zeros_like(edge_probabilities)
    selected: List[Tuple[int, HardSelectionOutput]] = []
    constraints_relaxed = False
    pool_limit = max(int(subgraph_count), int(subgraph_count) * int(candidate_multiplier))
    for _ in range(int(subgraph_count)):
        residual_nodes = node_probabilities * torch.pow(
            node_probabilities.new_full((), float(node_reuse_decay)), node_use
        )
        residual_edges = edge_probabilities * torch.pow(
            edge_probabilities.new_full((), float(edge_reuse_decay)), edge_use
        )
        seed_order, distance_relaxed = _diverse_seed_order(
            residual_nodes,
            communities,
            coordinates,
            selected,
            min_seed_distance,
        )
        constraints_relaxed = constraints_relaxed or distance_relaxed
        candidates = []
        signatures = set()
        for seed in seed_order[:pool_limit]:
            candidate = _grow_one(
                seed,
                residual_nodes,
                residual_edges,
                valid,
                target_nodes,
                edge_ratio,
                edge_minimum,
            )
            if (
                candidate.actual_node_count < node_minimum
                or candidate.actual_edge_count < edge_minimum
            ):
                continue
            signature = (
                tuple(torch.nonzero(candidate.hard_node_mask, as_tuple=False).flatten().tolist()),
                tuple(
                    tuple(pair)
                    for pair in torch.nonzero(
                        torch.triu(candidate.hard_edge_mask, diagonal=1),
                        as_tuple=False,
                    ).tolist()
                ),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            quality = _candidate_quality(
                candidate, node_probabilities, edge_probabilities
            )
            node_values = []
            edge_values = []
            for _, prior in selected:
                node_value, edge_value = _selection_overlaps(candidate, prior)
                node_values.append(node_value)
                edge_values.append(edge_value)
            union = torch.zeros_like(candidate.hard_node_mask)
            for _, prior in selected:
                union |= prior.hard_node_mask
            unique_fraction = (
                1.0
                if not selected
                else float((candidate.hard_node_mask & ~union).sum())
                / float(max(1, int(candidate.hard_node_mask.sum())))
            )
            candidates.append(
                (
                    int(seed),
                    candidate,
                    quality,
                    max(node_values) if node_values else 0.0,
                    max(edge_values) if edge_values else 0.0,
                    unique_fraction,
                )
            )
        if not candidates:
            raise ValueError("diverse fixed-K selection cannot build a real candidate")
        best_quality = max(item[2] for item in candidates)
        quality_floor = float(quality_floor_ratio) * best_quality
        feasible = [
            item for item in candidates
            if item[2] >= quality_floor
            and item[3] <= float(max_node_overlap)
            and item[4] <= float(max_edge_overlap)
            and item[5] >= float(min_unique_node_fraction)
        ]
        if feasible:
            chosen = max(
                feasible,
                key=lambda item: (item[2], item[5], -item[3], -item[4], -item[0]),
            )
        else:
            constraints_relaxed = True
            chosen = min(
                candidates,
                key=lambda item: (
                    max(0.0, quality_floor - item[2]),
                    max(0.0, item[3] - float(max_node_overlap)),
                    max(0.0, item[4] - float(max_edge_overlap)),
                    max(0.0, float(min_unique_node_fraction) - item[5]),
                    -item[2],
                    item[0],
                ),
            )
        selected.append((chosen[0], chosen[1]))
        node_use += chosen[1].hard_node_mask.to(node_use.dtype)
        edge_use += chosen[1].hard_edge_mask.to(edge_use.dtype)
    return selected, constraints_relaxed


def select_fixed_k_subgraphs(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    communities: torch.Tensor,
    edge_presence_mask: torch.Tensor,
    subgraph_count: int = 5,
    candidate_multiplier: int = 2,
    total_node_ratio: float = 0.50,
    edge_ratio: float = 0.30,
    node_minimum: int = 2,
    edge_minimum: int = 1,
    overlap_penalty: float = 0.25,
    coordinates: Optional[torch.Tensor] = None,
    diversity_enabled: bool = False,
    per_object_node_ratio: float = 0.10,
    node_reuse_decay: float = 0.25,
    edge_reuse_decay: float = 0.10,
    max_node_overlap: float = 0.40,
    max_edge_overlap: float = 0.25,
    min_unique_node_fraction: float = 0.50,
    quality_floor_ratio: float = 0.80,
    min_seed_distance: float = 0.15,
) -> FixedKSelectionOutput:
    """Return a hard union plus exactly K connected, overlap-tolerant objects.

    A valid window is never silently padded with an empty slot.  On a
    degenerate toy graph with fewer than K distinct connected candidates, the
    best real candidate is repeated deterministically; ordinary brain graphs
    use distinct candidates and may still overlap where structurally useful.
    """

    if subgraph_count < 1 or candidate_multiplier < 1:
        raise ValueError("fixed-K selector sizes must be positive")
    count = int(node_probabilities.numel())
    if tuple(communities.shape) != (count,):
        raise ValueError("fixed-K communities do not align with nodes")
    if tuple(edge_probabilities.shape) != (count, count):
        raise ValueError("fixed-K edge probabilities must be square")
    if not 0.0 < total_node_ratio <= 1.0 or not 0.0 < edge_ratio <= 1.0:
        raise ValueError("fixed-K selection ratios must lie in (0,1]")
    if diversity_enabled:
        if coordinates is None or tuple(coordinates.shape) != (count, 3):
            raise ValueError(
                "diverse fixed-K selection requires aligned 3-D coordinates"
            )
        ratios = (
            per_object_node_ratio,
            node_reuse_decay,
            edge_reuse_decay,
            max_node_overlap,
            max_edge_overlap,
            min_unique_node_fraction,
            quality_floor_ratio,
        )
        if any(float(value) <= 0.0 or float(value) > 1.0 for value in ratios):
            raise ValueError("diverse fixed-K ratios must lie in (0,1]")
        if min_seed_distance < 0.0:
            raise ValueError("minimum seed distance cannot be negative")
    # Hardening is intentionally non-differentiable.  Take one bulk snapshot
    # per window instead of repeatedly converting CUDA scalars inside Python
    # sorting/growth loops.  The selected masks are rebound to the live scorer
    # tensors at the end so the straight-through estimator is unchanged.
    valid = _valid_edges(edge_presence_mask).to(node_probabilities.device)
    decision_nodes = node_probabilities.detach().to(device="cpu")
    decision_edges = edge_probabilities.detach().to(device="cpu")
    decision_communities = communities.detach().to(device="cpu")
    decision_valid = valid.detach().to(device="cpu")
    decision_coordinates = (
        coordinates.detach().to(device="cpu")
        if coordinates is not None
        else None
    )
    total_target = min(
        count,
        max(
            int(node_minimum) * int(subgraph_count),
            int(math.ceil(float(total_node_ratio) * count)),
        ),
    )
    per_object_target = min(
        count,
        max(
            int(node_minimum),
            int(math.ceil(float(total_target) / float(subgraph_count))),
        ),
    )
    diversity_constraint_relaxed = False
    if diversity_enabled:
        selected, diversity_constraint_relaxed = _select_diverse_fixed_k(
            decision_nodes,
            decision_edges,
            decision_communities,
            decision_coordinates,
            decision_valid,
            subgraph_count,
            candidate_multiplier,
            per_object_node_ratio,
            edge_ratio,
            node_minimum,
            edge_minimum,
            node_reuse_decay,
            edge_reuse_decay,
            max_node_overlap,
            max_edge_overlap,
            min_unique_node_fraction,
            quality_floor_ratio,
            min_seed_distance,
        )
    else:
        seeds = _seed_order(decision_nodes, decision_communities)
        pool_limit = min(len(seeds), int(subgraph_count) * int(candidate_multiplier))
        candidates = []
        candidate_signatures = set()
        # The configured pool is the normal fast path.  Remaining seeds are only
        # visited when invalid/duplicate candidates would otherwise violate the
        # exact-K contract.
        for seed_index, seed in enumerate(seeds):
            if seed_index >= pool_limit and len(candidates) >= int(subgraph_count):
                break
            selection = _grow_one(
                seed,
                decision_nodes,
                decision_edges,
                decision_valid,
                per_object_target,
                edge_ratio,
                edge_minimum,
            )
            if (
                selection.actual_node_count >= node_minimum
                and selection.actual_edge_count >= edge_minimum
            ):
                signature = (
                    tuple(
                        torch.nonzero(
                            selection.hard_node_mask, as_tuple=False
                        ).flatten().detach().cpu().tolist()
                    ),
                    tuple(
                        tuple(pair)
                        for pair in torch.nonzero(
                            torch.triu(selection.hard_edge_mask, diagonal=1),
                            as_tuple=False,
                        ).detach().cpu().tolist()
                    ),
                )
                if signature not in candidate_signatures:
                    candidate_signatures.add(signature)
                    candidates.append((int(seed), selection))

        if not candidates:
            raise ValueError(
                "valid window cannot supply a non-empty connected hard subgraph"
            )

        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < int(subgraph_count):
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    _candidate_score(remaining[index][1])
                    - float(overlap_penalty)
                    * sum(
                        _mask_overlap(remaining[index][1], prior[1])
                        for prior in selected
                    ),
                    -remaining[index][0],
                ),
            )
            selected.append(remaining.pop(best_index))

        # The legacy profile retains its exact-K fallback for reproducibility.
        if len(selected) < int(subgraph_count):
            ranked_fallback = sorted(
                candidates,
                key=lambda item: (-_candidate_score(item[1]), item[0]),
            )
            cursor = 0
            while len(selected) < int(subgraph_count):
                selected.append(ranked_fallback[cursor % len(ranked_fallback)])
                cursor += 1

    union_node = torch.zeros(
        count, dtype=torch.bool, device=decision_nodes.device
    )
    union_edge = torch.zeros_like(decision_valid)
    union_candidate = torch.zeros_like(union_node)
    for _, item in selected:
        union_node |= item.hard_node_mask
        union_edge |= item.hard_edge_mask
        union_candidate |= item.candidate_node_mask
    union_candidate_edges = (
        decision_valid & union_candidate[:, None] & union_candidate[None, :]
    )
    union_decision = _selection_from_masks(
        decision_nodes,
        decision_edges,
        decision_valid,
        union_node,
        union_edge,
        union_candidate,
        int(union_candidate.sum()),
        int(torch.triu(union_edge, diagonal=1).sum()),
        int(torch.triu(union_candidate_edges, diagonal=1).sum()),
        "learned_k_union",
    )
    union = _rebind_selection(
        union_decision, node_probabilities, edge_probabilities, valid
    )
    rebound_subgraphs = tuple(
        _rebind_selection(item[1], node_probabilities, edge_probabilities, valid)
        for item in selected
    )
    mask = torch.zeros(
        int(subgraph_count), dtype=torch.bool, device=node_probabilities.device
    )
    mask[:] = True
    seed_tensor = torch.full(
        (int(subgraph_count),),
        -1,
        dtype=torch.long,
        device=node_probabilities.device,
    )
    if selected:
        seed_tensor[: len(selected)] = torch.tensor(
            [item[0] for item in selected],
            dtype=torch.long,
            device=node_probabilities.device,
        )
    diagnostics = _selection_diagnostics(
        selected,
        decision_coordinates,
        diversity_constraint_relaxed,
        node_probabilities.device,
    )
    return FixedKSelectionOutput(
        union=union,
        subgraphs=rebound_subgraphs,
        subgraph_mask=mask,
        seed_indices=seed_tensor,
        pairwise_node_overlap=diagnostics[0],
        pairwise_edge_overlap=diagnostics[1],
        unique_node_fractions=diagnostics[2],
        seed_distance_matrix=diagnostics[3],
        union_efficiency=diagnostics[4],
        diversity_constraint_relaxed=diagnostics[5],
    )


def select_object_conditioned_subgraphs(
    global_node_probabilities: torch.Tensor,
    global_edge_probabilities: torch.Tensor,
    object_node_probabilities: torch.Tensor,
    object_edge_probabilities: torch.Tensor,
    edge_presence_mask: torch.Tensor,
    per_object_node_ratio: float = 0.10,
    edge_ratio: float = 0.30,
    node_minimum: int = 2,
    edge_minimum: int = 1,
    candidate_multiplier: int = 4,
    overlap_penalty: float = 0.25,
    max_node_overlap: float = 0.40,
    max_edge_overlap: float = 0.30,
) -> FixedKSelectionOutput:
    """Harden K independently learned object fields and return their union.

    This is intentionally different from :func:`select_fixed_k_subgraphs`:
    each object owns a differentiable node/edge field.  The discrete candidate
    search is only a guardrail for pathological overlap, not the mechanism
    that creates object diversity.
    """

    if object_node_probabilities.ndim != 2:
        raise ValueError("object node probabilities must have shape [K,N]")
    object_count, count = object_node_probabilities.shape
    if object_count < 2 or count < 2:
        raise ValueError("object-conditioned hardening requires K>=2 and N>=2")
    if tuple(global_node_probabilities.shape) != (count,):
        raise ValueError("global node probabilities do not align")
    if tuple(global_edge_probabilities.shape) != (count, count):
        raise ValueError("global edge probabilities do not align")
    if tuple(object_edge_probabilities.shape) != (
        object_count, count, count
    ):
        raise ValueError("object edge probabilities must have shape [K,N,N]")
    if not 0.0 < per_object_node_ratio <= 1.0:
        raise ValueError("object node ratio must lie in (0,1]")
    if not 0.0 < edge_ratio <= 1.0:
        raise ValueError("object edge ratio must lie in (0,1]")
    if candidate_multiplier < 1:
        raise ValueError("object candidate multiplier must be positive")
    if not 0.0 <= max_node_overlap <= 1.0:
        raise ValueError("maximum node overlap must lie in [0,1]")
    if not 0.0 <= max_edge_overlap <= 1.0:
        raise ValueError("maximum edge overlap must lie in [0,1]")

    device = global_node_probabilities.device
    valid = _valid_edges(edge_presence_mask).to(device)
    # One fused snapshot is materially faster than five independent .cpu()
    # calls on CUDA because it creates one synchronization point per window.
    # Slicing the detached CPU buffer preserves exactly the same decisions.
    snapshot_dtype = global_node_probabilities.dtype
    lengths = (
        count * count,
        count,
        count * count,
        int(object_count) * count,
        int(object_count) * count * count,
    )
    snapshot = torch.cat(
        (
            valid.to(snapshot_dtype).reshape(-1),
            global_node_probabilities.detach().reshape(-1),
            global_edge_probabilities.detach().reshape(-1),
            object_node_probabilities.detach().reshape(-1),
            object_edge_probabilities.detach().reshape(-1),
        )
    ).detach().cpu()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    decision_valid = snapshot[
        offsets[0] : offsets[1]
    ].reshape(count, count).to(torch.bool)
    decision_global_nodes = snapshot[
        offsets[1] : offsets[2]
    ].reshape(count)
    decision_global_edges = snapshot[
        offsets[2] : offsets[3]
    ].reshape(count, count)
    decision_object_nodes = snapshot[
        offsets[3] : offsets[4]
    ].reshape(object_count, count)
    decision_object_edges = snapshot[
        offsets[4] : offsets[5]
    ].reshape(object_count, count, count)
    target_nodes = min(
        int(count),
        max(int(node_minimum), int(math.ceil(per_object_node_ratio * count))),
    )
    selected: List[Tuple[int, HardSelectionOutput]] = []
    relaxed = False
    for object_index in range(int(object_count)):
        node_scores = decision_object_nodes[object_index]
        edge_scores = decision_object_edges[object_index]
        seeds = torch.argsort(node_scores, descending=True).tolist()
        candidates = []
        signatures = set()
        pool_limit = min(
            len(seeds), max(int(object_count), int(candidate_multiplier))
        )
        def consider_seed(seed):
            candidate = _grow_one(
                int(seed),
                node_scores,
                edge_scores,
                decision_valid,
                target_nodes,
                edge_ratio,
                edge_minimum,
            )
            if (
                candidate.actual_node_count < int(node_minimum)
                or candidate.actual_edge_count < int(edge_minimum)
            ):
                return
            signature = (
                tuple(torch.nonzero(candidate.hard_node_mask).flatten().tolist()),
                tuple(
                    tuple(pair)
                    for pair in torch.nonzero(
                        torch.triu(candidate.hard_edge_mask, diagonal=1)
                    ).tolist()
                ),
            )
            if signature in signatures:
                return
            signatures.add(signature)
            node_overlap = 0.0
            edge_overlap = 0.0
            for _, prior in selected:
                current_node, current_edge = _selection_overlaps(candidate, prior)
                node_overlap = max(node_overlap, current_node)
                edge_overlap = max(edge_overlap, current_edge)
            quality = _candidate_quality(candidate, node_scores, edge_scores)
            violation = (
                max(0.0, node_overlap - float(max_node_overlap))
                + max(0.0, edge_overlap - float(max_edge_overlap))
            )
            objective = quality - float(overlap_penalty) * (
                node_overlap + edge_overlap
            )
            candidates.append(
                (violation, -objective, int(seed), candidate)
            )
        for seed in seeds[:pool_limit]:
            consider_seed(seed)
        # Sparse or disconnected windows can place all preferred seeds on
        # isolated nodes.  Only in that failure case, scan the deterministic
        # remainder so the object still resolves to a valid connected graph.
        if not candidates:
            for seed in seeds[pool_limit:]:
                consider_seed(seed)
                if candidates:
                    break
        if not candidates:
            raise ValueError("object field cannot supply a connected hard subgraph")
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = candidates[0]
        relaxed = relaxed or chosen[0] > 0.0
        selected.append((chosen[2], chosen[3]))

    union_node = torch.zeros(count, dtype=torch.bool)
    union_edge = torch.zeros_like(decision_valid)
    union_candidate = torch.zeros_like(union_node)
    for _, item in selected:
        union_node |= item.hard_node_mask
        union_edge |= item.hard_edge_mask
        union_candidate |= item.candidate_node_mask
    union_candidate_edges = (
        decision_valid & union_candidate[:, None] & union_candidate[None, :]
    )
    union_decision = _selection_from_masks(
        decision_global_nodes,
        decision_global_edges,
        decision_valid,
        union_node,
        union_edge,
        union_candidate,
        int(union_candidate.sum()),
        int(torch.triu(union_edge, diagonal=1).sum()),
        int(torch.triu(union_candidate_edges, diagonal=1).sum()),
        "learned_multi_object_union",
    )
    # The hard union is the forward decision, while its backward path follows
    # the differentiable union of the K object fields.  Consequently the task
    # loss reaches both the global scorer and every object query/decoder.
    object_node_union = 1.0 - torch.prod(
        1.0 - object_node_probabilities, dim=0
    )
    object_edge_union = 1.0 - torch.prod(
        1.0 - object_edge_probabilities, dim=0
    )
    differentiable_union_nodes = (
        global_node_probabilities * object_node_union
    )
    differentiable_union_edges = (
        global_edge_probabilities * object_edge_union
    )
    union = _rebind_selection(
        union_decision,
        differentiable_union_nodes,
        differentiable_union_edges,
        valid,
    )
    rebound = tuple(
        _rebind_selection(
            item,
            object_node_probabilities[index],
            object_edge_probabilities[index],
            valid,
        )
        for index, (_, item) in enumerate(selected)
    )
    diagnostics = _selection_diagnostics(
        selected, None, relaxed, device
    )
    return FixedKSelectionOutput(
        union=union,
        subgraphs=rebound,
        subgraph_mask=torch.ones(object_count, dtype=torch.bool, device=device),
        seed_indices=torch.tensor(
            [item[0] for item in selected], dtype=torch.long, device=device
        ),
        pairwise_node_overlap=diagnostics[0],
        pairwise_edge_overlap=diagnostics[1],
        unique_node_fractions=diagnostics[2],
        seed_distance_matrix=diagnostics[3],
        union_efficiency=diagnostics[4],
        diversity_constraint_relaxed=diagnostics[5],
    )
