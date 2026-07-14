"""Community-seeded hard extraction from a frozen soft_graph model."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.features.graph_features import GraphTimepointFeatures
from keysubgraph.models.soft_extractor import SoftGraphClassifier, TimepointSelection


@dataclass(frozen=True)
class HardExtractionConfig:
    seeds_per_community: int = 1
    neighborhood_hops: int = 1
    max_nodes: int = 20
    max_edges: int = 40
    min_nodes: int = 2
    min_edges: int = 1
    top_k: int = 5
    overlap_threshold: float = 1.0
    node_score_weight: float = 0.35
    edge_score_weight: float = 0.35
    connectivity_weight: float = 0.20
    dynamic_weight: float = 0.10
    local_confidence_weight: float = 0.0
    use_local_confidence_score: bool = False
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        positive = (
            self.seeds_per_community,
            self.max_nodes,
            self.max_edges,
            self.min_nodes,
            self.min_edges,
            self.top_k,
        )
        if any(value < 1 for value in positive) or self.neighborhood_hops < 0:
            raise ValueError("hard extraction sizes are invalid")
        if self.min_nodes > self.max_nodes or self.min_edges > self.max_edges:
            raise ValueError("minimum candidate sizes exceed maximum sizes")
        if not 0.0 <= self.overlap_threshold <= 2.0:
            raise ValueError("overlap_threshold must be in [0, 2]")
        weights = (
            self.node_score_weight,
            self.edge_score_weight,
            self.connectivity_weight,
            self.dynamic_weight,
            self.local_confidence_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("candidate score weights must be non-negative")
        if self.use_local_confidence_score or self.local_confidence_weight != 0.0:
            raise ValueError(
                "local candidate head is disabled in the verified soft_graph baseline"
            )
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class HardSubgraphCandidate:
    seed_node: int
    node_ids: Tuple[int, ...]
    node_names: Tuple[str, ...]
    edge_index: Tuple[Tuple[int, int], ...]
    original_edge_weights: Tuple[float, ...]
    node_scores: Tuple[float, ...]
    edge_scores: Tuple[float, ...]
    community_labels: Tuple[int, ...]
    delta_degree: Tuple[float, ...]
    delta_degree_mask: Tuple[bool, ...]
    delta_edge_weight: Tuple[float, ...]
    delta_edge_mask: Tuple[bool, ...]
    score_node: float
    score_edge: float
    score_connectivity: float
    score_dynamic: float
    score_local_confidence: float
    candidate_score: float

    def edge_set(self) -> Set[Tuple[int, int]]:
        return set(self.edge_index)

    def node_set(self) -> Set[int]:
        return set(self.node_ids)


@dataclass(frozen=True)
class HardTimepointResult:
    time_index: int
    original_node_count: int
    original_edge_count: int
    candidate_pool: Tuple[HardSubgraphCandidate, ...]
    selected_subgraphs: Tuple[HardSubgraphCandidate, ...]
    num_valid_subgraphs: int
    subgraph_mask: Tuple[bool, ...]


@dataclass(frozen=True)
class HardSampleResult:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    split: str
    relative_path: str
    edge_presence_threshold: float
    timepoints: Tuple[HardTimepointResult, ...]


def candidate_overlap(
    left: HardSubgraphCandidate,
    right: HardSubgraphCandidate,
    epsilon: float = 1e-8,
) -> float:
    left_nodes, right_nodes = left.node_set(), right.node_set()
    left_edges, right_edges = left.edge_set(), right.edge_set()
    node_union = left_nodes | right_nodes
    edge_union = left_edges | right_edges
    node_term = len(left_nodes & right_nodes) / (len(node_union) + epsilon)
    edge_term = len(left_edges & right_edges) / (len(edge_union) + epsilon)
    return node_term + edge_term


def _largest_connected_component_size(
    node_ids: Sequence[int], edges: Sequence[Tuple[int, int]]
) -> int:
    adjacency = {node: set() for node in node_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    largest = 0
    unseen = set(node_ids)
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        largest = max(largest, size)
    return largest


class HardSubgraphExtractor:
    """Non-differentiable export path; construction freezes the model."""

    def __init__(
        self,
        model: SoftGraphClassifier,
        config: Optional[HardExtractionConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or HardExtractionConfig()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _community_seeds(
        self, communities: torch.Tensor, node_scores: torch.Tensor
    ) -> List[int]:
        seeds = []
        for community_id in sorted(int(item) for item in torch.unique(communities).tolist()):
            members = torch.nonzero(communities == community_id, as_tuple=False).flatten().tolist()
            members.sort(key=lambda node: (-float(node_scores[node]), node))
            seeds.extend(members[: self.config.seeds_per_community])
        return seeds

    def _hop_nodes(self, seed: int, edge_mask: torch.Tensor) -> Set[int]:
        visited = {seed}
        frontier = {seed}
        for _ in range(self.config.neighborhood_hops):
            next_frontier = set()
            for node in frontier:
                neighbors = torch.nonzero(edge_mask[node], as_tuple=False).flatten().tolist()
                next_frontier.update(int(item) for item in neighbors)
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def _candidate(
        self,
        sample: GraphSequenceSample,
        time_index: int,
        seed: int,
        features: GraphTimepointFeatures,
        selection: TimepointSelection,
    ) -> Optional[HardSubgraphCandidate]:
        expanded = self._hop_nodes(seed, features.edge_mask)
        ranked_nodes = sorted(
            expanded, key=lambda node: (-float(selection.node_scores[node]), node)
        )[: self.config.max_nodes]
        node_ids = tuple(sorted(ranked_nodes))
        if len(node_ids) < self.config.min_nodes:
            return None
        node_set = set(node_ids)
        possible_edges = []
        for left in node_ids:
            for right in node_ids:
                if left < right and right in node_set and bool(features.edge_mask[left, right]):
                    possible_edges.append((left, right))
        possible_edges.sort(
            key=lambda edge: (
                -float(selection.edge_scores[edge[0], edge[1]]),
                edge[0],
                edge[1],
            )
        )
        edges = tuple(possible_edges[: self.config.max_edges])
        if len(edges) < self.config.min_edges:
            return None

        node_score = sum(float(selection.node_scores[node]) for node in node_ids) / len(node_ids)
        edge_score = sum(
            float(selection.edge_scores[left, right]) for left, right in edges
        ) / len(edges)
        connectivity = _largest_connected_component_size(node_ids, edges) / len(node_ids)
        valid_dynamic_nodes = [
            node for node in node_ids if bool(features.delta_degree_mask[node])
        ]
        dynamic = (
            sum(float(features.delta_degree[node].abs()) for node in valid_dynamic_nodes)
            / len(valid_dynamic_nodes)
            if valid_dynamic_nodes
            else 0.0
        )
        total = (
            self.config.node_score_weight * node_score
            + self.config.edge_score_weight * edge_score
            + self.config.connectivity_weight * connectivity
            + self.config.dynamic_weight * dynamic
        )
        adjacency = sample.adjacency[time_index]
        communities = sample.communities[time_index]
        return HardSubgraphCandidate(
            seed_node=seed,
            node_ids=node_ids,
            node_names=tuple(sample.node_names[time_index][node] for node in node_ids),
            edge_index=edges,
            original_edge_weights=tuple(float(adjacency[left, right]) for left, right in edges),
            node_scores=tuple(float(selection.node_scores[node]) for node in node_ids),
            edge_scores=tuple(float(selection.edge_scores[left, right]) for left, right in edges),
            community_labels=tuple(int(communities[node]) for node in node_ids),
            delta_degree=tuple(float(features.delta_degree[node]) for node in node_ids),
            delta_degree_mask=tuple(bool(features.delta_degree_mask[node]) for node in node_ids),
            delta_edge_weight=tuple(float(features.delta_edge_weight[left, right]) for left, right in edges),
            delta_edge_mask=tuple(bool(features.delta_edge_mask[left, right]) for left, right in edges),
            score_node=node_score,
            score_edge=edge_score,
            score_connectivity=connectivity,
            score_dynamic=dynamic,
            score_local_confidence=0.0,
            candidate_score=total,
        )

    def _deduplicate(
        self, candidates: Sequence[HardSubgraphCandidate]
    ) -> List[HardSubgraphCandidate]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (-candidate.candidate_score, candidate.seed_node),
        )
        kept = []
        for candidate in ordered:
            if all(
                candidate_overlap(candidate, existing, self.config.epsilon)
                <= self.config.overlap_threshold
                for existing in kept
            ):
                kept.append(candidate)
        return kept

    def extract_sample(self, sample: GraphSequenceSample) -> HardSampleResult:
        timepoints = []
        with torch.no_grad():
            for time_index in range(sample.num_timepoints):
                features, selection = self.model.score_timepoint(sample, time_index)
                seeds = self._community_seeds(
                    sample.communities[time_index], selection.node_scores
                )
                candidates = []
                for seed in seeds:
                    candidate = self._candidate(
                        sample, time_index, seed, features, selection
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                deduplicated = self._deduplicate(candidates)
                selected = tuple(deduplicated[: self.config.top_k])
                edge_count = int(torch.triu(features.edge_mask, diagonal=1).sum())
                timepoints.append(
                    HardTimepointResult(
                        time_index=time_index,
                        original_node_count=int(features.node_features.shape[0]),
                        original_edge_count=edge_count,
                        candidate_pool=tuple(candidates),
                        selected_subgraphs=selected,
                        num_valid_subgraphs=len(selected),
                        subgraph_mask=tuple(
                            index < len(selected) for index in range(self.config.top_k)
                        ),
                    )
                )
        return HardSampleResult(
            sample_key=sample.sample_key,
            sample_id=sample.sample_id,
            site=sample.site,
            subject_id=sample.subject_id,
            session_id=sample.session_id,
            label=sample.label,
            split=sample.split,
            relative_path=sample.relative_path,
            edge_presence_threshold=sample.edge_presence_threshold,
            timepoints=tuple(timepoints),
        )


def _candidate_dict(
    candidate: HardSubgraphCandidate,
    result: HardSampleResult,
    timepoint: HardTimepointResult,
    subgraph_index: Optional[int],
) -> Dict[str, Any]:
    payload = asdict(candidate)
    payload.update(
        {
            "sample_id": result.sample_id,
            "site": result.site,
            "label": result.label,
            "split": result.split,
            "fold": None,
            "time_index": timepoint.time_index,
            "subgraph_index": subgraph_index,
            "edge_presence_threshold": result.edge_presence_threshold,
            "time_mask": True,
            "node_mask": [True] * len(candidate.node_ids),
            "subgraph_mask": subgraph_index is not None,
            "num_valid_subgraphs": timepoint.num_valid_subgraphs,
            "original_graph_ref": result.relative_path,
            "candidate_pool_ref": "{}#time={}".format(
                result.sample_key, timepoint.time_index
            ),
        }
    )
    return payload


def export_hard_sample(
    result: HardSampleResult,
    output_path: Path,
    config: HardExtractionConfig,
    checkpoint_path: Path,
    data_protocol_sha256: str,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("hard-subgraph export already exists")
    payload = {
        "schema_version": 1,
        "sample_key": result.sample_key,
        "sample_id": result.sample_id,
        "site": result.site,
        "subject_id": result.subject_id,
        "session_id": result.session_id,
        "label": result.label,
        "split": result.split,
        "relative_path": result.relative_path,
        "edge_presence_threshold": result.edge_presence_threshold,
        "hard_extraction_config": asdict(config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "data_protocol_sha256": data_protocol_sha256,
        "timepoints": [],
    }
    for timepoint in result.timepoints:
        payload["timepoints"].append(
            {
                "time_index": timepoint.time_index,
                "time_mask": True,
                "original_node_count": timepoint.original_node_count,
                "original_edge_count": timepoint.original_edge_count,
                "num_valid_subgraphs": timepoint.num_valid_subgraphs,
                "subgraph_mask": list(timepoint.subgraph_mask),
                "candidate_pool": [
                    _candidate_dict(candidate, result, timepoint, None)
                    for candidate in timepoint.candidate_pool
                ],
                "subgraphs": [
                    _candidate_dict(candidate, result, timepoint, index)
                    for index, candidate in enumerate(timepoint.selected_subgraphs)
                ],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path
