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
from keysubgraph.models.tg_sgw_types import TGSGWTheoryConfig
from keysubgraph.theory import (
    CandidateScoreStandardizer,
    EvolutionRepresentation,
    FidelityResult,
    HardExportFidelityEvaluator,
    HardUnionGraph,
    HardUnionGraphBuilder,
    SpectralGWEvolutionEncoder,
    SpectralGWGreedyExporter,
)


@dataclass(frozen=True)
class HardExtractionConfig:
    seeds_per_community: int = 1
    neighborhood_hops: int = 1
    max_nodes: int = 20
    max_edges: int = 80
    min_nodes: int = 2
    min_edges: int = 1
    top_k: int = 4
    overlap_threshold: float = 0.60
    node_score_weight: float = 0.35
    edge_score_weight: float = 0.35
    connectivity_weight: float = 0.20
    dynamic_weight: float = 0.10
    local_confidence_weight: float = 0.0
    use_local_confidence_score: bool = False
    strategy: str = "spectral_gw_greedy"
    beta_lambda: float = 0.20
    beta_gw: float = 0.10
    beta_overlap: float = 0.10
    beta_size: float = 0.05
    min_export_gain: float = 0.0
    prefilter_discriminative_top_r1: int = 32
    prefilter_spectral_top_r2: int = 8
    max_union_nodes: Optional[int] = None
    max_union_edges: Optional[int] = None
    eval_gw_entropic_reg: float = 0.01
    eval_gw_max_iter: int = 100
    eval_gw_sinkhorn_iter: int = 100
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        positive = (
            self.seeds_per_community,
            self.max_nodes,
            self.max_edges,
            self.min_nodes,
            self.min_edges,
            self.top_k,
            self.prefilter_discriminative_top_r1,
            self.prefilter_spectral_top_r2,
        )
        if any(value < 1 for value in positive) or self.neighborhood_hops < 0:
            raise ValueError("hard extraction sizes are invalid")
        if self.min_nodes > self.max_nodes or self.min_edges > self.max_edges:
            raise ValueError("minimum candidate sizes exceed maximum sizes")
        if not (
            self.top_k
            <= self.prefilter_spectral_top_r2
            <= self.prefilter_discriminative_top_r1
        ):
            raise ValueError("hard export requires K <= R2 <= R1")
        if self.max_union_nodes is not None and self.max_union_nodes < 1:
            raise ValueError("max_union_nodes must be positive")
        if self.max_union_edges is not None and self.max_union_edges < 1:
            raise ValueError("max_union_edges must be positive")
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
        if self.strategy != "spectral_gw_greedy":
            raise ValueError("strong theory export requires spectral_gw_greedy")
        if any(
            value < 0.0
            for value in (self.beta_lambda, self.beta_gw, self.beta_overlap, self.beta_size)
        ):
            raise ValueError("spectral/GW/overlap penalties must be non-negative")
        if self.eval_gw_entropic_reg <= 0.0:
            raise ValueError("evaluation GW regularization must be positive")
        if self.eval_gw_max_iter < 1 or self.eval_gw_sinkhorn_iter < 1:
            raise ValueError("evaluation GW iterations must be positive")
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
class HardCandidatePoolResult:
    candidates: Tuple[HardSubgraphCandidate, ...]
    window_valid: bool
    rejected_invalid: int
    removed_duplicates: int


class HardCandidatePoolBuilder(object):
    """Validate and deterministically deduplicate frozen hard candidates."""

    def __init__(
        self,
        min_nodes: int,
        min_edges: int,
        max_nodes: int,
        max_edges: int,
        require_connected: bool = True,
        tolerance: float = 1.0e-6,
    ) -> None:
        if min_nodes < 1 or min_edges < 1:
            raise ValueError("candidate minimum sizes must be positive")
        if max_nodes < min_nodes or max_edges < min_edges:
            raise ValueError("candidate maximum sizes must cover minimum sizes")
        if tolerance < 0.0:
            raise ValueError("candidate validation tolerance must be non-negative")
        self.min_nodes = int(min_nodes)
        self.min_edges = int(min_edges)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.require_connected = bool(require_connected)
        self.tolerance = float(tolerance)

    @staticmethod
    def _signature(candidate: HardSubgraphCandidate):
        nodes = tuple(sorted(set(int(node) for node in candidate.node_ids)))
        edges = tuple(
            sorted(
                set(
                    (min(int(left), int(right)), max(int(left), int(right)))
                    for left, right in candidate.edge_index
                )
            )
        )
        return nodes, edges

    def _is_valid(
        self,
        candidate: HardSubgraphCandidate,
        original_adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> bool:
        nodes, edges = self._signature(candidate)
        raw_edges = tuple(
            (min(int(left), int(right)), max(int(left), int(right)))
            for left, right in candidate.edge_index
        )
        node_count = int(original_adjacency.shape[0])
        if not self.min_nodes <= len(nodes) <= self.max_nodes:
            return False
        if not self.min_edges <= len(edges) <= self.max_edges:
            return False
        if len(nodes) != len(candidate.node_ids) or len(edges) != len(candidate.edge_index):
            return False
        node_set = set(nodes)
        if nodes[0] < 0 or nodes[-1] >= node_count:
            return False
        if len(candidate.original_edge_weights) != len(edges):
            return False
        for edge_index, (left, right) in enumerate(raw_edges):
            if left == right or left not in node_set or right not in node_set:
                return False
            if not bool(edge_mask[left, right]):
                return False
            source_weight = float(original_adjacency[left, right].detach().cpu())
            if abs(source_weight - float(candidate.original_edge_weights[edge_index])) > self.tolerance:
                return False
        if self.require_connected and _largest_connected_component_size(nodes, edges) != len(nodes):
            return False
        return True

    def finalize(
        self,
        candidates: Sequence[HardSubgraphCandidate],
        original_adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> HardCandidatePoolResult:
        if original_adjacency.ndim != 2 or original_adjacency.shape[0] != original_adjacency.shape[1]:
            raise ValueError("candidate source adjacency must be square")
        if tuple(edge_mask.shape) != tuple(original_adjacency.shape):
            raise ValueError("candidate edge mask must match adjacency")
        valid = [
            candidate
            for candidate in candidates
            if self._is_valid(candidate, original_adjacency, edge_mask)
        ]
        ordered = sorted(
            valid,
            key=lambda item: (-float(item.candidate_score), int(item.seed_node)),
        )
        unique = []
        signatures = set()
        for candidate in ordered:
            signature = self._signature(candidate)
            if signature in signatures:
                continue
            signatures.add(signature)
            unique.append(candidate)
        return HardCandidatePoolResult(
            candidates=tuple(unique),
            window_valid=bool(unique),
            rejected_invalid=len(candidates) - len(valid),
            removed_duplicates=len(valid) - len(unique),
        )


@dataclass(frozen=True)
class HardTimepointResult:
    time_index: int
    time_start: float
    original_node_count: int
    original_edge_count: int
    candidate_pool: Tuple[HardSubgraphCandidate, ...]
    selected_subgraphs: Tuple[HardSubgraphCandidate, ...]
    num_valid_subgraphs: int
    subgraph_mask: Tuple[bool, ...]
    union_graph: Optional[HardUnionGraph]
    fidelity: Optional[FidelityResult]
    spectral_gw_greedy_trace: Tuple[Dict[str, Any], ...]
    window_valid: bool
    candidate_rejected_invalid: int
    candidate_removed_duplicates: int
    prefilter_summary: Dict[str, Any]


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
    laplacian_eta: float
    heat_kernel_t: float
    node_measure: str
    timepoints: Tuple[HardTimepointResult, ...]
    H_SGW_full: Tuple[float, ...]
    H_SGW_soft: Tuple[float, ...]
    H_SGW_hard: Tuple[float, ...]
    Gamma_SGW_full: Tuple[Tuple[float, ...], ...]
    Gamma_SGW_soft: Tuple[Tuple[float, ...], ...]
    Gamma_SGW_hard: Tuple[Tuple[float, ...], ...]
    Gamma_SGW_full_mask: Tuple[bool, ...]
    Gamma_SGW_soft_mask: Tuple[bool, ...]
    Gamma_SGW_hard_mask: Tuple[bool, ...]
    num_valid_hard_windows: int
    eligible_for_stage_c: bool
    exclusion_reason: Optional[str]
    candidate_score_scaler: Optional[Dict[str, Any]]


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
        model: Any,
        config: Optional[HardExtractionConfig] = None,
        candidate_score_scaler: Optional[CandidateScoreStandardizer] = None,
    ) -> None:
        self.model = model
        self.config = config or HardExtractionConfig()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        model_config = self.model.config
        theory_defaults = TGSGWTheoryConfig()
        self.fidelity = HardExportFidelityEvaluator(
            laplacian_eta=model_config.laplacian_eta,
            heat_kernel_t=getattr(
                model_config,
                "heat_kernel_t",
                getattr(model_config, "diffusion_time", 1.0),
            ),
            spectral_quantile_grid=getattr(
                model_config,
                "spectral_quantile_grid",
                theory_defaults.spectral_quantile_grid,
            ),
            train_entropic_reg=getattr(model_config, "gw_entropic_reg", 5.0e-2),
            train_max_iter=getattr(model_config, "gw_max_iter", 20),
            train_sinkhorn_iter=getattr(model_config, "gw_sinkhorn_iter", 20),
            tolerance=getattr(model_config, "gw_tolerance", 1.0e-7),
            eval_entropic_reg=self.config.eval_gw_entropic_reg,
            eval_max_iter=self.config.eval_gw_max_iter,
            eval_sinkhorn_iter=self.config.eval_gw_sinkhorn_iter,
        )
        self.greedy = SpectralGWGreedyExporter(
            self.fidelity,
            beta_lambda=self.config.beta_lambda,
            beta_gw=self.config.beta_gw,
            beta_overlap=self.config.beta_overlap,
            min_export_gain=self.config.min_export_gain,
            epsilon=self.config.epsilon,
            beta_size=self.config.beta_size,
            candidate_score_scaler=candidate_score_scaler,
            prefilter_r1=self.config.prefilter_discriminative_top_r1,
            prefilter_r2=self.config.prefilter_spectral_top_r2,
            max_union_nodes=self.config.max_union_nodes,
            max_union_edges=self.config.max_union_edges,
        )
        self.candidate_score_scaler = candidate_score_scaler
        self.evolution = SpectralGWEvolutionEncoder(self.fidelity)
        self.candidate_pool_builder = HardCandidatePoolBuilder(
            min_nodes=self.config.min_nodes,
            min_edges=self.config.min_edges,
            max_nodes=self.config.max_nodes,
            max_edges=self.config.max_edges,
        )
        self.union_builder = HardUnionGraphBuilder()

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

    def build_candidate_pool(self, sample: GraphSequenceSample, time_index: int):
        """Build a frozen window candidate pool without running Top-K selection."""

        if self.model.training:
            raise RuntimeError("hard candidate generation requires model.eval()")
        with torch.no_grad():
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
            adjacency = sample.adjacency[time_index]
            pool = self.candidate_pool_builder.finalize(
                candidates, adjacency, features.edge_mask
            )
            deduplicated = tuple(self._deduplicate(pool.candidates))
        return features, selection, pool, deduplicated

    def extract_sample(self, sample: GraphSequenceSample) -> HardSampleResult:
        timepoints = []
        full_adjacencies = []
        soft_adjacencies = []
        hard_adjacencies = []
        full_edge_masks = []
        hard_edge_masks = []
        with torch.no_grad():
            for time_index in range(sample.num_timepoints):
                features, selection, pool, deduplicated = self.build_candidate_pool(
                    sample, time_index
                )
                adjacency = sample.adjacency[time_index]
                selected, union_graph, greedy_trace = self.greedy.select(
                    deduplicated,
                    adjacency,
                    sample.node_names[time_index],
                    features.edge_mask,
                    selection.soft_adjacency,
                    self.config.top_k,
                    communities=sample.communities[time_index],
                    edge_presence_threshold=sample.edge_presence_threshold,
                )
                fidelity = (
                    self.fidelity.evaluate(
                        adjacency,
                        selection.soft_adjacency,
                        features.edge_mask,
                        union_graph,
                    )
                    if union_graph is not None
                    else None
                )
                edge_count = int(torch.triu(features.edge_mask, diagonal=1).sum())
                timepoints.append(
                    HardTimepointResult(
                        time_index=time_index,
                        time_start=float(sample.window_starts[time_index].detach().cpu()),
                        original_node_count=int(features.node_features.shape[0]),
                        original_edge_count=edge_count,
                        candidate_pool=tuple(deduplicated),
                        selected_subgraphs=selected,
                        num_valid_subgraphs=len(selected),
                        subgraph_mask=tuple(
                            index < len(selected) for index in range(self.config.top_k)
                        ),
                        union_graph=union_graph,
                        fidelity=fidelity,
                        spectral_gw_greedy_trace=greedy_trace,
                        window_valid=union_graph is not None,
                        candidate_rejected_invalid=pool.rejected_invalid,
                        candidate_removed_duplicates=pool.removed_duplicates,
                        prefilter_summary=dict(self.greedy.last_prefilter_summary),
                    )
                )
                full_adjacencies.append(adjacency)
                soft_adjacencies.append(selection.soft_adjacency)
                full_edge_masks.append(features.edge_mask)
                hard_adjacencies.append(
                    union_graph.adjacency if union_graph is not None else None
                )
                hard_edge_masks.append(
                    union_graph.adjacency.abs() > 0.0
                    if union_graph is not None
                    else None
                )
        time_values = [float(value) for value in sample.window_starts.detach().cpu()]
        full_evolution = self.evolution.encode(
            full_adjacencies, full_edge_masks, time_values
        )
        soft_evolution = self.evolution.encode(
            soft_adjacencies, full_edge_masks, time_values
        )
        hard_evolution = self.evolution.encode(
            hard_adjacencies, hard_edge_masks, time_values
        )
        valid_hard_windows = sum(
            1 for timepoint in timepoints if timepoint.window_valid
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
            laplacian_eta=self.model.config.laplacian_eta,
            heat_kernel_t=getattr(
                self.model.config,
                "heat_kernel_t",
                getattr(self.model.config, "diffusion_time", 1.0),
            ),
            node_measure=getattr(self.model.config, "node_measure", "uniform"),
            timepoints=tuple(timepoints),
            H_SGW_full=full_evolution.aggregated,
            H_SGW_soft=soft_evolution.aggregated,
            H_SGW_hard=hard_evolution.aggregated,
            Gamma_SGW_full=full_evolution.gamma,
            Gamma_SGW_soft=soft_evolution.gamma,
            Gamma_SGW_hard=hard_evolution.gamma,
            Gamma_SGW_full_mask=full_evolution.step_mask,
            Gamma_SGW_soft_mask=soft_evolution.step_mask,
            Gamma_SGW_hard_mask=hard_evolution.step_mask,
            num_valid_hard_windows=valid_hard_windows,
            eligible_for_stage_c=valid_hard_windows >= 2,
            exclusion_reason=(
                None
                if valid_hard_windows >= 2
                else "fewer_than_two_valid_hard_windows"
            ),
            candidate_score_scaler=(
                self.candidate_score_scaler.to_dict()
                if self.candidate_score_scaler is not None
                else None
            ),
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
        "laplacian_eta": result.laplacian_eta,
        "heat_kernel_t": result.heat_kernel_t,
        "node_measure": result.node_measure,
        "hard_extraction_config": asdict(config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "data_protocol_sha256": data_protocol_sha256,
        "timepoints": [],
        "H_SGW_full": list(result.H_SGW_full),
        "H_SGW_soft": list(result.H_SGW_soft),
        "H_SGW_hard": list(result.H_SGW_hard),
        "Gamma_SGW_full": [list(item) for item in result.Gamma_SGW_full],
        "Gamma_SGW_soft": [list(item) for item in result.Gamma_SGW_soft],
        "Gamma_SGW_hard": [list(item) for item in result.Gamma_SGW_hard],
        "Gamma_SGW_full_mask": list(result.Gamma_SGW_full_mask),
        "Gamma_SGW_soft_mask": list(result.Gamma_SGW_soft_mask),
        "Gamma_SGW_hard_mask": list(result.Gamma_SGW_hard_mask),
        "num_valid_hard_windows": result.num_valid_hard_windows,
        "eligible_for_stage_c": result.eligible_for_stage_c,
        "exclusion_reason": result.exclusion_reason,
        "candidate_score_scaler": result.candidate_score_scaler,
    }
    for timepoint in result.timepoints:
        union = timepoint.union_graph
        fidelity = timepoint.fidelity.to_dict() if timepoint.fidelity is not None else {}
        payload["timepoints"].append(
            {
                "time_index": timepoint.time_index,
                "time_start": timepoint.time_start,
                "time_mask": True,
                "original_node_count": timepoint.original_node_count,
                "original_edge_count": timepoint.original_edge_count,
                "num_valid_subgraphs": timepoint.num_valid_subgraphs,
                "window_valid": timepoint.window_valid,
                "candidate_rejected_invalid": timepoint.candidate_rejected_invalid,
                "candidate_removed_duplicates": timepoint.candidate_removed_duplicates,
                "prefilter_summary": dict(timepoint.prefilter_summary),
                "subgraph_mask": list(timepoint.subgraph_mask),
                "hard_union_available": union is not None,
                "union_node_ids": list(union.node_ids) if union is not None else None,
                "union_node_names": list(union.node_names) if union is not None else None,
                "union_edge_index": [list(edge) for edge in union.edge_index] if union is not None else None,
                "union_original_edge_weights": list(union.original_edge_weights) if union is not None else None,
                "union_community_labels": list(union.community_labels) if union is not None else None,
                "union_num_nodes": union.num_nodes if union is not None else 0,
                "union_num_edges": union.num_edges if union is not None else 0,
                "spectral_gw_greedy_trace": list(timepoint.spectral_gw_greedy_trace),
                **fidelity,
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
