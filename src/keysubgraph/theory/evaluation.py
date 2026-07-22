"""Frozen hard-export fidelity, greedy selection, evolution and theory bounds."""

from __future__ import absolute_import, division, print_function

import math
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .spectral_gw import (
    DifferentiableGWLoss,
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    SpectralStateExtractor,
    spectral_w1,
    spectral_winf_exact,
)


@dataclass(frozen=True)
class HardUnionGraph:
    node_ids: Tuple[int, ...]
    node_names: Tuple[str, ...]
    edge_index: Tuple[Tuple[int, int], ...]
    original_edge_weights: Tuple[float, ...]
    adjacency: torch.Tensor
    community_labels: Tuple[int, ...] = ()
    edge_presence_threshold: float = 0.0

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index)


def build_hard_union_graph(
    candidates: Sequence[Any],
    original_adjacency: torch.Tensor,
    node_names: Sequence[str],
    edge_mask: torch.Tensor,
    communities: Optional[torch.Tensor] = None,
    edge_presence_threshold: float = 0.0,
    max_union_nodes: Optional[int] = None,
    max_union_edges: Optional[int] = None,
) -> Optional[HardUnionGraph]:
    """Union selected candidates while restoring signed weights from the source graph."""

    if not candidates:
        return None
    if original_adjacency.ndim != 2 or original_adjacency.shape[0] != original_adjacency.shape[1]:
        raise ValueError("original adjacency must be square")
    node_count = int(original_adjacency.shape[0])
    if tuple(edge_mask.shape) != (node_count, node_count):
        raise ValueError("edge mask must match original adjacency")
    if len(node_names) != node_count:
        raise ValueError("node names must align with original adjacency")
    if communities is not None and tuple(communities.shape) != (node_count,):
        raise ValueError("community labels must align with original adjacency")
    if edge_presence_threshold < 0.0:
        raise ValueError("edge presence threshold must be non-negative")
    nodes = sorted(set(node for candidate in candidates for node in candidate.node_ids))
    if not nodes:
        return None
    if nodes[0] < 0 or nodes[-1] >= node_count:
        raise ValueError("hard union contains an out-of-range node")
    edges = sorted(
        set(
            (min(left, right), max(left, right))
            for candidate in candidates
            for left, right in candidate.edge_index
        )
    )
    for candidate in candidates:
        candidate_nodes = set(int(node) for node in candidate.node_ids)
        if any(
            int(left) not in candidate_nodes or int(right) not in candidate_nodes
            for left, right in candidate.edge_index
        ):
            raise ValueError("candidate edge endpoint is absent from candidate nodes")
    if max_union_nodes is not None and len(nodes) > int(max_union_nodes):
        raise ValueError("hard union exceeds its node budget")
    if max_union_edges is not None and len(edges) > int(max_union_edges):
        raise ValueError("hard union exceeds its edge budget")
    for left, right in edges:
        if left == right or not bool(edge_mask[left, right]):
            raise ValueError("hard union contains an edge absent from the original graph")
    local = {node: index for index, node in enumerate(nodes)}
    adjacency = original_adjacency.new_zeros((len(nodes), len(nodes)))
    weights = []
    for left, right in edges:
        weight = original_adjacency[left, right]
        if float(weight.abs().detach().cpu()) == 0.0:
            raise ValueError("hard union attempted to create a zero/absent edge")
        adjacency[local[left], local[right]] = weight
        adjacency[local[right], local[left]] = weight
        weights.append(float(weight.detach().cpu()))
    return HardUnionGraph(
        node_ids=tuple(nodes),
        node_names=tuple(str(node_names[node]) for node in nodes),
        edge_index=tuple(edges),
        original_edge_weights=tuple(weights),
        adjacency=adjacency,
        community_labels=(
            tuple(int(communities[node]) for node in nodes)
            if communities is not None
            else ()
        ),
        edge_presence_threshold=float(edge_presence_threshold),
    )


class HardUnionGraphBuilder(object):
    """Canonical non-induced hard union with explicit empty-window semantics."""

    def __init__(
        self,
        max_union_nodes: Optional[int] = None,
        max_union_edges: Optional[int] = None,
    ) -> None:
        if max_union_nodes is not None and int(max_union_nodes) < 1:
            raise ValueError("max_union_nodes must be positive")
        if max_union_edges is not None and int(max_union_edges) < 1:
            raise ValueError("max_union_edges must be positive")
        self.max_union_nodes = (
            None if max_union_nodes is None else int(max_union_nodes)
        )
        self.max_union_edges = (
            None if max_union_edges is None else int(max_union_edges)
        )

    def build(
        self,
        original_adjacency: torch.Tensor,
        node_names: Sequence[str],
        edge_mask: torch.Tensor,
        selected_subgraphs: Sequence[Any],
        communities: Optional[torch.Tensor] = None,
        edge_presence_threshold: float = 0.0,
    ) -> Tuple[Optional[HardUnionGraph], bool]:
        graph = build_hard_union_graph(
            selected_subgraphs,
            original_adjacency,
            node_names,
            edge_mask,
            communities=communities,
            edge_presence_threshold=edge_presence_threshold,
            max_union_nodes=self.max_union_nodes,
            max_union_edges=self.max_union_edges,
        )
        return graph, graph is not None


@dataclass(frozen=True)
class FidelityResult:
    full_laplacian_eigenvalues: Tuple[float, ...]
    soft_laplacian_eigenvalues: Tuple[float, ...]
    hard_union_laplacian_eigenvalues: Tuple[float, ...]
    full_spectral_quantiles: Tuple[float, ...]
    soft_spectral_quantiles: Tuple[float, ...]
    hard_spectral_quantiles: Tuple[float, ...]
    full_spectral_gap: float
    soft_spectral_gap: float
    hard_union_spectral_gap: float
    full_to_soft_laplacian_fro_error: float
    full_to_soft_laplacian_operator_error: float
    full_to_soft_spectral_winf: float
    full_to_soft_gw_error: float
    soft_to_hard_spectral_winf: float
    soft_to_hard_gw_error: float
    full_to_hard_spectral_winf: float
    full_to_hard_gw_error: float
    gw_solver_error_proxy: float
    gw_solver_converged: bool
    gw_solver_iterations: int
    gw_solver_residual: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardExportFidelityEvaluator:
    """Evaluate full->soft, soft->hard and full->hard spectral--GW errors."""

    def __init__(
        self,
        laplacian_eta: float,
        heat_kernel_t: float,
        spectral_quantile_grid: Sequence[float],
        train_entropic_reg: float = 5.0e-2,
        train_max_iter: int = 20,
        train_sinkhorn_iter: int = 20,
        tolerance: float = 1.0e-7,
        eval_entropic_reg: float = 1.0e-2,
        eval_max_iter: int = 100,
        eval_sinkhorn_iter: int = 100,
    ) -> None:
        self.laplacian = SignedLaplacianBuilder(laplacian_eta)
        self.spectral = SpectralStateExtractor(spectral_quantile_grid)
        self.heat = HeatKernelMetricBuilder(heat_kernel_t)
        self.train_gw = DifferentiableGWLoss(
            train_entropic_reg,
            train_max_iter,
            tolerance,
            train_sinkhorn_iter,
            failure_strategy="use_last",
        )
        self.eval_gw = DifferentiableGWLoss(
            eval_entropic_reg,
            eval_max_iter,
            tolerance,
            eval_sinkhorn_iter,
            failure_strategy="use_last",
        )

    def _geometry(self, adjacency: torch.Tensor, edge_mask: torch.Tensor):
        laplacian = self.laplacian(adjacency, edge_mask=edge_mask)
        spectrum = self.spectral(laplacian)
        metric = self.heat(laplacian).distance
        return laplacian, spectrum, metric

    def fast_soft_to_hard(
        self,
        soft_adjacency: torch.Tensor,
        soft_edge_mask: torch.Tensor,
        hard_union: HardUnionGraph,
    ) -> Tuple[float, float, bool, int, float]:
        _, soft_spectrum, soft_metric = self._geometry(soft_adjacency, soft_edge_mask)
        hard_mask = hard_union.adjacency.abs() > 0.0
        _, hard_spectrum, hard_metric = self._geometry(hard_union.adjacency, hard_mask)
        spectral_error = spectral_winf_exact(
            soft_spectrum.eigenvalues, hard_spectrum.eigenvalues
        )
        gw = self.train_gw(soft_metric, hard_metric)
        return (
            float(spectral_error.detach().cpu()),
            float(gw.distance.detach().cpu()),
            gw.converged,
            gw.iterations,
            gw.residual,
        )

    def fast_spectral_soft_to_hard(
        self,
        soft_adjacency: torch.Tensor,
        soft_edge_mask: torch.Tensor,
        hard_union: HardUnionGraph,
    ) -> float:
        """Compute the stage-two spectral filter without invoking GW."""

        _, soft_spectrum, _ = self._geometry(soft_adjacency, soft_edge_mask)
        hard_mask = hard_union.adjacency.abs() > hard_union.edge_presence_threshold
        _, hard_spectrum, _ = self._geometry(hard_union.adjacency, hard_mask)
        return float(
            spectral_winf_exact(
                soft_spectrum.eigenvalues, hard_spectrum.eigenvalues
            ).detach().cpu()
        )

    def evaluate(
        self,
        full_adjacency: torch.Tensor,
        soft_adjacency: torch.Tensor,
        edge_mask: torch.Tensor,
        hard_union: HardUnionGraph,
    ) -> FidelityResult:
        full_laplacian, full_spectrum, full_metric = self._geometry(
            full_adjacency, edge_mask
        )
        soft_laplacian, soft_spectrum, soft_metric = self._geometry(
            soft_adjacency, edge_mask
        )
        hard_mask = hard_union.adjacency.abs() > 0.0
        hard_laplacian, hard_spectrum, hard_metric = self._geometry(
            hard_union.adjacency, hard_mask
        )
        difference = full_laplacian - soft_laplacian
        full_soft_train = self.train_gw(full_metric, soft_metric)
        soft_hard_train = self.train_gw(soft_metric, hard_metric)
        full_hard_train = self.train_gw(full_metric, hard_metric)
        full_soft_eval = self.eval_gw(full_metric, soft_metric)
        soft_hard_eval = self.eval_gw(soft_metric, hard_metric)
        full_hard_eval = self.eval_gw(full_metric, hard_metric)
        solver_error = max(
            abs(float(full_soft_train.distance) - float(full_soft_eval.distance)),
            abs(float(soft_hard_train.distance) - float(soft_hard_eval.distance)),
            abs(float(full_hard_train.distance) - float(full_hard_eval.distance)),
        )
        all_solver_results = (
            full_soft_train,
            soft_hard_train,
            full_hard_train,
            full_soft_eval,
            soft_hard_eval,
            full_hard_eval,
        )
        return FidelityResult(
            tuple(float(value) for value in full_spectrum.eigenvalues.detach().cpu()),
            tuple(float(value) for value in soft_spectrum.eigenvalues.detach().cpu()),
            tuple(float(value) for value in hard_spectrum.eigenvalues.detach().cpu()),
            tuple(float(value) for value in full_spectrum.quantiles.detach().cpu()),
            tuple(float(value) for value in soft_spectrum.quantiles.detach().cpu()),
            tuple(float(value) for value in hard_spectrum.quantiles.detach().cpu()),
            float(full_spectrum.spectral_gap.detach().cpu()),
            float(soft_spectrum.spectral_gap.detach().cpu()),
            float(hard_spectrum.spectral_gap.detach().cpu()),
            float(torch.linalg.matrix_norm(difference, ord="fro").detach().cpu()),
            float(torch.linalg.eigvalsh(difference).abs().max().detach().cpu()),
            float(spectral_winf_exact(full_spectrum.eigenvalues, soft_spectrum.eigenvalues)),
            float(full_soft_eval.distance),
            float(spectral_winf_exact(soft_spectrum.eigenvalues, hard_spectrum.eigenvalues)),
            float(soft_hard_eval.distance),
            float(spectral_winf_exact(full_spectrum.eigenvalues, hard_spectrum.eigenvalues)),
            float(full_hard_eval.distance),
            solver_error,
            all(result.converged for result in all_solver_results),
            max(result.iterations for result in all_solver_results),
            max(result.residual for result in all_solver_results),
        )


def _candidate_overlap(left: Any, right: Any, epsilon: float) -> float:
    left_nodes, right_nodes = set(left.node_ids), set(right.node_ids)
    left_edges, right_edges = set(left.edge_index), set(right.edge_index)
    node_union = left_nodes | right_nodes
    edge_union = left_edges | right_edges
    return (
        len(left_nodes & right_nodes) / (len(node_union) + epsilon)
        + len(left_edges & right_edges) / (len(edge_union) + epsilon)
    )


@dataclass(frozen=True)
class CandidateScoreStandardizer:
    """Training-fold-only standardization of hard-candidate score components."""

    feature_names: Tuple[str, ...]
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    weights: Tuple[float, ...]
    fit_split: str
    standard_deviation_floor: float
    data_protocol_sha256: Optional[str] = None
    teacher_checkpoint_sha256: Optional[str] = None

    DEFAULT_FEATURE_NAMES: ClassVar[Tuple[str, ...]] = (
        "score_node",
        "score_edge",
        "score_connectivity",
        "score_dynamic",
    )

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size < 1 or not (
            len(self.mean) == len(self.scale) == len(self.weights) == size
        ):
            raise ValueError("candidate score scaler dimensions are inconsistent")
        if self.fit_split != "train":
            raise ValueError("candidate score scaler must be fitted on train only")
        if self.standard_deviation_floor <= 0.0:
            raise ValueError("candidate score scale floor must be positive")
        if any(value < self.standard_deviation_floor for value in self.scale):
            raise ValueError("candidate score scaler contains a sub-floor scale")
        if any(value < 0.0 for value in self.weights):
            raise ValueError("candidate score weights must be non-negative")

    @classmethod
    def fit(
        cls,
        candidates: Sequence[Any],
        weights: Sequence[float],
        fit_split: str = "train",
        standard_deviation_floor: float = 1.0e-6,
        feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
        data_protocol_sha256: Optional[str] = None,
        teacher_checkpoint_sha256: Optional[str] = None,
    ) -> "CandidateScoreStandardizer":
        if fit_split != "train":
            raise ValueError("candidate score scaler cannot fit validation or test")
        names = tuple(str(name) for name in feature_names)
        if not candidates:
            raise ValueError("candidate score scaler requires training candidates")
        if len(weights) != len(names):
            raise ValueError("candidate score weights do not match score components")
        matrix = torch.tensor(
            [
                [float(getattr(candidate, name)) for name in names]
                for candidate in candidates
            ],
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("candidate scores contain non-finite values")
        mean = matrix.mean(dim=0)
        scale = matrix.std(dim=0, unbiased=False).clamp_min(
            float(standard_deviation_floor)
        )
        return cls(
            feature_names=names,
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            weights=tuple(float(value) for value in weights),
            fit_split=fit_split,
            standard_deviation_floor=float(standard_deviation_floor),
            data_protocol_sha256=data_protocol_sha256,
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
        )

    def score(self, candidate: Any) -> float:
        standardized = [
            (float(getattr(candidate, name)) - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.mean, self.scale)
        ]
        return sum(
            weight * value for weight, value in zip(self.weights, standardized)
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        payload["artifact_type"] = "tg_sgw_candidate_score_scaler"
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateScoreStandardizer":
        if payload.get("schema_version") != 1 or payload.get("artifact_type") != "tg_sgw_candidate_score_scaler":
            raise ValueError("unsupported candidate score scaler artifact")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weights=tuple(float(value) for value in payload["weights"]),
            fit_split=str(payload["fit_split"]),
            standard_deviation_floor=float(payload["standard_deviation_floor"]),
            data_protocol_sha256=payload.get("data_protocol_sha256"),
            teacher_checkpoint_sha256=payload.get("teacher_checkpoint_sha256"),
        )

    def save(self, path: Path, overwrite: bool = False) -> Path:
        path = Path(path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError("candidate score scaler already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
        return path

    @classmethod
    def load(cls, path: Path) -> "CandidateScoreStandardizer":
        with Path(path).resolve().open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class SpectralGWGreedyExporter:
    """Select at most K candidates using the set-level spectral--GW objective."""

    def __init__(
        self,
        fidelity_evaluator: HardExportFidelityEvaluator,
        beta_lambda: float,
        beta_gw: float,
        beta_overlap: float,
        min_export_gain: float = 0.0,
        epsilon: float = 1.0e-8,
        beta_size: float = 0.0,
        candidate_score_scaler: Optional[CandidateScoreStandardizer] = None,
        prefilter_r1: Optional[int] = None,
        prefilter_r2: Optional[int] = None,
        max_union_nodes: Optional[int] = None,
        max_union_edges: Optional[int] = None,
    ) -> None:
        if any(value < 0.0 for value in (beta_lambda, beta_gw, beta_overlap, beta_size)):
            raise ValueError("greedy export penalties must be non-negative")
        if prefilter_r1 is not None and int(prefilter_r1) < 1:
            raise ValueError("prefilter_r1 must be positive")
        if prefilter_r2 is not None and int(prefilter_r2) < 1:
            raise ValueError("prefilter_r2 must be positive")
        if prefilter_r1 is not None and prefilter_r2 is not None and int(prefilter_r2) > int(prefilter_r1):
            raise ValueError("prefilter_r2 cannot exceed prefilter_r1")
        self.fidelity = fidelity_evaluator
        self.beta_lambda = float(beta_lambda)
        self.beta_gw = float(beta_gw)
        self.beta_overlap = float(beta_overlap)
        self.beta_size = float(beta_size)
        self.min_export_gain = float(min_export_gain)
        self.epsilon = float(epsilon)
        self.candidate_score_scaler = candidate_score_scaler
        self.prefilter_r1 = None if prefilter_r1 is None else int(prefilter_r1)
        self.prefilter_r2 = None if prefilter_r2 is None else int(prefilter_r2)
        self.union_builder = HardUnionGraphBuilder(max_union_nodes, max_union_edges)
        self.last_prefilter_summary: Dict[str, Any] = {}

    def _discriminative_score(self, candidate: Any) -> float:
        if self.candidate_score_scaler is None:
            return float(candidate.candidate_score)
        return float(self.candidate_score_scaler.score(candidate))

    def select(
        self,
        candidates: Sequence[Any],
        original_adjacency: torch.Tensor,
        node_names: Sequence[str],
        edge_mask: torch.Tensor,
        soft_adjacency: torch.Tensor,
        max_k: int,
        communities: Optional[torch.Tensor] = None,
        edge_presence_threshold: float = 0.0,
    ) -> Tuple[Tuple[Any, ...], Optional[HardUnionGraph], Tuple[Dict[str, Any], ...]]:
        if max_k < 1:
            raise ValueError("max_k must be positive")
        if self.prefilter_r2 is not None and max_k > self.prefilter_r2:
            raise ValueError("max_k cannot exceed prefilter_r2")
        if not candidates:
            self.last_prefilter_summary = {
                "input_candidates": 0,
                "discriminative_r1": 0,
                "spectral_r2": 0,
                "gw_evaluated_sets": 0,
            }
            return (), None, ()

        discriminative_scores = {
            index: self._discriminative_score(candidate)
            for index, candidate in enumerate(candidates)
        }
        stage_one = sorted(
            range(len(candidates)),
            key=lambda index: (
                -discriminative_scores[index],
                int(candidates[index].seed_node),
            ),
        )[: min(len(candidates), self.prefilter_r1 or len(candidates))]
        spectral_trials = []
        for index in stage_one:
            try:
                union, valid = self.union_builder.build(
                    original_adjacency,
                    node_names,
                    edge_mask,
                    (candidates[index],),
                    communities=communities,
                    edge_presence_threshold=edge_presence_threshold,
                )
            except ValueError:
                continue
            if not valid or union is None:
                continue
            spectral_error = self.fidelity.fast_spectral_soft_to_hard(
                soft_adjacency, edge_mask, union
            )
            spectral_trials.append(
                (
                    spectral_error,
                    -discriminative_scores[index],
                    int(candidates[index].seed_node),
                    index,
                )
            )
        spectral_trials.sort()
        stage_two = [
            item[3]
            for item in spectral_trials[
                : min(len(spectral_trials), self.prefilter_r2 or len(spectral_trials))
            ]
        ]
        selected_indices: List[int] = []
        remaining = list(stage_two)
        current_objective = 0.0
        cache: Dict[Tuple[int, ...], Tuple[HardUnionGraph, float, float, bool, int, float]] = {}
        trace: List[Dict[str, Any]] = []
        gw_evaluated_sets = 0
        source_edge_count = max(
            1, int(torch.triu(edge_mask, diagonal=1).sum().detach().cpu())
        )
        while len(selected_indices) < max_k and remaining:
            trials = []
            for index in remaining:
                key = tuple(sorted(selected_indices + [index]))
                if key not in cache:
                    try:
                        union, valid = self.union_builder.build(
                            original_adjacency,
                            node_names,
                            edge_mask,
                            [candidates[item] for item in key],
                            communities=communities,
                            edge_presence_threshold=edge_presence_threshold,
                        )
                    except ValueError:
                        continue
                    if not valid or union is None:
                        continue
                    spectral_error, gw_error, converged, iterations, residual = (
                        self.fidelity.fast_soft_to_hard(
                            soft_adjacency, edge_mask, union
                        )
                    )
                    gw_evaluated_sets += 1
                    cache[key] = (
                        union,
                        spectral_error,
                        gw_error,
                        converged,
                        iterations,
                        residual,
                    )
                union, spectral_error, gw_error, converged, iterations, residual = cache[key]
                chosen_indices = list(key)
                chosen = [candidates[item] for item in chosen_indices]
                score_sum = sum(discriminative_scores[item] for item in chosen_indices)
                overlap_penalty = sum(
                    _candidate_overlap(chosen[left], chosen[right], self.epsilon)
                    for left in range(len(chosen))
                    for right in range(left + 1, len(chosen))
                )
                node_denominator = float(
                    self.union_builder.max_union_nodes or original_adjacency.shape[0]
                )
                edge_denominator = float(
                    self.union_builder.max_union_edges or source_edge_count
                )
                size_penalty = (
                    union.num_nodes / node_denominator
                    + union.num_edges / edge_denominator
                )
                objective = (
                    score_sum
                    - self.beta_lambda * spectral_error
                    - self.beta_gw * gw_error
                    - self.beta_overlap * overlap_penalty
                    - self.beta_size * size_penalty
                )
                trials.append(
                    (
                        objective - current_objective,
                        -int(candidates[index].seed_node),
                        index,
                        objective,
                        score_sum,
                        spectral_error,
                        gw_error,
                        overlap_penalty,
                        size_penalty,
                        union,
                        converged,
                        iterations,
                        residual,
                    )
                )
            if not trials:
                break
            best = max(trials, key=lambda item: (item[0], item[1]))
            marginal_gain = best[0]
            if marginal_gain <= self.min_export_gain:
                break
            index = best[2]
            selected_indices.append(index)
            remaining.remove(index)
            current_objective = best[3]
            trace.append(
                {
                    "candidate_id": int(candidates[index].seed_node),
                    "marginal_gain": marginal_gain,
                    "objective": current_objective,
                    "discriminative_score_sum": best[4],
                    "old_score_sum": best[4],
                    "spectral_error": best[5],
                    "gw_error": best[6],
                    "overlap_penalty": best[7],
                    "size_penalty": best[8],
                    "selected_union_num_nodes": best[9].num_nodes,
                    "selected_union_num_edges": best[9].num_edges,
                    "gw_solver_converged": best[10],
                    "gw_solver_iterations": best[11],
                    "gw_solver_residual": best[12],
                    "prefilter_r1_count": len(stage_one),
                    "prefilter_r2_count": len(stage_two),
                }
            )
        selected = tuple(candidates[index] for index in selected_indices)
        union, _ = self.union_builder.build(
            original_adjacency,
            node_names,
            edge_mask,
            selected,
            communities=communities,
            edge_presence_threshold=edge_presence_threshold,
        )
        self.last_prefilter_summary = {
            "input_candidates": len(candidates),
            "discriminative_r1": len(stage_one),
            "spectral_r2": len(stage_two),
            "gw_evaluated_sets": gw_evaluated_sets,
            "candidate_score_scaler_fit_split": (
                self.candidate_score_scaler.fit_split
                if self.candidate_score_scaler is not None
                else None
            ),
        }
        return selected, union, tuple(trace)


@dataclass(frozen=True)
class EvolutionRepresentation:
    gamma: Tuple[Tuple[float, ...], ...]
    step_mask: Tuple[bool, ...]
    aggregated: Tuple[float, ...]
    gw_solver_converged: Tuple[bool, ...]


class SpectralGWEvolutionEncoder:
    """Mask-aware fixed aggregation of spectral directions and spectral/GW speeds."""

    def __init__(self, fidelity: HardExportFidelityEvaluator) -> None:
        self.fidelity = fidelity

    def encode(
        self,
        adjacencies: Sequence[Optional[torch.Tensor]],
        edge_masks: Sequence[Optional[torch.Tensor]],
        time_values: Sequence[float],
    ) -> EvolutionRepresentation:
        if not (len(adjacencies) == len(edge_masks) == len(time_values)):
            raise ValueError("evolution inputs must have equal sequence lengths")
        states = []
        for adjacency, edge_mask in zip(adjacencies, edge_masks):
            if adjacency is None or edge_mask is None:
                states.append(None)
            else:
                _, spectrum, metric = self.fidelity._geometry(adjacency, edge_mask)
                states.append((spectrum, metric))
        gamma_tensors = []
        step_mask = []
        convergence = []
        path_length = 0.0
        for index in range(max(0, len(states) - 1)):
            left, right = states[index], states[index + 1]
            if left is None or right is None:
                step_mask.append(False)
                continue
            tau = float(time_values[index + 1]) - float(time_values[index])
            if tau <= 0.0:
                raise ValueError("time values must be strictly increasing")
            delta = right[0].quantiles - left[0].quantiles
            spectral_distance = spectral_w1(left[0].eigenvalues, right[0].eigenvalues)
            gw = self.fidelity.eval_gw(left[1], right[1])
            spec_speed = spectral_distance / tau
            gw_speed = gw.distance / tau
            gamma_tensors.append(torch.cat((delta, spec_speed.reshape(1), gw_speed.reshape(1))))
            step_mask.append(True)
            convergence.append(gw.converged)
            path_length += math.sqrt(float(spectral_distance) ** 2 + float(gw.distance) ** 2)
        feature_dim = len(self.fidelity.spectral.quantile_grid) + 2
        if gamma_tensors:
            stacked = torch.stack(gamma_tensors)
            mean = stacked.mean(dim=0)
            standard_deviation = stacked.std(dim=0, unbiased=False)
            max_spec = stacked[:, -2].max()
            max_gw = stacked[:, -1].max()
            aggregated = torch.cat(
                (
                    mean,
                    standard_deviation,
                    mean.new_tensor([path_length]),
                    max_spec.reshape(1),
                    max_gw.reshape(1),
                )
            )
        else:
            aggregated = torch.zeros(feature_dim * 2 + 3, dtype=torch.float32)
        return EvolutionRepresentation(
            gamma=tuple(
                tuple(float(value) for value in item.detach().cpu())
                for item in gamma_tensors
            ),
            step_mask=tuple(step_mask),
            aggregated=tuple(float(value) for value in aggregated.detach().cpu()),
            gw_solver_converged=tuple(convergence),
        )


class TheoryBoundEvaluator:
    """Compute class separation, representation radii and a bootstrap lower bound."""

    def __init__(self, bootstrap_repeats: int = 1000, seed: int = 42) -> None:
        if bootstrap_repeats < 1:
            raise ValueError("bootstrap_repeats must be positive")
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.seed = int(seed)

    @staticmethod
    def _components(full: np.ndarray, hard: np.ndarray, labels: np.ndarray):
        classes = sorted(set(int(value) for value in labels.tolist()))
        if classes != [0, 1]:
            raise ValueError("theory bound requires both binary classes")
        means = [full[labels == label].mean(axis=0) for label in classes]
        separation = float(np.linalg.norm(means[0] - means[1]))
        radii = [
            float(np.linalg.norm(full[labels == label] - hard[labels == label], axis=1).mean())
            for label in classes
        ]
        return separation, radii[0], radii[1], separation - radii[0] - radii[1]

    def evaluate(
        self,
        full_representations: Sequence[Sequence[float]],
        hard_representations: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> Dict[str, Any]:
        full = np.asarray(full_representations, dtype=np.float64)
        hard = np.asarray(hard_representations, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int64)
        if full.ndim != 2 or hard.shape != full.shape or labels_array.shape != (len(full),):
            raise ValueError("theory-bound arrays have incompatible shapes")
        separation, eta_a, eta_b, lower = self._components(full, hard, labels_array)
        random_state = np.random.RandomState(self.seed)
        bootstrap = []
        class_indices = [np.flatnonzero(labels_array == label) for label in (0, 1)]
        for _ in range(self.bootstrap_repeats):
            sampled = np.concatenate(
                [
                    random_state.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
            bootstrap.append(
                self._components(full[sampled], hard[sampled], labels_array[sampled])[3]
            )
        interval = np.percentile(np.asarray(bootstrap), [2.5, 97.5])
        return {
            "delta_ab": separation,
            "eta_a": eta_a,
            "eta_b": eta_b,
            "lower_bound": lower,
            "lower_bound_positive": bool(lower > 0.0),
            "bootstrap_confidence_interval": [float(interval[0]), float(interval[1])],
            "sufficient_condition_status": (
                "verified" if lower > 0.0 else "not_verified"
            ),
            "bootstrap_repeats": self.bootstrap_repeats,
            "bootstrap_seed": self.seed,
        }


def one_step_sgw_error_bound(
    spectral_error_left: float,
    spectral_error_right: float,
    gw_error_left: float,
    gw_error_right: float,
    quantile_count: int,
    tau: float,
) -> float:
    """Triangle-inequality bound for one spectral--GW evolution step."""

    if quantile_count < 1 or tau <= 0.0:
        raise ValueError("quantile_count and tau must be positive")
    spectral_sum = float(spectral_error_left) + float(spectral_error_right)
    gw_sum = float(gw_error_left) + float(gw_error_right)
    return math.sqrt(
        quantile_count * spectral_sum ** 2
        + (spectral_sum / tau) ** 2
        + (gw_sum / tau) ** 2
    )
