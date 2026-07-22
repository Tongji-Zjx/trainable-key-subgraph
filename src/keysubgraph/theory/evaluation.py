"""Frozen hard-export fidelity, greedy selection, evolution and theory bounds."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
) -> Optional[HardUnionGraph]:
    """Union selected candidates while restoring signed weights from the source graph."""

    if not candidates:
        return None
    nodes = sorted(set(node for candidate in candidates for node in candidate.node_ids))
    if not nodes:
        return None
    edges = sorted(
        set(
            (min(left, right), max(left, right))
            for candidate in candidates
            for left, right in candidate.edge_index
        )
    )
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
    )


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
    ) -> None:
        if any(value < 0.0 for value in (beta_lambda, beta_gw, beta_overlap)):
            raise ValueError("greedy export penalties must be non-negative")
        self.fidelity = fidelity_evaluator
        self.beta_lambda = float(beta_lambda)
        self.beta_gw = float(beta_gw)
        self.beta_overlap = float(beta_overlap)
        self.min_export_gain = float(min_export_gain)
        self.epsilon = float(epsilon)

    def select(
        self,
        candidates: Sequence[Any],
        original_adjacency: torch.Tensor,
        node_names: Sequence[str],
        edge_mask: torch.Tensor,
        soft_adjacency: torch.Tensor,
        max_k: int,
    ) -> Tuple[Tuple[Any, ...], Optional[HardUnionGraph], Tuple[Dict[str, Any], ...]]:
        if max_k < 1:
            raise ValueError("max_k must be positive")
        selected_indices: List[int] = []
        remaining = list(range(len(candidates)))
        current_objective = 0.0
        cache: Dict[Tuple[int, ...], Tuple[HardUnionGraph, float, float, bool, int, float]] = {}
        trace: List[Dict[str, Any]] = []
        while len(selected_indices) < max_k and remaining:
            trials = []
            for index in remaining:
                key = tuple(sorted(selected_indices + [index]))
                if key not in cache:
                    union = build_hard_union_graph(
                        [candidates[item] for item in key],
                        original_adjacency,
                        node_names,
                        edge_mask,
                    )
                    if union is None:
                        continue
                    spectral_error, gw_error, converged, iterations, residual = (
                        self.fidelity.fast_soft_to_hard(
                            soft_adjacency, edge_mask, union
                        )
                    )
                    cache[key] = (
                        union,
                        spectral_error,
                        gw_error,
                        converged,
                        iterations,
                        residual,
                    )
                union, spectral_error, gw_error, converged, iterations, residual = cache[key]
                chosen = [candidates[item] for item in key]
                old_score_sum = sum(float(item.candidate_score) for item in chosen)
                overlap_penalty = sum(
                    _candidate_overlap(chosen[left], chosen[right], self.epsilon)
                    for left in range(len(chosen))
                    for right in range(left + 1, len(chosen))
                )
                objective = (
                    old_score_sum
                    - self.beta_lambda * spectral_error
                    - self.beta_gw * gw_error
                    - self.beta_overlap * overlap_penalty
                )
                trials.append(
                    (
                        objective - current_objective,
                        -int(candidates[index].seed_node),
                        index,
                        objective,
                        old_score_sum,
                        spectral_error,
                        gw_error,
                        overlap_penalty,
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
            if marginal_gain < self.min_export_gain:
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
                    "old_score_sum": best[4],
                    "spectral_error": best[5],
                    "gw_error": best[6],
                    "overlap_penalty": best[7],
                    "selected_union_num_nodes": best[8].num_nodes,
                    "selected_union_num_edges": best[8].num_edges,
                    "gw_solver_converged": best[9],
                    "gw_solver_iterations": best[10],
                    "gw_solver_residual": best[11],
                }
            )
        selected = tuple(candidates[index] for index in selected_indices)
        union = build_hard_union_graph(
            selected, original_adjacency, node_names, edge_mask
        )
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
