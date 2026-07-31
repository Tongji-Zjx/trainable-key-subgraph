"""Exact empirical class margins and extraction radii for Stage 0."""

from __future__ import absolute_import, division, print_function

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist


STAGE0_METRIC_NAMES = (
    "delta_full",
    "eta_0_pair",
    "eta_1_pair",
    "eta_0_ot",
    "eta_1_ot",
    "lower_bound_pair",
    "lower_bound_ot",
    "delta_hard",
)


def _matrix(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("{} must be a nonempty matrix".format(name))
    if not np.isfinite(result).all():
        raise ValueError("{} contains non-finite values".format(name))
    return result


def euclidean_cost(first: Any, second: Any) -> np.ndarray:
    first = _matrix(first, "first features")
    second = _matrix(second, "second features")
    if first.shape[1] != second.shape[1]:
        raise ValueError("feature matrices have different dimensions")
    return cdist(first, second, metric="euclidean")


@lru_cache(maxsize=64)
def _transport_constraints(
    source_count: int, target_count: int
) -> Tuple[Any, np.ndarray]:
    if source_count < 1 or target_count < 1:
        raise ValueError("transport counts must be positive")
    variable_count = source_count * target_count
    columns = np.arange(variable_count, dtype=np.int64)
    source_rows = np.repeat(
        np.arange(source_count, dtype=np.int64), target_count
    )
    target_rows = source_count + np.tile(
        np.arange(target_count, dtype=np.int64), source_count
    )
    rows = np.concatenate((source_rows, target_rows))
    cols = np.concatenate((columns, columns))
    data = np.ones(rows.shape[0], dtype=np.float64)
    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(source_count + target_count, variable_count),
    ).tocsr()
    # One equality is redundant.  Removing it makes HiGHS more stable while
    # preserving the exact uniform transportation polytope.
    matrix = matrix[:-1]
    values = np.concatenate(
        (
            np.full(source_count, 1.0 / float(source_count)),
            np.full(target_count - 1, 1.0 / float(target_count)),
        )
    )
    return matrix, values


def exact_uniform_wasserstein_from_cost(cost: Any) -> float:
    """Return exact uniform discrete OT for a precomputed ground cost."""

    cost = _matrix(cost, "ground cost")
    if bool((cost < 0.0).any()):
        raise ValueError("ground cost cannot be negative")
    source_count, target_count = cost.shape
    if source_count == target_count:
        rows, columns = linear_sum_assignment(cost)
        return float(cost[rows, columns].mean())
    constraints, values = _transport_constraints(
        source_count, target_count
    )
    result = linprog(
        cost.reshape(-1),
        A_eq=constraints,
        b_eq=values,
        bounds=(0.0, None),
        method="highs",
    )
    if not bool(result.success):
        raise RuntimeError(
            "exact uniform OT failed: {}".format(result.message)
        )
    return float(result.fun)


def exact_uniform_wasserstein(first: Any, second: Any) -> float:
    return exact_uniform_wasserstein_from_cost(
        euclidean_cost(first, second)
    )


def fit_train_only_standardizer(
    train_full_features: Any, epsilon: float = 1.0e-8
) -> Dict[str, Any]:
    values = _matrix(train_full_features, "train full features")
    if epsilon <= 0.0:
        raise ValueError("standardizer epsilon must be positive")
    mean = values.mean(axis=0)
    variance = np.square(values - mean).mean(axis=0)
    scale = np.sqrt(variance + float(epsilon))
    return {
        "fit_split": "train",
        "fit_source": "full_core_only",
        "sample_count": int(values.shape[0]),
        "mean": mean,
        "scale": scale,
        "epsilon": float(epsilon),
    }


def apply_standardizer(values: Any, scaler: Mapping[str, Any]) -> np.ndarray:
    values = _matrix(values, "features")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("standardizer dimension mismatch")
    if not np.isfinite(mean).all() or bool((scale <= 0.0).any()):
        raise ValueError("standardizer parameters are invalid")
    return (values - mean) / scale


def class_margin_metrics(
    full_features: Any,
    hard_features: Any,
    labels: Sequence[int],
    tolerance: float = 1.0e-8,
) -> Dict[str, Any]:
    full = _matrix(full_features, "full features")
    hard = _matrix(hard_features, "hard features")
    labels = np.asarray(labels, dtype=np.int64)
    if full.shape != hard.shape or labels.shape != (full.shape[0],):
        raise ValueError("Stage-0 paired features and labels are misaligned")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Stage-0 metrics require both classes")
    indices = {
        label: np.flatnonzero(labels == label) for label in (0, 1)
    }
    full_classes = {label: full[value] for label, value in indices.items()}
    hard_classes = {label: hard[value] for label, value in indices.items()}
    paired_errors = np.linalg.norm(full - hard, axis=1)
    eta_pair = {
        label: float(paired_errors[value].mean())
        for label, value in indices.items()
    }
    eta_ot = {
        label: exact_uniform_wasserstein(
            full_classes[label], hard_classes[label]
        )
        for label in (0, 1)
    }
    delta_full = exact_uniform_wasserstein(
        full_classes[0], full_classes[1]
    )
    delta_hard = exact_uniform_wasserstein(
        hard_classes[0], hard_classes[1]
    )
    lower_pair = delta_full - eta_pair[0] - eta_pair[1]
    lower_ot = delta_full - eta_ot[0] - eta_ot[1]
    ot_within_pair = all(
        eta_ot[label] <= eta_pair[label] + float(tolerance)
        for label in (0, 1)
    )
    hard_above_lower = delta_hard + float(tolerance) >= lower_ot
    return {
        "sample_count": int(full.shape[0]),
        "class_counts": {
            str(label): int(indices[label].size) for label in (0, 1)
        },
        "ground_metric": "euclidean",
        "delta_full": float(delta_full),
        "eta_0_pair": eta_pair[0],
        "eta_1_pair": eta_pair[1],
        "eta_0_ot": eta_ot[0],
        "eta_1_ot": eta_ot[1],
        "lower_bound_pair": float(lower_pair),
        "lower_bound_ot": float(lower_ot),
        "delta_hard": float(delta_hard),
        "checks": {
            "eta_ot_not_above_pair": bool(ot_within_pair),
            "hard_margin_not_below_ot_lower_bound": bool(
                hard_above_lower
            ),
        },
    }


def _bootstrap_once(
    random: np.random.RandomState,
    costs: Mapping[str, np.ndarray],
    paired: Mapping[int, np.ndarray],
) -> Dict[str, float]:
    count0 = paired[0].shape[0]
    count1 = paired[1].shape[0]
    index0 = random.randint(0, count0, size=count0)
    index1 = random.randint(0, count1, size=count1)
    delta_full = exact_uniform_wasserstein_from_cost(
        costs["full_01"][np.ix_(index0, index1)]
    )
    delta_hard = exact_uniform_wasserstein_from_cost(
        costs["hard_01"][np.ix_(index0, index1)]
    )
    eta_0_ot = exact_uniform_wasserstein_from_cost(
        costs["full_hard_0"][np.ix_(index0, index0)]
    )
    eta_1_ot = exact_uniform_wasserstein_from_cost(
        costs["full_hard_1"][np.ix_(index1, index1)]
    )
    eta_0_pair = float(paired[0][index0].mean())
    eta_1_pair = float(paired[1][index1].mean())
    return {
        "delta_full": delta_full,
        "eta_0_pair": eta_0_pair,
        "eta_1_pair": eta_1_pair,
        "eta_0_ot": eta_0_ot,
        "eta_1_ot": eta_1_ot,
        "lower_bound_pair": (
            delta_full - eta_0_pair - eta_1_pair
        ),
        "lower_bound_ot": delta_full - eta_0_ot - eta_1_ot,
        "delta_hard": delta_hard,
    }


def stratified_paired_bootstrap(
    full_features: Any,
    hard_features: Any,
    labels: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    """Stratified paired bootstrap with cached pairwise ground costs."""

    full = _matrix(full_features, "full features")
    hard = _matrix(hard_features, "hard features")
    labels = np.asarray(labels, dtype=np.int64)
    if full.shape != hard.shape or labels.shape != (full.shape[0],):
        raise ValueError("bootstrap inputs are misaligned")
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    if any(value.size < 1 for value in indices.values()):
        raise ValueError("bootstrap requires both classes")
    full_classes = {label: full[value] for label, value in indices.items()}
    hard_classes = {label: hard[value] for label, value in indices.items()}
    paired = {
        label: np.linalg.norm(
            full_classes[label] - hard_classes[label], axis=1
        )
        for label in (0, 1)
    }
    costs = {
        "full_01": euclidean_cost(full_classes[0], full_classes[1]),
        "hard_01": euclidean_cost(hard_classes[0], hard_classes[1]),
        "full_hard_0": euclidean_cost(
            full_classes[0], hard_classes[0]
        ),
        "full_hard_1": euclidean_cost(
            full_classes[1], hard_classes[1]
        ),
    }
    random = np.random.RandomState(int(seed))
    samples = {name: [] for name in STAGE0_METRIC_NAMES}
    for _ in range(int(repeats)):
        current = _bootstrap_once(random, costs, paired)
        for name in STAGE0_METRIC_NAMES:
            samples[name].append(float(current[name]))
    intervals = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "mean": float(array.mean()),
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
        }
    return {
        "method": "stratified_paired_exact_discrete_ot",
        "repeats": int(repeats),
        "seed": int(seed),
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def component_margin_metrics(
    full_features: Any,
    hard_features: Any,
    labels: Sequence[int],
) -> Dict[str, Any]:
    full = _matrix(full_features, "full features")
    hard = _matrix(hard_features, "hard features")
    if full.shape[1] != 18 or hard.shape != full.shape:
        raise ValueError("component diagnostics require paired 18-D cores")
    blocks = {
        "spectral_direction": slice(0, 16),
        "spectral_speed": slice(16, 17),
        "gw_speed": slice(17, 18),
    }
    return {
        name: class_margin_metrics(
            full[:, block], hard[:, block], labels
        )
        for name, block in blocks.items()
    }
