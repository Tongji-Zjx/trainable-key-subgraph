"""Leakage-aware scalar fusion for frozen subgraph and static branches."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from keysubgraph.tge.trainer import classification_metrics, site_stratified_roc_auc


@dataclass(frozen=True)
class SafeFusionConfig:
    dataset: str
    weight_grid_step: float = 0.05
    stability_penalty: float = 0.50
    near_best_tolerance: float = 0.002
    minimum_mean_auc_gain: float = 0.005
    maximum_worst_rotation_drop: float = 0.01
    maximum_site_auc_drop: float = 0.01
    minimum_non_decreasing_rotations: int = 3
    epsilon: float = 1.0e-8

    def __post_init__(self):
        if self.dataset.lower() not in ("adhd", "wmrc"):
            raise ValueError("safe fusion dataset must be ADHD or WMRC")
        if not 0.0 < self.weight_grid_step <= 1.0:
            raise ValueError("fusion weight grid step must be in (0,1]")
        if min(
            self.stability_penalty,
            self.near_best_tolerance,
            self.minimum_mean_auc_gain,
            self.maximum_worst_rotation_drop,
            self.maximum_site_auc_drop,
        ) < 0.0:
            raise ValueError("safe fusion tolerances must be non-negative")
        if self.minimum_non_decreasing_rotations < 1:
            raise ValueError("minimum non-decreasing rotations must be positive")


def _as_array(values, dtype):
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("safe fusion arrays must be non-empty and one-dimensional")
    return result


def validate_fusion_fold(fold: Mapping[str, object]) -> dict:
    required = ("sample_keys", "sites", "labels", "subgraph_logits", "background_logits")
    if any(name not in fold for name in required):
        raise ValueError("safe fusion fold is missing required arrays")
    result = {
        "sample_keys": _as_array(fold["sample_keys"], str),
        "sites": _as_array(fold["sites"], str),
        "labels": _as_array(fold["labels"], np.int64),
        "subgraph_logits": _as_array(fold["subgraph_logits"], np.float64),
        "background_logits": _as_array(fold["background_logits"], np.float64),
    }
    count = result["labels"].size
    if any(value.size != count for value in result.values()):
        raise ValueError("safe fusion fold arrays do not align")
    if len(set(result["sample_keys"].tolist())) != count:
        raise ValueError("safe fusion sample keys must be unique inside a fold")
    if set(result["labels"].tolist()) != {0, 1}:
        raise ValueError("safe fusion fold must contain both classes")
    if not np.isfinite(result["subgraph_logits"]).all() or not np.isfinite(
        result["background_logits"]
    ).all():
        raise ValueError("safe fusion logits must be finite")
    return result


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def score_logits(labels, sites, logits):
    labels = np.asarray(labels, dtype=np.int64)
    probability = sigmoid(logits)
    result = classification_metrics(labels, probability, 0.5)
    result["site_stratified_roc_auc"] = site_stratified_roc_auc(
        labels.tolist(), probability.tolist(), np.asarray(sites, dtype=str).tolist()
    )
    result["logit_mean"] = float(np.mean(logits))
    result["logit_standard_deviation"] = float(np.std(logits))
    return result


def fit_zero_preserving_scale(validation_folds, epsilon=1.0e-8):
    folds = [validate_fusion_fold(fold) for fold in validation_folds]
    subgraph = np.concatenate([fold["subgraph_logits"] for fold in folds])
    background = np.concatenate([fold["background_logits"] for fold in folds])
    subgraph_scale = float(np.std(subgraph))
    background_scale = float(np.std(background))
    if subgraph_scale <= epsilon or background_scale <= epsilon:
        raise ValueError("a fusion branch has degenerate development-OOF logits")
    return {
        "subgraph_standard_deviation": subgraph_scale,
        "background_standard_deviation": background_scale,
        "background_scale_ratio": subgraph_scale / background_scale,
        "centering_applied": False,
    }


def fuse_logits(subgraph_logits, background_logits, subgraph_weight, scale_ratio):
    weight = float(subgraph_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("subgraph fusion weight must be in [0,1]")
    subgraph = np.asarray(subgraph_logits, dtype=np.float64)
    background = np.asarray(background_logits, dtype=np.float64)
    if subgraph.shape != background.shape:
        raise ValueError("fusion logit arrays do not align")
    if weight == 1.0:
        return subgraph.copy()
    if weight == 0.0:
        return float(scale_ratio) * background
    return weight * subgraph + (1.0 - weight) * float(scale_ratio) * background


def apply_safe_fusion(selection, subgraph_logits, background_logits):
    source = str(selection["selected_source"])
    if source == "subgraph_exact_fallback":
        return np.asarray(subgraph_logits, dtype=np.float64).copy()
    if source == "background_exact_fallback":
        return np.asarray(background_logits, dtype=np.float64).copy()
    if source != "safe_convex_fusion":
        raise ValueError("unknown safe fusion source")
    return fuse_logits(
        subgraph_logits,
        background_logits,
        selection["selected_subgraph_weight"],
        selection["scale"]["background_scale_ratio"],
    )


def ranking_pair_changes(labels, baseline_logits, candidate_logits):
    labels = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    baseline_correct = baseline[positive, None] > baseline[negative][None, :]
    candidate_correct = candidate[positive, None] > candidate[negative][None, :]
    corrected = int(np.logical_and(~baseline_correct, candidate_correct).sum())
    corrupted = int(np.logical_and(baseline_correct, ~candidate_correct).sum())
    total = int(baseline_correct.size)
    return {
        "pair_count": total,
        "corrected_pair_count": corrected,
        "corrupted_pair_count": corrupted,
        "corrected_pair_ratio": corrected / float(total),
        "corrupted_pair_ratio": corrupted / float(total),
        "net_corrected_pair_count": corrected - corrupted,
    }


def _candidate_row(folds, weight, scale_ratio, stability_penalty):
    rotation_metrics = []
    differences = []
    all_labels = []
    all_sites = []
    all_subgraph = []
    all_fused = []
    for rotation, fold in enumerate(folds):
        fused = fuse_logits(
            fold["subgraph_logits"], fold["background_logits"], weight, scale_ratio
        )
        baseline_metrics = score_logits(
            fold["labels"], fold["sites"], fold["subgraph_logits"]
        )
        metrics = score_logits(fold["labels"], fold["sites"], fused)
        difference = float(metrics["roc_auc"]) - float(baseline_metrics["roc_auc"])
        differences.append(difference)
        rotation_metrics.append(
            {
                "rotation": rotation,
                "roc_auc": float(metrics["roc_auc"]),
                "baseline_roc_auc": float(baseline_metrics["roc_auc"]),
                "auc_difference": difference,
                "accuracy": float(metrics["accuracy"]),
                "site_stratified_roc_auc": metrics["site_stratified_roc_auc"],
            }
        )
        all_labels.append(fold["labels"])
        all_sites.append(fold["sites"])
        all_subgraph.append(fold["subgraph_logits"])
        all_fused.append(fused)
    aucs = np.asarray([row["roc_auc"] for row in rotation_metrics], dtype=np.float64)
    labels = np.concatenate(all_labels)
    sites = np.concatenate(all_sites)
    subgraph = np.concatenate(all_subgraph)
    fused = np.concatenate(all_fused)
    pooled = score_logits(labels, sites, fused)
    pooled_baseline = score_logits(labels, sites, subgraph)
    pairs = ranking_pair_changes(labels, subgraph, fused)
    return {
        "subgraph_weight": float(weight),
        "mean_rotation_roc_auc": float(np.mean(aucs)),
        "std_rotation_roc_auc": float(np.std(aucs)),
        "stability_objective": float(np.mean(aucs) - stability_penalty * np.std(aucs)),
        "mean_auc_gain": float(np.mean(differences)),
        "worst_rotation_auc_difference": float(np.min(differences)),
        "non_decreasing_rotation_count": int(np.sum(np.asarray(differences) >= -1.0e-12)),
        "development_oof_roc_auc": float(pooled["roc_auc"]),
        "development_oof_accuracy": float(pooled["accuracy"]),
        "development_oof_site_auc": pooled["site_stratified_roc_auc"],
        "site_auc_difference": (
            None
            if pooled["site_stratified_roc_auc"] is None
            or pooled_baseline["site_stratified_roc_auc"] is None
            else float(pooled["site_stratified_roc_auc"])
            - float(pooled_baseline["site_stratified_roc_auc"])
        ),
        "ranking_pairs": pairs,
        "rotations": rotation_metrics,
    }


def _weight_grid(step):
    count = int(round(1.0 / float(step)))
    if not np.isclose(count * float(step), 1.0, atol=1.0e-9):
        raise ValueError("fusion grid step must divide one exactly")
    return [float(index * step) for index in range(count + 1)]


def select_safe_fusion(validation_folds: Sequence[Mapping[str, object]], config):
    """Select one shared weight from disjoint development validation rotations."""

    folds = [validate_fusion_fold(fold) for fold in validation_folds]
    if len(folds) < config.minimum_non_decreasing_rotations:
        raise ValueError("not enough validation rotations for the no-harm rule")
    seen = set()
    for fold in folds:
        current = set(fold["sample_keys"].tolist())
        if seen.intersection(current):
            raise ValueError("development validation rotations are not disjoint")
        seen.update(current)
    scale = fit_zero_preserving_scale(folds, config.epsilon)
    candidates = [
        _candidate_row(
            folds,
            weight,
            scale["background_scale_ratio"],
            config.stability_penalty,
        )
        for weight in _weight_grid(config.weight_grid_step)
    ]
    best_objective = max(row["stability_objective"] for row in candidates)
    near_best = [
        row for row in candidates
        if row["stability_objective"] >= best_objective - config.near_best_tolerance
    ]
    if config.dataset.lower() == "adhd":
        proposed = max(near_best, key=lambda row: row["subgraph_weight"])
    else:
        proposed = max(
            near_best,
            key=lambda row: (row["stability_objective"], row["mean_rotation_roc_auc"]),
        )
    site_ok = (
        proposed["site_auc_difference"] is None
        or proposed["site_auc_difference"] >= -config.maximum_site_auc_drop
    )
    common_checks = {
        "minimum_mean_auc_gain": (
            proposed["mean_auc_gain"] >= config.minimum_mean_auc_gain
        ),
        "minimum_non_decreasing_rotations": (
            proposed["non_decreasing_rotation_count"]
            >= config.minimum_non_decreasing_rotations
        ),
        "maximum_worst_rotation_drop": (
            proposed["worst_rotation_auc_difference"]
            >= -config.maximum_worst_rotation_drop
        ),
        "maximum_site_auc_drop": bool(site_ok),
    }
    if config.dataset.lower() == "wmrc":
        common_checks["positive_net_corrected_pairs"] = (
            proposed["ranking_pairs"]["net_corrected_pair_count"] > 0
        )
    accepted = all(common_checks.values())
    if accepted:
        source = "safe_convex_fusion"
        selected_weight = proposed["subgraph_weight"]
        fallback_reason = None
    elif config.dataset.lower() == "adhd":
        source = "subgraph_exact_fallback"
        selected_weight = 1.0
        fallback_reason = "ADHD development-OOF no-harm checks did not all pass"
    else:
        subgraph = next(row for row in candidates if row["subgraph_weight"] == 1.0)
        background = next(row for row in candidates if row["subgraph_weight"] == 0.0)
        if background["stability_objective"] > subgraph["stability_objective"]:
            source = "background_exact_fallback"
            selected_weight = 0.0
        else:
            source = "subgraph_exact_fallback"
            selected_weight = 1.0
        fallback_reason = "WMRC fusion checks failed; selected the stabler single branch"
    return {
        "artifact_type": "mokse_background_safe_fusion_selection_v1",
        "config": asdict(config),
        "development_sample_count": len(seen),
        "development_rotation_count": len(folds),
        "scale": scale,
        "proposed_candidate": proposed,
        "acceptance_checks": common_checks,
        "fusion_accepted": bool(accepted),
        "selected_source": source,
        "selected_subgraph_weight": float(selected_weight),
        "fallback_reason": fallback_reason,
        "candidates": candidates,
        "fixed_test_used_for_selection": False,
        "decision_threshold": 0.5,
    }

