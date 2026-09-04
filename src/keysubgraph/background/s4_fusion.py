"""Strict OOF seed ensembling and anchored residual fusion for S4."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from .safe_fusion import ranking_pair_changes, score_logits


def _vector(values, dtype, name):
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("{} must be a non-empty vector".format(name))
    return result


def _validate_seed_payload(payload, require_oof=False):
    required = ("seed", "sample_keys", "logits")
    if any(name not in payload for name in required):
        raise ValueError("S4 seed prediction payload is incomplete")
    if require_oof and payload.get("prediction_role") != "development_oof":
        raise ValueError("S4 seed fitting requires development-OOF predictions")
    sample_keys = _vector(payload["sample_keys"], str, "sample_keys")
    logits = _vector(payload["logits"], np.float64, "logits")
    if sample_keys.shape != logits.shape:
        raise ValueError("S4 seed sample/logit arrays do not align")
    if len(set(sample_keys.tolist())) != sample_keys.size:
        raise ValueError("S4 seed sample keys must be unique")
    if not np.isfinite(logits).all():
        raise ValueError("S4 seed logits must be finite")
    return {
        "seed": int(payload["seed"]),
        "sample_keys": sample_keys,
        "logits": logits,
        "prediction_role": payload.get("prediction_role"),
    }


def _align_seed_payloads(payloads, expected_seeds=None, require_oof=False):
    rows = [_validate_seed_payload(row, require_oof=require_oof) for row in payloads]
    if len(rows) < 2:
        raise ValueError("S4 robust ensemble requires at least two fixed seeds")
    by_seed = {row["seed"]: row for row in rows}
    if len(by_seed) != len(rows):
        raise ValueError("S4 robust ensemble seeds must be unique")
    seeds = tuple(sorted(by_seed))
    if expected_seeds is not None and seeds != tuple(sorted(int(x) for x in expected_seeds)):
        raise ValueError("S4 robust ensemble seed set mismatch")
    reference = by_seed[seeds[0]]["sample_keys"]
    reference_set = set(reference.tolist())
    aligned = []
    for seed in seeds:
        row = by_seed[seed]
        if set(row["sample_keys"].tolist()) != reference_set:
            raise ValueError("S4 seed prediction cohorts differ")
        position = {key: index for index, key in enumerate(row["sample_keys"].tolist())}
        aligned.append(row["logits"][[position[key] for key in reference]])
    return seeds, reference, np.stack(aligned, axis=1)


def _seed_ensemble_arrays(logit_matrix, means, scales, tau, epsilon):
    standardized = (logit_matrix - means[None, :]) / scales[None, :]
    standardized_median = np.median(standardized, axis=1)
    uncertainty = np.median(
        np.abs(standardized - standardized_median[:, None]), axis=1
    )
    if tau <= epsilon:
        reliability = np.where(uncertainty <= epsilon, 1.0, 0.0)
    else:
        reliability = 1.0 / (1.0 + uncertainty / (tau + epsilon))
    return {
        # Raw logits retain the classifier's zero decision point for ACC@0.5.
        "raw_median_logit": np.median(logit_matrix, axis=1),
        # Standardized scores are used only by complementary residual fusion.
        "standardized_median_score": standardized_median,
        "standardized_seed_logits": standardized,
        "uncertainty": uncertainty,
        "reliability": reliability,
    }


def fit_s4_seed_ensemble(
    development_oof_predictions: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int] = (43, 44, 45),
    epsilon: float = 1.0e-8,
):
    """Fit per-seed score scales exclusively from aligned development OOF logits."""

    seeds, sample_keys, matrix = _align_seed_payloads(
        development_oof_predictions,
        expected_seeds=expected_seeds,
        require_oof=True,
    )
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    if np.any(scales <= float(epsilon)):
        raise ValueError("an S4 seed has degenerate development-OOF logits")
    provisional = _seed_ensemble_arrays(
        matrix, means, scales, tau=1.0, epsilon=float(epsilon)
    )
    tau = float(np.median(provisional["uncertainty"]))
    return {
        "artifact_type": "mokse_s4_seed_ensemble_fit_v1",
        "seeds": list(seeds),
        "development_oof_sample_count": int(sample_keys.size),
        "seed_logit_mean": means.tolist(),
        "seed_logit_scale": scales.tolist(),
        "uncertainty_tau": tau,
        "epsilon": float(epsilon),
        "representation_averaging": False,
        "fit_prediction_role": "development_oof",
        "test_used_for_fit": False,
    }


def apply_s4_seed_ensemble(
    fit: Mapping[str, object],
    seed_predictions: Sequence[Mapping[str, object]],
):
    if fit.get("artifact_type") != "mokse_s4_seed_ensemble_fit_v1":
        raise ValueError("S4 seed ensemble artifact type mismatch")
    seeds = tuple(int(value) for value in fit["seeds"])
    observed, sample_keys, matrix = _align_seed_payloads(
        seed_predictions, expected_seeds=seeds, require_oof=False
    )
    if observed != tuple(sorted(seeds)):
        raise ValueError("S4 seed application order mismatch")
    means = np.asarray(fit["seed_logit_mean"], dtype=np.float64)
    scales = np.asarray(fit["seed_logit_scale"], dtype=np.float64)
    arrays = _seed_ensemble_arrays(
        matrix,
        means,
        scales,
        float(fit["uncertainty_tau"]),
        float(fit["epsilon"]),
    )
    arrays["sample_keys"] = sample_keys
    arrays["seeds"] = np.asarray(seeds, dtype=np.int64)
    return arrays


@dataclass(frozen=True)
class S4AnchoredFusionConfig:
    dataset: str
    beta_grid: Tuple[float, ...] = (
        0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    )
    stability_penalty: float = 0.50
    near_best_tolerance: float = 0.001
    minimum_mean_auc_gain: float = 0.003
    maximum_worst_fold_auc_drop: float = 0.01
    maximum_site_auc_drop: float = 0.01
    maximum_mean_accuracy_drop: float = 0.005
    maximum_mean_auprc_drop: float = 0.01
    minimum_non_decreasing_folds: int = 3
    epsilon: float = 1.0e-8

    def __post_init__(self):
        if self.dataset.lower() not in ("adhd", "wmrc"):
            raise ValueError("S4 fusion dataset must be ADHD or WMRC")
        if not self.beta_grid or min(self.beta_grid) < 0.0 or max(self.beta_grid) > 0.40:
            raise ValueError("S4 beta grid must be non-empty and inside [0,0.40]")
        if 0.0 not in self.beta_grid:
            raise ValueError("S4 beta grid must contain the exact fallback beta=0")
        if self.minimum_non_decreasing_folds < 1:
            raise ValueError("S4 minimum non-decreasing folds must be positive")
        numeric = (
            self.stability_penalty,
            self.near_best_tolerance,
            self.minimum_mean_auc_gain,
            self.maximum_worst_fold_auc_drop,
            self.maximum_site_auc_drop,
            self.maximum_mean_accuracy_drop,
            self.maximum_mean_auprc_drop,
            self.epsilon,
        )
        if any(float(value) < 0.0 for value in numeric) or self.epsilon <= 0.0:
            raise ValueError("S4 fusion tolerances must be non-negative")


@dataclass(frozen=True)
class S4StaticPromotionConfig:
    """Explicit test-guided S3-to-S4 promotion policy.

    This policy is intentionally separate from fusion fitting.  Its outputs
    are exploratory because fixed-test metrics are used for architecture
    promotion, but the fixed test still never fits model weights or beta.
    """

    minimum_mean_auc_gain: float = 0.0
    maximum_worst_fold_auc_drop: float = 0.01
    maximum_auc_std_increase: float = 0.005
    maximum_mean_accuracy_drop: float = 0.005
    maximum_mean_auprc_drop: float = 0.01
    maximum_mean_site_auc_drop: float = 0.01

    def __post_init__(self):
        if min(
            self.minimum_mean_auc_gain,
            self.maximum_worst_fold_auc_drop,
            self.maximum_auc_std_increase,
            self.maximum_mean_accuracy_drop,
            self.maximum_mean_auprc_drop,
            self.maximum_mean_site_auc_drop,
        ) < 0.0:
            raise ValueError("S4 static promotion tolerances must be non-negative")


def _summarize_static_metrics(rows):
    if len(rows) < 2:
        raise ValueError("S4 static promotion requires multiple folds")
    required = ("roc_auc", "accuracy", "auprc")
    if any(any(name not in row for name in required) for row in rows):
        raise ValueError("S4 static promotion metrics are incomplete")
    auc = np.asarray([float(row["roc_auc"]) for row in rows], dtype=np.float64)
    accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
    auprc = np.asarray([float(row["auprc"]) for row in rows], dtype=np.float64)
    site_auc = [
        float(row["site_stratified_roc_auc"])
        for row in rows if row.get("site_stratified_roc_auc") is not None
    ]
    return {
        "fold_count": len(rows),
        "mean_roc_auc": float(np.mean(auc)),
        "std_roc_auc": float(np.std(auc)),
        "worst_fold_roc_auc": float(np.min(auc)),
        "mean_accuracy": float(np.mean(accuracy)),
        "mean_auprc": float(np.mean(auprc)),
        "mean_site_auc": None if not site_auc else float(np.mean(site_auc)),
    }


def select_s4_static_promotion(
    s3_fixed_test_metrics: Sequence[Mapping[str, object]],
    s4_fixed_test_metrics: Sequence[Mapping[str, object]],
    config: S4StaticPromotionConfig = S4StaticPromotionConfig(),
):
    """Promote S4 using fixed-test metrics with an explicit leakage label."""

    if len(s3_fixed_test_metrics) != len(s4_fixed_test_metrics):
        raise ValueError("S3 and S4 promotion fold counts differ")
    s3 = _summarize_static_metrics(s3_fixed_test_metrics)
    s4 = _summarize_static_metrics(s4_fixed_test_metrics)
    site_difference = (
        None if s3["mean_site_auc"] is None or s4["mean_site_auc"] is None
        else float(s4["mean_site_auc"] - s3["mean_site_auc"])
    )
    differences = {
        "mean_auc_gain": float(s4["mean_roc_auc"] - s3["mean_roc_auc"]),
        "worst_fold_auc_gain": float(
            s4["worst_fold_roc_auc"] - s3["worst_fold_roc_auc"]
        ),
        "auc_std_change": float(s4["std_roc_auc"] - s3["std_roc_auc"]),
        "mean_accuracy_gain": float(s4["mean_accuracy"] - s3["mean_accuracy"]),
        "mean_auprc_gain": float(s4["mean_auprc"] - s3["mean_auprc"]),
        "mean_site_auc_gain": site_difference,
    }
    checks = {
        "minimum_mean_auc_gain": (
            differences["mean_auc_gain"] >= config.minimum_mean_auc_gain
        ),
        "maximum_worst_fold_auc_drop": (
            differences["worst_fold_auc_gain"]
            >= -config.maximum_worst_fold_auc_drop
        ),
        "maximum_auc_std_increase": (
            differences["auc_std_change"] <= config.maximum_auc_std_increase
        ),
        "maximum_mean_accuracy_drop": (
            differences["mean_accuracy_gain"] >= -config.maximum_mean_accuracy_drop
        ),
        "maximum_mean_auprc_drop": (
            differences["mean_auprc_gain"] >= -config.maximum_mean_auprc_drop
        ),
        "maximum_mean_site_auc_drop": (
            site_difference is None
            or site_difference >= -config.maximum_mean_site_auc_drop
        ),
    }
    promoted = bool(all(checks.values()))
    return {
        "artifact_type": "mokse_s4_test_guided_static_promotion_v1",
        "selection_data": "fixed_test",
        "test_guided_architecture_selection": True,
        "unbiased_generalization_estimate": False,
        "test_used_to_fit_model_weights": False,
        "test_used_to_fit_fusion_beta": False,
        "config": asdict(config),
        "s3": s3,
        "s4": s4,
        "differences": differences,
        "acceptance_checks": checks,
        "s4_promoted": promoted,
        "selected_static_stage": "s4" if promoted else "s3",
    }


def _validate_oof_fold(payload):
    required = (
        "sample_keys", "sites", "labels", "subgraph_logits",
        "static_scores", "static_uncertainty",
    )
    if any(name not in payload for name in required):
        raise ValueError("S4 anchored fusion fold is incomplete")
    if payload.get("prediction_role") != "development_oof":
        raise ValueError("S4 fusion selection requires development-OOF folds")
    result = {
        "sample_keys": _vector(payload["sample_keys"], str, "sample_keys"),
        "sites": _vector(payload["sites"], str, "sites"),
        "labels": _vector(payload["labels"], np.int64, "labels"),
        "subgraph_logits": _vector(
            payload["subgraph_logits"], np.float64, "subgraph_logits"
        ),
        "static_scores": _vector(
            payload["static_scores"], np.float64, "static_scores"
        ),
        "static_uncertainty": _vector(
            payload["static_uncertainty"], np.float64, "static_uncertainty"
        ),
    }
    count = result["labels"].size
    if any(value.size != count for value in result.values()):
        raise ValueError("S4 anchored fusion fold arrays do not align")
    if set(result["labels"].tolist()) != {0, 1}:
        raise ValueError("S4 anchored fusion fold must contain both classes")
    if len(set(result["sample_keys"].tolist())) != count:
        raise ValueError("S4 anchored fusion fold sample keys must be unique")
    if not all(np.isfinite(result[name]).all() for name in (
        "subgraph_logits", "static_scores", "static_uncertainty"
    )):
        raise ValueError("S4 anchored fusion inputs must be finite")
    if np.any(result["static_uncertainty"] < 0.0):
        raise ValueError("S4 static uncertainty cannot be negative")
    return result


def _fit_complement_calibration(folds, epsilon):
    subgraph = np.concatenate([fold["subgraph_logits"] for fold in folds])
    static = np.concatenate([fold["static_scores"] for fold in folds])
    uncertainty = np.concatenate([fold["static_uncertainty"] for fold in folds])
    subgraph_mean = float(np.mean(subgraph))
    subgraph_scale = float(np.std(subgraph))
    static_mean = float(np.mean(static))
    static_scale = float(np.std(static))
    if min(subgraph_scale, static_scale) <= epsilon:
        raise ValueError("S4 calibration received a degenerate score channel")
    s = (subgraph - subgraph_mean) / subgraph_scale
    b = (static - static_mean) / static_scale
    eta = float(np.mean(b * s) / (np.mean(s * s) + epsilon))
    residual = b - eta * s
    residual_scale = float(np.std(residual))
    if residual_scale <= epsilon:
        raise ValueError("S4 complementary residual is degenerate")
    tau = float(np.median(uncertainty))
    if tau <= epsilon:
        reliability = np.where(uncertainty <= epsilon, 1.0, 0.0)
    else:
        reliability = 1.0 / (1.0 + uncertainty / (tau + epsilon))
    correction = reliability * residual / residual_scale
    return {
        "subgraph_mean": subgraph_mean,
        "subgraph_scale": subgraph_scale,
        "static_mean": static_mean,
        "static_scale": static_scale,
        "eta": eta,
        "residual_scale": residual_scale,
        "uncertainty_tau": tau,
        # Train/OOF centering prevents a systematic intercept shift at ACC@0.5.
        "correction_mean": float(np.mean(correction)),
        "epsilon": float(epsilon),
        "centering_source": "development_oof",
    }


def _apply_calibration(calibration, subgraph_logits, static_scores, uncertainty):
    subgraph = np.asarray(subgraph_logits, dtype=np.float64)
    static = np.asarray(static_scores, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if subgraph.shape != static.shape or subgraph.shape != uncertainty.shape:
        raise ValueError("S4 fusion application arrays do not align")
    epsilon = float(calibration["epsilon"])
    s = (
        subgraph - float(calibration["subgraph_mean"])
    ) / float(calibration["subgraph_scale"])
    b = (
        static - float(calibration["static_mean"])
    ) / float(calibration["static_scale"])
    residual = b - float(calibration["eta"]) * s
    tau = float(calibration["uncertainty_tau"])
    if tau <= epsilon:
        reliability = np.where(uncertainty <= epsilon, 1.0, 0.0)
    else:
        reliability = 1.0 / (1.0 + uncertainty / (tau + epsilon))
    correction = reliability * residual / float(calibration["residual_scale"])
    correction = correction - float(calibration["correction_mean"])
    return correction, reliability, residual


def apply_s4_anchored_fusion(
    selection: Mapping[str, object],
    subgraph_logits,
    static_scores,
    static_uncertainty,
):
    if selection.get("artifact_type") != "mokse_s4_anchored_fusion_selection_v1":
        raise ValueError("S4 anchored fusion artifact type mismatch")
    subgraph = np.asarray(subgraph_logits, dtype=np.float64)
    beta = float(selection["selected_beta"])
    if beta == 0.0:
        _, reliability, residual = _apply_calibration(
            selection["final_calibration"],
            subgraph,
            static_scores,
            static_uncertainty,
        )
        return {
            "fused_logits": subgraph.copy(),
            "correction": np.zeros_like(subgraph),
            "reliability": reliability,
            "static_residual": residual,
        }
    correction, reliability, residual = _apply_calibration(
        selection["final_calibration"],
        subgraph,
        static_scores,
        static_uncertainty,
    )
    scaled = (
        beta
        * float(selection["final_calibration"]["subgraph_scale"])
        * correction
    )
    return {
        "fused_logits": subgraph + scaled,
        "correction": scaled,
        "reliability": reliability,
        "static_residual": residual,
    }


def _mean_optional(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _candidate(beta, folds, calibrations, config):
    rows = []
    all_labels = []
    all_subgraph = []
    all_fused = []
    for index, (fold, calibration) in enumerate(zip(folds, calibrations)):
        if float(beta) == 0.0:
            fused = fold["subgraph_logits"].copy()
        else:
            correction, _, _ = _apply_calibration(
                calibration,
                fold["subgraph_logits"],
                fold["static_scores"],
                fold["static_uncertainty"],
            )
            fused = fold["subgraph_logits"] + (
                float(beta) * float(calibration["subgraph_scale"]) * correction
            )
        baseline = score_logits(
            fold["labels"], fold["sites"], fold["subgraph_logits"]
        )
        metrics = score_logits(fold["labels"], fold["sites"], fused)
        rows.append({
            "fold": index,
            "roc_auc": float(metrics["roc_auc"]),
            "baseline_roc_auc": float(baseline["roc_auc"]),
            "auc_difference": float(metrics["roc_auc"] - baseline["roc_auc"]),
            "accuracy": float(metrics["accuracy"]),
            "baseline_accuracy": float(baseline["accuracy"]),
            "accuracy_difference": float(metrics["accuracy"] - baseline["accuracy"]),
            "auprc": float(metrics["auprc"]),
            "baseline_auprc": float(baseline["auprc"]),
            "auprc_difference": float(metrics["auprc"] - baseline["auprc"]),
            "site_auc": metrics["site_stratified_roc_auc"],
            "baseline_site_auc": baseline["site_stratified_roc_auc"],
        })
        all_labels.append(fold["labels"])
        all_subgraph.append(fold["subgraph_logits"])
        all_fused.append(fused)
    aucs = np.asarray([row["roc_auc"] for row in rows], dtype=np.float64)
    auc_differences = np.asarray(
        [row["auc_difference"] for row in rows], dtype=np.float64
    )
    site_differences = [
        row["site_auc"] - row["baseline_site_auc"]
        for row in rows
        if row["site_auc"] is not None and row["baseline_site_auc"] is not None
    ]
    pairs = ranking_pair_changes(
        np.concatenate(all_labels),
        np.concatenate(all_subgraph),
        np.concatenate(all_fused),
    )
    result = {
        "beta": float(beta),
        "mean_fold_roc_auc": float(np.mean(aucs)),
        "std_fold_roc_auc": float(np.std(aucs)),
        "stability_objective": float(
            np.mean(aucs) - config.stability_penalty * np.std(aucs)
        ),
        "mean_auc_gain": float(np.mean(auc_differences)),
        "worst_fold_auc_difference": float(np.min(auc_differences)),
        "non_decreasing_fold_count": int(np.sum(auc_differences >= -1.0e-12)),
        "mean_accuracy_difference": float(np.mean([
            row["accuracy_difference"] for row in rows
        ])),
        "mean_auprc_difference": float(np.mean([
            row["auprc_difference"] for row in rows
        ])),
        "mean_site_auc_difference": _mean_optional(site_differences),
        "ranking_pairs": pairs,
        "folds": rows,
    }
    checks = {
        "positive_beta": float(beta) > 0.0,
        "minimum_mean_auc_gain": result["mean_auc_gain"] >= config.minimum_mean_auc_gain,
        "maximum_worst_fold_auc_drop": (
            result["worst_fold_auc_difference"] >= -config.maximum_worst_fold_auc_drop
        ),
        "minimum_non_decreasing_folds": (
            result["non_decreasing_fold_count"] >= config.minimum_non_decreasing_folds
        ),
        "maximum_mean_accuracy_drop": (
            result["mean_accuracy_difference"] >= -config.maximum_mean_accuracy_drop
        ),
        "maximum_mean_auprc_drop": (
            result["mean_auprc_difference"] >= -config.maximum_mean_auprc_drop
        ),
        "maximum_site_auc_drop": (
            result["mean_site_auc_difference"] is None
            or result["mean_site_auc_difference"] >= -config.maximum_site_auc_drop
        ),
    }
    if config.dataset.lower() == "wmrc":
        checks["positive_net_corrected_pairs"] = (
            pairs["net_corrected_pair_count"] > 0
        )
    result["acceptance_checks"] = checks
    result["accepted"] = bool(all(checks.values()))
    return result


def select_s4_anchored_fusion(
    development_oof_folds: Sequence[Mapping[str, object]],
    config: S4AnchoredFusionConfig,
):
    """Select shared beta with leave-one-fold-out calibration and exact fallback."""

    folds = [_validate_oof_fold(fold) for fold in development_oof_folds]
    if len(folds) < max(2, config.minimum_non_decreasing_folds):
        raise ValueError("not enough S4 development-OOF folds")
    seen = set()
    for fold in folds:
        current = set(fold["sample_keys"].tolist())
        if seen.intersection(current):
            raise ValueError("S4 development-OOF folds overlap")
        seen.update(current)
    calibrations = [
        _fit_complement_calibration(
            [other for other_index, other in enumerate(folds) if other_index != index],
            config.epsilon,
        )
        for index in range(len(folds))
    ]
    candidates = [
        _candidate(beta, folds, calibrations, config)
        for beta in sorted(set(float(value) for value in config.beta_grid))
    ]
    eligible = [row for row in candidates if row["accepted"]]
    if eligible:
        best_objective = max(row["stability_objective"] for row in eligible)
        near_best = [
            row for row in eligible
            if row["stability_objective"] >= best_objective - config.near_best_tolerance
        ]
        selected = min(near_best, key=lambda row: row["beta"])
        selected_beta = float(selected["beta"])
        source = "anchored_static_complement"
        fallback_reason = None
    else:
        selected = next(row for row in candidates if row["beta"] == 0.0)
        selected_beta = 0.0
        source = "subgraph_exact_fallback"
        fallback_reason = "no positive beta passed all development-OOF no-harm checks"
    final_calibration = _fit_complement_calibration(folds, config.epsilon)
    return {
        "artifact_type": "mokse_s4_anchored_fusion_selection_v1",
        "config": asdict(config),
        "development_oof_sample_count": len(seen),
        "development_oof_fold_count": len(folds),
        "calibration_mode": "leave_one_fold_out_for_selection_then_all_oof_for_application",
        "leave_one_fold_out_calibrations": calibrations,
        "selected_source": source,
        "selected_beta": selected_beta,
        "selected_candidate": selected,
        "fallback_reason": fallback_reason,
        "final_calibration": final_calibration,
        "candidates": candidates,
        "fixed_test_used_for_selection": False,
        "decision_threshold": 0.5,
    }
