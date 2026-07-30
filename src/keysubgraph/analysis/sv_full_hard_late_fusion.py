"""Frozen validation-selected late fusion of full and hard graph channels."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import log_loss

from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)


DEFAULT_FULL_HARD_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _logit(values: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _prediction_lookup(
    payload: Mapping[str, Any], name: str, expected_split: str
) -> Dict[str, Tuple[int, str, float]]:
    if payload.get("artifact_type") != (
        "sv_hard_sgw_signed_gin_evaluation"
    ):
        raise ValueError("{} is not an SV evaluation".format(name))
    if payload.get("split") != expected_split:
        raise ValueError("{} has the wrong split".format(name))
    rows = payload.get("predictions", ())
    lookup = {}
    for row in rows:
        key = str(row["sample_key"])
        if key in lookup:
            raise ValueError("{} has duplicate samples".format(name))
        probability = float(row["positive_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("{} has invalid probabilities".format(name))
        lookup[key] = (
            int(row["label"]),
            str(row.get("site", "")),
            probability,
        )
    if not lookup:
        raise ValueError("{} has no predictions".format(name))
    return lookup


def _validate_branch_pair(
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    branch: str,
    expected_mode: str,
) -> None:
    if validation.get("variant") != test.get("variant"):
        raise ValueError("{} branch variants disagree".format(branch))
    for field in ("checkpoint_sha256", "scaler_sha256"):
        if validation.get(field) != test.get(field):
            raise ValueError(
                "{} branch {} disagrees".format(branch, field)
            )
    validation_provenance = validation.get("provenance")
    test_provenance = test.get("provenance")
    if not isinstance(validation_provenance, dict) or not isinstance(
        test_provenance, dict
    ):
        raise ValueError(
            "{} branch evaluation lacks provenance".format(branch)
        )
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
        "training_seed",
    )
    if any(
        validation_provenance.get(key) != test_provenance.get(key)
        for key in keys
    ):
        raise ValueError(
            "{} validation/test provenance disagrees".format(branch)
        )
    if validation_provenance.get("selection_mode") != expected_mode:
        raise ValueError(
            "{} branch must use {} selection".format(
                branch, expected_mode
            )
        )
    validation_lookup = _prediction_lookup(
        validation, branch + "_validation", "validation"
    )
    test_lookup = _prediction_lookup(test, branch + "_test", "test")
    if set(validation_lookup) & set(test_lookup):
        raise ValueError("{} validation and test overlap".format(branch))


def _align_branches(
    hard: Mapping[str, Any],
    full: Mapping[str, Any],
    split: str,
) -> Dict[str, Any]:
    hard_lookup = _prediction_lookup(hard, "hard_" + split, split)
    full_lookup = _prediction_lookup(full, "full_" + split, split)
    if set(hard_lookup) != set(full_lookup):
        raise ValueError(
            "{} full/hard samples do not align".format(split)
        )
    keys = sorted(hard_lookup)
    labels = []
    sites = []
    hard_probabilities = []
    full_probabilities = []
    for key in keys:
        hard_label, hard_site, hard_probability = hard_lookup[key]
        full_label, full_site, full_probability = full_lookup[key]
        if hard_label != full_label or hard_site != full_site:
            raise ValueError(
                "{} full/hard identities disagree".format(split)
            )
        labels.append(hard_label)
        sites.append(hard_site)
        hard_probabilities.append(hard_probability)
        full_probabilities.append(full_probability)
    if set(labels) != {0, 1}:
        raise ValueError("{} requires both classes".format(split))
    return {
        "sample_keys": keys,
        "labels": labels,
        "sites": sites,
        "hard_probabilities": np.asarray(
            hard_probabilities, dtype=np.float64
        ),
        "full_probabilities": np.asarray(
            full_probabilities, dtype=np.float64
        ),
    }


def _normalization(
    validation: Mapping[str, Any], epsilon: float
) -> Dict[str, Dict[str, float]]:
    result = {}
    for name in ("hard", "full"):
        values = _logit(
            validation[name + "_probabilities"], epsilon
        )
        standard_deviation = float(values.std())
        if not math.isfinite(standard_deviation) or (
            standard_deviation <= 1.0e-12
        ):
            raise ValueError(
                "{} validation logits have zero variance".format(name)
            )
        result[name] = {
            "fit_split": "validation",
            "mean": float(values.mean()),
            "standard_deviation": standard_deviation,
        }
    return result


def _standardized_logits(
    partition: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
    epsilon: float,
) -> Dict[str, np.ndarray]:
    result = {}
    for name in ("hard", "full"):
        values = _logit(
            partition[name + "_probabilities"], epsilon
        )
        result[name] = (
            values - float(normalization[name]["mean"])
        ) / float(normalization[name]["standard_deviation"])
    return result


def _fuse(
    standardized: Mapping[str, np.ndarray], hard_weight: float
) -> np.ndarray:
    return _sigmoid(
        hard_weight * standardized["hard"]
        + (1.0 - hard_weight) * standardized["full"]
    )


def _metrics_with_sites(
    labels: Sequence[int],
    probabilities: Sequence[float],
    sites: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    metrics = binary_metrics(
        list(labels), list(probabilities), float(threshold)
    )
    metrics["site_stratified_roc_auc"] = site_stratified_roc_auc(
        list(labels), list(probabilities), list(sites)
    )
    return metrics


def build_sv_full_hard_late_fusion(
    hard_validation: Mapping[str, Any],
    hard_test: Mapping[str, Any],
    full_validation: Mapping[str, Any],
    full_test: Mapping[str, Any],
    alpha_grid: Sequence[float] = DEFAULT_FULL_HARD_ALPHA_GRID,
    epsilon: float = 1.0e-6,
) -> Dict[str, Any]:
    """Select the hard-channel weight on validation and freeze it to test."""
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("fusion epsilon must lie in (0,0.5)")
    grid = sorted(set(float(value) for value in alpha_grid))
    if not grid or any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("fusion alpha grid must lie in [0,1]")
    if 0.0 not in grid or 0.5 not in grid or 1.0 not in grid:
        raise ValueError("fusion grid must contain 0, 0.5 and 1")
    _validate_branch_pair(
        hard_validation, hard_test, "hard", "learned"
    )
    _validate_branch_pair(
        full_validation, full_test, "full", "full"
    )
    if hard_validation["variant"] != full_validation["variant"]:
        raise ValueError("full and hard downstream variants must match")
    hard_provenance = hard_validation["provenance"]
    full_provenance = full_validation["provenance"]
    if (
        hard_provenance["protocol_sha256"]
        != full_provenance["protocol_sha256"]
        or int(hard_provenance["selection_seed"])
        != int(full_provenance["selection_seed"])
        or int(hard_provenance["training_seed"])
        != int(full_provenance["training_seed"])
    ):
        raise ValueError(
            "full and hard branches do not share protocol/seeds"
        )
    validation = _align_branches(
        hard_validation, full_validation, "validation"
    )
    test = _align_branches(hard_test, full_test, "test")
    normalization = _normalization(validation, epsilon)
    validation_logits = _standardized_logits(
        validation, normalization, epsilon
    )
    test_logits = _standardized_logits(test, normalization, epsilon)
    candidates = []
    for hard_weight in grid:
        probabilities = _fuse(validation_logits, hard_weight)
        threshold = fit_binary_threshold(
            validation["labels"],
            probabilities.tolist(),
            "balanced_accuracy",
        )
        metrics = _metrics_with_sites(
            validation["labels"],
            probabilities.tolist(),
            validation["sites"],
            threshold,
        )
        candidates.append(
            {
                "hard_weight": hard_weight,
                "full_weight": 1.0 - hard_weight,
                "validation_roc_auc": metrics["roc_auc"],
                "validation_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "validation_log_loss": float(
                    log_loss(
                        validation["labels"],
                        probabilities,
                        labels=[0, 1],
                    )
                ),
                "validation_threshold": threshold,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["validation_roc_auc"]),
            -float(row["validation_log_loss"]),
            -abs(float(row["hard_weight"]) - 0.5),
        ),
    )
    selected_weight = float(selected["hard_weight"])
    validation_probabilities = _fuse(
        validation_logits, selected_weight
    )
    test_probabilities = _fuse(test_logits, selected_weight)
    thresholds = {
        policy: fit_binary_threshold(
            validation["labels"],
            validation_probabilities.tolist(),
            policy,
        )
        for policy in ("balanced_accuracy", "accuracy")
    }

    def partition_payload(partition, probabilities):
        metrics = {
            policy: _metrics_with_sites(
                partition["labels"],
                probabilities.tolist(),
                partition["sites"],
                threshold,
            )
            for policy, threshold in thresholds.items()
        }
        predictions = []
        for index, key in enumerate(partition["sample_keys"]):
            row = {
                "sample_key": key,
                "site": partition["sites"][index],
                "label": int(partition["labels"][index]),
                "hard_probability": float(
                    partition["hard_probabilities"][index]
                ),
                "full_probability": float(
                    partition["full_probabilities"][index]
                ),
                "fused_probability": float(probabilities[index]),
            }
            for policy, threshold in thresholds.items():
                row[policy + "_prediction"] = int(
                    probabilities[index] >= threshold
                )
            predictions.append(row)
        return {"metrics": metrics, "predictions": predictions}

    equal_validation = _fuse(validation_logits, 0.5)
    return {
        "schema_version": 1,
        "artifact_type": "sv_full_hard_frozen_late_fusion",
        "variant": hard_validation["variant"],
        "test_used_for_selection": False,
        "weight_fit_split": "validation",
        "normalization_fit_split": "validation",
        "threshold_fit_split": "validation",
        "updated_parameter_count": 0,
        "selection_metric": "validation_roc_auc",
        "tie_breakers": (
            "validation_log_loss_then_weight_closest_to_equal"
        ),
        "alpha_semantics": (
            "fused_logit=hard_weight*hard_zlogit+"
            "(1-hard_weight)*full_zlogit"
        ),
        "alpha_grid": grid,
        "selected_hard_weight": selected_weight,
        "selected_full_weight": 1.0 - selected_weight,
        "normalization": normalization,
        "thresholds": thresholds,
        "candidates": candidates,
        "standalone_validation_auc": {
            "hard": binary_metrics(
                validation["labels"],
                validation["hard_probabilities"].tolist(),
                0.5,
            )["roc_auc"],
            "full": binary_metrics(
                validation["labels"],
                validation["full_probabilities"].tolist(),
                0.5,
            )["roc_auc"],
            "equal_fusion": binary_metrics(
                validation["labels"],
                equal_validation.tolist(),
                0.5,
            )["roc_auc"],
        },
        "validation": partition_payload(
            validation, validation_probabilities
        ),
        "test": partition_payload(test, test_probabilities),
        "provenance": {
            "protocol_sha256": hard_provenance["protocol_sha256"],
            "selection_seed": int(
                hard_provenance["selection_seed"]
            ),
            "hard_checkpoint_sha256": hard_validation[
                "checkpoint_sha256"
            ],
            "full_checkpoint_sha256": full_validation[
                "checkpoint_sha256"
            ],
            "hard_selector_checkpoint_sha256": hard_provenance[
                "selector_checkpoint_sha256"
            ],
            "full_selection_mode": full_provenance[
                "selection_mode"
            ],
        },
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# SV Full Graph + Hard Graph 冻结晚期融合",
        "",
        "- 下游变体：`{}`".format(payload["variant"]),
        "- 权重拟合集：validation",
        "- 阈值拟合集：validation",
        "- Test 参与模型选择：否",
        "- 更新参数量：0",
        "- Hard 权重：{:.2f}".format(
            payload["selected_hard_weight"]
        ),
        "- Full 权重：{:.2f}".format(
            payload["selected_full_weight"]
        ),
        "",
        "## Validation 权重搜索",
        "",
        "| Hard权重 | Full权重 | AUC | BA | Log-loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "| {hard_weight:.2f} | {full_weight:.2f} | "
            "{validation_roc_auc:.6f} | "
            "{validation_balanced_accuracy:.6f} | "
            "{validation_log_loss:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 冻结权重结果",
            "",
            "| 阈值策略 | Split | AUC | Site-AUC | BA | "
            "Accuracy | F1 | Threshold |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in ("balanced_accuracy", "accuracy"):
        for split in ("validation", "test"):
            metrics = payload[split]["metrics"][policy]
            site = metrics["site_stratified_roc_auc"]
            lines.append(
                "| {} | {} | {:.6f} | {} | {:.6f} | {:.6f} | "
                "{:.6f} | {:.6f} |".format(
                    policy,
                    split,
                    metrics["roc_auc"],
                    "N/A" if site is None else "{:.6f}".format(site),
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                    metrics["f1"],
                    metrics["threshold"],
                )
            )
    lines.extend(
        [
            "",
            "> Full 与 Hard 通道独立训练；仅在 validation 上选择融合"
            "权重和阈值，然后原样冻结到 test。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def write_sv_full_hard_late_fusion(
    payload: Mapping[str, Any], output_dir: Path
) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("SV full-hard fusion output exists")
    output_dir.mkdir(parents=True)
    paths = {
        "evaluation": output_dir / "evaluation.json",
        "model_spec": output_dir / "model_spec.json",
        "summary": output_dir / "summary.md",
        "validation_predictions": output_dir
        / "validation_predictions.csv",
        "test_predictions": output_dir / "test_predictions.csv",
    }
    _atomic_json(paths["evaluation"], payload)
    _atomic_json(
        paths["model_spec"],
        {
            key: payload[key]
            for key in (
                "schema_version",
                "artifact_type",
                "variant",
                "test_used_for_selection",
                "weight_fit_split",
                "normalization_fit_split",
                "threshold_fit_split",
                "updated_parameter_count",
                "selection_metric",
                "alpha_semantics",
                "alpha_grid",
                "selected_hard_weight",
                "selected_full_weight",
                "normalization",
                "thresholds",
                "provenance",
            )
        },
    )
    _write_csv(
        paths["validation_predictions"],
        payload["validation"]["predictions"],
    )
    _write_csv(paths["test_predictions"], payload["test"]["predictions"])
    temporary = paths["summary"].with_suffix(".md.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_markdown(payload))
    os.replace(str(temporary), str(paths["summary"]))
    return {name: str(path) for name, path in paths.items()}
