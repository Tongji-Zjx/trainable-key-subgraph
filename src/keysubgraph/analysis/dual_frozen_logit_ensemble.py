"""Parameter-free equal-weight logit ensembles for frozen D3 paths."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)


def _validate_component(
    name: str, partition: Mapping[str, Sequence[Any]]
) -> Dict[str, Any]:
    keys = [str(value) for value in partition["sample_keys"]]
    labels = [int(value) for value in partition["labels"]]
    probabilities = [
        float(value) for value in partition["probabilities"]
    ]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(
            "{} ensemble keys are empty or duplicated".format(name)
        )
    if len(labels) != len(keys) or len(probabilities) != len(keys):
        raise ValueError("{} ensemble predictions are misaligned".format(name))
    if set(labels) != {0, 1}:
        raise ValueError("{} ensemble requires both classes".format(name))
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ):
        raise ValueError(
            "{} ensemble probabilities are invalid".format(name)
        )
    return {
        key: (label, probability)
        for key, label, probability in zip(keys, labels, probabilities)
    }


def _align_components(
    components: Mapping[str, Mapping[str, Sequence[Any]]],
    split: str,
) -> Dict[str, Any]:
    if len(components) < 2:
        raise ValueError("a frozen ensemble requires at least two components")
    names = sorted(str(name) for name in components)
    if len(set(names)) != len(names):
        raise ValueError("frozen ensemble component names are duplicated")
    lookups = {
        name: _validate_component(
            "{}:{}".format(split, name), components[name]
        )
        for name in names
    }
    sample_keys = sorted(lookups[names[0]])
    expected = set(sample_keys)
    for name in names[1:]:
        if set(lookups[name]) != expected:
            raise ValueError(
                "{} ensemble components cover different samples".format(split)
            )
    labels = []
    probabilities = {name: [] for name in names}
    for sample_key in sample_keys:
        component_labels = [
            lookups[name][sample_key][0] for name in names
        ]
        if len(set(component_labels)) != 1:
            raise ValueError(
                "{} ensemble components disagree on labels".format(split)
            )
        labels.append(component_labels[0])
        for name in names:
            probabilities[name].append(lookups[name][sample_key][1])
    return {
        "component_names": names,
        "sample_keys": sample_keys,
        "labels": labels,
        "probabilities": probabilities,
    }


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _fuse(
    aligned: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
    epsilon: float,
) -> Dict[str, Any]:
    standardized = []
    for name in aligned["component_names"]:
        values = _logit(
            np.asarray(aligned["probabilities"][name], dtype=np.float64),
            epsilon,
        )
        statistics = normalization[name]
        standardized.append(
            (values - float(statistics["mean"]))
            / float(statistics["standard_deviation"])
        )
    fused_logit = np.mean(np.stack(standardized, axis=0), axis=0)
    fused_probability = _sigmoid(fused_logit)
    return {
        "logits": fused_logit.tolist(),
        "probabilities": fused_probability.tolist(),
    }


def build_frozen_equal_logit_ensemble(
    validation_components: Mapping[
        str, Mapping[str, Sequence[Any]]
    ],
    test_components: Mapping[str, Mapping[str, Sequence[Any]]],
    ensemble_scope: str,
    epsilon: float = 1.0e-6,
) -> Dict[str, Any]:
    """Standardize component logits on validation and average equally."""
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("frozen ensemble epsilon must be in (0,0.5)")
    validation = _align_components(validation_components, "validation")
    test = _align_components(test_components, "test")
    if validation["component_names"] != test["component_names"]:
        raise ValueError(
            "validation and test ensemble components disagree"
        )
    if set(validation["sample_keys"]) & set(test["sample_keys"]):
        raise ValueError("frozen ensemble validation and test overlap")
    normalization = {}
    for name in validation["component_names"]:
        values = _logit(
            np.asarray(
                validation["probabilities"][name], dtype=np.float64
            ),
            epsilon,
        )
        standard_deviation = float(values.std())
        if not math.isfinite(standard_deviation) or (
            standard_deviation <= 1.0e-12
        ):
            raise ValueError(
                "{} validation logits have zero variance".format(name)
            )
        normalization[name] = {
            "fit_split": "validation",
            "mean": float(values.mean()),
            "standard_deviation": standard_deviation,
        }
    validation_fused = _fuse(validation, normalization, epsilon)
    test_fused = _fuse(test, normalization, epsilon)
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            validation["labels"],
            validation_fused["probabilities"],
            "balanced_accuracy",
        ),
        "accuracy": fit_binary_threshold(
            validation["labels"],
            validation_fused["probabilities"],
            "accuracy",
        ),
    }

    def partition_payload(aligned, fused):
        metrics = {
            name: binary_metrics(
                aligned["labels"],
                fused["probabilities"],
                threshold,
            )
            for name, threshold in thresholds.items()
        }
        predictions = []
        for index, sample_key in enumerate(aligned["sample_keys"]):
            row = {
                "sample_key": sample_key,
                "label": int(aligned["labels"][index]),
                "fused_logit": float(fused["logits"][index]),
                "positive_probability": float(
                    fused["probabilities"][index]
                ),
            }
            for component in aligned["component_names"]:
                row["{}_probability".format(component)] = float(
                    aligned["probabilities"][component][index]
                )
            for policy, threshold in thresholds.items():
                row["{}_prediction".format(policy)] = int(
                    fused["probabilities"][index] >= threshold
                )
            predictions.append(row)
        return {"metrics": metrics, "predictions": predictions}

    return {
        "artifact": "dual_d3_frozen_equal_logit_ensemble",
        "schema_version": 1,
        "ensemble_scope": str(ensemble_scope),
        "component_names": validation["component_names"],
        "component_count": len(validation["component_names"]),
        "weight_per_component": 1.0
        / float(len(validation["component_names"])),
        "normalization": normalization,
        "normalization_fit_split": "validation",
        "updated_parameter_count": 0,
        "threshold_fit_split": "validation",
        "thresholds": thresholds,
        "primary_threshold_policy": "balanced_accuracy",
        "primary_ranking_metric": "roc_auc",
        "validation": partition_payload(validation, validation_fused),
        "test": partition_payload(test, test_fused),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty frozen ensemble predictions")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(evaluation: Mapping[str, Any]) -> str:
    lines = [
        "# D3 冻结等权 Logit 集成",
        "",
        "- 集成范围：{}".format(evaluation["ensemble_scope"]),
        "- 组成数量：{}".format(evaluation["component_count"]),
        "- 单项权重：{:.6f}".format(
            evaluation["weight_per_component"]
        ),
        "- 标准化拟合集：validation",
        "- 阈值拟合集：validation",
        "- 更新参数量：0",
        "",
        "| 阈值策略 | Split | AUROC | BA | Accuracy | F1 | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ("balanced_accuracy", "accuracy"):
        for split in ("validation", "test"):
            metrics = evaluation[split]["metrics"][policy]
            lines.append(
                "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | "
                "{:.6f} |".format(
                    policy,
                    split,
                    metrics["roc_auc"],
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                    metrics["f1"],
                    metrics["threshold"],
                )
            )
    lines.extend(
        [
            "",
            "> 权重固定等分；标准化与阈值仅使用 validation，test 未参与选择。",
            "",
        ]
    )
    return "\n".join(lines)


def write_frozen_equal_logit_ensemble_artifacts(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    """Write an immutable ensemble evaluation bundle."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output_dir / "evaluation.json",
        "model_spec": output_dir / "model_spec.json",
        "summary": output_dir / "summary.md",
        "validation_predictions": output_dir
        / "validation_predictions.csv",
        "test_predictions": output_dir / "test_predictions.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("frozen ensemble artifacts already exist")
    payload = dict(evaluation)
    payload["provenance"] = dict(provenance)
    _atomic_json(paths["evaluation"], payload)
    _atomic_json(
        paths["model_spec"],
        {
            "artifact": "dual_d3_frozen_equal_logit_ensemble_spec",
            "schema_version": 1,
            "ensemble_scope": evaluation["ensemble_scope"],
            "component_names": evaluation["component_names"],
            "weight_per_component": evaluation["weight_per_component"],
            "normalization": evaluation["normalization"],
            "primary_threshold_policy": evaluation[
                "primary_threshold_policy"
            ],
            "frozen_threshold": evaluation["thresholds"][
                evaluation["primary_threshold_policy"]
            ],
            "updated_parameter_count": 0,
            "provenance": dict(provenance),
        },
    )
    _write_csv(
        paths["validation_predictions"],
        evaluation["validation"]["predictions"],
    )
    _write_csv(
        paths["test_predictions"], evaluation["test"]["predictions"]
    )
    temporary = paths["summary"].with_suffix(".md.tmp")
    temporary.write_text(_markdown(evaluation) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(paths["summary"]))
    return paths
