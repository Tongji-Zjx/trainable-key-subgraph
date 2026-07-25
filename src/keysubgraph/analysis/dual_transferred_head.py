"""Frozen-head evaluation for cached Exact-SGW representations."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)


def _validate_partition(
    name: str,
    sample_keys: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> Dict[str, Any]:
    keys = [str(value) for value in sample_keys]
    labels_list = [int(value) for value in labels]
    probabilities_list = [float(value) for value in probabilities]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(
            "{} transferred-head keys are empty or duplicated".format(name)
        )
    if len(labels_list) != len(keys) or len(probabilities_list) != len(keys):
        raise ValueError(
            "{} transferred-head predictions are misaligned".format(name)
        )
    if set(labels_list) != {0, 1}:
        raise ValueError(
            "{} transferred-head data must contain both classes".format(name)
        )
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities_list
    ):
        raise ValueError(
            "{} transferred-head probabilities are invalid".format(name)
        )
    return {
        "sample_keys": keys,
        "labels": labels_list,
        "probabilities": probabilities_list,
    }


def _prediction_rows(
    partition: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> Sequence[Dict[str, Any]]:
    rows = []
    for key, label, probability in zip(
        partition["sample_keys"],
        partition["labels"],
        partition["probabilities"],
    ):
        row = {
            "sample_key": key,
            "label": int(label),
            "positive_probability": float(probability),
        }
        for name, threshold in thresholds.items():
            row["{}_prediction".format(name)] = int(
                probability >= threshold
            )
        rows.append(row)
    return rows


def build_transferred_head_evaluation(
    validation_sample_keys: Sequence[str],
    validation_labels: Sequence[int],
    validation_probabilities: Sequence[float],
    test_sample_keys: Sequence[str],
    test_labels: Sequence[int],
    test_probabilities: Sequence[float],
    original_proxy_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit thresholds on validation and freeze them for test evaluation."""
    validation = _validate_partition(
        "validation",
        validation_sample_keys,
        validation_labels,
        validation_probabilities,
    )
    test = _validate_partition(
        "test",
        test_sample_keys,
        test_labels,
        test_probabilities,
    )
    if set(validation["sample_keys"]) & set(test["sample_keys"]):
        raise ValueError("transferred-head validation and test overlap")
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            validation["labels"],
            validation["probabilities"],
            "balanced_accuracy",
        ),
        "accuracy": fit_binary_threshold(
            validation["labels"],
            validation["probabilities"],
            "accuracy",
        ),
    }
    if original_proxy_threshold is not None:
        threshold = float(original_proxy_threshold)
        if not np.isfinite(threshold):
            raise ValueError("original proxy threshold is non-finite")
        thresholds["original_proxy"] = threshold
    validation_metrics = {
        name: binary_metrics(
            validation["labels"],
            validation["probabilities"],
            threshold,
        )
        for name, threshold in thresholds.items()
    }
    test_metrics = {
        name: binary_metrics(
            test["labels"], test["probabilities"], threshold
        )
        for name, threshold in thresholds.items()
    }
    return {
        "artifact": "dual_d3_transferred_proxy_head_evaluation",
        "schema_version": 1,
        "architecture": {
            "input": "cached_exact_sgw_raw_34d",
            "classifier": "frozen_selector_proxy_head_34_64_2",
            "uses_exact_sgw_scaler": False,
            "updated_parameter_count": 0,
        },
        "threshold_fit_split": "validation",
        "thresholds": thresholds,
        "primary_threshold_policy": "balanced_accuracy",
        "primary_ranking_metric": "roc_auc",
        "validation": {
            "metrics": validation_metrics,
            "predictions": _prediction_rows(validation, thresholds),
        },
        "test": {
            "metrics": test_metrics,
            "predictions": _prediction_rows(test, thresholds),
        },
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
        raise ValueError("cannot write empty transferred-head predictions")
    path = Path(path).resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# D3 Transferred-Head 实验",
        "",
        "- Exact-SGW scaler：不使用",
        "- Proxy head：冻结",
        "- 更新参数量：0",
        "- 阈值拟合集：validation",
        "",
        "| 阈值策略 | Split | AUROC | BA | Accuracy | F1 | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy in payload["thresholds"]:
        for split in ("validation", "test"):
            metrics = payload[split]["metrics"][policy]
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
            "> Test 始终使用 validation 冻结阈值，未在 test 上重新选择。",
            "",
        ]
    )
    return "\n".join(lines)


def write_transferred_head_artifacts(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output_dir / "evaluation.json",
        "summary": output_dir / "summary.md",
        "validation_predictions": output_dir
        / "validation_predictions.csv",
        "test_predictions": output_dir / "test_predictions.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("transferred-head artifacts already exist")
    payload = dict(evaluation)
    payload["provenance"] = dict(provenance)
    _atomic_json(paths["evaluation"], payload)
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
