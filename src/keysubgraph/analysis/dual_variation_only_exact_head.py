"""Formal evaluation for frozen D3 Variation-Only Exact-Head."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
            "{} Variation-Only keys are empty or duplicated".format(name)
        )
    if len(labels_list) != len(keys) or len(probabilities_list) != len(keys):
        raise ValueError(
            "{} Variation-Only predictions are misaligned".format(name)
        )
    if set(labels_list) != {0, 1}:
        raise ValueError(
            "{} Variation-Only data must contain both classes".format(name)
        )
    if any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities_list
    ):
        raise ValueError(
            "{} Variation-Only probabilities are invalid".format(name)
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


def build_variation_only_exact_head_evaluation(
    validation_sample_keys: Sequence[str],
    validation_labels: Sequence[int],
    validation_probabilities: Sequence[float],
    test_sample_keys: Sequence[str],
    test_labels: Sequence[int],
    test_probabilities: Sequence[float],
) -> Dict[str, Any]:
    """Fit B thresholds on validation and freeze them for test."""
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
        raise ValueError("Variation-Only validation and test overlap")
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
        "artifact": "dual_d3_variation_only_exact_head_evaluation",
        "schema_version": 1,
        "architecture": {
            "path": "B_variation_only",
            "source_representation": "frozen_selector_proxy_raw_34d",
            "retained_raw_dimensions": [18, 34],
            "masked_raw_dimensions": [0, 18],
            "mask_value": "exact_sgw_train_only_mean",
            "normalization": "frozen_exact_sgw_train_only_standardizer",
            "classifier": "frozen_exact_sgw_auxiliary_head",
            "selector_frozen": True,
            "scaler_frozen": True,
            "classifier_frozen": True,
            "updated_parameter_count": 0,
            "all_34_proxy_path_removed_or_modified": False,
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
        raise ValueError("cannot write empty Variation-Only predictions")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(evaluation: Mapping[str, Any]) -> str:
    lines = [
        "# D3 B路径（Variation-Only Exact-Head）正式评估",
        "",
        "- 输入：冻结selector生成的34维Proxy表示",
        "- 屏蔽：原始维度0–17替换为Exact-SGW train-only均值",
        "- 保留：原始维度18–33的16维variation",
        "- 标准化与分类头：冻结的Exact-SGW scaler与分类头",
        "- 更新参数量：0",
        "- 阈值只在validation拟合，然后冻结到test",
        "- 原all-34 Proxy路径代码未删除、未修改",
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
            "> Test始终使用validation冻结阈值；未在test上重新选阈值。"
            "本入口只验证B路径，不改变原all-34 Proxy实现。",
            "",
        ]
    )
    return "\n".join(lines)


def write_variation_only_exact_head_artifacts(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    """Write an immutable formal B-path evaluation bundle."""
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
        raise FileExistsError(
            "Variation-Only Exact-Head artifacts already exist"
        )
    payload = dict(evaluation)
    payload["provenance"] = dict(provenance)
    _atomic_json(paths["evaluation"], payload)
    _atomic_json(
        paths["model_spec"],
        {
            "artifact": "dual_d3_variation_only_exact_head_model_spec",
            "schema_version": 1,
            "architecture": evaluation["architecture"],
            "primary_threshold_policy": evaluation[
                "primary_threshold_policy"
            ],
            "frozen_threshold": evaluation["thresholds"][
                evaluation["primary_threshold_policy"]
            ],
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
