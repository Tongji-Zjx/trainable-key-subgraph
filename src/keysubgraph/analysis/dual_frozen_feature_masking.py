"""Frozen validation-only feature masking for the D3 classification path."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)


FEATURE_MASK_CONDITIONS = (
    {
        "code": "A",
        "name": "all_34",
        "description": "all 34 Proxy features",
        "masked_ranges": (),
    },
    {
        "code": "B",
        "name": "variation_only",
        "description": "variation only; dimensions 0:18 set to train mean",
        "masked_ranges": ((0, 18),),
    },
    {
        "code": "C",
        "name": "spectral_delta_variation",
        "description": (
            "spectral_delta plus variation; both speed dimensions masked"
        ),
        "masked_ranges": ((16, 18),),
    },
    {
        "code": "D",
        "name": "no_gw_speed",
        "description": "GW speed masked",
        "masked_ranges": ((17, 18),),
    },
    {
        "code": "E",
        "name": "no_spectral_or_gw_speed",
        "description": "spectral speed and GW speed masked",
        "masked_ranges": ((16, 18),),
    },
)


def _condition(code: str) -> Mapping[str, Any]:
    matches = [
        item for item in FEATURE_MASK_CONDITIONS if item["code"] == code
    ]
    if len(matches) != 1:
        raise ValueError("unknown frozen feature-mask condition: {}".format(code))
    return matches[0]


def apply_frozen_feature_mask(
    values: np.ndarray,
    train_mean: np.ndarray,
    condition_code: str,
) -> np.ndarray:
    """Replace masked raw dimensions by the frozen train-only mean."""
    values = np.asarray(values, dtype=np.float64)
    train_mean = np.asarray(train_mean, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 34:
        raise ValueError("feature masking requires a finite [N,34] array")
    if train_mean.shape != (34,):
        raise ValueError("feature masking requires a 34-D train mean")
    if not np.isfinite(values).all() or not np.isfinite(train_mean).all():
        raise ValueError("feature masking inputs must be finite")
    masked = values.copy()
    for start, stop in _condition(condition_code)["masked_ranges"]:
        masked[:, start:stop] = train_mean[start:stop]
    return masked


def _validate_predictions(
    sample_keys: Sequence[str],
    labels: Sequence[int],
    probabilities: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    keys = [str(value) for value in sample_keys]
    labels_list = [int(value) for value in labels]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("feature-mask sample keys are empty or duplicated")
    if len(labels_list) != len(keys) or set(labels_list) != {0, 1}:
        raise ValueError("feature-mask labels are invalid or misaligned")
    expected = {item["code"] for item in FEATURE_MASK_CONDITIONS}
    if set(probabilities) != expected:
        raise ValueError("feature-mask probabilities do not cover A-E")
    checked = {}
    for code in sorted(expected):
        values = np.asarray(probabilities[code], dtype=np.float64)
        if (
            values.shape != (len(keys),)
            or not np.isfinite(values).all()
            or bool((values < 0.0).any())
            or bool((values > 1.0).any())
        ):
            raise ValueError(
                "feature-mask probabilities are invalid for {}".format(code)
            )
        checked[code] = values
    return {"keys": keys, "labels": labels_list, "probabilities": checked}


def _safe_spearman(left: np.ndarray, right: np.ndarray):
    if left.std() <= 1.0e-12 or right.std() <= 1.0e-12:
        return None
    value = float(spearmanr(left, right)[0])
    return value if np.isfinite(value) else None


def build_frozen_feature_mask_evaluation(
    sample_keys: Sequence[str],
    labels: Sequence[int],
    probabilities: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    """Compare A-E with one frozen head and no test-set access."""
    checked = _validate_predictions(sample_keys, labels, probabilities)
    labels_list = checked["labels"]
    values = checked["probabilities"]
    duplicate_max_difference = float(np.max(np.abs(values["C"] - values["E"])))
    if duplicate_max_difference > 1.0e-12:
        raise ValueError("conditions C and E must be numerically identical")
    shared_threshold = fit_binary_threshold(
        labels_list,
        values["A"].tolist(),
        "balanced_accuracy",
    )
    baseline_auc = binary_metrics(
        labels_list, values["A"].tolist(), shared_threshold
    )["roc_auc"]
    conditions = []
    for specification in FEATURE_MASK_CONDITIONS:
        code = specification["code"]
        metrics = binary_metrics(
            labels_list, values[code].tolist(), shared_threshold
        )
        if code == "A":
            probability_spearman = 1.0
            mean_absolute_difference = 0.0
        else:
            probability_spearman = _safe_spearman(
                values["A"], values[code]
            )
            mean_absolute_difference = float(
                np.mean(np.abs(values["A"] - values[code]))
            )
        conditions.append(
            {
                "code": code,
                "name": specification["name"],
                "description": specification["description"],
                "masked_ranges": [
                    [int(start), int(stop)]
                    for start, stop in specification["masked_ranges"]
                ],
                "metrics": metrics,
                "auc_delta_vs_A": float(metrics["roc_auc"] - baseline_auc),
                "probability_spearman_vs_A": probability_spearman,
                "mean_absolute_probability_difference_vs_A": (
                    mean_absolute_difference
                ),
            }
        )
    auc = {
        row["code"]: float(row["metrics"]["roc_auc"])
        for row in conditions
    }
    contrasts = {
        "gw_speed_incremental_auc_A_minus_D": auc["A"] - auc["D"],
        "spectral_speed_incremental_auc_D_minus_E": auc["D"] - auc["E"],
        "both_speeds_incremental_auc_A_minus_E": auc["A"] - auc["E"],
        "spectral_delta_incremental_auc_E_minus_B": auc["E"] - auc["B"],
    }
    prediction_rows = []
    for index, (key, label) in enumerate(
        zip(checked["keys"], labels_list)
    ):
        row = {"sample_key": key, "label": int(label)}
        for code in sorted(values):
            probability = float(values[code][index])
            row["{}_probability".format(code)] = probability
            row["{}_prediction".format(code)] = int(
                probability >= shared_threshold
            )
        prediction_rows.append(row)
    return {
        "artifact": "dual_d3_frozen_feature_mask_diagnostic",
        "schema_version": 1,
        "split": "validation",
        "test_split_used": False,
        "updated_parameter_count": 0,
        "primary_metric": "roc_auc",
        "shared_threshold": {
            "fit_condition": "A",
            "fit_split": "validation",
            "metric": "balanced_accuracy",
            "value": float(shared_threshold),
            "applied_unchanged_to_all_conditions": True,
        },
        "conditions": conditions,
        "contrasts": contrasts,
        "duplicate_condition_check": {
            "conditions": ["C", "E"],
            "reason": (
                "both definitions mask dimensions 16:18 and are identical"
            ),
            "maximum_probability_difference": duplicate_max_difference,
            "passed": True,
        },
        "predictions": prediction_rows,
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
        raise ValueError("cannot write empty feature-mask CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(evaluation: Mapping[str, Any]) -> str:
    lines = [
        "# D3 冻结特征屏蔽诊断",
        "",
        "- 数据：validation（test 未使用）",
        "- Selector、train-only scaler、Exact-SGW 分类头：全部冻结",
        "- 更新参数量：0",
        "- 主比较指标：AUROC",
        "- 辅助分类指标：所有条件共享 A 在 validation 上确定的 BA 阈值",
        "",
        "| 条件 | 保留/屏蔽定义 | AUROC | ΔAUC vs A | BA | Accuracy | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["conditions"]:
        metrics = row["metrics"]
        lines.append(
            "| {} | {} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | "
            "{:.6f} |".format(
                row["code"],
                row["description"],
                metrics["roc_auc"],
                row["auc_delta_vs_A"],
                metrics["balanced_accuracy"],
                metrics["accuracy"],
                metrics["f1"],
            )
        )
    lines.extend(
        [
            "",
            "## 预定义对比",
            "",
            "| 对比 | AUROC差值 | 正值含义 |",
            "|---|---:|---|",
            "| A − D | {:+.6f} | GW speed 在其余特征存在时有增益 |".format(
                evaluation["contrasts"][
                    "gw_speed_incremental_auc_A_minus_D"
                ]
            ),
            "| D − E | {:+.6f} | 屏蔽 GW 后 spectral speed 仍有增益 |".format(
                evaluation["contrasts"][
                    "spectral_speed_incremental_auc_D_minus_E"
                ]
            ),
            "| A − E | {:+.6f} | 两个 speed 合计有增益 |".format(
                evaluation["contrasts"][
                    "both_speeds_incremental_auc_A_minus_E"
                ]
            ),
            "| E − B | {:+.6f} | spectral_delta 在 variation 上有增益 |".format(
                evaluation["contrasts"][
                    "spectral_delta_incremental_auc_E_minus_B"
                ]
            ),
            "",
            "> C 与 E 按给定定义完全相同；程序已验证其逐样本概率一致。"
            "本诊断不重新训练，也不使用 test 做架构选择。",
            "",
        ]
    )
    return "\n".join(lines)


def write_frozen_feature_mask_artifacts(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    """Write an immutable validation-only diagnostic bundle."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output_dir / "evaluation.json",
        "experiment_spec": output_dir / "experiment_spec.json",
        "condition_metrics": output_dir / "condition_metrics.csv",
        "validation_predictions": output_dir / "validation_predictions.csv",
        "summary": output_dir / "summary.md",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("frozen feature-mask artifacts already exist")
    payload = dict(evaluation)
    payload["provenance"] = dict(provenance)
    _atomic_json(paths["evaluation"], payload)
    _atomic_json(
        paths["experiment_spec"],
        {
            "artifact": "dual_d3_frozen_feature_mask_spec",
            "schema_version": 1,
            "conditions": list(FEATURE_MASK_CONDITIONS),
            "split": "validation",
            "test_split_used": False,
            "updated_parameter_count": 0,
            "primary_metric": "roc_auc",
            "provenance": dict(provenance),
        },
    )
    metric_rows = []
    for row in evaluation["conditions"]:
        metrics = row["metrics"]
        metric_rows.append(
            {
                "condition": row["code"],
                "name": row["name"],
                "roc_auc": metrics["roc_auc"],
                "auc_delta_vs_A": row["auc_delta_vs_A"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "probability_spearman_vs_A": row[
                    "probability_spearman_vs_A"
                ],
                "mean_absolute_probability_difference_vs_A": row[
                    "mean_absolute_probability_difference_vs_A"
                ],
            }
        )
    _write_csv(paths["condition_metrics"], metric_rows)
    _write_csv(
        paths["validation_predictions"], evaluation["predictions"]
    )
    temporary = paths["summary"].with_suffix(".md.tmp")
    temporary.write_text(_markdown(evaluation) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(paths["summary"]))
    return paths
