"""Frozen diagnostics for D3 classification bottlenecks."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


FEATURE_BLOCKS = (
    ("spectral_delta", 0, 16),
    ("spectral_speed", 16, 17),
    ("gw_speed", 17, 18),
    ("variation", 18, 34),
)


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() <= 1.0e-12 or right.std() <= 1.0e-12:
        return None
    value = float(spearmanr(left, right)[0])
    return value if math.isfinite(value) else None


def _effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 1.0e-20:
        return 0.0
    probabilities = energy / total
    entropy = -float(
        np.sum(
            probabilities[probabilities > 0.0]
            * np.log(probabilities[probabilities > 0.0])
        )
    )
    return float(np.exp(entropy))


def _representation_statistics(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    zero = values[labels == 0]
    one = values[labels == 1]
    zero_centroid = zero.mean(axis=0)
    one_centroid = one.mean(axis=0)
    difference = one_centroid - zero_centroid
    within = float(
        np.mean(np.sum((zero - zero_centroid) ** 2, axis=1))
        + np.mean(np.sum((one - one_centroid) ** 2, axis=1))
    )
    norms = np.linalg.norm(values, axis=1)
    return {
        "sample_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "mean_feature_variance": float(values.var(axis=0).mean()),
        "effective_rank": _effective_rank(values),
        "centroid_distance": float(np.linalg.norm(difference)),
        "fisher_ratio": float(np.dot(difference, difference))
        / max(1.0e-12, within),
        "active_feature_fraction": float(
            np.mean(values.std(axis=0) > 1.0e-6)
        ),
        "mean_norm": float(norms.mean()),
        "norm_standard_deviation": float(norms.std()),
    }


def _drift_statistics(train, validation):
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
    if validation.ndim == 1:
        validation = validation[:, None]
    if train.shape[1] != validation.shape[1]:
        raise ValueError("drift representations have different dimensions")
    pooled = np.sqrt(
        0.5 * (train.var(axis=0) + validation.var(axis=0))
    )
    standardized = np.abs(
        validation.mean(axis=0) - train.mean(axis=0)
    ) / np.maximum(pooled, 1.0e-12)
    train_covariance = np.cov(train, rowvar=False)
    validation_covariance = np.cov(validation, rowvar=False)
    train_covariance = np.atleast_2d(train_covariance)
    validation_covariance = np.atleast_2d(validation_covariance)
    denominator = max(
        1.0e-12, float(np.linalg.norm(train_covariance))
    )
    return {
        "median_absolute_standardized_mean_shift": float(
            np.median(standardized)
        ),
        "maximum_absolute_standardized_mean_shift": float(
            np.max(standardized)
        ),
        "covariance_relative_frobenius_shift": float(
            np.linalg.norm(validation_covariance - train_covariance)
        )
        / denominator,
    }


def _validate_inputs(
    train_labels,
    validation_labels,
    train_proxy,
    train_exact,
    validation_proxy,
    validation_exact,
    path_probabilities,
    layer_representations,
):
    labels = {
        "train": np.asarray(train_labels, dtype=np.int64),
        "validation": np.asarray(validation_labels, dtype=np.int64),
    }
    features = {
        "train_proxy": np.asarray(train_proxy, dtype=np.float64),
        "train_exact": np.asarray(train_exact, dtype=np.float64),
        "validation_proxy": np.asarray(
            validation_proxy, dtype=np.float64
        ),
        "validation_exact": np.asarray(
            validation_exact, dtype=np.float64
        ),
    }
    for split in ("train", "validation"):
        if set(labels[split].tolist()) != {0, 1}:
            raise ValueError("bottleneck labels must contain both classes")
        count = int(labels[split].size)
        for source in ("proxy", "exact"):
            values = features["{}_{}".format(split, source)]
            if values.shape != (count, 34) or not np.isfinite(values).all():
                raise ValueError("bottleneck features must be finite [N,34]")
        for values in path_probabilities[split].values():
            array = np.asarray(values, dtype=np.float64)
            if (
                array.shape != (count,)
                or not np.isfinite(array).all()
                or bool((array < 0.0).any())
                or bool((array > 1.0).any())
            ):
                raise ValueError("bottleneck path probabilities are invalid")
        for values in layer_representations[split].values():
            array = np.asarray(values, dtype=np.float64)
            if (
                array.ndim not in (1, 2)
                or array.shape[0] != count
                or not np.isfinite(array).all()
            ):
                raise ValueError("bottleneck layer values are invalid")
    if set(path_probabilities["train"]) != set(
        path_probabilities["validation"]
    ):
        raise ValueError("train and validation paths disagree")
    if set(layer_representations["train"]) != set(
        layer_representations["validation"]
    ):
        raise ValueError("train and validation layers disagree")
    return labels, features


def analyze_dual_classification_bottleneck(
    train_labels: Sequence[int],
    validation_labels: Sequence[int],
    train_proxy: np.ndarray,
    train_exact: np.ndarray,
    validation_proxy: np.ndarray,
    validation_exact: np.ndarray,
    path_probabilities: Mapping[
        str, Mapping[str, Sequence[float]]
    ],
    permutation_aucs: Mapping[str, Sequence[float]],
    layer_representations: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    selector_stability_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate all frozen diagnostics without fitting a model."""
    labels, features = _validate_inputs(
        train_labels,
        validation_labels,
        train_proxy,
        train_exact,
        validation_proxy,
        validation_exact,
        path_probabilities,
        layer_representations,
    )
    baseline_name = "proxy_all"
    if baseline_name not in path_probabilities["validation"]:
        raise ValueError("bottleneck paths require proxy_all")
    path_rows = []
    baseline_auc = _auc(
        labels["validation"],
        path_probabilities["validation"][baseline_name],
    )
    for name in sorted(path_probabilities["validation"]):
        train_auc = _auc(
            labels["train"], path_probabilities["train"][name]
        )
        validation_auc = _auc(
            labels["validation"],
            path_probabilities["validation"][name],
        )
        path_rows.append(
            {
                "path": name,
                "train_auc": train_auc,
                "validation_auc": validation_auc,
                "validation_minus_proxy_auc": validation_auc
                - baseline_auc,
                "train_validation_auc_gap": train_auc - validation_auc,
            }
        )
    block_rows = []
    for name, start, stop in FEATURE_BLOCKS:
        values = [
            float(value) for value in permutation_aucs.get(name, ())
        ]
        if not values:
            raise ValueError(
                "missing permutation repetitions for {}".format(name)
            )
        replacement = next(
            row
            for row in path_rows
            if row["path"] == "replace_{}".format(name)
        )
        block_rows.append(
            {
                "block": name,
                "start": start,
                "stop": stop,
                "dimension": stop - start,
                "replacement_validation_auc": replacement[
                    "validation_auc"
                ],
                "replacement_auc_delta": replacement[
                    "validation_minus_proxy_auc"
                ],
                "permutation_auc_mean": float(np.mean(values)),
                "permutation_auc_std": float(np.std(values)),
                "permutation_auc_drop": baseline_auc
                - float(np.mean(values)),
            }
        )
    feature_rows = []
    validation_probability = np.asarray(
        path_probabilities["validation"][baseline_name],
        dtype=np.float64,
    )
    for index in range(34):
        block = next(
            name
            for name, start, stop in FEATURE_BLOCKS
            if start <= index < stop
        )
        row = {"dimension": index, "block": block}
        for source in ("proxy", "exact"):
            values = features["validation_{}".format(source)][:, index]
            zero = values[labels["validation"] == 0]
            one = values[labels["validation"] == 1]
            pooled = math.sqrt(
                0.5 * (float(zero.var()) + float(one.var()))
            )
            auc = _auc(labels["validation"], values)
            row.update(
                {
                    "{}_auc".format(source): auc,
                    "{}_direction_free_auc".format(source): max(
                        auc, 1.0 - auc
                    ),
                    "{}_standardized_class_effect".format(source): float(
                        one.mean() - zero.mean()
                    )
                    / max(1.0e-12, pooled),
                    "{}_probability_spearman".format(
                        source
                    ): _safe_spearman(values, validation_probability),
                }
            )
        feature_rows.append(row)
    layer_rows = []
    drift_rows = []
    for name in sorted(layer_representations["train"]):
        for split in ("train", "validation"):
            statistics = _representation_statistics(
                layer_representations[split][name], labels[split]
            )
            layer_rows.append(
                dict({"layer": name, "split": split}, **statistics)
            )
        drift_rows.append(
            dict(
                {"layer": name},
                **_drift_statistics(
                    layer_representations["train"][name],
                    layer_representations["validation"][name],
                )
            )
        )
    stability_rows = [dict(row) for row in selector_stability_rows]
    stability_summary = {}
    numeric_keys = (
        "node_score_margin",
        "edge_score_margin",
        "node_perturbation_jaccard",
        "edge_perturbation_jaccard",
        "temporal_node_jaccard",
        "temporal_edge_jaccard",
    )
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in stability_rows
            if row.get(key) is not None
            and math.isfinite(float(row[key]))
        ]
        stability_summary[key] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "minimum": float(np.min(values)) if values else None,
        }
    proxy_input = next(
        row
        for row in layer_rows
        if row["layer"] == "proxy_scaled_input"
        and row["split"] == "validation"
    )
    proxy_hidden = next(
        row
        for row in layer_rows
        if row["layer"] == "proxy_hidden_activation"
        and row["split"] == "validation"
    )
    proxy_logits = next(
        row
        for row in layer_rows
        if row["layer"] == "proxy_logits"
        and row["split"] == "validation"
    )
    best_replacement = max(
        block_rows, key=lambda row: row["replacement_auc_delta"]
    )
    strongest_permutation = max(
        block_rows, key=lambda row: row["permutation_auc_drop"]
    )
    summary = {
        "train_sample_count": int(labels["train"].size),
        "validation_sample_count": int(labels["validation"].size),
        "baseline_proxy_validation_auc": baseline_auc,
        "best_replacement_block": best_replacement["block"],
        "best_replacement_auc_delta": best_replacement[
            "replacement_auc_delta"
        ],
        "strongest_permutation_block": strongest_permutation["block"],
        "strongest_permutation_auc_drop": strongest_permutation[
            "permutation_auc_drop"
        ],
        "proxy_input_effective_rank": proxy_input["effective_rank"],
        "proxy_hidden_effective_rank": proxy_hidden["effective_rank"],
        "proxy_logits_effective_rank": proxy_logits["effective_rank"],
        "effective_rank_retention": proxy_hidden["effective_rank"]
        / max(1.0e-12, proxy_input["effective_rank"]),
        "proxy_logits_mean_feature_variance": proxy_logits[
            "mean_feature_variance"
        ],
        "selector_stability": stability_summary,
    }
    return {
        "summary": summary,
        "path_rows": path_rows,
        "block_rows": block_rows,
        "feature_rows": feature_rows,
        "layer_rows": layer_rows,
        "drift_rows": drift_rows,
        "selector_stability_rows": stability_rows,
    }


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty bottleneck diagnostic CSV")
    path = Path(path).resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(analysis):
    summary = analysis["summary"]
    lines = [
        "# D3 分类瓶颈一次性诊断",
        "",
        "- Train 样本数：{}".format(summary["train_sample_count"]),
        "- Validation 样本数：{}".format(
            summary["validation_sample_count"]
        ),
        "- ProxyInput 基准 validation AUROC：{:.6f}".format(
            summary["baseline_proxy_validation_auc"]
        ),
        "- 最佳 Exact 分块替换：{} ({:+.6f})".format(
            summary["best_replacement_block"],
            summary["best_replacement_auc_delta"],
        ),
        "- 最强置换重要块：{} (AUROC下降 {:+.6f})".format(
            summary["strongest_permutation_block"],
            summary["strongest_permutation_auc_drop"],
        ),
        "- 输入→logits 有效秩保留率：{:.4f}".format(
            summary["effective_rank_retention"]
        ),
        "",
        "## 冻结路径与分块替换",
        "",
        "| 路径 | Train AUC | Validation AUC | 相对Proxy变化 | 泛化差距 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analysis["path_rows"]:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:+.6f} | {:+.6f} |".format(
                row["path"],
                row["train_auc"],
                row["validation_auc"],
                row["validation_minus_proxy_auc"],
                row["train_validation_auc_gap"],
            )
        )
    lines.extend(
        [
            "",
            "## 特征块置换",
            "",
            "| 特征块 | 替换AUC变化 | 置换后AUC | 置换AUC下降 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in analysis["block_rows"]:
        lines.append(
            "| {} | {:+.6f} | {:.6f} | {:+.6f} |".format(
                row["block"],
                row["replacement_auc_delta"],
                row["permutation_auc_mean"],
                row["permutation_auc_drop"],
            )
        )
    lines.extend(
        [
            "",
            "## Selector 稳定性",
            "",
            "| 指标 | 均值 | 中位数 | 最小值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in summary["selector_stability"].items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                name,
                (
                    "{:.6f}".format(values["mean"])
                    if values["mean"] is not None
                    else "N/A"
                ),
                (
                    "{:.6f}".format(values["median"])
                    if values["median"] is not None
                    else "N/A"
                ),
                (
                    "{:.6f}".format(values["minimum"])
                    if values["minimum"] is not None
                    else "N/A"
                ),
            )
        )
    lines.extend(
        [
            "",
            "> 本诊断不更新模型参数；架构判断应以 validation 为主，"
            "test 不参与瓶颈选择。",
            "",
        ]
    )
    return "\n".join(lines)


def write_dual_classification_bottleneck_artifacts(
    output_dir: Path,
    analysis: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / "summary.json",
        "summary_markdown": output_dir / "summary.md",
        "paths": output_dir / "path_diagnostics.csv",
        "blocks": output_dir / "feature_block_diagnostics.csv",
        "features": output_dir / "feature_dimension_diagnostics.csv",
        "layers": output_dir / "layer_diagnostics.csv",
        "drift": output_dir / "train_validation_drift.csv",
        "selector": output_dir / "selector_stability.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("classification diagnostic already exists")
    _atomic_json(
        paths["summary_json"],
        {
            "artifact": "dual_d3_classification_bottleneck_diagnostic",
            "schema_version": 1,
            "provenance": dict(provenance),
            **dict(analysis["summary"]),
        },
    )
    _write_csv(paths["paths"], analysis["path_rows"])
    _write_csv(paths["blocks"], analysis["block_rows"])
    _write_csv(paths["features"], analysis["feature_rows"])
    _write_csv(paths["layers"], analysis["layer_rows"])
    _write_csv(paths["drift"], analysis["drift_rows"])
    _write_csv(
        paths["selector"], analysis["selector_stability_rows"]
    )
    temporary = paths["summary_markdown"].with_suffix(".md.tmp")
    temporary.write_text(_markdown(analysis) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(paths["summary_markdown"]))
    return paths
