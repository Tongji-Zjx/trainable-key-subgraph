"""Read-only statistics for Dual-STSE proxy versus Exact-SGW alignment."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


FEATURE_BLOCKS = (
    ("spectral_delta", 0, 16),
    ("spectral_speed", 16, 17),
    ("gw_speed", 17, 18),
    ("variation", 18, 34),
)


def _feature_name(index: int) -> str:
    if index < 16:
        return "spectral_delta_q{:02d}".format(index)
    if index == 16:
        return "spectral_speed"
    if index == 17:
        return "geometry_speed_proxy_vs_exact_gw"
    return "spectral_variation_q{:02d}".format(index - 18)


def _block_name(index: int) -> str:
    for name, start, stop in FEATURE_BLOCKS:
        if start <= index < stop:
            return name
    raise IndexError("SGW feature index is outside [0,34)")


def _finite_or_none(value: Any) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_correlation(
    left: np.ndarray, right: np.ndarray, method: str
) -> Optional[float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or left.size != right.size:
        return None
    if np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return None
    if method == "pearson":
        value = pearsonr(left, right)[0]
    elif method == "spearman":
        value = spearmanr(left, right)[0]
    elif method == "kendall":
        value = kendalltau(left, right)[0]
    else:
        raise ValueError("unsupported correlation method")
    return _finite_or_none(value)


def _cosine(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return None
    return _finite_or_none(float(np.dot(left, right)) / denominator)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int)
    return {
        "sample_count": int(labels.size),
        "threshold": float(threshold),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if set(labels.tolist()) == {0, 1}
            else None
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def _class_separation(
    values: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    zero = values[labels == 0]
    one = values[labels == 1]
    if zero.size < 1 or one.size < 1:
        raise ValueError("class separation requires both classes")
    zero_centroid = zero.mean(axis=0)
    one_centroid = one.mean(axis=0)
    difference = one_centroid - zero_centroid
    within_zero = float(
        np.mean(np.sum((zero - zero_centroid) ** 2, axis=1))
    )
    within_one = float(
        np.mean(np.sum((one - one_centroid) ** 2, axis=1))
    )
    squared_distance = float(np.dot(difference, difference))
    return {
        "centroid_euclidean_distance": math.sqrt(squared_distance),
        "centroid_cosine_similarity": _cosine(
            zero_centroid, one_centroid
        ),
        "class_0_within_squared_distance": within_zero,
        "class_1_within_squared_distance": within_one,
        "fisher_ratio": squared_distance
        / max(1.0e-12, within_zero + within_one),
    }


def _validate_inputs(
    sample_keys: Sequence[str],
    labels: Sequence[int],
    proxy_features: np.ndarray,
    exact_features: np.ndarray,
    probabilities: Mapping[str, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    proxy = np.asarray(proxy_features, dtype=np.float64)
    exact = np.asarray(exact_features, dtype=np.float64)
    count = len(sample_keys)
    if len(set(str(key) for key in sample_keys)) != count:
        raise ValueError("alignment samples contain duplicate keys")
    if labels_array.shape != (count,) or set(labels_array.tolist()) != {0, 1}:
        raise ValueError("alignment labels must contain both binary classes")
    if proxy.shape != (count, 34) or exact.shape != (count, 34):
        raise ValueError("proxy and Exact-SGW features must be [N,34]")
    if not np.isfinite(proxy).all() or not np.isfinite(exact).all():
        raise ValueError("proxy or Exact-SGW features are non-finite")
    required = {"proxy_proxy", "exact_proxy", "exact_exact", "proxy_exact"}
    if set(probabilities) != required:
        raise ValueError("alignment requires the four classification paths")
    for values in probabilities.values():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError("classification probabilities are invalid")
        if bool((array < 0.0).any()) or bool((array > 1.0).any()):
            raise ValueError("classification probabilities leave [0,1]")
    return labels_array, proxy, exact


def analyze_proxy_exact_alignment(
    sample_keys: Sequence[str],
    labels: Sequence[int],
    proxy_features: np.ndarray,
    exact_features: np.ndarray,
    probabilities: Mapping[str, Sequence[float]],
    proxy_threshold: float,
    exact_threshold: float,
    proxy_transition_masks: Sequence[np.ndarray],
    exact_transition_masks: Sequence[np.ndarray],
    proxy_standardized: Optional[np.ndarray] = None,
    exact_standardized: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze aligned frozen representations without fitting any model."""
    labels_array, proxy, exact = _validate_inputs(
        sample_keys,
        labels,
        proxy_features,
        exact_features,
        probabilities,
    )
    count = len(sample_keys)
    if (
        len(proxy_transition_masks) != count
        or len(exact_transition_masks) != count
    ):
        raise ValueError("transition masks do not align with samples")
    probability_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in probabilities.items()
    }
    difference = proxy - exact
    sample_rows: List[Dict[str, Any]] = []
    mask_equal_count = 0
    mask_position_total = 0
    mask_position_equal = 0
    for index, sample_key in enumerate(sample_keys):
        row: Dict[str, Any] = {
            "sample_key": str(sample_key),
            "label": int(labels_array[index]),
        }
        for name, start, stop in (("all", 0, 34),) + FEATURE_BLOCKS:
            left = proxy[index, start:stop]
            right = exact[index, start:stop]
            delta = left - right
            rmse = math.sqrt(float(np.mean(delta ** 2)))
            denominator = math.sqrt(float(np.mean(right ** 2)))
            row["{}_mae".format(name)] = float(np.mean(np.abs(delta)))
            row["{}_rmse".format(name)] = rmse
            row["{}_normalized_rmse".format(name)] = rmse / max(
                1.0e-12, denominator
            )
            row["{}_cosine".format(name)] = _cosine(left, right)
            row["{}_pearson".format(name)] = _safe_correlation(
                left, right, "pearson"
            )
            row["{}_spearman".format(name)] = _safe_correlation(
                left, right, "spearman"
            )
        proxy_mask = np.asarray(
            proxy_transition_masks[index], dtype=np.bool_
        ).reshape(-1)
        exact_mask = np.asarray(
            exact_transition_masks[index], dtype=np.bool_
        ).reshape(-1)
        same_length = proxy_mask.size == exact_mask.size
        maximum = max(proxy_mask.size, exact_mask.size)
        left_padded = np.zeros(maximum, dtype=np.bool_)
        right_padded = np.zeros(maximum, dtype=np.bool_)
        left_padded[: proxy_mask.size] = proxy_mask
        right_padded[: exact_mask.size] = exact_mask
        equal_positions = int(np.sum(left_padded == right_padded))
        masks_equal = same_length and bool(np.array_equal(proxy_mask, exact_mask))
        mask_equal_count += int(masks_equal)
        mask_position_total += maximum
        mask_position_equal += equal_positions
        row.update(
            {
                "proxy_valid_transition_count": int(proxy_mask.sum()),
                "exact_valid_transition_count": int(exact_mask.sum()),
                "transition_mask_length_equal": bool(same_length),
                "transition_mask_equal": bool(masks_equal),
                "transition_mask_position_agreement": (
                    float(equal_positions) / float(maximum)
                    if maximum
                    else 1.0
                ),
            }
        )
        for path, values in probability_arrays.items():
            row["{}_probability".format(path)] = float(values[index])
        sample_rows.append(row)

    dimension_rows: List[Dict[str, Any]] = []
    for index in range(34):
        proxy_values = proxy[:, index]
        exact_values = exact[:, index]
        proxy_zero = proxy_values[labels_array == 0]
        proxy_one = proxy_values[labels_array == 1]
        exact_zero = exact_values[labels_array == 0]
        exact_one = exact_values[labels_array == 1]
        proxy_effect = float(proxy_one.mean() - proxy_zero.mean())
        exact_effect = float(exact_one.mean() - exact_zero.mean())
        pooled_proxy = math.sqrt(
            0.5 * (float(proxy_zero.var()) + float(proxy_one.var()))
        )
        pooled_exact = math.sqrt(
            0.5 * (float(exact_zero.var()) + float(exact_one.var()))
        )
        dimension_rows.append(
            {
                "dimension": index,
                "feature_name": _feature_name(index),
                "block": _block_name(index),
                "proxy_mean": float(proxy_values.mean()),
                "proxy_std": float(proxy_values.std()),
                "exact_mean": float(exact_values.mean()),
                "exact_std": float(exact_values.std()),
                "proxy_minus_exact_bias": float(
                    (proxy_values - exact_values).mean()
                ),
                "mae": float(
                    np.mean(np.abs(proxy_values - exact_values))
                ),
                "rmse": math.sqrt(
                    float(np.mean((proxy_values - exact_values) ** 2))
                ),
                "pearson": _safe_correlation(
                    proxy_values, exact_values, "pearson"
                ),
                "spearman": _safe_correlation(
                    proxy_values, exact_values, "spearman"
                ),
                "proxy_to_exact_std_ratio": float(proxy_values.std())
                / max(1.0e-12, float(exact_values.std())),
                "proxy_class_mean_difference": proxy_effect,
                "exact_class_mean_difference": exact_effect,
                "proxy_standardized_class_effect": proxy_effect
                / max(1.0e-12, pooled_proxy),
                "exact_standardized_class_effect": exact_effect
                / max(1.0e-12, pooled_exact),
                "class_effect_direction_equal": bool(
                    np.sign(proxy_effect) == np.sign(exact_effect)
                ),
            }
        )

    block_summary = {}
    for name, start, stop in (("all", 0, 34),) + FEATURE_BLOCKS:
        block_difference = difference[:, start:stop]
        exact_block = exact[:, start:stop]
        block_dimensions = dimension_rows[start:stop]
        block_summary[name] = {
            "mae": float(np.mean(np.abs(block_difference))),
            "rmse": math.sqrt(float(np.mean(block_difference ** 2))),
            "normalized_rmse": math.sqrt(
                float(np.mean(block_difference ** 2))
            )
            / max(1.0e-12, math.sqrt(float(np.mean(exact_block ** 2)))),
            "median_sample_cosine": _median(
                [
                    row["{}_cosine".format(name)]
                    for row in sample_rows
                ]
            ),
            "median_dimension_pearson": _median(
                [row["pearson"] for row in block_dimensions]
            ),
            "median_dimension_spearman": _median(
                [row["spearman"] for row in block_dimensions]
            ),
            "class_effect_direction_agreement": float(
                np.mean(
                    [
                        row["class_effect_direction_equal"]
                        for row in block_dimensions
                    ]
                )
            ),
        }

    path_thresholds = {
        "proxy_proxy": float(proxy_threshold),
        "exact_proxy": float(proxy_threshold),
        "exact_exact": float(exact_threshold),
        "proxy_exact": float(exact_threshold),
    }
    classification_rows = []
    classification = {}
    for path, values in probability_arrays.items():
        metrics = _binary_metrics(
            labels_array, values, path_thresholds[path]
        )
        classification[path] = metrics
        classification_rows.append(
            dict({"path": path}, **metrics)
        )
    proxy_predictions = (
        probability_arrays["proxy_proxy"] >= float(proxy_threshold)
    )
    exact_predictions = (
        probability_arrays["exact_exact"] >= float(exact_threshold)
    )
    probability_alignment = {
        "proxy_proxy_vs_exact_exact": {
            "pearson": _safe_correlation(
                probability_arrays["proxy_proxy"],
                probability_arrays["exact_exact"],
                "pearson",
            ),
            "spearman": _safe_correlation(
                probability_arrays["proxy_proxy"],
                probability_arrays["exact_exact"],
                "spearman",
            ),
            "kendall": _safe_correlation(
                probability_arrays["proxy_proxy"],
                probability_arrays["exact_exact"],
                "kendall",
            ),
            "mean_absolute_probability_difference": float(
                np.mean(
                    np.abs(
                        probability_arrays["proxy_proxy"]
                        - probability_arrays["exact_exact"]
                    )
                )
            ),
            "prediction_disagreement_rate": float(
                np.mean(proxy_predictions != exact_predictions)
            ),
        },
        "proxy_proxy_vs_exact_proxy": {
            "spearman": _safe_correlation(
                probability_arrays["proxy_proxy"],
                probability_arrays["exact_proxy"],
                "spearman",
            )
        },
        "exact_exact_vs_proxy_exact": {
            "spearman": _safe_correlation(
                probability_arrays["exact_exact"],
                probability_arrays["proxy_exact"],
                "spearman",
            )
        },
    }
    probability_rows = []
    for index, sample_key in enumerate(sample_keys):
        probability_rows.append(
            {
                "sample_key": str(sample_key),
                "label": int(labels_array[index]),
                **{
                    "{}_probability".format(path): float(values[index])
                    for path, values in probability_arrays.items()
                },
                "proxy_proxy_prediction": int(proxy_predictions[index]),
                "exact_exact_prediction": int(exact_predictions[index]),
            }
        )
    class_separation = {
        "proxy_raw": _class_separation(proxy, labels_array),
        "exact_raw": _class_separation(exact, labels_array),
    }
    if proxy_standardized is not None and exact_standardized is not None:
        proxy_scaled = np.asarray(proxy_standardized, dtype=np.float64)
        exact_scaled = np.asarray(exact_standardized, dtype=np.float64)
        if proxy_scaled.shape != proxy.shape or exact_scaled.shape != exact.shape:
            raise ValueError("standardized alignment values are malformed")
        class_separation.update(
            {
                "proxy_exact_scaler_standardized": _class_separation(
                    proxy_scaled, labels_array
                ),
                "exact_standardized": _class_separation(
                    exact_scaled, labels_array
                ),
            }
        )
    return {
        "summary": {
            "sample_count": count,
            "class_counts": {
                "0": int(np.sum(labels_array == 0)),
                "1": int(np.sum(labels_array == 1)),
            },
            "feature_blocks": block_summary,
            "transition_masks": {
                "exact_sample_match_rate": float(mask_equal_count)
                / float(count),
                "position_agreement_rate": float(mask_position_equal)
                / float(max(1, mask_position_total)),
            },
            "classification_paths": classification,
            "probability_alignment": probability_alignment,
            "class_separation": class_separation,
        },
        "sample_rows": sample_rows,
        "dimension_rows": dimension_rows,
        "classification_rows": classification_rows,
        "probability_rows": probability_rows,
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
        raise ValueError("cannot write an empty alignment CSV")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _markdown(summary: Mapping[str, Any]) -> str:
    blocks = summary["feature_blocks"]
    paths = summary["classification_paths"]
    lines = [
        "# D3 Proxy–Exact 对齐诊断",
        "",
        "- 样本数：{}".format(summary["sample_count"]),
        "- transition mask 完全一致率：{:.2%}".format(
            summary["transition_masks"]["exact_sample_match_rate"]
        ),
        "",
        "## 表示对齐",
        "",
        "| 特征块 | NRMSE | 中位逐维 Pearson | 类别效应方向一致率 |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "all",
        "spectral_delta",
        "spectral_speed",
        "gw_speed",
        "variation",
    ):
        item = blocks[name]
        correlation = item["median_dimension_pearson"]
        lines.append(
            "| {} | {:.6f} | {} | {:.2%} |".format(
                name,
                item["normalized_rmse"],
                (
                    "{:.6f}".format(correlation)
                    if correlation is not None
                    else "N/A"
                ),
                item["class_effect_direction_agreement"],
            )
        )
    lines.extend(
        [
            "",
            "## 四条冻结分类路径",
            "",
            "| 路径 | AUROC | BA | Accuracy | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "proxy_proxy",
        "exact_proxy",
        "exact_exact",
        "proxy_exact",
    ):
        item = paths[name]
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                name,
                (
                    "{:.6f}".format(item["roc_auc"])
                    if item["roc_auc"] is not None
                    else "N/A"
                ),
                item["balanced_accuracy"],
                item["accuracy"],
                item["f1"],
            )
        )
    probability = summary["probability_alignment"][
        "proxy_proxy_vs_exact_exact"
    ]
    lines.extend(
        [
            "",
            "## 判别排序传递",
            "",
            "- Proxy 与 Exact 概率 Spearman：{}".format(
                probability["spearman"]
            ),
            "- Proxy 与 Exact 概率 Kendall：{}".format(
                probability["kendall"]
            ),
            "- 冻结阈值预测不一致率：{:.2%}".format(
                probability["prediction_disagreement_rate"]
            ),
            "",
            "> 本报告是冻结产物的只读诊断，不包含重新训练或测试集调参。",
            "",
        ]
    )
    return "\n".join(lines)


def write_proxy_exact_alignment_artifacts(
    output_dir: Path,
    analysis: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / "summary.json",
        "summary_markdown": output_dir / "summary.md",
        "sample_alignment": output_dir / "sample_alignment.csv",
        "dimension_alignment": output_dir / "dimension_alignment.csv",
        "classification_paths": output_dir / "classification_paths.csv",
        "probability_alignment": output_dir / "probability_alignment.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "proxy–Exact alignment artifacts already exist"
        )
    _atomic_json(
        paths["summary_json"],
        {
            "artifact": "dual_proxy_exact_alignment",
            "schema_version": 1,
            "provenance": dict(provenance),
            **dict(analysis["summary"]),
        },
    )
    _write_csv(paths["sample_alignment"], analysis["sample_rows"])
    _write_csv(paths["dimension_alignment"], analysis["dimension_rows"])
    _write_csv(
        paths["classification_paths"], analysis["classification_rows"]
    )
    _write_csv(
        paths["probability_alignment"], analysis["probability_rows"]
    )
    markdown_path = paths["summary_markdown"]
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(
        _markdown(analysis["summary"]) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(markdown_path))
    return paths
