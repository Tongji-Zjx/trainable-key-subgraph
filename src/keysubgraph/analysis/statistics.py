"""Sample-level Mann-Whitney tests, BH-FDR, discrepancy, and CSV output."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import mannwhitneyu

from .structural_metrics import METRIC_NAMES, aggregate_sample_metrics, compute_subgraph_metrics


def apply_bh_fdr(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def _cliffs_delta(class_zero: np.ndarray, class_one: np.ndarray) -> float:
    comparisons = class_one[:, None] - class_zero[None, :]
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def _iqr(values: np.ndarray) -> float:
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def run_univariate_tests(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    sources = sorted(set(row["source"] for row in sample_rows))
    for source in sources:
        source_rows = [row for row in sample_rows if row["source"] == source]
        source_results = []
        for metric in METRIC_NAMES:
            class_values = {}
            for label in (0, 1):
                class_values[label] = np.asarray(
                    [
                        float(row[metric])
                        for row in source_rows
                        if int(row["label"]) == label
                        and math.isfinite(float(row[metric]))
                    ],
                    dtype=np.float64,
                )
            zero, one = class_values[0], class_values[1]
            if zero.size == 0 or one.size == 0:
                p_value = 1.0
                u_statistic = float("nan")
                cliffs = float("nan")
            else:
                test = mannwhitneyu(zero, one, alternative="two-sided")
                u_statistic = float(test.statistic)
                p_value = float(test.pvalue)
                cliffs = _cliffs_delta(zero, one)
            mean_zero = float(np.mean(zero)) if zero.size else float("nan")
            mean_one = float(np.mean(one)) if one.size else float("nan")
            std_zero = float(np.std(zero, ddof=1)) if zero.size > 1 else 0.0
            std_one = float(np.std(one, ddof=1)) if one.size > 1 else 0.0
            discrepancy = (
                abs(mean_one - mean_zero) / (std_zero + std_one + 1e-8)
                if zero.size and one.size
                else float("nan")
            )
            median_zero = float(np.median(zero)) if zero.size else float("nan")
            median_one = float(np.median(one)) if one.size else float("nan")
            source_results.append(
                {
                    "source": source,
                    "metric": metric,
                    "class_0_valid_samples": int(zero.size),
                    "class_1_valid_samples": int(one.size),
                    "class_0_median": median_zero,
                    "class_1_median": median_one,
                    "class_0_iqr": _iqr(zero) if zero.size else float("nan"),
                    "class_1_iqr": _iqr(one) if one.size else float("nan"),
                    "class_0_mean": mean_zero,
                    "class_1_mean": mean_one,
                    "class_0_std": std_zero,
                    "class_1_std": std_one,
                    "u_statistic": u_statistic,
                    "p_value": p_value,
                    "direction": (
                        1 if median_one > median_zero else -1 if median_one < median_zero else 0
                    ) if zero.size and one.size else 0,
                    "standardized_discrepancy": discrepancy,
                    "cliffs_delta": cliffs,
                }
            )
        q_values = apply_bh_fdr([row["p_value"] for row in source_results])
        for row, q_value in zip(source_results, q_values):
            row["q_value"] = q_value
            row["significant_fdr_0_05"] = q_value < 0.05
        results.extend(source_results)
    return results


def summarize_sources(test_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in test_rows:
        groups[row["source"]].append(row)
    summaries = []
    for source, rows in sorted(groups.items()):
        discrepancies = [
            float(row["standardized_discrepancy"])
            for row in rows
            if math.isfinite(float(row["standardized_discrepancy"]))
        ]
        summaries.append(
            {
                "source": source,
                "significant_metric_count": sum(bool(row["significant_fdr_0_05"]) for row in rows),
                "mean_q_value": sum(float(row["q_value"]) for row in rows) / len(rows),
                "delta_total": sum(discrepancies) / len(discrepancies) if discrepancies else float("nan"),
            }
        )
    return summaries


def _atomic_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_structural_analysis(
    subgraph_records: Iterable[Dict[str, Any]], output_dir: Path
) -> Dict[str, Path]:
    subgraph_rows = [compute_subgraph_metrics(record) for record in subgraph_records]
    if not subgraph_rows:
        raise ValueError("no valid subgraphs were provided")
    sample_rows = aggregate_sample_metrics(subgraph_rows)
    test_rows = run_univariate_tests(sample_rows)
    comparison_rows = summarize_sources(test_rows)
    output_dir = Path(output_dir).resolve()
    paths = {
        "subgraph_metrics": output_dir / "subgraph_level_metrics.csv",
        "sample_metrics": output_dir / "sample_level_metrics.csv",
        "tests": output_dir / "univariate_test_results.csv",
        "comparison": output_dir / "control_group_comparison.csv",
        "summary": output_dir / "analysis_summary.json",
    }
    _atomic_csv(paths["subgraph_metrics"], subgraph_rows)
    _atomic_csv(paths["sample_metrics"], sample_rows)
    _atomic_csv(paths["tests"], test_rows)
    _atomic_csv(paths["comparison"], comparison_rows)
    summary = {
        "subgraph_count": len(subgraph_rows),
        "sample_source_count": len(sample_rows),
        "sources": comparison_rows,
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["summary"].with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(paths["summary"]))
    return paths
