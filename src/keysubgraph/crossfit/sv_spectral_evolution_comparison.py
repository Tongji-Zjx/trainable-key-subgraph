"""Paired OOF comparison of S, S+E, and fixed time-shuffled S+E."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from keysubgraph.crossfit.sv_branch_ablation import (
    _auc,
    _paired_statistics,
)
from keysubgraph.crossfit.sv_signed_gin_summary import (
    _classification_metrics,
    _site_stratified_auc,
)


def _read_summary(directory: Path, variant: str):
    directory = Path(directory).resolve()
    with (directory / "summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    if (
        summary.get("artifact_type")
        != "sv_signed_gin_crossfit_oof_summary"
        or summary.get("variant") != variant
        or not summary.get("checks", {}).get(
            "every_sample_predicted_once"
        )
    ):
        raise ValueError("invalid spectral evolution OOF summary")
    with (directory / "oof_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_key = {str(row["sample_key"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate spectral evolution OOF sample")
    return summary, by_key


def _metrics(keys, rows):
    labels = np.asarray(
        [int(rows[key]["label"]) for key in keys], dtype=np.int64
    )
    scores = np.asarray(
        [float(rows[key]["positive_probability"]) for key in keys],
        dtype=np.float64,
    )
    sites = [str(rows[key]["site"]) for key in keys]
    classification = [
        {
            "label": int(rows[key]["label"]),
            "predicted_label": int(rows[key]["predicted_label"]),
        }
        for key in keys
    ]
    site = _site_stratified_auc(
        labels.tolist(), scores.tolist(), sites
    )
    return {
        "pooled_oof_roc_auc": _auc(labels, scores),
        "site_stratified_oof_roc_auc": site["roc_auc"],
        **_classification_metrics(classification),
    }, labels, scores


def compare_sv_spectral_evolution(
    datasets: Sequence[Tuple[str, Path, Path, Path]],
    output_dir: Path,
    bootstrap_repeats: int = 10000,
    permutation_repeats: int = 10000,
    seed: int = 202607,
    overwrite: bool = False,
) -> Dict:
    """Compare frozen S with neural E and its order-shuffled inference."""

    if not datasets:
        raise ValueError("spectral evolution comparison requires datasets")
    output_dir = Path(output_dir).resolve()
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    if (
        (json_path.exists() or markdown_path.exists())
        and not overwrite
    ):
        raise FileExistsError("spectral evolution comparison exists")
    variants = {
        "S": "static_spectral_only",
        "SE": "static_spectral_neural_evolution",
        "SE_shuffled": "static_spectral_neural_evolution_time_shuffled",
    }
    results = {}
    for dataset_index, (name, s_dir, se_dir, shuffled_dir) in enumerate(
        datasets
    ):
        if name in results:
            raise ValueError("duplicate spectral evolution dataset")
        directories = {
            "S": s_dir,
            "SE": se_dir,
            "SE_shuffled": shuffled_dir,
        }
        rows = {
            key: _read_summary(directories[key], variants[key])[1]
            for key in directories
        }
        keys = sorted(rows["S"])
        if any(set(current) != set(keys) for current in rows.values()):
            raise ValueError("S/SE/shuffled OOF sample sets disagree")
        for sample_key in keys:
            metadata = rows["S"][sample_key]
            for condition in ("SE", "SE_shuffled"):
                if any(
                    rows[condition][sample_key][field] != metadata[field]
                    for field in ("fold", "site", "label")
                ):
                    raise ValueError("paired OOF metadata disagree")
        metrics = {}
        probabilities = {}
        labels = None
        for condition in ("S", "SE", "SE_shuffled"):
            current, current_labels, scores = _metrics(
                keys, rows[condition]
            )
            metrics[condition] = current
            probabilities[condition] = scores
            labels = current_labels if labels is None else labels
        contrasts = {
            "SE_minus_S": _paired_statistics(
                labels,
                probabilities["SE"],
                probabilities["S"],
                bootstrap_repeats,
                permutation_repeats,
                seed + dataset_index * 100,
            ),
            "SE_minus_shuffled": _paired_statistics(
                labels,
                probabilities["SE"],
                probabilities["SE_shuffled"],
                bootstrap_repeats,
                permutation_repeats,
                seed + dataset_index * 100 + 1,
            ),
        }
        results[name] = {
            "sample_count": len(keys),
            "models": metrics,
            "contrasts": contrasts,
            "source_summaries": {
                key: str(Path(value).resolve())
                for key, value in directories.items()
            },
        }
    payload = {
        "artifact_type": "sv_spectral_evolution_paired_oof_comparison",
        "models": variants,
        "seed": int(seed),
        "datasets": results,
        "interpretation_rule": {
            "SE_minus_S_positive": (
                "learned dynamic evolution adds OOF discrimination"
            ),
            "SE_minus_shuffled_positive": (
                "temporal order contributes beyond transition inventory"
            ),
        },
    }
    lines = [
        "# Static-spectral + Neural Evolution 配对 OOF 比较",
        "",
        "- S：冻结的 Static-spectral 基线",
        "- SE：S + 有符号谱差分 Masked Residual TCN",
        "- SE-shuffled：冻结同一 SE，仅固定打乱每段时间顺序",
        "- 所有比较使用相同外折样本和各折 inner-validation 冻结阈值。",
        "",
    ]
    for name, result in results.items():
        lines.extend(
            [
                "## {}".format(name),
                "",
                "| 路径 | OOF AUC | Site-AUC | BA | Accuracy | F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in ("S", "SE", "SE_shuffled"):
            values = result["models"][condition]
            lines.append(
                "| {condition} | {auc:.6f} | {site:.6f} | "
                "{ba:.6f} | {accuracy:.6f} | {f1:.6f} |".format(
                    condition=condition,
                    auc=values["pooled_oof_roc_auc"],
                    site=values["site_stratified_oof_roc_auc"],
                    ba=values["balanced_accuracy"],
                    accuracy=values["accuracy"],
                    f1=values["f1"],
                )
            )
        lines.extend(
            [
                "",
                "| 配对比较 | ΔAUC | Bootstrap 95% CI | 置换 p |",
                "|---|---:|---:|---:|",
            ]
        )
        for contrast in ("SE_minus_S", "SE_minus_shuffled"):
            values = result["contrasts"][contrast]
            lines.append(
                "| {contrast} | {delta:+.6f} | "
                "[{lower:+.6f}, {upper:+.6f}] | {p:.6f} |".format(
                    contrast=contrast,
                    delta=values["auc_difference"],
                    lower=values["bootstrap_95_ci"][0],
                    upper=values["bootstrap_95_ci"][1],
                    p=values["paired_swap_two_sided_p"],
                )
            )
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(json_path))
    markdown_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "comparison_json": json_path,
        "comparison_markdown": markdown_path,
        "datasets": results,
    }
