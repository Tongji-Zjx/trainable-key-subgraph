"""Paired OOF comparison for nested S, SV, and SVG classifiers."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from .sv_signed_gin_summary import (
    _classification_metrics,
    _site_stratified_auc,
)


SV_BRANCH_ABLATION_VARIANTS = {
    "S": "static_spectral_only",
    "SV": "static_spectral_variation_late_fusion",
    "SVG": "signed_gin_multibranch_late_fusion",
}
SV_BRANCH_ABLATION_CONTRASTS = (
    ("SV_minus_S", "SV", "S"),
    ("SVG_minus_SV", "SVG", "SV"),
    ("SVG_minus_S", "SVG", "S"),
)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives < 1 or negatives < 1:
        raise ValueError("paired OOF AUC requires both classes")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels == 1].sum())
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values), probability))


def _read_oof_summary(directory: Path, expected_variant: str):
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    predictions_path = directory / "oof_predictions.csv"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if (
        summary.get("artifact_type")
        != "sv_signed_gin_crossfit_oof_summary"
        or summary.get("variant") != expected_variant
        or not summary.get("checks", {}).get(
            "every_sample_predicted_once"
        )
    ):
        raise ValueError("invalid S/SV/SVG OOF summary")
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary["metrics"]["sample_count"]):
        raise ValueError("OOF summary and prediction count disagree")
    by_key = {}
    for row in rows:
        key = str(row["sample_key"])
        if key in by_key:
            raise ValueError("duplicate sample in OOF predictions")
        by_key[key] = {
            "fold": int(row["fold"]),
            "site": str(row["site"]),
            "label": int(row["label"]),
            "positive_probability": float(
                row["positive_probability"]
            ),
            "threshold": float(row["threshold"]),
            "predicted_label": int(row["predicted_label"]),
        }
    return summary, by_key


def _paired_statistics(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    if bootstrap_repeats < 1 or permutation_repeats < 1:
        raise ValueError("paired repeat counts must be positive")
    observed = _auc(labels, left) - _auc(labels, right)
    rng = np.random.RandomState(int(seed))
    by_class = (
        np.flatnonzero(labels == 0),
        np.flatnonzero(labels == 1),
    )
    bootstrap = []
    for _ in range(int(bootstrap_repeats)):
        indices = np.concatenate(
            [
                rng.choice(values, size=len(values), replace=True)
                for values in by_class
            ]
        )
        bootstrap.append(
            _auc(labels[indices], left[indices])
            - _auc(labels[indices], right[indices])
        )
    extreme = 0
    for _ in range(int(permutation_repeats)):
        swap = rng.rand(len(labels)) < 0.5
        permuted_left = np.where(swap, right, left)
        permuted_right = np.where(swap, left, right)
        difference = _auc(labels, permuted_left) - _auc(
            labels, permuted_right
        )
        if abs(difference) >= abs(observed) - 1.0e-15:
            extreme += 1
    lower = _quantile(bootstrap, 0.025)
    upper = _quantile(bootstrap, 0.975)
    return {
        "auc_difference": observed,
        "bootstrap_95_ci": [lower, upper],
        "paired_swap_two_sided_p": (
            (extreme + 1.0) / float(permutation_repeats + 1)
        ),
        "bootstrap_repeats": int(bootstrap_repeats),
        "permutation_repeats": int(permutation_repeats),
        "direction": (
            "positive"
            if lower > 0.0
            else ("negative" if upper < 0.0 else "inconclusive")
        ),
    }


def compare_sv_branch_ablation(
    datasets: Sequence[Tuple[str, Path, Path, Path]],
    output_dir: Path,
    bootstrap_repeats: int = 10000,
    permutation_repeats: int = 10000,
    seed: int = 202607,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Compare nested S/SV/SVG OOF predictions within each dataset."""

    if not datasets:
        raise ValueError("at least one ablation dataset is required")
    output_dir = Path(output_dir).resolve()
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    if (
        (json_path.exists() or markdown_path.exists())
        and not overwrite
    ):
        raise FileExistsError("S/SV/SVG comparison output exists")
    results = {}
    for dataset_index, item in enumerate(datasets):
        name, s_dir, sv_dir, svg_dir = item
        if name in results:
            raise ValueError("duplicate ablation dataset name")
        directories = {"S": s_dir, "SV": sv_dir, "SVG": svg_dir}
        summaries = {}
        predictions = {}
        for model_name, directory in directories.items():
            summaries[model_name], predictions[model_name] = (
                _read_oof_summary(
                    directory,
                    SV_BRANCH_ABLATION_VARIANTS[model_name],
                )
            )
        keys = sorted(predictions["S"])
        if any(set(predictions[value]) != set(keys) for value in predictions):
            raise ValueError("S/SV/SVG OOF sample sets disagree")
        rows = []
        for key in keys:
            metadata = predictions["S"][key]
            for model_name in ("SV", "SVG"):
                current = predictions[model_name][key]
                for field in ("fold", "site", "label"):
                    if current[field] != metadata[field]:
                        raise ValueError(
                            "paired OOF sample metadata disagree"
                        )
            rows.append(metadata)
        labels = np.asarray(
            [row["label"] for row in rows], dtype=np.int64
        )
        sites = [row["site"] for row in rows]
        metrics = {}
        probabilities = {}
        for model_name in ("S", "SV", "SVG"):
            values = np.asarray(
                [
                    predictions[model_name][key][
                        "positive_probability"
                    ]
                    for key in keys
                ],
                dtype=np.float64,
            )
            probabilities[model_name] = values
            classification_rows = [
                {
                    "label": predictions[model_name][key]["label"],
                    "predicted_label": predictions[model_name][key][
                        "predicted_label"
                    ],
                }
                for key in keys
            ]
            site_auc = _site_stratified_auc(
                labels.tolist(), values.tolist(), sites
            )
            metrics[model_name] = {
                "pooled_oof_roc_auc": _auc(labels, values),
                "site_stratified_oof_roc_auc": site_auc["roc_auc"],
                "eligible_site_count": site_auc[
                    "eligible_site_count"
                ],
                **_classification_metrics(classification_rows),
            }
        contrasts = {}
        for contrast_index, (contrast, left, right) in enumerate(
            SV_BRANCH_ABLATION_CONTRASTS
        ):
            contrasts[contrast] = _paired_statistics(
                labels,
                probabilities[left],
                probabilities[right],
                bootstrap_repeats,
                permutation_repeats,
                seed + dataset_index * 100 + contrast_index,
            )
        results[name] = {
            "sample_count": len(keys),
            "models": metrics,
            "contrasts": contrasts,
            "source_summaries": {
                model_name: str(Path(directory).resolve())
                for model_name, directory in directories.items()
            },
        }
    payload = {
        "artifact_type": "sv_branch_ablation_paired_oof_comparison",
        "models": dict(SV_BRANCH_ABLATION_VARIANTS),
        "nested_design": {
            "S": "16-dimensional static-spectral only",
            "SV": "S plus 16-dimensional variation",
            "SVG": "SV plus SignedGIN",
        },
        "seed": int(seed),
        "datasets": results,
    }
    lines = [
        "# S / SV / SVG 配对 OOF 比较",
        "",
        "- S：Static-spectral only",
        "- SV：Static-spectral + Variation",
        "- SVG：Static-spectral + Variation + SignedGIN",
        "- 三者使用同一外折、同一样本和各折 inner-validation 冻结阈值。",
        "",
    ]
    for name, result in results.items():
        lines.extend(
            [
                "## {}".format(name),
                "",
                "| 模型 | Pooled OOF AUC | Site-AUC | BA | Accuracy | F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name in ("S", "SV", "SVG"):
            values = result["models"][model_name]
            lines.append(
                "| {name} | {auc:.6f} | {site:.6f} | {ba:.6f} | "
                "{accuracy:.6f} | {f1:.6f} |".format(
                    name=model_name,
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
                "| 配对对比 | ΔAUC | Bootstrap 95% CI | 置换 p | 方向 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for contrast, _, _ in SV_BRANCH_ABLATION_CONTRASTS:
            values = result["contrasts"][contrast]
            lines.append(
                "| {name} | {delta:+.6f} | [{lower:+.6f}, "
                "{upper:+.6f}] | {p:.6f} | {direction} |".format(
                    name=contrast,
                    delta=values["auc_difference"],
                    lower=values["bootstrap_95_ci"][0],
                    upper=values["bootstrap_95_ci"][1],
                    p=values["paired_swap_two_sided_p"],
                    direction=values["direction"],
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
