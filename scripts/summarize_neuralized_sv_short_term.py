"""Summarize paired 3-fold ST versus ST+neuralized-S/V predictions."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    _classification_metrics,
    _mean_std,
    _roc_auc,
    _site_stratified_auc,
)
from keysubgraph.models.neuralized_sv import NEURALIZED_SV_VARIANTS  # noqa: E402


def _read_json_predictions(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("short-term evaluation has no predictions")
    return rows


def _read_csv_predictions(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("fusion evaluation has no predictions")
    return rows


def _normalize(rows, fold):
    output = []
    seen = set()
    for row in rows:
        key = str(row["sample_key"])
        if key in seen:
            raise ValueError("paired summary found duplicate sample")
        seen.add(key)
        probability = float(row["positive_probability"])
        threshold = float(row["threshold"])
        output.append(
            {
                "fold": int(fold),
                "sample_key": key,
                "site": str(row["site"]),
                "label": int(row["label"]),
                "positive_probability": probability,
                "threshold": threshold,
                "predicted_label": int(probability >= threshold),
            }
        )
    return output


def _metrics(rows):
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["positive_probability"]) for row in rows]
    sites = [str(row["site"]) for row in rows]
    value = {
        "sample_count": len(rows),
        "roc_auc": _roc_auc(labels, scores),
        "site_stratified_roc_auc": _site_stratified_auc(
            labels, scores, sites
        )["roc_auc"],
    }
    value.update(_classification_metrics(rows))
    return value


def _paired_mean_fold_bootstrap(reference, candidate, repeats, seed):
    if set(reference) != set(candidate):
        raise ValueError("paired fusion predictions are not aligned")
    folds = sorted({int(row["fold"]) for row in reference.values()})
    strata = defaultdict(list)
    for key, row in reference.items():
        strata[(int(row["fold"]), str(row["site"]), int(row["label"]))].append(key)
    rng = np.random.RandomState(int(seed))
    values = []
    for _ in range(int(repeats)):
        selected = []
        for keys in strata.values():
            selected.extend(rng.choice(keys, size=len(keys), replace=True).tolist())
        deltas = []
        for fold in folds:
            current = [
                key for key in selected if int(reference[key]["fold"]) == fold
            ]
            labels = [int(reference[key]["label"]) for key in current]
            if len(set(labels)) < 2:
                continue
            reference_auc = _roc_auc(
                labels,
                [float(reference[key]["positive_probability"]) for key in current],
            )
            candidate_auc = _roc_auc(
                labels,
                [float(candidate[key]["positive_probability"]) for key in current],
            )
            deltas.append(float(candidate_auc) - float(reference_auc))
        if len(deltas) == len(folds):
            values.append(float(np.mean(deltas)))
    if not values:
        raise ValueError("paired fusion bootstrap produced no valid sample")
    array = np.asarray(values)
    interval = [
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    ]
    return {
        "repeats": int(len(array)),
        "mean": float(array.mean()),
        "confidence_interval_95": interval,
        "probability_positive": float(np.mean(array > 0.0)),
        "statistically_significant_positive": bool(interval[0] > 0.0),
    }


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--short-term-seed", type=int, required=True)
    parser.add_argument("--neural-seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=NEURALIZED_SV_VARIANTS,
        default=list(NEURALIZED_SV_VARIANTS),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with args.fold_assignments.open("r", encoding="utf-8") as handle:
        fold_count = int(json.load(handle)["num_outer_folds"])
    reference_rows = []
    reference_folds = []
    for fold in range(fold_count):
        path = (
            args.source_crossfit_root
            / "fold_{}".format(fold)
            / "author_short_term_no_coord"
            / "evaluation_seed{}".format(args.short_term_seed)
            / "test_evaluation.json"
        )
        rows = _normalize(_read_json_predictions(path), fold)
        reference_rows.extend(rows)
        reference_folds.append(_metrics(rows))
    reference_by_key = {row["sample_key"]: row for row in reference_rows}
    if len(reference_by_key) != len(reference_rows):
        raise ValueError("short-term OOF predictions contain duplicates")
    reference = {
        "pooled": _metrics(reference_rows),
        "folds": reference_folds,
        "mean_fold_roc_auc": _mean_std(
            [float(item["roc_auc"]) for item in reference_folds]
        ),
    }

    candidates = {}
    for offset, variant in enumerate(args.variants):
        rows_all = []
        folds = []
        for fold in range(fold_count):
            path = (
                args.fusion_root
                / "fold_{}".format(fold)
                / "{}_seed{}".format(variant, args.neural_seed)
                / "predictions.csv"
            )
            rows = _normalize(_read_csv_predictions(path), fold)
            rows_all.extend(rows)
            folds.append(_metrics(rows))
        by_key = {row["sample_key"]: row for row in rows_all}
        if len(by_key) != len(rows_all) or set(by_key) != set(reference_by_key):
            raise ValueError("fused OOF predictions do not align with short-term")
        for key in by_key:
            if (
                by_key[key]["label"] != reference_by_key[key]["label"]
                or by_key[key]["site"] != reference_by_key[key]["site"]
                or by_key[key]["fold"] != reference_by_key[key]["fold"]
            ):
                raise ValueError("fused and short-term OOF metadata differ")
        mean_fold = _mean_std([float(item["roc_auc"]) for item in folds])
        delta = float(mean_fold["mean"]) - float(
            reference["mean_fold_roc_auc"]["mean"]
        )
        bootstrap = _paired_mean_fold_bootstrap(
            reference_by_key,
            by_key,
            args.bootstrap_repeats,
            args.bootstrap_seed + offset,
        )
        candidates[variant] = {
            "pooled": _metrics(rows_all),
            "folds": folds,
            "mean_fold_roc_auc": mean_fold,
            "mean_fold_auc_delta_vs_short_term": delta,
            "paired_mean_fold_bootstrap": bootstrap,
            "acceptance": {
                "positive_mean_fold_delta": delta > 0.0,
                "minimum_practical_delta_0_01": delta >= 0.01,
                "significant_positive_95ci": bootstrap[
                    "statistically_significant_positive"
                ],
            },
        }
    primary = "NSV_safe_residual"
    result = {
        "artifact_type": "neuralized_sv_short_term_paired_oof_summary",
        "dataset": args.dataset,
        "primary_metric": "mean_fold_roc_auc",
        "primary_variant": primary,
        "short_term": reference,
        "candidates": candidates,
        "primary_acceptance_passed": bool(
            candidates[primary]["acceptance"]["significant_positive_95ci"]
        ),
        "selection_note": (
            "NSV_safe_residual is preregistered primary; NS and NV are mechanism ablations"
        ),
        "test_used_for_training_or_fusion_fit": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "summary.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError("corrected neural S/V summary exists")
    _atomic_json(target, result)
    lines = [
        "# {}：短期分支与神经化 S/V 配对 OOF\n".format(args.dataset),
        "- 主指标：mean-fold AUROC",
        "- 主候选：`{}`".format(primary),
        "- 融合参数：每折仅由 inner-validation 拟合",
        "- Outer-test 用于训练或调参：否\n",
        "| 模型 | Mean-fold AUC | Δ vs ST | Pooled AUC | Site-AUC | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
        "| ST | {:.6f} | — | {:.6f} | {:.6f} | — |".format(
            reference["mean_fold_roc_auc"]["mean"],
            reference["pooled"]["roc_auc"],
            reference["pooled"]["site_stratified_roc_auc"],
        ),
    ]
    for variant in args.variants:
        item = candidates[variant]
        interval = item["paired_mean_fold_bootstrap"]["confidence_interval_95"]
        lines.append(
            "| {} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | [{:+.6f}, {:+.6f}] |".format(
                variant,
                item["mean_fold_roc_auc"]["mean"],
                item["mean_fold_auc_delta_vs_short_term"],
                item["pooled"]["roc_auc"],
                item["pooled"]["site_stratified_roc_auc"],
                interval[0],
                interval[1],
            )
        )
    lines.extend(
        [
            "",
            "主候选显著增益验收：**{}**。".format(
                "通过" if result["primary_acceptance_passed"] else "未通过"
            ),
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("corrected neural S/V summary:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

