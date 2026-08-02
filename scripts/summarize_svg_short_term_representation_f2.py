"""Summarize promoted representation-level F2 three-fold OOF results."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.structured_short_term_summary import (  # noqa: E402
    _expected_roles,
)
from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    _classification_metrics,
    _roc_auc,
    _site_stratified_auc,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--fusion-seed", type=int, default=42)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    targets = (output / "summary.json", output / "oof_predictions.csv", output / "summary.md")
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("representation F2 OOF summary exists")
    assignments = _load(args.fold_assignments)
    if (
        assignments.get("purpose")
        != "confirmatory_cross_fitted_fold_roles"
        or not bool(assignments.get("immutable"))
    ):
        raise ValueError("unsupported representation F2 fold assignments")
    fold_count = int(assignments["num_outer_folds"])
    expected, expected_outer = _expected_roles(assignments, fold_count)
    rows = []
    folds = []
    for fold in range(fold_count):
        path = (
            args.output_root.resolve()
            / "fold_{}".format(fold)
            / "model_seed{}".format(args.fusion_seed)
            / "test_evaluation.json"
        )
        result = _load(path)
        if (
            result.get("artifact_type")
            != "svg_short_term_representation_f2_evaluation"
            or result.get("split") != "test"
            or result.get("threshold_fit_split") != "validation"
        ):
            raise ValueError("invalid representation F2 fold evaluation")
        threshold = float(result["threshold"])
        predictions = result["metrics"].get("predictions", [])
        by_key = {str(row["sample_key"]): row for row in predictions}
        if set(by_key) != set(expected[fold]["outer_test"]):
            raise ValueError("representation F2 outer-test assignments mismatch")
        fold_rows = []
        for key in sorted(by_key):
            row = by_key[key]
            probability = float(row["positive_probability"])
            current = {
                "fold": fold,
                "sample_key": key,
                "site": str(row["site"]),
                "label": int(row["label"]),
                "positive_probability": probability,
                "anchor_positive_probability": float(
                    row["anchor_positive_probability"]
                ),
                "threshold": threshold,
                "predicted_label": int(probability >= threshold),
            }
            fold_rows.append(current)
            rows.append(current)
        labels = [row["label"] for row in fold_rows]
        scores = [row["positive_probability"] for row in fold_rows]
        anchor_scores = [
            row["anchor_positive_probability"] for row in fold_rows
        ]
        sites = [row["site"] for row in fold_rows]
        classification = _classification_metrics(fold_rows)
        site = _site_stratified_auc(labels, scores, sites)
        folds.append(
            {
                "fold": fold,
                "sample_count": len(fold_rows),
                "roc_auc": _roc_auc(labels, scores),
                "anchor_roc_auc": _roc_auc(labels, anchor_scores),
                "site_stratified_roc_auc": site["roc_auc"],
                "balanced_accuracy": classification["balanced_accuracy"],
                "accuracy": classification["accuracy"],
                "f1": classification["f1"],
                "threshold": threshold,
                "gate": float(result["metrics"]["gate"]),
            }
        )
    if (
        len(rows) != len(expected_outer)
        or len({row["sample_key"] for row in rows}) != len(rows)
        or {row["sample_key"] for row in rows} != set(expected_outer)
    ):
        raise ValueError("representation F2 OOF coverage mismatch")
    rows.sort(key=lambda row: (int(row["fold"]), row["sample_key"]))
    labels = [row["label"] for row in rows]
    scores = [row["positive_probability"] for row in rows]
    anchor_scores = [row["anchor_positive_probability"] for row in rows]
    sites = [row["site"] for row in rows]
    classification = _classification_metrics(rows)
    site = _site_stratified_auc(labels, scores, sites)
    fold_auc = [float(row["roc_auc"]) for row in folds]
    anchor_fold_auc = [float(row["anchor_roc_auc"]) for row in folds]
    metrics = {
        "sample_count": len(rows),
        "mean_fold_roc_auc": statistics.mean(fold_auc),
        "fold_roc_auc_population_sd": statistics.pstdev(fold_auc),
        "mean_fold_anchor_roc_auc": statistics.mean(anchor_fold_auc),
        "mean_fold_delta_vs_anchor": statistics.mean(fold_auc)
        - statistics.mean(anchor_fold_auc),
        "pooled_oof_roc_auc": _roc_auc(labels, scores),
        "pooled_anchor_roc_auc": _roc_auc(labels, anchor_scores),
        "pooled_oof_site_stratified_roc_auc": site["roc_auc"],
        "mean_gate": statistics.mean(float(row["gate"]) for row in folds),
    }
    metrics.update(classification)
    payload = {
        "artifact_type": "svg_short_term_representation_f2_oof_summary",
        "dataset": args.dataset,
        "primary_metric": "mean_outer_fold_roc_auc",
        "fusion_seed": int(args.fusion_seed),
        "folds": folds,
        "metrics": metrics,
        "checks": {
            "every_sample_predicted_once": True,
            "outer_test_assignments_match": True,
            "test_threshold_fitting": False,
            "frozen_base_encoders": True,
        },
    }
    _atomic_json(targets[0], payload)
    output.mkdir(parents=True, exist_ok=True)
    with targets[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# {} 表示级 F2 三折 OOF".format(args.dataset),
        "",
        "- G2 logit：冻结锚点",
        "- 短期分支：冻结隐藏表示",
        "- 仅训练：零初始化残差头与全局门控",
        "- 主指标：三折 outer-test AUROC 算术平均",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        "| Mean-fold AUROC | {:.6f} ± {:.6f} |".format(
            metrics["mean_fold_roc_auc"],
            metrics["fold_roc_auc_population_sd"],
        ),
        "| G2 anchor Mean-fold AUROC | {:.6f} |".format(
            metrics["mean_fold_anchor_roc_auc"]
        ),
        "| ΔAUC vs G2 anchor | {:+.6f} |".format(
            metrics["mean_fold_delta_vs_anchor"]
        ),
        "| Pooled OOF AUROC | {:.6f} |".format(
            metrics["pooled_oof_roc_auc"]
        ),
        "| Site-stratified OOF AUROC | {} |".format(
            "N/A"
            if metrics["pooled_oof_site_stratified_roc_auc"] is None
            else "{:.6f}".format(
                metrics["pooled_oof_site_stratified_roc_auc"]
            )
        ),
        "| Accuracy | {:.6f} |".format(metrics["accuracy"]),
        "| Balanced Accuracy | {:.6f} |".format(metrics["balanced_accuracy"]),
        "| F1 | {:.6f} |".format(metrics["f1"]),
        "| Mean gate | {:.6f} |".format(metrics["mean_gate"]),
        "",
        "| Fold | AUC | Anchor AUC | Site-AUC | BA | Accuracy | F1 | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            "| {fold} | {roc_auc:.6f} | {anchor_roc_auc:.6f} | {site} | "
            "{balanced_accuracy:.6f} | {accuracy:.6f} | {f1:.6f} | "
            "{gate:.6f} |".format(
                site=(
                    "N/A"
                    if row["site_stratified_roc_auc"] is None
                    else "{:.6f}".format(
                        row["site_stratified_roc_auc"]
                    )
                ),
                **row
            )
        )
    targets[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(targets[2].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
