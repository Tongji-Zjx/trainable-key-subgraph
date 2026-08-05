"""Strict three-fold OOF summary for the frozen final multi-view fusion."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


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


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    output = (args.output_dir or root / "oof_summary_seed{}".format(args.seed)).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError("final multi-view OOF summary exists")
    assignments = _read(args.fold_assignments)
    if assignments.get("purpose") != "confirmatory_cross_fitted_fold_roles":
        raise ValueError("unexpected fold assignments purpose")
    if not assignments.get("immutable") or int(assignments.get("num_outer_folds", 0)) != 3:
        raise ValueError("final multi-view summary requires immutable three-fold assignments")
    assignment_rows = assignments["assignments"]
    expected = {
        fold: {
            row["sample_key"]: row
            for row in assignment_rows
            if int(row["outer_fold"]) == fold and row["role"] == "outer_test"
        }
        for fold in range(3)
    }
    rows, folds = [], []
    for fold in range(3):
        fold_root = root / "fold_{}".format(fold)
        spec = _read(fold_root / "fold_complete.json")
        if int(spec.get("outer_fold", -1)) != fold or int(spec.get("seed", -1)) != args.seed:
            raise ValueError("fold completion metadata mismatch")
        architecture = spec.get("architecture", {})
        if architecture.get("static_mode") != "residual" or architecture.get("v_mode") != "uot" or architecture.get("use_g") is not True:
            raise ValueError("fold did not use frozen S-residual/UOT/G architecture")
        validation = _read(fold_root / "fusion_seed{}".format(args.seed) / "validation_evaluation.json")
        test = _read(fold_root / "fusion_seed{}".format(args.seed) / "test_evaluation.json")
        if validation.get("split") != "validation" or test.get("split") != "test":
            raise ValueError("fold evaluation split mismatch")
        threshold = float(validation["threshold"])
        if abs(float(test["threshold"]) - threshold) > 1.0e-12:
            raise ValueError("fold test threshold was refit")
        predictions = {row["sample_key"]: row for row in test["predictions"]}
        if set(predictions) != set(expected[fold]):
            raise ValueError("fold outer-test prediction coverage mismatch")
        fold_rows = []
        for key in sorted(predictions):
            prediction = predictions[key]
            assignment = expected[fold][key]
            if int(prediction["label"]) != int(assignment["label"]):
                raise ValueError("fold prediction label mismatch")
            probability = float(prediction["positive_probability"])
            row = {
                "fold": fold,
                "sample_key": key,
                "site": assignment["site"],
                "label": int(assignment["label"]),
                "positive_probability": probability,
                "threshold": threshold,
                "predicted_label": int(probability >= threshold),
            }
            rows.append(row)
            fold_rows.append(row)
        labels = [row["label"] for row in fold_rows]
        scores = [row["positive_probability"] for row in fold_rows]
        sites = [row["site"] for row in fold_rows]
        classification = _classification_metrics(fold_rows)
        site_auc = _site_stratified_auc(labels, scores, sites)
        folds.append({
            "fold": fold,
            "sample_count": len(fold_rows),
            "roc_auc": _roc_auc(labels, scores),
            "site_stratified_roc_auc": site_auc["roc_auc"],
            "accuracy": classification["accuracy"],
            "balanced_accuracy": classification["balanced_accuracy"],
            "f1": classification["f1"],
            "threshold": threshold,
        })
    if len(rows) != 938 or len({row["sample_key"] for row in rows}) != 938:
        raise ValueError("OOF predictions must cover all 938 samples exactly once")
    rows.sort(key=lambda row: (row["fold"], row["sample_key"]))
    labels = [row["label"] for row in rows]
    scores = [row["positive_probability"] for row in rows]
    sites = [row["site"] for row in rows]
    classification = _classification_metrics(rows)
    site_auc = _site_stratified_auc(labels, scores, sites)
    fold_auc = [row["roc_auc"] for row in folds]
    metrics = {
        "sample_count": len(rows),
        "class_counts": dict(Counter(str(value) for value in labels)),
        "mean_fold_roc_auc": _mean_std(fold_auc),
        "pooled_oof_roc_auc": _roc_auc(labels, scores),
        "pooled_oof_site_stratified_roc_auc": site_auc["roc_auc"],
        "accuracy": classification["accuracy"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "f1": classification["f1"],
    }
    payload = {
        "artifact_type": "multiview_final_crossfit_oof_summary_v1",
        "seed": args.seed,
        "architecture": {
            "critical": "S_residual + UOT + G",
            "short_term": "author_no_coordinate_short_term",
            "fusion": "critical_representation_residual",
        },
        "primary_metric": "mean_fold_roc_auc",
        "folds": folds,
        "metrics": metrics,
        "checks": {
            "every_sample_predicted_once": True,
            "outer_test_assignments_match": True,
            "test_threshold_fitting": False,
            "architecture_frozen_before_outer_test": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    mean_auc = metrics["mean_fold_roc_auc"]
    lines = [
        "# 最终多视图融合架构 3-fold OOF",
        "",
        "- 固定关键分支：`S_residual + UOT + G`",
        "- 短期分支：`author_no_coordinate_short_term`",
        "- 训练种子：`{}`".format(args.seed),
        "- 主指标：三折 Mean AUROC",
        "- 每折阈值仅由该折 inner-validation 冻结",
        "",
        "| 指标 | 数值 |", "|---|---:|",
        "| Mean-fold AUROC | {:.6f} ± {:.6f} |".format(mean_auc["mean"], mean_auc["standard_deviation"]),
        "| Pooled OOF AUROC | {:.6f} |".format(metrics["pooled_oof_roc_auc"]),
        "| Site-stratified OOF AUROC | {:.6f} |".format(metrics["pooled_oof_site_stratified_roc_auc"]),
        "| Accuracy | {:.6f} |".format(metrics["accuracy"]),
        "| Balanced Accuracy | {:.6f} |".format(metrics["balanced_accuracy"]),
        "| F1 | {:.6f} |".format(metrics["f1"]),
        "", "| Fold | N | AUROC | Site-AUC | BA | Accuracy | F1 |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append("| {fold} | {sample_count} | {roc_auc:.6f} | {site_stratified_roc_auc:.6f} | {balanced_accuracy:.6f} | {accuracy:.6f} | {f1:.6f} |".format(**row))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
