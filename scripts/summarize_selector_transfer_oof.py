"""Strictly summarize Full-Soft-Hard selector outer-fold predictions."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    _classification_metrics,
    _roc_auc,
    _site_stratified_auc,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    assignment_path = root / "assignments" / "fold_assignments.json"
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    if (
        assignments.get("purpose")
        != "confirmatory_cross_fitted_fold_roles"
        or not assignments.get("immutable")
    ):
        raise ValueError("unsupported cross-fit assignments")
    num_folds = int(assignments["num_outer_folds"])
    rows = []
    folds = []
    for fold in range(num_folds):
        expected = {
            item["sample_key"]: item
            for item in assignments["assignments"]
            if int(item["outer_fold"]) == fold
            and item["role"] == "outer_test"
        }
        path = (
            root / "fold_{}".format(fold)
            / "selector_transfer_full_soft_hard_seed{}".format(args.seed)
            / "outer_fold_evaluation.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("selector_objective") != "full_soft_hard"
            or payload.get("test_used_for_fitting") is not False
            or payload["probe"].get("threshold_fit_split") != "validation"
        ):
            raise ValueError("selector OOF evaluation metadata mismatch")
        predictions = payload["evaluations"]["test"]["predictions"]
        actual = {item["sample_key"]: item for item in predictions}
        if len(actual) != len(predictions) or set(actual) != set(expected):
            raise ValueError("selector OOF outer-test coverage mismatch")
        threshold = float(payload["probe"]["threshold"])
        fold_rows = []
        for key in sorted(actual):
            prediction = actual[key]
            assignment = expected[key]
            if (
                int(prediction["label"]) != int(assignment["label"])
                or str(prediction["site"]) != str(assignment["site"])
            ):
                raise ValueError("selector OOF metadata mismatch")
            probability = float(prediction["positive_probability"])
            row = {
                "fold": fold,
                "sample_key": key,
                "site": str(prediction["site"]),
                "label": int(prediction["label"]),
                "positive_probability": probability,
                "threshold": threshold,
                "predicted_label": int(probability >= threshold),
            }
            fold_rows.append(row)
            rows.append(row)
        labels = [item["label"] for item in fold_rows]
        scores = [item["positive_probability"] for item in fold_rows]
        sites = [item["site"] for item in fold_rows]
        classification = _classification_metrics(fold_rows)
        site_auc = _site_stratified_auc(labels, scores, sites)
        folds.append({
            "fold": fold,
            "sample_count": len(fold_rows),
            "roc_auc": _roc_auc(labels, scores),
            "site_stratified_roc_auc": site_auc["roc_auc"],
            "balanced_accuracy": classification["balanced_accuracy"],
            "accuracy": classification["accuracy"],
            "f1": classification["f1"],
            "threshold": threshold,
            "evaluation": str(path),
        })
    if len({item["sample_key"] for item in rows}) != len(rows):
        raise ValueError("selector OOF samples are duplicated")
    labels = [item["label"] for item in rows]
    scores = [item["positive_probability"] for item in rows]
    sites = [item["site"] for item in rows]
    classification = _classification_metrics(rows)
    site_auc = _site_stratified_auc(labels, scores, sites)
    fold_aucs = [float(item["roc_auc"]) for item in folds]
    mean_auc = sum(fold_aucs) / float(len(fold_aucs))
    std_auc = math.sqrt(
        sum((value - mean_auc) ** 2 for value in fold_aucs)
        / float(len(fold_aucs))
    )
    metrics = {
        "sample_count": len(rows),
        "class_counts": dict(Counter(str(value) for value in labels)),
        "pooled_oof_roc_auc": _roc_auc(labels, scores),
        "pooled_oof_site_stratified_roc_auc": site_auc["roc_auc"],
        "mean_fold_roc_auc": mean_auc,
        "standard_deviation_fold_roc_auc": std_auc,
        **classification
    }
    result = {
        "artifact_type": "selector_transfer_full_soft_hard_oof_summary",
        "selector_objective": "full_soft_hard",
        "seed": int(args.seed),
        "num_outer_folds": num_folds,
        "test_used_for_fitting": False,
        "checks": {
            "outer_test_once_per_sample": True,
            "threshold_fit_split": "validation",
            "probe_fit_split": "train",
        },
        "folds": folds,
        "metrics": metrics,
    }
    output = (
        root / "oof_summary"
        / "selector_transfer_full_soft_hard_seed{}".format(args.seed)
    )
    targets = (
        output / "summary.json",
        output / "oof_predictions.csv",
        output / "summary.md",
    )
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("selector OOF summary exists")
    _write(targets[0], result)
    output.mkdir(parents=True, exist_ok=True)
    temporary = targets[1].with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(targets[1]))
    lines = [
        "# Full-Soft-Hard Selector 3-fold OOF",
        "",
        "- Probe fit: inner-train only",
        "- Threshold fit: inner-validation only",
        "- Outer-test used for fitting: no",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Pooled OOF AUROC | {:.6f} |".format(
            metrics["pooled_oof_roc_auc"]
        ),
        "| Site-stratified OOF AUROC | {} |".format(
            "N/A" if metrics["pooled_oof_site_stratified_roc_auc"] is None
            else "{:.6f}".format(
                metrics["pooled_oof_site_stratified_roc_auc"]
            )
        ),
        "| Mean fold AUROC | {:.6f} ± {:.6f} |".format(
            mean_auc, std_auc
        ),
        "| Accuracy | {:.6f} |".format(metrics["accuracy"]),
        "| Balanced Accuracy | {:.6f} |".format(
            metrics["balanced_accuracy"]
        ),
        "| F1 | {:.6f} |".format(metrics["f1"]),
        "",
    ]
    targets[2].write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("summary:", targets[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
