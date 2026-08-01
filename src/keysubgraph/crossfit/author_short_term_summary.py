"""Strict OOF summary for the author short-term reproduction."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from keysubgraph.crossfit.author_short_term_runner import (
    AUTHOR_SHORT_TERM_BRANCH,
)
from keysubgraph.crossfit.structured_short_term_summary import (
    _expected_roles,
    _load_json,
    _validate_evaluation,
)
from keysubgraph.crossfit.sv_signed_gin_summary import (
    _classification_metrics,
    _mean_std,
    _roc_auc,
    _sha256,
    _site_stratified_auc,
)
from keysubgraph.models.author_short_term import AUTHOR_SHORT_TERM_MODEL_NAME
from keysubgraph.training.author_short_term_trainer import (
    author_short_term_training_config,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def summarize_author_short_term_crossfit(
    output_root: Path,
    fold_assignments: Path,
    profile: str,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    fold_assignments = Path(fold_assignments).resolve()
    frozen_seed = author_short_term_training_config(
        profile, seed=seed
    ).seed
    if output_dir is None:
        output_dir = (
            output_root
            / "oof_summary"
            / "author_short_term_no_coord_{}_seed{}".format(
                profile, frozen_seed
            )
        )
    output_dir = Path(output_dir).resolve()
    outputs = (
        output_dir / "summary.json",
        output_dir / "oof_predictions.csv",
        output_dir / "summary.md",
    )
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError("author short-term OOF summary exists")
    assignments = _load_json(fold_assignments)
    if (
        assignments.get("purpose")
        != "confirmatory_cross_fitted_fold_roles"
        or not assignments.get("immutable")
    ):
        raise ValueError("unsupported cross-fit assignments")
    fold_count = int(assignments["num_outer_folds"])
    expected, expected_outer = _expected_roles(assignments, fold_count)
    rows: List[Dict[str, Any]] = []
    folds = []
    evaluation_hashes = {}
    for fold in range(fold_count):
        evaluation_dir = (
            output_root
            / "fold_{}".format(fold)
            / AUTHOR_SHORT_TERM_BRANCH
            / "evaluation_seed{}".format(frozen_seed)
        )
        validation_path = evaluation_dir / "validation_evaluation.json"
        test_path = evaluation_dir / "test_evaluation.json"
        validation = _load_json(validation_path)
        if validation.get("profile") != profile:
            raise ValueError("validation profile mismatch")
        threshold, _ = _validate_evaluation(
            validation,
            "validation",
            expected[fold]["inner_validation"],
            expected_model_name=AUTHOR_SHORT_TERM_MODEL_NAME,
        )
        test = _load_json(test_path)
        if test.get("profile") != profile:
            raise ValueError("test profile mismatch")
        _, predictions = _validate_evaluation(
            test,
            "test",
            expected[fold]["outer_test"],
            expected_model_name=AUTHOR_SHORT_TERM_MODEL_NAME,
            expected_threshold=threshold,
        )
        fold_rows = []
        for sample_key in sorted(predictions):
            prediction = predictions[sample_key]
            probability = float(prediction["positive_probability"])
            row = {
                "fold": fold,
                "sample_key": sample_key,
                "site": str(prediction["site"]),
                "label": int(prediction["label"]),
                "positive_probability": probability,
                "threshold": threshold,
                "predicted_label": int(probability >= threshold),
            }
            fold_rows.append(row)
            rows.append(row)
        labels = [row["label"] for row in fold_rows]
        scores = [row["positive_probability"] for row in fold_rows]
        sites = [row["site"] for row in fold_rows]
        classification = _classification_metrics(fold_rows)
        site_auc = _site_stratified_auc(labels, scores, sites)
        folds.append(
            {
                "fold": fold,
                "sample_count": len(fold_rows),
                "class_counts": dict(
                    Counter(str(value) for value in labels)
                ),
                "roc_auc": _roc_auc(labels, scores),
                "site_stratified_roc_auc": site_auc["roc_auc"],
                "balanced_accuracy": classification[
                    "balanced_accuracy"
                ],
                "accuracy": classification["accuracy"],
                "f1": classification["f1"],
                "threshold": threshold,
            }
        )
        evaluation_hashes[str(fold)] = {
            "validation": _sha256(validation_path),
            "outer_test": _sha256(test_path),
        }
    if len(rows) != len(expected_outer):
        raise ValueError("OOF row count mismatch")
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("OOF predictions contain duplicates")
    if {row["sample_key"] for row in rows} != set(expected_outer):
        raise ValueError("OOF sample coverage mismatch")
    rows.sort(key=lambda row: (row["fold"], row["sample_key"]))
    labels = [row["label"] for row in rows]
    scores = [row["positive_probability"] for row in rows]
    sites = [row["site"] for row in rows]
    classification = _classification_metrics(rows)
    site_auc = _site_stratified_auc(labels, scores, sites)
    fold_auc = [float(row["roc_auc"]) for row in folds]
    metrics = {
        "sample_count": len(rows),
        "class_counts": dict(Counter(str(value) for value in labels)),
        "pooled_oof_roc_auc": _roc_auc(labels, scores),
        "pooled_oof_site_stratified_roc_auc": site_auc["roc_auc"],
        "eligible_site_count": site_auc["eligible_site_count"],
        "eligible_site_pair_count": site_auc["eligible_pair_count"],
        "outer_fold_roc_auc": _mean_std(fold_auc),
    }
    metrics.update(classification)
    payload = {
        "artifact_type": "author_short_term_crossfit_oof_summary_v1",
        "model_name": AUTHOR_SHORT_TERM_MODEL_NAME,
        "profile": profile,
        "seed": frozen_seed,
        "num_outer_folds": fold_count,
        "fold_assignments": str(fold_assignments),
        "fold_assignments_sha256": _sha256(fold_assignments),
        "evaluation_sha256": evaluation_hashes,
        "threshold_policy": (
            "each fold freezes the author grid threshold on inner-validation"
        ),
        "checks": {
            "every_sample_predicted_once": True,
            "outer_test_assignments_match": True,
            "test_threshold_fitting": False,
            "coordinates_disabled": True,
        },
        "folds": folds,
        "metrics": metrics,
    }
    _atomic_json(outputs[0], payload)
    _atomic_csv(outputs[1], rows)
    lines = [
        "# 作者短期分支无坐标版 3-fold OOF 汇总",
        "",
        "- 数据配置：`{}`".format(profile),
        "- 模型：`{}`".format(AUTHOR_SHORT_TERM_MODEL_NAME),
        "- OOF 样本数：{}".format(len(rows)),
        "- 每个样本恰好一次 outer-test 预测：是",
        "- 坐标输入：关闭",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
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
        "| Mean fold AUROC | {:.6f} ± {:.6f} |".format(
            metrics["outer_fold_roc_auc"]["mean"],
            metrics["outer_fold_roc_auc"]["standard_deviation"],
        ),
        "| Accuracy | {:.6f} |".format(metrics["accuracy"]),
        "| Balanced Accuracy | {:.6f} |".format(
            metrics["balanced_accuracy"]
        ),
        "| F1 | {:.6f} |".format(metrics["f1"]),
        "",
        "| Fold | N | AUROC | Site-AUC | BA | Accuracy | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            "| {fold} | {sample_count} | {roc_auc:.6f} | {site} | "
            "{balanced_accuracy:.6f} | {accuracy:.6f} | {f1:.6f} |".format(
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
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_json": outputs[0],
        "predictions_csv": outputs[1],
        "summary_markdown": outputs[2],
        "metrics": metrics,
    }

