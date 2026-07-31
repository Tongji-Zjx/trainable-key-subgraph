"""Strict OOF aggregation for structured short-term cross-fit evaluations."""

from __future__ import absolute_import, division, print_function

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .sv_signed_gin_summary import (
    _atomic_csv,
    _atomic_json,
    _classification_metrics,
    _mean_std,
    _roc_auc,
    _sha256,
    _site_stratified_auc,
)
from keysubgraph.models.structured_short_term import (
    PAPER_ALIGNED_MODEL_NAME,
    PAPER_ALIGNED_VARIANT,
    PAPER_ALIGNED_PST_MODEL_NAME,
    PAPER_ALIGNED_PST_VARIANT,
    STRUCTURED_SAFE_MODEL_NAME,
    STRUCTURED_SAFE_VARIANT,
)


MODEL_NAME = STRUCTURED_SAFE_MODEL_NAME


def _load_json(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_roles(assignment_payload, num_folds):
    expected = {}
    outer_all = {}
    assignments = assignment_payload.get("assignments", [])
    for fold in range(num_folds):
        expected[fold] = {}
        for role in ("inner_validation", "outer_test"):
            rows = [
                row
                for row in assignments
                if int(row["outer_fold"]) == fold
                and str(row["role"]) == role
            ]
            mapped = {str(row["sample_key"]): row for row in rows}
            if len(mapped) != len(rows) or not rows:
                raise ValueError(
                    "cross-fit assignment role is empty or duplicated"
                )
            expected[fold][role] = mapped
        for key, row in expected[fold]["outer_test"].items():
            if key in outer_all:
                raise ValueError(
                    "sample is assigned to multiple outer-test folds"
                )
            outer_all[key] = row
    return expected, outer_all


def _validate_evaluation(
    evaluation,
    split,
    expected,
    expected_model_name=MODEL_NAME,
    expected_threshold=None,
):
    if (
        evaluation.get("model_name") != expected_model_name
        or evaluation.get("split") != split
        or evaluation.get("threshold_source") != "frozen_validation"
        or evaluation.get("threshold_fit_split") != "validation"
        or evaluation.get("threshold_strategy") != "balanced_accuracy"
    ):
        raise ValueError(
            "structured short-term evaluation metadata mismatch"
        )
    threshold = float(evaluation["threshold"])
    if (
        expected_threshold is not None
        and abs(threshold - float(expected_threshold)) > 1.0e-12
    ):
        raise ValueError(
            "validation and outer-test evaluations use different thresholds"
        )
    predictions = evaluation.get("predictions", [])
    actual = {str(row["sample_key"]): row for row in predictions}
    if len(actual) != len(predictions):
        raise ValueError("evaluation contains duplicate samples")
    if set(actual) != set(expected):
        raise ValueError(
            "evaluation predictions differ from frozen fold assignment"
        )
    for key, prediction in actual.items():
        assignment = expected[key]
        if (
            int(prediction["label"]) != int(assignment["label"])
            or str(prediction["site"]) != str(assignment["site"])
        ):
            raise ValueError("evaluation prediction metadata mismatch")
        probability = float(prediction["positive_probability"])
        if probability < 0.0 or probability > 1.0:
            raise ValueError("evaluation probability is outside [0, 1]")
        recorded = int(prediction["prediction"])
        if recorded != int(probability >= threshold):
            raise ValueError("evaluation prediction disagrees with threshold")
    return threshold, actual


def summarize_structured_short_term_crossfit(
    output_root: Path,
    fold_assignments: Path,
    seed: int = 42,
    output_dir: Path = None,
    overwrite: bool = False,
    model_variant: str = STRUCTURED_SAFE_VARIANT,
) -> Dict[str, Any]:
    """Validate and aggregate exactly one outer-test prediction per sample."""

    output_root = Path(output_root).resolve()
    fold_assignments = Path(fold_assignments).resolve()
    if model_variant == PAPER_ALIGNED_PST_VARIANT:
        model_name = PAPER_ALIGNED_PST_MODEL_NAME
        branch_name = "paper_aligned_short_term_with_pst"
        summary_name = "paper_aligned_short_term_with_pst_seed{}".format(
            seed
        )
    elif model_variant == PAPER_ALIGNED_VARIANT:
        model_name = PAPER_ALIGNED_MODEL_NAME
        branch_name = "paper_aligned_short_term"
        summary_name = "paper_aligned_short_term_seed{}".format(seed)
    elif model_variant == STRUCTURED_SAFE_VARIANT:
        model_name = STRUCTURED_SAFE_MODEL_NAME
        branch_name = "structured_short_term"
        summary_name = "structured_short_term_seed{}".format(seed)
    else:
        raise ValueError("unsupported structured short-term variant")
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else output_root
        / "oof_summary"
        / summary_name
    )
    outputs = (
        output_dir / "summary.json",
        output_dir / "oof_predictions.csv",
        output_dir / "summary.md",
    )
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError(
            "structured short-term cross-fit summary already exists"
        )
    assignment_payload = _load_json(fold_assignments)
    if (
        assignment_payload.get("purpose")
        != "confirmatory_cross_fitted_fold_roles"
        or not assignment_payload.get("immutable")
    ):
        raise ValueError("unsupported cross-fit assignment artifact")
    num_folds = int(assignment_payload["num_outer_folds"])
    expected, expected_outer = _expected_roles(
        assignment_payload,
        num_folds,
    )

    oof_rows: List[Dict[str, Any]] = []
    fold_rows = []
    evaluation_hashes = {}
    for fold in range(num_folds):
        evaluation_dir = (
            output_root
            / "fold_{}".format(fold)
            / branch_name
            / "evaluation_seed{}".format(seed)
        )
        validation_path = evaluation_dir / "validation_evaluation.json"
        test_path = evaluation_dir / "test_evaluation.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                "missing validation evaluation: {}".format(validation_path)
            )
        if not test_path.is_file():
            raise FileNotFoundError(
                "missing outer-test evaluation: {}".format(test_path)
            )
        validation = _load_json(validation_path)
        threshold, _ = _validate_evaluation(
            validation,
            "validation",
            expected[fold]["inner_validation"],
            expected_model_name=model_name,
        )
        test = _load_json(test_path)
        _, predictions = _validate_evaluation(
            test,
            "test",
            expected[fold]["outer_test"],
            expected_model_name=model_name,
            expected_threshold=threshold,
        )

        fold_prediction_rows = []
        for key in sorted(predictions):
            prediction = predictions[key]
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
            fold_prediction_rows.append(row)
            oof_rows.append(row)
        labels = [row["label"] for row in fold_prediction_rows]
        scores = [
            row["positive_probability"] for row in fold_prediction_rows
        ]
        sites = [row["site"] for row in fold_prediction_rows]
        site_auc = _site_stratified_auc(labels, scores, sites)
        classification = _classification_metrics(fold_prediction_rows)
        fold_rows.append(
            {
                "fold": fold,
                "sample_count": len(fold_prediction_rows),
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
                "validation_evaluation": str(validation_path),
                "outer_test_evaluation": str(test_path),
            }
        )
        evaluation_hashes[str(fold)] = {
            "validation": _sha256(validation_path),
            "outer_test": _sha256(test_path),
        }

    if len(oof_rows) != len(expected_outer):
        raise ValueError("OOF predictions do not cover every sample once")
    if len({row["sample_key"] for row in oof_rows}) != len(oof_rows):
        raise ValueError("OOF predictions contain duplicate samples")
    if {row["sample_key"] for row in oof_rows} != set(expected_outer):
        raise ValueError("OOF predictions do not match outer assignments")
    oof_rows.sort(key=lambda row: (row["fold"], row["sample_key"]))
    labels = [row["label"] for row in oof_rows]
    scores = [row["positive_probability"] for row in oof_rows]
    sites = [row["site"] for row in oof_rows]
    pooled_auc = _roc_auc(labels, scores)
    site_auc = _site_stratified_auc(labels, scores, sites)
    classification = _classification_metrics(oof_rows)
    fold_auc = [float(row["roc_auc"]) for row in fold_rows]
    metrics = {
        "sample_count": len(oof_rows),
        "class_counts": dict(Counter(str(value) for value in labels)),
        "pooled_oof_roc_auc": pooled_auc,
        "pooled_oof_site_stratified_roc_auc": site_auc["roc_auc"],
        "eligible_site_count": site_auc["eligible_site_count"],
        "eligible_site_pair_count": site_auc["eligible_pair_count"],
        "outer_fold_roc_auc": _mean_std(fold_auc),
    }
    metrics.update(classification)
    payload = {
        "artifact_type": (
            "structured_short_term_crossfit_oof_summary"
        ),
        "model_name": model_name,
        "model_variant": model_variant,
        "seed": int(seed),
        "num_outer_folds": num_folds,
        "threshold_policy": (
            "each fold uses its inner-validation balanced-accuracy threshold"
        ),
        "fold_assignments": str(fold_assignments),
        "fold_assignments_sha256": _sha256(fold_assignments),
        "evaluation_sha256": evaluation_hashes,
        "checks": {
            "every_sample_predicted_once": True,
            "outer_test_assignments_match": True,
            "inner_validation_assignments_match": True,
            "validation_and_test_thresholds_match": True,
            "test_threshold_fitting": False,
        },
        "folds": fold_rows,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(outputs[0], payload)
    _atomic_csv(outputs[1], oof_rows)
    lines = [
        "# 结构化短期分支 3-fold OOF 汇总",
        "",
        "- 模型：`{}`".format(model_name),
        "- 外折数：{}".format(num_folds),
        "- OOF 样本数：{}".format(len(oof_rows)),
        "- 每个样本恰好一次 outer-test 预测：是",
        "- 阈值：各折仅由该折 inner-validation 冻结",
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        "| Pooled OOF AUROC | {:.6f} |".format(pooled_auc),
        "| Site-stratified OOF AUROC | {} |".format(
            "N/A"
            if site_auc["roc_auc"] is None
            else "{:.6f}".format(site_auc["roc_auc"])
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
        "## 各外折",
        "",
        "| Fold | N | AUROC | Site-AUC | BA | Accuracy | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(
            "| {fold} | {sample_count} | {roc_auc:.6f} | "
            "{site_auc} | {balanced_accuracy:.6f} | "
            "{accuracy:.6f} | {f1:.6f} |".format(
                site_auc=(
                    "N/A"
                    if row["site_stratified_roc_auc"] is None
                    else "{:.6f}".format(
                        row["site_stratified_roc_auc"]
                    )
                ),
                **row
            )
        )
    outputs[2].write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return {
        "summary_json": outputs[0],
        "predictions_csv": outputs[1],
        "summary_markdown": outputs[2],
        "metrics": metrics,
    }
