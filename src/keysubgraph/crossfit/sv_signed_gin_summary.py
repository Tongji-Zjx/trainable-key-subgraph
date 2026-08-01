"""Strict OOF aggregation for SV Signed-GIN cross-fit evaluations."""

from __future__ import absolute_import, division, print_function

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roc_auc(labels: Sequence[int], scores: Sequence[float]):
    if len(labels) != len(scores) or not labels:
        raise ValueError("OOF labels and scores are empty or misaligned")
    positives = sum(int(value) == 1 for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    indexed = sorted(
        enumerate(float(value) for value in scores),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while (
            end < len(indexed)
            and indexed[end][1] == indexed[start][1]
        ):
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    positive_rank_sum = sum(
        ranks[index]
        for index, label in enumerate(labels)
        if int(label) == 1
    )
    return (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def _site_stratified_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    sites: Sequence[str],
):
    numerator = 0.0
    denominator = 0.0
    eligible = 0
    for site in sorted(set(str(value) for value in sites)):
        indices = [
            index
            for index, value in enumerate(sites)
            if str(value) == site
        ]
        current_labels = [int(labels[index]) for index in indices]
        current_scores = [float(scores[index]) for index in indices]
        auc = _roc_auc(current_labels, current_scores)
        if auc is None:
            continue
        positives = sum(value == 1 for value in current_labels)
        negatives = len(current_labels) - positives
        pairs = float(positives * negatives)
        numerator += float(auc) * pairs
        denominator += pairs
        eligible += 1
    return {
        "roc_auc": (
            numerator / denominator if denominator > 0.0 else None
        ),
        "eligible_site_count": eligible,
        "eligible_pair_count": int(denominator),
    }


def _classification_metrics(rows: Sequence[Mapping[str, Any]]):
    counts = Counter()
    for row in rows:
        label = int(row["label"])
        predicted = int(row["predicted_label"])
        if label == 1 and predicted == 1:
            counts["tp"] += 1
        elif label == 1:
            counts["fn"] += 1
        elif predicted == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    sensitivity = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = sensitivity
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": (tp + tn) / float(len(rows)),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(
        len(values)
    )
    return {"mean": mean, "standard_deviation": math.sqrt(variance)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "fold",
                "sample_key",
                "site",
                "label",
                "positive_probability",
                "threshold",
                "predicted_label",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def summarize_sv_signed_gin_crossfit(
    output_root: Path,
    fold_assignments: Path,
    variant: str = "signed_gin_static_variation",
    seed: int = 42,
    run_name: str = None,
    output_dir: Path = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate and aggregate one frozen outer-test prediction per sample."""

    output_root = Path(output_root).resolve()
    fold_assignments = Path(fold_assignments).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else output_root / "oof_summary" / "{}_seed{}".format(variant, seed)
    )
    model_directory = (
        str(run_name)
        if run_name is not None
        else "{}_seed{}".format(variant, seed)
    )
    if (
        not model_directory
        or "/" in model_directory
        or "\\" in model_directory
    ):
        raise ValueError("SV summary run name must be one directory")
    outputs = (
        output_dir / "summary.json",
        output_dir / "oof_predictions.csv",
        output_dir / "summary.md",
    )
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError("SV cross-fit summary already exists")
    with fold_assignments.open("r", encoding="utf-8") as handle:
        assignment_payload = json.load(handle)
    if (
        assignment_payload.get("purpose")
        != "confirmatory_cross_fitted_fold_roles"
        or not assignment_payload.get("immutable")
    ):
        raise ValueError("unsupported cross-fit assignment artifact")
    num_folds = int(assignment_payload["num_outer_folds"])
    expected_by_fold = {}
    expected_all = {}
    for fold in range(num_folds):
        rows = [
            row
            for row in assignment_payload["assignments"]
            if int(row["outer_fold"]) == fold
            and row["role"] == "outer_test"
        ]
        expected_by_fold[fold] = {
            row["sample_key"]: row for row in rows
        }
        for row in rows:
            key = row["sample_key"]
            if key in expected_all:
                raise ValueError(
                    "sample is assigned to multiple outer-test folds"
                )
            expected_all[key] = row
    oof_rows: List[Dict[str, Any]] = []
    fold_rows = []
    evaluation_hashes = {}
    for fold in range(num_folds):
        path = (
            output_root
            / "fold_{}".format(fold)
            / "models"
            / model_directory
            / "outer_test_evaluation.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "missing outer-test evaluation: {}".format(path)
            )
        with path.open("r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        if (
            evaluation.get("split") != "test"
            or evaluation.get("variant") != variant
            or evaluation.get("threshold_fit_split") != "validation"
            or evaluation.get("threshold_strategy")
            != "balanced_accuracy"
        ):
            raise ValueError("outer-test evaluation metadata mismatch")
        expected = expected_by_fold[fold]
        predictions = evaluation.get("predictions", [])
        actual = {row["sample_key"]: row for row in predictions}
        if len(actual) != len(predictions):
            raise ValueError("outer-test evaluation has duplicate samples")
        if set(actual) != set(expected):
            raise ValueError(
                "outer-test predictions differ from frozen fold assignment"
            )
        threshold = float(evaluation["threshold"])
        for key in sorted(actual):
            prediction = actual[key]
            assignment = expected[key]
            if (
                int(prediction["label"]) != int(assignment["label"])
                or str(prediction["site"]) != str(assignment["site"])
            ):
                raise ValueError("OOF prediction metadata mismatch")
            probability = float(prediction["positive_probability"])
            oof_rows.append(
                {
                    "fold": fold,
                    "sample_key": key,
                    "site": str(prediction["site"]),
                    "label": int(prediction["label"]),
                    "positive_probability": probability,
                    "threshold": threshold,
                    "predicted_label": int(probability >= threshold),
                }
            )
        metrics = evaluation["metrics"]
        fold_rows.append(
            {
                "fold": fold,
                "sample_count": len(predictions),
                "class_counts": dict(
                    Counter(
                        str(int(row["label"])) for row in predictions
                    )
                ),
                "roc_auc": metrics.get("roc_auc"),
                "site_stratified_roc_auc": metrics.get(
                    "site_stratified_roc_auc"
                ),
                "balanced_accuracy": metrics.get(
                    "balanced_accuracy"
                ),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "threshold": threshold,
                "evaluation": str(path),
            }
        )
        evaluation_hashes[str(fold)] = _sha256(path)
    if len(oof_rows) != len(expected_all):
        raise ValueError("OOF predictions do not cover every sample once")
    if len({row["sample_key"] for row in oof_rows}) != len(oof_rows):
        raise ValueError("OOF predictions contain duplicate samples")
    oof_rows.sort(key=lambda row: (row["fold"], row["sample_key"]))
    labels = [int(row["label"]) for row in oof_rows]
    scores = [float(row["positive_probability"]) for row in oof_rows]
    sites = [str(row["site"]) for row in oof_rows]
    pooled_auc = _roc_auc(labels, scores)
    site_auc = _site_stratified_auc(labels, scores, sites)
    classification = _classification_metrics(oof_rows)
    fold_auc_values = [
        float(row["roc_auc"])
        for row in fold_rows
        if row["roc_auc"] is not None
    ]
    fold_site_auc_values = [
        float(row["site_stratified_roc_auc"])
        for row in fold_rows
        if row["site_stratified_roc_auc"] is not None
    ]
    metrics = {
        "sample_count": len(oof_rows),
        "class_counts": dict(
            Counter(str(value) for value in labels)
        ),
        "pooled_oof_roc_auc": pooled_auc,
        "pooled_oof_site_stratified_roc_auc": site_auc["roc_auc"],
        "eligible_site_count": site_auc["eligible_site_count"],
        "eligible_site_pair_count": site_auc[
            "eligible_pair_count"
        ],
        "outer_fold_roc_auc": _mean_std(fold_auc_values),
        "outer_fold_site_stratified_roc_auc": (
            _mean_std(fold_site_auc_values)
            if fold_site_auc_values
            else None
        ),
        **classification
    }
    payload = {
        "artifact_type": "sv_signed_gin_crossfit_oof_summary",
        "variant": variant,
        "run_name": model_directory,
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
            "test_threshold_fitting": False,
        },
        "folds": fold_rows,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(outputs[0], payload)
    _atomic_csv(outputs[1], oof_rows)
    lines = [
        "# SV Signed-GIN 交叉拟合 OOF 汇总",
        "",
        "- 模型：`{}`".format(variant),
        "- 外折数：{}".format(num_folds),
        "- OOF 样本数：{}".format(len(oof_rows)),
        "- 每个样本恰好一次外折预测：是",
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
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "summary_json": outputs[0],
        "predictions_csv": outputs[1],
        "summary_markdown": outputs[2],
        "metrics": metrics,
    }
