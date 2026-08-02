"""Summarize the two-fold SVG-v2 screen without reading outer-test data."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NAME", "EXPERIMENT_ROOT", "BASELINE_ROOT"),
        required=True,
        help="repeat once per dataset",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=("A1", "B1", "C3", "F1", "G2"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1))
    parser.add_argument(
        "--baseline-variant",
        default="signed_gin_multibranch_late_fusion",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(path):
    payload = _read(path)
    metrics = payload.get("metrics", {}).get("balanced_accuracy")
    if not isinstance(metrics, dict):
        raise ValueError("validation evaluation has no BA-threshold metrics")
    auc = metrics.get("roc_auc")
    site_auc = metrics.get("site_stratified_roc_auc")
    if auc is None or site_auc is None:
        raise ValueError("validation AUC/Site-AUC is unavailable")
    return {"roc_auc": float(auc), "site_stratified_roc_auc": float(site_auc)}


def _mean(values):
    return sum(values) / float(len(values))


def build_summary(datasets, candidates, folds, baseline_variant, seed):
    if len(datasets) < 2:
        raise ValueError("SVG-v2 screen requires both registered datasets")
    if len(folds) != 2 or len(set(folds)) != 2:
        raise ValueError("streamlined SVG-v2 screen requires two folds")
    result = {
        "artifact_type": "svg_v2_two_fold_validation_screen",
        "primary_metric": "mean_inner_validation_fold_roc_auc",
        "pooled_auc_used": False,
        "test_used": False,
        "folds": list(folds),
        "seed": int(seed),
        "baseline_variant": baseline_variant,
        "datasets": {},
        "candidates": {},
    }
    for name, experiment_root, baseline_root in datasets:
        if name in result["datasets"]:
            raise ValueError("duplicate SVG-v2 screen dataset")
        experiment_root = Path(experiment_root).resolve()
        baseline_root = Path(baseline_root).resolve()
        baseline = []
        observed = {candidate: [] for candidate in candidates}
        for fold in folds:
            baseline.append(
                _metrics(
                    baseline_root
                    / "fold_{}".format(fold)
                    / "models"
                    / "{}_seed{}".format(baseline_variant, seed)
                    / "best_evaluation.json"
                )
            )
            for candidate in candidates:
                observed[candidate].append(
                    _metrics(
                        experiment_root
                        / "fold_{}".format(fold)
                        / "models"
                        / "{}_seed{}".format(candidate, seed)
                        / "best_evaluation.json"
                    )
                )
        base_auc = _mean([row["roc_auc"] for row in baseline])
        base_site = _mean(
            [row["site_stratified_roc_auc"] for row in baseline]
        )
        dataset_result = {
            "baseline": {
                "folds": baseline,
                "mean_roc_auc": base_auc,
                "mean_site_stratified_roc_auc": base_site,
            },
            "candidates": {},
        }
        for candidate, rows in observed.items():
            mean_auc = _mean([row["roc_auc"] for row in rows])
            mean_site = _mean(
                [row["site_stratified_roc_auc"] for row in rows]
            )
            dataset_result["candidates"][candidate] = {
                "folds": rows,
                "mean_roc_auc": mean_auc,
                "mean_site_stratified_roc_auc": mean_site,
                "delta_roc_auc": mean_auc - base_auc,
                "delta_site_stratified_roc_auc": mean_site - base_site,
                "positive_fold_count": sum(
                    row["roc_auc"] > baseline[index]["roc_auc"]
                    for index, row in enumerate(rows)
                ),
            }
        result["datasets"][name] = dataset_result

    for candidate in candidates:
        rows = [
            result["datasets"][name]["candidates"][candidate]
            for name in sorted(result["datasets"])
        ]
        auc_deltas = [row["delta_roc_auc"] for row in rows]
        classification_gain = (
            max(auc_deltas) >= 0.005 and min(auc_deltas) >= -0.015
        )
        site_gain = any(
            row["delta_site_stratified_roc_auc"] >= 0.01
            and row["delta_roc_auc"] >= -0.01
            for row in rows
        )
        two_fold_positive = any(
            row["positive_fold_count"] == len(folds) for row in rows
        )
        checks = {
            "classification_gain": classification_gain,
            "site_stratified_gain": site_gain,
            "both_screen_folds_positive_in_one_dataset": two_fold_positive,
        }
        result["candidates"][candidate] = {
            "eligible_for_round2": any(checks.values()),
            "checks": checks,
        }
    result["eligible_candidates"] = [
        name
        for name in candidates
        if result["candidates"][name]["eligible_for_round2"]
    ]
    return result


def _markdown(result):
    lines = [
        "# SVG-v2 两折开发筛选",
        "",
        "- outer-test 使用：否",
        "- 主指标：两折 inner-validation AUROC 算术平均",
        "- pooled AUROC 使用：否",
        "- fold：{}".format(", ".join(map(str, result["folds"]))),
        "- seed：{}".format(result["seed"]),
        "",
        "| 候选 | 数据集 | ΔMean-fold AUC | ΔMean-fold Site-AUC | 正向fold | 进入下一轮 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for candidate in result["candidates"]:
        eligible = result["candidates"][candidate]["eligible_for_round2"]
        for dataset, payload in result["datasets"].items():
            row = payload["candidates"][candidate]
            lines.append(
                "| {} | {} | {:+.6f} | {:+.6f} | {}/{} | {} |".format(
                    candidate,
                    dataset,
                    row["delta_roc_auc"],
                    row["delta_site_stratified_roc_auc"],
                    row["positive_fold_count"],
                    len(result["folds"]),
                    "是" if eligible else "否",
                )
            )
    lines.extend(
        [
            "",
            "进入下一轮：{}".format(
                ", ".join(result["eligible_candidates"]) or "无"
            ),
            "",
            "> 本报告只读取inner-validation产物；候选冻结前不得生成outer-test结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    json_path = output / "screen_summary.json"
    markdown_path = output / "summary.md"
    if (json_path.exists() or markdown_path.exists()) and not args.overwrite:
        raise FileExistsError("SVG-v2 screen summary already exists")
    result = build_summary(
        args.dataset,
        args.candidates,
        args.folds,
        args.baseline_variant,
        args.seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(json_path))
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
