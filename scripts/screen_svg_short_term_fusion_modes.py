"""Compare frozen ST+G2 F0/F1/F2 fusion modes over three outer folds.

F0 is calibrated nonnegative sparse fusion.  F1 and F2 are deliberately
cheap decision-level residual screens over frozen predictions: F1 anchors on
the short-term branch and F2 anchors on G2.  A winning residual mode must be
promoted separately before it can be described as representation-level
residual fusion.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.svg_v2_f0_fusion import (  # noqa: E402
    apply_multi_f0_fusion,
    apply_residual_logit_fusion,
    crossfit_classification_metrics,
    fit_multi_f0_fusion,
    fit_residual_logit_fusion,
    read_prediction_artifact,
)


MODES = ("F0", "F1", "F2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--short-term-seed", type=int, required=True)
    parser.add_argument("--g2-seed", type=int, default=43)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--l1-weight", type=float, default=1.0e-3)
    parser.add_argument("--initial-gate", type=float, default=0.01)
    parser.add_argument("--residual-auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-3)
    parser.add_argument("--optimization-steps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _artifacts(args, fold):
    source = args.source_crossfit_root.resolve() / "fold_{}".format(fold)
    short = (
        source
        / "author_short_term_no_coord"
        / "evaluation_seed{}".format(args.short_term_seed)
    )
    g2 = (
        args.g2_root.resolve()
        / "fold_{}".format(fold)
        / "models"
        / "G2_seed{}".format(args.g2_seed)
    )
    return {
        "short_term": {
            "fit": short / "validation_predictions.csv",
            "evaluate": short / "test_predictions.csv",
        },
        "g2": {
            "fit": g2 / "best_evaluation.json",
            "evaluate": g2 / "outer_test_evaluation.json",
        },
    }


def _fit_mode(args, mode, fit):
    if mode == "F0":
        return fit_multi_f0_fusion(
            fit,
            l1_weight=args.l1_weight,
            optimization_steps=args.optimization_steps,
        )
    anchor_name, residual_name = (
        ("short_term", "g2") if mode == "F1" else ("g2", "short_term")
    )
    return fit_residual_logit_fusion(
        anchor_name,
        fit[anchor_name],
        residual_name,
        fit[residual_name],
        initial_gate=args.initial_gate,
        auxiliary_weight=args.residual_auxiliary_weight,
        l2_weight=args.residual_l2_weight,
        optimization_steps=args.optimization_steps,
    )


def _apply_mode(mode, fitted, evaluate):
    if mode == "F0":
        return apply_multi_f0_fusion(fitted, evaluate)
    return apply_residual_logit_fusion(
        fitted,
        evaluate[str(fitted["anchor_name"])],
        evaluate[str(fitted["residual_name"])],
    )


def _write_predictions(path, rows):
    fields = (
        "fold",
        "sample_key",
        "site",
        "label",
        "positive_probability",
        "threshold",
        "predicted_label",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mode_result(args, mode):
    predictions = []
    fold_results = []
    provenance = []
    for fold in (0, 1, 2):
        artifacts = _artifacts(args, fold)
        missing = [
            str(paths[partition])
            for paths in artifacts.values()
            for partition in ("fit", "evaluate")
            if not paths[partition].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "fusion mode {} is missing artifacts: {}".format(mode, missing)
            )
        fit = {
            name: read_prediction_artifact(paths["fit"])
            for name, paths in artifacts.items()
        }
        evaluate = {
            name: read_prediction_artifact(paths["evaluate"])
            for name, paths in artifacts.items()
        }
        fitted = _fit_mode(args, mode, fit)
        evaluated = _apply_mode(mode, fitted, evaluate)
        current = []
        for row in evaluated["predictions"]:
            enriched = dict(row)
            enriched["fold"] = int(fold)
            current.append(enriched)
        predictions.extend(current)
        model_spec = dict(fitted)
        model_spec.pop("fit_sample_keys")
        model_spec.pop("fit_sites")
        fold_results.append(
            {
                "fold": int(fold),
                "fit_and_evaluation_disjoint": True,
                "fit_sample_count": len(fit["short_term"]),
                "evaluation_sample_count": len(evaluate["short_term"]),
                "fitted": model_spec,
                "metrics": evaluated["metrics"],
            }
        )
        provenance.append(
            {
                "fold": int(fold),
                "experts": {
                    name: {
                        partition: {
                            "path": str(paths[partition].resolve()),
                            "sha256": _sha256(paths[partition]),
                        }
                        for partition in ("fit", "evaluate")
                    }
                    for name, paths in artifacts.items()
                },
            }
        )
    if len({row["sample_key"] for row in predictions}) != len(predictions):
        raise RuntimeError("fusion mode generated duplicate OOF samples")
    predictions.sort(key=lambda row: (int(row["fold"]), row["sample_key"]))
    fold_auc = [float(row["metrics"]["roc_auc"]) for row in fold_results]
    fold_site_auc = [
        float(row["metrics"]["site_stratified_roc_auc"])
        for row in fold_results
    ]
    metrics = crossfit_classification_metrics(predictions)
    metrics.update(
        {
            "mean_fold_roc_auc": statistics.mean(fold_auc),
            "fold_roc_auc_population_sd": statistics.pstdev(fold_auc),
            "mean_fold_site_stratified_roc_auc": statistics.mean(fold_site_auc),
            "fold_site_stratified_roc_auc_population_sd": statistics.pstdev(
                fold_site_auc
            ),
        }
    )
    return {
        "artifact_type": "svg_short_term_st_g2_fusion_mode_screen",
        "dataset": args.dataset,
        "mode": mode,
        "experts": ["short_term", "g2"],
        "fusion_protocol": "inner_validation_fit_outer_test_evaluation",
        "residual_scope": (
            "not_applicable"
            if mode == "F0"
            else "decision_level_screen_not_representation_level"
        ),
        "strict_nested_stacking": False,
        "exploratory_mode_selection": True,
        "outer_test_used_for_mode_ranking": True,
        "test_threshold_fitting": False,
        "primary_metric": "mean_outer_fold_roc_auc",
        "seeds": {
            "short_term": int(args.short_term_seed),
            "g2": int(args.g2_seed),
        },
        "fold_results": fold_results,
        "metrics": metrics,
        "provenance": provenance,
        "predictions": predictions,
    }


def _summary_markdown(payload):
    lines = [
        "# {} ST+G2 融合模式筛选".format(payload["dataset"]),
        "",
        "- 主指标：三折 outer-test AUROC 算术平均",
        "- F0：校准、非负、稀疏的决策层融合",
        "- F1：短期分支为锚点、G2 为决策层残差",
        "- F2：G2 为锚点、短期分支为决策层残差",
        "- 每折仅用 inner-validation 拟合融合与阈值，并冻结到 outer-test",
        "- F1/F2 是低成本决策层筛选，不等同于表示层残差网络",
        "",
        "| 排名 | 模式 | Mean-fold AUC | Site-AUC | Pooled AUC | BA | Accuracy | F1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["ranking"], 1):
        metrics = row["metrics"]
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                index,
                row["mode"],
                metrics["mean_fold_roc_auc"],
                metrics["mean_fold_site_stratified_roc_auc"],
                metrics["roc_auc"],
                metrics["balanced_accuracy"],
                metrics["accuracy"],
                metrics["f1"],
            )
        )
    lines.extend(
        [
            "",
            "> 本报告用于探索性融合方式选择；不是选择完成后的独立确认结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("fusion mode names must be unique")
    if args.optimization_steps < 1:
        raise ValueError("optimization steps must be positive")
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError("fusion mode summary already exists")
    output.mkdir(parents=True, exist_ok=True)
    compact = []
    for mode in args.modes:
        result = _mode_result(args, mode)
        mode_output = output / mode
        mode_output.mkdir(parents=True, exist_ok=True)
        predictions = result.pop("predictions")
        _atomic_json(mode_output / "evaluation.json", result)
        _write_predictions(mode_output / "oof_predictions.csv", predictions)
        compact.append({"mode": mode, "metrics": result["metrics"]})
        print(
            "FINISH {} mean_fold_auc={:.6f}".format(
                mode, result["metrics"]["mean_fold_roc_auc"]
            ),
            flush=True,
        )
    compact.sort(
        key=lambda row: (-float(row["metrics"]["mean_fold_roc_auc"]), row["mode"])
    )
    payload = {
        "artifact_type": "svg_short_term_st_g2_fusion_mode_screen_summary",
        "dataset": args.dataset,
        "primary_metric": "mean_outer_fold_roc_auc",
        "exploratory_mode_selection": True,
        "ranking": compact,
    }
    _atomic_json(summary_path, payload)
    with (output / "summary.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(_summary_markdown(payload))
    print("summary: {}".format(summary_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
