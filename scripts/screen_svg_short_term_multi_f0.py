"""Screen sparse multi-expert F0 fusions on frozen three-fold predictions.

This is an explicitly exploratory candidate-ranking entry point.  Every
outer-test sample remains unseen by the fold-specific F0 fit, but the script
ranks candidates using their outer-test fold results and is therefore not a
fresh confirmatory estimate after a winner is selected.
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
    crossfit_classification_metrics,
    fit_multi_f0_fusion,
    read_prediction_artifact,
)


CANDIDATES = {
    "st_g2": ("short_term", "g2"),
    "st_s_g2": ("short_term", "s", "g2"),
    "st_s_g2_c3": ("short_term", "s", "g2", "c3"),
    "st_svg": ("short_term", "svg"),
    "st_s": ("short_term", "s"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--c3-root", type=Path, required=True)
    parser.add_argument("--short-term-seed", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--g2-seed", type=int, default=43)
    parser.add_argument("--c3-seed", type=int, default=42)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATES),
        default=tuple(CANDIDATES),
    )
    parser.add_argument("--l1-weight", type=float, default=1.0e-3)
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
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _expert_artifacts(args, fold):
    source = args.source_crossfit_root.resolve() / "fold_{}".format(fold)
    short = (
        source
        / "author_short_term_no_coord"
        / "evaluation_seed{}".format(args.short_term_seed)
    )
    models = source / "models"
    g2 = (
        args.g2_root.resolve()
        / "fold_{}".format(fold)
        / "models"
        / "G2_seed{}".format(args.g2_seed)
    )
    c3 = (
        args.c3_root.resolve()
        / "fold_{}".format(fold)
        / "models"
        / "C3_seed{}".format(args.c3_seed)
    )
    return {
        "short_term": {
            "fit": short / "validation_predictions.csv",
            "evaluate": short / "test_predictions.csv",
        },
        "s": {
            "fit": models
            / "static_spectral_only_seed{}".format(args.base_seed)
            / "best_evaluation.json",
            "evaluate": models
            / "static_spectral_only_seed{}".format(args.base_seed)
            / "outer_test_evaluation.json",
        },
        "svg": {
            "fit": models
            / "signed_gin_multibranch_late_fusion_seed{}".format(
                args.base_seed
            )
            / "best_evaluation.json",
            "evaluate": models
            / "signed_gin_multibranch_late_fusion_seed{}".format(
                args.base_seed
            )
            / "outer_test_evaluation.json",
        },
        "g2": {
            "fit": g2 / "best_evaluation.json",
            "evaluate": g2 / "outer_test_evaluation.json",
        },
        "c3": {
            "fit": c3 / "best_evaluation.json",
            "evaluate": c3 / "outer_test_evaluation.json",
        },
    }


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


def _candidate_result(args, candidate):
    names = CANDIDATES[candidate]
    predictions = []
    fold_results = []
    provenance = []
    for fold in (0, 1, 2):
        artifacts = _expert_artifacts(args, fold)
        selected = {name: artifacts[name] for name in names}
        missing = [
            str(paths[partition])
            for paths in selected.values()
            for partition in ("fit", "evaluate")
            if not paths[partition].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "candidate {} is missing artifacts: {}".format(
                    candidate, missing
                )
            )
        fit = {
            name: read_prediction_artifact(paths["fit"])
            for name, paths in selected.items()
        }
        evaluate = {
            name: read_prediction_artifact(paths["evaluate"])
            for name, paths in selected.items()
        }
        fitted = fit_multi_f0_fusion(
            fit,
            l1_weight=args.l1_weight,
            optimization_steps=args.optimization_steps,
        )
        evaluated = apply_multi_f0_fusion(fitted, evaluate)
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
                "fit_sample_count": len(next(iter(fit.values()))),
                "evaluation_sample_count": len(next(iter(evaluate.values()))),
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
                    for name, paths in selected.items()
                },
            }
        )
    if len({row["sample_key"] for row in predictions}) != len(predictions):
        raise RuntimeError("multi-expert F0 generated duplicate OOF samples")
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
            "mean_fold_site_stratified_roc_auc": statistics.mean(
                fold_site_auc
            ),
            "fold_site_stratified_roc_auc_population_sd": (
                statistics.pstdev(fold_site_auc)
            ),
        }
    )
    return {
        "artifact_type": "svg_short_term_multi_expert_f0_screen",
        "dataset": args.dataset,
        "candidate": candidate,
        "experts": list(names),
        "fusion_protocol": "inner_validation_fit_outer_test_evaluation",
        "strict_nested_stacking": False,
        "exploratory_candidate_selection": True,
        "outer_test_used_for_candidate_ranking": True,
        "test_threshold_fitting": False,
        "primary_metric": "mean_outer_fold_roc_auc",
        "seeds": {
            "short_term": int(args.short_term_seed),
            "base": int(args.base_seed),
            "g2": int(args.g2_seed),
            "c3": int(args.c3_seed),
        },
        "l1_weight": float(args.l1_weight),
        "optimization_steps": int(args.optimization_steps),
        "fold_results": fold_results,
        "metrics": metrics,
        "provenance": provenance,
        "predictions": predictions,
    }


def _summary_markdown(payload):
    lines = [
        "# {} ST + SVG候选多专家F0筛选".format(payload["dataset"]),
        "",
        "- 主指标：三折 outer-test AUROC 算术平均",
        "- 每折F0只在该折inner-validation拟合，阈值冻结到outer-test",
        "- 本报告用于探索性候选排序，不是候选选定后的无偏确认结果",
        "",
        "| 排名 | 候选 | 专家 | Mean-fold AUC | Site-AUC | Pooled AUC | BA | Accuracy | F1 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["ranking"], 1):
        metrics = row["metrics"]
        lines.append(
            "| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                index,
                row["candidate"],
                "+".join(row["experts"]),
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
            "> 候选、seed与后续超参数若继续依据同一批outer OOF结果选择，最终成绩必须标记为探索性best-configuration结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if len(args.candidates) != len(set(args.candidates)):
        raise ValueError("candidate names must be unique")
    if args.l1_weight < 0.0 or args.optimization_steps < 1:
        raise ValueError("invalid multi-expert F0 screen configuration")
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError("multi-expert F0 summary already exists")
    output.mkdir(parents=True, exist_ok=True)
    compact = []
    for candidate in args.candidates:
        result = _candidate_result(args, candidate)
        candidate_output = output / candidate
        candidate_output.mkdir(parents=True, exist_ok=True)
        predictions = result.pop("predictions")
        _atomic_json(candidate_output / "evaluation.json", result)
        _write_predictions(candidate_output / "oof_predictions.csv", predictions)
        compact.append(
            {
                "candidate": candidate,
                "experts": result["experts"],
                "metrics": result["metrics"],
            }
        )
        print(
            "FINISH {} mean_fold_auc={:.6f}".format(
                candidate, result["metrics"]["mean_fold_roc_auc"]
            ),
            flush=True,
        )
    compact.sort(
        key=lambda row: (
            -float(row["metrics"]["mean_fold_roc_auc"]),
            row["candidate"],
        )
    )
    payload = {
        "artifact_type": "svg_short_term_multi_expert_f0_screen_summary",
        "dataset": args.dataset,
        "primary_metric": "mean_outer_fold_roc_auc",
        "exploratory_candidate_selection": True,
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
