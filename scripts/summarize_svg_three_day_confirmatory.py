"""Summarize three-day SVG confirmation with mean-fold AUROC primary."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)
from keysubgraph.crossfit.svg_three_day_runner import (  # noqa: E402
    SVG_THREE_DAY_ALL_CANDIDATES,
    SVG_THREE_DAY_CANDIDATE_SPECS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-crossfit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=SVG_THREE_DAY_ALL_CANDIDATES,
        required=True,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean_std(values):
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {"mean": mean, "standard_deviation": math.sqrt(variance)}


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _seed_row(result, seed):
    metrics = result["metrics"]
    site = metrics["outer_fold_site_stratified_roc_auc"]
    return {
        "seed": int(seed),
        "mean_fold_roc_auc": metrics["outer_fold_roc_auc"]["mean"],
        "mean_fold_site_stratified_roc_auc": (
            site["mean"] if site is not None else None
        ),
        "pooled_oof_roc_auc": metrics["pooled_oof_roc_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
    }


def main():
    args = parse_args()
    if len(args.candidates) != len(set(args.candidates)):
        raise ValueError("confirmatory candidates must be unique")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("confirmatory seeds must be unique and non-empty")
    source = args.source_crossfit_root.resolve()
    output_root = args.output_root.resolve()
    assignments = source / "assignments" / "fold_assignments.json"
    if not assignments.is_file():
        raise FileNotFoundError(str(assignments))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_root / "confirmatory_summary"
    )
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    if (json_path.exists() or markdown_path.exists()) and not args.overwrite:
        raise FileExistsError("three-day confirmatory summary exists")

    payload = {
        "artifact_type": "svg_three_day_confirmatory_summary",
        "primary_metric": "seed_mean_of_outer_fold_roc_auc_mean",
        "pooled_oof_roc_auc_is_auxiliary": True,
        "fold_assignments": str(assignments),
        "seeds": list(args.seeds),
        "candidates": {},
    }
    for candidate in args.candidates:
        variant = str(SVG_THREE_DAY_CANDIDATE_SPECS[candidate]["variant"])
        rows = []
        for seed in args.seeds:
            summary_dir = output_root / "oof_summary" / "{}_seed{}".format(
                candidate, seed
            )
            result = summarize_sv_signed_gin_crossfit(
                output_root,
                assignments,
                variant=variant,
                seed=seed,
                run_name="{}_seed{}".format(candidate, seed),
                output_dir=summary_dir,
                overwrite=args.overwrite,
            )
            rows.append(_seed_row(result, seed))
        candidate_payload = {
            "variant": variant,
            "seeds": rows,
            "mean_fold_roc_auc": _mean_std(
                [row["mean_fold_roc_auc"] for row in rows]
            ),
            "balanced_accuracy": _mean_std(
                [row["balanced_accuracy"] for row in rows]
            ),
            "accuracy": _mean_std([row["accuracy"] for row in rows]),
            "f1": _mean_std([row["f1"] for row in rows]),
        }
        site_values = [
            row["mean_fold_site_stratified_roc_auc"]
            for row in rows
            if row["mean_fold_site_stratified_roc_auc"] is not None
        ]
        candidate_payload["mean_fold_site_stratified_roc_auc"] = (
            _mean_std(site_values) if site_values else None
        )
        payload["candidates"][candidate] = candidate_payload

    baseline = payload["candidates"].get("BASELINE")
    if baseline is not None:
        baseline_by_seed = {
            row["seed"]: row["mean_fold_roc_auc"]
            for row in baseline["seeds"]
        }
        for candidate, values in payload["candidates"].items():
            if candidate == "BASELINE":
                continue
            deltas = [
                row["mean_fold_roc_auc"] - baseline_by_seed[row["seed"]]
                for row in values["seeds"]
            ]
            values["paired_delta_mean_fold_roc_auc"] = _mean_std(deltas)
            values["positive_seed_count"] = sum(value > 0.0 for value in deltas)

    lines = [
        "# SVG 三天精简方案确认性汇总",
        "",
        "- 主指标：每个 seed 的三折 AUROC 算术平均，再跨 seed 汇总",
        "- Pooled OOF AUROC：仅作辅助诊断",
        "",
        "| 候选 | Mean-fold AUROC | Mean-fold Site-AUC | BA | Accuracy | F1 | 配对ΔAUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, values in payload["candidates"].items():
        site = values["mean_fold_site_stratified_roc_auc"]
        delta = values.get("paired_delta_mean_fold_roc_auc")
        lines.append(
            "| {} | {:.6f} ± {:.6f} | {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                candidate,
                values["mean_fold_roc_auc"]["mean"],
                values["mean_fold_roc_auc"]["standard_deviation"],
                (
                    "N/A"
                    if site is None
                    else "{:.6f} ± {:.6f}".format(
                        site["mean"], site["standard_deviation"]
                    )
                ),
                values["balanced_accuracy"]["mean"],
                values["accuracy"]["mean"],
                values["f1"]["mean"],
                (
                    "N/A"
                    if delta is None
                    else "{:+.6f} ± {:.6f}".format(
                        delta["mean"], delta["standard_deviation"]
                    )
                ),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(json_path, payload)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
