#!/usr/bin/env python3
"""Run and summarize the four-rotation S0-S3 static-branch matrix."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("s0", "s1", "s2", "s3")
FOLDS = (0, 1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("adhd", "wmrc"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--spectral-dim", type=int, default=8)
    parser.add_argument("--lambda-rank", type=float, default=0.05)
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def fold_paths(source_root, fold):
    root = source_root / "fold_{}".format(fold)
    paths = {
        "checkpoint": root / "neural" / "best_checkpoint.pt",
        "train": root / "cache" / "train" / "manifest.json",
        "validation": root / "cache" / "validation" / "manifest.json",
        "test": root / "cache" / "test" / "manifest.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("missing {} artifact: {}".format(name, path))
    return paths


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: {}\n".format(" ".join(str(value) for value in command)))
        handle.flush()
        return subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def update_status(path, dataset, completed, active=None, failed=None):
    atomic_json(
        path,
        {
            "artifact_type": "mokse_background_safe_static_matrix_status_v1",
            "dataset": dataset,
            "completed": list(completed),
            "completed_count": len(completed),
            "total_count": len(STAGES) * len(FOLDS),
            "active": active,
            "failed": failed,
        },
    )


def select_stage(args):
    command = [
        sys.executable,
        "-u",
        "scripts/select_mokse_background_safe_stage.py",
    ]
    for stage in STAGES:
        for fold in FOLDS:
            command.extend(
                [
                    "--{}-fold-dir".format(stage),
                    str(args.output_dir / "fold_{}".format(fold) / stage),
                ]
            )
    command.extend(["--output-dir", str(args.output_dir / "stage_selection")])
    return run_logged(command, args.output_dir / "logs" / "stage_selection.log")


def summarize(args):
    rows = []
    summary = {}
    for stage in STAGES:
        stage_rows = []
        for fold in FOLDS:
            path = args.output_dir / "fold_{}".format(fold) / stage / "evaluation.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "stage": stage,
                "fold": fold,
                "best_epoch": int(payload["best_epoch"]),
                "ensemble_size": int(payload["checkpoint_ensemble_size"]),
                "validation": payload["metrics"]["validation"],
                "test": payload["metrics"]["test"],
            }
            rows.append(row)
            stage_rows.append(row)
        aggregate = {}
        for split in ("validation", "test"):
            aggregate[split] = {}
            for metric in (
                "roc_auc",
                "accuracy",
                "balanced_accuracy",
                "auprc",
                "f1",
                "site_stratified_roc_auc",
            ):
                values = [
                    float(row[split][metric])
                    for row in stage_rows
                    if row[split].get(metric) is not None
                ]
                aggregate[split][metric + "_mean"] = (
                    float(np.mean(values)) if values else None
                )
                aggregate[split][metric + "_std"] = (
                    float(np.std(values)) if values else None
                )
        summary[stage] = aggregate
    selection_path = args.output_dir / "stage_selection" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    report = {
        "artifact_type": "mokse_background_safe_static_matrix_summary_v1",
        "dataset": args.dataset,
        "candidate_count": len(rows),
        "all_candidates_evaluated_on_fixed_test": True,
        "fixed_test_used_for_stage_selection": False,
        "selected_stage": selection["selected_stage"],
        "stage_summary": summary,
        "fold_rows": rows,
    }
    atomic_json(args.output_dir / "matrix_summary.json", report)
    lines = [
        "# {} MoKSE-Net-BG-Safe S0–S3静态分支评估".format(args.dataset.upper()),
        "",
        "- 四个候选均已在固定test评估：是",
        "- test参与checkpoint或阶段选择：否",
        "- development validation选中阶段：`{}`".format(selection["selected_stage"]),
        "",
        "| Stage | Val AUC | Val ACC | Test AUC | Test ACC | Test BA | Test AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        validation = summary[stage]["validation"]
        test = summary[stage]["test"]
        lines.append(
            "| {} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} | {:.6f} |".format(
                stage.upper(),
                validation["roc_auc_mean"], validation["roc_auc_std"],
                validation["accuracy_mean"], validation["accuracy_std"],
                test["roc_auc_mean"], test["roc_auc_std"],
                test["accuracy_mean"], test["accuracy_std"],
                test["balanced_accuracy_mean"], test["auprc_mean"],
            )
        )
    lines.extend(
        (
            "",
            "> 固定test结果只作并列审计，不能反向用于选择S0–S3。",
        )
    )
    atomic_text(args.output_dir / "matrix_summary.md", "\n".join(lines) + "\n")
    return report


def main():
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.global_root = args.global_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    status_path = args.output_dir / "matrix_status.json"
    for stage in STAGES:
        for fold in FOLDS:
            token = "{}_fold{}".format(stage, fold)
            output = args.output_dir / "fold_{}".format(fold) / stage
            if (output / "run_manifest.json").is_file():
                completed.append(token)
                update_status(status_path, args.dataset, completed)
                continue
            paths = fold_paths(args.source_root, fold)
            update_status(status_path, args.dataset, completed, active=token)
            command = [
                sys.executable,
                "-u",
                "scripts/run_mokse_background_safe_fold.py",
                "--checkpoint", str(paths["checkpoint"]),
                "--train-manifest", str(paths["train"]),
                "--validation-manifest", str(paths["validation"]),
                "--test-manifest", str(paths["test"]),
                "--evaluate-test",
                "--global-root", str(args.global_root),
                "--cache-dir", str(args.cache_dir),
                "--output-dir", str(output),
                "--stage", stage,
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--seed", str(args.seed),
                "--hidden-dim", str(args.hidden_dim),
                "--dropout", str(args.dropout),
                "--spectral-dim", str(args.spectral_dim),
                "--lambda-rank", str(args.lambda_rank),
            ]
            returncode = run_logged(
                command, args.output_dir / "logs" / (token + ".log")
            )
            if returncode != 0:
                update_status(
                    status_path,
                    args.dataset,
                    completed,
                    failed={"task": token, "returncode": returncode},
                )
                return returncode
            completed.append(token)
            update_status(status_path, args.dataset, completed)
    returncode = select_stage(args)
    if returncode != 0:
        update_status(
            status_path,
            args.dataset,
            completed,
            failed={"task": "stage_selection", "returncode": returncode},
        )
        return returncode
    report = summarize(args)
    update_status(status_path, args.dataset, completed)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
