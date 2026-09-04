#!/usr/bin/env python3
"""Refit downstream C4/XGB and S4 for one conditional development-OOF fold."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("adhd", "wmrc"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--source-fold-dir", type=Path, required=True)
    parser.add_argument("--tge-config", type=Path, required=True)
    parser.add_argument("--xgb-config", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--static-source-root", type=Path, required=True)
    parser.add_argument("--subgraph-test-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inner-split-seed", type=int, default=20260905)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--static-seeds", default="43,44,45")
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: {}\n".format(" ".join(str(x) for x in command)))
        handle.flush()
        result = subprocess.run(
            [str(x) for x in command],
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def copy_with_sidecar(source, target):
    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))
    sidecar = source.with_suffix(source.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    shutil.copy2(str(sidecar), str(target.with_suffix(target.suffix + ".json")))


def write_subgraph_oof_provenance(path, cache_report, xgb_dir):
    xgb_evaluation = json.loads(
        (Path(xgb_dir) / "evaluation.json").read_text(encoding="utf-8")
    )
    xgb_config_path = Path(
        xgb_evaluation.get("provenance", {}).get("config", "")
    )
    xgb_config = (
        json.loads(xgb_config_path.read_text(encoding="utf-8"))
        if xgb_config_path.is_file()
        else {}
    )
    provenance = {
        "artifact_type": "mokse_conditional_oof_subgraph_prediction_v1",
        "prediction_role": "development_oof",
        "conditional_on_frozen_selector_and_trajectory_cache": True,
        "end_to_end_selector_oof": False,
        "oof_disjointness_audit": {
            "train_sample_count": cache_report["sample_count"]["inner_train"],
            "checkpoint_validation_sample_count": cache_report["sample_count"]["inner_validation"],
            "oof_sample_count": cache_report["sample_count"]["oof_target"],
            "all_disjoint": True,
        },
        "xgb_directory": str(Path(xgb_dir).resolve()),
        "historical_fixed_test_guided_hyperparameters": bool(
            xgb_config.get("historical_fixed_test_guided_hyperparameters", False)
        ),
        "test_used_for_fit": False,
    }
    atomic_json(path.with_suffix(path.suffix + ".json"), provenance)


def main():
    args = parse_args()
    seeds = tuple(int(value) for value in args.static_seeds.split(",") if value)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct S4 seeds are required")
    source = args.source_fold_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    derived = output / "derived_cache"
    derivation_path = derived / "derivation.json"
    if not derivation_path.is_file():
        run(
            (
                sys.executable,
                "-u",
                "scripts/build_mokse_s4_conditional_oof_cache.py",
                "--train-manifest", source / "cache/train/manifest.json",
                "--target-validation-manifest", source / "cache/validation/manifest.json",
                "--output-dir", derived,
                "--inner-validation-fraction", args.inner_validation_fraction,
                "--seed", args.inner_split_seed,
            ),
            logs / "derive_cache.log",
        )
    cache_report = json.loads(derivation_path.read_text(encoding="utf-8"))
    train_manifest = Path(cache_report["manifests"]["inner_train"])
    validation_manifest = Path(cache_report["manifests"]["inner_validation"])
    target_manifest = Path(cache_report["manifests"]["oof_target"])

    neural = output / "subgraph/neural"
    checkpoint = neural / "best_checkpoint.pt"
    if not checkpoint.is_file():
        command = [
            sys.executable, "-u", "scripts/train_tge_gnn.py",
            "--config", args.tge_config.resolve(),
            "--train-manifest", train_manifest,
            "--validation-manifest", validation_manifest,
            "--output-dir", neural,
            "--device", args.device,
            "--cache-dataset-device",
        ]
        last = neural / "last_checkpoint.pt"
        if last.is_file():
            command.extend(("--resume-checkpoint", last))
        run(command, logs / "tge_train.log")

    xgb_dir = output / "subgraph/xgb"
    if not (xgb_dir / "validation_evaluation.json").is_file():
        run(
            (
                sys.executable, "-u", "scripts/run_tge_c4_xgb_residual.py",
                "--config", args.xgb_config.resolve(),
                "--train-manifest", train_manifest,
                "--validation-manifest", validation_manifest,
                "--test-manifest", target_manifest,
                "--checkpoint", checkpoint,
                "--output-dir", xgb_dir,
                "--phase", "validation",
                "--device", args.device,
                "--batch-size", 32,
            ),
            logs / "xgb.log",
        )
    if not (xgb_dir / "test_predictions.csv").is_file():
        run(
            (
                sys.executable, "-u", "scripts/run_tge_c4_xgb_residual.py",
                "--config", args.xgb_config.resolve(),
                "--train-manifest", train_manifest,
                "--validation-manifest", validation_manifest,
                "--test-manifest", target_manifest,
                "--checkpoint", checkpoint,
                "--output-dir", xgb_dir,
                "--phase", "test",
                "--device", args.device,
                "--batch-size", 32,
            ),
            logs / "xgb.log",
        )
    subgraph_export = output / "subgraph/predictions"
    subgraph_export.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        str(xgb_dir / "test_predictions.csv"),
        str(subgraph_export / "oof_predictions.csv"),
    )
    write_subgraph_oof_provenance(
        subgraph_export / "oof_predictions.csv", cache_report, xgb_dir
    )
    copy_with_sidecar(
        args.subgraph_test_dir.resolve() / "test_predictions.csv",
        subgraph_export / "test_predictions.csv",
    )

    for seed in seeds:
        static = output / "static/seed_{}".format(seed)
        if not (static / "run_manifest.json").is_file():
            run(
                (
                    sys.executable, "-u", "scripts/run_mokse_background_safe_fold.py",
                    "--checkpoint", checkpoint,
                    "--train-manifest", train_manifest,
                    "--validation-manifest", validation_manifest,
                    "--test-manifest", target_manifest,
                    "--evaluate-test",
                    "--global-root", args.global_root.resolve(),
                    "--cache-dir", args.feature_cache_dir.resolve(),
                    "--output-dir", static,
                    "--stage", "s4",
                    "--device", args.device,
                    "--epochs", 120,
                    "--batch-size", 16,
                    "--learning-rate", 0.001,
                    "--weight-decay", 0.0001,
                    "--patience", 15,
                    "--seed", seed,
                    "--hidden-dim", 64,
                    "--dropout", 0.1,
                    "--spectral-dim", 8,
                    "--lambda-rank", 0.05,
                ),
                logs / "static_seed_{}.log".format(seed),
            )
        if not (static / "oof_features.npz").is_file():
            run(
                (
                    sys.executable, "-u", "scripts/export_mokse_s4_checkpoint.py",
                    "--run-dir", static,
                    "--tge-checkpoint", checkpoint,
                    "--manifest", target_manifest,
                    "--manifest-split", "test",
                    "--prediction-role", "development_oof",
                    "--train-manifest", train_manifest,
                    "--checkpoint-validation-manifest", validation_manifest,
                    "--global-root", args.global_root.resolve(),
                    "--cache-dir", args.feature_cache_dir.resolve(),
                    "--output", static / "oof_features.npz",
                    "--device", args.device,
                    "--batch-size", 32,
                ),
                logs / "static_seed_{}_export.log".format(seed),
            )
        fixed_source = args.static_source_root.resolve() / (
            "fold_{}/seed_{}".format(args.fold, seed)
        )
        copy_with_sidecar(
            fixed_source / "test_features.npz", static / "test_features.npz"
        )

    summary = {
        "artifact_type": "mokse_s4_conditional_oof_fold_v1",
        "dataset": args.dataset,
        "fold": args.fold,
        "conditional_on_frozen_selector_and_trajectory_cache": True,
        "end_to_end_selector_oof": False,
        "cache_derivation": str(derivation_path),
        "tge_checkpoint": str(checkpoint),
        "xgb_directory": str(xgb_dir),
        "static_seeds": list(seeds),
        "complete": True,
    }
    atomic_json(output / "fold_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
