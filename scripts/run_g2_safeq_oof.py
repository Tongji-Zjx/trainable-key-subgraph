"""Run resumable three-fold G2-SafeQ cache, training and evaluation."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--g2-seed", type=int, required=True)
    parser.add_argument("--safeq-seed", type=int)
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precompute-batch-size", type=int, default=4)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def _path(value):
    return str(Path(value).resolve())


def build_fold_stages(args, fold):
    base = Path(args.base_root).resolve() / "fold_{}".format(fold)
    spectral = Path(args.spectral_root).resolve() / "fold_{}".format(fold)
    output = Path(args.output_root).resolve() / "fold_{}".format(fold)
    checkpoint = (
        spectral
        / "models"
        / "G2_seed{}".format(args.g2_seed)
        / "best_checkpoint.pt"
    )
    scaler = base / "scaler.json"
    spectral_scaler = spectral / "spectral_scaler.json"
    stages = []
    for split in ("train", "validation", "test"):
        cache = output / "cache" / split
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "precompute_g2_safeq.py"),
            "--split",
            split,
            "--manifest",
            _path(base / "cache" / split / "manifest.json"),
            "--scaler",
            _path(scaler),
            "--spectral-manifest",
            _path(spectral / "spectral_cache" / split / "manifest.json"),
            "--spectral-scaler",
            _path(spectral_scaler),
            "--g2-checkpoint",
            _path(checkpoint),
            "--output-dir",
            _path(cache),
            "--device",
            args.device,
            "--batch-size",
            str(args.precompute_batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        stages.append(
            ("cache_{}".format(split), command, cache / "manifest.json")
        )
    model_dir = output / "model"
    safeq_seed = args.g2_seed if args.safeq_seed is None else args.safeq_seed
    stages.append(
        (
            "train",
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "scripts" / "train_g2_safeq.py"),
                "--train-manifest",
                _path(output / "cache" / "train" / "manifest.json"),
                "--validation-manifest",
                _path(output / "cache" / "validation" / "manifest.json"),
                "--output-dir",
                _path(model_dir),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.training_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(safeq_seed),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
            ],
            model_dir / "best_checkpoint.pt",
        )
    )
    for split in ("validation", "test"):
        evaluation = output / "evaluation" / "{}.json".format(split)
        stages.append(
            (
                "evaluate_{}".format(split),
                [
                    sys.executable,
                    "-u",
                    str(PROJECT_ROOT / "scripts" / "evaluate_g2_safeq.py"),
                    "--manifest",
                    _path(output / "cache" / split / "manifest.json"),
                    "--checkpoint",
                    _path(model_dir / "best_checkpoint.pt"),
                    "--output",
                    _path(evaluation),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.evaluation_batch_size),
                    "--num-workers",
                    str(args.num_workers),
                ],
                evaluation,
            )
        )
    return stages


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    folds = tuple(int(value) for value in args.folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("SafeQ folds must be non-empty and unique")
    if any(value < 0 for value in folds):
        raise ValueError("SafeQ folds cannot be negative")
    if (
        args.precompute_batch_size < 1
        or args.training_batch_size < 1
        or args.evaluation_batch_size < 1
        or args.num_workers < 0
    ):
        raise ValueError("invalid SafeQ runner loader configuration")
    all_stages = []
    for fold in folds:
        all_stages.extend(
            (fold, name, command, indicator)
            for name, command, indicator in build_fold_stages(args, fold)
        )
    if args.print_only:
        for fold, name, command, indicator in all_stages:
            print(
                "fold={} stage={} indicator={}\n  {}".format(
                    fold,
                    name,
                    indicator,
                    " ".join(str(value) for value in command),
                )
            )
        return 0
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for fold, name, command, indicator in all_stages:
        if Path(indicator).is_file():
            print(
                "SKIP fold={} stage={}: {} exists".format(
                    fold, name, indicator
                ),
                flush=True,
            )
            continue
        print("START fold={} stage={}".format(fold, name), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not Path(indicator).is_file():
            raise RuntimeError(
                "SafeQ stage did not create indicator: {}".format(indicator)
            )
        print("FINISH fold={} stage={}".format(fold, name), flush=True)
    _atomic_json(
        output_root / "run_spec.json",
        {
            "artifact_type": "g2_safeq_oof_run",
            "base_root": _path(args.base_root),
            "spectral_root": _path(args.spectral_root),
            "output_root": _path(args.output_root),
            "g2_seed": int(args.g2_seed),
            "safeq_seed": int(
                args.g2_seed if args.safeq_seed is None else args.safeq_seed
            ),
            "folds": list(folds),
            "mixing_fit_split": "validation",
            "threshold_fit_split": "validation",
            "outer_test_tuning": False,
        },
    )
    print("SafeQ OOF stages complete: {}".format(output_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
