"""Run one resumable Full-Soft-Hard selector outer fold."""

from __future__ import absolute_import, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _command(*values):
    return [str(value) for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.fold < 0:
        raise ValueError("fold must be non-negative")
    fold_root = args.output_root.resolve() / "fold_{}".format(args.fold)
    protocol = fold_root / "protocol" / "data_protocol.json"
    if not protocol.is_file():
        raise FileNotFoundError(str(protocol))
    run = fold_root / "selector_transfer_full_soft_hard_seed{}".format(
        args.seed
    )
    selector = run / "selector"
    checkpoint = selector / "best_checkpoint.pt"
    cache = run / "cache"
    commands = [
        (
            "train_selector",
            _command(
                sys.executable, "-u",
                PROJECT_ROOT / "scripts" / "train_dual_selector.py",
                "--protocol", protocol,
                "--output-dir", selector,
                "--device", args.device,
                "--epochs", args.epochs,
                "--batch-size", 1,
                "--num-workers", args.num_workers,
                "--seed", args.seed,
                "--learning-rate", 0.001,
                "--weight-decay", 0.0001,
                "--gradient-clip", 1.0,
                "--early-stopping-patience", 15,
                "--selector-objective", "full_soft_hard",
            ),
            selector / "best_evaluation.json",
        )
    ]
    for split in ("train", "validation", "test"):
        directory = cache / split
        commands.append((
            "cache_{}".format(split),
            _command(
                sys.executable, "-u",
                PROJECT_ROOT / "scripts" / "precompute_sv_signed_gin_cache.py",
                "--protocol", protocol,
                "--selector-checkpoint", checkpoint,
                "--selection-mode", "learned",
                "--split", split,
                "--output-dir", directory,
                "--device", args.device,
                "--num-workers", args.num_workers,
                "--selection-seed", args.seed,
            ),
            directory / "manifest.json",
        ))
    evaluation = run / "outer_fold_evaluation.json"
    commands.append((
        "evaluate_outer_fold",
        _command(
            sys.executable, "-u",
            PROJECT_ROOT / "scripts" / "evaluate_selector_transfer_oof_fold.py",
            "--train-manifest", cache / "train" / "manifest.json",
            "--validation-manifest", cache / "validation" / "manifest.json",
            "--test-manifest", cache / "test" / "manifest.json",
            "--output", evaluation,
            "--seed", args.seed,
        ),
        evaluation,
    ))
    if args.print_only:
        print(json.dumps([
            {
                "stage": name,
                "command": command,
                "completion_artifact": str(artifact),
            }
            for name, command, artifact in commands
        ], ensure_ascii=False, indent=2))
        return 0
    for name, command, artifact in commands:
        if artifact.is_file():
            print("SKIP {}: {}".format(name, artifact), flush=True)
            continue
        print("START {}".format(name), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not artifact.is_file():
            raise RuntimeError(
                "stage did not create {}".format(artifact)
            )
        print("FINISH {}".format(name), flush=True)
    print("FOLD {} COMPLETE".format(args.fold), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
