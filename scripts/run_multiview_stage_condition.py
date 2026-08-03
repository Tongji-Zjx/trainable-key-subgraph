"""Run one resumable Stage-1/2/3 multi-view condition without test leakage."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402


def _run(name, command, done_path, print_only=False):
    row = {"name": name, "command": command, "done": str(done_path)}
    if print_only:
        return row
    if Path(done_path).is_file():
        print("SKIP {}: {} exists".format(name, done_path), flush=True)
        return row
    print("START {}".format(name), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    if not Path(done_path).is_file():
        raise RuntimeError("{} did not create {}".format(name, done_path))
    print("FINISH {}".format(name), flush=True)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--static-mode", choices=("stable", "neural", "residual"), required=True)
    parser.add_argument("--enable-v", action="store_true")
    parser.add_argument("--enable-g", action="store_true")
    parser.add_argument("--shuffle-correspondence", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lambda-q", type=float, default=0.1)
    parser.add_argument("--lambda-delta-q", type=float, default=0.1)
    parser.add_argument("--diagnostic-max-samples", type=int, default=64)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.shuffle_correspondence and not args.enable_v:
        parser.error("--shuffle-correspondence requires --enable-v")
    if args.evaluate_test and args.test_manifest is None:
        parser.error("--evaluate-test requires --test-manifest")
    return args


def main():
    args = parse_args()
    output = args.output_dir.absolute()
    checkpoint = output / "best_checkpoint.pt"
    python = sys.executable
    train = [
        python, "-u", "scripts/train_multiview_critical.py",
        "--train-manifest", str(args.train_manifest),
        "--validation-manifest", str(args.validation_manifest),
        "--scaler", str(args.scaler),
        "--output-dir", str(output),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--gradient-clip", str(args.gradient_clip),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--lambda-q", str(args.lambda_q),
        "--lambda-delta-q", str(args.lambda_delta_q),
        "--static-mode", args.static_mode,
    ]
    if not args.enable_v:
        train.append("--disable-v")
    if not args.enable_g:
        train.append("--disable-g")
    if args.shuffle_correspondence:
        train.append("--shuffle-correspondence")
    commands = [("train", train, checkpoint)]
    for split, manifest in (("validation", args.validation_manifest),):
        path = output / "{}_evaluation.json".format(split)
        commands.append((
            "evaluate_{}".format(split),
            [python, "-u", "scripts/evaluate_multiview_critical.py",
             "--manifest", str(manifest), "--scaler", str(args.scaler),
             "--checkpoint", str(checkpoint), "--output", str(path),
             "--device", args.device, "--batch-size", str(args.batch_size),
             "--num-workers", str(args.num_workers)],
            path,
        ))
    diagnostic = output / "validation_diagnostic.json"
    commands.append((
        "diagnose_validation",
        [python, "-u", "scripts/diagnose_multiview_critical.py",
         "--manifest", str(args.validation_manifest), "--scaler", str(args.scaler),
         "--checkpoint", str(checkpoint), "--output", str(diagnostic),
         "--device", args.device, "--num-workers", str(args.num_workers),
         "--max-samples", str(args.diagnostic_max_samples)],
        diagnostic,
    ))
    if args.evaluate_test:
        path = output / "test_evaluation.json"
        commands.append((
            "evaluate_test",
            [python, "-u", "scripts/evaluate_multiview_critical.py",
             "--manifest", str(args.test_manifest), "--scaler", str(args.scaler),
             "--checkpoint", str(checkpoint), "--output", str(path),
             "--device", args.device, "--batch-size", str(args.batch_size),
             "--num-workers", str(args.num_workers)],
            path,
        ))
    plan = [_run(name, command, done, args.print_only) for name, command, done in commands]
    if args.print_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        (output / "condition_spec.json").write_text(
            json.dumps({
                "condition_name": args.condition_name,
                "static_mode": args.static_mode,
                "enable_v": bool(args.enable_v),
                "enable_g": bool(args.enable_g),
                "correspondence": "shuffled" if args.shuffle_correspondence else "uot",
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "gradient_clip": float(args.gradient_clip),
                "early_stopping_patience": int(args.early_stopping_patience),
                "lambda_q": float(args.lambda_q),
                "lambda_delta_q": float(args.lambda_delta_q),
                "train_manifest": str(args.train_manifest),
                "train_manifest_sha256": file_sha256(args.train_manifest),
                "validation_manifest": str(args.validation_manifest),
                "validation_manifest_sha256": file_sha256(args.validation_manifest),
                "scaler": str(args.scaler),
                "scaler_sha256": file_sha256(args.scaler),
                "test_evaluated": bool(args.evaluate_test),
                "test_manifest": str(args.test_manifest) if args.test_manifest is not None else None,
                "test_manifest_sha256": (
                    file_sha256(args.test_manifest) if args.test_manifest is not None else None
                ),
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("condition completed: {}".format(args.condition_name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
