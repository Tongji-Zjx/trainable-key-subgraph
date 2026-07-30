"""Run independent Full/Hard SV channels and frozen late fusion."""

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

from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SV_SIGNED_GIN_VARIANTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--selector-checkpoint", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=SV_SIGNED_GIN_VARIANTS,
        default="sv_static_variation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15
    )
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "composite_auc"),
        default="composite_auc",
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_experiment_steps(args):
    root = Path(args.output_dir)
    python = sys.executable
    steps = []
    for source, mode in (("hard", "learned"), ("full", "full")):
        for split in ("train", "validation", "test"):
            output = root / "cache" / source / split
            command = [
                python,
                "-u",
                "scripts/precompute_sv_signed_gin_cache.py",
                "--protocol",
                str(args.protocol),
                "--selection-mode",
                mode,
                "--split",
                split,
                "--output-dir",
                str(output),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
                "--selection-seed",
                str(args.seed),
            ]
            if source == "hard":
                command.extend(
                    [
                        "--selector-checkpoint",
                        str(args.selector_checkpoint),
                    ]
                )
            steps.append(
                {
                    "name": "cache_{}_{}".format(source, split),
                    "expected": output / "manifest.json",
                    "command": command,
                }
            )
        scaler = root / "scalers" / "{}.json".format(source)
        steps.append(
            {
                "name": "scaler_" + source,
                "expected": scaler,
                "command": [
                    python,
                    "-u",
                    "scripts/fit_sv_signed_gin_scalers.py",
                    "--train-manifest",
                    str(root / "cache" / source / "train/manifest.json"),
                    "--output",
                    str(scaler),
                ],
            }
        )
        training = root / "training" / source
        steps.append(
            {
                "name": "train_" + source,
                "expected": training / "best_checkpoint.pt",
                "command": [
                    python,
                    "-u",
                    "scripts/train_sv_signed_gin.py",
                    "--train-manifest",
                    str(root / "cache" / source / "train/manifest.json"),
                    "--validation-manifest",
                    str(
                        root
                        / "cache"
                        / source
                        / "validation/manifest.json"
                    ),
                    "--scaler",
                    str(scaler),
                    "--variant",
                    args.variant,
                    "--output-dir",
                    str(training),
                    "--device",
                    args.device,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--gradient-accumulation-steps",
                    str(args.gradient_accumulation_steps),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.seed),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--gradient-clip",
                    str(args.gradient_clip),
                    "--early-stopping-patience",
                    str(args.early_stopping_patience),
                    "--selection-metric",
                    args.selection_metric,
                ],
            }
        )
        for split in ("validation", "test"):
            evaluation = (
                root
                / "evaluation"
                / "{}_{}.json".format(source, split)
            )
            steps.append(
                {
                    "name": "evaluate_{}_{}".format(source, split),
                    "expected": evaluation,
                    "command": [
                        python,
                        "-u",
                        "scripts/evaluate_sv_signed_gin.py",
                        "--manifest",
                        str(
                            root
                            / "cache"
                            / source
                            / split
                            / "manifest.json"
                        ),
                        "--scaler",
                        str(scaler),
                        "--checkpoint",
                        str(training / "best_checkpoint.pt"),
                        "--threshold-strategy",
                        "balanced_accuracy",
                        "--output",
                        str(evaluation),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                    ],
                }
            )
    fusion = root / "fusion"
    steps.append(
        {
            "name": "fuse_full_hard",
            "expected": fusion / "evaluation.json",
            "command": [
                python,
                "-u",
                "scripts/evaluate_sv_full_hard_late_fusion.py",
                "--hard-validation",
                str(root / "evaluation/hard_validation.json"),
                "--hard-test",
                str(root / "evaluation/hard_test.json"),
                "--full-validation",
                str(root / "evaluation/full_validation.json"),
                "--full-test",
                str(root / "evaluation/full_test.json"),
                "--alpha-grid",
                *[str(value) for value in args.alpha_grid],
                "--output-dir",
                str(fusion),
            ],
        }
    )
    return steps


def main():
    args = parse_args()
    args.protocol = Path(args.protocol).resolve()
    args.selector_checkpoint = Path(
        args.selector_checkpoint
    ).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    steps = build_experiment_steps(args)
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "name": step["name"],
                        "expected": str(step["expected"]),
                        "command": step["command"],
                    }
                    for step in steps
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not Path(args.protocol).is_file():
        raise FileNotFoundError(str(args.protocol))
    if not Path(args.selector_checkpoint).is_file():
        raise FileNotFoundError(str(args.selector_checkpoint))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for step in steps:
        expected = Path(step["expected"])
        if expected.is_file():
            print("SKIP {}: {} exists".format(step["name"], expected))
            continue
        print("START {}".format(step["name"]), flush=True)
        subprocess.run(
            step["command"], cwd=str(PROJECT_ROOT), check=True
        )
        if not expected.is_file():
            raise RuntimeError(
                "{} did not create {}".format(step["name"], expected)
            )
        print("FINISH {}".format(step["name"]), flush=True)
    completion = Path(args.output_dir) / "COMPLETE"
    completion.write_text("complete\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(Path(args.output_dir).resolve()),
                "fusion_summary": str(
                    (
                        Path(args.output_dir)
                        / "fusion"
                        / "summary.md"
                    ).resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
