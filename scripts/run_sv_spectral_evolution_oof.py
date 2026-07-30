"""Run resumable WMRC/ADHD S+E OOF training and time-order diagnostics."""

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

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)
from keysubgraph.crossfit.sv_spectral_evolution_comparison import (  # noqa: E402
    compare_sv_spectral_evolution,
)


S_VARIANT = "static_spectral_only"
SE_VARIANT = "static_spectral_neural_evolution"
SHUFFLED_VARIANT = "static_spectral_neural_evolution_time_shuffled"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "CROSSFIT_ROOT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument("--statistics-seed", type=int, default=202607)
    return parser.parse_args()


def _run(stage, command, artifact):
    artifact = Path(artifact)
    if artifact.is_file():
        print("SKIP {}: {} exists".format(stage, artifact), flush=True)
        return
    print("START {}".format(stage), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    if not artifact.is_file():
        raise RuntimeError(
            "stage did not create completion artifact: {}".format(artifact)
        )
    print("FINISH {}".format(stage), flush=True)


def _complete_summary(path):
    return all(
        (Path(path) / name).is_file()
        for name in ("summary.json", "oof_predictions.csv", "summary.md")
    )


def main():
    args = parse_args()
    datasets = []
    for name, root_value in args.dataset:
        root = Path(root_value).resolve()
        assignments = root / "assignments" / "fold_assignments.json"
        with assignments.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        fold_count = int(payload["num_outer_folds"])
        for fold in range(fold_count):
            fold_root = root / "fold_{}".format(fold)
            cache = fold_root / "cache"
            static_scaler = fold_root / "scaler.json"
            transition_scaler = (
                fold_root / "spectral_transition_scaler.json"
            )
            anchor = (
                fold_root
                / "models"
                / "{}_seed{}".format(S_VARIANT, args.seed)
                / "best_checkpoint.pt"
            )
            run = (
                fold_root
                / "models"
                / "{}_seed{}".format(SE_VARIANT, args.seed)
            )
            shuffled_run = (
                fold_root
                / "models"
                / "{}_seed{}".format(SHUFFLED_VARIANT, args.seed)
            )
            required = (
                cache / "train" / "manifest.json",
                cache / "validation" / "manifest.json",
                cache / "test" / "manifest.json",
                static_scaler,
                anchor,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "S reuse artifacts are missing: {}".format(missing)
                )
            _run(
                "{} fold{} transition_scaler".format(name, fold),
                [
                    sys.executable,
                    "-u",
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "fit_sv_spectral_transition_scaler.py"
                    ),
                    "--train-manifest",
                    str(cache / "train" / "manifest.json"),
                    "--output",
                    str(transition_scaler),
                ],
                transition_scaler,
            )
            _run(
                "{} fold{} train_SE".format(name, fold),
                [
                    sys.executable,
                    "-u",
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "train_sv_spectral_evolution.py"
                    ),
                    "--train-manifest",
                    str(cache / "train" / "manifest.json"),
                    "--validation-manifest",
                    str(cache / "validation" / "manifest.json"),
                    "--static-scaler",
                    str(static_scaler),
                    "--transition-scaler",
                    str(transition_scaler),
                    "--anchor-checkpoint",
                    str(anchor),
                    "--output-dir",
                    str(run),
                    "--device",
                    args.device,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.seed),
                    "--learning-rate",
                    "0.001",
                    "--weight-decay",
                    "0.0001",
                    "--gradient-clip",
                    "1.0",
                    "--early-stopping-patience",
                    "15",
                    "--selection-metric",
                    "composite_auc",
                    "--auxiliary-loss-weight",
                    "0.25",
                ],
                run / "best_evaluation.json",
            )
            base_evaluation = [
                sys.executable,
                "-u",
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "evaluate_sv_spectral_evolution.py"
                ),
                "--manifest",
                str(cache / "test" / "manifest.json"),
                "--static-scaler",
                str(static_scaler),
                "--transition-scaler",
                str(transition_scaler),
                "--anchor-checkpoint",
                str(anchor),
                "--checkpoint",
                str(run / "best_checkpoint.pt"),
                "--threshold-strategy",
                "balanced_accuracy",
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
            original_output = run / "outer_test_evaluation.json"
            _run(
                "{} fold{} evaluate_SE".format(name, fold),
                base_evaluation
                + [
                    "--output",
                    str(original_output),
                    "--evaluation-variant",
                    SE_VARIANT,
                ],
                original_output,
            )
            shuffled_output = (
                shuffled_run / "outer_test_evaluation.json"
            )
            _run(
                "{} fold{} evaluate_shuffled".format(name, fold),
                base_evaluation
                + [
                    "--output",
                    str(shuffled_output),
                    "--shuffle-time",
                    "--shuffle-seed",
                    str(args.shuffle_seed),
                    "--evaluation-variant",
                    SHUFFLED_VARIANT,
                ],
                shuffled_output,
            )
        summary_dirs = {
            "S": root / "summary_{}_v1".format(S_VARIANT),
            "SE": root / "summary_{}_v1".format(SE_VARIANT),
            "SE_shuffled": root
            / "summary_{}_v1".format(SHUFFLED_VARIANT),
        }
        for condition, variant in (
            ("S", S_VARIANT),
            ("SE", SE_VARIANT),
            ("SE_shuffled", SHUFFLED_VARIANT),
        ):
            if not _complete_summary(summary_dirs[condition]):
                summarize_sv_signed_gin_crossfit(
                    root,
                    assignments,
                    variant=variant,
                    seed=args.seed,
                    output_dir=summary_dirs[condition],
                )
        datasets.append(
            (
                name,
                summary_dirs["S"],
                summary_dirs["SE"],
                summary_dirs["SE_shuffled"],
            )
        )
    result = compare_sv_spectral_evolution(
        datasets,
        args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.statistics_seed,
    )
    print(
        json.dumps(
            {
                "comparison_json": str(result["comparison_json"]),
                "comparison_markdown": str(
                    result["comparison_markdown"]
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
