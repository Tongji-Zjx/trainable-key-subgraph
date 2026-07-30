"""Run resumable paired S/SV folds and compare them with frozen SVG OOF."""

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

from keysubgraph.crossfit.sv_branch_ablation import (  # noqa: E402
    compare_sv_branch_ablation,
)
from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)


S_VARIANT = "static_spectral_only"
SV_VARIANT = "static_spectral_variation_late_fusion"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("NAME", "CROSSFIT_ROOT", "SVG_SUMMARY"),
        required=True,
        help="repeat for each dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument("--statistics-seed", type=int, default=202607)
    return parser.parse_args()


def _summary_dir(root, variant):
    return Path(root) / "summary_{}_v1".format(variant)


def _complete_summary(path):
    return all(
        (Path(path) / name).is_file()
        for name in ("summary.json", "oof_predictions.csv", "summary.md")
    )


def main():
    args = parse_args()
    datasets = []
    for name, root_value, svg_summary_value in args.dataset:
        root = Path(root_value).resolve()
        assignments = root / "assignments" / "fold_assignments.json"
        with assignments.open("r", encoding="utf-8") as handle:
            assignment_payload = json.load(handle)
        fold_count = int(assignment_payload["num_outer_folds"])
        for fold in range(fold_count):
            command = [
                sys.executable,
                "-u",
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "run_sv_signed_gin_crossfit_fold.py"
                ),
                "--fold",
                str(fold),
                "--output-root",
                str(root),
                "--variants",
                S_VARIANT,
                SV_VARIANT,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--model-epochs",
                str(args.model_epochs),
                "--num-workers",
                str(args.num_workers),
            ]
            print(
                "START dataset={} fold={} variants=S,SV".format(
                    name, fold
                ),
                flush=True,
            )
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
            print(
                "FINISH dataset={} fold={}".format(name, fold),
                flush=True,
            )
        summary_dirs = {}
        for variant in (S_VARIANT, SV_VARIANT):
            summary_dir = _summary_dir(root, variant)
            if _complete_summary(summary_dir):
                print(
                    "SKIP summary dataset={} variant={}".format(
                        name, variant
                    ),
                    flush=True,
                )
            else:
                summarize_sv_signed_gin_crossfit(
                    output_root=root,
                    fold_assignments=assignments,
                    variant=variant,
                    seed=args.seed,
                    output_dir=summary_dir,
                )
                print(
                    "FINISH summary dataset={} variant={}".format(
                        name, variant
                    ),
                    flush=True,
                )
            summary_dirs[variant] = summary_dir
        svg_summary = Path(svg_summary_value).resolve()
        if not _complete_summary(svg_summary):
            raise FileNotFoundError(
                "frozen SVG OOF summary is incomplete: {}".format(
                    svg_summary
                )
            )
        datasets.append(
            (
                name,
                summary_dirs[S_VARIANT],
                summary_dirs[SV_VARIANT],
                svg_summary,
            )
        )
    comparison = compare_sv_branch_ablation(
        datasets=datasets,
        output_dir=args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.statistics_seed,
    )
    print(
        json.dumps(
            {
                "comparison_json": str(
                    comparison["comparison_json"]
                ),
                "comparison_markdown": str(
                    comparison["comparison_markdown"]
                ),
                "datasets": sorted(comparison["datasets"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
