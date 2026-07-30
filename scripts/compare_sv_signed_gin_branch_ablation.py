"""Compare paired S/SV/SVG OOF predictions across one or more datasets."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_branch_ablation import (  # noqa: E402
    compare_sv_branch_ablation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=4,
        action="append",
        metavar=("NAME", "S_SUMMARY", "SV_SUMMARY", "SVG_SUMMARY"),
        required=True,
        help="repeat for each dataset; paths are OOF summary directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [
        (name, Path(s_path), Path(sv_path), Path(svg_path))
        for name, s_path, sv_path, svg_path in args.dataset
    ]
    result = compare_sv_branch_ablation(
        datasets=datasets,
        output_dir=args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "comparison_json": str(result["comparison_json"]),
                "comparison_markdown": str(
                    result["comparison_markdown"]
                ),
                "datasets": sorted(result["datasets"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
