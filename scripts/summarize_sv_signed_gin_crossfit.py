"""Audit and summarize all outer-fold SV Signed-GIN predictions."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument(
        "--variant", default="signed_gin_static_variation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        help="optional model directory name, e.g. C3_G2_seed42",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assignments = (
        args.fold_assignments
        if args.fold_assignments is not None
        else args.output_root
        / "assignments"
        / "fold_assignments.json"
    )
    result = summarize_sv_signed_gin_crossfit(
        args.output_root,
        assignments,
        variant=args.variant,
        seed=args.seed,
        run_name=args.run_name,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in result.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
