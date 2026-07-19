"""Build immutable subject-grouped outer folds for confirmatory cross-fitting."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.crossfit_split import (  # noqa: E402
    create_crossfit_fold_assignments,
    create_outer_folds,
    write_crossfit_fold_artifacts,
    write_outer_fold_artifacts,
)
from keysubgraph.data.data_split import read_sample_index  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-index",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "index_no_coords" / "sample_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "crossfit",
    )
    parser.add_argument("--num-outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--inner-seed", type=int, default=202608)
    parser.add_argument("--inner-validation-ratio", type=float, default=0.1875)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = read_sample_index(args.sample_index)
    assignments = create_outer_folds(
        samples, num_folds=args.num_outer_folds, seed=args.seed
    )
    outer_result = write_outer_fold_artifacts(
        assignments,
        args.output_dir,
        args.sample_index,
        num_folds=args.num_outer_folds,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    fold_assignments = create_crossfit_fold_assignments(
        samples,
        assignments,
        inner_validation_ratio=args.inner_validation_ratio,
        seed=args.inner_seed,
    )
    fold_result = write_crossfit_fold_artifacts(
        fold_assignments,
        args.output_dir,
        Path(outer_result["json"]),
        args.sample_index,
        inner_validation_ratio=args.inner_validation_ratio,
        seed=args.inner_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(
        {"outer": outer_result, "folds": fold_result},
        ensure_ascii=False, indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
