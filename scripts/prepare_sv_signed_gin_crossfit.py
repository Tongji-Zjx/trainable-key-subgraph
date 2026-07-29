"""Prepare immutable group-aware folds for SV Signed-GIN cross-fitting."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_protocol import (  # noqa: E402
    prepare_fold_protocol,
)
from keysubgraph.data.crossfit_split import (  # noqa: E402
    create_crossfit_fold_assignments,
    create_outer_folds,
    write_crossfit_fold_artifacts,
    write_outer_fold_artifacts,
)
from keysubgraph.data.data_protocol import (  # noqa: E402
    validate_data_protocol,
)
from keysubgraph.data.data_split import (  # noqa: E402
    read_sample_index,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_wmrc_no_coord.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "sv_signed_gin_crossfit"
        / "wmrc_3fold_seed202607",
    )
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=202607)
    parser.add_argument("--inner-seed", type=int, default=202608)
    parser.add_argument(
        "--inner-validation-ratio", type=float, default=0.1875
    )
    parser.add_argument(
        "--group-key",
        choices=("subject_id", "site_subject"),
        default="site_subject",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(
        args.source_protocol, PROJECT_ROOT
    )
    sample_index = (
        PROJECT_ROOT / protocol["paths"]["sample_index_csv"]
    )
    samples = read_sample_index(sample_index)
    assignment_root = args.output_root / "assignments"
    outer = create_outer_folds(
        samples,
        num_folds=args.num_folds,
        seed=args.outer_seed,
        group_key=args.group_key,
    )
    outer_result = write_outer_fold_artifacts(
        outer,
        assignment_root,
        sample_index,
        num_folds=args.num_folds,
        seed=args.outer_seed,
        overwrite=args.overwrite,
        group_key=args.group_key,
    )
    roles = create_crossfit_fold_assignments(
        samples,
        outer,
        inner_validation_ratio=args.inner_validation_ratio,
        seed=args.inner_seed,
        group_key=args.group_key,
    )
    role_result = write_crossfit_fold_artifacts(
        roles,
        assignment_root,
        Path(outer_result["json"]),
        sample_index,
        inner_validation_ratio=args.inner_validation_ratio,
        seed=args.inner_seed,
        overwrite=args.overwrite,
        group_key=args.group_key,
    )
    folds = []
    for fold in range(args.num_folds):
        result = prepare_fold_protocol(
            PROJECT_ROOT,
            role_result["json"],
            args.source_protocol,
            fold,
            args.output_root,
            overwrite=args.overwrite,
        )
        folds.append(
            {
                "fold": fold,
                "protocol": str(result["protocol"]),
                "summary": result["summary"],
            }
        )
    output = {
        "artifact_type": "sv_signed_gin_crossfit_preparation",
        "source_protocol": str(args.source_protocol),
        "source_sample_count": len(samples),
        "num_folds": args.num_folds,
        "outer_seed": args.outer_seed,
        "inner_seed": args.inner_seed,
        "inner_validation_ratio": args.inner_validation_ratio,
        "group_key": args.group_key,
        "outer_assignments": outer_result,
        "fold_assignments": role_result,
        "folds": folds,
    }
    print(
        json.dumps(
            output, ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
