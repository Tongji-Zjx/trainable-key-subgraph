"""Freeze an exploratory protocol that assigns every indexed sample to ``all``."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import (  # noqa: E402
    freeze_data_protocol,
    validate_data_protocol,
)
from keysubgraph.data.data_split import read_sample_index  # noqa: E402
from keysubgraph.data.full_cohort import (  # noqa: E402
    FULL_COHORT_MODE,
    create_full_cohort_assignments,
    write_full_cohort_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "adhd_5_0.5"
    )
    parser.add_argument(
        "--sample-index",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "index_no_coords" / "sample_index.csv",
    )
    parser.add_argument(
        "--assignment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "all_samples_protocol",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_all_samples.json",
    )
    parser.add_argument("--edge-presence-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        csv_path = args.assignment_dir.resolve() / "splits.csv"
        json_path = args.assignment_dir.resolve() / "splits.json"
        if csv_path.exists() != json_path.exists():
            raise RuntimeError("only one full-cohort assignment artifact exists")
        if not csv_path.exists():
            samples = read_sample_index(args.sample_index)
            assignments = create_full_cohort_assignments(samples, seed=args.seed)
            write_full_cohort_artifacts(
                assignments, args.sample_index, args.assignment_dir
            )
        protocol = validate_data_protocol(args.output, PROJECT_ROOT)
        if protocol.get("experiment_mode") != FULL_COHORT_MODE:
            raise ValueError("existing protocol is not an all-sample protocol")
        print("Existing all-sample protocol found; hashes are valid and it will be reused.")
    else:
        samples = read_sample_index(args.sample_index)
        assignments = create_full_cohort_assignments(samples, seed=args.seed)
        artifacts = write_full_cohort_artifacts(
            assignments,
            args.sample_index,
            args.assignment_dir,
            overwrite=args.overwrite,
        )
        protocol = freeze_data_protocol(
            project_root=PROJECT_ROOT,
            dataset_root=args.dataset_root,
            sample_index_csv=args.sample_index,
            splits_csv=artifacts["csv"],
            splits_json=artifacts["json"],
            output_path=args.output,
            edge_presence_threshold=args.edge_presence_threshold,
            protocol_name="all_samples_exploratory",
            overwrite=args.overwrite,
        )
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
