"""Create or report the reusable train/validation/test split files."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import (  # noqa: E402
    SplitConfig,
    create_data_splits,
    file_sha256,
    read_sample_index,
    read_split_assignments,
    summarize_assignments,
    validate_assignments,
    write_split_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "index" / "sample_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing split artifacts. Normally they must be reused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.output_dir.resolve() / "splits.csv"
    json_path = args.output_dir.resolve() / "splits.json"
    if (csv_path.exists() or json_path.exists()) and not args.overwrite:
        if not (csv_path.exists() and json_path.exists()):
            raise RuntimeError("only one split artifact exists; manual inspection is required")
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        current_digest = file_sha256(args.index_csv)
        if payload.get("source_index_sha256") != current_digest:
            raise RuntimeError(
                "sample index has changed since the split was created; "
                "review it and explicitly use --overwrite if a new split is intended"
            )
        stored_ratios = payload["ratios"]
        stored_config = SplitConfig(
            train_ratio=stored_ratios["train"],
            validation_ratio=stored_ratios["validation"],
            test_ratio=stored_ratios["test"],
            seed=payload["seed"],
            max_class_ratio_deviation=payload["summary"]["checks"][
                "allowed_class_ratio_deviation"
            ],
        )
        assignments = read_split_assignments(csv_path)
        validate_assignments(assignments, stored_config)
        if [item.to_dict() for item in assignments] != payload.get("assignments"):
            raise RuntimeError("splits.csv and splits.json assignments do not match")
        print("Existing split artifacts found; reusing them without random re-splitting.")
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    config = SplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    samples = read_sample_index(args.index_csv)
    assignments = create_data_splits(samples, config)
    paths = write_split_artifacts(
        assignments, args.output_dir, args.index_csv, config, overwrite=args.overwrite
    )
    print(json.dumps(summarize_assignments(assignments, config), ensure_ascii=False, indent=2, sort_keys=True))
    for name, path in sorted(paths.items()):
        print("{}: {}".format(name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
