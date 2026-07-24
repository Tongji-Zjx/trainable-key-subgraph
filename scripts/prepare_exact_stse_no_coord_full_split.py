"""Create and verify the frozen 938-sample Exact-STSE-NoCoord split."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
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


EXPECTED_SAMPLE_COUNT = 938
EXPECTED_PROTOCOL_NAME = "exact_stse_no_coord_full_cohort"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "index_no_coords"
        / "sample_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "exact_stse_no_coord_full_splits",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_no_coord_full.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SplitConfig(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=2026,
    )
    samples = read_sample_index(args.index_csv)
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "expected {} valid samples, found {}".format(
                EXPECTED_SAMPLE_COUNT, len(samples)
            )
        )

    csv_path = args.output_dir.resolve() / "splits.csv"
    json_path = args.output_dir.resolve() / "splits.json"
    if args.overwrite or not (csv_path.is_file() and json_path.is_file()):
        if not args.overwrite and (csv_path.exists() or json_path.exists()):
            raise RuntimeError(
                "only one split artifact exists; inspect it before using "
                "--overwrite"
            )
        assignments = create_data_splits(samples, config)
        write_split_artifacts(
            assignments,
            args.output_dir,
            args.index_csv,
            config,
            overwrite=args.overwrite,
        )
    else:
        assignments = read_split_assignments(csv_path)
        validate_assignments(assignments, config)
        with json_path.open("r", encoding="utf-8") as handle:
            split_payload = json.load(handle)
        if split_payload.get("source_index_sha256") != file_sha256(
            args.index_csv
        ):
            raise ValueError("existing split was made from another index")
        if int(split_payload.get("seed", -1)) != config.seed:
            raise ValueError("existing split uses the wrong split seed")
        if [item.to_dict() for item in assignments] != split_payload.get(
            "assignments"
        ):
            raise ValueError("splits.csv and splits.json disagree")

    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != EXPECTED_PROTOCOL_NAME:
        raise ValueError("unexpected protocol name")
    if int(protocol.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("unexpected protocol sample count")
    if int(protocol.get("split_seed", -1)) != config.seed:
        raise ValueError("protocol split seed does not match")

    summary = summarize_assignments(assignments, config)
    result = {
        "passed": True,
        "protocol": str(args.protocol.resolve()),
        "sample_count": len(assignments),
        "split_seed": config.seed,
        "split_counts": {
            name: values["sample_count"]
            for name, values in summary["splits"].items()
        },
        "class_counts": {
            name: values["class_counts"]
            for name, values in summary["splits"].items()
        },
        "checks": summary["checks"],
        "sha256": {
            "sample_index_csv": file_sha256(args.index_csv),
            "splits_csv": file_sha256(csv_path),
            "splits_json": file_sha256(json_path),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
