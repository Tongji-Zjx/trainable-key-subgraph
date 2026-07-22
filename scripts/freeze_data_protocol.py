"""Freeze the validated data artifacts used by all later experiments."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "adhd_5_0.5")
    parser.add_argument("--sample-index", type=Path, default=PROJECT_ROOT / "outputs" / "index" / "sample_index.csv")
    parser.add_argument("--splits-csv", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "splits.csv")
    parser.add_argument("--splits-json", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "splits.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol.json")
    parser.add_argument("--edge-presence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--protocol-name",
        choices=("strict_theory", "all_samples_exploratory"),
        default="strict_theory",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        payload = validate_data_protocol(args.output, PROJECT_ROOT)
        print("Existing data protocol found; hashes are valid and it will be reused.")
    else:
        payload = freeze_data_protocol(
            project_root=PROJECT_ROOT,
            dataset_root=args.dataset_root,
            sample_index_csv=args.sample_index,
            splits_csv=args.splits_csv,
            splits_json=args.splits_json,
            output_path=args.output,
            edge_presence_threshold=args.edge_presence_threshold,
            protocol_name=args.protocol_name,
            overwrite=args.overwrite,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
