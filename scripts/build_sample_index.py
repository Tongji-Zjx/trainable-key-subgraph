"""Build the portable sample index and exclusion manifest."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.sample_index import (  # noqa: E402
    IndexBuildConfig,
    build_sample_index,
    summarize_records,
    write_index_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "adhd_5_0.5",
        help="Dataset root with <site>/<label>/*.pt layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "index",
        help="Directory for CSV and JSON index artifacts.",
    )
    parser.add_argument(
        "--edge-presence-threshold",
        type=float,
        default=0.0,
        help="An edge exists when abs(A_ij) is greater than this value.",
    )
    parser.add_argument(
        "--allow-zero-coords",
        action="store_true",
        help="Record all-zero coordinate samples as included (not recommended yet).",
    )
    parser.add_argument(
        "--allow-noncontiguous-communities",
        action="store_true",
        help="Do not exclude non-negative but non-contiguous community labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = IndexBuildConfig(
        dataset_root=args.data_root,
        require_valid_coords=not args.allow_zero_coords,
        require_contiguous_communities=not args.allow_noncontiguous_communities,
        edge_presence_threshold=args.edge_presence_threshold,
    )
    records = build_sample_index(config)
    paths = write_index_artifacts(records, args.output_dir)
    summary = summarize_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("Artifacts:")
    for name, path in sorted(paths.items()):
        print("  {}: {}".format(name, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
