"""Build one fold-wide, partition-aware Key/Random control manifest."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.crossfit_controls import build_crossfit_key_random_exports  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--key-export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args()
    payload = build_crossfit_key_random_exports(
        PROJECT_ROOT, args.protocol, args.key_export_root, args.output_root,
        args.random_seed, args.repeat_index,
    )
    print(json.dumps({
        "manifest": str((args.output_root / "key_random_control_manifest.json").resolve()),
        "partition_counts": {
            split: len(item["included_sample_keys"])
            for split, item in payload["partition_inventories"].items()
        },
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
