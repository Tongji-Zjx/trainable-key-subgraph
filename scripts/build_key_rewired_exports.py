"""Materialize matched Key and signed endpoint-rewired Key exports."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.key_rewiring import build_key_rewired_exports  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_all_samples.json",
    )
    parser.add_argument("--key-export-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rewiring-seed", type=int, default=2026)
    parser.add_argument("--max-attempts", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_key_rewired_exports(
        PROJECT_ROOT,
        args.protocol,
        args.key_export_dir,
        args.split,
        args.output_root,
        rewiring_seed=args.rewiring_seed,
        max_attempts=args.max_attempts,
    )
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "included_sample_count": len(payload["included_sample_keys"]),
        "excluded_sample_count": len(payload["excluded_samples"]),
        "sources": payload["sources"],
        "rewiring_seed": payload["rewiring_seed"],
        "rewiring_summary": payload["rewiring_summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
