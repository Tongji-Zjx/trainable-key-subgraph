"""Materialize matched high-score and random Key edge-deletion doses."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.edge_perturbation import (  # noqa: E402
    EDGE_PERTURBATION_RATIOS,
    build_edge_perturbation_exports,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_all_samples.json",
    )
    parser.add_argument("--key-export-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--perturbation-seed", type=int, default=2026)
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=EDGE_PERTURBATION_RATIOS,
        help="Increasing ratios beginning with 0; defaults to 0 .10 .25 .50.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_edge_perturbation_exports(
        PROJECT_ROOT,
        args.protocol,
        args.key_export_dir,
        args.split,
        args.output_root,
        ratios=args.ratios,
        perturbation_seed=args.perturbation_seed,
    )
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "included_sample_count": len(payload["included_sample_keys"]),
        "excluded_sample_count": len(payload["excluded_samples"]),
        "sources": payload["sources"],
        "perturbation_seed": payload["perturbation_seed"],
        "perturbation_summary": payload["perturbation_summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
