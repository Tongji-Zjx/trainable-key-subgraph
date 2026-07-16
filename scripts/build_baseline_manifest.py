"""Validate hard-subgraph exports and freeze a baseline input manifest."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_manifest import build_baseline_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol.json"
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baseline_manifest"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--evidence-level",
        choices=("exploratory_in_sample", "confirmatory_cross_fitted"),
        default="exploratory_in_sample",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_baseline_manifest(
        project_root=PROJECT_ROOT,
        protocol_path=args.protocol,
        export_dir=args.export_dir,
        split=args.split,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        evidence_level=args.evidence_level,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "sample_count": payload["sample_count"],
                "timepoint_count": payload["timepoint_count"],
                "subgraph_count": payload["subgraph_count"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "data_protocol_sha256": payload["data_protocol_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
