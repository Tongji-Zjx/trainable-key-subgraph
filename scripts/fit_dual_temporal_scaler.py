"""Fit and freeze the D3-B temporal 16-D train-only scaler."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.dual_temporal_manifest import (  # noqa: E402
    read_dual_temporal_manifest,
)
from keysubgraph.data.dual_temporal_scaler import (  # noqa: E402
    fit_dual_temporal_standardizer,
    save_dual_temporal_standardizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, records = read_dual_temporal_manifest(args.train_manifest)
    if manifest["split"] != "train":
        raise ValueError("temporal scaler must be fitted on train")
    scaler = fit_dual_temporal_standardizer(
        records, args.train_manifest
    )
    path = save_dual_temporal_standardizer(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "scaler": str(path),
                "valid_transition_count": scaler.valid_transition_count,
                "train_sample_count": scaler.train_sample_count,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
