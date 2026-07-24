"""Fit the Dual-STSE exact-SGW standardizer from train records only."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (  # noqa: E402
    fit_dual_sgw_standardizer,
    save_dual_sgw_standardizer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload, records, _ = read_dual_sgw_manifest(args.train_manifest)
    if payload["split"] != "train":
        raise ValueError("dual scaler manifest must be train")
    scaler = fit_dual_sgw_standardizer(records)
    path = save_dual_sgw_standardizer(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "sample_count": scaler.sample_count,
                "protocol_sha256": scaler.protocol_sha256,
                "selector_checkpoint_sha256": (
                    scaler.selector_checkpoint_sha256
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
