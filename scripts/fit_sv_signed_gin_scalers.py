"""Fit the frozen train-only node/static/variation SV standardizers."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_signed_gin_manifest import (  # noqa: E402
    read_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (  # noqa: E402
    fit_sv_signed_gin_standardizers,
    save_sv_signed_gin_standardizers,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, records = read_sv_signed_gin_manifest(
        args.train_manifest
    )
    if manifest["split"] != "train":
        raise ValueError("SV scalers must be fitted on train")
    scaler = fit_sv_signed_gin_standardizers(
        records, file_sha256(args.train_manifest)
    )
    path = save_sv_signed_gin_standardizers(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "scaler": str(path),
                "train_sample_count": scaler.train_sample_count,
                "train_node_count": scaler.train_node_count,
                "selection_mode": scaler.selection_mode,
                "selection_seed": scaler.selection_seed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
