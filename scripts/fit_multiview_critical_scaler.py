"""Fit the revised multi-view scaler from the train manifest only."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.multiview_critical import (  # noqa: E402
    fit_multiview_scaler,
    read_multiview_manifest,
    save_multiview_scaler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest, records = read_multiview_manifest(args.train_manifest, PROJECT_ROOT)
    if manifest["split"] != "train":
        raise ValueError("multi-view scaler manifest must be train")
    scaler = fit_multiview_scaler(records, file_sha256(args.train_manifest))
    path = save_multiview_scaler(scaler, args.output, overwrite=args.overwrite)
    print("multi-view scaler: {} samples={} ".format(path, len(records)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
