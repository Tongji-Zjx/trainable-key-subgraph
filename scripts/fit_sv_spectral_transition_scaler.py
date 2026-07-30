"""Fit the neural evolution transition scaler from one train manifest."""

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
from keysubgraph.data.sv_spectral_evolution import (  # noqa: E402
    fit_sv_spectral_transition_standardizer,
    save_sv_spectral_transition_standardizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--laplacian-eta", type=float, default=1.0e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, records = read_sv_signed_gin_manifest(args.train_manifest)
    if manifest["split"] != "train":
        raise ValueError("transition scaler requires a train manifest")
    scaler = fit_sv_spectral_transition_standardizer(
        records,
        file_sha256(args.train_manifest),
        laplacian_eta=args.laplacian_eta,
    )
    save_sv_spectral_transition_standardizer(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "train_sample_count": scaler.train_sample_count,
                "train_transition_count": scaler.train_transition_count,
                "laplacian_eta": scaler.laplacian_eta,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
