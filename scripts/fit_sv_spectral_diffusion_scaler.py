"""Fit the SVG-v2 HKS and signed-delta scaler from inner-train only."""

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
from keysubgraph.data.sv_spectral_diffusion import (  # noqa: E402
    fit_sv_spectral_diffusion_standardizer,
    read_sv_spectral_diffusion_manifest,
    save_sv_spectral_diffusion_standardizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, records = read_sv_spectral_diffusion_manifest(
        args.train_manifest
    )
    scaler = fit_sv_spectral_diffusion_standardizer(
        manifest,
        records,
        file_sha256(args.train_manifest),
    )
    output = save_sv_spectral_diffusion_standardizer(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "train_sample_count": scaler.train_sample_count,
                "hks_scale_minimum": float(scaler.hks_scale.min()),
                "delta_scale_minimum": float(scaler.delta_scale.min()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
