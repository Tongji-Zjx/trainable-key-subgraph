"""Fit the theory-geometry standardizer from the training sidecar only."""

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
from keysubgraph.data.sv_theory_geometry import (  # noqa: E402
    fit_sv_theory_feature_standardizer,
    load_sv_theory_feature_payload,
    save_sv_theory_feature_standardizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = load_sv_theory_feature_payload(args.train_cache)
    scaler = fit_sv_theory_feature_standardizer(
        payload, file_sha256(args.train_cache)
    )
    output = save_sv_theory_feature_standardizer(
        scaler, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "train_sample_count": scaler.train_sample_count,
                "train_manifest_sha256": (
                    scaler.train_manifest_sha256
                ),
                "spectral_direction_scale_minimum": float(
                    scaler.spectral_direction_scale.min()
                ),
                "diffusion_geometry_scale_minimum": float(
                    scaler.diffusion_geometry_scale.min()
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
