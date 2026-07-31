"""Precompute fixed signed-direction and multi-scale diffusion sidecars."""

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
from keysubgraph.data.sv_theory_geometry import (  # noqa: E402
    build_sv_theory_feature_payload,
    save_sv_theory_feature_payload,
)
from keysubgraph.features.sv_theory_geometry import (  # noqa: E402
    SVTheoryGeometryExtractor,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--laplacian-eta", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest, records = read_sv_signed_gin_manifest(args.manifest)
    extractor = SVTheoryGeometryExtractor(
        laplacian_eta=args.laplacian_eta
    )
    device = args.device

    def report_progress(index, total, sample_key):
        if index % 25 == 0 or index == total:
            print(
                "processed {}/{} {}".format(
                    index, total, sample_key
                ),
                flush=True,
            )

    payload = build_sv_theory_feature_payload(
        records,
        file_sha256(args.manifest),
        extractor,
        device=device,
        progress_callback=report_progress,
    )
    output = save_sv_theory_feature_payload(
        payload, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "split": manifest["split"],
                "sample_count": payload["sample_count"],
                "spectral_direction_dim": payload["dimensions"][
                    "spectral_direction"
                ],
                "diffusion_geometry_dim": payload["dimensions"][
                    "diffusion_geometry"
                ],
                "diffusion_time_scales": payload["configuration"][
                    "diffusion_time_scales"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
