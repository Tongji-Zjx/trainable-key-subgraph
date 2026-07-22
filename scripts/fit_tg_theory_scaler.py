"""Fit the 34-D TG-SGW standardizer from eligible training artifacts only."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.features import TGTheoryFeatureStandardizer  # noqa: E402
from keysubgraph.theory import load_tg_sgw_feature_artifact  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standard-deviation-floor", type=float, default=1.0e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.feature_dir.resolve().glob("*.pt"))
    artifacts = [load_tg_sgw_feature_artifact(path) for path in paths]
    eligible = [
        item for item in artifacts
        if item.split == "train" and item.eligible_for_stage_c
    ]
    if not eligible:
        raise ValueError("no eligible training SGW feature artifacts found")
    if any(item.split != "train" for item in eligible):
        raise RuntimeError("non-training feature reached theory scaler fitting")
    protocol_hashes = {item.data_protocol_sha256 for item in eligible}
    teacher_hashes = {item.teacher_checkpoint_sha256 for item in eligible}
    if len(protocol_hashes) != 1 or len(teacher_hashes) != 1:
        raise ValueError("training SGW features mix incompatible artifacts")
    scaler = TGTheoryFeatureStandardizer.fit(
        [item.features.h_classification for item in eligible],
        fit_split="train",
        standard_deviation_floor=args.standard_deviation_floor,
        data_protocol_sha256=next(iter(protocol_hashes)),
        teacher_checkpoint_sha256=next(iter(teacher_hashes)),
    )
    scaler.save(args.output, overwrite=args.overwrite)
    print(json.dumps({
        "fit_split": scaler.fit_split,
        "training_sample_count": len(eligible),
        "feature_dim": len(scaler.mean),
        "minimum_scale": min(scaler.scale),
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
