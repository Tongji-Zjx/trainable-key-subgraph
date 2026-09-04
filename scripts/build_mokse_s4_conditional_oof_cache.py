#!/usr/bin/env python3
"""Build nested TGE manifests for leakage-safe S4 fusion development OOF."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.conditional_oof import build_conditional_oof_cache  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--target-validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_conditional_oof_cache(
        args.train_manifest,
        args.target_validation_manifest,
        args.output_dir,
        validation_fraction=args.inner_validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

