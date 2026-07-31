"""Evaluate one frozen Full-Soft-Hard selector on an outer fold."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.selector_transfer_oof import (  # noqa: E402
    evaluate_selector_transfer_outer_fold,
    write_selector_transfer_outer_fold,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = evaluate_selector_transfer_outer_fold(
        args.train_manifest,
        args.validation_manifest,
        args.test_manifest,
        seed=args.seed,
    )
    path = write_selector_transfer_outer_fold(
        payload,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "output": str(path),
        "validation": payload["evaluations"]["validation"]["metrics"],
        "outer_test": payload["evaluations"]["test"]["metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
