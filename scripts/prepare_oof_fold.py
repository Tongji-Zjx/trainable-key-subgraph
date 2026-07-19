"""Materialize one cross-fitting fold as a frozen train/validation/test protocol."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_protocol import prepare_fold_protocol  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-assignments", type=Path, default=PROJECT_ROOT / "configs/crossfit/fold_assignments.json")
    parser.add_argument("--source-protocol", type=Path, default=PROJECT_ROOT / "configs/data_protocol_all_samples.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/crossfit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_fold_protocol(
        PROJECT_ROOT, args.fold_assignments, args.source_protocol,
        args.fold, args.output_root, args.overwrite,
    )
    print(json.dumps({
        "output_dir": str(result["output_dir"]), "protocol": str(result["protocol"]),
        "summary": result["summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
