"""Freeze the A-D cross-fitted downstream run plan."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.model_matrix import (  # noqa: E402
    build_oof_run_plan,
    write_oof_run_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-assignments", type=Path,
        default=PROJECT_ROOT / "configs/crossfit/fold_assignments.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "configs/crossfit/oof_run_plan.json"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    args = parser.parse_args()
    payload = build_oof_run_plan(args.fold_assignments, seeds=tuple(args.seeds))
    write_oof_run_plan(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "run_count": len(payload["runs"]),
        "fold_count": payload["fold_count"],
        "seeds": payload["seeds"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
