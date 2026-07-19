"""Freeze the OOF test-time edge-perturbation inference plan."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.perturbation_plan import (  # noqa: E402
    build_perturbation_inference_plan,
    write_perturbation_inference_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-plan", type=Path,
        default=PROJECT_ROOT / "configs/crossfit/oof_run_plan.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "configs/crossfit/perturbation_inference_plan.json"
    )
    parser.add_argument("--base-seed", type=int, default=2026)
    args = parser.parse_args()
    with args.run_plan.resolve().open("r", encoding="utf-8") as handle:
        run_plan = json.load(handle)
    payload = build_perturbation_inference_plan(run_plan, args.base_seed)
    write_perturbation_inference_plan(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "inference_count": len(payload["inferences"]),
        "retrain": payload["retrain"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
