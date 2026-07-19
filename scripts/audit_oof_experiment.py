"""Run local preflight audits for frozen cross-fitting experiment artifacts."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.audit import (  # noqa: E402
    audit_fold_assignments, audit_perturbation_plan, audit_run_plan,
)


def _load(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=PROJECT_ROOT / "configs/crossfit/fold_assignments.json")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "configs/crossfit/oof_run_plan.json")
    parser.add_argument("--perturbations", type=Path, default=PROJECT_ROOT / "configs/crossfit/perturbation_inference_plan.json")
    args = parser.parse_args()
    result = {
        "folds": audit_fold_assignments(_load(args.folds)),
        "runs": audit_run_plan(_load(args.runs)),
        "perturbations": audit_perturbation_plan(_load(args.perturbations)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
