"""Analyze one completed A-D OOF fold and its 13 perturbation conditions."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_analysis import analyze_fold_predictions  # noqa: E402


def _read_predictions(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prediction artifact has no sample rows: {}".format(path))
    return rows


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--downstream-root", type=Path, default=PROJECT_ROOT / "outputs/crossfit/downstream")
    parser.add_argument("--perturbation-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_rows = []
    for variant in ("A", "B", "C", "D"):
        path = args.downstream_root / "fold{}_{}_seed{}".format(args.fold, variant, args.model_seed) / "outer_test_predictions.json"
        for row in _read_predictions(path):
            current = dict(row)
            current.update({"outer_fold": args.fold, "model_seed": args.model_seed, "variant": variant})
            model_rows.append(current)
    summary_path = args.perturbation_root / "perturbation_run_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        perturbation_summary = json.load(handle)
    perturbation_rows = []
    for condition in perturbation_summary["conditions"]:
        for row in _read_predictions(condition["predictions"]):
            current = dict(row)
            current.update({
                "outer_fold": args.fold, "model_seed": args.model_seed,
                "mode": condition["mode"], "dose": condition["dose"],
                "repeat_index": condition["repeat_index"],
            })
            perturbation_rows.append(current)
    result = analyze_fold_predictions(
        model_rows, perturbation_rows, args.bootstrap_repeats, args.bootstrap_seed
    )
    result.update({
        "schema_version": 1, "outer_fold": args.fold,
        "model_seed": args.model_seed,
        "bootstrap_repeats": args.bootstrap_repeats,
    })
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "model_results": result["model_results"],
        "dose_results": result["dose_results"],
        "dose_slope_result": result["dose_slope_result"],
        "coverage_audit": result["coverage_audit"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
