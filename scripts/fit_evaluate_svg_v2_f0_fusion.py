"""Fit frozen nonnegative F0 fusion and evaluate a disjoint partition."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.svg_v2_f0_fusion import (  # noqa: E402
    apply_f0_fusion,
    fit_f0_fusion,
    read_prediction_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-short-term", type=Path, required=True)
    parser.add_argument("--fit-svg", type=Path, required=True)
    parser.add_argument("--evaluate-short-term", type=Path, required=True)
    parser.add_argument("--evaluate-svg", type=Path, required=True)
    parser.add_argument(
        "--fusion-protocol",
        choices=("validation_only", "strict_crossfit"),
        required=True,
    )
    parser.add_argument("--l1-weight", type=float, default=1.0e-3)
    parser.add_argument("--optimization-steps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    result_path = output / "evaluation.json"
    prediction_path = output / "predictions.csv"
    if (result_path.exists() or prediction_path.exists()) and not args.overwrite:
        raise FileExistsError("F0 fusion output exists")
    fit_short = read_prediction_csv(args.fit_short_term)
    fit_svg = read_prediction_csv(args.fit_svg)
    eval_short = read_prediction_csv(args.evaluate_short_term)
    eval_svg = read_prediction_csv(args.evaluate_svg)
    fit_keys = set(fit_short)
    evaluate_keys = set(eval_short)
    if fit_keys.intersection(evaluate_keys):
        raise ValueError("F0 fit and evaluation samples overlap")
    fitted = fit_f0_fusion(
        fit_short,
        fit_svg,
        l1_weight=args.l1_weight,
        optimization_steps=args.optimization_steps,
    )
    evaluated = apply_f0_fusion(fitted, eval_short, eval_svg)
    # Sample identities are retained in predictions, not duplicated in model spec.
    fitted.pop("fit_sample_keys")
    fitted.pop("fit_sites")
    result = {
        "artifact_type": "svg_v2_short_term_f0_fusion_evaluation",
        "fusion_protocol": args.fusion_protocol,
        "fit_and_evaluation_disjoint": True,
        "test_threshold_fitting": False,
        "fitted": fitted,
        "metrics": evaluated["metrics"],
        "source_sha256": {
            "fit_short_term": _sha256(args.fit_short_term),
            "fit_svg": _sha256(args.fit_svg),
            "evaluate_short_term": _sha256(args.evaluate_short_term),
            "evaluate_svg": _sha256(args.evaluate_svg),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, result)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "sample_key",
            "site",
            "label",
            "positive_probability",
            "threshold",
            "predicted_label",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluated["predictions"])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
