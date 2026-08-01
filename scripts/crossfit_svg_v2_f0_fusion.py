"""Run the explicitly non-nested outer-OOF F0 fusion diagnostic."""

from __future__ import absolute_import, print_function

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
    crossfit_oof_f0_fusion,
    read_crossfit_prediction_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-term-oof", type=Path, required=True)
    parser.add_argument("--svg-oof", type=Path, required=True)
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
    prediction_path = output / "oof_predictions.csv"
    if (result_path.exists() or prediction_path.exists()) and not args.overwrite:
        raise FileExistsError("outer-OOF F0 diagnostic output exists")
    short_term = read_crossfit_prediction_csv(args.short_term_oof)
    svg = read_crossfit_prediction_csv(args.svg_oof)
    result = crossfit_oof_f0_fusion(
        short_term,
        svg,
        l1_weight=args.l1_weight,
        optimization_steps=args.optimization_steps,
    )
    predictions = result.pop("predictions")
    payload = {
        "artifact_type": "svg_v2_short_term_f0_outer_oof_diagnostic",
        "fusion_protocol": "outer_oof_crossfit_surrogate",
        "strict_nested_stacking": False,
        "formal_f0_estimate": False,
        "every_sample_predicted_once": True,
        "held_fold_threshold_fitting": False,
        "source_sha256": {
            "short_term_oof": _sha256(args.short_term_oof),
            "svg_oof": _sha256(args.svg_oof),
        },
        **result,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, payload)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "fold",
            "sample_key",
            "site",
            "label",
            "positive_probability",
            "threshold",
            "predicted_label",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
