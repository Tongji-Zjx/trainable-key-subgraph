"""Fit F0 on each inner-validation split and evaluate its outer-test split."""

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
    apply_f0_fusion,
    crossfit_classification_metrics,
    fit_f0_fusion,
    read_prediction_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-output-root", type=Path, required=True)
    parser.add_argument("--short-term-crossfit-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2))
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


def _short_term_files(root, fold):
    base = root / "fold_{}".format(fold) / "author_short_term_no_coord"
    validation = sorted(base.glob("evaluation_seed*/validation_predictions.csv"))
    test = sorted(base.glob("evaluation_seed*/test_predictions.csv"))
    if len(validation) != 1 or len(test) != 1:
        raise ValueError(
            "expected one frozen short-term evaluation for fold {}".format(fold)
        )
    if validation[0].parent != test[0].parent:
        raise ValueError("short-term validation/test provenance mismatch")
    return validation[0], test[0]


def main():
    args = parse_args()
    svg_root = args.svg_output_root.resolve()
    short_root = args.short_term_crossfit_root.resolve()
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    predictions_path = output / "oof_predictions.csv"
    if (summary_path.exists() or predictions_path.exists()) and not args.overwrite:
        raise FileExistsError("F0 validation-crossfit output exists")
    if len(args.folds) < 2 or len(set(args.folds)) != len(args.folds):
        raise ValueError("F0 folds must be unique and include at least two folds")

    all_predictions = []
    fold_results = []
    source_hashes = {}
    for fold in args.folds:
        short_validation, short_test = _short_term_files(short_root, fold)
        model = (
            svg_root
            / "fold_{}".format(fold)
            / "models"
            / "{}_seed{}".format(args.candidate, args.seed)
        )
        svg_validation = model / "best_evaluation.json"
        svg_test = model / "outer_test_evaluation.json"
        required = (short_validation, short_test, svg_validation, svg_test)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("F0 source artifacts are missing: {}".format(missing))

        fitted = fit_f0_fusion(
            read_prediction_artifact(short_validation),
            read_prediction_artifact(svg_validation),
            l1_weight=args.l1_weight,
            optimization_steps=args.optimization_steps,
        )
        evaluated = apply_f0_fusion(
            fitted,
            read_prediction_artifact(short_test),
            read_prediction_artifact(svg_test),
        )
        current = []
        for row in evaluated["predictions"]:
            enriched = dict(row)
            enriched["fold"] = int(fold)
            current.append(enriched)
        all_predictions.extend(current)
        spec = dict(fitted)
        spec.pop("fit_sample_keys")
        spec.pop("fit_sites")
        fold_results.append(
            {
                "fold": int(fold),
                "validation_sample_count": len(
                    read_prediction_artifact(short_validation)
                ),
                "outer_test_sample_count": len(current),
                "fit_and_evaluation_disjoint": True,
                "fitted": spec,
                "metrics": evaluated["metrics"],
            }
        )
        for name, path in zip(
            ("short_validation", "short_test", "svg_validation", "svg_test"),
            required,
        ):
            source_hashes["fold{}_{}".format(fold, name)] = _sha256(path)

    keys = [row["sample_key"] for row in all_predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("F0 outer-test samples overlap across folds")
    all_predictions.sort(key=lambda row: (int(row["fold"]), row["sample_key"]))
    payload = {
        "artifact_type": "svg_v2_short_term_f0_validation_crossfit_summary",
        "fusion_protocol": "per_outer_fold_validation_only",
        "candidate": args.candidate,
        "seed": int(args.seed),
        "folds": list(args.folds),
        "every_sample_predicted_once": True,
        "outer_test_threshold_fitting": False,
        "fold_results": fold_results,
        "metrics": crossfit_classification_metrics(all_predictions),
        "source_sha256": source_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, payload)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerows(all_predictions)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
