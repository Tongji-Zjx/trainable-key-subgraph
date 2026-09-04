#!/usr/bin/env python3
"""Export validation/test logits from a frozen four-rotation MoKSE XGB head.

The historic MoKSE XGB search persisted the four boosters and fixed-test
predictions, but not the matching validation predictions.  Safe late fusion
needs both splits.  This utility replays the already-frozen boosters without
fitting or selecting anything and writes the common prediction CSV schema
consumed by ``fit_mokse_background_safe_fusion.py``.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results", type=Path, required=True)
    parser.add_argument(
        "--fold-dir", action="append", type=Path, required=True,
        help="four historic MoKSE fold directories containing fusion/*_features.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verification-tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_features(fold_dir, split, input_mode):
    path = Path(fold_dir) / "fusion" / (split + "_features.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    base_name = "evolution_logits" if input_mode == "evolution" else "fused_logits"
    required = (
        "sample_keys", "sites", "labels", "evolution_representations", base_name,
    )
    if any(name not in payload for name in required):
        raise ValueError("incomplete frozen feature artifact: {}".format(path))
    base = payload[base_name].astype(np.float32)
    features = np.concatenate(
        (
            payload["evolution_representations"].astype(np.float32),
            base[:, None],
        ),
        axis=1,
    )
    return {
        "path": path,
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "features": features,
        "base": base,
    }


def write_predictions(path, split, logits):
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("sample_key", "site", "label", "final_logit", "probability", "prediction")
        )
        for key, site, label, logit, score in zip(
            split["sample_keys"], split["sites"], split["labels"], logits, probability
        ):
            writer.writerow(
                (key, site, int(label), float(logit), float(score), int(score >= 0.5))
            )


def compare_persisted_test(path, sample_keys, logits, tolerance):
    if not path.is_file():
        return {"available": False, "passed": None, "maximum_absolute_difference": None}
    stored = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stored[str(row["sample_key"])] = float(row["final_logit"])
    if set(stored) != set(sample_keys.tolist()):
        raise ValueError("persisted fixed-test prediction cohort differs: {}".format(path))
    difference = max(
        abs(float(logit) - stored[str(key)])
        for key, logit in zip(sample_keys, logits)
    )
    if difference > tolerance:
        raise ValueError(
            "frozen XGB replay differs from persisted test predictions: {:.3e}".format(
                difference
            )
        )
    return {
        "available": True,
        "passed": True,
        "maximum_absolute_difference": float(difference),
    }


def main():
    args = parse_args()
    if len(args.fold_dir) != 4:
        raise ValueError("exactly four fold directories are required")
    search_path = args.search_results.resolve()
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if search.get("artifact_type") != "mokse_background_shared_xgb_test_guided_search_v1":
        raise ValueError("unsupported frozen XGB search artifact")
    input_mode = str(search["input_mode"])
    residual_alpha = float(search["best"]["candidate"]["residual_alpha"])
    booster_root = search_path.parent / "boosters"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import xgboost as xgb

    fold_reports = []
    for fold_index, fold_dir in enumerate(args.fold_dir):
        booster_path = booster_root / "fold_{}_booster.json".format(fold_index)
        if not booster_path.is_file():
            raise FileNotFoundError(booster_path)
        booster = xgb.Booster()
        booster.load_model(str(booster_path))
        output = args.output_dir / "fold_{}".format(fold_index)
        output.mkdir(parents=True, exist_ok=True)
        split_reports = {}
        for split_name in ("validation", "test"):
            split = load_features(fold_dir.resolve(), split_name, input_mode)
            matrix = xgb.DMatrix(split["features"], base_margin=split["base"])
            total = booster.predict(matrix, output_margin=True)
            residual = total - split["base"]
            final = split["base"] + residual_alpha * residual
            if not np.isfinite(final).all():
                raise ValueError("frozen XGB replay produced non-finite logits")
            prediction_path = output / (split_name + "_predictions.csv")
            write_predictions(prediction_path, split, final)
            split_reports[split_name] = {
                "sample_count": int(final.size),
                "feature_source": str(split["path"].resolve()),
                "feature_source_sha256": file_sha256(split["path"]),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": file_sha256(prediction_path),
            }
            if split_name == "test":
                persisted = search_path.parent / (
                    "fold_{}_test_predictions.csv".format(fold_index)
                )
                split_reports[split_name]["persisted_test_replay"] = (
                    compare_persisted_test(
                        persisted,
                        split["sample_keys"],
                        final,
                        args.verification_tolerance,
                    )
                )
        fold_reports.append(
            {
                "fold": fold_index,
                "booster": str(booster_path.resolve()),
                "booster_sha256": file_sha256(booster_path),
                "splits": split_reports,
            }
        )
    manifest = {
        "artifact_type": "mokse_frozen_subgraph_final_prediction_export_v1",
        "fit_performed": False,
        "selection_performed": False,
        "input_mode": input_mode,
        "residual_alpha": residual_alpha,
        "search_results": str(search_path),
        "search_results_sha256": file_sha256(search_path),
        "historic_search_used_test_for_parameter_selection": bool(
            search.get("test_used_for_parameter_selection", False)
        ),
        "folds": fold_reports,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
