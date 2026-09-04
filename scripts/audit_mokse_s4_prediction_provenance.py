#!/usr/bin/env python3
"""Audit frozen S4 prediction cohorts and write strict role provenance."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-role", choices=("development_oof", "fixed_test"), required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument(
        "--fit-manifest", action="append", type=Path, default=[],
        help="all disjoint manifests used for fitting or checkpoint selection",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_keys(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest does not contain a records list: {}".format(path))
    keys = [str(row.get("sample_key", "")) for row in records]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("manifest sample keys are empty or duplicated: {}".format(path))
    return set(keys)


def prediction_keys(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [str(row.get("sample_key", "")) for row in rows]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("prediction sample keys are empty or duplicated")
    if any("final_logit" not in row for row in rows):
        raise ValueError("predictions do not contain final_logit")
    return set(keys)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    args = parse_args()
    predictions = args.predictions.resolve()
    prediction_manifest = args.prediction_manifest.resolve()
    observed = prediction_keys(predictions)
    expected = manifest_keys(prediction_manifest)
    if observed != expected:
        raise ValueError("prediction CSV and prediction manifest cohorts differ")

    fit_manifests = [path.resolve() for path in args.fit_manifest]
    fit_sets = [manifest_keys(path) for path in fit_manifests]
    all_disjoint = True
    if args.prediction_role == "development_oof":
        if len(fit_sets) < 2:
            raise ValueError(
                "development OOF audit requires both fitting and checkpoint-selection manifests"
            )
        for index, current in enumerate(fit_sets):
            if observed.intersection(current):
                raise ValueError("development OOF cohort overlaps a fit manifest")
            for previous in fit_sets[:index]:
                if current.intersection(previous):
                    raise ValueError("fit/checkpoint-selection manifests overlap")

    output = (
        args.output.resolve()
        if args.output is not None
        else predictions.with_suffix(predictions.suffix + ".json")
    )
    payload = {
        "artifact_type": "mokse_s4_prediction_provenance_v1",
        "prediction_role": args.prediction_role,
        "predictions": str(predictions),
        "predictions_sha256": file_sha256(predictions),
        "prediction_manifest": str(prediction_manifest),
        "prediction_manifest_sha256": file_sha256(prediction_manifest),
        "sample_count": len(observed),
        "fit_manifests": [str(path) for path in fit_manifests],
        "fit_manifest_sha256": [file_sha256(path) for path in fit_manifests],
        "oof_disjointness_audit": {
            "all_disjoint": bool(all_disjoint and args.prediction_role == "development_oof"),
            "fit_manifest_count": len(fit_manifests),
        },
        "test_used_for_fit": False,
    }
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
