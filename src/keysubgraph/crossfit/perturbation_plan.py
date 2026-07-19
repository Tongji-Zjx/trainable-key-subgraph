"""Test-time-only confirmatory edge-perturbation inference plan."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path


PERTURBATION_PLAN_SCHEMA_VERSION = 1
CONFIRMATORY_DOSES = (0.0, 0.25, 0.50)
RANDOM_REPEATS = 5


def build_perturbation_inference_plan(oof_run_plan, base_seed=2026):
    """Reuse every A checkpoint for nested targeted/random test-time deletions."""

    if int(oof_run_plan.get("schema_version", -1)) != 1:
        raise ValueError("unsupported OOF run plan")
    if base_seed < 0:
        raise ValueError("perturbation seed must be non-negative")
    model_runs = [row for row in oof_run_plan["runs"] if row["variant"] == "A"]
    expected = int(oof_run_plan["fold_count"]) * len(oof_run_plan["seeds"])
    if len(model_runs) != expected:
        raise ValueError("OOF plan does not contain one A model per fold/seed")
    records = []
    for model_run in model_runs:
        conditions = [("none", 0.0, None, base_seed)]
        for dose in CONFIRMATORY_DOSES[1:]:
            conditions.append(("targeted", dose, None, base_seed))
            for repeat in range(RANDOM_REPEATS):
                conditions.append(("random", dose, repeat, base_seed + repeat))
        for mode, dose, repeat, perturbation_seed in conditions:
            repeat_code = "single" if repeat is None else "repeat{}".format(repeat)
            condition_id = "{}_dose{:03d}_{}".format(
                mode, int(round(dose * 100.0)), repeat_code
            )
            inference_id = "{}_{}".format(model_run["run_id"], condition_id)
            records.append({
                "inference_id": inference_id,
                "outer_fold": model_run["outer_fold"],
                "model_seed": model_run["seed"],
                "checkpoint": model_run["checkpoint"],
                "test_manifest": model_run["test_manifest"],
                "mode": mode,
                "dose": dose,
                "repeat_index": repeat,
                "perturbation_seed": perturbation_seed,
                "retrain": False,
                "predictions": (
                    "outputs/crossfit/perturbation/{}/predictions.json".format(inference_id)
                ),
            })
    identifiers = [row["inference_id"] for row in records]
    predictions = [row["predictions"] for row in records]
    if len(set(identifiers)) != len(records) or len(set(predictions)) != len(records):
        raise AssertionError("perturbation inference paths collide")
    return {
        "schema_version": PERTURBATION_PLAN_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "confirmatory_test_time_edge_dose_response",
        "doses": list(CONFIRMATORY_DOSES),
        "random_repeats": RANDOM_REPEATS,
        "base_seed": int(base_seed),
        "retrain": False,
        "expected_inference_count": expected * 13,
        "inferences": records,
    }


def write_perturbation_inference_plan(payload, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path
