"""Fail-closed audits for cross-fitted splits, runs, predictions, and perturbations."""

from __future__ import absolute_import, division, print_function

from collections import defaultdict
from pathlib import Path

from keysubgraph.data.data_split import file_sha256


def audit_file_hash(path, expected_sha256):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != str(expected_sha256):
        raise ValueError("artifact is missing or its SHA-256 differs: {}".format(path))
    return {"path": str(path.resolve()), "sha256": str(expected_sha256), "valid": True}


def audit_fold_assignments(payload):
    assignments = payload.get("assignments", [])
    fold_count = int(payload.get("num_outer_folds", payload.get("fold_count", 0)))
    if fold_count < 2 or not assignments:
        raise ValueError("crossfit assignments are empty or invalid")
    outer_occurrences = defaultdict(int)
    all_samples = set()
    for fold in range(fold_count):
        current = [row for row in assignments if int(row["outer_fold"]) == fold]
        if not current:
            raise ValueError("crossfit fold is empty")
        sample_roles = defaultdict(set)
        subject_roles = defaultdict(set)
        for row in current:
            role = str(row["role"])
            if role not in ("inner_train", "inner_validation", "outer_test"):
                raise ValueError("unknown crossfit role")
            sample_key = str(row["sample_key"])
            subject_id = str(row["subject_id"])
            sample_roles[sample_key].add(role)
            subject_roles[subject_id].add(role)
            all_samples.add(sample_key)
            if role == "outer_test":
                outer_occurrences[sample_key] += 1
        if any(len(roles) != 1 for roles in sample_roles.values()):
            raise ValueError("sample crosses roles within an outer fold")
        if any(len(roles) != 1 for roles in subject_roles.values()):
            raise ValueError("subject crosses roles within an outer fold")
        for role in ("inner_train", "inner_validation", "outer_test"):
            labels = {int(row["label"]) for row in current if row["role"] == role}
            if labels != {0, 1}:
                raise ValueError("crossfit role lacks one class")
    if set(outer_occurrences) != all_samples or any(value != 1 for value in outer_occurrences.values()):
        raise ValueError("each sample must occur in outer_test exactly once")
    return {
        "valid": True, "fold_count": fold_count,
        "sample_count": len(all_samples), "outer_test_once": True,
    }


def audit_run_plan(plan):
    runs = plan.get("runs", [])
    expected = int(plan.get("expected_run_count", -1))
    if len(runs) != expected or expected != 4 * int(plan["fold_count"]) * len(plan["seeds"]):
        raise ValueError("OOF run matrix is incomplete")
    identities = set()
    checkpoints = set()
    for row in runs:
        identity = (int(row["outer_fold"]), int(row["seed"]), str(row["variant"]))
        if identity in identities or row["checkpoint"] in checkpoints:
            raise ValueError("OOF run/checkpoint is duplicated")
        identities.add(identity)
        checkpoints.add(row["checkpoint"])
        expected_source = "key" if row["variant"] in ("A", "C") else "random"
        expected_encoder = "signed" if row["variant"] in ("A", "B") else "node_only"
        if row["source"] != expected_source or row["encoder_type"] != expected_encoder:
            raise ValueError("OOF variant semantics differ from A-D")
        if row["history_mode"] != "independent_bag":
            raise ValueError("OOF temporal aggregation differs")
    return {"valid": True, "run_count": len(runs), "unique_checkpoints": len(checkpoints)}


def audit_perturbation_plan(plan):
    rows = plan.get("inferences", [])
    if plan.get("retrain") is not False or len(rows) != int(plan.get("expected_inference_count", -1)):
        raise ValueError("perturbation plan is incomplete or retrains models")
    groups = defaultdict(list)
    for row in rows:
        if row.get("retrain") is not False:
            raise ValueError("perturbation inference requests retraining")
        groups[(int(row["outer_fold"]), int(row["model_seed"]))].append(row)
    for key, members in groups.items():
        if len(members) != 13 or len({row["checkpoint"] for row in members}) != 1:
            raise ValueError("perturbation conditions do not share one A checkpoint")
        if len([row for row in members if row["mode"] == "none" and row["dose"] == 0.0]) != 1:
            raise ValueError("perturbation baseline is missing")
        for dose in (0.25, 0.50):
            targeted = [row for row in members if row["mode"] == "targeted" and row["dose"] == dose]
            random_rows = [row for row in members if row["mode"] == "random" and row["dose"] == dose]
            if len(targeted) != 1 or {row["repeat_index"] for row in random_rows} != set(range(5)):
                raise ValueError("perturbation targeted/random inventory differs")
    return {"valid": True, "model_count": len(groups), "inference_count": len(rows)}


def audit_oof_prediction_coverage(predictions, expected_sample_keys, seeds=(42, 43, 44)):
    expected_sample_keys = set(str(value) for value in expected_sample_keys)
    groups = defaultdict(set)
    duplicates = set()
    for row in predictions:
        key = (str(row["sample_key"]), int(row["model_seed"]))
        variant = str(row["variant"])
        if variant in groups[key]:
            duplicates.add(key + (variant,))
        groups[key].add(variant)
    if duplicates:
        raise ValueError("duplicate OOF sample prediction")
    expected_groups = {(sample, int(seed)) for sample in expected_sample_keys for seed in seeds}
    if set(groups) != expected_groups or any(value != {"A", "B", "C", "D"} for value in groups.values()):
        raise ValueError("OOF prediction coverage is incomplete")
    return {
        "valid": True, "sample_count": len(expected_sample_keys),
        "prediction_count": len(predictions), "outer_test_once_per_variant_seed": True,
    }
