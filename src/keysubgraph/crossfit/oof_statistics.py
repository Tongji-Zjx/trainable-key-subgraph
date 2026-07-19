"""Paired subject-level statistics for OOF model and perturbation predictions."""

from __future__ import absolute_import, division, print_function

import math
import random
from collections import defaultdict


VARIANTS = ("A", "B", "C", "D")


def binary_log_loss(label, probability):
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    label = int(label)
    if label not in (0, 1):
        raise ValueError("binary label must be zero or one")
    return -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))


def _identity(row):
    return (int(row["outer_fold"]), int(row["model_seed"]), str(row["sample_key"]))


def compute_model_contrasts(predictions):
    """Strictly pair A-D OOF predictions and calculate DSC, SEG, and TPA."""

    groups = defaultdict(dict)
    metadata = {}
    for row in predictions:
        variant = str(row["variant"])
        if variant not in VARIANTS:
            raise ValueError("unknown OOF model variant")
        key = _identity(row)
        if variant in groups[key]:
            raise ValueError("duplicate OOF variant prediction")
        groups[key][variant] = row
        current = (str(row["subject_id"]), str(row.get("session_id", "")), int(row["label"]))
        if key in metadata and metadata[key] != current:
            raise ValueError("paired predictions have inconsistent metadata")
        metadata[key] = current
    results = []
    for key in sorted(groups):
        if set(groups[key]) != set(VARIANTS):
            raise ValueError("incomplete A-D prediction pairing")
        losses = {
            variant: binary_log_loss(
                groups[key][variant]["label"],
                groups[key][variant]["class_1_probability"],
            )
            for variant in VARIANTS
        }
        subject_id, session_id, label = metadata[key]
        results.append({
            "outer_fold": key[0], "model_seed": key[1], "sample_key": key[2],
            "subject_id": subject_id, "session_id": session_id, "label": label,
            "dsc": losses["B"] - losses["A"],
            "seg": losses["C"] - losses["A"],
            "tpa": (losses["B"] - losses["A"]) - (losses["D"] - losses["C"]),
        })
    if not results:
        raise ValueError("no OOF predictions")
    return results


def compute_perturbation_contrasts(predictions, doses=(0.25, 0.50), random_repeats=5):
    """Calculate targeted-minus-random loss damage after repeat averaging."""

    groups = defaultdict(list)
    metadata = {}
    for row in predictions:
        key = _identity(row) + (float(row["dose"]), str(row["mode"]))
        groups[key].append(row)
        sample_key = key[:3]
        current = (str(row["subject_id"]), str(row.get("session_id", "")), int(row["label"]))
        if sample_key in metadata and metadata[sample_key] != current:
            raise ValueError("perturbation metadata differs")
        metadata[sample_key] = current
    results = []
    sample_keys = sorted(metadata)
    for sample_key in sample_keys:
        for dose in doses:
            targeted = groups.get(sample_key + (float(dose), "targeted"), [])
            random_rows = groups.get(sample_key + (float(dose), "random"), [])
            baseline = groups.get(sample_key + (0.0, "none"), [])
            if len(targeted) != 1 or len(baseline) != 1 or len(random_rows) != random_repeats:
                raise ValueError("incomplete perturbation dose/repeat pairing")
            repeats = {int(row["repeat_index"]) for row in random_rows}
            if repeats != set(range(random_repeats)):
                raise ValueError("random repeat indices are incomplete")
            label = metadata[sample_key][2]
            baseline_loss = binary_log_loss(label, baseline[0]["class_1_probability"])
            targeted_loss = binary_log_loss(label, targeted[0]["class_1_probability"])
            random_loss = sum(
                binary_log_loss(label, row["class_1_probability"])
                for row in random_rows
            ) / random_repeats
            results.append({
                "outer_fold": sample_key[0], "model_seed": sample_key[1],
                "sample_key": sample_key[2], "subject_id": metadata[sample_key][0],
                "session_id": metadata[sample_key][1], "label": label,
                "dose": float(dose),
                "targeted_damage": targeted_loss - baseline_loss,
                "random_damage": random_loss - baseline_loss,
                "dose_contrast": targeted_loss - random_loss,
            })
    if not results:
        raise ValueError("no perturbation predictions")
    return results


def aggregate_subjects(rows, value_fields):
    """Average sessions first, then model seeds, leaving subjects as analysis units."""

    sample_groups = defaultdict(list)
    for row in rows:
        dose = row.get("dose")
        key = (row["outer_fold"], row["model_seed"], row["subject_id"], dose)
        sample_groups[key].append(row)
    seed_rows = []
    for key, members in sorted(sample_groups.items()):
        seed_rows.append({
            "outer_fold": key[0], "model_seed": key[1], "subject_id": key[2],
            "dose": key[3], "sample_count": len(members),
            **{field: sum(float(row[field]) for row in members) / len(members) for field in value_fields},
        })
    subject_groups = defaultdict(list)
    for row in seed_rows:
        subject_groups[(row["outer_fold"], row["subject_id"], row["dose"])].append(row)
    subject_rows = []
    for key, members in sorted(subject_groups.items()):
        subject_rows.append({
            "outer_fold": key[0], "subject_id": key[1], "dose": key[2],
            "seed_count": len(members),
            **{field: sum(float(row[field]) for row in members) / len(members) for field in value_fields},
        })
    return subject_rows


def dose_slope(subject_dose_rows, value_field="dose_contrast"):
    """Calculate the prespecified through-origin slope for each subject."""

    groups = defaultdict(list)
    for row in subject_dose_rows:
        groups[(row["outer_fold"], row["subject_id"])].append(row)
    output = []
    for key, members in sorted(groups.items()):
        doses = {float(row["dose"]) for row in members}
        if doses != {0.25, 0.50}:
            raise ValueError("subject dose inventory is incomplete")
        numerator = sum(float(row["dose"]) * float(row[value_field]) for row in members)
        denominator = sum(float(row["dose"]) ** 2 for row in members)
        output.append({
            "outer_fold": key[0], "subject_id": key[1],
            "dose_slope": numerator / denominator,
        })
    return output


def bootstrap_subject_mean(subject_rows, value_field, repeats=5000, seed=42):
    """Deterministic percentile bootstrap with subjects as the only resampling unit."""

    values = [float(row[value_field]) for row in subject_rows]
    if not values or repeats < 1:
        raise ValueError("bootstrap requires subjects and positive repeats")
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        estimates.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    estimates.sort()
    lower = estimates[int(math.floor(0.025 * (repeats - 1)))]
    upper = estimates[int(math.ceil(0.975 * (repeats - 1)))]
    non_positive = sum(value <= 0.0 for value in estimates) / repeats
    non_negative = sum(value >= 0.0 for value in estimates) / repeats
    return {
        "subject_count": len(values), "mean": sum(values) / len(values),
        "ci95": [lower, upper], "bootstrap_repeats": repeats,
        "two_sided_p": min(1.0, 2.0 * min(non_positive, non_negative)),
        "seed": int(seed),
    }
