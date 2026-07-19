"""Fold-level OOF contrasts, subject aggregation, bootstrap, and coverage audit."""

from __future__ import absolute_import, print_function

from keysubgraph.crossfit.audit import audit_oof_prediction_coverage
from keysubgraph.crossfit.oof_statistics import (
    aggregate_subjects, bootstrap_subject_mean, compute_model_contrasts,
    compute_perturbation_contrasts, dose_slope,
)


def analyze_fold_predictions(model_predictions, perturbation_predictions, bootstrap_repeats=500, seed=42):
    model_contrasts = compute_model_contrasts(model_predictions)
    model_subjects = aggregate_subjects(model_contrasts, ("dsc", "seg", "tpa"))
    model_results = {
        field: bootstrap_subject_mean(model_subjects, field, bootstrap_repeats, seed)
        for field in ("dsc", "seg", "tpa")
    }
    perturbation_contrasts = compute_perturbation_contrasts(perturbation_predictions)
    dose_subjects = aggregate_subjects(
        perturbation_contrasts,
        ("targeted_damage", "random_damage", "dose_contrast"),
    )
    dose_results = {}
    for dose in (0.25, 0.50):
        rows = [row for row in dose_subjects if float(row["dose"]) == dose]
        dose_results[str(dose)] = bootstrap_subject_mean(
            rows, "dose_contrast", bootstrap_repeats, seed
        )
    slope_subjects = dose_slope(dose_subjects)
    slope_result = bootstrap_subject_mean(
        slope_subjects, "dose_slope", bootstrap_repeats, seed
    )
    sample_keys = {row["sample_key"] for row in model_predictions}
    seeds = sorted({int(row["model_seed"]) for row in model_predictions})
    coverage = audit_oof_prediction_coverage(model_predictions, sample_keys, seeds)
    return {
        "model_results": model_results, "dose_results": dose_results,
        "dose_slope_result": slope_result, "coverage_audit": coverage,
        "sample_contrasts": model_contrasts, "model_subjects": model_subjects,
        "perturbation_contrasts": perturbation_contrasts,
        "dose_subjects": dose_subjects, "slope_subjects": slope_subjects,
    }
