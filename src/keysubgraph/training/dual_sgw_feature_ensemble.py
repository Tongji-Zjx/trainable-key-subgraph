"""Probability averaging for independently trained SGW feature classifiers."""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)


def _validate_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    split: str,
) -> Dict[str, Any]:
    if len(evaluations) < 2:
        raise ValueError("an ensemble requires at least two evaluations")
    first = evaluations[0]
    if first.get("split") != split:
        raise ValueError("ensemble evaluation uses the wrong split")
    shared = (
        "classifier_type",
        "protocol_sha256",
        "manifest_sha256",
        "scaler_sha256",
        "manifest_provenance",
    )
    for evaluation in evaluations[1:]:
        if evaluation.get("split") != split:
            raise ValueError("ensemble evaluations mix data splits")
        for key in shared:
            if evaluation.get(key) != first.get(key):
                raise ValueError(
                    "ensemble evaluations disagree on {}".format(key)
                )
    seeds = [int(evaluation["seed"]) for evaluation in evaluations]
    if len(set(seeds)) != len(seeds):
        raise ValueError("ensemble component seeds must be distinct")
    return {
        "classifier_type": first["classifier_type"],
        "protocol_sha256": first["protocol_sha256"],
        "manifest_sha256": first["manifest_sha256"],
        "scaler_sha256": first["scaler_sha256"],
        "manifest_provenance": first["manifest_provenance"],
        "seeds": sorted(seeds),
    }


def _prediction_lookup(
    evaluation: Mapping[str, Any],
) -> Dict[str, Tuple[int, float]]:
    lookup: Dict[str, Tuple[int, float]] = {}
    for item in evaluation.get("predictions", []):
        key = str(item["sample_key"])
        if key in lookup:
            raise ValueError("evaluation contains duplicate sample keys")
        label = int(item["label"])
        probability = float(item["positive_probability"])
        if label not in (0, 1) or probability < 0.0 or probability > 1.0:
            raise ValueError("evaluation contains invalid predictions")
        lookup[key] = (label, probability)
    if not lookup:
        raise ValueError("evaluation contains no predictions")
    return lookup


def average_evaluation_probabilities(
    evaluations: Sequence[Mapping[str, Any]],
    split: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata = _validate_evaluations(evaluations, split)
    lookups = [_prediction_lookup(evaluation) for evaluation in evaluations]
    sample_keys = sorted(lookups[0])
    expected = set(sample_keys)
    for lookup in lookups[1:]:
        if set(lookup) != expected:
            raise ValueError("ensemble components cover different samples")
    predictions = []
    for sample_key in sample_keys:
        labels = [lookup[sample_key][0] for lookup in lookups]
        if len(set(labels)) != 1:
            raise ValueError("ensemble components disagree on labels")
        probabilities = [lookup[sample_key][1] for lookup in lookups]
        predictions.append(
            {
                "sample_key": sample_key,
                "label": labels[0],
                "positive_probability": sum(probabilities)
                / float(len(probabilities)),
                "component_probabilities": probabilities,
            }
        )
    return metadata, predictions


def build_dual_sgw_probability_ensemble(
    validation_evaluations: Sequence[Mapping[str, Any]],
    test_evaluations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    validation_metadata, validation_predictions = (
        average_evaluation_probabilities(
            validation_evaluations, "validation"
        )
    )
    test_metadata, test_predictions = average_evaluation_probabilities(
        test_evaluations, "test"
    )
    for key in (
        "classifier_type",
        "protocol_sha256",
        "scaler_sha256",
        "manifest_provenance",
        "seeds",
    ):
        if validation_metadata[key] != test_metadata[key]:
            raise ValueError(
                "validation and test ensembles disagree on {}".format(key)
            )
    validation_labels = [
        int(item["label"]) for item in validation_predictions
    ]
    validation_probabilities = [
        float(item["positive_probability"])
        for item in validation_predictions
    ]
    test_labels = [int(item["label"]) for item in test_predictions]
    test_probabilities = [
        float(item["positive_probability"]) for item in test_predictions
    ]
    thresholds = {
        "balanced_accuracy": fit_binary_threshold(
            validation_labels,
            validation_probabilities,
            "balanced_accuracy",
        ),
        "accuracy": fit_binary_threshold(
            validation_labels, validation_probabilities, "accuracy"
        ),
    }
    return {
        "artifact": "dual_sgw_feature_probability_ensemble",
        "schema_version": 1,
        "classifier_type": validation_metadata["classifier_type"],
        "component_seeds": validation_metadata["seeds"],
        "protocol_sha256": validation_metadata["protocol_sha256"],
        "scaler_sha256": validation_metadata["scaler_sha256"],
        "manifest_provenance": validation_metadata[
            "manifest_provenance"
        ],
        "validation_manifest_sha256": validation_metadata[
            "manifest_sha256"
        ],
        "test_manifest_sha256": test_metadata["manifest_sha256"],
        "validation_thresholds": thresholds,
        "validation": {
            "metrics": {
                name: binary_metrics(
                    validation_labels,
                    validation_probabilities,
                    threshold,
                )
                for name, threshold in thresholds.items()
            },
            "predictions": validation_predictions,
        },
        "test": {
            "metrics": {
                name: binary_metrics(
                    test_labels, test_probabilities, threshold
                )
                for name, threshold in thresholds.items()
            },
            "predictions": test_predictions,
        },
    }
