"""Strict outer-fold evaluation for a frozen selector-transfer probe."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from keysubgraph.analysis.selector_transfer_probe import _load_probe_split
from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)


def evaluate_selector_transfer_outer_fold(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit on inner-train, freeze on validation, evaluate outer-test once."""

    splits = {
        "train": _load_probe_split(train_manifest),
        "validation": _load_probe_split(validation_manifest),
        "test": _load_probe_split(test_manifest),
    }
    for name, data in splits.items():
        if data["manifest"]["split"] != name:
            raise ValueError("selector OOF manifest split mismatch")
        if data["features"].shape[1] != 44:
            raise ValueError("selector OOF probe requires 44-D features")
    protocols = {
        data["manifest"]["protocol_sha256"] for data in splits.values()
    }
    selectors = {
        data["manifest"]["selector_checkpoint_sha256"]
        for data in splits.values()
    }
    if len(protocols) != 1 or len(selectors) != 1:
        raise ValueError("selector OOF manifests disagree on provenance")
    key_sets = {
        name: set(data["sample_keys"]) for name, data in splits.items()
    }
    if (
        key_sets["train"] & key_sets["validation"]
        or key_sets["train"] & key_sets["test"]
        or key_sets["validation"] & key_sets["test"]
    ):
        raise ValueError("selector OOF splits overlap")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(splits["train"]["features"])
    validation_x = scaler.transform(splits["validation"]["features"])
    test_x = scaler.transform(splits["test"]["features"])
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=int(seed),
        solver="liblinear",
    )
    classifier.fit(train_x, splits["train"]["labels"])
    probabilities = {
        "train": classifier.predict_proba(train_x)[:, 1],
        "validation": classifier.predict_proba(validation_x)[:, 1],
        "test": classifier.predict_proba(test_x)[:, 1],
    }
    threshold = fit_binary_threshold(
        splits["validation"]["labels"].tolist(),
        probabilities["validation"].tolist(),
        "balanced_accuracy",
    )
    evaluations = {}
    for name in ("train", "validation", "test"):
        labels = splits[name]["labels"].tolist()
        scores = probabilities[name].tolist()
        metrics = binary_metrics(labels, scores, threshold)
        metrics["site_stratified_roc_auc"] = site_stratified_roc_auc(
            labels,
            scores,
            splits[name]["sites"],
        )
        evaluations[name] = {
            "metrics": metrics,
            "predictions": [
                {
                    "sample_key": key,
                    "site": str(site),
                    "label": int(label),
                    "positive_probability": float(score),
                    "prediction": int(float(score) >= threshold),
                }
                for key, site, label, score in zip(
                    splits[name]["sample_keys"],
                    splits[name]["sites"],
                    labels,
                    scores,
                )
            ],
        }
    return {
        "schema_version": 1,
        "artifact_type": "selector_transfer_outer_fold_evaluation",
        "selector_objective": "full_soft_hard",
        "feature_definition": "static_28_plus_variation_16",
        "probe": {
            "type": "balanced_logistic_regression",
            "seed": int(seed),
            "scaler_fit_split": "train",
            "classifier_fit_split": "train",
            "threshold_fit_split": "validation",
            "threshold_strategy": "balanced_accuracy",
            "threshold": float(threshold),
        },
        "protocol_sha256": next(iter(protocols)),
        "selector_checkpoint_sha256": next(iter(selectors)),
        "manifests": {
            "train": str(Path(train_manifest).resolve()),
            "validation": str(Path(validation_manifest).resolve()),
            "test": str(Path(test_manifest).resolve()),
        },
        "evaluations": evaluations,
        "test_used_for_fitting": False,
    }


def write_selector_transfer_outer_fold(
    payload: Mapping[str, Any],
    output: Path,
    overwrite: bool = False,
) -> Path:
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError("selector OOF evaluation already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(output))
    return output
