"""Leakage-safe calibrated nonnegative F0 fusion for SVG-v2."""

from __future__ import absolute_import, division, print_function

import csv
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch.nn import functional as F

from keysubgraph.training.dual_sgw_feature_trainer import (
    binary_metrics,
    fit_binary_threshold,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)


def _logit(probability: float, epsilon: float = 1.0e-6) -> float:
    value = min(1.0 - epsilon, max(epsilon, float(probability)))
    return math.log(value / (1.0 - value))


def read_prediction_csv(path: Path) -> Dict[str, Dict[str, object]]:
    rows = {}
    with Path(path).resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row["sample_key"])
            if key in rows:
                raise ValueError("fusion predictions contain duplicate samples")
            rows[key] = {
                "sample_key": key,
                "label": int(row["label"]),
                "site": str(row["site"]),
                "positive_probability": float(row["positive_probability"]),
            }
    if not rows:
        raise ValueError("fusion prediction file is empty")
    return rows


def align_fusion_predictions(
    short_term: Mapping[str, Mapping[str, object]],
    svg: Mapping[str, Mapping[str, object]],
) -> Tuple[Tuple[str, ...], torch.Tensor, torch.Tensor, Sequence[str]]:
    if set(short_term) != set(svg):
        raise ValueError("fusion branches do not cover the same samples")
    keys = tuple(sorted(short_term))
    labels = []
    sites = []
    logits = []
    for key in keys:
        left = short_term[key]
        right = svg[key]
        if int(left["label"]) != int(right["label"]) or str(
            left["site"]
        ) != str(right["site"]):
            raise ValueError("fusion branch metadata mismatch")
        labels.append(int(left["label"]))
        sites.append(str(left["site"]))
        logits.append(
            (
                _logit(left["positive_probability"]),
                _logit(right["positive_probability"]),
            )
        )
    return (
        keys,
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(labels, dtype=torch.float64),
        sites,
    )


def _fit_platt(logits: torch.Tensor, labels: torch.Tensor, steps: int):
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    raw_slope = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((intercept, raw_slope), lr=0.03)
    for _ in range(int(steps)):
        optimizer.zero_grad()
        slope = F.softplus(raw_slope)
        loss = F.binary_cross_entropy_with_logits(
            intercept + slope * logits, labels
        )
        loss.backward()
        optimizer.step()
    return float(intercept.detach()), float(F.softplus(raw_slope).detach())


def _fit_nonnegative_fusion(
    calibrated: torch.Tensor,
    labels: torch.Tensor,
    l1_weight: float,
    steps: int,
):
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    raw_weights = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((intercept, raw_weights), lr=0.03)
    for _ in range(int(steps)):
        optimizer.zero_grad()
        weights = F.softplus(raw_weights)
        fused = intercept + calibrated.matmul(weights)
        loss = F.binary_cross_entropy_with_logits(fused, labels)
        loss = loss + float(l1_weight) * weights.sum()
        loss.backward()
        optimizer.step()
    return float(intercept.detach()), F.softplus(raw_weights).detach()


def fit_f0_fusion(
    short_term_fit: Mapping[str, Mapping[str, object]],
    svg_fit: Mapping[str, Mapping[str, object]],
    l1_weight: float = 1.0e-3,
    optimization_steps: int = 2000,
) -> Dict[str, object]:
    if l1_weight < 0.0 or optimization_steps < 1:
        raise ValueError("invalid F0 optimization configuration")
    keys, logits, labels, sites = align_fusion_predictions(
        short_term_fit, svg_fit
    )
    if set(int(value) for value in labels.tolist()) != {0, 1}:
        raise ValueError("F0 fit set requires both classes")
    calibrators = []
    calibrated_columns = []
    for column in range(2):
        intercept, slope = _fit_platt(
            logits[:, column], labels, optimization_steps
        )
        calibrators.append({"intercept": intercept, "slope": slope})
        calibrated_columns.append(intercept + slope * logits[:, column])
    calibrated = torch.stack(calibrated_columns, dim=-1)
    intercept, weights = _fit_nonnegative_fusion(
        calibrated, labels, l1_weight, optimization_steps
    )
    probabilities = torch.sigmoid(
        intercept + calibrated.matmul(weights)
    ).tolist()
    threshold = fit_binary_threshold(
        [int(value) for value in labels.tolist()],
        probabilities,
        "balanced_accuracy",
    )
    return {
        "fit_sample_keys": list(keys),
        "fit_sites": list(sites),
        "calibrators": {
            "short_term": calibrators[0],
            "svg_v2": calibrators[1],
        },
        "fusion_intercept": intercept,
        "weights": {
            "short_term": float(weights[0]),
            "svg_v2": float(weights[1]),
        },
        "l1_weight": float(l1_weight),
        "optimization_steps": int(optimization_steps),
        "threshold": float(threshold),
    }


def apply_f0_fusion(
    fitted: Mapping[str, object],
    short_term: Mapping[str, Mapping[str, object]],
    svg: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    keys, logits, labels, sites = align_fusion_predictions(short_term, svg)
    calibrated = []
    for name, column in (("short_term", 0), ("svg_v2", 1)):
        values = fitted["calibrators"][name]
        calibrated.append(
            float(values["intercept"])
            + float(values["slope"]) * logits[:, column]
        )
    calibrated = torch.stack(calibrated, dim=-1)
    weights = torch.tensor(
        (
            fitted["weights"]["short_term"],
            fitted["weights"]["svg_v2"],
        ),
        dtype=torch.float64,
    )
    probabilities = torch.sigmoid(
        float(fitted["fusion_intercept"]) + calibrated.matmul(weights)
    ).tolist()
    labels_list = [int(value) for value in labels.tolist()]
    threshold = float(fitted["threshold"])
    metrics = binary_metrics(labels_list, probabilities, threshold)
    metrics["site_stratified_roc_auc"] = site_stratified_roc_auc(
        labels_list, probabilities, list(sites)
    )
    predictions = [
        {
            "sample_key": key,
            "site": str(sites[index]),
            "label": labels_list[index],
            "positive_probability": float(probabilities[index]),
            "threshold": threshold,
            "predicted_label": int(probabilities[index] >= threshold),
        }
        for index, key in enumerate(keys)
    ]
    return {"metrics": metrics, "predictions": predictions}
