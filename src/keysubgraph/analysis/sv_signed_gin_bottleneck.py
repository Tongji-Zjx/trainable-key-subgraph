"""Read-only bottleneck diagnostics for frozen SV Signed-GIN models."""

from __future__ import absolute_import, division, print_function

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from keysubgraph.models.sv_signed_gin import (
    SVSignedGINBatch,
    SVSignedGINClassifier,
    SVSignedGINSampleInput,
)
from keysubgraph.training.dual_sgw_feature_trainer import binary_metrics
from keysubgraph.training.sv_signed_gin_trainer import (
    site_stratified_roc_auc,
)


SV_CHANNEL_MASK_CONDITIONS = (
    "all",
    "mask_gin",
    "mask_static",
    "mask_variation",
    "gin_only",
    "static_only",
    "variation_only",
    "mask_static_spectral",
    "mask_static_structural",
    "static_spectral_only",
    "static_structural_only",
)


def _safe_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size < 2 or set(labels.tolist()) != {0, 1}:
        return None
    return float(roc_auc_score(labels, probabilities))


def _safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.size < 2
        or float(left.std()) <= 1.0e-12
        or float(right.std()) <= 1.0e-12
    ):
        return None
    value = float(spearmanr(left, right)[0])
    return value if math.isfinite(value) else None


def _effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 1.0e-20:
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def _mean_pairwise_cosine(values, maximum_rows=512):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return None
    if values.shape[0] > int(maximum_rows):
        indices = np.linspace(
            0, values.shape[0] - 1, int(maximum_rows)
        ).astype(np.int64)
        values = values[indices]
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1.0e-12
    values = values[valid]
    norms = norms[valid]
    if values.shape[0] < 2:
        return None
    normalized = values / norms[:, None]
    similarities = normalized.dot(normalized.T)
    upper = similarities[np.triu_indices(values.shape[0], k=1)]
    return float(upper.mean()) if upper.size else None


def representation_statistics(values, labels):
    """Summarize variance, rank, cosine collapse, and class separation."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or values.shape[0] != labels.size
        or not np.isfinite(values).all()
    ):
        raise ValueError("representation values must be finite [N,D]")
    feature_variance = values.var(axis=0)
    norms = np.linalg.norm(values, axis=1)
    effective_rank = _effective_rank(values)
    result = {
        "sample_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "mean_feature_variance": float(feature_variance.mean()),
        "median_feature_variance": float(np.median(feature_variance)),
        "active_feature_fraction": float(
            np.mean(np.sqrt(feature_variance) > 1.0e-6)
        ),
        "effective_rank": effective_rank,
        "normalized_effective_rank": (
            effective_rank / float(max(1, values.shape[1]))
        ),
        "mean_pairwise_cosine": _mean_pairwise_cosine(values),
        "mean_norm": float(norms.mean()),
        "norm_standard_deviation": float(norms.std()),
    }
    if set(labels.tolist()) == {0, 1}:
        zero = values[labels == 0]
        one = values[labels == 1]
        zero_centroid = zero.mean(axis=0)
        one_centroid = one.mean(axis=0)
        difference = one_centroid - zero_centroid
        within = (
            float(
                np.mean(
                    np.sum((zero - zero_centroid) ** 2, axis=1)
                )
            )
            + float(
                np.mean(
                    np.sum((one - one_centroid) ** 2, axis=1)
                )
            )
        )
        denominator = max(
            1.0e-12,
            float(np.linalg.norm(zero_centroid))
            * float(np.linalg.norm(one_centroid)),
        )
        result.update(
            {
                "class_centroid_distance": float(
                    np.linalg.norm(difference)
                ),
                "class_centroid_cosine": float(
                    np.dot(zero_centroid, one_centroid) / denominator
                ),
                "class_fisher_ratio": float(
                    np.dot(difference, difference)
                )
                / max(1.0e-12, within),
            }
        )
    return result


def representation_drift(train_values, validation_values):
    train = np.asarray(train_values, dtype=np.float64)
    validation = np.asarray(validation_values, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
    if validation.ndim == 1:
        validation = validation[:, None]
    if train.shape[1] != validation.shape[1]:
        raise ValueError("drift representations have different dimensions")
    pooled = np.sqrt(
        0.5 * (train.var(axis=0) + validation.var(axis=0))
    )
    standardized = np.abs(
        validation.mean(axis=0) - train.mean(axis=0)
    ) / np.maximum(pooled, 1.0e-12)
    return {
        "median_absolute_standardized_mean_shift": float(
            np.median(standardized)
        ),
        "maximum_absolute_standardized_mean_shift": float(
            np.max(standardized)
        ),
    }


def _frozen_probe(
    train_values,
    train_targets,
    validation_values,
    validation_targets,
    seed,
    binary,
):
    train_values = np.asarray(train_values, dtype=np.float64)
    validation_values = np.asarray(
        validation_values, dtype=np.float64
    )
    train_targets = np.asarray(train_targets)
    validation_targets = np.asarray(validation_targets)
    classes = sorted(set(train_targets.tolist()))
    known = np.asarray(
        [value in set(classes) for value in validation_targets],
        dtype=bool,
    )
    if len(classes) < 2 or int(known.sum()) < 2:
        return {
            "available": False,
            "reason": "probe requires at least two train/validation classes",
        }
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=int(seed),
            solver="lbfgs",
        ),
    )
    pipeline.fit(train_values, train_targets)
    prediction = pipeline.predict(validation_values[known])
    target = validation_targets[known]
    result = {
        "available": True,
        "train_sample_count": int(train_values.shape[0]),
        "validation_sample_count": int(known.sum()),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
    }
    probabilities = pipeline.predict_proba(validation_values[known])
    if binary and set(target.tolist()) == {0, 1}:
        positive_index = list(pipeline.classes_).index(1)
        result["roc_auc"] = float(
            roc_auc_score(target, probabilities[:, positive_index])
        )
    return result


def frozen_label_probe(
    train_values,
    train_labels,
    validation_values,
    validation_labels,
    seed=2026,
):
    return _frozen_probe(
        train_values,
        train_labels,
        validation_values,
        validation_labels,
        seed,
        True,
    )


def frozen_site_probe(
    train_values,
    train_sites,
    validation_values,
    validation_sites,
    seed=2026,
):
    return _frozen_probe(
        train_values,
        train_sites,
        validation_values,
        validation_sites,
        seed,
        False,
    )


def site_only_label_baseline(
    train_sites,
    train_labels,
    validation_sites,
    validation_labels,
):
    train_labels = np.asarray(train_labels, dtype=np.int64)
    global_rate = float(train_labels.mean())
    site_rates = {}
    for site in sorted(set(train_sites)):
        indices = [
            index
            for index, value in enumerate(train_sites)
            if value == site
        ]
        site_rates[str(site)] = float(train_labels[indices].mean())
    probabilities = [
        site_rates.get(str(site), global_rate)
        for site in validation_sites
    ]
    return {
        "roc_auc": _safe_auc(validation_labels, probabilities),
        "known_site_fraction": float(
            np.mean(
                [
                    str(site) in site_rates
                    for site in validation_sites
                ]
            )
        ),
        "global_train_positive_rate": global_rate,
        "site_positive_rates": site_rates,
    }


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 1:
        return {"count": 0}
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std()),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _per_site_auc_rows(labels, probabilities, sites):
    labels = list(int(value) for value in labels)
    probabilities = list(float(value) for value in probabilities)
    sites = list(str(value) for value in sites)
    if not (
        len(labels) == len(probabilities) == len(sites)
    ):
        raise ValueError("site-conditioned vectors are misaligned")
    rows = []
    for site in sorted(set(sites)):
        indices = [
            index for index, value in enumerate(sites) if value == site
        ]
        site_labels = [labels[index] for index in indices]
        site_probabilities = [
            probabilities[index] for index in indices
        ]
        class_counts = {
            str(label): int(
                sum(value == label for value in site_labels)
            )
            for label in (0, 1)
        }
        rows.append(
            {
                "site": site,
                "sample_count": len(indices),
                "class_0_count": class_counts["0"],
                "class_1_count": class_counts["1"],
                "roc_auc": _safe_auc(
                    site_labels, site_probabilities
                ),
            }
        )
    return rows


def diagnose_sv_sample(
    model: SVSignedGINClassifier,
    sample: SVSignedGINSampleInput,
):
    """Reproduce one SG2 forward pass and expose signed/pooling internals."""
    if model.config.variant != "signed_gin_static_variation":
        raise ValueError("bottleneck diagnosis requires SG2")
    layer_graph_means = [
        [] for _ in range(model.config.gin_layers + 1)
    ]
    window_embeddings = []
    cancellation = []
    attention_rows = []
    for window in sample.windows:
        states = model.encoder.node_projection(window.node_features)
        layer_graph_means[0].append(states.mean(dim=0))
        degree = window.adjacency.abs().sum(dim=1)
        for layer_index, layer in enumerate(model.encoder.layers):
            positive = window.adjacency.clamp_min(0.0)
            negative = -window.adjacency.clamp_max(0.0)
            positive_message = positive.matmul(states)
            negative_message = negative.matmul(states)
            net_message = positive_message - negative_message
            positive_norm = torch.linalg.vector_norm(
                positive_message, dim=1
            )
            negative_norm = torch.linalg.vector_norm(
                negative_message, dim=1
            )
            net_norm = torch.linalg.vector_norm(net_message, dim=1)
            ratio = net_norm / (
                positive_norm + negative_norm + 1.0e-12
            )
            cancellation.append(
                {
                    "layer": int(layer_index + 1),
                    "positive_norm": positive_norm.detach().cpu().numpy(),
                    "negative_norm": negative_norm.detach().cpu().numpy(),
                    "net_norm": net_norm.detach().cpu().numpy(),
                    "ratio": ratio.detach().cpu().numpy(),
                }
            )
            aggregated = (
                (1.0 + layer.epsilon.to(states)) * states
                + net_message
            )
            states = layer.mlp(aggregated)
            layer_graph_means[layer_index + 1].append(
                states.mean(dim=0)
            )
        if model.config.pooling == "attention":
            scores = model.encoder.attention(states).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            embedding = (states * weights[:, None]).sum(dim=0)
        elif model.config.pooling == "mean":
            weights = states.new_full(
                (states.shape[0],), 1.0 / float(states.shape[0])
            )
            embedding = states.mean(dim=0)
        else:
            maximum = states.max(dim=0)
            embedding = maximum.values
            weights = (
                states == maximum.values[None, :]
            ).any(dim=-1).to(states.dtype)
        window_embeddings.append(embedding)
        probability_weights = weights / weights.sum().clamp_min(1.0e-12)
        entropy = -(
            probability_weights
            * probability_weights.clamp_min(1.0e-12).log()
        ).sum()
        normalized_entropy = (
            entropy / math.log(float(weights.numel()))
            if weights.numel() > 1
            else entropy.new_tensor(1.0)
        )
        attention_rows.append(
            {
                "node_count": int(weights.numel()),
                "normalized_entropy": float(
                    normalized_entropy.detach().cpu()
                ),
                "maximum_weight": float(
                    probability_weights.max().detach().cpu()
                ),
                "effective_node_count": float(
                    entropy.exp().detach().cpu()
                ),
                "degree_spearman": _safe_spearman(
                    probability_weights.detach().cpu().numpy(),
                    degree.detach().cpu().numpy(),
                ),
            }
        )
    gin = torch.stack(window_embeddings, dim=0).mean(dim=0)
    gin_projection = model.gin_projection(gin[None, :]).squeeze(0)
    static_projection = model.static_projection(
        sample.static_features[None, :]
    ).squeeze(0)
    variation_projection = model.variation_projection(
        sample.variation[None, :]
    ).squeeze(0)
    final = torch.cat(
        (gin_projection, static_projection, variation_projection),
        dim=-1,
    )
    classifier_linear = model.classifier[0](final)
    classifier_hidden = model.classifier[1](classifier_linear)
    classifier_regularized = model.classifier[2](classifier_hidden)
    logits = model.classifier[3](classifier_regularized)
    probability = torch.softmax(logits, dim=-1)[1]
    representations = {
        "raw_static": sample.static_features.detach().cpu().numpy(),
        "raw_static_spectral": (
            sample.static_features[:16].detach().cpu().numpy()
        ),
        "raw_static_structural": (
            sample.static_features[16:].detach().cpu().numpy()
        ),
        "raw_variation": sample.variation.detach().cpu().numpy(),
        "gin_representation": gin.detach().cpu().numpy(),
        "gin_projection": gin_projection.detach().cpu().numpy(),
        "static_projection": static_projection.detach().cpu().numpy(),
        "variation_projection": (
            variation_projection.detach().cpu().numpy()
        ),
        "final_representation": final.detach().cpu().numpy(),
        "classifier_linear": classifier_linear.detach().cpu().numpy(),
        "classifier_hidden": classifier_hidden.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
        "positive_probability": np.asarray(
            [float(probability.detach().cpu())], dtype=np.float64
        ),
    }
    for index, values in enumerate(layer_graph_means):
        representations[
            "node_projection_graph_mean"
            if index == 0
            else "gin_layer_{}_graph_mean".format(index)
        ] = (
            torch.stack(values, dim=0)
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )
    return {
        "representations": representations,
        "logits": logits.detach(),
        "probability": float(probability.detach().cpu()),
        "cancellation": cancellation,
        "attention": attention_rows,
    }


def collect_sv_diagnostics(
    model: SVSignedGINClassifier,
    dataset,
    device,
    progress_callback=None,
):
    model.eval()
    representation_rows = {}
    cancellation_rows = []
    attention_rows = []
    probabilities = []
    maximum_forward_difference = 0.0
    with torch.no_grad():
        for index, cpu_sample in enumerate(dataset.samples):
            sample = cpu_sample.to(device)
            diagnosed = diagnose_sv_sample(model, sample)
            # One exact cross-check per split is sufficient because the
            # diagnostic follows the same frozen modules for every sample.
            # Avoiding a second full GIN pass for all remaining samples keeps
            # this cohort-wide audit inexpensive.
            if index == 0:
                actual = model(SVSignedGINBatch((sample,)))
                difference = float(
                    (
                        actual.logits.squeeze(0) - diagnosed["logits"]
                    )
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                )
                maximum_forward_difference = max(
                    maximum_forward_difference, difference
                )
            for name, values in diagnosed["representations"].items():
                representation_rows.setdefault(name, []).append(values)
            probabilities.append(diagnosed["probability"])
            cancellation_rows.extend(diagnosed["cancellation"])
            attention_rows.extend(diagnosed["attention"])
            if progress_callback is not None:
                progress_callback(index + 1, len(dataset))
    if maximum_forward_difference > 1.0e-5:
        raise RuntimeError(
            "diagnostic forward does not match frozen model"
        )
    representations = {
        name: np.stack(values, axis=0)
        for name, values in representation_rows.items()
    }
    cancellation_summary = {}
    for layer in range(1, model.config.gin_layers + 1):
        selected = [
            row for row in cancellation_rows if row["layer"] == layer
        ]
        ratios = np.concatenate([row["ratio"] for row in selected])
        cancellation_summary["layer_{}".format(layer)] = {
            "positive_message_norm": _distribution(
                np.concatenate(
                    [row["positive_norm"] for row in selected]
                )
            ),
            "negative_message_norm": _distribution(
                np.concatenate(
                    [row["negative_norm"] for row in selected]
                )
            ),
            "net_message_norm": _distribution(
                np.concatenate([row["net_norm"] for row in selected])
            ),
            "cancellation_ratio": _distribution(ratios),
            "fraction_ratio_below_0_20": float(
                np.mean(ratios < 0.20)
            ),
            "fraction_ratio_below_0_10": float(
                np.mean(ratios < 0.10)
            ),
        }
    degree_correlations = [
        row["degree_spearman"]
        for row in attention_rows
        if row["degree_spearman"] is not None
    ]
    attention_summary = {
        "window_count": len(attention_rows),
        "normalized_entropy": _distribution(
            [row["normalized_entropy"] for row in attention_rows]
        ),
        "maximum_weight": _distribution(
            [row["maximum_weight"] for row in attention_rows]
        ),
        "effective_node_count": _distribution(
            [row["effective_node_count"] for row in attention_rows]
        ),
        "degree_spearman": _distribution(degree_correlations),
        "fraction_maximum_weight_above_0_80": float(
            np.mean(
                [
                    row["maximum_weight"] > 0.80
                    for row in attention_rows
                ]
            )
        ),
        "fraction_normalized_entropy_below_0_20": float(
            np.mean(
                [
                    row["normalized_entropy"] < 0.20
                    for row in attention_rows
                ]
            )
        ),
    }
    return {
        "sample_keys": list(dataset.sample_keys),
        "labels": list(dataset.labels),
        "sites": list(dataset.sites),
        "representations": representations,
        "probabilities": np.asarray(
            probabilities, dtype=np.float64
        ),
        "maximum_forward_logit_difference": (
            maximum_forward_difference
        ),
        "signed_cancellation": cancellation_summary,
        "attention_pooling": attention_summary,
    }


def _masked_static_projection(
    model,
    raw_static,
    train_static_mean,
    condition,
    device,
):
    values = np.asarray(raw_static, dtype=np.float32).copy()
    mean = np.asarray(train_static_mean, dtype=np.float32)
    if condition in ("mask_static_spectral", "static_structural_only"):
        values[:, :16] = mean[None, :16]
    elif condition in (
        "mask_static_structural",
        "static_spectral_only",
    ):
        values[:, 16:] = mean[None, 16:]
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    with torch.no_grad():
        projected = model.static_projection(tensor)
    return projected


def frozen_channel_masking(
    model,
    train_collection,
    validation_collection,
    threshold,
    device,
):
    """Mask SG2 channels to train means and run the frozen classifier."""
    labels = validation_collection["labels"]
    sites = validation_collection["sites"]
    train_representations = train_collection["representations"]
    validation_representations = validation_collection[
        "representations"
    ]
    means = {
        name: np.asarray(
            train_representations[name], dtype=np.float64
        ).mean(axis=0)
        for name in (
            "gin_projection",
            "static_projection",
            "variation_projection",
        )
    }
    train_static_mean = np.asarray(
        train_representations["raw_static"], dtype=np.float64
    ).mean(axis=0)
    original = {
        name: torch.tensor(
            validation_representations[name],
            dtype=torch.float32,
            device=device,
        )
        for name in (
            "gin_projection",
            "static_projection",
            "variation_projection",
        )
    }
    count = original["gin_projection"].shape[0]

    def mean_channel(name):
        return torch.tensor(
            means[name], dtype=torch.float32, device=device
        )[None, :].expand(count, -1)

    rows = []
    baseline_auc = None
    with torch.no_grad():
        for condition in SV_CHANNEL_MASK_CONDITIONS:
            channels = {
                name: original[name]
                for name in (
                    "gin_projection",
                    "static_projection",
                    "variation_projection",
                )
            }
            if condition == "mask_gin":
                channels["gin_projection"] = mean_channel(
                    "gin_projection"
                )
            elif condition == "mask_static":
                channels["static_projection"] = mean_channel(
                    "static_projection"
                )
            elif condition == "mask_variation":
                channels["variation_projection"] = mean_channel(
                    "variation_projection"
                )
            elif condition == "gin_only":
                channels["static_projection"] = mean_channel(
                    "static_projection"
                )
                channels["variation_projection"] = mean_channel(
                    "variation_projection"
                )
            elif condition == "static_only":
                channels["gin_projection"] = mean_channel(
                    "gin_projection"
                )
                channels["variation_projection"] = mean_channel(
                    "variation_projection"
                )
            elif condition == "variation_only":
                channels["gin_projection"] = mean_channel(
                    "gin_projection"
                )
                channels["static_projection"] = mean_channel(
                    "static_projection"
                )
            elif condition in (
                "mask_static_spectral",
                "mask_static_structural",
                "static_spectral_only",
                "static_structural_only",
            ):
                channels["static_projection"] = (
                    _masked_static_projection(
                        model,
                        validation_representations["raw_static"],
                        train_static_mean,
                        condition,
                        device,
                    )
                )
                if condition in (
                    "static_spectral_only",
                    "static_structural_only",
                ):
                    channels["gin_projection"] = mean_channel(
                        "gin_projection"
                    )
                    channels["variation_projection"] = mean_channel(
                        "variation_projection"
                    )
            final = torch.cat(
                (
                    channels["gin_projection"],
                    channels["static_projection"],
                    channels["variation_projection"],
                ),
                dim=-1,
            )
            probabilities = torch.softmax(
                model.classifier(final), dim=-1
            )[:, 1].detach().cpu().numpy()
            metrics = binary_metrics(labels, probabilities, threshold)
            auc = _safe_auc(labels, probabilities)
            site_auc = site_stratified_roc_auc(
                list(int(value) for value in labels),
                list(float(value) for value in probabilities),
                list(str(value) for value in sites),
            )
            per_site = _per_site_auc_rows(
                labels, probabilities, sites
            )
            eligible_site_aucs = [
                row["roc_auc"]
                for row in per_site
                if row["roc_auc"] is not None
            ]
            if condition == "all":
                baseline_auc = auc
            rows.append(
                {
                    "condition": condition,
                    "roc_auc": auc,
                    "site_stratified_roc_auc": site_auc,
                    "global_minus_site_stratified_auc": (
                        None
                        if auc is None or site_auc is None
                        else float(auc - site_auc)
                    ),
                    "macro_site_roc_auc": (
                        float(np.mean(eligible_site_aucs))
                        if eligible_site_aucs
                        else None
                    ),
                    "eligible_site_count": len(
                        eligible_site_aucs
                    ),
                    "per_site": per_site,
                    "delta_auc_vs_all": (
                        None
                        if auc is None or baseline_auc is None
                        else float(auc - baseline_auc)
                    ),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "f1": metrics["f1"],
                    "threshold": float(threshold),
                    "threshold_source": (
                        "frozen checkpoint validation threshold"
                    ),
                }
            )
    return rows


def analyze_sv_signed_gin_bottleneck(
    train_collection,
    validation_collection,
    model,
    threshold,
    device,
    seed=2026,
):
    train_labels = train_collection["labels"]
    validation_labels = validation_collection["labels"]
    train_sites = train_collection["sites"]
    validation_sites = validation_collection["sites"]
    train_representations = train_collection["representations"]
    validation_representations = validation_collection[
        "representations"
    ]
    if set(train_representations) != set(validation_representations):
        raise ValueError("diagnostic splits expose different layers")
    layer_results = {}
    for name in train_representations:
        train_values = train_representations[name]
        validation_values = validation_representations[name]
        layer_results[name] = {
            "train": representation_statistics(
                train_values, train_labels
            ),
            "validation": representation_statistics(
                validation_values, validation_labels
            ),
            "train_validation_drift": representation_drift(
                train_values, validation_values
            ),
            "label_probe": frozen_label_probe(
                train_values,
                train_labels,
                validation_values,
                validation_labels,
                seed,
            ),
            "site_probe": frozen_site_probe(
                train_values,
                train_sites,
                validation_values,
                validation_sites,
                seed,
            ),
        }
    masking = frozen_channel_masking(
        model,
        train_collection,
        validation_collection,
        threshold,
        device,
    )
    result = {
        "artifact_type": "sv_signed_gin_read_only_bottleneck_diagnostic",
        "schema_version": 1,
        "analysis_splits": ["train", "validation"],
        "test_used": False,
        "parameter_update_count": 0,
        "model_variant": model.config.variant,
        "threshold": float(threshold),
        "threshold_source": (
            "frozen checkpoint validation balanced_accuracy threshold"
        ),
        "forward_consistency": {
            "train_maximum_absolute_logit_difference": (
                train_collection["maximum_forward_logit_difference"]
            ),
            "validation_maximum_absolute_logit_difference": (
                validation_collection[
                    "maximum_forward_logit_difference"
                ]
            ),
            "passed": bool(
                max(
                    train_collection[
                        "maximum_forward_logit_difference"
                    ],
                    validation_collection[
                        "maximum_forward_logit_difference"
                    ],
                )
                <= 1.0e-5
            ),
        },
        "channel_masking": masking,
        "representations": layer_results,
        "signed_cancellation": {
            "train": train_collection["signed_cancellation"],
            "validation": validation_collection[
                "signed_cancellation"
            ],
        },
        "attention_pooling": {
            "train": train_collection["attention_pooling"],
            "validation": validation_collection[
                "attention_pooling"
            ],
        },
        "site_only_label_baseline": site_only_label_baseline(
            train_sites,
            train_labels,
            validation_sites,
            validation_labels,
        ),
    }
    flags = []
    for row in masking:
        if (
            row["condition"] in (
                "mask_gin",
                "mask_static",
                "mask_variation",
            )
            and row["delta_auc_vs_all"] is not None
            and float(row["delta_auc_vs_all"]) >= -0.01
        ):
            flags.append(
                {
                    "category": "limited_channel_contribution",
                    "target": row["condition"].replace("mask_", ""),
                    "evidence": (
                        "masking changed validation AUROC by {:.6f}".format(
                            float(row["delta_auc_vs_all"])
                        )
                    ),
                    "threshold": "AUROC loss below 0.01",
                }
            )
        if (
            row["global_minus_site_stratified_auc"] is not None
            and float(
                row["global_minus_site_stratified_auc"]
            )
            > 0.05
        ):
            flags.append(
                {
                    "category": "site_sensitive_ranking",
                    "target": row["condition"],
                    "evidence": (
                        "global AUROC exceeds site-stratified AUROC "
                        "by {:.6f}".format(
                            float(
                                row[
                                    "global_minus_site_stratified_auc"
                                ]
                            )
                        )
                    ),
                }
            )
    for name, layer in layer_results.items():
        statistics = layer["validation"]
        reasons = []
        if (
            int(statistics["dimension"]) > 1
            and statistics["mean_pairwise_cosine"] is not None
            and float(statistics["mean_pairwise_cosine"]) > 0.995
        ):
            reasons.append("mean_pairwise_cosine>0.995")
        if float(statistics["mean_feature_variance"]) < 1.0e-6:
            reasons.append("mean_feature_variance<1e-6")
        if float(statistics["normalized_effective_rank"]) < 0.10:
            reasons.append("normalized_effective_rank<0.10")
        if reasons:
            flags.append(
                {
                    "category": "representation_collapse",
                    "target": name,
                    "evidence": ", ".join(reasons),
                }
            )
    for name, row in result["signed_cancellation"][
        "validation"
    ].items():
        if (
            float(row["cancellation_ratio"]["median"]) < 0.20
            or float(row["fraction_ratio_below_0_20"]) > 0.50
        ):
            flags.append(
                {
                    "category": "signed_message_cancellation",
                    "target": name,
                    "evidence": (
                        "median ratio={:.6f}, fraction<0.20={:.6f}".format(
                            float(
                                row["cancellation_ratio"]["median"]
                            ),
                            float(row["fraction_ratio_below_0_20"]),
                        )
                    ),
                }
            )
    attention = result["attention_pooling"]["validation"]
    if (
        float(attention["maximum_weight"]["median"]) > 0.80
        or float(attention["normalized_entropy"]["median"]) < 0.20
    ):
        flags.append(
            {
                "category": "attention_concentration",
                "target": "node_attention",
                "evidence": (
                    "median max={:.6f}, median normalized entropy={:.6f}".format(
                        float(attention["maximum_weight"]["median"]),
                        float(attention["normalized_entropy"]["median"]),
                    )
                ),
            }
        )
    result["automatic_flags"] = flags
    return result


def selection_control_probe(
    named_splits: Mapping[str, Mapping[str, Any]],
    seed=2026,
):
    """Compare learned/random/full caches via in-memory train-only probes."""
    rows = []
    reference = None
    for name in sorted(named_splits):
        train_dataset = named_splits[name]["train"]
        validation_dataset = named_splits[name]["validation"]
        identity = (
            tuple(train_dataset.sample_keys),
            tuple(train_dataset.labels),
            tuple(validation_dataset.sample_keys),
            tuple(validation_dataset.labels),
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(
                "selection control caches do not cover identical samples"
            )
        train_values = np.stack(
            [
                np.concatenate(
                    (
                        sample.static_features.numpy(),
                        sample.variation.numpy(),
                    )
                )
                for sample in train_dataset.samples
            ],
            axis=0,
        )
        validation_values = np.stack(
            [
                np.concatenate(
                    (
                        sample.static_features.numpy(),
                        sample.variation.numpy(),
                    )
                )
                for sample in validation_dataset.samples
            ],
            axis=0,
        )
        metrics = frozen_label_probe(
            train_values,
            train_dataset.labels,
            validation_values,
            validation_dataset.labels,
            seed,
        )
        rows.append(
            {
                "selection_mode": name,
                "feature_block": "standardized_static_plus_variation",
                **metrics
            }
        )
    return rows


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_csv(path, rows):
    path = Path(path).resolve()
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _format_optional(value, digits=6):
    return (
        "N/A"
        if value is None
        else ("{:.{}f}".format(float(value), int(digits)))
    )


def write_sv_signed_gin_bottleneck_artifacts(
    result,
    output_dir,
    selection_controls=None,
):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("bottleneck diagnostic output exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = dict(result)
    payload["selection_controls"] = (
        list(selection_controls) if selection_controls else []
    )
    _atomic_json(output_dir / "diagnostic.json", payload)
    channel_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "per_site"
        }
        for row in payload["channel_masking"]
    ]
    _write_csv(
        output_dir / "channel_masking.csv",
        channel_rows,
    )
    site_rows = []
    for row in payload["channel_masking"]:
        for site in row.get("per_site", []):
            site_rows.append(
                {
                    "condition": row["condition"],
                    **site
                }
            )
    _write_csv(
        output_dir / "channel_masking_by_site.csv",
        site_rows,
    )
    layer_rows = []
    for name, layer in payload["representations"].items():
        validation = layer["validation"]
        layer_rows.append(
            {
                "layer": name,
                "dimension": validation["dimension"],
                "validation_variance": validation[
                    "mean_feature_variance"
                ],
                "validation_effective_rank": validation[
                    "effective_rank"
                ],
                "validation_normalized_effective_rank": validation[
                    "normalized_effective_rank"
                ],
                "validation_mean_pairwise_cosine": validation[
                    "mean_pairwise_cosine"
                ],
                "validation_class_fisher_ratio": validation.get(
                    "class_fisher_ratio"
                ),
                "label_probe_auc": layer["label_probe"].get(
                    "roc_auc"
                ),
                "site_probe_balanced_accuracy": layer[
                    "site_probe"
                ].get("balanced_accuracy"),
            }
        )
    _write_csv(output_dir / "layer_statistics.csv", layer_rows)
    if selection_controls:
        _write_csv(
            output_dir / "selection_controls.csv",
            selection_controls,
        )
    lines = [
        "# SV-HardSGW SignedGIN 一次性只读瓶颈诊断",
        "",
        "- 仅使用：train、validation",
        "- test 使用：否",
        "- 参数更新量：0",
        "- 模型变体：{}".format(payload["model_variant"]),
        "- 前向复现一致性：{}".format(
            "通过"
            if payload["forward_consistency"]["passed"]
            else "失败"
        ),
        "",
        "## 冻结通道屏蔽",
        "",
        "| 条件 | 总体AUROC | Site-stratified AUROC | 总体−站点内 | ΔAUC vs all | BA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["channel_masking"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["condition"],
                _format_optional(row["roc_auc"]),
                _format_optional(
                    row["site_stratified_roc_auc"]
                ),
                _format_optional(
                    row["global_minus_site_stratified_auc"]
                ),
                _format_optional(row["delta_auc_vs_all"]),
                _format_optional(row["balanced_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## 表示层诊断（validation）",
            "",
            "| 层 | 方差 | 有效秩 | 归一化有效秩 | 余弦 | 标签探针AUC | 站点探针BA |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in layer_rows:
        lines.append(
            "| {} | {:.3e} | {} | {} | {} | {} | {} |".format(
                row["layer"],
                float(row["validation_variance"]),
                _format_optional(
                    row["validation_effective_rank"], 3
                ),
                _format_optional(
                    row["validation_normalized_effective_rank"], 3
                ),
                _format_optional(
                    row["validation_mean_pairwise_cosine"], 6
                ),
                _format_optional(row["label_probe_auc"], 6),
                _format_optional(
                    row["site_probe_balanced_accuracy"], 6
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Signed 正负消息抵消（validation）",
            "",
            "| 层 | 抵消比中位数 | 比例<0.20 | 比例<0.10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in payload["signed_cancellation"][
        "validation"
    ].items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                name,
                _format_optional(
                    row["cancellation_ratio"]["median"]
                ),
                _format_optional(
                    row["fraction_ratio_below_0_20"]
                ),
                _format_optional(
                    row["fraction_ratio_below_0_10"]
                ),
            )
        )
    attention = payload["attention_pooling"]["validation"]
    lines.extend(
        [
            "",
            "## Attention pooling（validation）",
            "",
            "- 归一化熵中位数：{}".format(
                _format_optional(
                    attention["normalized_entropy"]["median"]
                )
            ),
            "- 最大节点权重中位数：{}".format(
                _format_optional(
                    attention["maximum_weight"]["median"]
                )
            ),
            "- 有效节点数中位数：{}".format(
                _format_optional(
                    attention["effective_node_count"]["median"]
                )
            ),
            "- attention–|degree| Spearman 中位数：{}".format(
                _format_optional(
                    attention["degree_spearman"].get("median")
                )
            ),
            "",
            "## 站点基线",
            "",
            "- 仅由 train 站点阳性率预测 validation 的 AUROC：{}".format(
                _format_optional(
                    payload["site_only_label_baseline"]["roc_auc"]
                )
            ),
        ]
    )
    if selection_controls:
        lines.extend(
            [
                "",
                "## 选择器对照的冻结线性探针",
                "",
                "| 选择模式 | Validation AUROC | BA | Accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in selection_controls:
            lines.append(
                "| {} | {} | {} | {} |".format(
                    row["selection_mode"],
                    _format_optional(row.get("roc_auc")),
                    _format_optional(
                        row.get("balanced_accuracy")
                    ),
                    _format_optional(row.get("accuracy")),
                )
            )
    lines.extend(
        [
            "",
            "## 自动筛查标记",
            "",
        ]
    )
    if payload.get("automatic_flags"):
        for flag in payload["automatic_flags"]:
            lines.append(
                "- `{}` / `{}`：{}".format(
                    flag["category"],
                    flag["target"],
                    flag["evidence"],
                )
            )
    else:
        lines.append("- 未触发预设阈值。")
    lines.extend(
        [
            "",
            "> 所有神经网络权重均冻结；线性探针只在内存中用 train 拟合并在 validation 评估，未保存模型，未使用 test。",
            "",
        ]
    )
    summary = output_dir / "summary.md"
    with summary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return {
        "diagnostic": str(output_dir / "diagnostic.json"),
        "summary": str(summary),
        "channel_masking": str(output_dir / "channel_masking.csv"),
        "channel_masking_by_site": str(
            output_dir / "channel_masking_by_site.csv"
        ),
        "layer_statistics": str(output_dir / "layer_statistics.csv"),
    }
