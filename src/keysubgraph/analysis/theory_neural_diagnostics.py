"""Frozen train/validation diagnostics for Stage-1 neural experts."""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _representation_statistics(values, labels):
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    probabilities = energy / max(float(energy.sum()), 1.0e-12)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1.0e-12))))
    normalized_rank = effective_rank / float(max(1, array.shape[1]))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.maximum(norms, 1.0e-12)
    cosine = normalized.dot(normalized.T)
    upper = cosine[np.triu_indices(array.shape[0], k=1)]
    labels = np.asarray(labels, dtype=np.int64)
    centroids = [array[labels == value].mean(axis=0) for value in (0, 1)]
    between = float(np.sum((centroids[1] - centroids[0]) ** 2))
    within_values = []
    for value in (0, 1):
        current = array[labels == value]
        within_values.extend(np.sum((current - centroids[value]) ** 2, axis=1))
    within = float(np.mean(within_values))
    return {
        "dimension": int(array.shape[1]),
        "mean_feature_variance": float(np.var(array, axis=0).mean()),
        "effective_rank": effective_rank,
        "normalized_effective_rank": normalized_rank,
        "mean_pairwise_cosine": float(upper.mean()) if upper.size else 1.0,
        "between_class_squared_distance": between,
        "within_class_squared_distance": within,
        "fisher_ratio": between / max(within, 1.0e-12),
    }


def _binary_probe(train_x, train_y, validation_x, validation_y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear", random_state=0
        ),
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(validation_x)[:, 1]
    return float(roc_auc_score(validation_y, probability))


def _site_probe(train_x, train_sites, validation_x, validation_sites):
    classes = sorted(set(train_sites))
    eligible = [index for index, site in enumerate(validation_sites) if site in classes]
    if len(classes) < 2 or not eligible:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear",
            multi_class="ovr", random_state=0
        ),
    )
    model.fit(train_x, train_sites)
    actual = [validation_sites[index] for index in eligible]
    predicted = model.predict(np.asarray(validation_x)[eligible])
    return float(balanced_accuracy_score(actual, predicted))


def collect_theory_neural_diagnostic_inputs(model, loader, device):
    model.eval()
    payload = {
        "sample_keys": [], "labels": [], "sites": [], "representations": [],
        "node_summaries": [], "edge_summaries": [], "q_errors": [],
        "gamma_errors": [], "film_gamma_norms": [], "film_beta_norms": [],
        "positive_message_norms": [], "negative_message_norms": [],
        "signed_difference_norms": [],
    }
    site_by_key = {
        str(key): str(site)
        for key, site in zip(loader.dataset.sample_keys, loader.dataset.sites)
    }
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            payload["representations"].extend(output.representations.cpu().tolist())
            for sample_input, sample_output in zip(batch.samples, output.samples):
                payload["sample_keys"].append(sample_input.sample_key)
                payload["labels"].append(int(sample_input.label))
                payload["sites"].append(site_by_key[sample_input.sample_key])
                nodes = torch.cat([
                    window.node_features for window in sample_input.windows
                    if window is not None
                ], dim=0)
                node_mean = nodes.mean(dim=0)
                node_std = torch.sqrt((nodes - node_mean).square().mean(dim=0) + 1.0e-8)
                payload["node_summaries"].append(
                    torch.cat((node_mean, node_std)).cpu().tolist()
                )
                edges = torch.cat([
                    window.edge_features[window.adjacency.abs() > 0.0]
                    for window in sample_input.windows
                    if window is not None and bool((window.adjacency.abs() > 0.0).any())
                ], dim=0)
                edge_mean = edges.mean(dim=0)
                edge_std = torch.sqrt((edges - edge_mean).square().mean(dim=0) + 1.0e-8)
                payload["edge_summaries"].append(
                    torch.cat((edge_mean, edge_std)).cpu().tolist()
                )
                if sample_output.q_predictions is not None:
                    payload["q_errors"].append(float(
                        torch.nn.functional.mse_loss(
                            sample_output.q_predictions, sample_output.q_targets
                        ).cpu()
                    ))
                if (sample_output.gamma_predictions is not None
                        and sample_output.gamma_predictions.numel()):
                    payload["gamma_errors"].append(float(
                        torch.nn.functional.mse_loss(
                            sample_output.gamma_predictions,
                            sample_output.gamma_targets
                        ).cpu()
                    ))
                if sample_output.film_values is not None:
                    gamma, beta = sample_output.film_values.chunk(2, dim=-1)
                    payload["film_gamma_norms"].append(float(gamma.norm(dim=-1).mean().cpu()))
                    payload["film_beta_norms"].append(float(beta.norm(dim=-1).mean().cpu()))
                for window_norms in sample_output.message_norms:
                    for positive, negative, difference in window_norms:
                        payload["positive_message_norms"].append(float(positive.cpu()))
                        payload["negative_message_norms"].append(float(negative.cpu()))
                        payload["signed_difference_norms"].append(float(difference.cpu()))
    return payload


def _mean(values):
    return float(np.mean(values)) if values else None


def build_theory_neural_diagnostics(train, validation):
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    validation_labels = np.asarray(validation["labels"], dtype=np.int64)
    result: Dict[str, Any] = {
        "artifact_type": "theory_guided_neural_stage1_diagnostics",
        "uses_test": False,
        "parameters_updated": 0,
        "sample_counts": {
            "train": len(train_labels), "validation": len(validation_labels)
        },
        "representation": _representation_statistics(
            validation["representations"], validation_labels
        ),
        "label_probes": {},
        "site_probes": {},
        "mechanism": {},
    }
    for name in ("representations", "node_summaries", "edge_summaries"):
        result["label_probes"][name] = _binary_probe(
            np.asarray(train[name]), train_labels,
            np.asarray(validation[name]), validation_labels
        )
        result["site_probes"][name] = _site_probe(
            np.asarray(train[name]), train["sites"],
            np.asarray(validation[name]), validation["sites"]
        )
    for name in (
        "q_errors", "gamma_errors", "film_gamma_norms", "film_beta_norms",
        "positive_message_norms", "negative_message_norms",
        "signed_difference_norms",
    ):
        result["mechanism"][name] = {
            "train_mean": _mean(train[name]),
            "validation_mean": _mean(validation[name]),
        }
    return result
