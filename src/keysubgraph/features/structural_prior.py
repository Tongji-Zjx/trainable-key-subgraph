"""Training-fold-only structural standardization and statistical priors."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


# Temporal deltas and exactly complementary inter-community ratios are
# intentionally excluded from this experiment.
STATIC_WINDOW_STRUCTURAL_FEATURES = (
    "node_count",
    "edge_count",
    "density",
    "abs_edge_weight_mean",
    "abs_connection_sum",
    "positive_edge_weight_mean",
    "positive_connection_sum",
    "negative_edge_magnitude_mean",
    "negative_connection_magnitude_sum",
    "positive_intra_ratio",
    "negative_intra_ratio",
)
PRIOR_MODES = ("none", "uniform", "real", "permuted")
STRUCTURAL_GROUPS = ("A", "B", "C", "D", "E")


def compute_static_subgraph_features(
    adjacency: torch.Tensor, communities: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the 11 non-redundant static metrics and their validity mask."""

    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("structural adjacency must be square")
    if communities.shape != (adjacency.shape[0],):
        raise ValueError("communities do not align with structural adjacency")
    node_count = int(adjacency.shape[0])
    upper = torch.triu(adjacency.abs() > float(threshold), diagonal=1)
    edge_positions = upper.nonzero(as_tuple=False)
    if edge_positions.numel() == 0:
        raise ValueError("structural subgraph contains no edges")
    weights = adjacency[edge_positions[:, 0], edge_positions[:, 1]]
    positive = weights > 0.0
    negative = weights < 0.0
    absolute = weights.abs()
    same_community = (
        communities[edge_positions[:, 0]] == communities[edge_positions[:, 1]]
    )

    def mean_or_zero(values):
        return float(values.mean()) if values.numel() else 0.0

    positive_weights = weights[positive]
    negative_magnitudes = absolute[negative]
    values = torch.tensor(
        [
            float(node_count),
            float(weights.numel()),
            2.0 * float(weights.numel()) / (node_count * (node_count - 1)),
            float(absolute.mean()),
            float(absolute.sum()),
            mean_or_zero(positive_weights),
            float(positive_weights.sum()),
            mean_or_zero(negative_magnitudes),
            float(negative_magnitudes.sum()),
            mean_or_zero(same_community[positive].to(torch.float32)),
            mean_or_zero(same_community[negative].to(torch.float32)),
        ],
        dtype=adjacency.dtype,
        device=adjacency.device,
    )
    mask = torch.tensor(
        [
            True, True, True, True, True,
            bool(positive.any()), True,
            bool(negative.any()), True,
            bool(positive.any()), bool(negative.any()),
        ],
        dtype=torch.bool,
        device=adjacency.device,
    )
    return values, mask


def structural_group_configuration(group: str) -> Tuple[bool, str]:
    mapping = {
        "A": (False, "none"),
        "B": (True, "none"),
        "C": (True, "uniform"),
        "D": (True, "real"),
        "E": (True, "permuted"),
    }
    if group not in mapping:
        raise ValueError("unsupported structural experiment group")
    return mapping[group]


def _window_features(window) -> Tuple[np.ndarray, np.ndarray]:
    values = np.stack([
        subgraph.structural_features.detach().cpu().numpy()
        for subgraph in window.subgraphs
    ])
    masks = np.stack([
        subgraph.structural_mask.detach().cpu().numpy().astype(bool)
        for subgraph in window.subgraphs
    ])
    counts = masks.sum(axis=0)
    valid = counts > 0
    output = np.zeros(values.shape[1], dtype=np.float64)
    output[valid] = (values * masks).sum(axis=0)[valid] / counts[valid]
    return output, valid


def fit_structural_transform(
    dataset,
    group: str,
    beta: float = 1.0,
    permutation_seed: int = 42,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Fit standardization and effect-size weights using only one train Dataset."""

    use_features, prior_mode = structural_group_configuration(group)
    if beta < 0.0 or permutation_seed < 0 or epsilon <= 0.0:
        raise ValueError("invalid structural prior configuration")
    feature_count = len(STATIC_WINDOW_STRUCTURAL_FEATURES)
    if hasattr(dataset, "records"):
        frozen_sample_keys = [record.sample_key for record in dataset.records]
    else:
        frozen_sample_keys = [dataset[index].sample_key for index in range(len(dataset))]
    frozen_sample_key_hash = hashlib.sha256(
        json.dumps(sorted(frozen_sample_keys), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not use_features:
        return {
            "schema_version": 1,
            "fitted_on": "train_only",
            "structural_group": group,
            "use_structural_features": False,
            "prior_mode": prior_mode,
            "feature_names": list(STATIC_WINDOW_STRUCTURAL_FEATURES),
            "mean": [0.0] * feature_count,
            "std": [1.0] * feature_count,
            "valid_window_counts": [0] * feature_count,
            "effect_size": [0.0] * feature_count,
            "normalized_importance": [0.0] * feature_count,
            "prior_scale": [1.0] * feature_count,
            "permutation": list(range(feature_count)),
            "beta": float(beta),
            "permutation_seed": int(permutation_seed),
            "epsilon": float(epsilon),
            "sample_count": len(dataset),
            "window_count": 0,
            "train_sample_key_sha256": frozen_sample_key_hash,
        }

    sample_rows: List[Tuple[np.ndarray, np.ndarray, int]] = []
    all_values = []
    all_masks = []
    sample_keys = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        if sample.split != "train":
            raise ValueError("structural transform can only be fit on train samples")
        current_values = []
        current_masks = []
        for window in sample.windows:
            values, mask = _window_features(window)
            all_values.append(values)
            all_masks.append(mask)
            current_values.append(values)
            current_masks.append(mask)
        values_array = np.stack(current_values)
        masks_array = np.stack(current_masks)
        sample_rows.append((values_array, masks_array, int(sample.label)))
        sample_keys.append(sample.sample_key)
    if not all_values:
        raise ValueError("training Dataset contains no structural windows")
    values = np.stack(all_values)
    masks = np.stack(all_masks)
    counts = masks.sum(axis=0)
    if bool((counts == 0).any()):
        missing = [
            STATIC_WINDOW_STRUCTURAL_FEATURES[index]
            for index in np.flatnonzero(counts == 0)
        ]
        raise ValueError("structural features are absent from train: {}".format(missing))
    mean = (values * masks).sum(axis=0) / counts
    centered = (values - mean) * masks
    std = np.sqrt((centered * centered).sum(axis=0) / counts)
    safe_std = np.maximum(std, epsilon)

    sample_values = []
    sample_labels = []
    for current_values, current_masks, label in sample_rows:
        standardized = (current_values - mean) / safe_std
        current_counts = current_masks.sum(axis=0)
        current = np.zeros(feature_count, dtype=np.float64)
        valid = current_counts > 0
        current[valid] = (
            standardized * current_masks
        ).sum(axis=0)[valid] / current_counts[valid]
        sample_values.append(current)
        sample_labels.append(label)
    sample_values_array = np.stack(sample_values)
    labels = np.asarray(sample_labels, dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("structural prior training data must contain both classes")
    class_zero = sample_values_array[labels == 0]
    class_one = sample_values_array[labels == 1]
    if min(len(class_zero), len(class_one)) < 2:
        raise ValueError("each class needs at least two samples for structural prior")
    pooled_variance = (
        (len(class_zero) - 1) * class_zero.var(axis=0, ddof=1)
        + (len(class_one) - 1) * class_one.var(axis=0, ddof=1)
    ) / (len(class_zero) + len(class_one) - 2)
    effect = np.abs(class_one.mean(axis=0) - class_zero.mean(axis=0)) / np.maximum(
        np.sqrt(pooled_variance), epsilon
    )
    maximum = float(effect.max())
    importance = effect / maximum if maximum > 0.0 else np.zeros_like(effect)
    permutation = np.arange(feature_count)
    if prior_mode == "none":
        applied = np.zeros(feature_count, dtype=np.float64)
    elif prior_mode == "uniform":
        applied = np.full(feature_count, float(importance.mean()))
    elif prior_mode == "real":
        applied = importance.copy()
    else:
        rng = np.random.RandomState(permutation_seed)
        permutation = rng.permutation(feature_count)
        if np.array_equal(permutation, np.arange(feature_count)) and feature_count > 1:
            permutation = np.roll(permutation, 1)
        applied = importance[permutation]
    prior_scale = 1.0 + float(beta) * applied
    sample_key_hash = hashlib.sha256(
        json.dumps(sorted(sample_keys), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if sample_key_hash != frozen_sample_key_hash:
        raise RuntimeError("structural fit sample inventory changed during reading")
    return {
        "schema_version": 1,
        "fitted_on": "train_only",
        "structural_group": group,
        "use_structural_features": True,
        "prior_mode": prior_mode,
        "feature_names": list(STATIC_WINDOW_STRUCTURAL_FEATURES),
        "mean": mean.tolist(),
        "std": safe_std.tolist(),
        "raw_std": std.tolist(),
        "valid_window_counts": counts.astype(np.int64).tolist(),
        "effect_size": effect.tolist(),
        "normalized_importance": importance.tolist(),
        "prior_scale": prior_scale.tolist(),
        "permutation": permutation.astype(np.int64).tolist(),
        "beta": float(beta),
        "permutation_seed": int(permutation_seed),
        "epsilon": float(epsilon),
        "sample_count": len(dataset),
        "class_counts": {
            "0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))
        },
        "window_count": len(all_values),
        "train_sample_key_sha256": sample_key_hash,
        "effect_estimation_unit": "sample_mean_over_valid_windows",
        "standardization_unit": "valid_training_windows",
    }
