"""Diagnostics for full-graph ordering, input validity, and representation collapse."""

from __future__ import absolute_import, division, print_function

import math
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from keysubgraph.features.graph_features import GraphFeatureBuilder
from keysubgraph.models.full_graph_classifier import (
    FullGraphClassifierOutput,
    FullGraphSequenceClassifier,
    SignedGatedBiGRUPrototypeEncoder,
    SignedGNNTCNFullGraphEncoder,
)


class _ScalarStatistics(object):
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.squared_total = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, values: torch.Tensor) -> None:
        flattened = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if flattened.numel() == 0:
            return
        self.count += int(flattened.numel())
        self.total += float(flattened.sum())
        self.squared_total += float(flattened.square().sum())
        self.minimum = min(self.minimum, float(flattened.min()))
        self.maximum = max(self.maximum, float(flattened.max()))

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "minimum": None,
                "maximum": None,
            }
        mean = self.total / float(self.count)
        variance = max(
            0.0, self.squared_total / float(self.count) - mean * mean
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class _RowRepresentationStatistics(object):
    def __init__(self, maximum_cosine_rows: int = 512) -> None:
        self.count = 0
        self.dimension = None
        self.total = None
        self.squared_total = None
        self.norms = _ScalarStatistics()
        self.rows = []
        self.maximum_cosine_rows = int(maximum_cosine_rows)
        self.saved_row_count = 0

    def add(self, values: torch.Tensor) -> None:
        detached = values.detach().to(device="cpu", dtype=torch.float64)
        if detached.ndim == 1:
            detached = detached.unsqueeze(0)
        elif detached.ndim > 2:
            detached = detached.reshape(-1, detached.shape[-1])
        if detached.ndim != 2 or detached.shape[0] == 0:
            return
        dimension = int(detached.shape[-1])
        if self.dimension is None:
            self.dimension = dimension
            self.total = torch.zeros(dimension, dtype=torch.float64)
            self.squared_total = torch.zeros(dimension, dtype=torch.float64)
        if dimension != self.dimension:
            raise ValueError("representation dimension changed within one diagnostic")
        self.count += int(detached.shape[0])
        self.total += detached.sum(dim=0)
        self.squared_total += detached.square().sum(dim=0)
        self.norms.add(torch.linalg.vector_norm(detached, dim=-1))
        remaining = self.maximum_cosine_rows - self.saved_row_count
        if remaining > 0:
            saved = detached[:remaining].clone()
            self.rows.append(saved)
            self.saved_row_count += int(saved.shape[0])

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "row_count": 0,
                "feature_dim": None,
                "mean_feature_variance": None,
                "median_feature_variance": None,
                "active_feature_fraction": None,
                "mean_pairwise_cosine": None,
                "std_pairwise_cosine": None,
                "norm": self.norms.summary(),
            }
        mean = self.total / float(self.count)
        variance = (
            self.squared_total / float(self.count) - mean.square()
        ).clamp_min(0.0)
        cosine_mean = None
        cosine_std = None
        sampled = torch.cat(self.rows, dim=0) if self.rows else None
        if sampled is not None and sampled.shape[0] > 1:
            normalized = torch.nn.functional.normalize(sampled, dim=-1)
            matrix = normalized.matmul(normalized.transpose(0, 1))
            upper = torch.triu(
                torch.ones_like(matrix, dtype=torch.bool), diagonal=1
            )
            cosine = matrix[upper]
            cosine_mean = float(cosine.mean())
            cosine_std = float(cosine.std(unbiased=False))
        return {
            "row_count": self.count,
            "feature_dim": self.dimension,
            "mean_feature_variance": float(variance.mean()),
            "median_feature_variance": float(variance.median()),
            "minimum_feature_variance": float(variance.min()),
            "maximum_feature_variance": float(variance.max()),
            "active_feature_fraction": float(
                (variance > 1.0e-8).to(torch.float64).mean()
            ),
            "mean_pairwise_cosine": cosine_mean,
            "std_pairwise_cosine": cosine_std,
            "norm": self.norms.summary(),
        }


def validate_full_graph_batch_alignment(
    batch,
    output: FullGraphClassifierOutput,
    assignment_by_key: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if tuple(output.logits.shape[:1]) != (len(batch),):
        raise ValueError("logit batch order cannot match the input batch")
    if output.sequence_lengths.numel() != len(batch):
        raise ValueError("sequence length output cannot match the input batch")
    records = []
    for index, sample in enumerate(batch):
        assignment = assignment_by_key.get(sample.sample_key)
        if assignment is None:
            raise ValueError("batch sample is absent from split assignments")
        if int(assignment.label) != int(sample.label):
            raise ValueError("sample label does not match frozen assignment")
        if output.sequence_lengths[index].item() != sample.num_timepoints:
            raise ValueError("packed sequence order or length does not match sample")
        starts = sample.window_starts.detach().cpu()
        if starts.numel() != sample.num_timepoints:
            raise ValueError("window starts do not match graph sequence length")
        if starts.numel() > 1 and not bool((starts[1:] > starts[:-1]).all()):
            raise ValueError("window starts are not strictly increasing")
        records.append(
            {
                "batch_position": index,
                "sample_key": sample.sample_key,
                "label": int(sample.label),
                "num_timepoints": sample.num_timepoints,
                "node_counts": list(sample.node_counts),
                "window_starts": [float(value) for value in starts.tolist()],
            }
        )
    return records


def summarize_full_graph_inputs(
    samples: Iterable,
    feature_builder: Optional[GraphFeatureBuilder] = None,
) -> Dict[str, Any]:
    feature_builder = feature_builder or GraphFeatureBuilder()
    density = _ScalarStatistics()
    positive_weights = _ScalarStatistics()
    negative_magnitudes = _ScalarStatistics()
    absolute_weights = _ScalarStatistics()
    node_counts = _ScalarStatistics()
    timepoint_counts = _ScalarStatistics()
    sample_count = 0
    timepoint_count = 0
    positive_edge_count = 0
    negative_edge_count = 0
    empty_edge_timepoints = 0
    failures = []

    for sample in samples:
        sample_count += 1
        timepoint_counts.add(torch.tensor([sample.num_timepoints]))
        for time_index, adjacency in enumerate(sample.adjacency):
            timepoint_count += 1
            node_count = int(adjacency.shape[0])
            node_counts.add(torch.tensor([node_count]))
            mask = sample.edge_mask[time_index].to(dtype=torch.bool)
            expected = adjacency.abs() > float(sample.edge_presence_threshold)
            expected = expected.clone()
            expected.fill_diagonal_(False)
            checks = {
                "square": adjacency.ndim == 2
                and adjacency.shape[0] == adjacency.shape[1],
                "finite": bool(torch.isfinite(adjacency).all()),
                "symmetric": bool(
                    torch.allclose(
                        adjacency, adjacency.transpose(0, 1), atol=1.0e-6
                    )
                ),
                "zero_diagonal": bool(
                    torch.allclose(
                        torch.diagonal(adjacency),
                        torch.zeros_like(torch.diagonal(adjacency)),
                        atol=1.0e-7,
                    )
                ),
                "mask_shape": tuple(mask.shape) == tuple(adjacency.shape),
                "mask_symmetric": bool(torch.equal(mask, mask.transpose(0, 1))),
                "mask_matches_protocol": bool(torch.equal(mask, expected)),
            }
            features = feature_builder.build_timepoint(sample, time_index)
            checks["node_features_13d_finite"] = (
                features.node_features.shape[-1] == 13
                and bool(torch.isfinite(features.node_features).all())
            )
            checks["edge_features_4d_finite"] = (
                features.edge_features.shape[-1] == 4
                and bool(torch.isfinite(features.edge_features).all())
            )
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                failures.append(
                    {
                        "sample_key": sample.sample_key,
                        "time_index": time_index,
                        "failed_checks": failed,
                    }
                )
                continue

            upper = torch.triu(mask, diagonal=1)
            weights = adjacency[upper]
            possible = node_count * (node_count - 1) / 2.0
            density.add(
                torch.tensor(
                    [float(weights.numel()) / possible if possible > 0.0 else 0.0]
                )
            )
            if weights.numel() == 0:
                empty_edge_timepoints += 1
                continue
            positive = weights[weights > 0.0]
            negative = -weights[weights < 0.0]
            positive_edge_count += int(positive.numel())
            negative_edge_count += int(negative.numel())
            positive_weights.add(positive)
            negative_magnitudes.add(negative)
            absolute_weights.add(weights.abs())

    return {
        "sample_count": sample_count,
        "timepoint_count": timepoint_count,
        "timepoints_per_sample": timepoint_counts.summary(),
        "nodes_per_timepoint": node_counts.summary(),
        "edge_density": density.summary(),
        "positive_edge_count": positive_edge_count,
        "negative_edge_count": negative_edge_count,
        "positive_edge_weight": positive_weights.summary(),
        "negative_edge_magnitude": negative_magnitudes.summary(),
        "absolute_edge_weight": absolute_weights.summary(),
        "empty_edge_timepoints": empty_edge_timepoints,
        "validation_failure_count": len(failures),
        "validation_failures": failures[:100],
    }


class FullGraphRepresentationMonitor(object):
    """Collect graph/sample-level activation statistics through forward hooks."""

    def __init__(self, model: FullGraphSequenceClassifier) -> None:
        self.model = model
        self.statistics = {}
        self.handles = []
        self._register()

    def _accumulator(self, name: str) -> _RowRepresentationStatistics:
        if name not in self.statistics:
            self.statistics[name] = _RowRepresentationStatistics()
        return self.statistics[name]

    def _register_tensor(self, module, name: str, transform=None) -> None:
        def hook(_module, _inputs, output):
            value = transform(output) if transform is not None else output
            self._accumulator(name).add(value)

        self.handles.append(module.register_forward_hook(hook))

    def _register(self) -> None:
        encoder = self.model.encoder
        if isinstance(encoder, SignedGNNTCNFullGraphEncoder):
            for index, layer in enumerate(encoder.graph_encoder.layers):
                self._register_tensor(
                    layer,
                    "gnn_layer_{}_graph_mean".format(index + 1),
                    lambda output: output.mean(dim=0),
                )
        elif isinstance(encoder, SignedGatedBiGRUPrototypeEncoder):
            self._register_tensor(
                encoder.graph_encoder.node_projection,
                "node_projection_graph_mean",
                lambda output: output.mean(dim=0),
            )
            for index, layer in enumerate(encoder.graph_encoder.layers):
                self._register_tensor(
                    layer,
                    "gated_gnn_layer_{}_graph_mean".format(index + 1),
                    lambda output: output[0].mean(dim=0),
                )
                self._register_tensor(
                    layer,
                    "gated_gnn_layer_{}_valid_gates".format(index + 1),
                    lambda output: output[1][output[1] > 0.0].unsqueeze(-1),
                )
            self._register_tensor(
                encoder.prototype_codebook,
                "prototype_fused_representation",
                lambda output: output[0],
            )
            self._register_tensor(
                encoder.prototype_codebook,
                "prototype_attention",
                lambda output: output[1],
            )
        else:
            raise TypeError("unsupported full-graph encoder for diagnostics")

        if isinstance(encoder, SignedGatedBiGRUPrototypeEncoder):
            self._register_tensor(
                encoder.graph_pooling,
                "window_embedding_pre_normalization",
            )
            self._register_tensor(
                encoder.graph_pooling_normalization,
                "window_embedding",
            )
        else:
            self._register_tensor(encoder.graph_pooling, "window_embedding")
        self._register_tensor(
            encoder.temporal_encoder,
            "temporal_sequence_representation",
            lambda output: output[0],
        )
        linear_index = 0
        for module in self.model.classifier:
            if isinstance(module, nn.Linear):
                linear_index += 1
                self._register_tensor(
                    module,
                    "classifier_linear_{}".format(linear_index),
                )

    def add_model_output(self, output: FullGraphClassifierOutput) -> None:
        # MaskedTCNEncoder.forward_list calls its forward method directly, so
        # a module-level forward hook is not invoked on the controlled
        # baseline. Its encoder output is exactly the temporal sequence
        # representation; record the explicit alias here for a complete,
        # comparable layer table.
        if isinstance(self.model.encoder, SignedGNNTCNFullGraphEncoder):
            self._accumulator("temporal_sequence_representation").add(
                output.representation
            )
        self._accumulator("final_representation").add(output.representation)
        self._accumulator("logits").add(output.logits)
        probabilities = torch.softmax(output.logits.detach(), dim=-1)[:, 1:]
        self._accumulator("positive_probability").add(probabilities)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self) -> Dict[str, Any]:
        return {
            name: statistics.summary()
            for name, statistics in sorted(self.statistics.items())
        }
