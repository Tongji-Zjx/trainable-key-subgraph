"""Original-graph records compatible with the structural metric pipeline."""

from __future__ import absolute_import, print_function

import math
from typing import Any, Dict, Iterable, Iterator, Optional

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.features.graph_features import GraphFeatureBuilder
from keysubgraph.analysis.structural_metrics import METRIC_NAMES


def build_original_graph_record(
    sample: GraphSequenceSample,
    time_index: int,
    feature_builder: Optional[GraphFeatureBuilder] = None,
) -> Dict[str, Any]:
    """Represent one complete valid timepoint as a structural-analysis record."""

    builder = feature_builder or GraphFeatureBuilder()
    features = builder.build_timepoint(sample, time_index)
    adjacency = sample.adjacency[time_index]
    communities = sample.communities[time_index]
    node_count = int(adjacency.shape[0])
    nodes = list(range(node_count))
    upper_mask = torch.triu(features.edge_mask, diagonal=1)
    edge_tensor = torch.nonzero(upper_mask, as_tuple=False)
    if edge_tensor.numel() == 0:
        raise ValueError(
            "original graph has no valid edges: {} time {}".format(
                sample.sample_key, time_index
            )
        )
    edges = [(int(row[0]), int(row[1])) for row in edge_tensor.cpu().tolist()]
    return {
        "sample_id": sample.sample_id,
        "site": sample.site,
        "label": sample.label,
        "split": sample.split,
        "fold": None,
        "time_index": time_index,
        "subgraph_index": 0,
        "node_ids": nodes,
        "node_names": list(sample.node_names[time_index]),
        "edge_index": [list(edge) for edge in edges],
        "original_edge_weights": [
            float(adjacency[left, right]) for left, right in edges
        ],
        "edge_presence_threshold": sample.edge_presence_threshold,
        "community_labels": [int(value) for value in communities.cpu().tolist()],
        "delta_degree": [float(value) for value in features.delta_degree.cpu().tolist()],
        "delta_degree_mask": [
            bool(value) for value in features.delta_degree_mask.cpu().tolist()
        ],
        "delta_edge_weight": [
            float(features.delta_edge_weight[left, right]) for left, right in edges
        ],
        "delta_edge_mask": [
            bool(features.delta_edge_mask[left, right]) for left, right in edges
        ],
        "time_mask": True,
        "node_mask": [True] * node_count,
        "subgraph_mask": True,
        "num_valid_subgraphs": 1,
        "original_graph_ref": sample.relative_path,
        "candidate_pool_ref": None,
        "source": "original",
        "repeat_index": None,
    }


def iter_original_graph_records(
    samples: Iterable[GraphSequenceSample],
) -> Iterator[Dict[str, Any]]:
    """Yield complete-graph records without retaining raw graph records in memory."""

    builder = GraphFeatureBuilder()
    for sample in samples:
        for time_index in range(sample.num_timepoints):
            yield build_original_graph_record(sample, time_index, builder)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def compute_original_graph_metrics(
    sample: GraphSequenceSample,
    time_index: int,
    feature_builder: Optional[GraphFeatureBuilder] = None,
) -> Dict[str, Any]:
    """Compute the same 15 metrics directly on a complete graph with tensors."""

    builder = feature_builder or GraphFeatureBuilder()
    features = builder.build_timepoint(sample, time_index)
    adjacency = sample.adjacency[time_index]
    communities = sample.communities[time_index]
    node_count = int(adjacency.shape[0])
    upper_mask = torch.triu(features.edge_mask, diagonal=1)
    edge_index = torch.nonzero(upper_mask, as_tuple=False)
    if edge_index.numel() == 0:
        raise ValueError(
            "original graph has no valid edges: {} time {}".format(
                sample.sample_key, time_index
            )
        )
    left, right = edge_index[:, 0], edge_index[:, 1]
    weights = adjacency[left, right]
    absolute = weights.abs()
    threshold = sample.edge_presence_threshold
    positive = weights > threshold
    negative = weights < -threshold
    same_community = communities[left] == communities[right]
    edge_count = int(weights.numel())
    delta_node_values = features.delta_degree.abs()
    delta_edge_values = features.delta_edge_weight[left, right].abs()
    delta_edge_valid = features.delta_edge_mask[left, right]

    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    positive_sum = float(weights[positive].sum()) if positive_count else 0.0
    negative_sum = float(absolute[negative].sum()) if negative_count else 0.0
    result = {
        "sample_id": sample.sample_id,
        "site": sample.site,
        "label": sample.label,
        "split": sample.split,
        "fold": None,
        "time_index": time_index,
        "subgraph_index": 0,
        "source": "original",
        "repeat_index": None,
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "density": 2.0 * edge_count / (node_count * (node_count - 1)),
        "abs_edge_weight_mean": float(absolute.mean()),
        "abs_connection_sum": float(absolute.sum()),
        "positive_edge_weight_mean": _masked_mean(weights, positive),
        "positive_connection_sum": positive_sum,
        "negative_edge_magnitude_mean": _masked_mean(absolute, negative),
        "negative_connection_magnitude_sum": negative_sum,
        "node_dynamic_mean_abs": _masked_mean(
            delta_node_values, features.delta_degree_mask
        ),
        "edge_dynamic_mean_abs": _masked_mean(
            delta_edge_values, delta_edge_valid
        ),
        "positive_intra_ratio": (
            float((positive & same_community).sum()) / positive_count
            if positive_count
            else float("nan")
        ),
        "positive_inter_ratio": (
            float((positive & ~same_community).sum()) / positive_count
            if positive_count
            else float("nan")
        ),
        "negative_intra_ratio": (
            float((negative & same_community).sum()) / negative_count
            if negative_count
            else float("nan")
        ),
        "negative_inter_ratio": (
            float((negative & ~same_community).sum()) / negative_count
            if negative_count
            else float("nan")
        ),
    }
    for metric in METRIC_NAMES:
        value = result[metric]
        if not math.isnan(value) and not math.isfinite(value):
            raise ValueError("non-finite original-graph metric: {}".format(metric))
    return result


def iter_original_graph_metrics(
    samples: Iterable[GraphSequenceSample],
) -> Iterator[Dict[str, Any]]:
    """Yield tensor-computed complete-graph metrics one timepoint at a time."""

    builder = GraphFeatureBuilder()
    for sample in samples:
        for time_index in range(sample.num_timepoints):
            yield compute_original_graph_metrics(sample, time_index, builder)
