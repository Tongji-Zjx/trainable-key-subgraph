"""Signed structural metrics with per-metric missing-value masks."""

from __future__ import absolute_import, division, print_function

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


METRIC_NAMES = (
    "node_count",
    "edge_count",
    "density",
    "abs_edge_weight_mean",
    "abs_connection_sum",
    "positive_edge_weight_mean",
    "positive_connection_sum",
    "negative_edge_magnitude_mean",
    "negative_connection_magnitude_sum",
    "node_dynamic_mean_abs",
    "edge_dynamic_mean_abs",
    "positive_intra_ratio",
    "positive_inter_ratio",
    "negative_intra_ratio",
    "negative_inter_ratio",
)


def _mean_or_nan(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def compute_subgraph_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the verified 15-dimensional metric vector for one subgraph."""

    if not record.get("time_mask", True) or not record.get("subgraph_mask", True):
        raise ValueError("masked padding is not a valid subgraph record")
    node_ids = [int(item) for item in record["node_ids"]]
    node_count = len(node_ids)
    if node_count < 2 or len(set(node_ids)) != node_count:
        raise ValueError("subgraph node_ids must contain at least two unique nodes")
    node_set = set(node_ids)
    edges = [tuple(int(value) for value in edge) for edge in record["edge_index"]]
    weights = [float(value) for value in record["original_edge_weights"]]
    if len(edges) != len(weights) or not edges:
        raise ValueError("subgraph edges and weights must be non-empty and aligned")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate undirected edges")
    if any(left >= right or left not in node_set or right not in node_set for left, right in edges):
        raise ValueError("edges must be canonical i<j and reference selected nodes")
    threshold = float(record["edge_presence_threshold"])
    if threshold < 0.0 or any(abs(weight) <= threshold for weight in weights):
        raise ValueError("exported edge violates the frozen edge threshold")

    edge_count = len(edges)
    positive_indices = [index for index, weight in enumerate(weights) if weight > threshold]
    negative_indices = [index for index, weight in enumerate(weights) if weight < -threshold]
    positive_weights = [weights[index] for index in positive_indices]
    negative_magnitudes = [abs(weights[index]) for index in negative_indices]
    absolute_weights = [abs(weight) for weight in weights]

    community_values = [int(item) for item in record["community_labels"]]
    if len(community_values) != node_count:
        raise ValueError("community labels do not align with node_ids")
    community_by_node = dict(zip(node_ids, community_values))
    positive_intra = sum(
        community_by_node[edges[index][0]] == community_by_node[edges[index][1]]
        for index in positive_indices
    )
    negative_intra = sum(
        community_by_node[edges[index][0]] == community_by_node[edges[index][1]]
        for index in negative_indices
    )

    delta_degree = [float(item) for item in record["delta_degree"]]
    delta_degree_mask = [bool(item) for item in record["delta_degree_mask"]]
    if len(delta_degree) != node_count or len(delta_degree_mask) != node_count:
        raise ValueError("delta degree values do not align with nodes")
    valid_node_delta = [
        abs(value) for value, valid in zip(delta_degree, delta_degree_mask) if valid
    ]
    delta_edge = [float(item) for item in record["delta_edge_weight"]]
    delta_edge_mask = [bool(item) for item in record["delta_edge_mask"]]
    if len(delta_edge) != edge_count or len(delta_edge_mask) != edge_count:
        raise ValueError("delta edge values do not align with edges")
    valid_edge_delta = [
        abs(value) for value, valid in zip(delta_edge, delta_edge_mask) if valid
    ]

    result = {
        "sample_id": record["sample_id"],
        "site": record["site"],
        "label": int(record["label"]),
        "split": record["split"],
        "fold": record.get("fold"),
        "time_index": int(record["time_index"]),
        "subgraph_index": record.get("subgraph_index"),
        "source": record.get("source", "key"),
        "repeat_index": record.get("repeat_index"),
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "density": 2.0 * edge_count / (node_count * (node_count - 1)),
        "abs_edge_weight_mean": _mean_or_nan(absolute_weights),
        "abs_connection_sum": sum(absolute_weights),
        "positive_edge_weight_mean": _mean_or_nan(positive_weights),
        "positive_connection_sum": sum(positive_weights),
        "negative_edge_magnitude_mean": _mean_or_nan(negative_magnitudes),
        "negative_connection_magnitude_sum": sum(negative_magnitudes),
        "node_dynamic_mean_abs": _mean_or_nan(valid_node_delta),
        "edge_dynamic_mean_abs": _mean_or_nan(valid_edge_delta),
        "positive_intra_ratio": (
            positive_intra / len(positive_indices) if positive_indices else float("nan")
        ),
        "positive_inter_ratio": (
            1.0 - positive_intra / len(positive_indices)
            if positive_indices
            else float("nan")
        ),
        "negative_intra_ratio": (
            negative_intra / len(negative_indices) if negative_indices else float("nan")
        ),
        "negative_inter_ratio": (
            1.0 - negative_intra / len(negative_indices)
            if negative_indices
            else float("nan")
        ),
    }
    for metric in METRIC_NAMES:
        value = result[metric]
        if not math.isnan(value) and not math.isfinite(value):
            raise ValueError("non-finite structural metric: {}".format(metric))
    return result


def aggregate_sample_metrics(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average only valid subgraphs per sample/source/metric."""

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"],
            row["sample_id"],
            row["site"],
            int(row["label"]),
            row["split"],
            row.get("fold"),
        )
        groups[key].append(row)
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        source, sample_id, site, label, split, fold = key
        aggregated = {
            "source": source,
            "sample_id": sample_id,
            "site": site,
            "label": label,
            "split": split,
            "fold": fold,
            "repeat_index": None,
            "valid_subgraph_count": len(group_rows),
        }
        for metric in METRIC_NAMES:
            values = [
                float(row[metric])
                for row in group_rows
                if math.isfinite(float(row[metric]))
            ]
            aggregated[metric] = _mean_or_nan(values)
            aggregated[metric + "__valid_count"] = len(values)
        output.append(aggregated)
    return output
