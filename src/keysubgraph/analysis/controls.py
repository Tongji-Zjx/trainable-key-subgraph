"""Matched Random, Top-degree, and Low-score control subgraphs."""

from __future__ import absolute_import, division, print_function

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.features.graph_features import GraphFeatureBuilder, GraphTimepointFeatures


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2 ** 32)


def _possible_edges(nodes: Sequence[int], edge_mask: torch.Tensor) -> List[Tuple[int, int]]:
    return [
        (left, right)
        for left in nodes
        for right in nodes
        if left < right and bool(edge_mask[left, right])
    ]


def _spanning_tree(
    nodes: Sequence[int], edges: Sequence[Tuple[int, int]], rng: random.Random
) -> Optional[List[Tuple[int, int]]]:
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    shuffled = list(edges)
    rng.shuffle(shuffled)
    tree = []
    for left, right in shuffled:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
            tree.append((left, right))
    return tree if len(tree) == len(nodes) - 1 else None


def _record(
    sample: GraphSequenceSample,
    features: GraphTimepointFeatures,
    time_index: int,
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    source: str,
    subgraph_index: int,
    repeat_index: Optional[int] = None,
) -> Dict[str, Any]:
    adjacency = sample.adjacency[time_index]
    communities = sample.communities[time_index]
    return {
        "sample_id": sample.sample_id,
        "site": sample.site,
        "label": sample.label,
        "split": sample.split,
        "fold": None,
        "time_index": time_index,
        "subgraph_index": subgraph_index,
        "node_ids": list(nodes),
        "node_names": [sample.node_names[time_index][node] for node in nodes],
        "edge_index": [list(edge) for edge in edges],
        "original_edge_weights": [float(adjacency[left, right]) for left, right in edges],
        "edge_presence_threshold": sample.edge_presence_threshold,
        "community_labels": [int(communities[node]) for node in nodes],
        "delta_degree": [float(features.delta_degree[node]) for node in nodes],
        "delta_degree_mask": [bool(features.delta_degree_mask[node]) for node in nodes],
        "delta_edge_weight": [float(features.delta_edge_weight[left, right]) for left, right in edges],
        "delta_edge_mask": [bool(features.delta_edge_mask[left, right]) for left, right in edges],
        "time_mask": True,
        "node_mask": [True] * len(nodes),
        "subgraph_mask": True,
        "num_valid_subgraphs": 1,
        "original_graph_ref": sample.relative_path,
        "candidate_pool_ref": "{}#time={}".format(sample.sample_key, time_index),
        "source": source,
        "repeat_index": repeat_index,
    }


def generate_random_controls(
    sample: GraphSequenceSample,
    export_payload: Dict[str, Any],
    repeats: int = 100,
    seed: int = 42,
    max_attempts: int = 200,
) -> List[Dict[str, Any]]:
    if repeats < 1 or max_attempts < 1:
        raise ValueError("random control counts must be positive")
    builder = GraphFeatureBuilder()
    controls = []
    for timepoint in export_payload["timepoints"]:
        time_index = int(timepoint["time_index"])
        features = builder.build_timepoint(sample, time_index)
        all_nodes = list(range(sample.adjacency[time_index].shape[0]))
        for key_index, key in enumerate(timepoint["subgraphs"]):
            node_count = len(key["node_ids"])
            edge_count = len(key["edge_index"])
            require_connected = float(key.get("score_connectivity", 0.0)) >= 1.0 - 1e-8
            for repeat_index in range(repeats):
                rng = random.Random(
                    _stable_seed(seed, sample.sample_key, time_index, key_index, repeat_index)
                )
                selected = None
                for _ in range(max_attempts):
                    nodes = sorted(rng.sample(all_nodes, node_count))
                    possible = _possible_edges(nodes, features.edge_mask)
                    if len(possible) < edge_count:
                        continue
                    if require_connected:
                        if edge_count < node_count - 1:
                            continue
                        tree = _spanning_tree(nodes, possible, rng)
                        if tree is None:
                            continue
                        remaining = [edge for edge in possible if edge not in set(tree)]
                        rng.shuffle(remaining)
                        edges = tree + remaining[: edge_count - len(tree)]
                    else:
                        edges = rng.sample(possible, edge_count)
                    selected = (nodes, sorted(edges))
                    break
                if selected is None:
                    continue
                controls.append(
                    _record(
                        sample,
                        features,
                        time_index,
                        selected[0],
                        selected[1],
                        "random",
                        key_index,
                        repeat_index,
                    )
                )
    return controls


def generate_top_degree_controls(
    sample: GraphSequenceSample, export_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    builder = GraphFeatureBuilder()
    controls = []
    for timepoint in export_payload["timepoints"]:
        time_index = int(timepoint["time_index"])
        features = builder.build_timepoint(sample, time_index)
        degree_order = sorted(
            range(features.degree.numel()),
            key=lambda node: (-float(features.degree[node]), node),
        )
        for key_index, key in enumerate(timepoint["subgraphs"]):
            node_count = len(key["node_ids"])
            edge_count = len(key["edge_index"])
            nodes = sorted(degree_order[:node_count])
            possible = _possible_edges(nodes, features.edge_mask)
            possible.sort(
                key=lambda edge: (
                    -float(sample.adjacency[time_index][edge[0], edge[1]].abs()),
                    edge[0],
                    edge[1],
                )
            )
            if len(possible) < edge_count:
                continue
            controls.append(
                _record(
                    sample,
                    features,
                    time_index,
                    nodes,
                    sorted(possible[:edge_count]),
                    "top_degree",
                    key_index,
                )
            )
    return controls


def select_low_score_controls(export_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    controls = []
    for timepoint in export_payload["timepoints"]:
        pool = sorted(
            timepoint["candidate_pool"],
            key=lambda item: (float(item["candidate_score"]), int(item["seed_node"])),
        )
        used = set()
        for key_index, key in enumerate(timepoint["subgraphs"]):
            match = None
            for pool_index, candidate in enumerate(pool):
                if pool_index in used:
                    continue
                if len(candidate["node_ids"]) != len(key["node_ids"]):
                    continue
                if len(candidate["edge_index"]) != len(key["edge_index"]):
                    continue
                if (
                    candidate["node_ids"] == key["node_ids"]
                    and candidate["edge_index"] == key["edge_index"]
                ):
                    continue
                match = dict(candidate)
                used.add(pool_index)
                break
            if match is None:
                continue
            match["source"] = "low_score"
            match["repeat_index"] = None
            match["subgraph_index"] = key_index
            match["subgraph_mask"] = True
            controls.append(match)
    return controls
