"""Portable Stage-C cache built only from frozen hard-export artifacts."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .hard_graph_features import (
    HardGraphClassificationFeatures,
    HardGraphFeatureBuilder,
    HardGraphWindow,
)


HARD_GRAPH_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedHardSubgraph:
    adjacency: torch.Tensor
    union_node_indices: torch.Tensor
    candidate_score: float
    seed_node: int


@dataclass(frozen=True)
class CachedHardWindow:
    graph: HardGraphWindow
    features: HardGraphClassificationFeatures
    subgraphs: Tuple[CachedHardSubgraph, ...]


@dataclass(frozen=True)
class HardGraphSampleCache:
    sample_key: str
    sample_id: str
    label: int
    split: str
    windows: Tuple[Optional[CachedHardWindow], ...]
    time_values: Tuple[float, ...]
    time_mask: Tuple[bool, ...]
    eligible_for_stage_c: bool
    exclusion_reason: Optional[str]
    data_protocol_sha256: str
    teacher_checkpoint_sha256: str

    @property
    def num_valid_windows(self) -> int:
        return sum(bool(value) for value in self.time_mask)


def _adjacency_from_edges(
    node_ids: Sequence[int],
    edge_index: Sequence[Sequence[int]],
    edge_weights: Sequence[float],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if len(edge_index) != len(edge_weights):
        raise ValueError("hard graph edge indices and weights differ in length")
    if len(node_ids) != len(set(int(node) for node in node_ids)):
        raise ValueError("hard graph node identities must be unique")
    local = {int(node): index for index, node in enumerate(node_ids)}
    adjacency = torch.zeros((len(node_ids), len(node_ids)), dtype=dtype)
    seen = set()
    for edge, weight in zip(edge_index, edge_weights):
        if len(edge) != 2:
            raise ValueError("hard graph edges must contain two endpoints")
        left, right = int(edge[0]), int(edge[1])
        canonical = (min(left, right), max(left, right))
        if left == right or canonical in seen:
            raise ValueError("hard graph contains a loop or duplicate edge")
        if left not in local or right not in local:
            raise ValueError("hard graph edge endpoint is absent from node identities")
        value = float(weight)
        if not torch.isfinite(torch.tensor(value)) or value == 0.0:
            raise ValueError("hard graph edges require finite nonzero signed weights")
        seen.add(canonical)
        adjacency[local[left], local[right]] = value
        adjacency[local[right], local[left]] = value
    return adjacency


class HardExportFeatureAdapter(object):
    """Reconstruct and recompute Stage-C features without full-graph statistics."""

    def __init__(self, feature_builder: Optional[HardGraphFeatureBuilder] = None) -> None:
        self.feature_builder = feature_builder or HardGraphFeatureBuilder()

    @staticmethod
    def _window_from_payload(
        payload: Dict[str, Any], timepoint: Dict[str, Any]
    ) -> HardGraphWindow:
        valid = bool(timepoint.get("window_valid", timepoint.get("hard_union_available", False)))
        time_start = float(timepoint["time_start"])
        threshold = float(payload["edge_presence_threshold"])
        if not valid:
            return HardGraphWindow(
                adjacency=torch.zeros((0, 0), dtype=torch.float32),
                communities=torch.zeros(0, dtype=torch.long),
                node_names=(),
                node_ids=(),
                time_start=time_start,
                edge_presence_threshold=threshold,
                window_valid=False,
            )
        node_ids = tuple(int(value) for value in timepoint["union_node_ids"])
        node_names = tuple(str(value) for value in timepoint["union_node_names"])
        communities = tuple(int(value) for value in timepoint["union_community_labels"])
        if not (len(node_ids) == len(node_names) == len(communities)):
            raise ValueError("hard union node metadata are not aligned")
        adjacency = _adjacency_from_edges(
            node_ids,
            timepoint["union_edge_index"],
            timepoint["union_original_edge_weights"],
        )
        return HardGraphWindow(
            adjacency=adjacency,
            communities=torch.tensor(communities, dtype=torch.long),
            node_names=node_names,
            node_ids=tuple(str(value) for value in node_ids),
            time_start=time_start,
            edge_presence_threshold=threshold,
            window_valid=True,
        )

    @staticmethod
    def _subgraphs_from_payload(
        timepoint: Dict[str, Any], union_graph: HardGraphWindow
    ) -> Tuple[CachedHardSubgraph, ...]:
        union_ids = tuple(int(value) for value in union_graph.node_ids or ())
        union_local = {node: index for index, node in enumerate(union_ids)}
        results = []
        for item in timepoint.get("subgraphs", ()):
            node_ids = tuple(int(value) for value in item["node_ids"])
            if any(node not in union_local for node in node_ids):
                raise ValueError("selected subgraph is absent from its hard union")
            adjacency = _adjacency_from_edges(
                node_ids, item["edge_index"], item["original_edge_weights"]
            )
            results.append(
                CachedHardSubgraph(
                    adjacency=adjacency,
                    union_node_indices=torch.tensor(
                        [union_local[node] for node in node_ids], dtype=torch.long
                    ),
                    candidate_score=float(item["candidate_score"]),
                    seed_node=int(item["seed_node"]),
                )
            )
        return tuple(results)

    def build(self, payload: Dict[str, Any]) -> HardGraphSampleCache:
        required = (
            "sample_key",
            "sample_id",
            "label",
            "split",
            "edge_presence_threshold",
            "data_protocol_sha256",
            "checkpoint_sha256",
            "timepoints",
        )
        if any(name not in payload for name in required):
            raise ValueError("hard export is missing Stage-C metadata")
        timepoints = sorted(payload["timepoints"], key=lambda item: int(item["time_index"]))
        if [int(item["time_index"]) for item in timepoints] != list(range(len(timepoints))):
            raise ValueError("hard export time indices must be contiguous and ordered")
        graph_windows = tuple(
            self._window_from_payload(payload, timepoint) for timepoint in timepoints
        )
        features = self.feature_builder.build_sequence(graph_windows)
        cached = []
        for graph, feature, timepoint in zip(graph_windows, features, timepoints):
            if feature is None:
                cached.append(None)
                continue
            cached.append(
                CachedHardWindow(
                    graph=graph,
                    features=feature,
                    subgraphs=self._subgraphs_from_payload(timepoint, graph),
                )
            )
        time_mask = tuple(item is not None for item in cached)
        eligible = sum(time_mask) >= 2
        return HardGraphSampleCache(
            sample_key=str(payload["sample_key"]),
            sample_id=str(payload["sample_id"]),
            label=int(payload["label"]),
            split=str(payload["split"]),
            windows=tuple(cached),
            time_values=tuple(float(graph.time_start) for graph in graph_windows),
            time_mask=time_mask,
            eligible_for_stage_c=eligible,
            exclusion_reason=None if eligible else "fewer_than_two_valid_hard_windows",
            data_protocol_sha256=str(payload["data_protocol_sha256"]),
            teacher_checkpoint_sha256=str(payload["checkpoint_sha256"]),
        )


def save_hard_graph_cache(
    cache: HardGraphSampleCache, path: Path, overwrite: bool = False
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("hard graph feature cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": HARD_GRAPH_CACHE_SCHEMA_VERSION,
            "artifact_type": "tg_sgw_hard_graph_feature_cache",
            "cache": cache,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_hard_graph_cache(path: Path) -> HardGraphSampleCache:
    try:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if payload.get("schema_version") != HARD_GRAPH_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported hard graph feature cache schema")
    if payload.get("artifact_type") != "tg_sgw_hard_graph_feature_cache":
        raise ValueError("unexpected hard graph feature cache artifact")
    cache = payload.get("cache")
    if not isinstance(cache, HardGraphSampleCache):
        raise ValueError("hard graph feature cache payload is invalid")
    return cache
