"""Hard-subgraph sequence Dataset for the signed baseline classifier."""

from __future__ import absolute_import, division, print_function

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from keysubgraph.features.graph_features import GraphFeatureBuilder

from .baseline_manifest import (
    BaselineManifestRecord,
    read_baseline_manifest,
    resolve_manifest_path,
)
from .data_protocol import validate_data_protocol
from .graph_dataset import GraphSequenceDataset, GraphSequenceSample


@dataclass(frozen=True)
class BaselineSubgraph:
    """One exported hard subgraph with local node indexing."""

    node_ids: torch.Tensor
    node_names: Tuple[str, ...]
    community_labels: torch.Tensor
    adjacency: torch.Tensor
    edge_mask: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "BaselineSubgraph":
        return BaselineSubgraph(
            node_ids=self.node_ids.to(device=device, non_blocking=non_blocking),
            node_names=self.node_names,
            community_labels=self.community_labels.to(
                device=device, non_blocking=non_blocking
            ),
            adjacency=self.adjacency.to(device=device, non_blocking=non_blocking),
            edge_mask=self.edge_mask.to(device=device, non_blocking=non_blocking),
            node_features=self.node_features.to(
                device=device, non_blocking=non_blocking
            ),
            edge_index=self.edge_index.to(device=device, non_blocking=non_blocking),
            edge_weight=self.edge_weight.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class BaselineWindow:
    time_index: int
    subgraphs: Tuple[BaselineSubgraph, ...]

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "BaselineWindow":
        return BaselineWindow(
            time_index=self.time_index,
            subgraphs=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.subgraphs
            ),
        )


@dataclass(frozen=True)
class BaselineSequenceSample:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    split: str
    windows: Tuple[BaselineWindow, ...]

    @property
    def num_timepoints(self) -> int:
        return len(self.windows)

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "BaselineSequenceSample":
        return BaselineSequenceSample(
            sample_key=self.sample_key,
            sample_id=self.sample_id,
            site=self.site,
            subject_id=self.subject_id,
            session_id=self.session_id,
            label=self.label,
            split=self.split,
            windows=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.windows
            ),
        )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("hard-subgraph payload must be a dict")
    return payload


def _validate_export_metadata(
    payload: Dict[str, Any], record: BaselineManifestRecord, sample: GraphSequenceSample
) -> None:
    expected = (
        ("sample_key", record.sample_key),
        ("sample_id", sample.sample_id),
        ("site", sample.site),
        ("subject_id", sample.subject_id),
        ("session_id", sample.session_id),
        ("label", sample.label),
        ("split", sample.split),
        ("relative_path", sample.relative_path),
    )
    for name, value in expected:
        if payload.get(name) != value:
            raise ValueError("hard-subgraph metadata mismatch: {}".format(name))


def _local_subgraph(
    exported: Dict[str, Any],
    raw_sample: GraphSequenceSample,
    time_index: int,
    feature_builder: GraphFeatureBuilder,
) -> BaselineSubgraph:
    raw_adjacency = raw_sample.adjacency[time_index]
    raw_communities = raw_sample.communities[time_index]
    raw_names = raw_sample.node_names[time_index]
    node_ids_list = exported.get("node_ids")
    if not isinstance(node_ids_list, list) or len(node_ids_list) < 2:
        raise ValueError("subgraph node_ids must contain at least two nodes")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in node_ids_list):
        raise ValueError("subgraph node_ids must be integers")
    if len(set(node_ids_list)) != len(node_ids_list):
        raise ValueError("subgraph node_ids contain duplicates")
    if min(node_ids_list) < 0 or max(node_ids_list) >= raw_adjacency.shape[0]:
        raise ValueError("subgraph node_ids are outside the original graph")
    node_ids = torch.tensor(node_ids_list, dtype=torch.long)
    local_lookup = {node: index for index, node in enumerate(node_ids_list)}

    exported_names = exported.get("node_names")
    expected_names = [raw_names[index] for index in node_ids_list]
    if exported_names != expected_names:
        raise ValueError("subgraph node_names do not align with node_ids")
    exported_communities = exported.get("community_labels")
    expected_communities = [int(raw_communities[index]) for index in node_ids_list]
    if exported_communities != expected_communities:
        raise ValueError("subgraph community labels do not align with node_ids")
    communities = torch.tensor(expected_communities, dtype=torch.long)

    edges = exported.get("edge_index")
    weights = exported.get("original_edge_weights")
    if not isinstance(edges, list) or not isinstance(weights, list) or not edges:
        raise ValueError("subgraph edges must be non-empty lists")
    if len(edges) != len(weights):
        raise ValueError("edge_index and original_edge_weights differ in length")
    adjacency = torch.zeros(len(node_ids_list), len(node_ids_list), dtype=torch.float32)
    local_edges = []
    local_weights = []
    seen = set()
    threshold = raw_sample.edge_presence_threshold
    for edge, weight_value in zip(edges, weights):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("each edge_index item must contain two endpoints")
        left_global, right_global = edge
        if left_global not in local_lookup or right_global not in local_lookup:
            raise ValueError("edge endpoint is not present in node_ids")
        if left_global == right_global:
            raise ValueError("hard subgraph contains a self loop")
        canonical = tuple(sorted((left_global, right_global)))
        if canonical in seen:
            raise ValueError("hard subgraph contains a duplicate undirected edge")
        seen.add(canonical)
        weight = float(weight_value)
        raw_weight = float(raw_adjacency[left_global, right_global])
        if abs(raw_weight - weight) > 1e-6:
            raise ValueError("exported edge weight differs from original graph")
        if abs(weight) <= threshold:
            raise ValueError("exported edge does not satisfy edge presence threshold")
        left_local = local_lookup[left_global]
        right_local = local_lookup[right_global]
        adjacency[left_local, right_local] = weight
        adjacency[right_local, left_local] = weight
        local_edges.append((left_local, right_local))
        local_weights.append(weight)

    edge_mask = adjacency.abs() > threshold
    edge_mask.fill_diagonal_(False)
    node_features = feature_builder.build_static_node_features(
        adjacency, communities, threshold
    )
    edge_index = torch.tensor(local_edges, dtype=torch.long).transpose(0, 1).contiguous()
    edge_weight = torch.tensor(local_weights, dtype=torch.float32)
    return BaselineSubgraph(
        node_ids=node_ids,
        node_names=tuple(expected_names),
        community_labels=communities,
        adjacency=adjacency,
        edge_mask=edge_mask,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


class BaselineHardSubgraphDataset(Dataset):
    """Join a frozen hard-subgraph manifest with the original graph Dataset."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        verify_exports: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        payload, records = read_baseline_manifest(
            manifest_path, self.project_root, verify_exports=verify_exports
        )
        protocol_path = resolve_manifest_path(payload["data_protocol"], self.project_root)
        protocol = validate_data_protocol(protocol_path, self.project_root)
        split = str(payload["split"])
        paths = protocol["paths"]
        self.raw_dataset = GraphSequenceDataset(
            self.project_root / paths["dataset_root"],
            self.project_root / paths["sample_index_csv"],
            self.project_root / paths["splits_csv"],
            split=split,
            edge_presence_threshold=float(protocol["edge_presence_threshold"]),
        )
        raw_indices = {
            assignment.sample_key: index
            for index, assignment in enumerate(self.raw_dataset.assignments)
        }
        if set(raw_indices) != {record.sample_key for record in records}:
            raise ValueError("manifest and raw Dataset sample sets differ")
        self.records = records
        self.raw_indices = raw_indices
        self.split = split
        self.feature_builder = GraphFeatureBuilder()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BaselineSequenceSample:
        record = self.records[index]
        raw_sample = self.raw_dataset[self.raw_indices[record.sample_key]]
        export_path = resolve_manifest_path(
            record.hard_subgraph_json, self.project_root
        )
        payload = _read_json(export_path)
        _validate_export_metadata(payload, record, raw_sample)
        timepoints = payload.get("timepoints")
        if len(timepoints) != raw_sample.num_timepoints:
            raise ValueError("hard export and original graph timepoint counts differ")
        windows = []
        for expected_index, timepoint in enumerate(timepoints):
            if timepoint.get("time_index") != expected_index or not bool(
                timepoint.get("time_mask", False)
            ):
                raise ValueError("invalid timepoint index or mask")
            exported_subgraphs = timepoint.get("subgraphs")
            if not exported_subgraphs:
                raise ValueError("effective timepoint contains no hard subgraphs")
            subgraphs = tuple(
                _local_subgraph(
                    exported,
                    raw_sample,
                    expected_index,
                    self.feature_builder,
                )
                for exported in exported_subgraphs
            )
            windows.append(BaselineWindow(expected_index, subgraphs))
        if len(windows) != record.timepoint_count:
            raise ValueError("manifest timepoint count mismatch")
        if sum(len(window.subgraphs) for window in windows) != record.subgraph_count:
            raise ValueError("manifest subgraph count mismatch")
        return BaselineSequenceSample(
            sample_key=raw_sample.sample_key,
            sample_id=raw_sample.sample_id,
            site=raw_sample.site,
            subject_id=raw_sample.subject_id,
            session_id=raw_sample.session_id,
            label=raw_sample.label,
            split=raw_sample.split,
            windows=tuple(windows),
        )


def baseline_list_collate(
    samples: Sequence[BaselineSequenceSample],
) -> Tuple[BaselineSequenceSample, ...]:
    if not samples:
        raise ValueError("cannot collate an empty baseline batch")
    return tuple(samples)


def iter_subgraphs(sample: BaselineSequenceSample) -> Iterator[BaselineSubgraph]:
    for window in sample.windows:
        for subgraph in window.subgraphs:
            yield subgraph
