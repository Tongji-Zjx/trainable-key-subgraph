"""Immutable artifacts, train-only scaling and list batching for multi-view S/V/G."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.multiview_critical import (
    CriticalObjectFeatures,
    CriticalTransitionFeatures,
    CriticalWindowFeatures,
    MultiViewCriticalBatch,
    MultiViewCriticalSampleFeatures,
)


MULTIVIEW_ARTIFACT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class MultiViewCriticalRecord:
    sample_id: str
    subject_id: str
    site: str
    split: str
    features: MultiViewCriticalSampleFeatures
    protocol_sha256: str
    selector_checkpoint_sha256: str
    feature_schema_sha256: str
    precompute_seconds: float = 0.0
    peak_memory_mib: float = 0.0
    feature_config_json: str = "{}"
    git_commit: str = "unknown"


@dataclass(frozen=True)
class MultiViewCriticalScaler:
    node_mean: torch.Tensor
    node_scale: torch.Tensor
    edge_mean: torch.Tensor
    edge_scale: torch.Tensor
    spectral_mean: torch.Tensor
    spectral_scale: torch.Tensor
    stable_mean: torch.Tensor
    stable_scale: torch.Tensor
    q_mean: torch.Tensor
    q_scale: torch.Tensor
    delta_mean: torch.Tensor
    delta_scale: torch.Tensor
    train_manifest_sha256: str
    protocol_sha256: str
    selector_checkpoint_sha256: str
    feature_schema_sha256: str


def multiview_filename(sample_key):
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def _finite_shape(value, shape):
    return tuple(value.shape) == tuple(shape) and bool(torch.isfinite(value).all())


def validate_multiview_record(record):
    if not isinstance(record, MultiViewCriticalRecord):
        raise ValueError("invalid multi-view record type")
    sample = record.features
    if (
        not sample.sample_key
        or int(sample.label) not in (0, 1)
        or record.split not in ("train", "validation", "test")
        or not record.sample_id
    ):
        raise ValueError("invalid multi-view sample identity")
    count = len(sample.hard_windows)
    if count < 1 or tuple(sample.window_mask.shape) != (count,):
        raise ValueError("multi-view window mask mismatch")
    if tuple(sample.transitions) and len(sample.transitions) != max(0, count - 1):
        raise ValueError("multi-view transition count mismatch")
    if tuple(sample.transition_mask.shape) != (max(0, count - 1),):
        raise ValueError("multi-view transition mask mismatch")
    if not _finite_shape(sample.stable_static, (28,)):
        raise ValueError("multi-view stable feature schema mismatch")
    for index, window in enumerate(sample.hard_windows):
        if (window is not None) != bool(sample.window_mask[index]):
            raise ValueError("multi-view hard window/mask mismatch")
        if window is None:
            continue
        nodes = int(window.adjacency.shape[0])
        checks = (
            _finite_shape(window.node_features, (nodes, 15)),
            _finite_shape(window.spectral_features, (nodes, 9)),
            _finite_shape(window.adjacency, (nodes, nodes)),
            _finite_shape(window.edge_features, (nodes, nodes, 6)),
            _finite_shape(window.q_target, (16,)),
            tuple(window.communities.shape) == (nodes,),
            len(window.objects) >= 1,
            tuple(window.object_coupling.shape) == (len(window.objects), len(window.objects), 2),
        )
        if not all(checks):
            raise ValueError("multi-view hard window schema mismatch")
        for item in window.objects:
            object_nodes = int(item.adjacency.shape[0])
            if not all((
                _finite_shape(item.node_features, (object_nodes, 15)),
                _finite_shape(item.spectral_features, (object_nodes, 9)),
                _finite_shape(item.adjacency, (object_nodes, object_nodes)),
                _finite_shape(item.edge_features, (object_nodes, object_nodes, 6)),
                tuple(item.communities.shape) == (object_nodes,),
            )):
                raise ValueError("multi-view critical object schema mismatch")
    if sample.full_windows and len(sample.full_windows) != count:
        raise ValueError("multi-view full/hard window count mismatch")
    for index, window in enumerate(sample.full_windows):
        if window is None:
            continue
        nodes = int(window.adjacency.shape[0])
        if not all(
            (
                _finite_shape(window.node_features, (nodes, 15)),
                _finite_shape(window.spectral_features, (nodes, 9)),
                _finite_shape(window.adjacency, (nodes, nodes)),
                _finite_shape(window.edge_features, (nodes, nodes, 6)),
                tuple(window.communities.shape) == (nodes,),
                len(window.objects) == 0,
                tuple(window.object_coupling.shape) == (0, 0, 2),
            )
        ):
            raise ValueError("multi-view full window schema mismatch")
    for index, transition in enumerate(sample.transitions):
        if (transition is not None) != bool(sample.transition_mask[index]):
            raise ValueError("multi-view transition/mask mismatch")
        if transition is None:
            continue
        left = sample.hard_windows[transition.source_index]
        right = sample.hard_windows[transition.target_index]
        if (
            tuple(transition.object_cost.shape) != (len(left.objects), len(right.objects))
            or tuple(transition.transport_plan.shape) != tuple(transition.object_cost.shape)
            or not _finite_shape(transition.delta_q_target, (18,))
            or transition.delta_time <= 0.0
        ):
            raise ValueError("multi-view transition schema mismatch")
    if any(
        not str(value)
        for value in (
            record.protocol_sha256,
            record.selector_checkpoint_sha256,
            record.feature_schema_sha256,
        )
    ):
        raise ValueError("multi-view provenance is incomplete")
    if record.precompute_seconds < 0.0 or record.peak_memory_mib < 0.0:
        raise ValueError("multi-view precompute diagnostics are invalid")
    try:
        feature_config = json.loads(record.feature_config_json)
    except (TypeError, ValueError):
        raise ValueError("multi-view feature configuration is invalid")
    if not isinstance(feature_config, dict) or not str(record.git_commit):
        raise ValueError("multi-view feature configuration provenance is incomplete")


def save_multiview_record(record, path, overwrite=False):
    validate_multiview_record(record)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("multi-view artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": MULTIVIEW_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "theory_guided_multiview_critical_record",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_multiview_record(path):
    try:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if (
        payload.get("schema_version") != MULTIVIEW_ARTIFACT_SCHEMA_VERSION
        or payload.get("artifact_type") != "theory_guided_multiview_critical_record"
    ):
        raise ValueError("unsupported multi-view artifact")
    record = payload.get("record")
    validate_multiview_record(record)
    return record


def write_multiview_manifest(paths, output_path, project_root, overwrite=False):
    # Keep the logical project path here.  Resolving either side would follow an
    # ``outputs`` symlink and make an in-project artifact appear to live outside
    # an isolated git worktree.
    output_path = Path(output_path).absolute()
    project_root = Path(project_root).absolute()
    if output_path.exists() and not overwrite:
        raise FileExistsError("multi-view manifest already exists")
    records, provenance, split = [], None, None
    for path in paths:
        path = Path(path).absolute()
        record = load_multiview_record(path)
        current = (
            record.protocol_sha256,
            record.selector_checkpoint_sha256,
            record.feature_schema_sha256,
            record.feature_config_json,
            record.git_commit,
        )
        if provenance is None:
            provenance, split = current, record.split
        if current != provenance or record.split != split:
            raise ValueError("multi-view records have incompatible provenance")
        records.append(
            {
                "sample_key": record.features.sample_key,
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "site": record.site,
                "label": int(record.features.label),
                "feature_path": path.relative_to(project_root).as_posix(),
                "feature_sha256": file_sha256(path),
                "precompute_seconds": float(record.precompute_seconds),
                "peak_memory_mib": float(record.peak_memory_mib),
            }
        )
    records.sort(key=lambda item: item["sample_key"])
    if not records or len({item["sample_key"] for item in records}) != len(records):
        raise ValueError("multi-view manifest is empty or duplicated")
    payload = {
        "schema_version": MULTIVIEW_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "theory_guided_multiview_critical_manifest",
        "split": split,
        "sample_count": len(records),
        "protocol_sha256": provenance[0],
        "selector_checkpoint_sha256": provenance[1],
        "feature_schema_sha256": provenance[2],
        "feature_config": json.loads(provenance[3]),
        "git_commit": provenance[4],
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def read_multiview_manifest(path, project_root):
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != "theory_guided_multiview_critical_manifest":
        raise ValueError("unsupported multi-view manifest")
    records = []
    for row in payload.get("records", []):
        feature_path = project_root / row["feature_path"]
        if file_sha256(feature_path) != row["feature_sha256"]:
            raise ValueError("multi-view artifact hash mismatch")
        record = load_multiview_record(feature_path)
        if (
            record.features.sample_key != row["sample_key"]
            or int(record.features.label) != int(row["label"])
            or record.split != payload["split"]
            or record.protocol_sha256 != payload["protocol_sha256"]
            or record.selector_checkpoint_sha256 != payload["selector_checkpoint_sha256"]
            or record.feature_schema_sha256 != payload["feature_schema_sha256"]
            or json.loads(record.feature_config_json) != payload.get("feature_config", {})
            or record.git_commit != payload.get("git_commit")
        ):
            raise ValueError("multi-view manifest record mismatch")
        records.append(record)
    if len(records) != int(payload.get("sample_count", -1)):
        raise ValueError("multi-view manifest count mismatch")
    return payload, records


def _mean_scale(values):
    tensor = torch.cat(tuple(values), dim=0).to(torch.float32)
    mean = tensor.mean(dim=0)
    scale = tensor.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mean, scale


def fit_multiview_scaler(records, train_manifest_sha256):
    if not records or any(record.split != "train" for record in records):
        raise ValueError("multi-view scaler requires train records only")
    provenance = {
        (record.protocol_sha256, record.selector_checkpoint_sha256, record.feature_schema_sha256)
        for record in records
    }
    if len(provenance) != 1:
        raise ValueError("multi-view train records have incompatible provenance")
    nodes, edges, spectral, stable, q_values, delta_values = [], [], [], [], [], []
    for record in records:
        sample = record.features
        stable.append(sample.stable_static.reshape(1, -1))
        for window in sample.hard_windows:
            if window is None:
                continue
            nodes.append(window.node_features)
            edges.append(window.edge_features.reshape(-1, 6))
            spectral.append(window.spectral_features)
            q_values.append(window.q_target.reshape(1, -1))
        for window in sample.full_windows:
            if window is not None:
                nodes.append(window.node_features)
                edges.append(window.edge_features.reshape(-1, 6))
                spectral.append(window.spectral_features)
        for transition in sample.transitions:
            if transition is not None:
                delta_values.append(transition.delta_q_target.reshape(1, -1))
    if not all((nodes, edges, spectral, stable, q_values, delta_values)):
        raise ValueError("multi-view scaler received incomplete training features")
    node_mean, node_scale = _mean_scale(nodes)
    edge_mean, edge_scale = _mean_scale(edges)
    spectral_mean, spectral_scale = _mean_scale(spectral)
    stable_mean, stable_scale = _mean_scale(stable)
    q_mean, q_scale = _mean_scale(q_values)
    delta_mean, delta_scale = _mean_scale(delta_values)
    protocol, selector, schema = next(iter(provenance))
    return MultiViewCriticalScaler(
        node_mean, node_scale, edge_mean, edge_scale, spectral_mean, spectral_scale,
        stable_mean, stable_scale, q_mean, q_scale, delta_mean, delta_scale,
        str(train_manifest_sha256), protocol, selector, schema,
    )


def save_multiview_scaler(scaler, path, overwrite=False):
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("multi-view scaler already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"schema_version": 1, "artifact_type": "multiview_train_scaler", "scaler": scaler}, str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_multiview_scaler(path):
    try:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if payload.get("artifact_type") != "multiview_train_scaler":
        raise ValueError("unsupported multi-view scaler")
    scaler = payload.get("scaler")
    if not isinstance(scaler, MultiViewCriticalScaler):
        raise ValueError("invalid multi-view scaler payload")
    return scaler


def _standardize(value, mean, scale):
    return (value.to(torch.float32) - mean) / scale


def _standardize_object(item, scaler):
    return CriticalObjectFeatures(
        _standardize(item.node_features, scaler.node_mean, scaler.node_scale),
        _standardize(item.spectral_features, scaler.spectral_mean, scaler.spectral_scale),
        item.adjacency.to(torch.float32),
        _standardize(item.edge_features, scaler.edge_mean, scaler.edge_scale),
        item.communities.clone(),
        item.union_node_indices.clone(), float(item.mass),
    )


def _standardize_window(window, scaler, use_q=True):
    return CriticalWindowFeatures(
        _standardize(window.node_features, scaler.node_mean, scaler.node_scale),
        _standardize(window.spectral_features, scaler.spectral_mean, scaler.spectral_scale),
        window.adjacency.to(torch.float32),
        _standardize(window.edge_features, scaler.edge_mean, scaler.edge_scale),
        window.communities.clone(),
        _standardize(window.q_target, scaler.q_mean, scaler.q_scale) if use_q else window.q_target.to(torch.float32),
        tuple(_standardize_object(item, scaler) for item in window.objects),
        window.object_coupling.to(torch.float32), float(window.time_start),
    )


def standardize_multiview_sample(sample, scaler):
    transitions = []
    for item in sample.transitions:
        if item is None:
            transitions.append(None)
        else:
            transitions.append(
                CriticalTransitionFeatures(
                    item.source_index, item.target_index,
                    item.object_cost.to(torch.float32), item.transport_plan.to(torch.float32),
                    _standardize(item.delta_q_target, scaler.delta_mean, scaler.delta_scale),
                    item.delta_time, item.solver_converged,
                )
            )
    return MultiViewCriticalSampleFeatures(
        sample.sample_key, int(sample.label),
        _standardize(sample.stable_static, scaler.stable_mean, scaler.stable_scale),
        tuple(_standardize_window(item, scaler) if item is not None else None for item in sample.hard_windows),
        tuple(_standardize_window(item, scaler, use_q=False) if item is not None else None for item in sample.full_windows),
        tuple(transitions), sample.window_mask.clone(), sample.transition_mask.clone(),
    )


class MultiViewCriticalDataset(Dataset):
    def __init__(self, project_root, manifest_path, scaler_path, max_samples=None):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest, records = read_multiview_manifest(self.manifest_path, self.project_root)
        self.scaler = load_multiview_scaler(scaler_path)
        if (
            self.scaler.protocol_sha256 != self.manifest["protocol_sha256"]
            or self.scaler.selector_checkpoint_sha256 != self.manifest["selector_checkpoint_sha256"]
            or self.scaler.feature_schema_sha256 != self.manifest["feature_schema_sha256"]
        ):
            raise ValueError("multi-view dataset/scaler provenance mismatch")
        if self.manifest["split"] == "train" and self.scaler.train_manifest_sha256 != file_sha256(self.manifest_path):
            raise ValueError("multi-view train scaler manifest mismatch")
        selected = records if max_samples is None else records[: int(max_samples)]
        if not selected:
            raise ValueError("multi-view dataset cannot be empty")
        self.samples = tuple(standardize_multiview_sample(item.features, self.scaler) for item in selected)

    @property
    def split(self):
        return self.manifest["split"]

    @property
    def labels(self):
        return tuple(item.label for item in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_multiview(samples):
    if not samples:
        raise ValueError("cannot collate an empty multi-view batch")
    return MultiViewCriticalBatch(tuple(samples))


def _seed_worker(worker_id):
    del worker_id
    seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(seed)
    np.random.seed(seed)


def create_multiview_loader(dataset, batch_size, seed, shuffle, num_workers=0, pin_memory=False):
    if dataset.split != "train" and shuffle:
        raise ValueError("multi-view evaluation loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers),
        pin_memory=bool(pin_memory), collate_fn=collate_multiview,
        worker_init_fn=_seed_worker if num_workers else None, generator=generator,
    )


class PairedMultiViewAuthorDataset(Dataset):
    """Align a critical artifact dataset with the raw author branch by key."""

    def __init__(self, critical_dataset, graph_dataset):
        self.critical_dataset = critical_dataset
        self.graph_dataset = graph_dataset
        assignments = getattr(graph_dataset, "assignments", ())
        lookup = {
            str(item.sample_key): index for index, item in enumerate(assignments)
        }
        critical_keys = tuple(item.sample_key for item in critical_dataset.samples)
        if any(key not in lookup for key in critical_keys):
            raise ValueError("author raw dataset is missing critical samples")
        self.graph_indices = tuple(lookup[key] for key in critical_keys)
        self.split = critical_dataset.split

    @property
    def labels(self):
        return self.critical_dataset.labels

    def __len__(self):
        return len(self.critical_dataset)

    def __getitem__(self, index):
        critical = self.critical_dataset[index]
        graph = self.graph_dataset[self.graph_indices[index]]
        if graph.sample_key != critical.sample_key or int(graph.label) != int(critical.label):
            raise ValueError("critical/author sample alignment failed")
        return critical, graph


def collate_multiview_author(samples):
    if not samples:
        raise ValueError("cannot collate an empty multi-view/author batch")
    critical, graphs = zip(*samples)
    return MultiViewCriticalBatch(tuple(critical)), GraphSequenceBatch(tuple(graphs))


def create_multiview_author_loader(dataset, batch_size, seed, shuffle, num_workers=0, pin_memory=False):
    if dataset.split != "train" and shuffle:
        raise ValueError("paired evaluation loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers),
        pin_memory=bool(pin_memory), collate_fn=collate_multiview_author,
        worker_init_fn=_seed_worker if num_workers else None, generator=generator,
    )
