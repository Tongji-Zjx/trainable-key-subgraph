"""Immutable Stage-1 edge-aware neural artifacts."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from keysubgraph.features.sv_hard_graph_features import SV_NODE_FEATURE_DIM
from keysubgraph.features.theory_neural_features import THEORY_EDGE_FEATURE_DIM
from keysubgraph.theory.sgw_core_features import SGW_CORE_DIM, SGW_QUANTILE_DIM


THEORY_NEURAL_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TheoryNeuralWindowRecord:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    spectral_quantiles: torch.Tensor
    communities: torch.Tensor
    node_ids: Tuple[str, ...]
    time_start: float


@dataclass(frozen=True)
class TheoryNeuralRecord:
    sample_key: str
    sample_id: str
    subject_id: str
    site: str
    label: int
    split: str
    windows: Tuple[Optional[TheoryNeuralWindowRecord], ...]
    window_mask: torch.Tensor
    transition_features: torch.Tensor
    transition_mask: torch.Tensor
    gw_solver_converged: Tuple[bool, ...]
    protocol_sha256: str
    selector_checkpoint_sha256: str
    selection_mode: str
    selection_seed: int
    feature_schema_sha256: str


def validate_theory_neural_record(record: TheoryNeuralRecord) -> None:
    if not isinstance(record, TheoryNeuralRecord):
        raise ValueError("invalid Stage-1 artifact type")
    if (
        not record.sample_key
        or not record.sample_id
        or int(record.label) not in (0, 1)
        or record.split not in ("train", "validation", "test")
    ):
        raise ValueError("invalid Stage-1 sample identity")
    count = len(record.windows)
    if count < 1 or tuple(record.window_mask.shape) != (count,):
        raise ValueError("Stage-1 window mask mismatch")
    if tuple(record.transition_features.shape) != (
        max(0, count - 1),
        SGW_CORE_DIM,
    ) or tuple(record.transition_mask.shape) != (max(0, count - 1),):
        raise ValueError("Stage-1 transition target mismatch")
    if record.window_mask.dtype != torch.bool or record.transition_mask.dtype != torch.bool:
        raise ValueError("Stage-1 masks must be boolean")
    if not bool(torch.isfinite(record.transition_features).all()):
        raise ValueError("Stage-1 transitions contain non-finite values")
    valid_count = 0
    previous_time = None
    for index, window in enumerate(record.windows):
        if (window is not None) != bool(record.window_mask[index]):
            raise ValueError("Stage-1 window payload and mask disagree")
        if window is None:
            continue
        valid_count += 1
        nodes = int(window.node_features.shape[0])
        shapes = (
            tuple(window.node_features.shape) == (nodes, SV_NODE_FEATURE_DIM),
            tuple(window.adjacency.shape) == (nodes, nodes),
            tuple(window.edge_features.shape)
            == (nodes, nodes, THEORY_EDGE_FEATURE_DIM),
            tuple(window.spectral_quantiles.shape) == (SGW_QUANTILE_DIM,),
            tuple(window.communities.shape) == (nodes,),
            len(window.node_ids) == nodes,
            len(set(window.node_ids)) == nodes,
        )
        if nodes < 1 or not all(shapes):
            raise ValueError("Stage-1 window tensor schema mismatch")
        tensors = (
            window.node_features,
            window.adjacency,
            window.edge_features,
            window.spectral_quantiles,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Stage-1 window contains non-finite values")
        if not torch.allclose(
            window.adjacency,
            window.adjacency.transpose(0, 1),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("Stage-1 adjacency must be symmetric")
        if previous_time is not None and float(window.time_start) <= previous_time:
            raise ValueError("Stage-1 valid window times must increase")
        previous_time = float(window.time_start)
    if valid_count < 1:
        raise ValueError("Stage-1 sample has no valid window")
    expected_transition = record.window_mask[:-1] & record.window_mask[1:]
    if not torch.equal(expected_transition.cpu(), record.transition_mask.cpu()):
        raise ValueError("Stage-1 transition mask is inconsistent")
    if len(record.gw_solver_converged) != int(record.transition_mask.sum()):
        raise ValueError("Stage-1 GW convergence flags are misaligned")
    provenance = (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.selection_mode,
        record.feature_schema_sha256,
    )
    if any(not str(value) for value in provenance):
        raise ValueError("Stage-1 provenance is incomplete")


def save_theory_neural_record(record, path, overwrite=False):
    validate_theory_neural_record(record)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Stage-1 artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": THEORY_NEURAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "svg_theory_guided_neural_record",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_theory_neural_record(path):
    try:
        payload = torch.load(
            str(Path(path).resolve()), map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if (
        payload.get("schema_version") != THEORY_NEURAL_ARTIFACT_SCHEMA_VERSION
        or payload.get("artifact_type") != "svg_theory_guided_neural_record"
    ):
        raise ValueError("unsupported Stage-1 artifact")
    record = payload.get("record")
    validate_theory_neural_record(record)
    return record
