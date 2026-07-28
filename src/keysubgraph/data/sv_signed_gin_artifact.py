"""Immutable frozen hard-graph artifacts for SV Signed-GIN experiments."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from keysubgraph.features.sv_hard_graph_features import (
    SV_NODE_FEATURE_DIM,
    SV_STATIC_FEATURE_DIM,
    SV_VARIATION_DIM,
)


SV_SIGNED_GIN_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SVSignedGINWindowRecord:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    time_start: float


@dataclass(frozen=True)
class SVSignedGINRecord:
    sample_key: str
    sample_id: str
    subject_id: str
    site: str
    label: int
    split: str
    windows: Tuple[Optional[SVSignedGINWindowRecord], ...]
    static_features: torch.Tensor
    variation: torch.Tensor
    window_mask: torch.Tensor
    transition_mask: torch.Tensor
    protocol_sha256: str
    selector_checkpoint_sha256: str
    selection_mode: str
    selection_seed: int

    @property
    def valid_window_count(self) -> int:
        return int(self.window_mask.sum().item())

    @property
    def valid_transition_count(self) -> int:
        return int(self.transition_mask.sum().item())


def validate_sv_signed_gin_record(record: SVSignedGINRecord) -> None:
    if not isinstance(record, SVSignedGINRecord):
        raise ValueError("invalid SV Signed-GIN record type")
    if (
        not record.sample_key
        or not record.sample_id
        or record.split not in ("train", "validation", "test")
        or int(record.label) not in (0, 1)
    ):
        raise ValueError("SV Signed-GIN record identity is invalid")
    if not record.windows:
        raise ValueError("SV Signed-GIN record has no time windows")
    if tuple(record.window_mask.shape) != (len(record.windows),):
        raise ValueError("SV window mask does not align with windows")
    if tuple(record.transition_mask.shape) != (
        max(0, len(record.windows) - 1),
    ):
        raise ValueError("SV transition mask does not align with windows")
    if (
        record.window_mask.dtype != torch.bool
        or record.transition_mask.dtype != torch.bool
    ):
        raise ValueError("SV masks must be boolean")
    if tuple(record.static_features.shape) != (
        SV_STATIC_FEATURE_DIM,
    ) or tuple(record.variation.shape) != (SV_VARIATION_DIM,):
        raise ValueError("SV sample summaries must be 28-D/16-D")
    if not bool(torch.isfinite(record.static_features).all()) or not bool(
        torch.isfinite(record.variation).all()
    ):
        raise ValueError("SV sample summaries contain non-finite values")
    if record.valid_window_count < 1:
        raise ValueError("SV Signed-GIN record has no valid hard windows")
    previous_time = None
    for index, window in enumerate(record.windows):
        expected_valid = bool(record.window_mask[index])
        if (window is not None) != expected_valid:
            raise ValueError("SV window payload and mask disagree")
        if window is None:
            continue
        if not isinstance(window, SVSignedGINWindowRecord):
            raise ValueError("invalid SV hard-window record type")
        node_count = int(window.node_features.shape[0])
        if tuple(window.node_features.shape) != (
            node_count,
            SV_NODE_FEATURE_DIM,
        ) or tuple(window.adjacency.shape) != (
            node_count,
            node_count,
        ):
            raise ValueError("SV hard-window tensors are misaligned")
        if node_count < 1:
            raise ValueError("SV valid hard window is empty")
        if not bool(torch.isfinite(window.node_features).all()) or not bool(
            torch.isfinite(window.adjacency).all()
        ):
            raise ValueError("SV hard-window tensors are non-finite")
        if not torch.allclose(
            window.adjacency,
            window.adjacency.transpose(0, 1),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("SV hard adjacency must be symmetric")
        if not bool(
            (torch.diagonal(window.adjacency) == 0.0).all()
        ):
            raise ValueError("SV hard adjacency must have zero diagonal")
        if previous_time is not None and float(window.time_start) <= previous_time:
            raise ValueError("SV valid window times must increase")
        previous_time = float(window.time_start)
    expected_transition = torch.tensor(
        [
            bool(record.window_mask[index])
            and bool(record.window_mask[index + 1])
            for index in range(max(0, len(record.windows) - 1))
        ],
        dtype=torch.bool,
    )
    if not torch.equal(record.transition_mask.cpu(), expected_transition):
        raise ValueError("SV transition mask is inconsistent with windows")
    provenance = (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.selection_mode,
    )
    if any(not str(value) for value in provenance):
        raise ValueError("SV provenance fields are required")
    if record.selection_mode not in ("learned", "full", "random"):
        raise ValueError("SV selection mode is invalid")


def save_sv_signed_gin_record(
    record: SVSignedGINRecord,
    path: Path,
    overwrite: bool = False,
) -> Path:
    validate_sv_signed_gin_record(record)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("SV Signed-GIN artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": SV_SIGNED_GIN_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "sv_hard_sgw_signed_gin_record",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_sv_signed_gin_record(path: Path) -> SVSignedGINRecord:
    try:
        payload = torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            str(Path(path).resolve()), map_location="cpu"
        )
    if payload.get("schema_version") != (
        SV_SIGNED_GIN_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported SV Signed-GIN artifact schema")
    if payload.get("artifact_type") != "sv_hard_sgw_signed_gin_record":
        raise ValueError("unexpected SV Signed-GIN artifact")
    record = payload.get("record")
    validate_sv_signed_gin_record(record)
    return record
