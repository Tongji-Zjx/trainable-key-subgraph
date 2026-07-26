"""Immutable per-sample artifacts for D3-B temporal residual models."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path

import torch


DUAL_TEMPORAL_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DualTemporalVariationRecord:
    sample_key: str
    label: int
    split: str
    window_count: int
    transition_values: torch.Tensor
    transition_mask: torch.Tensor
    base_logits: torch.Tensor
    protocol_sha256: str
    selector_checkpoint_sha256: str
    exact_head_checkpoint_sha256: str
    sgw_scaler_sha256: str
    exact_manifest_sha256: str
    selection_mode: str
    selection_seed: int

    @property
    def valid_transition_count(self) -> int:
        return int(self.transition_mask.sum().item())


def validate_dual_temporal_record(
    record: DualTemporalVariationRecord,
) -> None:
    if not isinstance(record, DualTemporalVariationRecord):
        raise ValueError("invalid dual temporal record type")
    if not record.sample_key or record.split not in (
        "train",
        "validation",
        "test",
    ):
        raise ValueError("dual temporal identity is invalid")
    if int(record.label) not in (0, 1) or int(record.window_count) < 1:
        raise ValueError("dual temporal label/window count is invalid")
    expected_transitions = max(0, int(record.window_count) - 1)
    if tuple(record.transition_values.shape) != (
        expected_transitions,
        16,
    ):
        raise ValueError("dual temporal values must have shape [M-1,16]")
    if tuple(record.transition_mask.shape) != (expected_transitions,):
        raise ValueError("dual temporal mask must have shape [M-1]")
    if record.transition_mask.dtype != torch.bool:
        raise ValueError("dual temporal mask must be boolean")
    if tuple(record.base_logits.shape) != (2,):
        raise ValueError("dual temporal base logits must have shape [2]")
    if not bool(torch.isfinite(record.transition_values).all()) or not bool(
        torch.isfinite(record.base_logits).all()
    ):
        raise ValueError("dual temporal tensors must be finite")
    invalid = ~record.transition_mask
    if bool(invalid.any()) and not bool(
        (record.transition_values[invalid] == 0.0).all()
    ):
        raise ValueError("invalid temporal transitions must be zero")
    hashes = (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.exact_head_checkpoint_sha256,
        record.sgw_scaler_sha256,
        record.exact_manifest_sha256,
    )
    if any(not str(value) for value in hashes):
        raise ValueError("dual temporal provenance hashes are required")
    if record.selection_mode != "learned":
        raise ValueError("temporal residual requires learned selection")


def save_dual_temporal_record(
    record: DualTemporalVariationRecord,
    path: Path,
    overwrite: bool = False,
) -> Path:
    validate_dual_temporal_record(record)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("dual temporal artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": DUAL_TEMPORAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "dual_d3b_temporal_variation_record",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_dual_temporal_record(
    path: Path,
) -> DualTemporalVariationRecord:
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
        DUAL_TEMPORAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported dual temporal artifact schema")
    if payload.get("artifact_type") != (
        "dual_d3b_temporal_variation_record"
    ):
        raise ValueError("unexpected dual temporal artifact")
    record = payload.get("record")
    validate_dual_temporal_record(record)
    return record
