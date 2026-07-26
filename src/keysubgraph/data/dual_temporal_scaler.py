"""Train-only standardization for valid 16-D temporal transitions."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .dual_temporal_artifact import DualTemporalVariationRecord


DUAL_TEMPORAL_SCALER_SCHEMA_VERSION = 1


class DualTemporalStandardizer(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        valid_transition_count: int,
        train_sample_count: int,
        train_manifest_sha256: str,
        protocol_sha256: str,
        selector_checkpoint_sha256: str,
        exact_head_checkpoint_sha256: str,
        sgw_scaler_sha256: str,
        exact_manifest_sha256: str,
        selection_mode: str,
        selection_seed: int,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if tuple(mean.shape) != (16,) or tuple(scale.shape) != (16,):
            raise ValueError("temporal scaler requires 16-D vectors")
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0.0).any())
        ):
            raise ValueError("temporal scaler values are invalid")
        if valid_transition_count < 1 or train_sample_count < 1:
            raise ValueError("temporal scaler counts must be positive")
        if epsilon <= 0.0:
            raise ValueError("temporal scaler epsilon must be positive")
        self.register_buffer("mean", mean.detach().to(torch.float32))
        self.register_buffer("scale", scale.detach().to(torch.float32))
        self.valid_transition_count = int(valid_transition_count)
        self.train_sample_count = int(train_sample_count)
        self.train_manifest_sha256 = str(train_manifest_sha256)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(
            selector_checkpoint_sha256
        )
        self.exact_head_checkpoint_sha256 = str(
            exact_head_checkpoint_sha256
        )
        self.sgw_scaler_sha256 = str(sgw_scaler_sha256)
        self.exact_manifest_sha256 = str(exact_manifest_sha256)
        self.selection_mode = str(selection_mode)
        self.selection_seed = int(selection_seed)
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 16:
            raise ValueError("temporal values must end in dimension 16")
        return (
            values - self.mean.to(values)
        ) / self.scale.to(values)


def _record_provenance(record):
    return (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.exact_head_checkpoint_sha256,
        record.sgw_scaler_sha256,
        record.exact_manifest_sha256,
        record.selection_mode,
        int(record.selection_seed),
    )


def fit_dual_temporal_standardizer(
    records: Sequence[DualTemporalVariationRecord],
    train_manifest_sha256: str,
    epsilon: float = 1.0e-8,
) -> DualTemporalStandardizer:
    if not records or not train_manifest_sha256:
        raise ValueError("temporal scaler requires train records/manifest")
    if any(record.split != "train" for record in records):
        raise ValueError("temporal scaler may be fitted from train only")
    keys = [record.sample_key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("temporal scaler received duplicate samples")
    provenance = {_record_provenance(record) for record in records}
    if len(provenance) != 1:
        raise ValueError("temporal scaler records have mixed provenance")
    valid = [
        record.transition_values[record.transition_mask].to(torch.float64)
        for record in records
        if bool(record.transition_mask.any())
    ]
    if not valid:
        raise ValueError("temporal scaler has no valid train transitions")
    values = torch.cat(valid, dim=0)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("temporal scaler train values are not [N,16]")
    mean = values.mean(dim=0)
    variance = (values - mean).square().mean(dim=0)
    scale = torch.sqrt(variance + float(epsilon))
    fields = next(iter(provenance))
    return DualTemporalStandardizer(
        mean=mean,
        scale=scale,
        valid_transition_count=int(values.shape[0]),
        train_sample_count=len(records),
        train_manifest_sha256=train_manifest_sha256,
        protocol_sha256=fields[0],
        selector_checkpoint_sha256=fields[1],
        exact_head_checkpoint_sha256=fields[2],
        sgw_scaler_sha256=fields[3],
        exact_manifest_sha256=fields[4],
        selection_mode=fields[5],
        selection_seed=fields[6],
        epsilon=epsilon,
    )


def save_dual_temporal_standardizer(
    scaler: DualTemporalStandardizer,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("temporal scaler already exists")
    payload = {
        "schema_version": DUAL_TEMPORAL_SCALER_SCHEMA_VERSION,
        "artifact_type": "dual_d3b_temporal_train_only_scaler",
        "fit_split": "train",
        "dimension": 16,
        "valid_transition_count": scaler.valid_transition_count,
        "train_sample_count": scaler.train_sample_count,
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": (
            scaler.selector_checkpoint_sha256
        ),
        "exact_head_checkpoint_sha256": (
            scaler.exact_head_checkpoint_sha256
        ),
        "sgw_scaler_sha256": scaler.sgw_scaler_sha256,
        "exact_manifest_sha256": scaler.exact_manifest_sha256,
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "epsilon": scaler.epsilon,
        "mean": scaler.mean.detach().cpu().tolist(),
        "scale": scaler.scale.detach().cpu().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_dual_temporal_standardizer(
    path: Path,
) -> DualTemporalStandardizer:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != DUAL_TEMPORAL_SCALER_SCHEMA_VERSION:
        raise ValueError("unsupported temporal scaler schema")
    if payload.get("artifact_type") != (
        "dual_d3b_temporal_train_only_scaler"
    ):
        raise ValueError("unexpected temporal scaler artifact")
    if payload.get("fit_split") != "train" or payload.get("dimension") != 16:
        raise ValueError("temporal scaler violates train-only contract")
    return DualTemporalStandardizer(
        mean=torch.tensor(payload["mean"], dtype=torch.float32),
        scale=torch.tensor(payload["scale"], dtype=torch.float32),
        valid_transition_count=int(payload["valid_transition_count"]),
        train_sample_count=int(payload["train_sample_count"]),
        train_manifest_sha256=payload["train_manifest_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        selector_checkpoint_sha256=payload[
            "selector_checkpoint_sha256"
        ],
        exact_head_checkpoint_sha256=payload[
            "exact_head_checkpoint_sha256"
        ],
        sgw_scaler_sha256=payload["sgw_scaler_sha256"],
        exact_manifest_sha256=payload["exact_manifest_sha256"],
        selection_mode=payload["selection_mode"],
        selection_seed=int(payload["selection_seed"]),
        epsilon=float(payload["epsilon"]),
    )
