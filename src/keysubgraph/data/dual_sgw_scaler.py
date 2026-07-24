"""Train-only standardization for cached 34-D exact SGW features."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

DUAL_SGW_SCALER_SCHEMA_VERSION = 1


class DualSGWStandardizer(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        sample_count: int,
        protocol_sha256: str,
        selector_checkpoint_sha256: str,
        epsilon: float = 1.0e-8,
        selection_mode: str = "",
        selection_seed: int = -1,
    ) -> None:
        super().__init__()
        if tuple(mean.shape) != (34,) or tuple(scale.shape) != (34,):
            raise ValueError("dual SGW scaler must contain 34-D vectors")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise ValueError("dual SGW scaler values must be finite")
        if bool((scale <= 0.0).any()):
            raise ValueError("dual SGW scales must be positive")
        if int(sample_count) < 1:
            raise ValueError("dual SGW scaler sample count must be positive")
        if epsilon <= 0.0:
            raise ValueError("dual SGW scaler epsilon must be positive")
        self.register_buffer("mean", mean.detach().to(torch.float32))
        self.register_buffer("scale", scale.detach().to(torch.float32))
        self.sample_count = int(sample_count)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(
            selector_checkpoint_sha256
        )
        self.epsilon = float(epsilon)
        self.selection_mode = str(selection_mode)
        self.selection_seed = int(selection_seed)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 34:
            raise ValueError("dual SGW values must end in dimension 34")
        return (
            values - self.mean.to(values)
        ) / self.scale.to(values)


def fit_dual_sgw_standardizer(
    records: Sequence[Any],
    epsilon: float = 1.0e-8,
) -> DualSGWStandardizer:
    if not records:
        raise ValueError("cannot fit a scaler from no SGW records")
    if any(record.split != "train" for record in records):
        raise ValueError("dual SGW scaler may be fitted from train only")
    sample_keys = [record.sample_key for record in records]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("dual SGW scaler received duplicate samples")
    protocols = {record.protocol_sha256 for record in records}
    selectors = {
        record.selector_checkpoint_sha256 for record in records
    }
    modes = {record.selection_mode for record in records}
    seeds = {int(record.selection_seed) for record in records}
    if (
        len(protocols) != 1
        or len(selectors) != 1
        or len(modes) != 1
        or len(seeds) != 1
    ):
        raise ValueError("dual SGW scaler records have mixed provenance")
    values = torch.stack(
        [
            record.representation.detach().to(torch.float64)
            for record in records
        ],
        dim=0,
    )
    if tuple(values.shape[1:]) != (34,):
        raise ValueError("dual SGW training features are not 34-D")
    mean = values.mean(dim=0)
    variance = (values - mean).square().mean(dim=0)
    scale = torch.sqrt(variance + float(epsilon))
    return DualSGWStandardizer(
        mean=mean,
        scale=scale,
        sample_count=len(records),
        protocol_sha256=next(iter(protocols)),
        selector_checkpoint_sha256=next(iter(selectors)),
        epsilon=epsilon,
        selection_mode=next(iter(modes)),
        selection_seed=next(iter(seeds)),
    )


def save_dual_sgw_standardizer(
    scaler: DualSGWStandardizer,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("dual SGW scaler already exists")
    payload = {
        "schema_version": DUAL_SGW_SCALER_SCHEMA_VERSION,
        "artifact_type": "dual_stse_sgw_train_only_scaler",
        "fit_split": "train",
        "dimension": 34,
        "sample_count": scaler.sample_count,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": (
            scaler.selector_checkpoint_sha256
        ),
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "epsilon": scaler.epsilon,
        "mean": scaler.mean.detach().cpu().tolist(),
        "scale": scaler.scale.detach().cpu().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_dual_sgw_standardizer(path: Path) -> DualSGWStandardizer:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != DUAL_SGW_SCALER_SCHEMA_VERSION:
        raise ValueError("unsupported dual SGW scaler schema")
    if payload.get("artifact_type") != (
        "dual_stse_sgw_train_only_scaler"
    ):
        raise ValueError("unexpected dual SGW scaler artifact")
    if payload.get("fit_split") != "train" or payload.get("dimension") != 34:
        raise ValueError("dual SGW scaler violates the train-only contract")
    return DualSGWStandardizer(
        mean=torch.tensor(payload["mean"], dtype=torch.float32),
        scale=torch.tensor(payload["scale"], dtype=torch.float32),
        sample_count=int(payload["sample_count"]),
        protocol_sha256=payload["protocol_sha256"],
        selector_checkpoint_sha256=payload[
            "selector_checkpoint_sha256"
        ],
        epsilon=float(payload["epsilon"]),
        selection_mode=payload.get("selection_mode", ""),
        selection_seed=int(payload.get("selection_seed", -1)),
    )
