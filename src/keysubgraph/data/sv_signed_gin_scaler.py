"""Train-only node/static/variation scaling for SV Signed-GIN."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from keysubgraph.features.sv_hard_graph_features import (
    SV_NODE_FEATURE_DIM,
    SV_STATIC_FEATURE_DIM,
    SV_VARIATION_DIM,
)
from .sv_signed_gin_artifact import SVSignedGINRecord


SV_SIGNED_GIN_SCALER_SCHEMA_VERSION = 1


class SVSignedGINStandardizers(nn.Module):
    def __init__(
        self,
        node_mean: torch.Tensor,
        node_scale: torch.Tensor,
        static_mean: torch.Tensor,
        static_scale: torch.Tensor,
        variation_mean: torch.Tensor,
        variation_scale: torch.Tensor,
        train_sample_count: int,
        train_node_count: int,
        train_manifest_sha256: str,
        protocol_sha256: str,
        selector_checkpoint_sha256: str,
        selection_mode: str,
        selection_seed: int,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        expected = (
            (node_mean, SV_NODE_FEATURE_DIM),
            (node_scale, SV_NODE_FEATURE_DIM),
            (static_mean, SV_STATIC_FEATURE_DIM),
            (static_scale, SV_STATIC_FEATURE_DIM),
            (variation_mean, SV_VARIATION_DIM),
            (variation_scale, SV_VARIATION_DIM),
        )
        if any(tuple(value.shape) != (dimension,) for value, dimension in expected):
            raise ValueError("SV scaler dimensions are invalid")
        if any(not bool(torch.isfinite(value).all()) for value, _ in expected):
            raise ValueError("SV scaler contains non-finite values")
        if any(
            bool((value <= 0.0).any())
            for value in (node_scale, static_scale, variation_scale)
        ):
            raise ValueError("SV scaler scales must be positive")
        if train_sample_count < 1 or train_node_count < 1:
            raise ValueError("SV scaler counts must be positive")
        if epsilon <= 0.0:
            raise ValueError("SV scaler epsilon must be positive")
        self.register_buffer("node_mean", node_mean.to(torch.float32))
        self.register_buffer("node_scale", node_scale.to(torch.float32))
        self.register_buffer("static_mean", static_mean.to(torch.float32))
        self.register_buffer("static_scale", static_scale.to(torch.float32))
        self.register_buffer(
            "variation_mean", variation_mean.to(torch.float32)
        )
        self.register_buffer(
            "variation_scale", variation_scale.to(torch.float32)
        )
        self.train_sample_count = int(train_sample_count)
        self.train_node_count = int(train_node_count)
        self.train_manifest_sha256 = str(train_manifest_sha256)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(
            selector_checkpoint_sha256
        )
        self.selection_mode = str(selection_mode)
        self.selection_seed = int(selection_seed)
        self.epsilon = float(epsilon)

    def standardize_nodes(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != SV_NODE_FEATURE_DIM:
            raise ValueError("SV node values must end in dimension 15")
        return (
            values - self.node_mean.to(values)
        ) / self.node_scale.to(values)

    def standardize_static(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != SV_STATIC_FEATURE_DIM:
            raise ValueError("SV static values must end in dimension 28")
        return (
            values - self.static_mean.to(values)
        ) / self.static_scale.to(values)

    def standardize_variation(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != SV_VARIATION_DIM:
            raise ValueError("SV variation values must end in dimension 16")
        return (
            values - self.variation_mean.to(values)
        ) / self.variation_scale.to(values)


def _provenance(record: SVSignedGINRecord):
    return (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.selection_mode,
        int(record.selection_seed),
    )


def _mean_scale(values: torch.Tensor, epsilon: float):
    values = values.to(torch.float64)
    mean = values.mean(dim=0)
    variance = (values - mean).square().mean(dim=0)
    return mean, torch.sqrt(variance + float(epsilon))


def fit_sv_signed_gin_standardizers(
    records: Sequence[SVSignedGINRecord],
    train_manifest_sha256: str,
    epsilon: float = 1.0e-8,
) -> SVSignedGINStandardizers:
    if not records or not train_manifest_sha256:
        raise ValueError("SV scaler requires train records and manifest")
    if any(record.split != "train" for record in records):
        raise ValueError("SV scaler may be fitted from train only")
    keys = [record.sample_key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("SV scaler received duplicate train samples")
    provenance = {_provenance(record) for record in records}
    if len(provenance) != 1:
        raise ValueError("SV scaler records have mixed provenance")
    node_values = [
        window.node_features
        for record in records
        for window in record.windows
        if window is not None
    ]
    if not node_values:
        raise ValueError("SV scaler has no valid train nodes")
    nodes = torch.cat(node_values, dim=0)
    static = torch.stack(
        [record.static_features for record in records], dim=0
    )
    variation = torch.stack(
        [record.variation for record in records], dim=0
    )
    node_mean, node_scale = _mean_scale(nodes, epsilon)
    static_mean, static_scale = _mean_scale(static, epsilon)
    variation_mean, variation_scale = _mean_scale(
        variation, epsilon
    )
    fields = next(iter(provenance))
    return SVSignedGINStandardizers(
        node_mean=node_mean,
        node_scale=node_scale,
        static_mean=static_mean,
        static_scale=static_scale,
        variation_mean=variation_mean,
        variation_scale=variation_scale,
        train_sample_count=len(records),
        train_node_count=int(nodes.shape[0]),
        train_manifest_sha256=train_manifest_sha256,
        protocol_sha256=fields[0],
        selector_checkpoint_sha256=fields[1],
        selection_mode=fields[2],
        selection_seed=fields[3],
        epsilon=epsilon,
    )


def save_sv_signed_gin_standardizers(
    scaler: SVSignedGINStandardizers,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("SV Signed-GIN scaler already exists")
    payload = {
        "schema_version": SV_SIGNED_GIN_SCALER_SCHEMA_VERSION,
        "artifact_type": "sv_hard_sgw_signed_gin_train_only_scalers",
        "fit_split": "train",
        "dimensions": {
            "node": SV_NODE_FEATURE_DIM,
            "static": SV_STATIC_FEATURE_DIM,
            "variation": SV_VARIATION_DIM,
        },
        "train_sample_count": scaler.train_sample_count,
        "train_node_count": scaler.train_node_count,
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": (
            scaler.selector_checkpoint_sha256
        ),
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "epsilon": scaler.epsilon,
        "node_mean": scaler.node_mean.detach().cpu().tolist(),
        "node_scale": scaler.node_scale.detach().cpu().tolist(),
        "static_mean": scaler.static_mean.detach().cpu().tolist(),
        "static_scale": scaler.static_scale.detach().cpu().tolist(),
        "variation_mean": scaler.variation_mean.detach().cpu().tolist(),
        "variation_scale": scaler.variation_scale.detach().cpu().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_sv_signed_gin_standardizers(
    path: Path,
) -> SVSignedGINStandardizers:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != (
        SV_SIGNED_GIN_SCALER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported SV Signed-GIN scaler schema")
    if payload.get("artifact_type") != (
        "sv_hard_sgw_signed_gin_train_only_scalers"
    ) or payload.get("fit_split") != "train":
        raise ValueError("SV scaler violates the train-only contract")
    if payload.get("dimensions") != {
        "node": SV_NODE_FEATURE_DIM,
        "static": SV_STATIC_FEATURE_DIM,
        "variation": SV_VARIATION_DIM,
    }:
        raise ValueError("SV scaler schema dimensions are invalid")
    return SVSignedGINStandardizers(
        node_mean=torch.tensor(payload["node_mean"]),
        node_scale=torch.tensor(payload["node_scale"]),
        static_mean=torch.tensor(payload["static_mean"]),
        static_scale=torch.tensor(payload["static_scale"]),
        variation_mean=torch.tensor(payload["variation_mean"]),
        variation_scale=torch.tensor(payload["variation_scale"]),
        train_sample_count=int(payload["train_sample_count"]),
        train_node_count=int(payload["train_node_count"]),
        train_manifest_sha256=payload["train_manifest_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        selector_checkpoint_sha256=payload[
            "selector_checkpoint_sha256"
        ],
        selection_mode=payload["selection_mode"],
        selection_seed=int(payload["selection_seed"]),
        epsilon=float(payload["epsilon"]),
    )
