"""Canonical Stage-0 full/hard 18-D spectral--GW core features.

This module is deliberately model-free.  It wraps the existing exact
``SGWFeatureExtractor`` so Stage 0 and later auxiliary targets share one
quantile grid, signed Laplacian, heat-kernel metric and GW implementation.
"""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.theory.tg_features import (
    SGWFeatureExtractor,
    SGWSequenceFeatures,
    SGWTheoryFeatureConfig,
)


SGW_CORE_DIM = 18
SGW_QUANTILE_DIM = 16
SGW_STAGE0_SAMPLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SGWCoreConfig:
    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0
    spectral_quantile_grid: Tuple[float, ...] = tuple(
        0.05 + (0.90 / 15.0) * index for index in range(16)
    )
    spectral_w1_grid_size: int = 256
    gw_entropic_reg: float = 1.0e-2
    gw_max_iter: int = 100
    gw_sinkhorn_iter: int = 100
    gw_tolerance: float = 1.0e-7
    ground_metric: str = "euclidean_raw_18d"

    def __post_init__(self) -> None:
        if self.laplacian_eta <= 0.0 or self.diffusion_time <= 0.0:
            raise ValueError("Stage-0 eta and diffusion time must be positive")
        if len(self.spectral_quantile_grid) != SGW_QUANTILE_DIM:
            raise ValueError("Stage-0 requires exactly 16 spectral quantiles")
        if self.spectral_w1_grid_size < 2:
            raise ValueError("Stage-0 spectral W1 grid is too small")
        if (
            self.gw_entropic_reg <= 0.0
            or self.gw_max_iter < 1
            or self.gw_sinkhorn_iter < 1
            or self.gw_tolerance <= 0.0
        ):
            raise ValueError("Stage-0 GW configuration is invalid")
        if self.ground_metric != "euclidean_raw_18d":
            raise ValueError("Stage-0 primary ground metric is frozen to raw L2")

    def theory_config(self) -> SGWTheoryFeatureConfig:
        return SGWTheoryFeatureConfig(
            laplacian_eta=self.laplacian_eta,
            diffusion_time=self.diffusion_time,
            spectral_quantile_grid=self.spectral_quantile_grid,
            spectral_w1_grid_size=self.spectral_w1_grid_size,
            time_quantity="speed",
        )

    def build_extractor(self) -> SGWFeatureExtractor:
        return SGWFeatureExtractor(
            theory_config=self.theory_config(),
            gw_entropic_reg=self.gw_entropic_reg,
            gw_max_iter=self.gw_max_iter,
            gw_sinkhorn_iter=self.gw_sinkhorn_iter,
            gw_tolerance=self.gw_tolerance,
        )

    def schema_sha256(self) -> str:
        payload = {
            "artifact": "sgw_stage0_core",
            "schema_version": SGW_STAGE0_SAMPLE_SCHEMA_VERSION,
            "core_dim": SGW_CORE_DIM,
            "quantile_dim": SGW_QUANTILE_DIM,
            "config": asdict(self),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SGWCoreSequence:
    core: torch.Tensor
    window_quantiles: torch.Tensor
    window_mask: torch.Tensor
    transition_features: torch.Tensor
    transition_mask: torch.Tensor
    gw_solver_converged: Tuple[bool, ...]

    @property
    def valid_transition_count(self) -> int:
        return int(self.transition_mask.sum().item())


def _window_from_adjacency(
    adjacency: torch.Tensor,
    time_start: float,
    edge_presence_threshold: float,
) -> HardGraphWindow:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Stage-0 adjacency must be square")
    if adjacency.shape[0] < 1 or not bool(torch.isfinite(adjacency).all()):
        raise ValueError("Stage-0 adjacency is empty or non-finite")
    adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(0.0)
    mask = adjacency.abs() > float(edge_presence_threshold)
    mask.fill_diagonal_(False)
    adjacency = adjacency * mask.to(adjacency.dtype)
    count = int(adjacency.shape[0])
    return HardGraphWindow(
        adjacency=adjacency,
        communities=torch.zeros(
            count, dtype=torch.long, device=adjacency.device
        ),
        node_names=tuple(str(index) for index in range(count)),
        node_ids=tuple(str(index) for index in range(count)),
        time_start=float(time_start),
        edge_presence_threshold=float(edge_presence_threshold),
        window_valid=True,
    )


def compute_sgw_core_sequence(
    adjacencies: Sequence[Optional[torch.Tensor]],
    time_values: Sequence[float],
    edge_presence_threshold: float,
    config: Optional[SGWCoreConfig] = None,
    extractor: Optional[SGWFeatureExtractor] = None,
) -> SGWCoreSequence:
    """Compute exact per-window states and the canonical mean 18-D core."""

    if not adjacencies or len(adjacencies) != len(time_values):
        raise ValueError("Stage-0 adjacency sequence and times must align")
    if edge_presence_threshold < 0.0:
        raise ValueError("Stage-0 edge threshold must be non-negative")
    for left, right in zip(time_values[:-1], time_values[1:]):
        if float(right) <= float(left):
            raise ValueError("Stage-0 window times must strictly increase")
    config = config or SGWCoreConfig()
    extractor = extractor or config.build_extractor()
    states = []
    reference = next(
        (value for value in adjacencies if value is not None), None
    )
    if reference is None:
        raise ValueError("Stage-0 sequence has no valid graph")
    quantiles = reference.new_zeros(
        (len(adjacencies), SGW_QUANTILE_DIM)
    )
    window_mask = torch.zeros(
        len(adjacencies), dtype=torch.bool, device=reference.device
    )
    for index, adjacency in enumerate(adjacencies):
        if adjacency is None:
            states.append(None)
            continue
        window = _window_from_adjacency(
            adjacency,
            float(time_values[index]),
            edge_presence_threshold,
        )
        state = extractor.compute_window_state(window)
        states.append(state)
        quantiles[index] = state.spectral_quantiles
        window_mask[index] = True
    features: SGWSequenceFeatures = extractor.compute_sequence_feature(
        states, time_values
    )
    if tuple(features.h_core.shape) != (SGW_CORE_DIM,):
        raise RuntimeError("Stage-0 core is not 18-D")
    if not bool(torch.isfinite(features.h_core).all()):
        raise ValueError("Stage-0 core contains non-finite values")
    return SGWCoreSequence(
        core=features.h_core,
        window_quantiles=quantiles,
        window_mask=window_mask,
        transition_features=features.transition_features,
        transition_mask=features.transition_mask,
        gw_solver_converged=features.gw_solver_converged,
    )


def save_stage0_sample_artifact(
    path: Path,
    payload: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Stage-0 sample artifact already exists")
    required = (
        "sample_key",
        "label",
        "split",
        "full",
        "hard",
        "provenance",
    )
    if any(name not in payload for name in required):
        raise ValueError("Stage-0 sample artifact is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": SGW_STAGE0_SAMPLE_SCHEMA_VERSION,
            "artifact_type": "sgw_stage0_full_hard_pair",
            "payload": dict(payload),
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_stage0_sample_artifact(path: Path) -> Dict[str, Any]:
    try:
        value = torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        value = torch.load(str(Path(path).resolve()), map_location="cpu")
    if (
        value.get("schema_version")
        != SGW_STAGE0_SAMPLE_SCHEMA_VERSION
        or value.get("artifact_type") != "sgw_stage0_full_hard_pair"
    ):
        raise ValueError("unsupported Stage-0 sample artifact")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid Stage-0 sample payload")
    for side in ("full", "hard"):
        item = payload.get(side, {})
        if tuple(item.get("core", torch.empty(0)).shape) != (
            SGW_CORE_DIM,
        ):
            raise ValueError("Stage-0 sample core dimension mismatch")
    return payload
