"""Canonical 18-D core and 34-D TG-SGW hard-sequence representations."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch

from keysubgraph.features.hard_graph_features import HardGraphWindow
from .spectral_gw import (
    DifferentiableGWLoss,
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    SpectralStateExtractor,
)


TG_SGW_FEATURE_ARTIFACT_SCHEMA_VERSION = 1


def _default_quantile_grid() -> Tuple[float, ...]:
    return tuple(0.05 + (0.90 / 15.0) * index for index in range(16))


@dataclass(frozen=True)
class SGWTheoryFeatureConfig:
    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0
    spectral_quantile_grid: Tuple[float, ...] = _default_quantile_grid()
    spectral_w1_grid_size: int = 256
    time_quantity: str = "speed"

    def __post_init__(self) -> None:
        if self.laplacian_eta <= 0.0 or self.diffusion_time <= 0.0:
            raise ValueError("SGW eta and diffusion time must be positive")
        if len(self.spectral_quantile_grid) != 16:
            raise ValueError("TG-SGW requires exactly 16 spectral quantiles")
        if self.spectral_w1_grid_size < 2:
            raise ValueError("spectral W1 grid must contain at least two points")
        if self.time_quantity != "speed":
            raise ValueError("canonical TG-SGW core features use speed semantics")


@dataclass(frozen=True)
class SGWWindowState:
    eigenvalues: torch.Tensor
    spectral_quantiles: torch.Tensor
    diffusion_distance: torch.Tensor
    node_measure: torch.Tensor


@dataclass(frozen=True)
class SGWSequenceFeatures:
    h_core: torch.Tensor
    h_variation: torch.Tensor
    h_classification: torch.Tensor
    transition_features: torch.Tensor
    transition_mask: torch.Tensor
    gw_solver_converged: Tuple[bool, ...]
    time_quantity: str = "speed"


@dataclass(frozen=True)
class TGSGWFeatureArtifact:
    sample_key: str
    sample_id: str
    label: int
    split: str
    features: SGWSequenceFeatures
    eligible_for_stage_c: bool
    data_protocol_sha256: str
    teacher_checkpoint_sha256: str


def _empirical_quantiles(values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    sorted_values = torch.sort(values).values
    indices = torch.ceil(probabilities * sorted_values.numel()).to(dtype=torch.long) - 1
    return sorted_values.index_select(0, indices.clamp(0, sorted_values.numel() - 1))


class SGWFeatureExtractor(object):
    """Mask-aware Stage-C theory branch using the canonical speed semantics."""

    def __init__(
        self,
        theory_config: Optional[Any] = None,
        gw_entropic_reg: float = 1.0e-2,
        gw_max_iter: int = 100,
        gw_sinkhorn_iter: int = 100,
        gw_tolerance: float = 1.0e-7,
    ) -> None:
        self.config = theory_config or SGWTheoryFeatureConfig()
        if len(self.config.spectral_quantile_grid) != 16:
            raise ValueError("TG-SGW requires exactly 16 spectral quantiles")
        if self.config.time_quantity != "speed":
            raise ValueError("canonical TG-SGW core features use speed semantics")
        self.laplacian = SignedLaplacianBuilder(self.config.laplacian_eta)
        self.spectral = SpectralStateExtractor(self.config.spectral_quantile_grid)
        self.heat = HeatKernelMetricBuilder(self.config.diffusion_time)
        self.gw = DifferentiableGWLoss(
            entropic_reg=gw_entropic_reg,
            max_iter=gw_max_iter,
            tolerance=gw_tolerance,
            sinkhorn_iter=gw_sinkhorn_iter,
            failure_strategy="use_last",
        )
        size = self.config.spectral_w1_grid_size
        self.spectral_w1_grid = tuple(
            (index + 0.5) / float(size) for index in range(size)
        )

    def compute_window_state(self, hard_union_graph: HardGraphWindow) -> SGWWindowState:
        if not hard_union_graph.window_valid or hard_union_graph.num_nodes < 1:
            raise ValueError("invalid or empty hard graphs cannot enter SGW decomposition")
        adjacency = hard_union_graph.adjacency
        edge_mask = adjacency.abs() > float(hard_union_graph.edge_presence_threshold)
        edge_mask.fill_diagonal_(False)
        with torch.no_grad():
            laplacian = self.laplacian(adjacency, edge_mask=edge_mask)
            spectrum = self.spectral(laplacian)
            diffusion = self.heat(laplacian).distance
            measure = adjacency.new_full(
                (adjacency.shape[0],), 1.0 / float(adjacency.shape[0])
            )
        return SGWWindowState(
            eigenvalues=spectrum.eigenvalues,
            spectral_quantiles=spectrum.quantiles,
            diffusion_distance=diffusion,
            node_measure=measure,
        )

    def _spectral_w1_dense(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        probabilities = first.new_tensor(self.spectral_w1_grid)
        return (
            _empirical_quantiles(first, probabilities)
            - _empirical_quantiles(second, probabilities)
        ).abs().mean()

    def compute_sequence_feature(
        self,
        window_states: Sequence[Optional[SGWWindowState]],
        time_values: Sequence[float],
    ) -> SGWSequenceFeatures:
        if len(window_states) != len(time_values) or len(window_states) < 1:
            raise ValueError("SGW states and times must be non-empty and aligned")
        for left, right in zip(time_values[:-1], time_values[1:]):
            if float(right) <= float(left):
                raise ValueError("SGW time values must be strictly increasing")
        transition_count = max(0, len(window_states) - 1)
        core_dim = len(self.config.spectral_quantile_grid) + 2
        quantile_dim = len(self.config.spectral_quantile_grid)
        reference = next((state.eigenvalues for state in window_states if state is not None), None)
        dtype = reference.dtype if reference is not None else torch.float32
        device = reference.device if reference is not None else torch.device("cpu")
        transitions = torch.zeros(
            (transition_count, core_dim), dtype=dtype, device=device
        )
        mask = torch.zeros(transition_count, dtype=torch.bool, device=device)
        convergence = []
        with torch.no_grad():
            for index in range(transition_count):
                first, second = window_states[index], window_states[index + 1]
                if first is None or second is None:
                    continue
                tau = float(time_values[index + 1]) - float(time_values[index])
                delta_quantiles = second.spectral_quantiles - first.spectral_quantiles
                spectral_speed = self._spectral_w1_dense(
                    first.eigenvalues, second.eigenvalues
                ) / tau
                gw_result = self.gw(
                    first.diffusion_distance,
                    second.diffusion_distance,
                    first.node_measure,
                    second.node_measure,
                )
                gw_speed = gw_result.structural_cost_sqrt / tau
                transitions[index] = torch.cat(
                    (
                        delta_quantiles,
                        spectral_speed.reshape(1),
                        gw_speed.reshape(1),
                    )
                )
                mask[index] = True
                convergence.append(bool(gw_result.converged))
        if bool(mask.any()):
            valid = transitions[mask]
            h_core = valid.mean(dim=0)
            h_variation = valid[:, :quantile_dim].abs().mean(dim=0)
        else:
            h_core = torch.zeros(core_dim, dtype=dtype, device=device)
            h_variation = torch.zeros(quantile_dim, dtype=dtype, device=device)
        h_classification = torch.cat((h_core, h_variation), dim=0)
        if tuple(h_core.shape) != (18,) or tuple(h_classification.shape) != (34,):
            raise RuntimeError("TG-SGW theory representation dimensions are invalid")
        return SGWSequenceFeatures(
            h_core=h_core,
            h_variation=h_variation,
            h_classification=h_classification,
            transition_features=transitions,
            transition_mask=mask,
            gw_solver_converged=tuple(convergence),
            time_quantity=self.config.time_quantity,
        )

    def compute_hard_graph_sequence(
        self,
        windows: Sequence[Optional[HardGraphWindow]],
        time_values: Sequence[float],
    ) -> SGWSequenceFeatures:
        states = tuple(
            self.compute_window_state(window) if window is not None else None
            for window in windows
        )
        return self.compute_sequence_feature(states, time_values)


def save_tg_sgw_feature_artifact(
    artifact: TGSGWFeatureArtifact, path: Path, overwrite: bool = False
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("TG-SGW feature artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": TG_SGW_FEATURE_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tg_sgw_theory_features",
            "artifact": artifact,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_tg_sgw_feature_artifact(path: Path) -> TGSGWFeatureArtifact:
    try:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if payload.get("schema_version") != TG_SGW_FEATURE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported TG-SGW feature artifact schema")
    if payload.get("artifact_type") != "tg_sgw_theory_features":
        raise ValueError("unexpected TG-SGW feature artifact")
    artifact = payload.get("artifact")
    if not isinstance(artifact, TGSGWFeatureArtifact):
        raise ValueError("invalid TG-SGW feature artifact payload")
    return artifact
