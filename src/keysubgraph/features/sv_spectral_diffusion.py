"""Exact spectral-diffusion states for frozen signed hard graphs.

The eigendecomposition is intentionally computed once during cache building.
Training reuses the eigenbasis to evaluate exact heat-kernel messages without
materialising one dense kernel for every diffusion scale and epoch.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from keysubgraph.features.sv_hard_graph_features import sv_quantile_grid
from keysubgraph.theory.spectral_gw import SignedLaplacianBuilder


SV_HKS_TIME_SCALES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
SV_DIFFUSION_MESSAGE_TIME_SCALES = (0.5, 2.0, 10.0)
SV_HKS_DIM = len(SV_HKS_TIME_SCALES)
SV_SPECTRAL_STATE_DIM = 16


@dataclass(frozen=True)
class SVSpectralDiffusionWindowFeatures:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    hks: torch.Tensor
    spectral_quantiles: torch.Tensor


@dataclass(frozen=True)
class SVSpectralDiffusionSampleFeatures:
    windows: Tuple[Optional[SVSpectralDiffusionWindowFeatures], ...]
    window_mask: torch.Tensor
    transition_mask: torch.Tensor


def exact_heat_diffusion_message(
    node_states: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    time_scale: float,
) -> torch.Tensor:
    """Return ``exp(-tL) H`` from a cached symmetric eigendecomposition."""

    if node_states.ndim != 2:
        raise ValueError("diffusion node states must have shape [N,H]")
    node_count = int(node_states.shape[0])
    if tuple(eigenvalues.shape) != (node_count,) or tuple(
        eigenvectors.shape
    ) != (node_count, node_count):
        raise ValueError("diffusion eigenbasis does not align with nodes")
    if float(time_scale) <= 0.0:
        raise ValueError("diffusion time scale must be positive")
    values = eigenvalues.to(node_states)
    vectors = eigenvectors.to(node_states)
    coefficients = vectors.transpose(0, 1).matmul(node_states)
    decays = torch.exp(-float(time_scale) * values)
    return vectors.matmul(decays[:, None] * coefficients)


class SVSpectralDiffusionExtractor(object):
    """Build exact HKS, spectral state and reusable heat-kernel basis."""

    def __init__(
        self,
        laplacian_eta: float = 1.0e-3,
        hks_time_scales: Sequence[float] = SV_HKS_TIME_SCALES,
    ) -> None:
        times = tuple(float(value) for value in hks_time_scales)
        if (
            float(laplacian_eta) <= 0.0
            or not times
            or any(value <= 0.0 for value in times)
            or any(left >= right for left, right in zip(times[:-1], times[1:]))
        ):
            raise ValueError("invalid SV spectral-diffusion configuration")
        if len(sv_quantile_grid()) != SV_SPECTRAL_STATE_DIM:
            raise RuntimeError("SV spectral state schema must be 16-D")
        self.laplacian_eta = float(laplacian_eta)
        self.hks_time_scales = times
        self.laplacian = SignedLaplacianBuilder(self.laplacian_eta)

    def build_window(
        self, adjacency: torch.Tensor
    ) -> SVSpectralDiffusionWindowFeatures:
        if (
            adjacency.ndim != 2
            or adjacency.shape[0] != adjacency.shape[1]
            or adjacency.shape[0] < 1
        ):
            raise ValueError("spectral-diffusion adjacency must be square")
        if not bool(torch.isfinite(adjacency).all()):
            raise ValueError("spectral-diffusion adjacency is non-finite")
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        edge_mask = adjacency.abs() > 0.0
        edge_mask.fill_diagonal_(False)
        laplacian = self.laplacian(adjacency, edge_mask=edge_mask)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        if not bool(torch.isfinite(eigenvalues).all()) or not bool(
            torch.isfinite(eigenvectors).all()
        ):
            raise RuntimeError("spectral-diffusion eigendecomposition failed")

        count = int(eigenvalues.numel())
        indices = [
            min(count - 1, max(0, int(math.ceil(q * count)) - 1))
            for q in sv_quantile_grid()
        ]
        spectral_quantiles = eigenvalues.index_select(
            0,
            torch.tensor(indices, dtype=torch.long, device=eigenvalues.device),
        )
        times = eigenvalues.new_tensor(self.hks_time_scales)
        decays = torch.exp(-eigenvalues[:, None] * times[None, :])
        hks = eigenvectors.square().matmul(decays)
        if tuple(hks.shape) != (count, len(self.hks_time_scales)) or not bool(
            torch.isfinite(hks).all()
        ):
            raise RuntimeError("SV HKS construction failed")
        return SVSpectralDiffusionWindowFeatures(
            eigenvalues=eigenvalues.to(torch.float32),
            eigenvectors=eigenvectors.to(torch.float32),
            hks=hks.to(torch.float32),
            spectral_quantiles=spectral_quantiles.to(torch.float32),
        )

    def build(
        self, windows: Sequence[Optional[object]]
    ) -> SVSpectralDiffusionSampleFeatures:
        if not windows:
            raise ValueError("spectral-diffusion sequence cannot be empty")
        output = []
        mask = []
        for window in windows:
            if window is None:
                output.append(None)
                mask.append(False)
                continue
            output.append(self.build_window(window.adjacency))
            mask.append(True)
        window_mask = torch.tensor(mask, dtype=torch.bool)
        if not bool(window_mask.any()):
            raise ValueError("spectral-diffusion sequence has no valid window")
        transition_mask = torch.tensor(
            [left and right for left, right in zip(mask[:-1], mask[1:])],
            dtype=torch.bool,
        )
        return SVSpectralDiffusionSampleFeatures(
            windows=tuple(output),
            window_mask=window_mask,
            transition_mask=transition_mask,
        )
