"""Exact per-transition spectral variation from frozen hard graph windows."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.theory.spectral_gw import (
    SignedLaplacianBuilder,
    SpectralStateExtractor,
)


def temporal_quantile_grid() -> Tuple[float, ...]:
    return tuple(0.05 + (0.90 / 15.0) * index for index in range(16))


@dataclass(frozen=True)
class DualTemporalVariation:
    values: torch.Tensor
    mask: torch.Tensor
    spectral_quantiles: torch.Tensor
    window_mask: torch.Tensor


class DualTemporalVariationExtractor(object):
    """Build |delta spectral quantile| without computing Exact GW."""

    def __init__(self, laplacian_eta: float = 1.0e-3) -> None:
        if laplacian_eta <= 0.0:
            raise ValueError("temporal Laplacian eta must be positive")
        self.laplacian = SignedLaplacianBuilder(laplacian_eta)
        self.spectral = SpectralStateExtractor(temporal_quantile_grid())

    def _window_quantiles(
        self, window: Optional[HardGraphWindow]
    ) -> Optional[torch.Tensor]:
        if window is None or not window.window_valid:
            return None
        adjacency = window.adjacency
        if (
            adjacency.ndim != 2
            or adjacency.shape[0] != adjacency.shape[1]
            or adjacency.shape[0] < 1
            or not bool(torch.isfinite(adjacency).all())
        ):
            raise ValueError("temporal hard adjacency is invalid")
        threshold = float(window.edge_presence_threshold)
        if threshold < 0.0:
            raise ValueError("temporal edge threshold must be non-negative")
        edge_mask = adjacency.abs() > threshold
        edge_mask = edge_mask.clone()
        edge_mask.fill_diagonal_(False)
        laplacian = self.laplacian(adjacency, edge_mask=edge_mask)
        return self.spectral(laplacian).quantiles

    def compute(
        self, windows: Sequence[Optional[HardGraphWindow]]
    ) -> DualTemporalVariation:
        if not windows:
            raise ValueError("temporal hard graph sequence cannot be empty")
        states = tuple(self._window_quantiles(window) for window in windows)
        reference = next(
            (state for state in states if state is not None),
            torch.zeros(16, dtype=torch.float32),
        )
        quantiles = reference.new_zeros((len(states), 16))
        window_mask = torch.zeros(
            len(states), dtype=torch.bool, device=reference.device
        )
        for index, state in enumerate(states):
            if state is not None:
                quantiles[index] = state
                window_mask[index] = True
        transition_count = max(0, len(states) - 1)
        values = reference.new_zeros((transition_count, 16))
        mask = torch.zeros(
            transition_count, dtype=torch.bool, device=reference.device
        )
        for index in range(transition_count):
            if states[index] is None or states[index + 1] is None:
                continue
            values[index] = (
                states[index + 1] - states[index]
            ).abs()
            mask[index] = True
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("temporal variation contains non-finite values")
        return DualTemporalVariation(
            values=values,
            mask=mask,
            spectral_quantiles=quantiles,
            window_mask=window_mask,
        )
