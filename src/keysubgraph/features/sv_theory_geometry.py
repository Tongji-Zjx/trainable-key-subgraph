"""Fixed theory features derived from frozen signed hard-graph sequences.

The default SVG model already uses the mean signed-Laplacian spectrum and the
mean absolute adjacent spectral change.  This module adds two complementary
objects without changing the selector or the existing cache schema:

1. the signed (direction-preserving) adjacent spectral displacement; and
2. multi-scale heat-kernel diffusion-geometry summaries.

Both outputs are permutation invariant, fixed dimensional and mask aware.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from keysubgraph.features.sv_hard_graph_features import sv_quantile_grid
from keysubgraph.theory.spectral_gw import SignedLaplacianBuilder


SV_SPECTRAL_DIRECTION_DIM = 16
SV_DIFFUSION_TIME_SCALES = (0.25, 0.50, 1.00, 2.00)
SV_DIFFUSION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SV_DIFFUSION_STATISTICS_PER_SCALE = (
    2 + len(SV_DIFFUSION_QUANTILES)
)
SV_DIFFUSION_GEOMETRY_DIM = (
    len(SV_DIFFUSION_TIME_SCALES)
    * SV_DIFFUSION_STATISTICS_PER_SCALE
)


@dataclass(frozen=True)
class SVTheoryGeometryFeatures:
    spectral_direction: torch.Tensor
    diffusion_geometry: torch.Tensor
    window_mask: torch.Tensor
    transition_mask: torch.Tensor
    diffusion_time_scales: Tuple[float, ...]
    diffusion_quantiles: Tuple[float, ...]


def _empirical_quantiles(
    values: torch.Tensor, probabilities: Sequence[float]
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("diffusion quantiles require a nonempty vector")
    sorted_values = torch.sort(values).values
    count = int(sorted_values.numel())
    indices = [
        min(
            count - 1,
            max(0, int(math.ceil(float(probability) * count)) - 1),
        )
        for probability in probabilities
    ]
    return sorted_values.index_select(
        0,
        torch.tensor(
            indices, dtype=torch.long, device=sorted_values.device
        ),
    )


class SVTheoryGeometryExtractor(object):
    """Extract direction-preserving spectrum and multi-scale diffusion state."""

    spectral_direction_dim = SV_SPECTRAL_DIRECTION_DIM
    diffusion_geometry_dim = SV_DIFFUSION_GEOMETRY_DIM

    def __init__(
        self,
        laplacian_eta: float = 1.0e-3,
        diffusion_time_scales: Sequence[float] = (
            SV_DIFFUSION_TIME_SCALES
        ),
        diffusion_quantiles: Sequence[float] = (
            SV_DIFFUSION_QUANTILES
        ),
        epsilon: float = 1.0e-8,
    ) -> None:
        times = tuple(float(value) for value in diffusion_time_scales)
        quantiles = tuple(float(value) for value in diffusion_quantiles)
        if (
            laplacian_eta <= 0.0
            or epsilon <= 0.0
            or not times
            or any(value <= 0.0 for value in times)
        ):
            raise ValueError("invalid SV theory-geometry configuration")
        if (
            not quantiles
            or any(value <= 0.0 or value >= 1.0 for value in quantiles)
            or any(
                left >= right
                for left, right in zip(quantiles[:-1], quantiles[1:])
            )
        ):
            raise ValueError(
                "diffusion quantiles must be strictly increasing in (0,1)"
            )
        if len(sv_quantile_grid()) != SV_SPECTRAL_DIRECTION_DIM:
            raise RuntimeError("SV spectral direction schema is not 16-D")
        self.laplacian_eta = float(laplacian_eta)
        self.diffusion_time_scales = times
        self.diffusion_quantiles = quantiles
        self.epsilon = float(epsilon)
        self.laplacian = SignedLaplacianBuilder(self.laplacian_eta)

    @property
    def diffusion_output_dim(self) -> int:
        return len(self.diffusion_time_scales) * (
            2 + len(self.diffusion_quantiles)
        )

    def _window_state(
        self, adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            adjacency.ndim != 2
            or adjacency.shape[0] != adjacency.shape[1]
            or adjacency.shape[0] < 1
        ):
            raise ValueError("SV theory adjacency must be nonempty and square")
        if not bool(torch.isfinite(adjacency).all()):
            raise ValueError("SV theory adjacency contains non-finite values")
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        edge_mask = adjacency.abs() > 0.0
        edge_mask.fill_diagonal_(False)
        laplacian = self.laplacian(
            adjacency, edge_mask=edge_mask
        )
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        except RuntimeError as error:
            raise RuntimeError(
                "SV theory Laplacian eigendecomposition failed"
            ) from error
        if not bool(torch.isfinite(eigenvalues).all()) or not bool(
            torch.isfinite(eigenvectors).all()
        ):
            raise RuntimeError(
                "SV theory eigendecomposition returned non-finite values"
            )

        count = int(eigenvalues.numel())
        spectral_indices = [
            min(
                count - 1,
                max(0, int(math.ceil(value * count)) - 1),
            )
            for value in sv_quantile_grid()
        ]
        spectrum = eigenvalues.index_select(
            0,
            torch.tensor(
                spectral_indices,
                dtype=torch.long,
                device=eigenvalues.device,
            ),
        )

        diffusion_parts = []
        upper = torch.triu(
            torch.ones(
                (count, count),
                dtype=torch.bool,
                device=adjacency.device,
            ),
            diagonal=1,
        )
        for time_scale in self.diffusion_time_scales:
            decays = torch.exp(-float(time_scale) * eigenvalues)
            kernel = (
                eigenvectors * decays.unsqueeze(0)
            ).matmul(eigenvectors.transpose(0, 1))
            kernel = 0.5 * (kernel + kernel.transpose(0, 1))
            distance = torch.cdist(kernel, kernel, p=2)
            pair_values = distance[upper]
            if pair_values.numel() < 1:
                summary = adjacency.new_zeros(
                    (2 + len(self.diffusion_quantiles),)
                )
            else:
                mean = pair_values.mean()
                variance = (pair_values - mean).square().mean()
                summary = torch.cat(
                    (
                        mean.reshape(1),
                        torch.sqrt(
                            variance + self.epsilon
                        ).reshape(1),
                        _empirical_quantiles(
                            pair_values, self.diffusion_quantiles
                        ),
                    ),
                    dim=0,
                )
            diffusion_parts.append(summary)
        diffusion = torch.cat(diffusion_parts, dim=0)
        if (
            tuple(spectrum.shape) != (SV_SPECTRAL_DIRECTION_DIM,)
            or tuple(diffusion.shape)
            != (self.diffusion_output_dim,)
            or not bool(torch.isfinite(spectrum).all())
            or not bool(torch.isfinite(diffusion).all())
        ):
            raise RuntimeError("SV theory window state is invalid")
        return spectrum, diffusion

    def build(
        self, windows: Sequence[Optional[object]]
    ) -> SVTheoryGeometryFeatures:
        if not windows:
            raise ValueError("SV theory extractor requires windows")
        spectra = []
        diffusions = []
        state_by_position = []
        window_mask_values = []
        reference = None
        for window in windows:
            if window is None:
                state_by_position.append(None)
                window_mask_values.append(False)
                continue
            adjacency = window.adjacency
            if reference is None:
                reference = adjacency
            spectrum, diffusion = self._window_state(adjacency)
            spectra.append(spectrum)
            diffusions.append(diffusion)
            state_by_position.append(spectrum)
            window_mask_values.append(True)
        if reference is None or not spectra:
            raise ValueError("SV theory sequence has no valid windows")

        directions = []
        transition_mask_values = []
        for left, right in zip(
            state_by_position[:-1], state_by_position[1:]
        ):
            valid = left is not None and right is not None
            transition_mask_values.append(valid)
            if valid:
                directions.append(right - left)
        spectral_direction = (
            torch.stack(directions, dim=0).mean(dim=0)
            if directions
            else reference.new_zeros((SV_SPECTRAL_DIRECTION_DIM,))
        )
        diffusion_geometry = torch.stack(
            diffusions, dim=0
        ).mean(dim=0)
        window_mask = torch.tensor(
            window_mask_values,
            dtype=torch.bool,
            device=reference.device,
        )
        transition_mask = torch.tensor(
            transition_mask_values,
            dtype=torch.bool,
            device=reference.device,
        )
        if (
            tuple(spectral_direction.shape)
            != (SV_SPECTRAL_DIRECTION_DIM,)
            or tuple(diffusion_geometry.shape)
            != (self.diffusion_output_dim,)
            or not bool(torch.isfinite(spectral_direction).all())
            or not bool(torch.isfinite(diffusion_geometry).all())
        ):
            raise RuntimeError("SV theory sample features are invalid")
        return SVTheoryGeometryFeatures(
            spectral_direction=spectral_direction,
            diffusion_geometry=diffusion_geometry,
            window_mask=window_mask,
            transition_mask=transition_mask,
            diffusion_time_scales=self.diffusion_time_scales,
            diffusion_quantiles=self.diffusion_quantiles,
        )
