"""Differentiable signed-Laplacian, spectrum, heat-kernel and GW utilities.

The routines deliberately operate on one variable-size graph at a time.  This
matches the project's list-based batching contract and prevents padded nodes
from entering a degree, spectrum, heat kernel, or node measure.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


def _validate_square(matrix: torch.Tensor, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("{} must be a square rank-2 tensor".format(name))
    if matrix.shape[0] < 1:
        raise ValueError("{} cannot be empty".format(name))
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("{} contains non-finite values".format(name))


def _valid_indices(size: int, node_mask: Optional[torch.Tensor], device) -> torch.Tensor:
    if node_mask is None:
        return torch.arange(size, device=device)
    if tuple(node_mask.shape) != (size,):
        raise ValueError("node_mask must have shape [N]")
    indices = torch.nonzero(node_mask.to(device=device, dtype=torch.bool), as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("graph has no valid nodes")
    return indices


class SignedLaplacianBuilder(nn.Module):
    """Build the regularized normalized signed Laplacian from valid edges."""

    def __init__(self, eta: float = 1.0e-4) -> None:
        super().__init__()
        if eta <= 0.0:
            raise ValueError("laplacian eta must be positive")
        self.eta = float(eta)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _validate_square(adjacency, "adjacency")
        size = adjacency.shape[0]
        valid = torch.ones(size, dtype=torch.bool, device=adjacency.device)
        if node_mask is not None:
            if tuple(node_mask.shape) != (size,):
                raise ValueError("node_mask must have shape [N]")
            valid = node_mask.to(device=adjacency.device, dtype=torch.bool)
        if not bool(valid.any()):
            raise ValueError("graph has no valid nodes")

        pair_mask = valid[:, None] & valid[None, :]
        if edge_mask is not None:
            if tuple(edge_mask.shape) != tuple(adjacency.shape):
                raise ValueError("edge_mask must match adjacency")
            pair_mask = pair_mask & edge_mask.to(device=adjacency.device, dtype=torch.bool)
        pair_mask = pair_mask.clone()
        pair_mask.fill_diagonal_(False)

        signed = adjacency * pair_mask.to(dtype=adjacency.dtype)
        signed = 0.5 * (signed + signed.transpose(0, 1))
        degree = signed.abs().sum(dim=-1)
        inverse_sqrt = torch.zeros_like(degree)
        inverse_sqrt[valid] = (degree[valid] + self.eta).rsqrt()
        laplacian = torch.diag(degree) - signed
        regularizer = torch.diag(valid.to(dtype=adjacency.dtype) * self.eta)
        normalized = inverse_sqrt[:, None] * (laplacian + regularizer) * inverse_sqrt[None, :]
        normalized = normalized * pair_mask.logical_or(torch.diag(valid)).to(adjacency.dtype)
        return 0.5 * (normalized + normalized.transpose(0, 1))


@dataclass(frozen=True)
class LaplacianFidelityResult:
    normalized_frobenius_squared: torch.Tensor
    frobenius_norm: torch.Tensor
    operator_norm: torch.Tensor
    valid_node_count: int


def laplacian_fidelity_metrics(
    full_laplacian: torch.Tensor,
    soft_laplacian: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
) -> LaplacianFidelityResult:
    """Return the train loss and the unnormalized theory diagnostics."""

    _validate_square(full_laplacian, "full_laplacian")
    _validate_square(soft_laplacian, "soft_laplacian")
    if tuple(full_laplacian.shape) != tuple(soft_laplacian.shape):
        raise ValueError("full and soft Laplacians must have the same shape")
    indices = _valid_indices(full_laplacian.shape[0], node_mask, full_laplacian.device)
    difference = (full_laplacian - soft_laplacian).index_select(0, indices).index_select(1, indices)
    difference = 0.5 * (difference + difference.transpose(0, 1))
    count = int(indices.numel())
    squared_frobenius = difference.square().sum()
    frobenius = torch.sqrt(squared_frobenius.clamp_min(0.0))
    operator = torch.linalg.eigvalsh(difference).abs().max()
    return LaplacianFidelityResult(
        normalized_frobenius_squared=squared_frobenius / float(count * count),
        frobenius_norm=frobenius,
        operator_norm=operator,
        valid_node_count=count,
    )


@dataclass(frozen=True)
class SpectralState:
    eigenvalues: torch.Tensor
    quantiles: torch.Tensor
    spectral_gap: torch.Tensor
    valid_node_count: int


class SpectralStateExtractor(nn.Module):
    """Extract a fixed-dimensional empirical spectral-quantile state."""

    def __init__(
        self,
        quantile_grid: Sequence[float] = (
            0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
        ),
    ) -> None:
        super().__init__()
        grid = tuple(float(value) for value in quantile_grid)
        if not grid or any(value <= 0.0 or value >= 1.0 for value in grid):
            raise ValueError("spectral quantiles must lie strictly inside (0, 1)")
        if any(left >= right for left, right in zip(grid[:-1], grid[1:])):
            raise ValueError("spectral quantile grid must be strictly increasing")
        self.quantile_grid = grid

    def forward(
        self, laplacian: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> SpectralState:
        _validate_square(laplacian, "laplacian")
        indices = _valid_indices(laplacian.shape[0], node_mask, laplacian.device)
        valid_laplacian = laplacian.index_select(0, indices).index_select(1, indices)
        try:
            eigenvalues = torch.linalg.eigvalsh(valid_laplacian)
        except RuntimeError as error:
            raise RuntimeError("signed Laplacian eigendecomposition failed") from error
        if not bool(torch.isfinite(eigenvalues).all()):
            raise RuntimeError("signed Laplacian eigendecomposition returned non-finite values")
        count = int(eigenvalues.numel())
        quantile_indices = [
            min(count - 1, max(0, int(math.ceil(value * count)) - 1))
            for value in self.quantile_grid
        ]
        quantiles = eigenvalues.index_select(
            0, torch.tensor(quantile_indices, dtype=torch.long, device=eigenvalues.device)
        )
        gap = (
            eigenvalues[1] - eigenvalues[0]
            if count > 1
            else eigenvalues.new_zeros(())
        )
        return SpectralState(eigenvalues, quantiles, gap, count)


@dataclass(frozen=True)
class HeatKernelState:
    kernel: torch.Tensor
    distance: torch.Tensor
    valid_node_count: int


class HeatKernelMetricBuilder(nn.Module):
    """Construct the heat kernel and its row-wise diffusion pseudometric."""

    def __init__(self, time_scale: float) -> None:
        super().__init__()
        if time_scale <= 0.0:
            raise ValueError("heat kernel time_scale must be positive")
        self.time_scale = float(time_scale)

    def forward(
        self, laplacian: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> HeatKernelState:
        _validate_square(laplacian, "laplacian")
        indices = _valid_indices(laplacian.shape[0], node_mask, laplacian.device)
        valid_laplacian = laplacian.index_select(0, indices).index_select(1, indices)
        # matrix_exp is mathematically identical to the eigendecomposition for
        # symmetric L, while its backward is stable at repeated eigenvalues.
        # Eigenvectors from eigh have undefined derivatives in that case and
        # caused non-finite CUDA gradients in the fidelity training path.
        try:
            kernel = torch.matrix_exp(-self.time_scale * valid_laplacian)
        except RuntimeError as error:
            raise RuntimeError("heat-kernel matrix exponential failed") from error
        kernel = 0.5 * (kernel + kernel.transpose(0, 1))
        distance = torch.cdist(kernel, kernel, p=2)
        distance = 0.5 * (distance + distance.transpose(0, 1))
        distance = distance - torch.diag(torch.diagonal(distance))
        distance = distance.clamp_min(0.0)
        return HeatKernelState(kernel, distance, int(indices.numel()))


def _empirical_quantile(sorted_values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    count = sorted_values.numel()
    indices = torch.ceil(probabilities * count).to(dtype=torch.long) - 1
    indices = indices.clamp(0, count - 1)
    return sorted_values.index_select(0, indices)


def spectral_winf_exact(spectrum_a: torch.Tensor, spectrum_b: torch.Tensor) -> torch.Tensor:
    """Exact W-infinity for equally weighted one-dimensional empirical measures."""

    if spectrum_a.ndim != 1 or spectrum_b.ndim != 1:
        raise ValueError("spectra must be rank-1")
    if spectrum_a.numel() < 1 or spectrum_b.numel() < 1:
        raise ValueError("spectra cannot be empty")
    a = torch.sort(spectrum_a).values
    b = torch.sort(spectrum_b).values
    breakpoints = sorted(
        set(
            [0.0, 1.0]
            + [index / float(a.numel()) for index in range(1, a.numel())]
            + [index / float(b.numel()) for index in range(1, b.numel())]
        )
    )
    midpoints = [
        0.5 * (left + right)
        for left, right in zip(breakpoints[:-1], breakpoints[1:])
        if right > left
    ]
    probabilities = a.new_tensor(midpoints)
    return (_empirical_quantile(a, probabilities) - _empirical_quantile(b, probabilities)).abs().max()


def spectral_winf_dense(
    spectrum_a: torch.Tensor, spectrum_b: torch.Tensor, grid_size: int = 10001
) -> torch.Tensor:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    a = torch.sort(spectrum_a).values
    b = torch.sort(spectrum_b).values
    probabilities = torch.linspace(
        0.5 / grid_size,
        1.0 - 0.5 / grid_size,
        grid_size,
        dtype=a.dtype,
        device=a.device,
    )
    return (_empirical_quantile(a, probabilities) - _empirical_quantile(b, probabilities)).abs().max()


def spectral_w1(spectrum_a: torch.Tensor, spectrum_b: torch.Tensor) -> torch.Tensor:
    """Exact W1 for equally weighted one-dimensional empirical measures."""

    a = torch.sort(spectrum_a).values
    b = torch.sort(spectrum_b).values
    breakpoints = sorted(
        set(
            [0.0, 1.0]
            + [index / float(a.numel()) for index in range(1, a.numel())]
            + [index / float(b.numel()) for index in range(1, b.numel())]
        )
    )
    total = a.new_zeros(())
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        if right <= left:
            continue
        probability = a.new_tensor([0.5 * (left + right)])
        difference = (_empirical_quantile(a, probability) - _empirical_quantile(b, probability)).abs()[0]
        total = total + difference * (right - left)
    return total


@dataclass(frozen=True)
class GWResult:
    distance: torch.Tensor
    squared_distance: torch.Tensor
    coupling: torch.Tensor
    converged: bool
    iterations: int
    residual: float
    solver: str
    regularized_objective: Optional[torch.Tensor] = None

    @property
    def structural_cost_sqrt(self) -> torch.Tensor:
        return self.distance

    @property
    def structural_cost_squared(self) -> torch.Tensor:
        return self.squared_distance

    @property
    def returned_distance_excludes_entropy_term(self) -> bool:
        return True


@dataclass(frozen=True)
class IdentityCouplingGWResult:
    squared_upper_bound: torch.Tensor
    distance_upper_bound: torch.Tensor
    measure: torch.Tensor


def gw_identity_coupling_upper_bound(
    full_distance: torch.Tensor,
    soft_distance: torch.Tensor,
    measure: Optional[torch.Tensor] = None,
) -> IdentityCouplingGWResult:
    """Differentiable same-node upper bound for squared second-order GW."""

    _validate_square(full_distance, "full_distance")
    _validate_square(soft_distance, "soft_distance")
    if tuple(full_distance.shape) != tuple(soft_distance.shape):
        raise ValueError("identity coupling requires equal node dimensions")
    count = full_distance.shape[0]
    if measure is None:
        probability = full_distance.new_full((count,), 1.0 / float(count))
    else:
        if tuple(measure.shape) != (count,):
            raise ValueError("identity-coupling measure has the wrong shape")
        probability = measure.to(device=full_distance.device, dtype=full_distance.dtype)
        if bool((probability < 0.0).any()) or not bool(torch.isfinite(probability).all()):
            raise ValueError("identity-coupling measure must be finite and non-negative")
        total = probability.sum()
        if float(total.detach().cpu()) <= 0.0:
            raise ValueError("identity-coupling measure must have positive mass")
        probability = probability / total
    pair_mass = probability[:, None] * probability[None, :]
    squared = ((full_distance - soft_distance).square() * pair_mass).sum()
    return IdentityCouplingGWResult(
        squared_upper_bound=squared,
        distance_upper_bound=torch.sqrt(squared.clamp_min(0.0)),
        measure=probability,
    )


class DifferentiableGWLoss(nn.Module):
    """Entropic squared-loss Gromov--Wasserstein solver.

    A non-converged solve either raises or returns the last non-zero iterate with
    ``converged=False``.  It is never silently replaced by zero.
    """

    def __init__(
        self,
        entropic_reg: float = 5.0e-2,
        max_iter: int = 50,
        tolerance: float = 1.0e-7,
        sinkhorn_iter: int = 50,
        failure_strategy: str = "use_last",
    ) -> None:
        super().__init__()
        if entropic_reg <= 0.0:
            raise ValueError("entropic_reg must be positive")
        if max_iter < 1 or sinkhorn_iter < 1:
            raise ValueError("GW and Sinkhorn iteration counts must be positive")
        if tolerance <= 0.0:
            raise ValueError("GW tolerance must be positive")
        if failure_strategy not in ("use_last", "raise"):
            raise ValueError("unsupported GW failure strategy")
        self.entropic_reg = float(entropic_reg)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.failure_strategy = failure_strategy

    @staticmethod
    def _measure(size: int, reference: torch.Tensor, measure: Optional[torch.Tensor]) -> torch.Tensor:
        if measure is None:
            return reference.new_full((size,), 1.0 / size)
        if tuple(measure.shape) != (size,):
            raise ValueError("node measure has the wrong shape")
        measure = measure.to(device=reference.device, dtype=reference.dtype)
        if bool((measure < 0).any()) or not bool(torch.isfinite(measure).all()):
            raise ValueError("node measure must be finite and non-negative")
        total = measure.sum()
        if float(total.detach().cpu()) <= 0.0:
            raise ValueError("node measure must have positive mass")
        return measure / total

    @staticmethod
    def _structural_cost(
        first: torch.Tensor,
        second: torch.Tensor,
        coupling: torch.Tensor,
        first_measure: torch.Tensor,
        second_measure: torch.Tensor,
    ) -> torch.Tensor:
        first_term = (first.pow(2) @ first_measure)[:, None]
        second_term = (second.pow(2) @ second_measure)[None, :]
        return first_term + second_term - 2.0 * first @ coupling @ second.transpose(0, 1)

    def _sinkhorn(
        self, cost: torch.Tensor, first_measure: torch.Tensor, second_measure: torch.Tensor
    ) -> torch.Tensor:
        log_kernel = -cost / self.entropic_reg
        log_first = torch.log(first_measure.clamp_min(torch.finfo(cost.dtype).tiny))
        log_second = torch.log(second_measure.clamp_min(torch.finfo(cost.dtype).tiny))
        log_u = torch.zeros_like(log_first)
        log_v = torch.zeros_like(log_second)
        for _ in range(self.sinkhorn_iter):
            log_u = log_first - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            log_v = log_second - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        coupling = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
        return coupling / coupling.sum().clamp_min(torch.finfo(cost.dtype).tiny)

    def forward(
        self,
        first_distance: torch.Tensor,
        second_distance: torch.Tensor,
        first_measure: Optional[torch.Tensor] = None,
        second_measure: Optional[torch.Tensor] = None,
    ) -> GWResult:
        _validate_square(first_distance, "first_distance")
        _validate_square(second_distance, "second_distance")
        p = self._measure(first_distance.shape[0], first_distance, first_measure)
        q = self._measure(second_distance.shape[0], second_distance, second_measure)

        if first_distance.shape == second_distance.shape and bool(
            torch.allclose(first_distance, second_distance, atol=1.0e-10, rtol=1.0e-8)
        ):
            zero = (first_distance - second_distance).pow(2).sum()
            coupling = torch.diag(p)
            return GWResult(zero, zero, coupling, True, 0, 0.0, "entropic_gw", zero)
        if first_distance.shape == second_distance.shape:
            # Detect exact graph isomorphisms without relying on node IDs.  This
            # preserves the defining GW property that a pure node permutation
            # has zero distance, while all non-isomorphic pairs still use GW.
            first_signature = torch.sort(first_distance, dim=1).values
            second_signature = torch.sort(second_distance, dim=1).values
            assignment_cost = torch.cdist(first_signature, second_signature).detach().cpu().numpy()
            row_indices, column_indices = linear_sum_assignment(assignment_cost)
            if tuple(row_indices.tolist()) == tuple(range(first_distance.shape[0])):
                permutation = torch.tensor(
                    column_indices,
                    dtype=torch.long,
                    device=second_distance.device,
                )
                aligned = second_distance.index_select(0, permutation).index_select(1, permutation)
                if bool(torch.allclose(first_distance, aligned, atol=1.0e-9, rtol=1.0e-7)):
                    zero = (first_distance - aligned).pow(2).sum()
                    coupling = first_distance.new_zeros(first_distance.shape)
                    coupling[
                        torch.arange(first_distance.shape[0], device=first_distance.device),
                        permutation,
                    ] = p
                    return GWResult(zero, zero, coupling, True, 0, 0.0, "entropic_gw", zero)

        coupling = p[:, None] * q[None, :]
        converged = False
        residual = math.inf
        iterations = self.max_iter
        for iteration in range(1, self.max_iter + 1):
            structural_cost = self._structural_cost(
                first_distance, second_distance, coupling, p, q
            )
            updated = self._sinkhorn(structural_cost, p, q)
            residual_tensor = (updated - coupling).abs().max()
            residual = float(residual_tensor.detach().cpu())
            coupling = updated
            if residual <= self.tolerance:
                converged = True
                iterations = iteration
                break
        if not converged and self.failure_strategy == "raise":
            raise RuntimeError(
                "entropic GW did not converge in {} iterations (residual={:.6g})".format(
                    self.max_iter, residual
                )
            )
        final_cost = self._structural_cost(first_distance, second_distance, coupling, p, q)
        squared = (final_cost * coupling).sum().clamp_min(0.0)
        distance = torch.sqrt(squared + torch.finfo(squared.dtype).eps)
        entropy = (
            coupling
            * (torch.log(coupling.clamp_min(torch.finfo(coupling.dtype).tiny)) - 1.0)
        ).sum()
        regularized_objective = squared + self.entropic_reg * entropy
        return GWResult(
            distance,
            squared,
            coupling,
            converged,
            iterations,
            residual,
            "entropic_gw",
            regularized_objective,
        )
