"""Differentiable SGW-aligned proxy used only to train the hard selector."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.theory.spectral_gw import (
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    SpectralStateExtractor,
    gw_identity_coupling_upper_bound,
    laplacian_fidelity_metrics,
)
from .dual_stse_hard_sgw_types import DualSTSEHardSGWConfig
from .hard_stse_types import HardWindowOutput


def _quantile_grid() -> Tuple[float, ...]:
    return tuple(0.05 + (0.90 / 15.0) * index for index in range(16))


def _empirical_quantiles(
    values: torch.Tensor, grid: Sequence[float]
) -> torch.Tensor:
    flattened = torch.sort(values.reshape(-1)).values
    if flattened.numel() < 1:
        raise ValueError("cannot summarize an empty empirical distribution")
    probabilities = flattened.new_tensor(tuple(grid))
    indices = (
        torch.ceil(probabilities * flattened.numel())
        .to(dtype=torch.long)
        .sub(1)
        .clamp(0, flattened.numel() - 1)
    )
    return flattened.index_select(0, indices)


@dataclass(frozen=True)
class DualSGWProxyOutput:
    core: torch.Tensor
    variation: torch.Tensor
    representation: torch.Tensor
    transition_mask: torch.Tensor
    laplacian_fidelity: torch.Tensor
    gw_fidelity: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DualSGWTransferOutput:
    spectral_quantization: torch.Tensor
    gw_quantization_proxy: torch.Tensor
    per_sample_spectral_quantization: torch.Tensor
    per_sample_gw_quantization_proxy: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class _ProxyWindowState:
    spectral_quantiles: torch.Tensor
    diffusion_quantiles: torch.Tensor
    laplacian: torch.Tensor
    diffusion_distance: torch.Tensor


class DualSGWProxy(nn.Module):
    """A variable-size, differentiable proxy with canonical 18/34 dimensions.

    It is not reported as exact GW.  Exact variable-size GW is computed only
    after freezing the selector.  Here, diffusion-distance distribution change
    is used as a stable GW-aligned scalar surrogate.
    """

    def __init__(
        self, config: Optional[DualSTSEHardSGWConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or DualSTSEHardSGWConfig()
        self.grid = _quantile_grid()
        self.laplacian = SignedLaplacianBuilder(
            self.config.laplacian_eta
        )
        self.spectral = SpectralStateExtractor(self.grid)
        self.heat = HeatKernelMetricBuilder(
            self.config.diffusion_time
        )

    @staticmethod
    def _window_adjacency(window) -> torch.Tensor:
        if hasattr(window, "adjacency_soft"):
            return window.adjacency_soft
        return window.adjacency_st

    @staticmethod
    def _window_node_mask(window) -> torch.Tensor:
        if hasattr(window, "node_mask"):
            return window.node_mask
        return window.hard_node_mask

    def _state(self, window) -> Optional[_ProxyWindowState]:
        if not window.window_valid:
            return None
        adjacency = self._window_adjacency(window)
        node_mask = self._window_node_mask(window).to(
            device=adjacency.device, dtype=torch.bool
        )
        laplacian = self.laplacian(
            adjacency, node_mask=node_mask
        )
        spectrum = self.spectral(laplacian, node_mask=node_mask)
        diffusion = self.heat(laplacian, node_mask=node_mask).distance
        upper = torch.triu(
            torch.ones_like(diffusion, dtype=torch.bool), diagonal=1
        )
        pair_values = diffusion[upper]
        if pair_values.numel() < 1:
            return None
        return _ProxyWindowState(
            spectral_quantiles=spectrum.quantiles,
            diffusion_quantiles=_empirical_quantiles(
                pair_values, self.grid
            ),
            laplacian=laplacian,
            diffusion_distance=diffusion,
        )

    def _sequence(
        self,
        windows: Sequence[Any],
        times: Sequence[float],
    ):
        if len(windows) != len(times) or len(windows) < 1:
            raise ValueError("proxy windows and times must be aligned")
        states = tuple(self._state(window) for window in windows)
        transitions = []
        valid_mask = []
        reference = self._window_adjacency(windows[0])
        for index in range(max(0, len(states) - 1)):
            left, right = states[index], states[index + 1]
            tau = float(times[index + 1]) - float(times[index])
            if tau <= 0.0:
                raise ValueError("proxy times must be strictly increasing")
            if left is None or right is None:
                transitions.append(
                    reference.new_zeros((self.config.sgw_core_dim,))
                )
                valid_mask.append(False)
                continue
            delta = (
                right.spectral_quantiles
                - left.spectral_quantiles
            )
            spectral_speed = delta.abs().mean() / tau
            diffusion_speed = (
                right.diffusion_quantiles
                - left.diffusion_quantiles
            ).abs().mean() / tau
            transitions.append(
                torch.cat(
                    (
                        delta,
                        spectral_speed.reshape(1),
                        diffusion_speed.reshape(1),
                    )
                )
            )
            valid_mask.append(True)
        if transitions:
            stacked = torch.stack(transitions, dim=0)
            mask = torch.tensor(
                valid_mask, dtype=torch.bool, device=stacked.device
            )
        else:
            stacked = reference.new_zeros(
                (0, self.config.sgw_core_dim)
            )
            mask = torch.zeros(
                0, dtype=torch.bool, device=reference.device
            )
        if bool(mask.any()):
            valid = stacked[mask]
            core = valid.mean(dim=0)
            variation = valid[:, :16].abs().mean(dim=0)
        else:
            core = reference.new_zeros((self.config.sgw_core_dim,))
            variation = reference.new_zeros(
                (self.config.sgw_variation_dim,)
            )
        return core, variation, mask, states

    def _fidelity(
        self,
        batch: ExactSTSEBatch,
        graph_windows: Sequence[Sequence[Any]],
        graph_states: Optional[
            Sequence[Sequence[Optional[_ProxyWindowState]]]
        ] = None,
    ):
        laplacian_terms: List[torch.Tensor] = []
        gw_terms: List[torch.Tensor] = []
        if graph_states is not None and len(graph_states) != len(
            graph_windows
        ):
            raise ValueError("proxy fidelity states must align with graphs")
        for sample_index, (exact_sample, windows) in enumerate(
            zip(batch, graph_windows)
        ):
            graph = exact_sample.graph
            for index, hard in enumerate(windows):
                if not hard.window_valid:
                    continue
                full_laplacian = self.laplacian(
                    graph.adjacency[index],
                    edge_mask=graph.edge_mask[index],
                )
                state = (
                    graph_states[sample_index][index]
                    if graph_states is not None
                    else None
                )
                if state is None:
                    graph_adjacency = self._window_adjacency(hard)
                    graph_laplacian = self.laplacian(
                        graph_adjacency,
                        edge_mask=graph.edge_mask[index],
                    )
                    graph_distance = self.heat(
                        graph_laplacian
                    ).distance
                else:
                    graph_laplacian = state.laplacian
                    graph_distance = state.diffusion_distance
                laplacian_terms.append(
                    laplacian_fidelity_metrics(
                        full_laplacian, graph_laplacian
                    ).normalized_frobenius_squared
                )
                full_distance = self.heat(full_laplacian).distance
                gw_terms.append(
                    gw_identity_coupling_upper_bound(
                        full_distance, graph_distance
                    ).squared_upper_bound
                )
        reference = self._window_adjacency(graph_windows[0][0])
        laplacian = (
            torch.stack(laplacian_terms).mean()
            if laplacian_terms
            else reference.new_zeros(())
        )
        gw = (
            torch.stack(gw_terms).mean()
            if gw_terms
            else reference.new_zeros(())
        )
        return laplacian, gw

    def forward(
        self,
        batch: ExactSTSEBatch,
        graph_windows: Sequence[Sequence[Any]],
        compute_fidelity: bool = True,
        reuse_fidelity_states: bool = False,
    ) -> DualSGWProxyOutput:
        if len(batch) != len(graph_windows) or len(batch) < 1:
            raise ValueError("proxy batch and hard graphs must be aligned")
        cores = []
        variations = []
        masks = []
        state_sequences = []
        for exact_sample, windows in zip(batch, graph_windows):
            core, variation, mask, states = self._sequence(
                windows,
                tuple(
                    float(value)
                    for value in exact_sample.graph.window_starts
                ),
            )
            cores.append(core)
            variations.append(variation)
            masks.append(mask)
            state_sequences.append(states)
        core_tensor = torch.stack(cores, dim=0)
        variation_tensor = torch.stack(variations, dim=0)
        representation = torch.cat(
            (core_tensor, variation_tensor), dim=-1
        )
        maximum = max(int(mask.numel()) for mask in masks)
        padded_mask = torch.zeros(
            (len(masks), maximum),
            dtype=torch.bool,
            device=representation.device,
        )
        for index, mask in enumerate(masks):
            padded_mask[index, : mask.numel()] = mask
        if compute_fidelity:
            laplacian, gw = self._fidelity(
                batch,
                graph_windows,
                graph_states=(
                    tuple(state_sequences)
                    if reuse_fidelity_states
                    else None
                ),
            )
        else:
            laplacian = representation.new_zeros(())
            gw = representation.new_zeros(())
        if tuple(representation.shape[1:]) != (
            self.config.sgw_output_dim,
        ):
            raise RuntimeError("proxy SGW representation is not 34-D")
        return DualSGWProxyOutput(
            core=core_tensor,
            variation=variation_tensor,
            representation=representation,
            transition_mask=padded_mask,
            laplacian_fidelity=laplacian,
            gw_fidelity=gw,
            diagnostics={
                "feature_semantics": "differentiable_sgw_aligned_proxy",
                "is_exact_gw": False,
                "valid_transition_count": int(padded_mask.sum()),
                "window_states": tuple(state_sequences),
            },
        )

    def compare_soft_and_hard(
        self,
        soft_windows: Sequence[Sequence[Any]],
        hard_windows: Sequence[Sequence[HardWindowOutput]],
        soft_states: Optional[
            Sequence[Sequence[Optional[_ProxyWindowState]]]
        ] = None,
        hard_states: Optional[
            Sequence[Sequence[Optional[_ProxyWindowState]]]
        ] = None,
    ) -> DualSGWTransferOutput:
        """Differentiable train-time proxies for soft-to-hard q terms."""

        if len(soft_windows) != len(hard_windows) or len(soft_windows) < 1:
            raise ValueError("soft and hard graph batches must align")
        spectral_per_sample = []
        gw_per_sample = []
        valid_window_count = 0
        for sample_index, (soft_sample, hard_sample) in enumerate(
            zip(soft_windows, hard_windows)
        ):
            if len(soft_sample) != len(hard_sample) or len(soft_sample) < 1:
                raise ValueError("soft and hard graph sequences must align")
            spectral_terms = []
            gw_terms = []
            for window_index, (soft_window, hard_window) in enumerate(
                zip(soft_sample, hard_sample)
            ):
                soft_state = (
                    soft_states[sample_index][window_index]
                    if soft_states is not None
                    else self._state(soft_window)
                )
                hard_state = (
                    hard_states[sample_index][window_index]
                    if hard_states is not None
                    else self._state(hard_window)
                )
                if soft_state is None or hard_state is None:
                    continue
                spectral_terms.append(
                    (
                        soft_state.spectral_quantiles
                        - hard_state.spectral_quantiles
                    )
                    .abs()
                    .mean()
                )
                gw_terms.append(
                    (
                        soft_state.diffusion_quantiles
                        - hard_state.diffusion_quantiles
                    )
                    .abs()
                    .mean()
                )
                valid_window_count += 1
            reference = self._window_adjacency(soft_sample[0])
            spectral_per_sample.append(
                torch.stack(spectral_terms).mean()
                if spectral_terms
                else reference.new_zeros(())
            )
            gw_per_sample.append(
                torch.stack(gw_terms).mean()
                if gw_terms
                else reference.new_zeros(())
            )
        spectral_tensor = torch.stack(spectral_per_sample)
        gw_tensor = torch.stack(gw_per_sample)
        return DualSGWTransferOutput(
            spectral_quantization=spectral_tensor.mean(),
            gw_quantization_proxy=gw_tensor.mean(),
            per_sample_spectral_quantization=spectral_tensor,
            per_sample_gw_quantization_proxy=gw_tensor,
            diagnostics={
                "gw_semantics": "diffusion_quantile_proxy_not_exact_gw",
                "valid_window_count": int(valid_window_count),
            },
        )
