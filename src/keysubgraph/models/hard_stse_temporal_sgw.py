"""Single-model Hard-STSE-Temporal-SGW classifier."""

from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.hard_stse_classification_features import (
    HardSTSEClassificationFeatureBuilder,
)
from keysubgraph.features.hard_stse_extractor_features import (
    HardSTSEExtractorFeatureBuilder,
)
from keysubgraph.features.hard_stse_hard_graph import build_hard_stse_window
from keysubgraph.theory.spectral_gw import (
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    gw_identity_coupling_upper_bound,
    laplacian_fidelity_metrics,
)
from .hard_stse_selector import HardSTSEScorer, select_hard_stse_window
from .hard_stse_sgw_branch import HardSTSESGWBranch
from .hard_stse_temporal import HardSTSETemporalEncoder
from .hard_stse_types import (
    HardSTSEConfig,
    HardSTSEModelOutput,
    HardWindowOutput,
)
from .hard_stse_window_encoder import HardSTSEWindowEncoder


def _classification_head(
    input_dim: int, hidden_dims: Tuple[int, ...], dropout: float
) -> nn.Module:
    modules: List[nn.Module] = []
    current = int(input_dim)
    for hidden in hidden_dims:
        modules.extend(
            (
                nn.Linear(current, int(hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )
        current = int(hidden)
    modules.append(nn.Linear(current, 2))
    return nn.Sequential(*modules)


class HardSTSETemporalSGWClassifier(nn.Module):
    """Hard graph classifier; initially M0 and extended by later stages."""

    model_name = "hard_stse_temporal_sgw"

    def __init__(self, config: Optional[HardSTSEConfig] = None) -> None:
        super().__init__()
        self.config = config or HardSTSEConfig()
        self.classification_features = HardSTSEClassificationFeatureBuilder(
            epsilon=self.config.epsilon
        )
        self.extractor_features = HardSTSEExtractorFeatureBuilder(
            epsilon=self.config.epsilon
        )
        self.scorer = (
            HardSTSEScorer(self.config)
            if self.config.selection_mode == "learned"
            else None
        )
        self.window_encoder = HardSTSEWindowEncoder(self.config)
        self.temporal_encoder = HardSTSETemporalEncoder(self.config)
        self.neural_head = _classification_head(
            self.config.neural_output_dim,
            self.config.fusion_hidden_dims,
            self.config.dropout,
        )
        self.sgw_branch = (
            HardSTSESGWBranch(self.config) if self.config.use_sgw else None
        )
        self.theory_head = (
            _classification_head(
                self.config.spectral_fixed_dim,
                self.config.fusion_hidden_dims,
                self.config.dropout,
            )
            if self.config.use_sgw
            else None
        )
        self.fusion_head = (
            _classification_head(
                self.config.neural_output_dim + self.config.theory_output_dim,
                self.config.fusion_hidden_dims,
                self.config.dropout,
            )
            if self.config.use_sgw
            else None
        )
        self.proxy_laplacian = SignedLaplacianBuilder(
            self.config.laplacian_eta
        )
        self.proxy_heat = HeatKernelMetricBuilder(
            self.config.diffusion_time
        )

    def _select_window(
        self,
        sample,
        time_index: int,
        epoch: int,
        random_selection_seed: int,
    ) -> HardWindowOutput:
        adjacency = sample.adjacency[time_index]
        count = adjacency.shape[0]
        if self.config.selection_mode == "learned":
            extractor = self.extractor_features.build_timepoint(
                sample, time_index
            )
            scores = self.scorer(
                extractor.node_features,
                extractor.edge_base_features,
                extractor.edge_presence_mask,
            )
            node_probabilities = scores.node_probabilities
            edge_probabilities = scores.edge_probabilities
            node_ratio, edge_ratio = self.config.selection_schedule.ratios(
                epoch
            )
        else:
            node_probabilities = adjacency.new_full((count,), 0.5)
            edge_probabilities = sample.edge_mask[time_index].to(
                device=adjacency.device, dtype=adjacency.dtype
            ) * 0.5
            if self.config.selection_mode == "full":
                node_ratio, edge_ratio = 1.0, 1.0
            else:
                node_ratio = self.config.selection_schedule.target_node_ratio
                edge_ratio = self.config.selection_schedule.target_edge_ratio
        selection = select_hard_stse_window(
            node_probabilities=node_probabilities,
            edge_probabilities=edge_probabilities,
            communities=sample.communities[time_index].to(adjacency.device),
            edge_presence_mask=sample.edge_mask[time_index].to(adjacency.device),
            node_ratio=node_ratio,
            edge_ratio=edge_ratio,
            node_minimum=self.config.node_minimum,
            edge_minimum=self.config.edge_minimum,
            selection_mode=self.config.selection_mode,
            sample_key=sample.sample_key,
            time_index=time_index,
            random_seed=random_selection_seed,
        )
        return build_hard_stse_window(sample, time_index, selection)

    def _encode_neural(
        self,
        batch: GraphSequenceBatch,
        epoch: int,
        random_selection_seed: int,
    ):
        sample_sequences = []
        all_hard_windows = []
        window_encodings = []
        for sample in batch:
            hard_windows = []
            embeddings = []
            previous = None
            for time_index in range(sample.num_timepoints):
                hard = self._select_window(
                    sample, time_index, epoch, random_selection_seed
                )
                hard_windows.append(hard)
                if not hard.window_valid:
                    previous = None
                    continue
                features = self.classification_features.build_timepoint(
                    sample, time_index, hard, previous
                )
                encoded = self.window_encoder(features)
                embeddings.append(encoded.embedding)
                window_encodings.append(encoded)
                previous = hard
            if not embeddings:
                raise ValueError("sample has no valid hard windows")
            sample_sequences.append(torch.stack(embeddings, dim=0))
            all_hard_windows.append(tuple(hard_windows))
        temporal = self.temporal_encoder(tuple(sample_sequences))
        return temporal, tuple(all_hard_windows), tuple(window_encodings)

    def _theory_proxies(
        self,
        batch: GraphSequenceBatch,
        hard_windows: Tuple[Tuple[HardWindowOutput, ...], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        laplacian_terms = []
        gw_terms = []
        for sample, sample_windows in zip(batch, hard_windows):
            for time_index, hard in enumerate(sample_windows):
                if not hard.window_valid:
                    continue
                adjacency = sample.adjacency[time_index]
                edge_mask = sample.edge_mask[time_index]
                full_laplacian = self.proxy_laplacian(
                    adjacency, edge_mask=edge_mask
                )
                hard_laplacian = self.proxy_laplacian(
                    hard.adjacency_st, edge_mask=edge_mask
                )
                laplacian_terms.append(
                    laplacian_fidelity_metrics(
                        full_laplacian, hard_laplacian
                    ).normalized_frobenius_squared
                )
                full_distance = self.proxy_heat(full_laplacian).distance
                hard_distance = self.proxy_heat(hard_laplacian).distance
                gw_terms.append(
                    gw_identity_coupling_upper_bound(
                        full_distance, hard_distance
                    ).squared_upper_bound
                )
        reference = next(self.parameters())
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
        batch: GraphSequenceBatch,
        epoch: int = 1,
        random_selection_seed: int = 42,
        compute_theory_proxies: bool = False,
    ) -> HardSTSEModelOutput:
        if len(batch) < 1:
            raise ValueError("Hard-STSE cannot process an empty batch")
        temporal, hard_windows, window_encodings = self._encode_neural(
            batch, epoch, random_selection_seed
        )
        neural_logits = self.neural_head(temporal.representation)
        theory_output = None
        theory_diagnostics = {}
        if self.sgw_branch is not None:
            time_values = tuple(
                tuple(float(value) for value in sample.window_starts)
                for sample in batch
            )
            theory_output, theory_diagnostics = self.sgw_branch(
                hard_windows, time_values
            )
            theory_logits = self.theory_head(theory_output.fixed)
            final_representation = torch.cat(
                (temporal.representation, theory_output.representation),
                dim=-1,
            )
            fusion_logits = self.fusion_head(final_representation)
        else:
            theory_logits = None
            final_representation = temporal.representation
            fusion_logits = neural_logits
        selections = [
            window.selection
            for sample_windows in hard_windows
            for window in sample_windows
        ]
        valid_selections = [
            item for item in selections if item.actual_edge_count > 0
        ]
        reference = neural_logits
        node_probability_mean = (
            torch.cat([item.node_probabilities.reshape(-1) for item in selections]).mean()
            if selections
            else reference.new_zeros(())
        )
        edge_probability_values = [
            item.edge_probabilities[item.edge_probabilities > 0.0]
            for item in selections
        ]
        edge_probability_values = [
            item for item in edge_probability_values if item.numel()
        ]
        edge_probability_mean = (
            torch.cat(edge_probability_values).mean()
            if edge_probability_values
            else reference.new_zeros(())
        )
        laplacian_proxy = reference.new_zeros(())
        gw_proxy = reference.new_zeros(())
        if compute_theory_proxies:
            laplacian_proxy, gw_proxy = self._theory_proxies(
                batch, hard_windows
            )
        diagnostics: Dict[str, object] = {
            "window_encodings": window_encodings,
            "temporal": temporal,
            "variant": self.config.variant,
            "node_probability_mean": node_probability_mean,
            "edge_probability_mean": edge_probability_mean,
            "requested_node_ratio": (
                sum(item.requested_node_count for item in selections)
                / float(sum(item.node_probabilities.numel() for item in selections))
            ),
            "actual_node_ratio": (
                sum(item.actual_node_count for item in selections)
                / float(sum(item.node_probabilities.numel() for item in selections))
            ),
            "requested_edge_count": sum(
                item.requested_edge_count for item in selections
            ),
            "actual_edge_count": sum(
                item.actual_edge_count for item in selections
            ),
            "candidate_edge_count": sum(
                item.candidate_edge_count for item in selections
            ),
            "original_edge_count": sum(
                item.original_edge_count for item in selections
            ),
            "actual_edge_candidate_ratio": (
                sum(item.actual_edge_count for item in selections)
                / float(max(
                    1,
                    sum(
                        item.candidate_edge_count
                        for item in selections
                    ),
                ))
            ),
            "actual_edge_original_ratio": (
                sum(item.actual_edge_count for item in selections)
                / float(max(
                    1,
                    sum(
                        item.original_edge_count
                        for item in selections
                    ),
                ))
            ),
            "valid_window_count": len(valid_selections),
            "total_window_count": len(selections),
            "laplacian_proxy": laplacian_proxy,
            "gw_proxy": gw_proxy,
            "selections": tuple(selections),
            "theory": theory_output,
            "theory_diagnostics": theory_diagnostics,
        }
        return HardSTSEModelOutput(
            fusion_logits=fusion_logits,
            neural_logits=neural_logits,
            theory_logits=theory_logits,
            neural_representation=temporal.representation,
            theory_representation=(
                theory_output.representation
                if theory_output is not None
                else None
            ),
            final_representation=final_representation,
            hard_windows=hard_windows,
            diagnostics=diagnostics,
        )
