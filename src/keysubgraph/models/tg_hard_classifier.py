"""Stage-C hard-graph neural branch and 226-D TG-SGW fusion classifier."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.features.hard_graph_cache import (
    CachedHardSubgraph,
    CachedHardWindow,
    HardGraphSampleCache,
)

from .masked_pooling import MaskedGraphPooling
from .masked_tcn import MaskedTCNEncoder
from .signed_graph_encoder import SignedGraphEncoder


@dataclass(frozen=True)
class TGHardClassifierConfig:
    node_feature_dim: int = 13
    signed_gnn_hidden_dim: int = 64
    signed_gnn_layers: int = 3
    graph_embedding_dim: int = 96
    temporal_hidden_dim: int = 96
    temporal_kernel_size: int = 3
    temporal_dilations: Tuple[int, ...] = (1, 2, 4)
    theory_feature_dim: int = 34
    final_representation_dim: int = 226
    classifier_hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.3
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.node_feature_dim != 13 or self.graph_embedding_dim != 96:
            raise ValueError("TG hard classifier requires 13-D nodes and 96-D graphs")
        if self.temporal_hidden_dim != 96 or self.theory_feature_dim != 34:
            raise ValueError("TG hard classifier requires 192-D neural and 34-D theory branches")
        if self.final_representation_dim != 226:
            raise ValueError("TG hard classifier final representation must be 226-D")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("TG hard classifier dropout must lie in [0, 1)")
        if self.signed_gnn_layers < 1 or any(value < 1 for value in self.classifier_hidden_dims):
            raise ValueError("TG hard classifier depths and widths must be positive")


@dataclass(frozen=True)
class TGHardClassifierOutput:
    logits: torch.Tensor
    final_representation: torch.Tensor
    neural_representation: torch.Tensor
    theory_representation: torch.Tensor
    encoded_windows: torch.Tensor
    time_mask: torch.Tensor


class TGHardSGWClassifier(nn.Module):
    training_stage = "hard_student"

    def __init__(self, config: Optional[TGHardClassifierConfig] = None) -> None:
        super().__init__()
        self.config = config or TGHardClassifierConfig()
        self.graph_encoder = SignedGraphEncoder(
            input_dim=self.config.node_feature_dim,
            hidden_dim=self.config.signed_gnn_hidden_dim,
            num_layers=self.config.signed_gnn_layers,
            dropout=self.config.dropout,
            epsilon=self.config.epsilon,
        )
        self.graph_pooling = MaskedGraphPooling(
            input_dim=self.config.signed_gnn_hidden_dim,
            output_dim=self.config.graph_embedding_dim,
            dropout=self.config.dropout,
            epsilon=self.config.epsilon,
        )
        self.set_phi = nn.Sequential(
            nn.Linear(96, 96), nn.GELU(), nn.Dropout(self.config.dropout)
        )
        self.set_rho = nn.Sequential(
            nn.Linear(96, 96), nn.GELU(), nn.Dropout(self.config.dropout)
        )
        self.window_projection = nn.Sequential(
            nn.Linear(192, 96), nn.GELU(), nn.Dropout(self.config.dropout)
        )
        self.temporal_encoder = MaskedTCNEncoder(
            input_dim=96,
            hidden_dim=self.config.temporal_hidden_dim,
            kernel_size=self.config.temporal_kernel_size,
            dilations=self.config.temporal_dilations,
            dropout=self.config.dropout,
            epsilon=self.config.epsilon,
        )
        self.neural_normalization = nn.LayerNorm(192)
        layers = []
        current = self.config.final_representation_dim
        for hidden in self.config.classifier_hidden_dims:
            layers.extend(
                (nn.Linear(current, hidden), nn.GELU(), nn.Dropout(self.config.dropout))
            )
            current = hidden
        layers.append(nn.Linear(current, 2))
        self.classifier = nn.Sequential(*layers)

    def _device_dtype(self):
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def _encode_graph(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        device, dtype = self._device_dtype()
        features = node_features.to(device=device, dtype=dtype)
        signed_adjacency = adjacency.to(device=device, dtype=dtype)
        hidden = self.graph_encoder(features, signed_adjacency)
        return self.graph_pooling(hidden)

    def _encode_subgraph(
        self, window: CachedHardWindow, subgraph: CachedHardSubgraph
    ) -> torch.Tensor:
        indices = subgraph.union_node_indices.to(
            device=window.features.node_features.device
        )
        node_features = window.features.node_features.index_select(0, indices)
        return self._encode_graph(node_features, subgraph.adjacency)

    def _window_embedding(self, window: CachedHardWindow) -> torch.Tensor:
        if not window.subgraphs:
            raise ValueError("a valid hard union requires at least one selected subgraph")
        union_embedding = self._encode_graph(
            window.features.node_features, window.graph.adjacency
        )
        subgraph_embeddings = torch.stack(
            [self._encode_subgraph(window, item) for item in window.subgraphs], dim=0
        )
        device, dtype = self._device_dtype()
        scores = torch.tensor(
            [item.candidate_score for item in window.subgraphs],
            device=device,
            dtype=dtype,
        )
        weights = torch.sigmoid(scores)
        transformed = self.set_phi(subgraph_embeddings)
        set_embedding = self.set_rho(
            (transformed * weights[:, None]).sum(dim=0)
            / weights.sum().clamp_min(self.config.epsilon)
        )
        return self.window_projection(torch.cat((set_embedding, union_embedding), dim=0))

    def _neural_batch(
        self, hard_batch: Sequence[HardGraphSampleCache]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hard_batch:
            raise ValueError("TG hard classifier cannot process an empty batch")
        if any(not sample.eligible_for_stage_c for sample in hard_batch):
            raise ValueError("ineligible hard samples cannot enter Stage C")
        device, dtype = self._device_dtype()
        max_windows = max(len(sample.windows) for sample in hard_batch)
        padded = torch.zeros(
            (len(hard_batch), max_windows, 96), device=device, dtype=dtype
        )
        mask = torch.zeros(
            (len(hard_batch), max_windows), device=device, dtype=torch.bool
        )
        for sample_index, sample in enumerate(hard_batch):
            for time_index, window in enumerate(sample.windows):
                if window is None:
                    continue
                padded[sample_index, time_index] = self._window_embedding(window)
                mask[sample_index, time_index] = True
        if bool((mask.sum(dim=1) < 2).any()):
            raise ValueError("Stage C requires at least two valid hard windows per sample")
        representation, encoded = self.temporal_encoder(padded, mask)
        return representation, encoded, mask

    def forward(
        self,
        hard_batch: Sequence[HardGraphSampleCache],
        standardized_sgw_features: torch.Tensor,
    ) -> TGHardClassifierOutput:
        if standardized_sgw_features.ndim != 2 or standardized_sgw_features.shape[1] != 34:
            raise ValueError("standardized SGW features must have shape [B, 34]")
        if len(hard_batch) != standardized_sgw_features.shape[0]:
            raise ValueError("hard batch and SGW feature batch sizes differ")
        neural, encoded, time_mask = self._neural_batch(hard_batch)
        theory = standardized_sgw_features.to(device=neural.device, dtype=neural.dtype)
        final = torch.cat((self.neural_normalization(neural), theory), dim=-1)
        if final.shape[1] != self.config.final_representation_dim:
            raise RuntimeError("TG hard fusion did not produce 226 dimensions")
        logits = self.classifier(final)
        return TGHardClassifierOutput(
            logits=logits,
            final_representation=final,
            neural_representation=neural,
            theory_representation=theory,
            encoded_windows=encoded,
            time_mask=time_mask,
        )
