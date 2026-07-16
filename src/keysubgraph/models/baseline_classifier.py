"""Unified signed-subgraph sequence baseline classifier."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass

import torch
from torch import nn

from keysubgraph.data.baseline_collate import BaselineBatch

from .baseline_subgraph_encoder import SignedSubgraphEncoder, WindowMeanPooling


HISTORY_MODES = ("full", "current_only", "truncate_history", "independent_bag")


@dataclass(frozen=True)
class BaselineModelConfig:
    node_feature_dim: int = 12
    node_hidden_dim: int = 64
    signed_gnn_layers: int = 2
    signed_gnn_dropout: float = 0.1
    use_residual: bool = True
    fusion_dim: int = 128
    gru_hidden_dim: int = 128
    classifier_hidden_dim: int = 64
    classifier_dropout: float = 0.2
    num_classes: int = 2
    use_structural_features: bool = False
    history_mode: str = "full"
    history_keep_ratio: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.node_feature_dim,
            self.node_hidden_dim,
            self.signed_gnn_layers,
            self.fusion_dim,
            self.gru_hidden_dim,
            self.classifier_hidden_dim,
            self.num_classes,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("baseline model dimensions must be positive")
        if self.num_classes != 2:
            raise ValueError("baseline currently supports binary classification only")
        if self.use_structural_features:
            raise ValueError("structural features are not enabled in the neutral baseline")
        if self.history_mode not in HISTORY_MODES:
            raise ValueError("unsupported baseline history_mode")
        if self.history_keep_ratio <= 0.0 or self.history_keep_ratio > 1.0:
            raise ValueError("history_keep_ratio must be in (0, 1]")
        if self.history_mode != "truncate_history" and self.history_keep_ratio != 1.0:
            raise ValueError(
                "history_keep_ratio differs from 1 only for truncate_history"
            )


@dataclass(frozen=True)
class BaselineModelOutput:
    logits: torch.Tensor
    subgraph_embeddings: torch.Tensor
    window_embeddings: torch.Tensor
    padded_window_embeddings: torch.Tensor
    hidden_states: torch.Tensor
    final_hidden_state: torch.Tensor
    time_mask: torch.Tensor
    history_mask: torch.Tensor


class SignedSequenceBaseline(nn.Module):
    """Signed subgraph encoder -> window mean -> GRU -> binary MLP."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.config = config
        self.subgraph_encoder = SignedSubgraphEncoder(
            node_feature_dim=config.node_feature_dim,
            hidden_dim=config.node_hidden_dim,
            layers=config.signed_gnn_layers,
            dropout=config.signed_gnn_dropout,
            residual=config.use_residual,
        )
        self.window_pooling = WindowMeanPooling()
        self.input_projection = nn.Linear(
            self.subgraph_encoder.output_dim, config.fusion_dim
        )
        self.input_normalization = nn.LayerNorm(config.fusion_dim)
        self.input_activation = nn.ReLU()
        self.gru_cell = nn.GRUCell(config.fusion_dim, config.gru_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.gru_hidden_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    @staticmethod
    def _pad_windows(
        flat_windows: torch.Tensor, window_index: torch.Tensor
    ) -> torch.Tensor:
        if flat_windows.dim() != 2 or window_index.dim() != 2:
            raise ValueError("invalid window embedding metadata")
        sentinel = flat_windows.new_zeros(1, flat_windows.shape[-1])
        values = torch.cat((flat_windows, sentinel), dim=0)
        safe_index = window_index.clone()
        safe_index[safe_index < 0] = flat_windows.shape[0]
        return values.index_select(0, safe_index.reshape(-1)).reshape(
            window_index.shape[0], window_index.shape[1], flat_windows.shape[-1]
        )

    def _history_mask(self, time_mask: torch.Tensor) -> torch.Tensor:
        """Return windows allowed to update recurrent state for this condition."""

        if time_mask.dim() != 2 or not bool(time_mask.any(dim=1).all()):
            raise ValueError("each baseline sample needs at least one valid timepoint")
        mode = self.config.history_mode
        if mode in ("full", "independent_bag"):
            return time_mask
        valid_counts = time_mask.sum(dim=1)
        positions = torch.arange(time_mask.shape[1], device=time_mask.device)
        if mode == "current_only":
            return time_mask & (positions.unsqueeze(0) == (valid_counts - 1).unsqueeze(1))
        keep_counts = torch.ceil(
            valid_counts.to(dtype=torch.float32) * self.config.history_keep_ratio
        ).to(dtype=torch.long).clamp_min(1)
        starts = valid_counts - keep_counts
        return time_mask & (positions.unsqueeze(0) >= starts.unsqueeze(1))

    def _encode_history(
        self, padded_windows: torch.Tensor, time_mask: torch.Tensor
    ):
        history_mask = self._history_mask(time_mask)
        zero_state = padded_windows.new_zeros(
            padded_windows.shape[0], self.config.gru_hidden_dim
        )
        hidden_states = []
        if self.config.history_mode == "independent_bag":
            state_sum = zero_state
            for time_index in range(padded_windows.shape[1]):
                candidate = self.gru_cell(
                    padded_windows[:, time_index], zero_state
                )
                valid = history_mask[:, time_index].unsqueeze(-1)
                independent_state = torch.where(
                    valid, candidate, torch.zeros_like(candidate)
                )
                state_sum = state_sum + independent_state
                hidden_states.append(independent_state)
            denominator = history_mask.sum(dim=1).clamp_min(1).to(
                dtype=padded_windows.dtype
            ).unsqueeze(-1)
            state = state_sum / denominator
        else:
            state = zero_state
            for time_index in range(padded_windows.shape[1]):
                candidate = self.gru_cell(padded_windows[:, time_index], state)
                active = history_mask[:, time_index].unsqueeze(-1)
                state = torch.where(active, candidate, state)
                hidden_states.append(state)
        if not hidden_states:
            raise ValueError("baseline batch contains no timepoints")
        return state, torch.stack(hidden_states, dim=1), history_mask

    def forward(self, batch: BaselineBatch) -> BaselineModelOutput:
        if batch.node_feature_dim != self.config.node_feature_dim:
            raise ValueError("batch node feature dimension does not match model")
        subgraph_embeddings = self.subgraph_encoder(
            batch.node_features, batch.adjacency, batch.node_mask
        )
        flat_window_embeddings = self.window_pooling(
            subgraph_embeddings,
            batch.subgraph_to_window,
            batch.window_count,
            batch.window_subgraph_count,
        )
        projected = self.input_projection(flat_window_embeddings)
        projected = self.input_activation(self.input_normalization(projected))
        padded_windows = self._pad_windows(projected, batch.window_index)

        state, stacked_states, history_mask = self._encode_history(
            padded_windows, batch.time_mask
        )
        logits = self.classifier(state)
        return BaselineModelOutput(
            logits=logits,
            subgraph_embeddings=subgraph_embeddings,
            window_embeddings=flat_window_embeddings,
            padded_window_embeddings=padded_windows,
            hidden_states=stacked_states,
            final_hidden_state=state,
            time_mask=batch.time_mask,
            history_mask=history_mask,
        )
