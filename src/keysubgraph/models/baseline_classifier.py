"""Unified signed-subgraph sequence baseline classifier."""

from __future__ import absolute_import, division, print_function

import hashlib
import random
from dataclasses import dataclass

import torch
from torch import nn

from keysubgraph.data.baseline_collate import BaselineBatch
from keysubgraph.features.structural_prior import PRIOR_MODES

from .baseline_subgraph_encoder import SignedSubgraphEncoder, WindowMeanPooling


HISTORY_MODES = ("full", "current_only", "truncate_history", "independent_bag")
TEMPORAL_ORDERS = ("ordered", "shuffled")


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
    structural_interface_version: int = 0
    structural_group: str = "neutral"
    structural_feature_dim: int = 11
    structural_hidden_dim: int = 32
    prior_mode: str = "none"
    prior_beta: float = 1.0
    prior_permutation_seed: int = 42
    history_mode: str = "full"
    history_keep_ratio: float = 1.0
    temporal_order: str = "ordered"
    permutation_seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = (
            self.node_feature_dim,
            self.node_hidden_dim,
            self.signed_gnn_layers,
            self.fusion_dim,
            self.gru_hidden_dim,
            self.classifier_hidden_dim,
            self.num_classes,
            self.structural_feature_dim,
            self.structural_hidden_dim,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("baseline model dimensions must be positive")
        if self.num_classes != 2:
            raise ValueError("baseline currently supports binary classification only")
        if self.structural_interface_version not in (0, 1):
            raise ValueError("unsupported structural interface version")
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError("unsupported structural prior mode")
        if self.prior_beta < 0.0 or self.prior_permutation_seed < 0:
            raise ValueError("invalid structural prior configuration")
        if self.structural_interface_version == 0:
            if self.use_structural_features or self.prior_mode != "none":
                raise ValueError("legacy structural interface must remain neutral")
            if self.structural_group != "neutral":
                raise ValueError("legacy structural interface uses neutral group")
        else:
            if self.structural_group not in ("A", "B", "C", "D", "E"):
                raise ValueError("structural interface v1 requires group A-E")
            expected = {
                "A": (False, "none"),
                "B": (True, "none"),
                "C": (True, "uniform"),
                "D": (True, "real"),
                "E": (True, "permuted"),
            }[self.structural_group]
            if (self.use_structural_features, self.prior_mode) != expected:
                raise ValueError("structural group configuration differs from A-E design")
            if not self.use_structural_features and (
                self.structural_group != "A" or self.prior_mode != "none"
            ):
                raise ValueError("only structural group A uses zero features")
            if self.use_structural_features and self.structural_group == "A":
                raise ValueError("structural group A must use zero features")
        if self.history_mode not in HISTORY_MODES:
            raise ValueError("unsupported baseline history_mode")
        if self.history_keep_ratio <= 0.0 or self.history_keep_ratio > 1.0:
            raise ValueError("history_keep_ratio must be in (0, 1]")
        if self.history_mode != "truncate_history" and self.history_keep_ratio != 1.0:
            raise ValueError(
                "history_keep_ratio differs from 1 only for truncate_history"
            )
        if self.temporal_order not in TEMPORAL_ORDERS:
            raise ValueError("unsupported baseline temporal_order")
        if self.permutation_seed < 0:
            raise ValueError("permutation_seed must be non-negative")
        if self.temporal_order == "shuffled" and self.history_mode != "full":
            raise ValueError("shuffled temporal order currently requires history_mode='full'")


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
    sequence_window_index: torch.Tensor


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
        projection_input_dim = self.subgraph_encoder.output_dim
        if config.structural_interface_version == 1:
            self.structural_projection = nn.Sequential(
                nn.Linear(
                    config.structural_feature_dim,
                    config.structural_hidden_dim,
                    bias=False,
                ),
                nn.ReLU(),
            )
            self.register_buffer(
                "structural_mean", torch.zeros(config.structural_feature_dim)
            )
            self.register_buffer(
                "structural_std", torch.ones(config.structural_feature_dim)
            )
            self.register_buffer(
                "structural_prior_scale", torch.ones(config.structural_feature_dim)
            )
            self.register_buffer("structural_transform_fitted", torch.tensor(False))
            projection_input_dim += config.structural_hidden_dim
        self.input_projection = nn.Linear(
            projection_input_dim, config.fusion_dim
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

    def configure_structural_transform(
        self, mean: torch.Tensor, std: torch.Tensor, prior_scale: torch.Tensor
    ) -> None:
        """Freeze one training-fold transform into model buffers."""

        if self.config.structural_interface_version != 1:
            raise ValueError("legacy baseline has no structural transform")
        expected = (self.config.structural_feature_dim,)
        if mean.shape != expected or std.shape != expected or prior_scale.shape != expected:
            raise ValueError("structural transform dimension differs from model")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("structural transform contains non-finite values")
        if bool((std <= 0.0).any()) or not bool(torch.isfinite(prior_scale).all()):
            raise ValueError("structural scale must be finite and positive")
        if bool((prior_scale <= 0.0).any()):
            raise ValueError("structural prior scale must be positive")
        self.structural_mean.copy_(mean.to(self.structural_mean))
        self.structural_std.copy_(std.to(self.structural_std))
        self.structural_prior_scale.copy_(prior_scale.to(self.structural_prior_scale))
        self.structural_transform_fitted.fill_(True)

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

    def _sequence_window_index(self, batch: BaselineBatch) -> torch.Tensor:
        """Return the frozen per-sample window order used by the sequence model."""

        if self.config.temporal_order == "ordered":
            return batch.window_index
        sequence_index = batch.window_index.clone()
        for sample_index, sample_key in enumerate(batch.sample_keys):
            valid_count = int(batch.time_mask[sample_index].sum().item())
            if valid_count < 2:
                continue
            material = "{}\0{}".format(
                self.config.permutation_seed, sample_key
            ).encode("utf-8")
            stable_seed = int.from_bytes(
                hashlib.sha256(material).digest()[:8], byteorder="big"
            )
            order = list(range(valid_count))
            random.Random(stable_seed).shuffle(order)
            permutation = torch.tensor(
                order, dtype=torch.long, device=batch.window_index.device
            )
            sequence_index[sample_index, :valid_count] = batch.window_index[
                sample_index, :valid_count
            ].index_select(0, permutation)
        return sequence_index

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
        fusion_input = flat_window_embeddings
        if self.config.structural_interface_version == 1:
            if not bool(self.structural_transform_fitted):
                raise ValueError("structural transform has not been configured")
            if batch.window_structural_features.shape != (
                batch.window_count, self.config.structural_feature_dim
            ):
                raise ValueError("window structural feature dimension differs")
            if batch.window_structural_mask.shape != batch.window_structural_features.shape:
                raise ValueError("window structural mask differs")
            if self.config.use_structural_features:
                structural = (
                    batch.window_structural_features - self.structural_mean
                ) / self.structural_std
                structural = structural.masked_fill(~batch.window_structural_mask, 0.0)
                structural = structural * self.structural_prior_scale
            else:
                structural = batch.window_structural_features.new_zeros(
                    batch.window_count, self.config.structural_feature_dim
                )
            structural_embedding = self.structural_projection(structural)
            fusion_input = torch.cat(
                (flat_window_embeddings, structural_embedding), dim=-1
            )
        projected = self.input_projection(fusion_input)
        projected = self.input_activation(self.input_normalization(projected))
        sequence_window_index = self._sequence_window_index(batch)
        padded_windows = self._pad_windows(projected, sequence_window_index)

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
            sequence_window_index=sequence_window_index,
        )
