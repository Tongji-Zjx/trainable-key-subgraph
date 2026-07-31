"""Stage-1 N0--N4 neural experts for frozen hard key-graph sequences."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from keysubgraph.features.sv_hard_graph_features import SV_NODE_FEATURE_DIM
from keysubgraph.features.theory_neural_features import THEORY_EDGE_FEATURE_DIM
from keysubgraph.models.sv_signed_gin import (
    SV_DEFAULT_VARIANT,
    SVSignedGINConfig,
    SVSignedGINSampleInput,
    SVSignedGINWindowInput,
    SignedGINKeySubgraphEncoder,
)
from keysubgraph.theory.sgw_core_features import SGW_CORE_DIM, SGW_QUANTILE_DIM


THEORY_NEURAL_VARIANTS = (
    "N0_signed_gin",
    "N1_edge_aware",
    "N2_spectral_film",
    "N3_theory_reconstruction",
    "N4_ema_center",
)


@dataclass(frozen=True)
class TheoryNeuralConfig:
    variant: str = "N1_edge_aware"
    node_feature_dim: int = SV_NODE_FEATURE_DIM
    edge_feature_dim: int = THEORY_EDGE_FEATURE_DIM
    quantile_dim: int = SGW_QUANTILE_DIM
    transition_dim: int = SGW_CORE_DIM
    hidden_dim: int = 64
    layers: int = 2
    classifier_hidden_dim: int = 16
    dropout: float = 0.10

    def __post_init__(self):
        if self.variant not in THEORY_NEURAL_VARIANTS:
            raise ValueError("unsupported Stage-1 neural variant")
        if (
            self.node_feature_dim != 15
            or self.edge_feature_dim != 6
            or self.quantile_dim != 16
            or self.transition_dim != 18
        ):
            raise ValueError("Stage-1 feature dimensions are frozen")
        if self.hidden_dim != 64 or self.layers != 2:
            raise ValueError("Stage-1 first implementation is frozen to H=64,L=2")
        if self.classifier_hidden_dim < 1 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid Stage-1 architecture configuration")

    @property
    def uses_edge_aware(self):
        return self.variant != "N0_signed_gin"

    @property
    def uses_film(self):
        return self.variant in (
            "N2_spectral_film",
            "N3_theory_reconstruction",
            "N4_ema_center",
        )

    @property
    def uses_reconstruction(self):
        return self.variant in (
            "N3_theory_reconstruction",
            "N4_ema_center",
        )

    @property
    def uses_center_loss(self):
        return self.variant == "N4_ema_center"


@dataclass(frozen=True)
class TheoryNeuralWindowInput:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    edge_features: torch.Tensor
    spectral_quantiles: torch.Tensor

    def to(self, device):
        return TheoryNeuralWindowInput(
            self.node_features.to(device),
            self.adjacency.to(device),
            self.edge_features.to(device),
            self.spectral_quantiles.to(device),
        )


@dataclass(frozen=True)
class TheoryNeuralSampleInput:
    sample_key: str
    label: int
    windows: Tuple[Optional[TheoryNeuralWindowInput], ...]
    window_mask: torch.Tensor
    transition_targets: torch.Tensor
    transition_mask: torch.Tensor

    def to(self, device):
        return TheoryNeuralSampleInput(
            self.sample_key,
            int(self.label),
            tuple(
                window.to(device) if window is not None else None
                for window in self.windows
            ),
            self.window_mask.to(device),
            self.transition_targets.to(device),
            self.transition_mask.to(device),
        )


@dataclass(frozen=True)
class TheoryNeuralBatch:
    samples: Tuple[TheoryNeuralSampleInput, ...]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def labels(self):
        return torch.tensor([sample.label for sample in self.samples], dtype=torch.long)

    @property
    def sample_keys(self):
        return tuple(sample.sample_key for sample in self.samples)

    def to(self, device):
        return TheoryNeuralBatch(tuple(sample.to(device) for sample in self.samples))


@dataclass(frozen=True)
class TheoryNeuralSampleOutput:
    representation: torch.Tensor
    window_embeddings: torch.Tensor
    window_mask: torch.Tensor
    q_predictions: Optional[torch.Tensor]
    q_targets: Optional[torch.Tensor]
    gamma_predictions: Optional[torch.Tensor]
    gamma_targets: Optional[torch.Tensor]
    message_norms: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    film_values: Optional[torch.Tensor]


@dataclass(frozen=True)
class TheoryNeuralOutput:
    logits: torch.Tensor
    representations: torch.Tensor
    samples: Tuple[TheoryNeuralSampleOutput, ...]
    diagnostics: Dict[str, Any]


class EdgeAwareSignedLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout):
        super().__init__()
        message_dim = 2 * hidden_dim + edge_dim
        self.positive = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.negative = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, states, adjacency, edge_features):
        count, hidden = states.shape
        if tuple(adjacency.shape) != (count, count) or tuple(edge_features.shape) != (
            count, count, THEORY_EDGE_FEATURE_DIM
        ):
            raise ValueError("edge-aware tensors are misaligned")
        edge_mask = adjacency.abs() > 0.0
        indices = torch.nonzero(edge_mask, as_tuple=False)
        positive_aggregate = states.new_zeros((count, hidden))
        negative_aggregate = states.new_zeros((count, hidden))
        if indices.numel() > 0:
            destination = indices[:, 0]
            source = indices[:, 1]
            pair = torch.cat(
                (
                    states.index_select(0, destination),
                    states.index_select(0, source),
                    edge_features[destination, source],
                ),
                dim=-1,
            )
            degree = adjacency.abs().sum(dim=-1).clamp_min(1.0e-8)
            weight = adjacency[destination, source].abs() / torch.sqrt(
                degree.index_select(0, destination)
                * degree.index_select(0, source)
            )
            positive_mask = adjacency[destination, source] > 0.0
            negative_mask = adjacency[destination, source] < 0.0
            if bool(positive_mask.any()):
                positive_aggregate.index_add_(
                    0,
                    destination[positive_mask],
                    self.positive(pair[positive_mask]) * weight[positive_mask, None],
                )
            if bool(negative_mask.any()):
                negative_aggregate.index_add_(
                    0,
                    destination[negative_mask],
                    self.negative(pair[negative_mask]) * weight[negative_mask, None],
                )
        update_input = torch.cat(
            (
                states,
                positive_aggregate,
                negative_aggregate,
                positive_aggregate - negative_aggregate,
            ),
            dim=-1,
        )
        output = self.norm(states + self.update(update_input))
        norms = (
            positive_aggregate.norm(dim=-1).mean(),
            negative_aggregate.norm(dim=-1).mean(),
            (positive_aggregate - negative_aggregate).norm(dim=-1).mean(),
        )
        return output, norms


class EdgeAwareSignedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_projection = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                EdgeAwareSignedLayer(
                    config.hidden_dim, config.edge_feature_dim, config.dropout
                )
                for _ in range(config.layers)
            ]
        )
        self.jumping_projection = nn.Linear(
            config.hidden_dim * (config.layers + 1), config.hidden_dim
        )
        self.film = nn.Sequential(
            nn.Linear(config.quantile_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.window_projection = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, window, use_film):
        states = self.node_projection(window.node_features)
        history = [states]
        norms = []
        for layer in self.layers:
            states, current_norms = layer(
                states, window.adjacency, window.edge_features
            )
            history.append(states)
            norms.append(current_norms)
        states = self.jumping_projection(torch.cat(history, dim=-1))
        film_values = None
        if use_film:
            film_values = self.film(window.spectral_quantiles)
            gamma, beta = film_values.chunk(2, dim=-1)
            states = (1.0 + gamma[None, :]) * states + beta[None, :]
        mean = states.mean(dim=0)
        std = torch.sqrt((states - mean).square().mean(dim=0) + 1.0e-8)
        return self.window_projection(torch.cat((mean, std), dim=-1)), tuple(norms), film_values


class TheoryGuidedNeuralClassifier(nn.Module):
    model_name = "svg_theory_guided_neural_stage1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or TheoryNeuralConfig()
        if self.config.uses_edge_aware:
            self.edge_encoder = EdgeAwareSignedEncoder(self.config)
            self.gin_encoder = None
        else:
            gin_config = SVSignedGINConfig(
                variant=SV_DEFAULT_VARIANT,
                gin_hidden_dim=self.config.hidden_dim,
                gin_layers=self.config.layers,
                dropout=self.config.dropout,
            )
            self.gin_encoder = SignedGINKeySubgraphEncoder(gin_config)
            self.edge_encoder = None
        self.sample_projection = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_dim, 2),
        )
        if self.config.uses_reconstruction:
            self.q_head = nn.Linear(self.config.hidden_dim, self.config.quantile_dim)
            self.gamma_pair = nn.Sequential(
                nn.Linear(4 * self.config.hidden_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            )
            self.gamma_head = nn.Linear(
                self.config.hidden_dim, self.config.transition_dim
            )
        else:
            self.q_head = None
            self.gamma_pair = None
            self.gamma_head = None

    def config_dict(self):
        return asdict(self.config)

    def _encode_n0(self, sample):
        valid = [window for window in sample.windows if window is not None]
        converted = SVSignedGINSampleInput(
            sample_key=sample.sample_key,
            label=sample.label,
            windows=tuple(
                SVSignedGINWindowInput(window.node_features, window.adjacency)
                for window in valid
            ),
            static_features=valid[0].node_features.new_zeros((28,)),
            variation=valid[0].node_features.new_zeros((16,)),
        )
        output = self.gin_encoder(converted)
        return output.representation, torch.stack(output.window_embeddings, dim=0)

    def _encode_edge(self, sample):
        embeddings = []
        positions = []
        all_norms = []
        films = []
        for index, window in enumerate(sample.windows):
            if window is None:
                continue
            embedding, norms, film = self.edge_encoder(
                window, self.config.uses_film
            )
            embeddings.append(embedding)
            positions.append(index)
            all_norms.append(norms)
            if film is not None:
                films.append(film)
        stacked = torch.stack(embeddings, dim=0)
        mean = stacked.mean(dim=0)
        std = torch.sqrt((stacked - mean).square().mean(dim=0) + 1.0e-8)
        representation = self.sample_projection(torch.cat((mean, std), dim=-1))
        q_predictions = self.q_head(stacked) if self.q_head is not None else None
        q_targets = (
            torch.stack(
                [sample.windows[index].spectral_quantiles for index in positions],
                dim=0,
            )
            if q_predictions is not None
            else None
        )
        gamma_predictions = []
        gamma_targets = []
        if self.gamma_head is not None:
            position_to_embedding = {
                position: stacked[offset] for offset, position in enumerate(positions)
            }
            for index in range(len(sample.windows) - 1):
                if not bool(sample.transition_mask[index]):
                    continue
                left = position_to_embedding[index]
                right = position_to_embedding[index + 1]
                pair = torch.cat((left, right, right - left, (right - left).abs()))
                gamma_predictions.append(self.gamma_head(self.gamma_pair(pair)))
                gamma_targets.append(sample.transition_targets[index])
        return TheoryNeuralSampleOutput(
            representation=representation,
            window_embeddings=stacked,
            window_mask=sample.window_mask,
            q_predictions=q_predictions,
            q_targets=q_targets,
            gamma_predictions=(
                torch.stack(gamma_predictions, dim=0)
                if gamma_predictions
                else stacked.new_zeros((0, self.config.transition_dim))
            ) if self.gamma_head is not None else None,
            gamma_targets=(
                torch.stack(gamma_targets, dim=0)
                if gamma_targets
                else stacked.new_zeros((0, self.config.transition_dim))
            ) if self.gamma_head is not None else None,
            message_norms=tuple(all_norms),
            film_values=torch.stack(films, dim=0) if films else None,
        )

    def forward(self, batch):
        if len(batch) < 1:
            raise ValueError("Stage-1 batch cannot be empty")
        outputs = []
        for sample in batch:
            if self.config.uses_edge_aware:
                outputs.append(self._encode_edge(sample))
            else:
                representation, windows = self._encode_n0(sample)
                outputs.append(
                    TheoryNeuralSampleOutput(
                        representation=representation,
                        window_embeddings=windows,
                        window_mask=sample.window_mask,
                        q_predictions=None,
                        q_targets=None,
                        gamma_predictions=None,
                        gamma_targets=None,
                        message_norms=(),
                        film_values=None,
                    )
                )
        representations = torch.stack(
            [output.representation for output in outputs], dim=0
        )
        logits = self.classifier(representations)
        return TheoryNeuralOutput(
            logits=logits,
            representations=representations,
            samples=tuple(outputs),
            diagnostics={
                "variant": self.config.variant,
                "uses_coordinates": False,
                "uses_site_input": False,
                "uses_raw_community_embedding": False,
                "uses_edge_aware_signed_channels": self.config.uses_edge_aware,
                "uses_spectral_film": self.config.uses_film,
                "uses_auxiliary_reconstruction": self.config.uses_reconstruction,
                "uses_ema_center_loss": self.config.uses_center_loss,
            },
        )
