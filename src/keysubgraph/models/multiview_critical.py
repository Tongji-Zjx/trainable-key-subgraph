"""Revised S/V/G critical channel and author short-term residual fusion."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from keysubgraph.features.multiview_critical import (
    MULTIVIEW_DELTA_Q_DIM,
    MULTIVIEW_Q_DIM,
    MULTIVIEW_SPECTRAL_FEATURE_DIM,
    MULTIVIEW_STABLE_STATIC_DIM,
    MultiViewCriticalBatch,
)


MULTIVIEW_CRITICAL_MODEL_NAME = "theory_guided_multiview_critical"


@dataclass(frozen=True)
class MultiViewCriticalConfig:
    node_feature_dim: int = 15
    edge_feature_dim: int = 6
    spectral_feature_dim: int = MULTIVIEW_SPECTRAL_FEATURE_DIM
    stable_static_dim: int = MULTIVIEW_STABLE_STATIC_DIM
    q_dim: int = MULTIVIEW_Q_DIM
    delta_q_dim: int = MULTIVIEW_DELTA_Q_DIM
    hidden_dim: int = 64
    static_layers: int = 3
    object_layers: int = 2
    full_layers: int = 2
    dropout: float = 0.10
    gcn_alpha: float = 0.10
    gcn_theta: float = 0.50
    static_mode: str = "residual"
    enable_static_attention: bool = True
    enable_v: bool = True
    enable_legacy_v: bool = False
    enable_g: bool = True
    correspondence_mode: str = "uot"
    initial_static_gate: float = 0.01
    initial_temporal_gate: float = 0.01
    initial_v_gate: float = 0.01
    initial_g_gate: float = 0.0
    classifier_hidden_dim: int = 32

    def __post_init__(self):
        if self.node_feature_dim != 15 or self.edge_feature_dim != 6 or self.spectral_feature_dim != 9:
            raise ValueError("multi-view node/edge/spectral schemas are frozen to 15/6/9")
        if self.stable_static_dim != 28 or self.q_dim != 16 or self.delta_q_dim != 18:
            raise ValueError("multi-view theory target schemas are frozen")
        if self.hidden_dim < 1 or min(self.static_layers, self.object_layers, self.full_layers) < 1:
            raise ValueError("multi-view encoder dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("multi-view dropout must lie in [0,1)")
        if self.static_mode not in ("stable", "neural", "residual"):
            raise ValueError("unsupported multi-view static mode")
        if self.correspondence_mode not in ("uot", "shuffled"):
            raise ValueError("unsupported object correspondence mode")
        if self.enable_v and self.enable_legacy_v:
            raise ValueError("new and legacy V branches are mutually exclusive")

    def to_dict(self):
        return asdict(self)


def _masked_mean_std(values):
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("masked list pooling requires [K,D] with K>0")
    mean = values.mean(dim=0)
    std = torch.sqrt((values - mean).square().mean(dim=0) + 1.0e-8)
    return torch.cat((mean, std), dim=-1)


class GatedAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.content = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, states):
        if states.ndim != 2 or states.shape[0] < 1:
            raise ValueError("attention pooling requires non-empty node states")
        scores = self.score(torch.tanh(self.content(states)) * torch.sigmoid(self.gate(states))).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return (weights[:, None] * states).sum(dim=0), weights


class SignedSpectralGCNIILayer(nn.Module):
    """Polarity-state GCNII update with positive-preserve/negative-swap."""

    def __init__(self, hidden_dim, edge_feature_dim, layer_index, alpha, theta, dropout):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(math.log(1.0 + float(theta) / float(layer_index + 1)))
        self.positive_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.negative_weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Linear(edge_feature_dim, 2)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)
        self.positive_norm = nn.LayerNorm(hidden_dim)
        self.negative_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _channels(adjacency, edge_modulation):
        positive = adjacency.clamp_min(0.0)
        negative = -adjacency.clamp_max(0.0)
        positive = positive * edge_modulation[..., 0]
        negative = negative * edge_modulation[..., 1]
        degree = (positive + negative).sum(dim=-1)
        inverse = torch.zeros_like(degree)
        valid = degree > 0.0
        inverse[valid] = degree[valid].rsqrt()
        return (
            inverse[:, None] * positive * inverse[None, :],
            inverse[:, None] * negative * inverse[None, :],
        )

    def forward(self, positive_state, negative_state, positive_initial, negative_initial, adjacency, edge_features):
        edge_modulation = 1.0 + 0.1 * torch.tanh(self.edge_gate(edge_features))
        positive_graph, negative_graph = self._channels(adjacency, edge_modulation)
        positive_preserve = positive_graph.matmul(positive_state)
        positive_swap = negative_graph.matmul(negative_state)
        negative_preserve = positive_graph.matmul(negative_state)
        negative_swap = negative_graph.matmul(positive_state)
        positive_message = positive_preserve + positive_swap
        negative_message = negative_preserve + negative_swap
        with torch.no_grad():
            numerator = positive_message.norm() + negative_message.norm()
            denominator = (
                positive_preserve.norm() + positive_swap.norm()
                + negative_preserve.norm() + negative_swap.norm()
            ).clamp_min(1.0e-8)
            self.last_message_diagnostics = {
                "positive_edge_message_norm": float(
                    (positive_preserve.norm() + negative_preserve.norm()).detach().cpu()
                ),
                "negative_edge_message_norm": float(
                    (positive_swap.norm() + negative_swap.norm()).detach().cpu()
                ),
                "signed_cancellation_ratio": float((numerator / denominator).detach().cpu()),
            }
        positive_base = (1.0 - self.alpha) * positive_message + self.alpha * positive_initial
        negative_base = (1.0 - self.alpha) * negative_message + self.alpha * negative_initial
        positive_update = (1.0 - self.beta) * positive_base + self.beta * self.positive_weight(positive_base)
        negative_update = (1.0 - self.beta) * negative_base + self.beta * self.negative_weight(negative_base)
        return (
            self.positive_norm(positive_state + self.dropout(torch.nn.functional.gelu(positive_update))),
            self.negative_norm(negative_state + self.dropout(torch.nn.functional.gelu(negative_update))),
        )


@dataclass(frozen=True)
class SignedSpectralEncoderOutput:
    graph_embedding: torch.Tensor
    node_states: torch.Tensor
    attention: torch.Tensor
    layer_states: Tuple[torch.Tensor, ...]


class SignedSpectralGCNIIEncoder(nn.Module):
    """Signed Spectral GCNII shared as code, never as branch parameters."""

    def __init__(self, input_dim, edge_feature_dim, hidden_dim, layers, dropout, alpha, theta, use_attention=True):
        super().__init__()
        self.positive_initial = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.negative_initial = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            [SignedSpectralGCNIILayer(hidden_dim, edge_feature_dim, index, alpha, theta, dropout) for index in range(layers)]
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.attention = GatedAttentionPool(hidden_dim)
        self.use_attention = bool(use_attention)
        pool_dim = 3 * hidden_dim if self.use_attention else 2 * hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, node_features, spectral_features, adjacency, edge_features):
        if node_features.ndim != 2 or spectral_features.ndim != 2:
            raise ValueError("Signed Spectral GCNII features must be matrices")
        if node_features.shape[0] != spectral_features.shape[0] or tuple(adjacency.shape) != (
            node_features.shape[0], node_features.shape[0]
        ):
            raise ValueError("Signed Spectral GCNII inputs are misaligned")
        if tuple(edge_features.shape[:2]) != tuple(adjacency.shape):
            raise ValueError("Signed Spectral GCNII edge features are misaligned")
        features = torch.cat((node_features, spectral_features), dim=-1)
        positive_initial = self.positive_initial(features)
        negative_initial = self.negative_initial(features)
        positive_state, negative_state = positive_initial, negative_initial
        history = []
        for layer in self.layers:
            positive_state, negative_state = layer(
                positive_state, negative_state, positive_initial, negative_initial, adjacency, edge_features
            )
            history.append(torch.cat((positive_state, negative_state), dim=-1))
        states = self.fusion(torch.cat((positive_state, negative_state), dim=-1))
        mean = states.mean(dim=0)
        std = torch.sqrt((states - mean).square().mean(dim=0) + 1.0e-8)
        attended, weights = self.attention(states)
        pooled = torch.cat((mean, std, attended), dim=-1) if self.use_attention else torch.cat((mean, std), dim=-1)
        return SignedSpectralEncoderOutput(
            graph_embedding=self.readout(pooled),
            node_states=states,
            attention=weights,
            layer_states=tuple(history),
        )


class ResidualTemporalGRU(nn.Module):
    def __init__(self, hidden_dim, dropout, initial_gate):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, values):
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("residual GRU requires a non-empty transition sequence")
        hidden = values.new_zeros((values.shape[1],))
        outputs = []
        for value in values:
            hidden = self.gru(self.input_dropout(value), hidden)
            outputs.append(value + torch.tanh(self.gate) * self.projection(hidden))
        return torch.stack(outputs, dim=0)


@dataclass(frozen=True)
class MultiViewCriticalSampleOutput:
    representation: torch.Tensor
    static_representation: torch.Tensor
    evolution_representation: torch.Tensor
    full_representation: torch.Tensor
    q_predictions: Optional[torch.Tensor]
    q_targets: Optional[torch.Tensor]
    delta_q_predictions: Optional[torch.Tensor]
    delta_q_targets: Optional[torch.Tensor]
    static_attention: Tuple[torch.Tensor, ...]
    object_attention: Tuple[Tuple[torch.Tensor, ...], ...]


@dataclass(frozen=True)
class MultiViewCriticalOutput:
    logits: torch.Tensor
    representations: torch.Tensor
    samples: Tuple[MultiViewCriticalSampleOutput, ...]
    diagnostics: Dict[str, Any]


class MultiViewCriticalClassifier(nn.Module):
    model_name = MULTIVIEW_CRITICAL_MODEL_NAME

    def __init__(self, config=None):
        super().__init__()
        self.config = config or MultiViewCriticalConfig()
        input_dim = self.config.node_feature_dim + self.config.spectral_feature_dim
        encoder_args = dict(
            input_dim=input_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            alpha=self.config.gcn_alpha,
            theta=self.config.gcn_theta,
        )
        self.static_encoder = SignedSpectralGCNIIEncoder(
            layers=self.config.static_layers,
            use_attention=self.config.enable_static_attention,
            **encoder_args
        )
        self.object_encoder = SignedSpectralGCNIIEncoder(
            layers=self.config.object_layers, use_attention=True, **encoder_args
        )
        self.full_encoder = SignedSpectralGCNIIEncoder(
            layers=self.config.full_layers, use_attention=False, **encoder_args
        )
        self.stable_projection = nn.Sequential(
            nn.Linear(self.config.stable_static_dim, self.config.hidden_dim),
            nn.GELU(), nn.LayerNorm(self.config.hidden_dim)
        )
        self.static_temporal = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.LayerNorm(self.config.hidden_dim)
        )
        self.static_residual = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.static_gate = nn.Parameter(torch.tensor(float(self.config.initial_static_gate)))
        self.q_decoder = nn.Linear(self.config.hidden_dim, self.config.q_dim)

        token_dim = 4 * self.config.hidden_dim + 4
        self.object_transition = nn.Sequential(
            nn.Linear(token_dim, self.config.hidden_dim), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim), nn.LayerNorm(self.config.hidden_dim)
        )
        self.object_context_projection = nn.Sequential(
            nn.Linear(6, self.config.hidden_dim), nn.GELU(), nn.LayerNorm(self.config.hidden_dim)
        )
        self.object_pool = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.LayerNorm(self.config.hidden_dim)
        )
        self.temporal = ResidualTemporalGRU(
            self.config.hidden_dim, self.config.dropout, self.config.initial_temporal_gate
        )
        self.evolution_pool = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.LayerNorm(self.config.hidden_dim)
        )
        self.delta_q_decoder = nn.Linear(self.config.hidden_dim, self.config.delta_q_dim)

        self.full_pool = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim), nn.GELU(), nn.LayerNorm(self.config.hidden_dim)
        )
        self.v_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.legacy_v_projection = nn.Sequential(
            nn.Linear(16, self.config.hidden_dim), nn.GELU(),
            nn.Dropout(self.config.dropout), nn.LayerNorm(self.config.hidden_dim)
        )
        self.g_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.v_gate = nn.Parameter(torch.tensor(float(self.config.initial_v_gate)))
        self.legacy_v_gate = nn.Parameter(torch.tensor(float(self.config.initial_v_gate)))
        self.g_gate = nn.Parameter(torch.tensor(float(self.config.initial_g_gate)))
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.classifier_hidden_dim),
            nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(self.config.classifier_hidden_dim, 2)
        )

    def config_dict(self):
        return self.config.to_dict()

    def _static(self, sample):
        base = self.stable_projection(sample.stable_static)
        embeddings, predictions, targets, attention = [], [], [], []
        if self.config.static_mode != "stable":
            for window in sample.hard_windows:
                if window is None:
                    continue
                output = self.static_encoder(
                    window.node_features, window.spectral_features,
                    window.adjacency, window.edge_features,
                )
                embeddings.append(output.graph_embedding)
                attention.append(output.attention)
                predictions.append(self.q_decoder(output.graph_embedding))
                targets.append(window.q_target)
        if embeddings:
            stack = torch.stack(embeddings, dim=0)
            neural = self.static_temporal(_masked_mean_std(stack))
        else:
            stack = base.new_zeros((0, self.config.hidden_dim))
            neural = base.new_zeros((self.config.hidden_dim,))
        if self.config.static_mode == "stable":
            representation = base
        elif self.config.static_mode == "neural":
            representation = neural
        else:
            representation = base + torch.tanh(self.static_gate) * self.static_residual(neural)
        return (
            representation,
            torch.stack(predictions) if predictions else None,
            torch.stack(targets) if targets else None,
            tuple(attention),
        )

    def _objects(self, sample):
        embeddings = {}
        attentions = {}
        for index, window in enumerate(sample.hard_windows):
            if window is None:
                continue
            encoded, weights = [], []
            for item in window.objects:
                output = self.object_encoder(
                    item.node_features, item.spectral_features,
                    item.adjacency, item.edge_features,
                )
                count = int(item.adjacency.shape[0])
                possible = float(max(1, count * max(0, count - 1)))
                positive = item.adjacency.clamp_min(0.0)
                negative = -item.adjacency.clamp_max(0.0)
                context = item.adjacency.new_tensor((
                    float(item.mass),
                    float((positive > 0.0).sum()) / possible,
                    float((negative > 0.0).sum()) / possible,
                    float(positive.sum()) / possible,
                    float(negative.sum()) / possible,
                    float((positive + negative).sum()) / possible,
                ))
                encoded.append(output.graph_embedding + self.object_context_projection(context))
                weights.append(output.attention)
            embeddings[index] = torch.stack(encoded, dim=0)
            attentions[index] = tuple(weights)
        return embeddings, attentions

    def _evolution(self, sample, reference):
        if not self.config.enable_v:
            zero = reference.new_zeros((self.config.hidden_dim,))
            return zero, None, None, ()
        object_embeddings, attentions = self._objects(sample)
        transition_inputs, targets = [], []
        for transition in sample.transitions:
            if transition is None:
                continue
            left = object_embeddings[transition.source_index]
            right = object_embeddings[transition.target_index]
            if self.config.correspondence_mode == "shuffled" and right.shape[0] > 1:
                right = torch.roll(right, shifts=1, dims=0)
            plan = transition.transport_plan
            if tuple(plan.shape) != (left.shape[0], right.shape[0]):
                raise ValueError("object transport plan does not align with embeddings")
            mass = plan.sum(dim=1)
            aligned = plan.matmul(right) / mass[:, None].clamp_min(1.0e-8)
            difference = (aligned - left) / float(transition.delta_time)
            cost = (plan * transition.object_cost).sum(dim=1) / mass.clamp_min(1.0e-8)
            coupling = sample.hard_windows[transition.source_index].object_coupling.sum(dim=1)
            token = torch.cat(
                (left, aligned, difference, difference.abs(), mass[:, None], cost[:, None], coupling), dim=1
            )
            object_states = self.object_transition(token)
            transition_inputs.append(self.object_pool(_masked_mean_std(object_states)))
            targets.append(transition.delta_q_target)
        if not transition_inputs:
            zero = reference.new_zeros((self.config.hidden_dim,))
            empty = reference.new_zeros((0, self.config.delta_q_dim))
            return zero, empty, empty, tuple(attentions.get(index, ()) for index in sorted(attentions))
        temporal = self.temporal(torch.stack(transition_inputs, dim=0))
        representation = self.evolution_pool(_masked_mean_std(temporal))
        return (
            representation,
            self.delta_q_decoder(temporal),
            torch.stack(targets, dim=0),
            tuple(attentions.get(index, ()) for index in sorted(attentions)),
        )

    def _full(self, sample, reference):
        if not self.config.enable_g:
            return reference.new_zeros((self.config.hidden_dim,))
        if not sample.full_windows:
            raise ValueError("G branch is enabled but full graph windows are absent")
        embeddings = []
        for window in sample.full_windows:
            if window is None:
                continue
            embeddings.append(
                self.full_encoder(
                    window.node_features, window.spectral_features,
                    window.adjacency, window.edge_features,
                ).graph_embedding
            )
        if not embeddings:
            raise ValueError("G branch has no valid full graph window")
        return self.full_pool(_masked_mean_std(torch.stack(embeddings, dim=0)))

    def _sample(self, sample):
        static, q_prediction, q_target, static_attention = self._static(sample)
        evolution, delta_prediction, delta_target, object_attention = self._evolution(sample, static)
        full = self._full(sample, static)
        representation = static
        if self.config.enable_v:
            representation = representation + torch.tanh(self.v_gate) * self.v_projection(evolution)
        if self.config.enable_legacy_v:
            if sample.legacy_variation is None or tuple(sample.legacy_variation.shape) != (16,):
                raise ValueError("legacy V branch requires a standardized 16-D variation")
            representation = representation + torch.tanh(self.legacy_v_gate) * self.legacy_v_projection(
                sample.legacy_variation
            )
        if self.config.enable_g:
            representation = representation + torch.tanh(self.g_gate) * self.g_projection(full)
        return MultiViewCriticalSampleOutput(
            representation, static, evolution, full,
            q_prediction, q_target, delta_prediction, delta_target,
            static_attention, object_attention,
        )

    def forward(self, batch):
        if not isinstance(batch, MultiViewCriticalBatch) or len(batch) < 1:
            raise ValueError("multi-view critical model requires a non-empty typed batch")
        samples = tuple(self._sample(sample) for sample in batch)
        representations = torch.stack([item.representation for item in samples], dim=0)
        return MultiViewCriticalOutput(
            logits=self.classifier(representations),
            representations=representations,
            samples=samples,
            diagnostics={
                "uses_signed_spectral_gcnii": True,
                "static_and_object_encoders_share_parameters": False,
                "uses_q_decoder": self.config.static_mode != "stable",
                "uses_uot_correspondence": self.config.enable_v,
                "uses_legacy_variation": self.config.enable_legacy_v,
                "correspondence_mode": self.config.correspondence_mode,
                "uses_unidirectional_residual_gru": self.config.enable_v,
                "uses_g_decoder": False,
                "static_gate": float(torch.tanh(self.static_gate).detach().cpu()),
                "temporal_gate": float(torch.tanh(self.temporal.gate).detach().cpu()),
                "v_gate": float(torch.tanh(self.v_gate).detach().cpu()),
                "legacy_v_gate": float(torch.tanh(self.legacy_v_gate).detach().cpu()),
                "g_gate": float(torch.tanh(self.g_gate).detach().cpu()),
            },
        )


@dataclass(frozen=True)
class MultiViewShortTermFusionOutput:
    logits: torch.Tensor
    critical: MultiViewCriticalOutput
    short_term: Any
    gate: torch.Tensor
    residual: torch.Tensor


class MultiViewCriticalShortTermFusion(nn.Module):
    """Author short-term logit anchor plus representation-level critical residual."""

    model_name = "theory_guided_multiview_with_author_short_term"

    def __init__(self, critical_model, author_short_term_model, initial_gate=0.01):
        super().__init__()
        self.critical_model = critical_model
        self.author_short_term_model = author_short_term_model
        short_dim = int(author_short_term_model.classifier.in_features)
        hidden = int(critical_model.config.hidden_dim)
        self.short_term_adapter = nn.Sequential(
            nn.Linear(short_dim, short_dim), nn.GELU(), nn.Linear(short_dim, short_dim)
        )
        self.critical_residual = nn.Sequential(
            nn.Linear(hidden, short_dim), nn.GELU(), nn.Linear(short_dim, short_dim)
        )
        self.short_term_gate = nn.Parameter(torch.tensor(float(initial_gate)))
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))
        for adapter in (self.short_term_adapter, self.critical_residual):
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.fusion_classifier = nn.Linear(short_dim, 1)
        with torch.no_grad():
            self.fusion_classifier.weight.copy_(author_short_term_model.classifier.weight)
            self.fusion_classifier.bias.copy_(author_short_term_model.classifier.bias)

    def freeze_base_encoders(self):
        for module in (self.critical_model, self.author_short_term_model):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, critical_batch, author_batch):
        if critical_batch.sample_keys != author_batch.sample_keys:
            raise ValueError("critical and author short-term sample order differs")
        critical = self.critical_model(critical_batch)
        short_term = self.author_short_term_model(author_batch)
        anchor = self.author_short_term_model.classifier_norm(short_term.final_representation)
        projected_anchor = anchor + torch.tanh(self.short_term_gate) * self.short_term_adapter(anchor)
        residual = self.critical_residual(critical.representations)
        fused = projected_anchor + torch.tanh(self.gate) * residual
        binary = self.fusion_classifier(fused).squeeze(-1)
        logits = torch.stack((-0.5 * binary, 0.5 * binary), dim=1)
        return MultiViewShortTermFusionOutput(logits, critical, short_term, torch.tanh(self.gate), residual)


def multiview_critical_loss(output, labels, lambda_q=0.1, lambda_delta_q=0.1, class_weights=None):
    labels = labels.to(device=output.logits.device, dtype=torch.long)
    per_sample = torch.nn.functional.cross_entropy(
        output.logits, labels, reduction="none"
    )
    if class_weights is not None:
        weights = class_weights.to(output.logits).index_select(0, labels)
        classification = (per_sample * weights).mean()
    else:
        classification = per_sample.mean()
    q_losses, delta_losses = [], []
    for sample in output.samples:
        if sample.q_predictions is not None and sample.q_predictions.numel() > 0:
            q_losses.append(torch.nn.functional.smooth_l1_loss(sample.q_predictions, sample.q_targets))
        if sample.delta_q_predictions is not None and sample.delta_q_predictions.numel() > 0:
            delta_losses.append(
                torch.nn.functional.smooth_l1_loss(sample.delta_q_predictions, sample.delta_q_targets)
            )
    q_loss = torch.stack(q_losses).mean() if q_losses else classification.new_zeros(())
    delta_loss = torch.stack(delta_losses).mean() if delta_losses else classification.new_zeros(())
    total = classification + float(lambda_q) * q_loss + float(lambda_delta_q) * delta_loss
    return {"loss": total, "classification_loss": classification, "q_loss": q_loss, "delta_q_loss": delta_loss}
