"""Small signed-GIN encoders for frozen SV-HardSGW key graphs."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from keysubgraph.features.sv_hard_graph_features import (
    SV_NODE_FEATURE_DIM,
    SV_STATIC_FEATURE_DIM,
    SV_VARIATION_DIM,
)


SV_SIGNED_GIN_VARIANTS = (
    "sv_static_variation",
    "signed_gin_variation",
    "signed_gin_static_variation",
    "signed_gin_multibranch_late_fusion",
)
SV_SIGNED_GIN_MESSAGE_MODES = (
    "signed_weighted",
    "signed_normalized",
    "unsigned_weighted",
    "unsigned_binary",
)
SV_SIGNED_GIN_POOLING_MODES = (
    "attention",
    "mean",
    "max",
    "mean_std",
)


@dataclass(frozen=True)
class SVSignedGINConfig:
    variant: str = "signed_gin_variation"
    node_feature_dim: int = SV_NODE_FEATURE_DIM
    static_feature_dim: int = SV_STATIC_FEATURE_DIM
    variation_dim: int = SV_VARIATION_DIM
    gin_hidden_dim: int = 64
    gin_layers: int = 2
    attention_hidden_dim: int = 32
    channel_projection_dim: int = 16
    fusion_hidden_dim: int = 16
    dropout: float = 0.10
    learnable_epsilon: bool = True
    message_mode: str = "signed_weighted"
    pooling: str = "attention"
    gin_residual: bool = False
    gin_jumping_knowledge: bool = False

    def __post_init__(self) -> None:
        if self.variant not in SV_SIGNED_GIN_VARIANTS:
            raise ValueError("unsupported SV Signed-GIN variant")
        expected = (
            (self.node_feature_dim, SV_NODE_FEATURE_DIM),
            (self.static_feature_dim, SV_STATIC_FEATURE_DIM),
            (self.variation_dim, SV_VARIATION_DIM),
        )
        if any(value != required for value, required in expected):
            raise ValueError("SV feature dimensions are frozen to 15/28/16")
        dimensions = (
            self.gin_hidden_dim,
            self.gin_layers,
            self.attention_hidden_dim,
            self.channel_projection_dim,
            self.fusion_hidden_dim,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("SV Signed-GIN dimensions must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("SV Signed-GIN dropout must lie in [0,1)")
        if self.message_mode not in SV_SIGNED_GIN_MESSAGE_MODES:
            raise ValueError("unsupported SV Signed-GIN message mode")
        if self.pooling not in SV_SIGNED_GIN_POOLING_MODES:
            raise ValueError("unsupported SV Signed-GIN pooling")

    @property
    def uses_gin(self) -> bool:
        return self.variant != "sv_static_variation"

    @property
    def uses_static(self) -> bool:
        return self.variant != "signed_gin_variation"

    @property
    def uses_late_fusion(self) -> bool:
        return self.variant == "signed_gin_multibranch_late_fusion"

    @property
    def gin_output_dim(self) -> int:
        return self.gin_hidden_dim * (
            2 if self.pooling == "mean_std" else 1
        )

    @property
    def static_input_dim(self) -> int:
        return (
            16 if self.uses_late_fusion else self.static_feature_dim
        )

    @property
    def fusion_input_dim(self) -> int:
        channel_count = 1  # Variation is always present.
        channel_count += int(self.uses_gin)
        channel_count += int(self.uses_static)
        return channel_count * self.channel_projection_dim


@dataclass(frozen=True)
class SVSignedGINWindowInput:
    node_features: torch.Tensor
    adjacency: torch.Tensor

    def to(self, device) -> "SVSignedGINWindowInput":
        return SVSignedGINWindowInput(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
        )


@dataclass(frozen=True)
class SVSignedGINSampleInput:
    sample_key: str
    label: int
    windows: Tuple[SVSignedGINWindowInput, ...]
    static_features: torch.Tensor
    variation: torch.Tensor

    def to(self, device) -> "SVSignedGINSampleInput":
        return SVSignedGINSampleInput(
            sample_key=self.sample_key,
            label=int(self.label),
            windows=tuple(window.to(device) for window in self.windows),
            static_features=self.static_features.to(device),
            variation=self.variation.to(device),
        )


@dataclass(frozen=True)
class SVSignedGINBatch:
    samples: Tuple[SVSignedGINSampleInput, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(
            [sample.label for sample in self.samples], dtype=torch.long
        )

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(sample.sample_key for sample in self.samples)

    def to(self, device) -> "SVSignedGINBatch":
        return SVSignedGINBatch(
            tuple(sample.to(device) for sample in self.samples)
        )


@dataclass(frozen=True)
class SVSignedGINEncoderOutput:
    representation: torch.Tensor
    window_embeddings: Tuple[torch.Tensor, ...]
    node_attention: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class SVSignedGINOutput:
    logits: torch.Tensor
    final_representation: torch.Tensor
    gin_representation: Optional[torch.Tensor]
    static_projection: Optional[torch.Tensor]
    variation_projection: torch.Tensor
    gin_projection: Optional[torch.Tensor]
    encoder_outputs: Tuple[SVSignedGINEncoderOutput, ...]
    diagnostics: Dict[str, Any]
    branch_logits: Optional[Dict[str, torch.Tensor]] = None
    fusion_weights: Optional[torch.Tensor] = None


class SignedGINLayer(nn.Module):
    """Magnitude-aware positive-minus-negative GIN update."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        learnable_epsilon: bool = True,
        message_mode: str = "signed_weighted",
    ) -> None:
        super().__init__()
        epsilon = torch.zeros((), dtype=torch.float32)
        if learnable_epsilon:
            self.epsilon = nn.Parameter(epsilon)
        else:
            self.register_buffer("epsilon", epsilon)
        if message_mode not in SV_SIGNED_GIN_MESSAGE_MODES:
            raise ValueError("unsupported Signed-GIN message mode")
        self.message_mode = str(message_mode)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def signed_aggregate(
        self, node_states: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if node_states.ndim != 2:
            raise ValueError("Signed GIN node states must be [N,H]")
        if tuple(adjacency.shape) != (
            node_states.shape[0],
            node_states.shape[0],
        ):
            raise ValueError("Signed GIN adjacency must align with nodes")
        if not bool(torch.isfinite(adjacency).all()):
            raise ValueError("Signed GIN adjacency contains non-finite values")
        if self.message_mode == "signed_weighted":
            positive = adjacency.clamp_min(0.0)
            negative_magnitude = -adjacency.clamp_max(0.0)
            message = positive.matmul(node_states) - (
                negative_magnitude.matmul(node_states)
            )
        elif self.message_mode == "signed_normalized":
            absolute_degree = adjacency.abs().sum(dim=-1).clamp_min(
                1.0e-8
            )
            inverse_sqrt = absolute_degree.rsqrt()
            normalized = (
                inverse_sqrt[:, None]
                * adjacency
                * inverse_sqrt[None, :]
            )
            message = normalized.matmul(node_states)
        elif self.message_mode == "unsigned_weighted":
            message = adjacency.abs().matmul(node_states)
        else:
            message = (adjacency != 0.0).to(
                node_states.dtype
            ).matmul(node_states)
        return (1.0 + self.epsilon.to(node_states)) * node_states + message

    def forward(
        self, node_states: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(self.signed_aggregate(node_states, adjacency))


class SignedGINKeySubgraphEncoder(nn.Module):
    """Encode each hard window, then mean-pool valid windows."""

    def __init__(
        self, config: Optional[SVSignedGINConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or SVSignedGINConfig()
        self.node_projection = nn.Sequential(
            nn.Linear(
                self.config.node_feature_dim,
                self.config.gin_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(self.config.gin_hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                SignedGINLayer(
                    self.config.gin_hidden_dim,
                    self.config.dropout,
                    self.config.learnable_epsilon,
                    self.config.message_mode,
                )
                for _ in range(self.config.gin_layers)
            ]
        )
        self.residual_norms = (
            nn.ModuleList(
                [
                    nn.LayerNorm(self.config.gin_hidden_dim)
                    for _ in range(self.config.gin_layers)
                ]
            )
            if self.config.gin_residual
            else None
        )
        self.jumping_projection = (
            nn.Sequential(
                nn.Linear(
                    self.config.gin_hidden_dim
                    * (self.config.gin_layers + 1),
                    self.config.gin_hidden_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(self.config.gin_hidden_dim),
            )
            if self.config.gin_jumping_knowledge
            else None
        )
        self.attention = nn.Sequential(
            nn.Linear(
                self.config.gin_hidden_dim,
                self.config.attention_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(self.config.attention_hidden_dim, 1),
        )

    def encode_window(
        self, window: SVSignedGINWindowInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_features = window.node_features
        adjacency = window.adjacency
        if (
            node_features.ndim != 2
            or node_features.shape[-1] != self.config.node_feature_dim
            or node_features.shape[0] < 1
        ):
            raise ValueError("Signed GIN window has invalid node features")
        if tuple(adjacency.shape) != (
            node_features.shape[0],
            node_features.shape[0],
        ):
            raise ValueError("Signed GIN window adjacency is misaligned")
        states = self.node_projection(node_features)
        history = [states]
        for layer_index, layer in enumerate(self.layers):
            updated = layer(states, adjacency)
            states = (
                self.residual_norms[layer_index](states + updated)
                if self.config.gin_residual
                else updated
            )
            history.append(states)
        if self.jumping_projection is not None:
            states = self.jumping_projection(
                torch.cat(history, dim=-1)
            )
        if self.config.pooling == "attention":
            scores = self.attention(states).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            embedding = (states * weights[:, None]).sum(dim=0)
        elif self.config.pooling == "mean":
            weights = states.new_full(
                (states.shape[0],), 1.0 / float(states.shape[0])
            )
            embedding = states.mean(dim=0)
        elif self.config.pooling == "mean_std":
            weights = states.new_full(
                (states.shape[0],), 1.0 / float(states.shape[0])
            )
            mean = states.mean(dim=0)
            variance = (states - mean).square().mean(dim=0)
            embedding = torch.cat(
                (mean, torch.sqrt(variance + 1.0e-8)), dim=-1
            )
        else:
            maximum = states.max(dim=0)
            embedding = maximum.values
            # This diagnostic mask may sum above one when different channels
            # choose different nodes; it is not an attention probability.
            weights = (
                states == maximum.values[None, :]
            ).any(dim=-1).to(states.dtype)
        return embedding, weights

    def forward(
        self, sample: SVSignedGINSampleInput
    ) -> SVSignedGINEncoderOutput:
        if not sample.windows:
            raise ValueError("Signed GIN sample has no valid hard windows")
        embeddings = []
        attention = []
        for window in sample.windows:
            embedding, weights = self.encode_window(window)
            embeddings.append(embedding)
            attention.append(weights)
        stacked = torch.stack(embeddings, dim=0)
        representation = stacked.mean(dim=0)
        return SVSignedGINEncoderOutput(
            representation=representation,
            window_embeddings=tuple(embeddings),
            node_attention=tuple(attention),
        )


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
    )


class SVSignedGINClassifier(nn.Module):
    """SG0/SG1/SG2 classifiers with equal-width feature channels."""

    model_name = "sv_hard_sgw_signed_gin"

    def __init__(
        self, config: Optional[SVSignedGINConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or SVSignedGINConfig()
        self.encoder = (
            SignedGINKeySubgraphEncoder(self.config)
            if self.config.uses_gin
            else None
        )
        self.gin_projection = (
            _projection(
                self.config.gin_output_dim,
                self.config.channel_projection_dim,
            )
            if self.config.uses_gin
            else None
        )
        self.static_projection = (
            _projection(
                self.config.static_input_dim,
                self.config.channel_projection_dim,
            )
            if self.config.uses_static
            else None
        )
        self.variation_projection = _projection(
            self.config.variation_dim,
            self.config.channel_projection_dim,
        )
        if self.config.uses_late_fusion:
            self.branch_classifiers = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(
                            self.config.channel_projection_dim,
                            self.config.fusion_hidden_dim,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.config.fusion_hidden_dim, 2),
                    )
                    for name in ("gin", "static_spectral", "variation")
                }
            )
            self.fusion_log_weights = nn.Parameter(
                torch.zeros(3, dtype=torch.float32)
            )
            self.classifier = None
        else:
            self.branch_classifiers = None
            self.register_parameter("fusion_log_weights", None)
            self.classifier = nn.Sequential(
                nn.Linear(
                    self.config.fusion_input_dim,
                    self.config.fusion_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.fusion_hidden_dim, 2),
            )

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(self, batch: SVSignedGINBatch) -> SVSignedGINOutput:
        if len(batch) < 1:
            raise ValueError("SV Signed-GIN batch cannot be empty")
        static = torch.stack(
            [sample.static_features for sample in batch], dim=0
        )
        variation = torch.stack(
            [sample.variation for sample in batch], dim=0
        )
        if tuple(static.shape[1:]) != (
            self.config.static_feature_dim,
        ) or tuple(variation.shape[1:]) != (
            self.config.variation_dim,
        ):
            raise ValueError("SV batch summary dimensions are invalid")

        channels = []
        encoder_outputs = ()
        gin_representation = None
        gin_projected = None
        if self.config.uses_gin:
            outputs = tuple(self.encoder(sample) for sample in batch)
            gin_representation = torch.stack(
                [output.representation for output in outputs], dim=0
            )
            gin_projected = self.gin_projection(gin_representation)
            channels.append(gin_projected)
            encoder_outputs = outputs

        static_projected = None
        if self.config.uses_static:
            static_input = (
                static[:, :16]
                if self.config.uses_late_fusion
                else static
            )
            static_projected = self.static_projection(static_input)
            channels.append(static_projected)
        variation_projected = self.variation_projection(variation)
        channels.append(variation_projected)
        final = torch.cat(channels, dim=-1)
        branch_logits = None
        fusion_weights = None
        if self.config.uses_late_fusion:
            branch_logits = {
                "gin": self.branch_classifiers["gin"](
                    gin_projected
                ),
                "static_spectral": self.branch_classifiers[
                    "static_spectral"
                ](static_projected),
                "variation": self.branch_classifiers["variation"](
                    variation_projected
                ),
            }
            fusion_weights = torch.softmax(
                self.fusion_log_weights, dim=0
            )
            stacked_logits = torch.stack(
                [
                    branch_logits["gin"],
                    branch_logits["static_spectral"],
                    branch_logits["variation"],
                ],
                dim=0,
            )
            logits = (
                stacked_logits * fusion_weights[:, None, None]
            ).sum(dim=0)
        else:
            logits = self.classifier(final)
        if tuple(logits.shape) != (len(batch), 2):
            raise RuntimeError("SV Signed-GIN logits must have shape [B,2]")
        return SVSignedGINOutput(
            logits=logits,
            final_representation=final,
            gin_representation=gin_representation,
            static_projection=static_projected,
            variation_projection=variation_projected,
            gin_projection=gin_projected,
            encoder_outputs=encoder_outputs,
            diagnostics={
                "variant": self.config.variant,
                "uses_coordinates": False,
                "uses_site_input": False,
                "uses_raw_community_embedding": False,
                "preserves_signed_edges": (
                    self.config.message_mode
                    in ("signed_weighted", "signed_normalized")
                ),
                "message_mode": self.config.message_mode,
                "pooling": self.config.pooling,
                "gin_residual": self.config.gin_residual,
                "gin_jumping_knowledge": (
                    self.config.gin_jumping_knowledge
                ),
                "fusion_mode": (
                    "nonnegative_logit"
                    if self.config.uses_late_fusion
                    else "feature_concatenation"
                ),
            },
            branch_logits=branch_logits,
            fusion_weights=fusion_weights,
        )
