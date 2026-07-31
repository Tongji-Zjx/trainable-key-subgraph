"""Small signed-GIN encoders for frozen SV-HardSGW key graphs."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from keysubgraph.features.sv_hard_graph_features import (
    SV_NODE_FEATURE_DIM,
    SV_STATIC_FEATURE_DIM,
    SV_VARIATION_DIM,
)


SV_SIGNED_GIN_VARIANTS = (
    "sv_static_variation",
    "static_spectral_only",
    "static_spectral_variation_late_fusion",
    "signed_gin_variation",
    "signed_gin_static_variation",
    "signed_gin_multibranch_late_fusion",
    "signed_gin_static_anchor_residual",
    "signed_gin_static_anchor_residual_attention",
)
SV_DEFAULT_VARIANT = "signed_gin_multibranch_late_fusion"
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
    variant: str = SV_DEFAULT_VARIANT
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
    message_mode: Optional[str] = None
    pooling: Optional[str] = None
    gin_residual: Optional[bool] = None
    gin_jumping_knowledge: Optional[bool] = None
    gin_compact_readout: Optional[bool] = None
    gin_batch_normalization: Optional[bool] = None
    gin_residual_attention: bool = False
    residual_gate_initial_logit: float = -6.0

    def __post_init__(self) -> None:
        if self.variant not in SV_SIGNED_GIN_VARIANTS:
            raise ValueError("unsupported SV Signed-GIN variant")
        is_default_svg = self.variant == SV_DEFAULT_VARIANT
        defaults = {
            "message_mode": (
                "signed_normalized"
                if is_default_svg
                else "signed_weighted"
            ),
            "pooling": "mean_std" if is_default_svg else "attention",
            "gin_residual": bool(is_default_svg),
            "gin_jumping_knowledge": bool(is_default_svg),
            "gin_compact_readout": bool(is_default_svg),
            "gin_batch_normalization": bool(is_default_svg),
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
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
        if not -20.0 <= self.residual_gate_initial_logit <= 0.0:
            raise ValueError(
                "SV residual gate initial logit must lie in [-20,0]"
            )
        if self.message_mode not in SV_SIGNED_GIN_MESSAGE_MODES:
            raise ValueError("unsupported SV Signed-GIN message mode")
        if self.pooling not in SV_SIGNED_GIN_POOLING_MODES:
            raise ValueError("unsupported SV Signed-GIN pooling")
        if self.gin_residual_attention and (
            self.variant
            != "signed_gin_static_anchor_residual_attention"
            or self.pooling != "mean_std"
            or not self.gin_compact_readout
        ):
            raise ValueError(
                "residual attention requires compact mean-std "
                "static-anchor residual fusion"
            )
        if (
            self.variant
            == "signed_gin_static_anchor_residual_attention"
            and not self.gin_residual_attention
        ):
            raise ValueError(
                "residual-attention variant must enable its attention"
            )

    @property
    def uses_gin(self) -> bool:
        return self.variant not in (
            "sv_static_variation",
            "static_spectral_only",
            "static_spectral_variation_late_fusion",
        )

    @property
    def uses_static(self) -> bool:
        return self.variant != "signed_gin_variation"

    @property
    def uses_variation(self) -> bool:
        return self.variant != "static_spectral_only"

    @property
    def uses_late_fusion(self) -> bool:
        return self.variant in (
            "static_spectral_only",
            "static_spectral_variation_late_fusion",
            "signed_gin_multibranch_late_fusion",
            "signed_gin_static_anchor_residual",
            "signed_gin_static_anchor_residual_attention",
        )

    @property
    def uses_residual_fusion(self) -> bool:
        return self.variant in (
            "signed_gin_static_anchor_residual",
            "signed_gin_static_anchor_residual_attention",
        )

    @property
    def gin_output_dim(self) -> int:
        if self.gin_compact_readout:
            return 4 * self.channel_projection_dim
        return self.gin_hidden_dim * (
            2 if self.pooling == "mean_std" else 1
        )

    @property
    def gin_window_output_dim(self) -> int:
        if self.gin_compact_readout:
            return 2 * self.channel_projection_dim
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
        channel_count = int(self.uses_variation)
        channel_count += int(self.uses_gin)
        channel_count += int(self.uses_static)
        return channel_count * self.channel_projection_dim

    @property
    def active_branch_names(self) -> Tuple[str, ...]:
        names = []
        if self.uses_gin:
            names.append("gin")
        if self.uses_static:
            names.append("static_spectral")
        if self.uses_variation:
            names.append("variation")
        return tuple(names)


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
    variation_projection: Optional[torch.Tensor]
    gin_projection: Optional[torch.Tensor]
    encoder_outputs: Tuple[SVSignedGINEncoderOutput, ...]
    diagnostics: Dict[str, Any]
    branch_logits: Optional[Dict[str, torch.Tensor]] = None
    fusion_weights: Optional[torch.Tensor] = None
    residual_gates: Optional[Dict[str, torch.Tensor]] = None
    gin_normalized_representation: Optional[torch.Tensor] = None


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
        self.window_readout_projection = (
            nn.Sequential(
                nn.Linear(
                    self.config.gin_hidden_dim
                    * (2 if self.config.pooling == "mean_std" else 1),
                    self.config.gin_window_output_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(self.config.gin_window_output_dim),
            )
            if self.config.gin_compact_readout
            else None
        )
        self.attention_residual_projection = None
        self.attention_residual_gate_logit = None
        if self.config.gin_residual_attention:
            self.attention_residual_projection = nn.Sequential(
                nn.Linear(
                    self.config.gin_hidden_dim,
                    self.config.gin_window_output_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    self.config.gin_window_output_dim,
                    self.config.gin_window_output_dim,
                ),
            )
            nn.init.zeros_(
                self.attention_residual_projection[-1].weight
            )
            nn.init.zeros_(
                self.attention_residual_projection[-1].bias
            )
            self.attention_residual_gate_logit = nn.Parameter(
                torch.tensor(
                    self.config.residual_gate_initial_logit,
                    dtype=torch.float32,
                )
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
        attention_embedding = None
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
            if self.config.gin_residual_attention:
                scores = self.attention(states).squeeze(-1)
                weights = torch.softmax(scores, dim=0)
                attention_embedding = (
                    states * weights[:, None]
                ).sum(dim=0)
        else:
            maximum = states.max(dim=0)
            embedding = maximum.values
            # This diagnostic mask may sum above one when different channels
            # choose different nodes; it is not an attention probability.
            weights = (
                states == maximum.values[None, :]
            ).any(dim=-1).to(states.dtype)
        if self.window_readout_projection is not None:
            embedding = self.window_readout_projection(embedding)
        if self.config.gin_residual_attention:
            embedding = (
                embedding
                + torch.sigmoid(
                    self.attention_residual_gate_logit
                )
                * self.attention_residual_projection(
                    attention_embedding
                )
            )
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
        if self.config.gin_compact_readout:
            mean = stacked.mean(dim=0)
            variance = (stacked - mean).square().mean(dim=0)
            representation = torch.cat(
                (mean, torch.sqrt(variance + 1.0e-8)), dim=-1
            )
        else:
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


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use frozen running statistics for singleton training batches."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training and value.shape[0] == 1:
            return F.batch_norm(
                value,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(value)


def _batch_normalized_projection(
    input_dim: int, output_dim: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim, bias=False),
        SafeBatchNorm1d(output_dim),
        nn.GELU(),
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
        self.gin_feature_normalization = (
            SafeBatchNorm1d(self.config.gin_output_dim)
            if self.config.uses_gin
            and self.config.gin_batch_normalization
            else None
        )
        self.gin_projection = (
            (
                _batch_normalized_projection(
                    self.config.gin_output_dim,
                    self.config.channel_projection_dim,
                )
                if self.config.gin_batch_normalization
                else _projection(
                    self.config.gin_output_dim,
                    self.config.channel_projection_dim,
                )
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
        self.variation_projection = (
            _projection(
                self.config.variation_dim,
                self.config.channel_projection_dim,
            )
            if self.config.uses_variation
            else None
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
                    for name in self.config.active_branch_names
                }
            )
            self.fusion_log_weights = nn.Parameter(
                torch.zeros(
                    len(self.config.active_branch_names),
                    dtype=torch.float32,
                ),
                requires_grad=not self.config.uses_residual_fusion,
            )
            if self.config.uses_residual_fusion:
                self.residual_gate_logits = nn.ParameterDict(
                    {
                        name: nn.Parameter(
                            torch.tensor(
                                self.config.residual_gate_initial_logit,
                                dtype=torch.float32,
                            )
                        )
                        for name in ("gin", "variation")
                    }
                )
                for name in ("gin", "variation"):
                    output_layer = self.branch_classifiers[name][-1]
                    nn.init.zeros_(output_layer.weight)
                    nn.init.zeros_(output_layer.bias)
            else:
                self.residual_gate_logits = None
            self.classifier = None
        else:
            self.branch_classifiers = None
            self.register_parameter("fusion_log_weights", None)
            self.residual_gate_logits = None
            self.classifier = nn.Sequential(
                nn.Linear(
                    self.config.fusion_input_dim,
                    self.config.fusion_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.fusion_hidden_dim, 2),
            )
        self._training_stage = (
            "residual_experts"
            if self.config.uses_residual_fusion
            else "joint"
        )

    @property
    def training_stage(self) -> str:
        return self._training_stage

    def set_training_stage(self, stage: str) -> None:
        """Freeze the V1A anchor or experts for its two training stages."""

        if not self.config.uses_residual_fusion:
            if stage != "joint":
                raise ValueError(
                    "training stages only apply to residual fusion"
                )
            self._training_stage = "joint"
            return
        if stage not in ("static_anchor", "residual_experts"):
            raise ValueError("unsupported residual-fusion training stage")
        self._training_stage = str(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "static_anchor":
            for module in (
                self.static_projection,
                self.branch_classifiers["static_spectral"],
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            for module in (
                self.encoder,
                self.gin_feature_normalization,
                self.gin_projection,
                self.variation_projection,
                self.branch_classifiers["gin"],
                self.branch_classifiers["variation"],
            ):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
            for parameter in self.residual_gate_logits.parameters():
                parameter.requires_grad_(True)

    def reset_residual_fusion_parameters(self, seed: int) -> None:
        """Deterministically initialize paired V1 candidates by component."""

        if not self.config.uses_residual_fusion:
            raise ValueError(
                "residual parameter reset requires residual fusion"
            )

        def reset_module(module):
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        components = (
            (self.encoder, 101),
            (self.gin_feature_normalization, 102),
            (self.gin_projection, 103),
            (self.static_projection, 104),
            (self.variation_projection, 105),
            (self.branch_classifiers["gin"], 106),
            (self.branch_classifiers["static_spectral"], 107),
            (self.branch_classifiers["variation"], 108),
        )
        with torch.random.fork_rng(devices=[]):
            for module, offset in components:
                if module is None:
                    continue
                torch.manual_seed(int(seed) + offset)
                module.apply(reset_module)
        with torch.no_grad():
            self.fusion_log_weights.zero_()
            for value in self.residual_gate_logits.values():
                value.fill_(self.config.residual_gate_initial_logit)
            for name in ("gin", "variation"):
                output_layer = self.branch_classifiers[name][-1]
                output_layer.weight.zero_()
                output_layer.bias.zero_()
            if self.config.gin_residual_attention:
                self.encoder.attention_residual_gate_logit.fill_(
                    self.config.residual_gate_initial_logit
                )
                output_layer = (
                    self.encoder.attention_residual_projection[-1]
                )
                output_layer.weight.zero_()
                output_layer.bias.zero_()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.config.uses_residual_fusion:
            if self._training_stage == "static_anchor":
                for module in (
                    self.encoder,
                    self.gin_feature_normalization,
                    self.gin_projection,
                    self.variation_projection,
                    self.branch_classifiers["gin"],
                    self.branch_classifiers["variation"],
                ):
                    if module is not None:
                        module.eval()
            elif self._training_stage == "residual_experts":
                self.static_projection.eval()
                self.branch_classifiers["static_spectral"].eval()
        return self

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
        gin_normalized_representation = None
        gin_projected = None
        skip_frozen_gin = (
            self.config.uses_residual_fusion
            and self._training_stage == "static_anchor"
        )
        if self.config.uses_gin and not skip_frozen_gin:
            outputs = tuple(self.encoder(sample) for sample in batch)
            gin_representation = torch.stack(
                [output.representation for output in outputs], dim=0
            )
            gin_normalized_representation = gin_representation
            if self.gin_feature_normalization is not None:
                gin_normalized_representation = (
                    self.gin_feature_normalization(
                        gin_representation
                    )
                )
            gin_projected = self.gin_projection(
                gin_normalized_representation
            )
            channels.append(gin_projected)
            encoder_outputs = outputs
        elif self.config.uses_gin:
            gin_projected = static.new_zeros(
                (len(batch), self.config.channel_projection_dim)
            )
            channels.append(gin_projected)

        static_projected = None
        if self.config.uses_static:
            static_input = (
                static[:, :16]
                if self.config.uses_late_fusion
                else static
            )
            static_projected = self.static_projection(static_input)
            channels.append(static_projected)
        variation_projected = None
        if self.config.uses_variation:
            variation_projected = self.variation_projection(variation)
            channels.append(variation_projected)
        final = torch.cat(channels, dim=-1)
        branch_logits = None
        fusion_weights = None
        residual_gates = None
        if self.config.uses_late_fusion:
            projected = {
                "gin": gin_projected,
                "static_spectral": static_projected,
                "variation": variation_projected,
            }
            branch_logits = {
                name: self.branch_classifiers[name](projected[name])
                for name in self.config.active_branch_names
            }
            if self.config.uses_residual_fusion:
                residual_gates = {
                    name: torch.sigmoid(value)
                    for name, value in self.residual_gate_logits.items()
                }
                if self.config.gin_residual_attention:
                    residual_gates["attention"] = torch.sigmoid(
                        self.encoder.attention_residual_gate_logit
                    )
                logits = branch_logits["static_spectral"]
                if self._training_stage != "static_anchor":
                    logits = (
                        logits
                        + residual_gates["gin"] * branch_logits["gin"]
                        + residual_gates["variation"]
                        * branch_logits["variation"]
                    )
            else:
                fusion_weights = torch.softmax(
                    self.fusion_log_weights, dim=0
                )
                stacked_logits = torch.stack(
                    [
                        branch_logits[name]
                        for name in self.config.active_branch_names
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
                "gin_compact_readout": (
                    self.config.gin_compact_readout
                ),
                "gin_batch_normalization": (
                    self.config.gin_batch_normalization
                ),
                "gin_residual_attention": (
                    self.config.gin_residual_attention
                ),
                "fusion_mode": (
                    "static_anchor_zero_output_residual"
                    if self.config.uses_residual_fusion
                    else (
                        "nonnegative_logit"
                        if self.config.uses_late_fusion
                        else "feature_concatenation"
                    )
                ),
                "training_stage": self._training_stage,
                "active_branches": self.config.active_branch_names,
            },
            branch_logits=branch_logits,
            fusion_weights=fusion_weights,
            residual_gates=residual_gates,
            gin_normalized_representation=(
                gin_normalized_representation
            ),
        )
