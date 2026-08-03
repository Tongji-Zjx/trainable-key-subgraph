"""Corrected theory-guided neural S/V experts for frozen hard graphs.

The static expert consumes both the signed hard graph and its invariant
spectral-quantile state.  The evolution expert consumes adjacent encoded
windows, keeps transition direction, and is supervised by the canonical
signed spectral--GW transition vector.  No coordinate, ROI identity, site
label, or raw community embedding enters either expert.
"""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from keysubgraph.features.sv_hard_graph_features import SV_NODE_FEATURE_DIM
from keysubgraph.features.theory_neural_features import THEORY_EDGE_FEATURE_DIM
from keysubgraph.models.theory_guided_neural import (
    EdgeAwareSignedEncoder,
    TheoryNeuralBatch,
    TheoryNeuralOutput,
    TheoryNeuralSampleOutput,
)
from keysubgraph.theory.sgw_core_features import SGW_CORE_DIM, SGW_QUANTILE_DIM


NEURALIZED_SV_MODEL_NAME = "svg_corrected_neuralized_sv"
NEURALIZED_SV_VARIANTS = (
    "NS_static_spectral",
    "NV_dynamic_evolution",
    "NSV_safe_residual",
)


@dataclass(frozen=True)
class NeuralizedSVConfig:
    variant: str = "NSV_safe_residual"
    node_feature_dim: int = SV_NODE_FEATURE_DIM
    edge_feature_dim: int = THEORY_EDGE_FEATURE_DIM
    quantile_dim: int = SGW_QUANTILE_DIM
    transition_dim: int = SGW_CORE_DIM
    hidden_dim: int = 64
    layers: int = 2
    spectral_token_dim: int = 32
    transition_token_dim: int = 32
    classifier_hidden_dim: int = 32
    dropout: float = 0.10
    initial_evolution_gate: float = 0.10

    def __post_init__(self) -> None:
        if self.variant not in NEURALIZED_SV_VARIANTS:
            raise ValueError("unsupported corrected neural S/V variant")
        if (
            self.node_feature_dim != 15
            or self.edge_feature_dim != 6
            or self.quantile_dim != 16
            or self.transition_dim != 18
        ):
            raise ValueError("corrected neural S/V feature dimensions are frozen")
        if self.hidden_dim != 64 or self.layers != 2:
            raise ValueError("corrected neural S/V first implementation uses H=64,L=2")
        if self.spectral_token_dim < 1 or self.transition_token_dim < 1:
            raise ValueError("corrected neural S/V token dimensions must be positive")
        if self.classifier_hidden_dim < 1 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid corrected neural S/V classifier configuration")
        if not 0.0 < self.initial_evolution_gate < 1.0:
            raise ValueError("initial evolution gate must lie in (0,1)")

    @property
    def uses_static(self) -> bool:
        return self.variant in ("NS_static_spectral", "NSV_safe_residual")

    @property
    def uses_evolution(self) -> bool:
        return self.variant in ("NV_dynamic_evolution", "NSV_safe_residual")

    @property
    def uses_center_loss(self) -> bool:
        return False


def _masked_mean_std(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("neural S/V pooling requires at least one state")
    mean = values.mean(dim=0)
    std = torch.sqrt((values - mean).square().mean(dim=0) + 1.0e-8)
    return torch.cat((mean, std), dim=-1)


class NeuralizedSVClassifier(nn.Module):
    """Independent neural S/V experts with a safe residual joint variant."""

    model_name = NEURALIZED_SV_MODEL_NAME

    def __init__(self, config: Optional[NeuralizedSVConfig] = None) -> None:
        super().__init__()
        self.config = config or NeuralizedSVConfig()
        # Reuse the audited signed edge-aware encoder.  It separates positive
        # and negative messages and consumes A, |A|, delta-A, |delta-A|,
        # delta-validity and same-community indicators.
        self.graph_encoder = EdgeAwareSignedEncoder(self.config)
        self.spectral_token = nn.Sequential(
            nn.Linear(self.config.quantile_dim, self.config.spectral_token_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.spectral_token_dim),
        )
        self.static_window_fusion = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim + self.config.spectral_token_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.static_pool = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.q_decoder = nn.Linear(
            self.config.hidden_dim, self.config.quantile_dim
        )

        pair_dim = 4 * self.config.hidden_dim
        self.graph_transition_encoder = nn.Sequential(
            nn.Linear(pair_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        # Exact theory quantities are label-free inputs.  Keeping their token
        # separate prevents the auxiliary decoder from learning an identity
        # map: Gamma is decoded only from the graph-pair latent below.
        self.transition_token = nn.Sequential(
            nn.Linear(self.config.transition_dim, self.config.transition_token_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.transition_token_dim),
        )
        self.transition_fusion = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim + self.config.transition_token_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.evolution_pool = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.gamma_decoder = nn.Linear(
            self.config.hidden_dim, self.config.transition_dim
        )

        self.evolution_residual = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.evolution_gate_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    self.config.initial_evolution_gate
                    / (1.0 - self.config.initial_evolution_gate)
                ),
                dtype=torch.float32,
            )
        )
        # Static pooling already ends in LayerNorm.  Avoid another transform
        # here so a zero residual is bitwise the static representation.
        self.joint_norm = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.classifier_hidden_dim, 2),
        )
        # At initialization NSV is exactly its static expert.  Evolution is
        # allowed to enter only after learning a non-zero residual direction.
        nn.init.zeros_(self.evolution_residual[-1].weight)
        nn.init.zeros_(self.evolution_residual[-1].bias)

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def _encode_sample(self, sample) -> TheoryNeuralSampleOutput:
        graph_embeddings = {}
        static_embeddings = []
        positions = []
        q_targets = []
        q_predictions = []
        all_norms = []
        film_values = []
        for index, window in enumerate(sample.windows):
            if window is None:
                continue
            graph, norms, film = self.graph_encoder(window, use_film=True)
            spectral = self.spectral_token(window.spectral_quantiles)
            static = self.static_window_fusion(torch.cat((graph, spectral), dim=-1))
            graph_embeddings[index] = graph
            positions.append(index)
            static_embeddings.append(static)
            q_targets.append(window.spectral_quantiles)
            q_predictions.append(self.q_decoder(static))
            all_norms.append(norms)
            if film is not None:
                film_values.append(film)
        if not static_embeddings:
            raise ValueError("corrected neural S/V sample has no valid window")
        static_stack = torch.stack(static_embeddings, dim=0)
        static_representation = self.static_pool(_masked_mean_std(static_stack))

        transition_states = []
        gamma_predictions = []
        gamma_targets = []
        for index in range(max(0, len(sample.windows) - 1)):
            if not bool(sample.transition_mask[index]):
                continue
            if index not in graph_embeddings or index + 1 not in graph_embeddings:
                raise ValueError("transition mask points to a missing graph window")
            left = graph_embeddings[index]
            right = graph_embeddings[index + 1]
            difference = right - left
            graph_transition = self.graph_transition_encoder(
                torch.cat((left, right, difference, difference.abs()), dim=-1)
            )
            target = sample.transition_targets[index]
            theory_token = self.transition_token(target)
            transition_states.append(
                self.transition_fusion(
                    torch.cat((graph_transition, theory_token), dim=-1)
                )
            )
            # Decode from graph_transition, not from the target token.
            gamma_predictions.append(self.gamma_decoder(graph_transition))
            gamma_targets.append(target)
        if transition_states:
            transition_stack = torch.stack(transition_states, dim=0)
            evolution_representation = self.evolution_pool(
                _masked_mean_std(transition_stack)
            )
        else:
            transition_stack = static_stack.new_zeros(
                (0, self.config.hidden_dim)
            )
            evolution_representation = static_representation.new_zeros(
                (self.config.hidden_dim,)
            )

        if self.config.variant == "NS_static_spectral":
            representation = static_representation
        elif self.config.variant == "NV_dynamic_evolution":
            representation = evolution_representation
        else:
            gate = torch.sigmoid(self.evolution_gate_logit)
            representation = self.joint_norm(
                static_representation
                + gate * self.evolution_residual(evolution_representation)
            )

        return TheoryNeuralSampleOutput(
            representation=representation,
            window_embeddings=static_stack,
            window_mask=sample.window_mask,
            q_predictions=(
                torch.stack(q_predictions, dim=0)
                if self.config.uses_static
                else None
            ),
            q_targets=(
                torch.stack(q_targets, dim=0)
                if self.config.uses_static
                else None
            ),
            gamma_predictions=(
                torch.stack(gamma_predictions, dim=0)
                if gamma_predictions and self.config.uses_evolution
                else transition_stack.new_zeros((0, self.config.transition_dim))
                if self.config.uses_evolution
                else None
            ),
            gamma_targets=(
                torch.stack(gamma_targets, dim=0)
                if gamma_targets and self.config.uses_evolution
                else transition_stack.new_zeros((0, self.config.transition_dim))
                if self.config.uses_evolution
                else None
            ),
            message_norms=tuple(all_norms),
            film_values=(
                torch.stack(film_values, dim=0) if film_values else None
            ),
        )

    def forward(self, batch: TheoryNeuralBatch) -> TheoryNeuralOutput:
        if len(batch) < 1:
            raise ValueError("corrected neural S/V batch cannot be empty")
        outputs = tuple(self._encode_sample(sample) for sample in batch)
        representations = torch.stack(
            [output.representation for output in outputs], dim=0
        )
        logits = self.classifier(representations)
        return TheoryNeuralOutput(
            logits=logits,
            representations=representations,
            samples=outputs,
            diagnostics={
                "variant": self.config.variant,
                "uses_coordinates": False,
                "uses_roi_identity": False,
                "uses_site_input": False,
                "uses_raw_community_embedding": False,
                "uses_signed_edge_channels": True,
                "uses_invariant_spectral_quantiles": True,
                "uses_directional_sgw_transitions": self.config.uses_evolution,
                "evolution_gate": float(
                    torch.sigmoid(self.evolution_gate_logit).detach().cpu()
                ),
            },
        )
