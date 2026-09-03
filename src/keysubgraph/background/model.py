"""Signed residual GCNII and bounded MoKSE background fusion."""

from __future__ import absolute_import, division, print_function

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class StaticBackgroundConfig:
    input_dim: int = 20
    hidden_dim: int = 64
    representation_dim: int = 24
    layers: int = 2
    dropout: float = 0.10
    initial_connection: float = 0.10
    identity_lambda: float = 0.50
    alpha_max: float = 0.50
    alpha_initial: float = 0.10

    def __post_init__(self):
        if min(self.input_dim, self.hidden_dim, self.representation_dim, self.layers) < 1:
            raise ValueError("background dimensions and layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("background dropout must be in [0,1)")
        if not 0.0 <= self.initial_connection <= 1.0:
            raise ValueError("initial connection must be in [0,1]")
        if self.identity_lambda <= 0.0:
            raise ValueError("identity lambda must be positive")
        if self.alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive")
        if not 0.0 < self.alpha_initial < self.alpha_max:
            raise ValueError("alpha_initial must be inside (0, alpha_max)")


class SignedWeightedGCNIILayer(nn.Module):
    def __init__(self, hidden_dim: int, layer_index: int, config: StaticBackgroundConfig):
        super().__init__()
        self.initial_connection = float(config.initial_connection)
        self.beta = float(math.log(config.identity_lambda / float(layer_index) + 1.0))
        self.message = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.identity_mapping = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, initial, positive, negative, mask):
        positive_message = torch.bmm(positive, state)
        negative_message = torch.bmm(negative, state)
        message = self.message(
            torch.cat(
                (state, positive_message, negative_message,
                 positive_message - negative_message),
                dim=-1,
            )
        )
        mixed = (
            (1.0 - self.initial_connection) * message
            + self.initial_connection * initial
        )
        identity = (1.0 - self.beta) * mixed + self.beta * self.identity_mapping(mixed)
        result = self.norm(state + self.dropout(torch.nn.functional.gelu(identity)))
        return result * mask.unsqueeze(-1).to(result.dtype)


def masked_mean_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / count
    variance = (((values - mean[:, None, :]) ** 2) * weights).sum(dim=1) / count
    return torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=-1)


class GlobalBackgroundGCN(nn.Module):
    def __init__(self, config: StaticBackgroundConfig = StaticBackgroundConfig()):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            SignedWeightedGCNIILayer(config.hidden_dim, index + 1, config)
            for index in range(config.layers)
        )
        pooled_dim = 2 * config.hidden_dim
        self.layer_residual = nn.Linear(pooled_dim, pooled_dim)
        self.layer_gate = nn.Parameter(torch.tensor(0.0))
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, config.representation_dim),
        )
        self.classifier_norm = nn.LayerNorm(config.representation_dim)
        self.classifier = nn.Linear(config.representation_dim, 1)

    def forward(self, node_features, positive, negative, node_mask):
        initial = self.input_projection(self.input_norm(node_features))
        initial = initial * node_mask.unsqueeze(-1).to(initial.dtype)
        state = initial
        pooled = []
        for layer in self.layers:
            state = layer(state, initial, positive, negative, node_mask)
            pooled.append(masked_mean_std(state, node_mask))
        graph = pooled[-1]
        if len(pooled) > 1:
            graph = graph + torch.sigmoid(self.layer_gate) * self.layer_residual(pooled[0])
        representation = self.projection(graph)
        logit = self.classifier(self.classifier_norm(representation)).squeeze(-1)
        return {
            "background_representation": representation,
            "background_logit": logit,
            "layer_gate": torch.sigmoid(self.layer_gate),
        }


class MoKSEBackgroundFusion(nn.Module):
    """A genuinely bounded global scalar residual fusion."""

    def __init__(self, alpha_max: float = 0.50, alpha_initial: float = 0.10):
        super().__init__()
        if not 0.0 < alpha_initial < alpha_max:
            raise ValueError("fusion alpha initialization must be inside its bound")
        self.alpha_max = float(alpha_max)
        ratio = alpha_initial / alpha_max
        self.alpha_parameter = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def forward(self, evolution_logit, background_logit):
        alpha = self.alpha_max * torch.sigmoid(self.alpha_parameter)
        bounded_background = torch.tanh(background_logit)
        residual = alpha * bounded_background
        return {
            "fused_logit": evolution_logit + residual,
            "fusion_alpha": alpha,
            "background_residual": residual,
        }


class MoKSEBackgroundModel(nn.Module):
    def __init__(self, config: StaticBackgroundConfig = StaticBackgroundConfig()):
        super().__init__()
        self.config = config
        self.background = GlobalBackgroundGCN(config)
        self.fusion = MoKSEBackgroundFusion(config.alpha_max, config.alpha_initial)

    def forward(self, node_features, positive, negative, node_mask, evolution_logit):
        output = self.background(node_features, positive, negative, node_mask)
        output.update(self.fusion(evolution_logit, output["background_logit"]))
        return output
