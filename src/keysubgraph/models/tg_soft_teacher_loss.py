"""Joint classification, budget and theory fidelity loss for the soft teacher."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .tg_soft_teacher import TGSoftTeacherOutput


@dataclass(frozen=True)
class TGSoftTeacherLossConfig:
    classification_weight: float = 1.0
    budget_weight: float = 0.1
    laplacian_max_weight: float = 0.5
    gw_identity_max_weight: float = 0.1
    supervised_contrastive_weight: float = 0.0
    target_node_ratio: float = 0.3
    target_edge_ratio: float = 0.2
    theory_warmup_epochs: int = 15
    contrastive_temperature: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.classification_weight,
            self.budget_weight,
            self.laplacian_max_weight,
            self.gw_identity_max_weight,
            self.supervised_contrastive_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("TG soft-teacher loss weights must be non-negative")
        if not 0.0 <= self.target_node_ratio <= 1.0 or not 0.0 <= self.target_edge_ratio <= 1.0:
            raise ValueError("target retention ratios must lie in [0, 1]")
        if self.theory_warmup_epochs < 0 or self.contrastive_temperature <= 0.0:
            raise ValueError("warmup must be non-negative and temperature positive")


@dataclass(frozen=True)
class TGSoftTeacherLoss:
    total: torch.Tensor
    classification: torch.Tensor
    budget: torch.Tensor
    node_budget: torch.Tensor
    edge_budget: torch.Tensor
    laplacian_fidelity: torch.Tensor
    gw_identity_upper_bound: torch.Tensor
    supervised_contrastive: torch.Tensor
    effective_laplacian_weight: float
    effective_gw_weight: float


def _warmup(target: float, epoch: int, warmup_epochs: int) -> float:
    if epoch < 1:
        raise ValueError("training epoch is one-based and must be positive")
    if warmup_epochs == 0:
        return float(target)
    return float(target) * min(1.0, float(epoch) / float(warmup_epochs))


def supervised_contrastive_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    normalized = F.normalize(representation, dim=-1)
    similarity = normalized.matmul(normalized.transpose(0, 1)) / float(temperature)
    terms = []
    for index in range(representation.shape[0]):
        others = torch.arange(representation.shape[0], device=representation.device) != index
        positives = others & (labels == labels[index])
        if not bool(positives.any()):
            continue
        log_denominator = torch.logsumexp(similarity[index, others], dim=0)
        terms.append(-(similarity[index, positives] - log_denominator).mean())
    if not terms:
        return representation.sum() * 0.0
    return torch.stack(terms).mean()


def compute_tg_soft_teacher_loss(
    output: TGSoftTeacherOutput,
    labels: torch.Tensor,
    epoch: int,
    config: Optional[TGSoftTeacherLossConfig] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> TGSoftTeacherLoss:
    config = config or TGSoftTeacherLossConfig()
    labels = labels.to(device=output.logits.device, dtype=torch.long)
    if class_weights is not None:
        class_weights = class_weights.to(device=output.logits.device, dtype=output.logits.dtype)
    classification = F.cross_entropy(output.logits, labels, weight=class_weights)
    node_budget = (output.node_retention_ratios - config.target_node_ratio).abs().mean()
    edge_budget = (output.edge_retention_ratios - config.target_edge_ratio).abs().mean()
    budget = node_budget + edge_budget
    laplacian = output.laplacian_normalized_frobenius.mean()
    gw_identity = output.gw_identity_upper_bounds_squared.mean()
    contrastive = supervised_contrastive_loss(
        output.representation, labels, config.contrastive_temperature
    )
    laplacian_weight = _warmup(
        config.laplacian_max_weight, epoch, config.theory_warmup_epochs
    )
    gw_weight = _warmup(
        config.gw_identity_max_weight, epoch, config.theory_warmup_epochs
    )
    total = (
        config.classification_weight * classification
        + config.budget_weight * budget
        + laplacian_weight * laplacian
        + gw_weight * gw_identity
        + config.supervised_contrastive_weight * contrastive
    )
    return TGSoftTeacherLoss(
        total,
        classification,
        budget,
        node_budget,
        edge_budget,
        laplacian,
        gw_identity,
        contrastive,
        laplacian_weight,
        gw_weight,
    )

