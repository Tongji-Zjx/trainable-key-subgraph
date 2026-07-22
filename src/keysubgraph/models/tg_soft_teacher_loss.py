"""Joint classification, budget and theory fidelity loss for the soft teacher."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .tg_soft_teacher import TGSoftTeacherOutput


TG_SOFT_TEACHER_ABLATIONS = (
    "classification_only",
    "classification_budget",
    "classification_budget_laplacian",
    "full",
)


def tg_soft_teacher_ablation_weights(name: str):
    """Return the fixed minimal nested ablation weights."""

    presets = {
        "classification_only": (0.0, 0.0, 0.0),
        "classification_budget": (0.10, 0.0, 0.0),
        "classification_budget_laplacian": (0.10, 0.50, 0.0),
        "full": (0.10, 0.50, 0.10),
    }
    if name not in presets:
        raise ValueError("unsupported TG soft-teacher ablation")
    return presets[name]


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
    per_sample_classification = F.cross_entropy(
        output.logits, labels, reduction="none"
    )
    if class_weights is None:
        classification = per_sample_classification.mean()
    else:
        # Dataset-balanced weights have expectation one over the training
        # cohort. Multiplying before the ordinary batch mean preserves their
        # effect for list batches of size one; PyTorch's weighted mean would
        # divide by the sole target weight and cancel it completely.
        classification = (
            per_sample_classification * class_weights.index_select(0, labels)
        ).mean()
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
    # Build only active branches into the autograd objective. Besides making
    # the ablations exact, this prevents a nominal ``0 * diagnostic`` branch
    # from participating in backward through expensive spectral operations.
    total = config.classification_weight * classification
    if config.budget_weight > 0.0:
        total = total + config.budget_weight * budget
    if laplacian_weight > 0.0:
        total = total + laplacian_weight * laplacian
    if gw_weight > 0.0:
        total = total + gw_weight * gw_identity
    if config.supervised_contrastive_weight > 0.0:
        total = total + config.supervised_contrastive_weight * contrastive
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
