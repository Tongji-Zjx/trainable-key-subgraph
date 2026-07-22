"""Classification and frozen-teacher distillation loss for Stage C."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .tg_soft_teacher_loss import supervised_contrastive_loss


@dataclass(frozen=True)
class TGHardStudentLossConfig:
    classification_weight: float = 1.0
    knowledge_distillation_weight: float = 0.50
    representation_distillation_weight: float = 0.10
    supervised_contrastive_weight: float = 0.05
    knowledge_distillation_temperature: float = 2.0
    contrastive_temperature: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.classification_weight,
            self.knowledge_distillation_weight,
            self.representation_distillation_weight,
            self.supervised_contrastive_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("TG hard-student loss weights must be non-negative")
        if self.knowledge_distillation_temperature <= 0.0:
            raise ValueError("knowledge-distillation temperature must be positive")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive temperature must be positive")


@dataclass(frozen=True)
class TGHardStudentLoss:
    total: torch.Tensor
    classification: torch.Tensor
    knowledge_distillation: torch.Tensor
    representation_distillation: torch.Tensor
    supervised_contrastive: torch.Tensor


def compute_tg_hard_student_loss(
    output,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_representation: torch.Tensor,
    config: Optional[TGHardStudentLossConfig] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> TGHardStudentLoss:
    config = config or TGHardStudentLossConfig()
    device, dtype = output.logits.device, output.logits.dtype
    labels = labels.to(device=device, dtype=torch.long)
    teacher_logits = teacher_logits.detach().to(device=device, dtype=dtype)
    teacher_representation = teacher_representation.detach().to(
        device=device, dtype=dtype
    )
    if teacher_logits.shape != output.logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if teacher_representation.shape != output.projected_neural_representation.shape:
        raise ValueError("student and teacher representations must have identical shapes")
    if class_weights is not None:
        class_weights = class_weights.to(device=device, dtype=dtype)
    classification = F.cross_entropy(output.logits, labels, weight=class_weights)
    temperature = float(config.knowledge_distillation_temperature)
    knowledge_distillation = (temperature ** 2) * F.kl_div(
        F.log_softmax(output.logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    )
    representation_distillation = F.mse_loss(
        output.projected_neural_representation, teacher_representation
    )
    contrastive = supervised_contrastive_loss(
        output.neural_representation,
        labels,
        config.contrastive_temperature,
    )
    total = (
        config.classification_weight * classification
        + config.knowledge_distillation_weight * knowledge_distillation
        + config.representation_distillation_weight * representation_distillation
        + config.supervised_contrastive_weight * contrastive
    )
    return TGHardStudentLoss(
        total,
        classification,
        knowledge_distillation,
        representation_distillation,
        contrastive,
    )
