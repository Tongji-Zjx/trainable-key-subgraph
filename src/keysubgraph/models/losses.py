"""Losses for the differentiable soft_graph baseline."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .soft_extractor import BatchModelOutput


@dataclass(frozen=True)
class SoftGraphLoss:
    total: torch.Tensor
    classification: torch.Tensor
    budget: torch.Tensor
    node_budget: torch.Tensor
    edge_budget: torch.Tensor


def compute_soft_graph_loss(
    output: BatchModelOutput,
    labels: torch.Tensor,
    target_node_ratio: float = 0.3,
    target_edge_ratio: float = 0.3,
    budget_weight: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> SoftGraphLoss:
    if target_node_ratio < 0.0 or target_node_ratio > 1.0:
        raise ValueError("target_node_ratio must be in [0, 1]")
    if target_edge_ratio < 0.0 or target_edge_ratio > 1.0:
        raise ValueError("target_edge_ratio must be in [0, 1]")
    if budget_weight < 0.0:
        raise ValueError("budget_weight must be non-negative")
    labels = labels.to(device=output.logits.device, dtype=torch.long)
    if class_weights is not None:
        class_weights = class_weights.to(device=output.logits.device, dtype=output.logits.dtype)
        if tuple(class_weights.shape) != (2,):
            raise ValueError("class_weights must have shape [2]")
    classification = F.cross_entropy(output.logits, labels, weight=class_weights)
    node_budget = (output.node_retention_ratios - target_node_ratio).abs().mean()
    edge_budget = (output.edge_retention_ratios - target_edge_ratio).abs().mean()
    budget = node_budget + edge_budget
    total = classification + budget_weight * budget
    return SoftGraphLoss(
        total=total,
        classification=classification,
        budget=budget,
        node_budget=node_budget,
        edge_budget=edge_budget,
    )
