"""Curriculum loss for the single-model Hard-STSE architecture."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .hard_stse_types import (
    HardSTSEConfig,
    HardSTSELossOutput,
    HardSTSEModelOutput,
)


@dataclass(frozen=True)
class HardSTSELossConfig:
    fusion_ce_weight: float = 1.0
    neural_aux_ce_weight: float = 0.3
    theory_aux_ce_weight: float = 0.3
    budget_weight_max: float = 0.10
    laplacian_weight_max: float = 0.05
    gw_proxy_weight_max: float = 0.02
    theory_ramp_epochs: int = 10

    def __post_init__(self) -> None:
        values = (
            self.fusion_ce_weight,
            self.neural_aux_ce_weight,
            self.theory_aux_ce_weight,
            self.budget_weight_max,
            self.laplacian_weight_max,
            self.gw_proxy_weight_max,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("Hard-STSE loss weights cannot be negative")
        if self.fusion_ce_weight <= 0.0:
            raise ValueError("fusion classification weight must be positive")
        if self.theory_ramp_epochs < 1:
            raise ValueError("theory ramp must contain at least one epoch")


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    if class_weights is None:
        return losses.mean()
    if tuple(class_weights.shape) != (logits.shape[-1],):
        raise ValueError("class weights do not match the classifier")
    sample_weights = class_weights.to(logits).index_select(0, labels)
    return (losses * sample_weights).mean()


class HardSTSECriterion(object):
    def __init__(
        self,
        model_config: HardSTSEConfig,
        loss_config: Optional[HardSTSELossConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.loss_config = loss_config or HardSTSELossConfig()

    def _curriculum_weights(self, epoch: int):
        schedule = self.model_config.selection_schedule
        if epoch <= schedule.high_retention_epochs:
            budget = 0.0
        elif epoch >= schedule.anneal_end_epoch:
            budget = self.loss_config.budget_weight_max
        else:
            progress = float(
                epoch - schedule.high_retention_epochs
            ) / float(
                schedule.anneal_end_epoch - schedule.high_retention_epochs
            )
            budget = progress * self.loss_config.budget_weight_max
        theory_start = schedule.anneal_end_epoch + 1
        theory_progress = max(
            0.0,
            min(
                1.0,
                float(epoch - theory_start + 1)
                / float(self.loss_config.theory_ramp_epochs),
            ),
        )
        return {
            "budget": budget,
            "laplacian": (
                theory_progress * self.loss_config.laplacian_weight_max
            ),
            "gw_proxy": (
                theory_progress * self.loss_config.gw_proxy_weight_max
            ),
        }

    def __call__(
        self,
        output: HardSTSEModelOutput,
        labels: torch.Tensor,
        epoch: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> HardSTSELossOutput:
        if labels.ndim != 1 or labels.shape[0] != output.fusion_logits.shape[0]:
            raise ValueError("labels do not align with Hard-STSE logits")
        fusion_ce = _weighted_cross_entropy(
            output.fusion_logits, labels, class_weights
        )
        neural_ce = _weighted_cross_entropy(
            output.neural_logits, labels, class_weights
        )
        zero = fusion_ce.new_zeros(())
        if output.theory_logits is not None:
            theory_ce = _weighted_cross_entropy(
                output.theory_logits, labels, class_weights
            )
            classification = (
                self.loss_config.fusion_ce_weight * fusion_ce
                + self.loss_config.neural_aux_ce_weight * neural_ce
                + self.loss_config.theory_aux_ce_weight * theory_ce
            )
        else:
            # In M0--M2 the fusion and neural heads are the same object; count
            # their CE exactly once.
            theory_ce = zero
            classification = fusion_ce

        selections = output.diagnostics.get("selections", ())
        node_values = [
            item.node_probabilities.reshape(-1) for item in selections
        ]
        edge_values = [
            item.edge_probabilities[
                item.edge_probabilities > 0.0
            ].reshape(-1)
            for item in selections
        ]
        edge_values = [item for item in edge_values if item.numel()]
        target_node, target_edge = (
            self.model_config.selection_schedule.ratios(epoch)
        )
        if self.model_config.selection_mode == "learned" and node_values:
            node_budget = (
                torch.cat(node_values).mean() - target_node
            ).abs()
            edge_budget = (
                (torch.cat(edge_values).mean() - target_edge).abs()
                if edge_values
                else zero
            )
        else:
            node_budget, edge_budget = zero, zero

        laplacian = output.diagnostics.get("laplacian_proxy", zero)
        gw_proxy = output.diagnostics.get("gw_proxy", zero)
        weights = self._curriculum_weights(epoch)
        total = (
            classification
            + weights["budget"] * (node_budget + edge_budget)
            + weights["laplacian"] * laplacian
            + weights["gw_proxy"] * gw_proxy
        )
        return HardSTSELossOutput(
            total=total,
            classification=classification,
            fusion_ce=fusion_ce,
            neural_ce=neural_ce,
            theory_ce=theory_ce,
            node_budget=node_budget,
            edge_budget=edge_budget,
            laplacian=laplacian,
            gw_proxy=gw_proxy,
            weights=weights,
        )
