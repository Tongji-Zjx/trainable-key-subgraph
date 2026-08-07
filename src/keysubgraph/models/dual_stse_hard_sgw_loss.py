"""Stage-aware multi-head loss for Dual-STSE-HardSGW."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from .dual_stse_hard_sgw_types import DualSTSEHardSGWOutput


@dataclass(frozen=True)
class DualSTSEHardSGWLossConfig:
    selector_objective: str = "current"
    fusion_ce_weight: float = 1.0
    stse_aux_ce_weight: float = 0.30
    sgw_aux_ce_weight: float = 0.50
    selector_proxy_ce_weight: float = 0.50
    selector_soft_ce_weight: float = 0.25
    selector_hard_ce_weight: float = 0.25
    node_budget_weight: float = 0.05
    edge_budget_weight: float = 0.05
    laplacian_weight: float = 0.05
    gw_proxy_weight: float = 0.02
    soft_hard_spectral_weight: float = 0.05
    soft_hard_gw_weight: float = 0.02
    soft_hard_kd_weight: float = 0.05
    object_overlap_weight: float = 0.10
    object_reconstruction_weight: float = 0.10
    object_coverage_weight: float = 0.05
    object_temporal_weight: float = 0.05
    soft_hard_kd_temperature: float = 1.0
    soft_warmup_epochs: int = 3
    target_node_ratio: float = 0.50
    target_edge_ratio: float = 0.30

    def __post_init__(self) -> None:
        if self.selector_objective not in (
            "current",
            "full_soft",
            "full_soft_hard",
        ):
            raise ValueError("unsupported selector objective")
        weights = (
            self.fusion_ce_weight,
            self.stse_aux_ce_weight,
            self.sgw_aux_ce_weight,
            self.selector_proxy_ce_weight,
            self.selector_soft_ce_weight,
            self.selector_hard_ce_weight,
            self.node_budget_weight,
            self.edge_budget_weight,
            self.laplacian_weight,
            self.gw_proxy_weight,
            self.soft_hard_spectral_weight,
            self.soft_hard_gw_weight,
            self.soft_hard_kd_weight,
            self.object_overlap_weight,
            self.object_reconstruction_weight,
            self.object_coverage_weight,
            self.object_temporal_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("dual loss weights cannot be negative")
        if self.fusion_ce_weight <= 0.0:
            raise ValueError("dual fusion CE weight must be positive")
        if self.soft_hard_kd_temperature <= 0.0:
            raise ValueError("soft-hard KD temperature must be positive")
        if self.soft_warmup_epochs < 0:
            raise ValueError("soft warmup epochs cannot be negative")
        for ratio in (self.target_node_ratio, self.target_edge_ratio):
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError("dual budget targets must lie in (0,1]")


@dataclass(frozen=True)
class DualSTSEHardSGWLoss:
    total: torch.Tensor
    fusion_ce: torch.Tensor
    stse_ce: torch.Tensor
    sgw_ce: torch.Tensor
    selector_proxy_ce: torch.Tensor
    node_budget: torch.Tensor
    edge_budget: torch.Tensor
    laplacian: torch.Tensor
    gw_proxy: torch.Tensor
    selector_soft_ce: torch.Tensor
    selector_hard_ce: torch.Tensor
    soft_hard_spectral: torch.Tensor
    soft_hard_gw: torch.Tensor
    soft_hard_kd: torch.Tensor
    object_overlap: torch.Tensor
    object_reconstruction: torch.Tensor
    object_coverage: torch.Tensor
    object_temporal: torch.Tensor
    stage: str
    weights: Dict[str, float]


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("dual classifier logits must have shape [B,2]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("dual labels do not align with logits")
    losses = F.cross_entropy(logits, labels, reduction="none")
    if class_weights is None:
        return losses.mean()
    if tuple(class_weights.shape) != (2,):
        raise ValueError("dual class weights must have shape [2]")
    sample_weights = class_weights.to(logits).index_select(0, labels)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(
        1.0e-12
    )


class DualSTSEHardSGWCriterion(object):
    def __init__(
        self,
        config: Optional[DualSTSEHardSGWLossConfig] = None,
    ) -> None:
        self.config = config or DualSTSEHardSGWLossConfig()

    def __call__(
        self,
        output: DualSTSEHardSGWOutput,
        labels: torch.Tensor,
        stage: str,
        class_weights: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> DualSTSEHardSGWLoss:
        if stage not in ("selector_proxy", "sgw_classifier", "fusion"):
            raise ValueError("unsupported dual loss stage")
        zero = output.stse_logits.new_zeros(())
        fusion_ce = zero
        stse_ce = zero
        sgw_ce = zero
        selector_ce = zero
        node_budget = zero
        edge_budget = zero
        laplacian = zero
        gw_proxy = zero
        selector_soft_ce = zero
        selector_hard_ce = zero
        soft_hard_spectral = zero
        soft_hard_gw = zero
        soft_hard_kd = zero
        object_overlap = zero
        object_reconstruction = zero
        object_coverage = zero
        object_temporal = zero

        if stage == "selector_proxy":
            if output.selector_proxy_logits is None:
                raise ValueError("selector stage requires proxy logits")
            selector_hard_ce = _weighted_ce(
                output.selector_proxy_logits, labels, class_weights
            )
            selector_ce = selector_hard_ce
            objective = self.config.selector_objective
            if objective == "current":
                proxy = output.diagnostics.get("proxy")
                if proxy is None:
                    raise ValueError(
                        "selector stage requires proxy diagnostics"
                    )
                laplacian = proxy.laplacian_fidelity
                gw_proxy = proxy.gw_fidelity
            else:
                soft_logits = output.selector_soft_proxy_logits
                soft_proxy = output.diagnostics.get("soft_proxy")
                if soft_logits is None or soft_proxy is None:
                    raise ValueError(
                        "transfer selector requires soft proxy outputs"
                    )
                selector_soft_ce = _weighted_ce(
                    soft_logits, labels, class_weights
                )
                laplacian = soft_proxy.laplacian_fidelity
                gw_proxy = soft_proxy.gw_fidelity
                transfer = output.diagnostics.get(
                    "soft_hard_transfer"
                )
                if transfer is None:
                    raise ValueError(
                        "transfer selector requires quantization outputs"
                    )
                soft_hard_spectral = (
                    transfer.spectral_quantization
                )
                soft_hard_gw = transfer.gw_quantization_proxy
                temperature = self.config.soft_hard_kd_temperature
                teacher = F.softmax(
                    soft_logits.detach() / temperature, dim=-1
                )
                soft_hard_kd = (
                    F.kl_div(
                        F.log_softmax(
                            output.selector_proxy_logits / temperature,
                            dim=-1,
                        ),
                        teacher,
                        reduction="batchmean",
                    )
                    * temperature
                    * temperature
                )
            selections = output.diagnostics.get("selection", {}).get(
                "selections", ()
            )
            object_terms = output.diagnostics.get("selection", {}).get(
                "multi_object_regularization"
            )
            if object_terms is not None:
                object_overlap = object_terms["overlap"]
                object_reconstruction = object_terms["reconstruction"]
                object_coverage = object_terms["coverage"]
                object_temporal = object_terms["temporal"]
            node_values = [
                item.node_probabilities.reshape(-1)
                for item in selections
            ]
            edge_values = [
                item.edge_probabilities[
                    item.edge_probabilities > 0.0
                ].reshape(-1)
                for item in selections
            ]
            edge_values = [item for item in edge_values if item.numel()]
            if node_values:
                node_budget = (
                    torch.cat(node_values).mean()
                    - self.config.target_node_ratio
                ).abs()
            if edge_values:
                edge_budget = (
                    torch.cat(edge_values).mean()
                    - self.config.target_edge_ratio
                ).abs()
            if objective == "current":
                total = (
                    self.config.selector_proxy_ce_weight * selector_ce
                    + self.config.node_budget_weight * node_budget
                    + self.config.edge_budget_weight * edge_budget
                    + self.config.laplacian_weight * laplacian
                    + self.config.gw_proxy_weight * gw_proxy
                )
            else:
                warmup = (
                    objective == "full_soft_hard"
                    and epoch is not None
                    and epoch <= self.config.soft_warmup_epochs
                )
                hard_weight = (
                    0.0
                    if warmup
                    else self.config.selector_hard_ce_weight
                )
                total = (
                    self.config.selector_soft_ce_weight
                    * selector_soft_ce
                    + hard_weight * selector_hard_ce
                    + self.config.node_budget_weight * node_budget
                    + self.config.edge_budget_weight * edge_budget
                    + self.config.laplacian_weight * laplacian
                    + self.config.gw_proxy_weight * gw_proxy
                )
                if objective == "full_soft_hard" and not warmup:
                    total = (
                        total
                        + self.config.soft_hard_spectral_weight
                        * soft_hard_spectral
                        + self.config.soft_hard_gw_weight
                        * soft_hard_gw
                        + self.config.soft_hard_kd_weight
                        * soft_hard_kd
                    )
            total = (
                total
                + self.config.object_overlap_weight * object_overlap
                + self.config.object_reconstruction_weight
                * object_reconstruction
                + self.config.object_coverage_weight * object_coverage
                + self.config.object_temporal_weight * object_temporal
            )
        elif stage == "sgw_classifier":
            if output.sgw_logits is None:
                raise ValueError("SGW stage requires exact SGW logits")
            sgw_ce = _weighted_ce(
                output.sgw_logits, labels, class_weights
            )
            total = sgw_ce
        else:
            if output.sgw_logits is None:
                raise ValueError("fusion stage requires SGW logits")
            fusion_ce = _weighted_ce(
                output.fusion_logits, labels, class_weights
            )
            stse_ce = _weighted_ce(
                output.stse_logits, labels, class_weights
            )
            sgw_ce = _weighted_ce(
                output.sgw_logits, labels, class_weights
            )
            total = (
                self.config.fusion_ce_weight * fusion_ce
                + self.config.stse_aux_ce_weight * stse_ce
                + self.config.sgw_aux_ce_weight * sgw_ce
            )
        return DualSTSEHardSGWLoss(
            total=total,
            fusion_ce=fusion_ce,
            stse_ce=stse_ce,
            sgw_ce=sgw_ce,
            selector_proxy_ce=selector_ce,
            node_budget=node_budget,
            edge_budget=edge_budget,
            laplacian=laplacian,
            gw_proxy=gw_proxy,
            selector_soft_ce=selector_soft_ce,
            selector_hard_ce=selector_hard_ce,
            soft_hard_spectral=soft_hard_spectral,
            soft_hard_gw=soft_hard_gw,
            soft_hard_kd=soft_hard_kd,
            object_overlap=object_overlap,
            object_reconstruction=object_reconstruction,
            object_coverage=object_coverage,
            object_temporal=object_temporal,
            stage=stage,
            weights={
                "fusion_ce": self.config.fusion_ce_weight,
                "stse_aux_ce": self.config.stse_aux_ce_weight,
                "sgw_aux_ce": self.config.sgw_aux_ce_weight,
                "selector_proxy_ce": (
                    self.config.selector_proxy_ce_weight
                ),
                "selector_soft_ce": self.config.selector_soft_ce_weight,
                "selector_hard_ce": self.config.selector_hard_ce_weight,
                "node_budget": self.config.node_budget_weight,
                "edge_budget": self.config.edge_budget_weight,
                "laplacian": self.config.laplacian_weight,
                "gw_proxy": self.config.gw_proxy_weight,
                "soft_hard_spectral": (
                    self.config.soft_hard_spectral_weight
                ),
                "soft_hard_gw": self.config.soft_hard_gw_weight,
                "soft_hard_kd": self.config.soft_hard_kd_weight,
                "object_overlap": self.config.object_overlap_weight,
                "object_reconstruction": (
                    self.config.object_reconstruction_weight
                ),
                "object_coverage": self.config.object_coverage_weight,
                "object_temporal": self.config.object_temporal_weight,
            },
        )
