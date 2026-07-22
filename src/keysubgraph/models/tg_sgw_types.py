"""Versioned contracts shared by the TG-SGW teacher and student models."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


TG_SGW_MODEL_NAME = "tg_sgw_net_v2"
TG_SGW_CHECKPOINT_SCHEMA_VERSION = 1
TG_SGW_SOFT_TEACHER_STAGE = "soft_teacher"
TG_SGW_HARD_STUDENT_STAGE = "hard_student"


def default_quantile_grid(count: int = 16) -> Tuple[float, ...]:
    """Return fixed interior quantiles without requiring NumPy."""

    if count < 1:
        raise ValueError("spectral quantile count must be positive")
    if count == 1:
        return (0.5,)
    lower, upper = 0.05, 0.95
    step = (upper - lower) / float(count - 1)
    return tuple(lower + step * index for index in range(count))


@dataclass(frozen=True)
class TGSGWTheoryConfig:
    """Canonical mathematical choices from the core derivation document."""

    laplacian_eta: float = 1.0e-3
    diffusion_time: float = 1.0
    spectral_quantile_grid: Tuple[float, ...] = default_quantile_grid()
    spectral_w1_grid_size: int = 256
    gw_order: int = 2
    gw_external_half_factor: bool = False
    node_measure: str = "uniform"
    time_quantity: str = "speed"
    laplacian_add_eta_to_numerator: bool = True

    def __post_init__(self) -> None:
        if self.laplacian_eta <= 0.0 or self.diffusion_time <= 0.0:
            raise ValueError("laplacian_eta and diffusion_time must be positive")
        grid = tuple(float(value) for value in self.spectral_quantile_grid)
        if not grid or any(value <= 0.0 or value >= 1.0 for value in grid):
            raise ValueError("spectral quantiles must lie strictly inside (0, 1)")
        if any(left >= right for left, right in zip(grid[:-1], grid[1:])):
            raise ValueError("spectral quantiles must be strictly increasing")
        if self.spectral_w1_grid_size < 2:
            raise ValueError("spectral_w1_grid_size must be at least two")
        if self.gw_order != 2 or self.gw_external_half_factor:
            raise ValueError("the canonical theory uses second-order GW without a half factor")
        if self.node_measure != "uniform":
            raise ValueError("the canonical first implementation uses a uniform node measure")
        if self.time_quantity != "speed":
            raise ValueError("the core derivation defines the 18-D representation with speeds")
        if not self.laplacian_add_eta_to_numerator:
            raise ValueError("the canonical Laplacian contains L_signed + eta I")

    @property
    def core_feature_dim(self) -> int:
        return len(self.spectral_quantile_grid) + 2

    @property
    def classification_feature_dim(self) -> int:
        return self.core_feature_dim + len(self.spectral_quantile_grid)


@dataclass(frozen=True)
class TGSGWDimensionConfig:
    extractor_node_feature_dim: int = 13
    classifier_node_feature_dim: int = 13
    edge_feature_dim: int = 4
    graph_embedding_dim: int = 96
    temporal_hidden_dim: int = 96
    neural_representation_dim: int = 192
    theory_core_dim: int = 18
    theory_variation_dim: int = 16
    theory_classification_dim: int = 34
    final_representation_dim: int = 226
    num_classes: int = 2

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(int(value) < 1 for value in values):
            raise ValueError("all TG-SGW dimensions must be positive")
        if self.theory_core_dim + self.theory_variation_dim != self.theory_classification_dim:
            raise ValueError("theory dimensions are inconsistent")
        if self.neural_representation_dim + self.theory_classification_dim != self.final_representation_dim:
            raise ValueError("final TG-SGW representation dimension is inconsistent")


@dataclass(frozen=True)
class TGSGWContract:
    theory: TGSGWTheoryConfig = TGSGWTheoryConfig()
    dimensions: TGSGWDimensionConfig = TGSGWDimensionConfig()
    model_name: str = TG_SGW_MODEL_NAME
    checkpoint_schema_version: int = TG_SGW_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.model_name != TG_SGW_MODEL_NAME:
            raise ValueError("unexpected TG-SGW model name")
        if self.checkpoint_schema_version != TG_SGW_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported TG-SGW checkpoint schema")
        if self.theory.core_feature_dim != self.dimensions.theory_core_dim:
            raise ValueError("quantile grid and core feature dimension disagree")
        if self.theory.classification_feature_dim != self.dimensions.theory_classification_dim:
            raise ValueError("quantile grid and classification feature dimension disagree")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_tg_sgw_checkpoint_header(payload: Dict[str, Any], stage: str) -> None:
    """Reject accidental loading of legacy or wrong-stage checkpoints."""

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("model_name") != TG_SGW_MODEL_NAME:
        raise ValueError("checkpoint is not a TG-SGW V2 checkpoint")
    if payload.get("schema_version") != TG_SGW_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported TG-SGW checkpoint schema version")
    if payload.get("stage") != stage:
        raise ValueError("TG-SGW checkpoint stage mismatch")

