"""Low-capacity classifiers for frozen 34-D exact-SGW features."""

from __future__ import absolute_import, division, print_function

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


DUAL_SGW_FEATURE_CLASSIFIERS = ("linear", "small_mlp")
DUAL_SGW_FEATURE_CLASSIFIER_MODEL_NAME = "dual_sgw_feature_classifier"
DUAL_SGW_FEATURE_CLASSIFIER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DualSGWFeatureClassifierConfig:
    classifier_type: str
    input_dim: int = 34
    hidden_dim: int = 16
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.classifier_type not in DUAL_SGW_FEATURE_CLASSIFIERS:
            raise ValueError("unsupported dual SGW feature classifier")
        if self.input_dim != 34 or self.hidden_dim < 1:
            raise ValueError("dual SGW classifier dimensions are invalid")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dual SGW classifier dropout is invalid")


class DualSGWFeatureClassifier(nn.Module):
    """Linear logistic or fixed 34->16->2 feature classifier."""

    model_name = DUAL_SGW_FEATURE_CLASSIFIER_MODEL_NAME

    def __init__(self, config: DualSGWFeatureClassifierConfig) -> None:
        super().__init__()
        self.config = config
        if config.classifier_type == "linear":
            self.network = nn.Linear(config.input_dim, 2)
        else:
            self.network = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 2),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != 34:
            raise ValueError(
                "dual SGW classifier expects features with shape [B,34]"
            )
        logits = self.network(features)
        if tuple(logits.shape) != (features.shape[0], 2):
            raise RuntimeError("dual SGW classifier produced invalid logits")
        return logits

    def config_dict(self) -> Dict:
        return asdict(self.config)

