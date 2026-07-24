"""Adapter that reuses the validated Exact-STSE-NoCoord implementation."""

from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from .exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
    ExactSTSEOutput,
)


@dataclass(frozen=True)
class DualSTSEChannelOutput:
    representation: torch.Tensor
    logits: torch.Tensor
    exact_output: ExactSTSEOutput


class ExistingNoCoordSTSEChannel(nn.Module):
    """Expose the existing 64-D subject embedding without changing its model."""

    def __init__(
        self, model: Optional[ExactSTSEClassifier] = None
    ) -> None:
        super().__init__()
        if model is None:
            model = ExactSTSEClassifier(
                ExactSTSEConfig(use_coordinates=False)
            )
        if model.config.use_coordinates:
            raise ValueError("the dual STSE channel must not use coordinates")
        if model.config.input_dim != 18 or model.config.hidden_dim != 64:
            raise ValueError(
                "the dual STSE channel requires the validated 18 -> 64 model"
            )
        self.model = model

    @property
    def config(self) -> ExactSTSEConfig:
        return self.model.config

    @property
    def representation_dim(self) -> int:
        return int(self.model.config.hidden_dim)

    def set_trainable(
        self, encoder: bool, classifier: bool
    ) -> None:
        for parameter in self.model.window_encoder.parameters():
            parameter.requires_grad_(bool(encoder))
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad_(bool(classifier))

    def forward(self, batch: ExactSTSEBatch) -> DualSTSEChannelOutput:
        output = self.model(batch)
        if output.subject_embedding.shape[-1] != self.representation_dim:
            raise RuntimeError("NoCoord-STSE returned an invalid representation")
        return DualSTSEChannelOutput(
            representation=output.subject_embedding,
            logits=output.logits,
            exact_output=output,
        )

