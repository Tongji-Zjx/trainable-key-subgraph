"""Detached exact 34-D SGW branch and immutable feature artifacts."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.theory.tg_features import (
    SGWFeatureExtractor,
    SGWTheoryFeatureConfig,
)
from .dual_stse_hard_sgw_types import DualSTSEHardSGWConfig
from .hard_stse_types import HardWindowOutput


DUAL_SGW_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DualExactSGWOutput:
    core: torch.Tensor
    variation: torch.Tensor
    representation: torch.Tensor
    transition_mask: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DualSGWFeatureRecord:
    sample_key: str
    label: int
    split: str
    selection_mode: str
    selection_seed: int
    core: torch.Tensor
    variation: torch.Tensor
    representation: torch.Tensor
    transition_mask: torch.Tensor
    protocol_sha256: str
    selector_checkpoint_sha256: str


class DualExactSGWBranch(nn.Module):
    """Compute exact cropped-graph SGW with no learned temporal encoder."""

    def __init__(
        self, config: DualSTSEHardSGWConfig = None
    ) -> None:
        super().__init__()
        self.config = config or DualSTSEHardSGWConfig()
        self.extractor = SGWFeatureExtractor(
            SGWTheoryFeatureConfig(
                laplacian_eta=self.config.laplacian_eta,
                diffusion_time=self.config.diffusion_time,
                time_quantity="speed",
            )
        )

    def forward(
        self,
        batch: ExactSTSEBatch,
        hard_windows: Sequence[Sequence[HardWindowOutput]],
    ) -> DualExactSGWOutput:
        if len(batch) != len(hard_windows) or len(batch) < 1:
            raise ValueError("exact SGW batch and hard graphs must align")
        extracted = []
        for sample, windows in zip(batch, hard_windows):
            if len(windows) != sample.num_timepoints:
                raise ValueError("exact SGW windows do not align with time")
            cropped = tuple(
                window.cropped_graph if window.window_valid else None
                for window in windows
            )
            times = tuple(
                float(value) for value in sample.graph.window_starts
            )
            extracted.append(
                self.extractor.compute_hard_graph_sequence(
                    cropped, times
                )
            )
        reference = batch.samples[0].graph.adjacency[0]
        core = torch.stack(
            [item.h_core.to(reference) for item in extracted], dim=0
        ).detach()
        variation = torch.stack(
            [item.h_variation.to(reference) for item in extracted],
            dim=0,
        ).detach()
        representation = torch.cat((core, variation), dim=-1).detach()
        maximum = max(
            int(item.transition_mask.numel()) for item in extracted
        )
        transition_mask = torch.zeros(
            (len(extracted), maximum),
            dtype=torch.bool,
            device=reference.device,
        )
        for index, item in enumerate(extracted):
            mask = item.transition_mask.to(
                device=reference.device, dtype=torch.bool
            )
            transition_mask[index, : mask.numel()] = mask
        if tuple(representation.shape[1:]) != (
            self.config.sgw_output_dim,
        ):
            raise RuntimeError("exact SGW output is not 34-D")
        return DualExactSGWOutput(
            core=core,
            variation=variation,
            representation=representation,
            transition_mask=transition_mask,
            diagnostics={
                "feature_semantics": "exact_cropped_graph_sgw",
                "is_exact_gw": True,
                "exact_features_detached": (
                    not representation.requires_grad
                ),
                "gw_solver_converged": tuple(
                    tuple(item.gw_solver_converged)
                    for item in extracted
                ),
                "valid_transition_count": int(transition_mask.sum()),
            },
        )


def save_dual_sgw_feature_record(
    record: DualSGWFeatureRecord,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("dual SGW feature artifact already exists")
    if tuple(record.representation.shape) != (34,):
        raise ValueError("cached dual SGW representation must be 34-D")
    if tuple(record.core.shape) != (18,) or tuple(
        record.variation.shape
    ) != (16,):
        raise ValueError("cached dual SGW core/variation dimensions are invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": DUAL_SGW_FEATURE_SCHEMA_VERSION,
            "artifact_type": "dual_stse_exact_sgw_feature",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_dual_sgw_feature_record(path: Path) -> DualSGWFeatureRecord:
    try:
        payload = torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            str(Path(path).resolve()), map_location="cpu"
        )
    if payload.get("schema_version") != DUAL_SGW_FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported dual SGW feature schema")
    if payload.get("artifact_type") != "dual_stse_exact_sgw_feature":
        raise ValueError("unexpected dual SGW feature artifact")
    record = payload.get("record")
    if not isinstance(record, DualSGWFeatureRecord):
        raise ValueError("invalid dual SGW feature record")
    if tuple(record.representation.shape) != (34,):
        raise ValueError("loaded dual SGW representation is not 34-D")
    return record

