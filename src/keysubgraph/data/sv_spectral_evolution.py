"""Leakage-safe spectral-transition data for the neural evolution branch."""

from __future__ import absolute_import, division, print_function

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.sv_signed_gin_artifact import SVSignedGINRecord
from keysubgraph.data.sv_signed_gin_manifest import (
    read_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (
    load_sv_signed_gin_standardizers,
)
from keysubgraph.models.sv_spectral_evolution import (
    SVSpectralEvolutionBatch,
    SVSpectralEvolutionSampleInput,
)
from keysubgraph.theory.spectral_gw import (
    SignedLaplacianBuilder,
    SpectralStateExtractor,
)


SV_SPECTRAL_STATE_DIM = 16
SV_SPECTRAL_TRANSITION_DIM = 32
SV_SPECTRAL_EVOLUTION_SCALER_SCHEMA_VERSION = 1


def spectral_quantile_grid() -> Tuple[float, ...]:
    """The frozen grid used by the existing 16-D Variation branch."""

    return tuple(0.05 + (0.90 / 15.0) * index for index in range(16))


def _record_provenance(record: SVSignedGINRecord) -> Tuple[str, str, str, int]:
    return (
        str(record.protocol_sha256),
        str(record.selector_checkpoint_sha256),
        str(record.selection_mode),
        int(record.selection_seed),
    )


def extract_spectral_transition_segments(
    record: SVSignedGINRecord,
    laplacian_eta: float = 1.0e-3,
    consistency_atol: float = 5.0e-5,
) -> Tuple[torch.Tensor, ...]:
    """Return contiguous signed/absolute spectral-difference segments.

    Invalid windows break a segment.  Consequently, a temporal convolution can
    never bridge a missing time point.  The absolute half of the returned
    transitions is checked against the immutable 16-D Variation feature.
    """

    if laplacian_eta <= 0.0 or consistency_atol <= 0.0:
        raise ValueError("spectral evolution constants must be positive")
    laplacian = SignedLaplacianBuilder(float(laplacian_eta))
    extractor = SpectralStateExtractor(spectral_quantile_grid())
    states: List[torch.Tensor] = []
    for window in record.windows:
        if window is None:
            states.append(None)
            continue
        adjacency = window.adjacency.to(torch.float32)
        edge_mask = adjacency.abs() > 0.0
        edge_mask = edge_mask.clone()
        edge_mask.fill_diagonal_(False)
        state = extractor(
            laplacian(adjacency, edge_mask=edge_mask)
        ).quantiles.to(torch.float32)
        if tuple(state.shape) != (SV_SPECTRAL_STATE_DIM,):
            raise RuntimeError("spectral state is not 16-D")
        states.append(state)

    segments: List[torch.Tensor] = []
    current: List[torch.Tensor] = []
    absolute_differences: List[torch.Tensor] = []
    for left, right in zip(states[:-1], states[1:]):
        if left is None or right is None:
            if current:
                segments.append(torch.stack(current, dim=0))
                current = []
            continue
        delta = right - left
        absolute = delta.abs()
        current.append(torch.cat((delta, absolute), dim=-1))
        absolute_differences.append(absolute)
    if current:
        segments.append(torch.stack(current, dim=0))

    expected_count = int(record.transition_mask.sum().item())
    actual_count = sum(int(segment.shape[0]) for segment in segments)
    if actual_count != expected_count:
        raise ValueError("spectral transitions disagree with frozen mask")
    reconstructed = (
        torch.stack(absolute_differences, dim=0).mean(dim=0)
        if absolute_differences
        else record.variation.new_zeros((SV_SPECTRAL_STATE_DIM,))
    )
    if not torch.allclose(
        reconstructed,
        record.variation.to(reconstructed),
        atol=float(consistency_atol),
        rtol=1.0e-4,
    ):
        maximum = float(
            (reconstructed - record.variation.to(reconstructed))
            .abs()
            .max()
            .item()
        )
        raise ValueError(
            "spectral sequence does not reconstruct frozen Variation "
            "(max_abs_difference={:.8g})".format(maximum)
        )
    if any(
        segment.ndim != 2
        or segment.shape[1] != SV_SPECTRAL_TRANSITION_DIM
        or not bool(torch.isfinite(segment).all())
        for segment in segments
    ):
        raise ValueError("spectral transition segment is invalid")
    return tuple(segments)


@dataclass(frozen=True)
class SVSpectralTransitionStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    train_sample_count: int
    train_transition_count: int
    train_manifest_sha256: str
    protocol_sha256: str
    selector_checkpoint_sha256: str
    selection_mode: str
    selection_seed: int
    laplacian_eta: float = 1.0e-3
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if tuple(self.mean.shape) != (SV_SPECTRAL_TRANSITION_DIM,) or tuple(
            self.scale.shape
        ) != (SV_SPECTRAL_TRANSITION_DIM,):
            raise ValueError("spectral transition scaler must be 32-D")
        if (
            not bool(torch.isfinite(self.mean).all())
            or not bool(torch.isfinite(self.scale).all())
            or bool((self.scale <= 0.0).any())
        ):
            raise ValueError("spectral transition scaler is invalid")
        if self.train_sample_count < 1 or self.train_transition_count < 1:
            raise ValueError("spectral transition scaler counts are invalid")

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != SV_SPECTRAL_TRANSITION_DIM:
            raise ValueError("spectral transitions must end in 32 dimensions")
        return (values - self.mean.to(values)) / self.scale.to(values)


def fit_sv_spectral_transition_standardizer(
    records: Sequence[SVSignedGINRecord],
    train_manifest_sha256: str,
    laplacian_eta: float = 1.0e-3,
    epsilon: float = 1.0e-8,
) -> SVSpectralTransitionStandardizer:
    if not records or not train_manifest_sha256:
        raise ValueError("transition scaler requires a train manifest")
    if any(record.split != "train" for record in records):
        raise ValueError("transition scaler may be fitted from train only")
    provenance = {_record_provenance(record) for record in records}
    if len(provenance) != 1:
        raise ValueError("transition scaler records mix provenance")
    values = [
        segment
        for record in records
        for segment in extract_spectral_transition_segments(
            record, laplacian_eta=laplacian_eta
        )
    ]
    if not values:
        raise ValueError("train split has no valid spectral transitions")
    stacked = torch.cat(values, dim=0).to(torch.float64)
    mean = stacked.mean(dim=0)
    scale = torch.sqrt(
        (stacked - mean).square().mean(dim=0) + float(epsilon)
    )
    fields = next(iter(provenance))
    return SVSpectralTransitionStandardizer(
        mean=mean.to(torch.float32),
        scale=scale.to(torch.float32),
        train_sample_count=len(records),
        train_transition_count=int(stacked.shape[0]),
        train_manifest_sha256=str(train_manifest_sha256),
        protocol_sha256=fields[0],
        selector_checkpoint_sha256=fields[1],
        selection_mode=fields[2],
        selection_seed=fields[3],
        laplacian_eta=float(laplacian_eta),
        epsilon=float(epsilon),
    )


def save_sv_spectral_transition_standardizer(
    scaler: SVSpectralTransitionStandardizer,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("spectral transition scaler already exists")
    payload = {
        "schema_version": SV_SPECTRAL_EVOLUTION_SCALER_SCHEMA_VERSION,
        "artifact_type": "sv_spectral_transition_train_only_scaler",
        "fit_split": "train",
        "dimension": SV_SPECTRAL_TRANSITION_DIM,
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "train_sample_count": scaler.train_sample_count,
        "train_transition_count": scaler.train_transition_count,
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": scaler.selector_checkpoint_sha256,
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "laplacian_eta": scaler.laplacian_eta,
        "epsilon": scaler.epsilon,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_sv_spectral_transition_standardizer(
    path: Path,
) -> SVSpectralTransitionStandardizer:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version")
        != SV_SPECTRAL_EVOLUTION_SCALER_SCHEMA_VERSION
        or payload.get("artifact_type")
        != "sv_spectral_transition_train_only_scaler"
        or payload.get("fit_split") != "train"
        or int(payload.get("dimension", -1))
        != SV_SPECTRAL_TRANSITION_DIM
    ):
        raise ValueError("unsupported spectral transition scaler")
    return SVSpectralTransitionStandardizer(
        mean=torch.tensor(payload["mean"], dtype=torch.float32),
        scale=torch.tensor(payload["scale"], dtype=torch.float32),
        train_sample_count=int(payload["train_sample_count"]),
        train_transition_count=int(payload["train_transition_count"]),
        train_manifest_sha256=str(payload["train_manifest_sha256"]),
        protocol_sha256=str(payload["protocol_sha256"]),
        selector_checkpoint_sha256=str(
            payload["selector_checkpoint_sha256"]
        ),
        selection_mode=str(payload["selection_mode"]),
        selection_seed=int(payload["selection_seed"]),
        laplacian_eta=float(payload["laplacian_eta"]),
        epsilon=float(payload["epsilon"]),
    )


class SVSpectralEvolutionDataset(Dataset):
    """Materialize static anchors and variable-length transition segments."""

    def __init__(
        self,
        manifest_path: Path,
        static_scaler_path: Path,
        transition_scaler_path: Path,
        shuffle_time: bool = False,
        shuffle_seed: int = 2026,
        max_samples=None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.static_scaler_path = Path(static_scaler_path).resolve()
        self.transition_scaler_path = Path(
            transition_scaler_path
        ).resolve()
        self.manifest, records = read_sv_signed_gin_manifest(
            self.manifest_path
        )
        self.static_scaler = load_sv_signed_gin_standardizers(
            self.static_scaler_path
        )
        self.transition_scaler = (
            load_sv_spectral_transition_standardizer(
                self.transition_scaler_path
            )
        )
        common = (
            self.manifest["protocol_sha256"],
            self.manifest["selector_checkpoint_sha256"],
            self.manifest["selection_mode"],
            int(self.manifest["selection_seed"]),
        )
        static_common = (
            self.static_scaler.protocol_sha256,
            self.static_scaler.selector_checkpoint_sha256,
            self.static_scaler.selection_mode,
            int(self.static_scaler.selection_seed),
        )
        transition_common = (
            self.transition_scaler.protocol_sha256,
            self.transition_scaler.selector_checkpoint_sha256,
            self.transition_scaler.selection_mode,
            int(self.transition_scaler.selection_seed),
        )
        if common != static_common or common != transition_common:
            raise ValueError("spectral evolution provenance mismatch")
        if self.manifest["split"] == "train":
            manifest_hash = file_sha256(self.manifest_path)
            if (
                self.static_scaler.train_manifest_sha256 != manifest_hash
                or self.transition_scaler.train_manifest_sha256
                != manifest_hash
            ):
                raise ValueError(
                    "spectral evolution train scaler manifest mismatch"
                )
        if max_samples is not None:
            if int(max_samples) < 1:
                raise ValueError("max_samples must be positive")
            records = records[: int(max_samples)]

        self.samples: List[SVSpectralEvolutionSampleInput] = []
        self.sites: List[str] = []
        self.subject_ids: List[str] = []
        for record_index, record in enumerate(records):
            segments = list(
                extract_spectral_transition_segments(
                    record,
                    laplacian_eta=self.transition_scaler.laplacian_eta,
                )
            )
            if shuffle_time:
                generator = torch.Generator()
                generator.manual_seed(
                    int(shuffle_seed) + 1000003 * record_index
                )
                segments = [
                    segment.index_select(
                        0,
                        torch.randperm(
                            segment.shape[0], generator=generator
                        ),
                    )
                    for segment in segments
                ]
            standardized = tuple(
                self.transition_scaler.standardize(segment)
                .detach()
                .clone()
                for segment in segments
            )
            self.samples.append(
                SVSpectralEvolutionSampleInput(
                    sample_key=record.sample_key,
                    label=int(record.label),
                    static_features=self.static_scaler.standardize_static(
                        record.static_features.to(torch.float32)
                    )
                    .detach()
                    .clone(),
                    transition_segments=standardized,
                )
            )
            self.sites.append(str(record.site))
            self.subject_ids.append(str(record.subject_id))
        if not self.samples:
            raise ValueError("spectral evolution dataset cannot be empty")

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(sample.label for sample in self.samples)

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(sample.sample_key for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SVSpectralEvolutionSampleInput:
        return self.samples[index]


def collate_sv_spectral_evolution(
    samples: Sequence[SVSpectralEvolutionSampleInput],
) -> SVSpectralEvolutionBatch:
    if not samples:
        raise ValueError("cannot collate an empty spectral evolution batch")
    return SVSpectralEvolutionBatch(tuple(samples))


def _seed_worker(worker_id):
    del worker_id
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_sv_spectral_evolution_loader(
    dataset: SVSpectralEvolutionDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid spectral evolution loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("validation/test loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_sv_spectral_evolution,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
