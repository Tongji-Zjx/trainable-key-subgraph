"""Immutable sidecars and train-only scaling for SV theory geometry."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.data.data_split import file_sha256
from keysubgraph.features.sv_theory_geometry import (
    SV_DIFFUSION_GEOMETRY_DIM,
    SV_DIFFUSION_QUANTILES,
    SV_DIFFUSION_TIME_SCALES,
    SV_SPECTRAL_DIRECTION_DIM,
    SVTheoryGeometryExtractor,
)
from .sv_signed_gin_artifact import SVSignedGINRecord
from .sv_signed_gin_dataset import SVSignedGINDataset


SV_THEORY_FEATURE_CACHE_SCHEMA_VERSION = 1
SV_THEORY_FEATURE_SCALER_SCHEMA_VERSION = 1


def _trusted_load(path: Path):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(Path(path).resolve()), map_location="cpu"
        )


def _record_provenance(record: SVSignedGINRecord) -> Tuple:
    return (
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.selection_mode,
        int(record.selection_seed),
    )


def build_sv_theory_feature_payload(
    records: Sequence[SVSignedGINRecord],
    source_manifest_sha256: str,
    extractor: SVTheoryGeometryExtractor,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    if not records or not source_manifest_sha256:
        raise ValueError(
            "SV theory cache requires records and a source manifest"
        )
    splits = {record.split for record in records}
    provenance = {_record_provenance(record) for record in records}
    keys = [record.sample_key for record in records]
    if (
        len(splits) != 1
        or len(provenance) != 1
        or len(set(keys)) != len(keys)
    ):
        raise ValueError(
            "SV theory cache records mix split, provenance or keys"
        )
    items = []
    sorted_records = sorted(
        records, key=lambda value: value.sample_key
    )
    for index, record in enumerate(sorted_records):
        if device is None:
            windows = record.windows
        else:
            windows = tuple(
                (
                    replace(
                        window,
                        node_features=window.node_features.to(device),
                        adjacency=window.adjacency.to(device),
                    )
                    if window is not None
                    else None
                )
                for window in record.windows
            )
        features = extractor.build(windows)
        if not torch.equal(
            features.window_mask.cpu(), record.window_mask.cpu()
        ) or not torch.equal(
            features.transition_mask.cpu(),
            record.transition_mask.cpu(),
        ):
            raise ValueError(
                "SV theory masks disagree with the source hard cache"
            )
        items.append(
            {
                "sample_key": record.sample_key,
                "label": int(record.label),
                "site": record.site,
                "subject_id": record.subject_id,
                "spectral_direction": (
                    features.spectral_direction.detach()
                    .cpu()
                    .to(torch.float32)
                ),
                "diffusion_geometry": (
                    features.diffusion_geometry.detach()
                    .cpu()
                    .to(torch.float32)
                ),
                "valid_window_count": record.valid_window_count,
                "valid_transition_count": (
                    record.valid_transition_count
                ),
            }
        )
        if progress_callback is not None:
            progress_callback(
                index + 1, len(sorted_records), record.sample_key
            )
    common = next(iter(provenance))
    return {
        "schema_version": SV_THEORY_FEATURE_CACHE_SCHEMA_VERSION,
        "artifact_type": "sv_hard_sgw_theory_geometry_cache",
        "source_manifest_sha256": str(source_manifest_sha256),
        "split": next(iter(splits)),
        "sample_count": len(items),
        "dimensions": {
            "spectral_direction": SV_SPECTRAL_DIRECTION_DIM,
            "diffusion_geometry": extractor.diffusion_output_dim,
        },
        "configuration": {
            "laplacian_eta": extractor.laplacian_eta,
            "diffusion_time_scales": list(
                extractor.diffusion_time_scales
            ),
            "diffusion_quantiles": list(
                extractor.diffusion_quantiles
            ),
            "epsilon": extractor.epsilon,
        },
        "protocol_sha256": common[0],
        "selector_checkpoint_sha256": common[1],
        "selection_mode": common[2],
        "selection_seed": common[3],
        "records": items,
    }


def validate_sv_theory_feature_payload(payload: Mapping) -> None:
    if payload.get("schema_version") != (
        SV_THEORY_FEATURE_CACHE_SCHEMA_VERSION
    ) or payload.get("artifact_type") != (
        "sv_hard_sgw_theory_geometry_cache"
    ):
        raise ValueError("unsupported SV theory feature cache")
    if payload.get("split") not in ("train", "validation", "test"):
        raise ValueError("SV theory cache split is invalid")
    if payload.get("dimensions") != {
        "spectral_direction": SV_SPECTRAL_DIRECTION_DIM,
        "diffusion_geometry": SV_DIFFUSION_GEOMETRY_DIM,
    }:
        raise ValueError("SV theory cache dimensions are invalid")
    configuration = payload.get("configuration", {})
    if tuple(configuration.get("diffusion_time_scales", ())) != tuple(
        SV_DIFFUSION_TIME_SCALES
    ) or tuple(configuration.get("diffusion_quantiles", ())) != tuple(
        SV_DIFFUSION_QUANTILES
    ):
        raise ValueError("SV theory cache uses an unsupported grid")
    records = payload.get("records", [])
    if len(records) != int(payload.get("sample_count", -1)):
        raise ValueError("SV theory cache sample count mismatch")
    seen = set()
    for record in records:
        key = str(record.get("sample_key", ""))
        direction = record.get("spectral_direction")
        diffusion = record.get("diffusion_geometry")
        if (
            not key
            or key in seen
            or int(record.get("label", -1)) not in (0, 1)
            or not isinstance(direction, torch.Tensor)
            or not isinstance(diffusion, torch.Tensor)
            or tuple(direction.shape)
            != (SV_SPECTRAL_DIRECTION_DIM,)
            or tuple(diffusion.shape)
            != (SV_DIFFUSION_GEOMETRY_DIM,)
            or not bool(torch.isfinite(direction).all())
            or not bool(torch.isfinite(diffusion).all())
        ):
            raise ValueError("SV theory cache record is invalid")
        seen.add(key)
    required = (
        "source_manifest_sha256",
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
    )
    if any(not str(payload.get(name, "")) for name in required):
        raise ValueError("SV theory cache provenance is incomplete")


def save_sv_theory_feature_payload(
    payload: Mapping, path: Path, overwrite: bool = False
) -> Path:
    validate_sv_theory_feature_payload(payload)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("SV theory feature cache already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(path))
    return path


def load_sv_theory_feature_payload(path: Path) -> Dict:
    payload = _trusted_load(path)
    validate_sv_theory_feature_payload(payload)
    return payload


class SVTheoryFeatureStandardizer(nn.Module):
    def __init__(
        self,
        spectral_direction_mean: torch.Tensor,
        spectral_direction_scale: torch.Tensor,
        diffusion_geometry_mean: torch.Tensor,
        diffusion_geometry_scale: torch.Tensor,
        train_sample_count: int,
        train_feature_cache_sha256: str,
        train_manifest_sha256: str,
        protocol_sha256: str,
        selector_checkpoint_sha256: str,
        selection_mode: str,
        selection_seed: int,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        expected = (
            (spectral_direction_mean, SV_SPECTRAL_DIRECTION_DIM),
            (spectral_direction_scale, SV_SPECTRAL_DIRECTION_DIM),
            (diffusion_geometry_mean, SV_DIFFUSION_GEOMETRY_DIM),
            (diffusion_geometry_scale, SV_DIFFUSION_GEOMETRY_DIM),
        )
        if any(
            tuple(value.shape) != (dimension,)
            for value, dimension in expected
        ):
            raise ValueError("SV theory scaler dimensions are invalid")
        if any(
            not bool(torch.isfinite(value).all())
            for value, _ in expected
        ) or any(
            bool((value <= 0.0).any())
            for value in (
                spectral_direction_scale,
                diffusion_geometry_scale,
            )
        ):
            raise ValueError("SV theory scaler values are invalid")
        if (
            train_sample_count < 1
            or epsilon <= 0.0
            or not train_feature_cache_sha256
            or not train_manifest_sha256
        ):
            raise ValueError("SV theory scaler metadata is invalid")
        self.register_buffer(
            "spectral_direction_mean",
            spectral_direction_mean.to(torch.float32),
        )
        self.register_buffer(
            "spectral_direction_scale",
            spectral_direction_scale.to(torch.float32),
        )
        self.register_buffer(
            "diffusion_geometry_mean",
            diffusion_geometry_mean.to(torch.float32),
        )
        self.register_buffer(
            "diffusion_geometry_scale",
            diffusion_geometry_scale.to(torch.float32),
        )
        self.train_sample_count = int(train_sample_count)
        self.train_feature_cache_sha256 = str(
            train_feature_cache_sha256
        )
        self.train_manifest_sha256 = str(train_manifest_sha256)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(
            selector_checkpoint_sha256
        )
        self.selection_mode = str(selection_mode)
        self.selection_seed = int(selection_seed)
        self.epsilon = float(epsilon)

    def standardize_spectral_direction(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        if value.shape[-1] != SV_SPECTRAL_DIRECTION_DIM:
            raise ValueError("spectral direction must end in dimension 16")
        return (
            value - self.spectral_direction_mean.to(value)
        ) / self.spectral_direction_scale.to(value)

    def standardize_diffusion_geometry(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        if value.shape[-1] != SV_DIFFUSION_GEOMETRY_DIM:
            raise ValueError("diffusion geometry has the wrong dimension")
        return (
            value - self.diffusion_geometry_mean.to(value)
        ) / self.diffusion_geometry_scale.to(value)


def _mean_scale(
    values: torch.Tensor, epsilon: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.to(torch.float64)
    mean = values.mean(dim=0)
    variance = (values - mean).square().mean(dim=0)
    return mean, torch.sqrt(variance + float(epsilon))


def fit_sv_theory_feature_standardizer(
    train_payload: Mapping,
    train_feature_cache_sha256: str,
    epsilon: float = 1.0e-8,
) -> SVTheoryFeatureStandardizer:
    validate_sv_theory_feature_payload(train_payload)
    if train_payload["split"] != "train":
        raise ValueError("SV theory scaler may be fitted from train only")
    if not train_feature_cache_sha256:
        raise ValueError("SV theory scaler requires a cache hash")
    records = train_payload["records"]
    direction = torch.stack(
        [record["spectral_direction"] for record in records], dim=0
    )
    diffusion = torch.stack(
        [record["diffusion_geometry"] for record in records], dim=0
    )
    direction_mean, direction_scale = _mean_scale(
        direction, epsilon
    )
    diffusion_mean, diffusion_scale = _mean_scale(
        diffusion, epsilon
    )
    return SVTheoryFeatureStandardizer(
        spectral_direction_mean=direction_mean,
        spectral_direction_scale=direction_scale,
        diffusion_geometry_mean=diffusion_mean,
        diffusion_geometry_scale=diffusion_scale,
        train_sample_count=len(records),
        train_feature_cache_sha256=train_feature_cache_sha256,
        train_manifest_sha256=train_payload[
            "source_manifest_sha256"
        ],
        protocol_sha256=train_payload["protocol_sha256"],
        selector_checkpoint_sha256=train_payload[
            "selector_checkpoint_sha256"
        ],
        selection_mode=train_payload["selection_mode"],
        selection_seed=int(train_payload["selection_seed"]),
        epsilon=epsilon,
    )


def save_sv_theory_feature_standardizer(
    scaler: SVTheoryFeatureStandardizer,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("SV theory feature scaler already exists")
    payload = {
        "schema_version": SV_THEORY_FEATURE_SCALER_SCHEMA_VERSION,
        "artifact_type": "sv_theory_geometry_train_only_scaler",
        "fit_split": "train",
        "dimensions": {
            "spectral_direction": SV_SPECTRAL_DIRECTION_DIM,
            "diffusion_geometry": SV_DIFFUSION_GEOMETRY_DIM,
        },
        "train_sample_count": scaler.train_sample_count,
        "train_feature_cache_sha256": (
            scaler.train_feature_cache_sha256
        ),
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": (
            scaler.selector_checkpoint_sha256
        ),
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "epsilon": scaler.epsilon,
        "spectral_direction_mean": (
            scaler.spectral_direction_mean.detach().cpu().tolist()
        ),
        "spectral_direction_scale": (
            scaler.spectral_direction_scale.detach().cpu().tolist()
        ),
        "diffusion_geometry_mean": (
            scaler.diffusion_geometry_mean.detach().cpu().tolist()
        ),
        "diffusion_geometry_scale": (
            scaler.diffusion_geometry_scale.detach().cpu().tolist()
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_sv_theory_feature_standardizer(
    path: Path,
) -> SVTheoryFeatureStandardizer:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version")
        != SV_THEORY_FEATURE_SCALER_SCHEMA_VERSION
        or payload.get("artifact_type")
        != "sv_theory_geometry_train_only_scaler"
        or payload.get("fit_split") != "train"
        or payload.get("dimensions")
        != {
            "spectral_direction": SV_SPECTRAL_DIRECTION_DIM,
            "diffusion_geometry": SV_DIFFUSION_GEOMETRY_DIM,
        }
    ):
        raise ValueError("unsupported SV theory feature scaler")
    return SVTheoryFeatureStandardizer(
        spectral_direction_mean=torch.tensor(
            payload["spectral_direction_mean"]
        ),
        spectral_direction_scale=torch.tensor(
            payload["spectral_direction_scale"]
        ),
        diffusion_geometry_mean=torch.tensor(
            payload["diffusion_geometry_mean"]
        ),
        diffusion_geometry_scale=torch.tensor(
            payload["diffusion_geometry_scale"]
        ),
        train_sample_count=int(payload["train_sample_count"]),
        train_feature_cache_sha256=payload[
            "train_feature_cache_sha256"
        ],
        train_manifest_sha256=payload["train_manifest_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        selector_checkpoint_sha256=payload[
            "selector_checkpoint_sha256"
        ],
        selection_mode=payload["selection_mode"],
        selection_seed=int(payload["selection_seed"]),
        epsilon=float(payload["epsilon"]),
    )


class SVTheoryAugmentedDataset(SVSignedGINDataset):
    """Attach standardized theory sidecars to the existing SVG samples."""

    def __init__(
        self,
        manifest_path: Path,
        scaler_path: Path,
        theory_feature_cache_path: Path,
        theory_scaler_path: Path,
        max_samples=None,
        include_windows: bool = True,
    ) -> None:
        super().__init__(
            manifest_path,
            scaler_path,
            max_samples=max_samples,
            include_windows=include_windows,
        )
        theory_path = Path(theory_feature_cache_path).resolve()
        theory_payload = load_sv_theory_feature_payload(theory_path)
        theory_scaler = load_sv_theory_feature_standardizer(
            theory_scaler_path
        )
        base_provenance = (
            self.manifest["protocol_sha256"],
            self.manifest["selector_checkpoint_sha256"],
            self.manifest["selection_mode"],
            int(self.manifest["selection_seed"]),
        )
        theory_provenance = (
            theory_payload["protocol_sha256"],
            theory_payload["selector_checkpoint_sha256"],
            theory_payload["selection_mode"],
            int(theory_payload["selection_seed"]),
        )
        scaler_provenance = (
            theory_scaler.protocol_sha256,
            theory_scaler.selector_checkpoint_sha256,
            theory_scaler.selection_mode,
            int(theory_scaler.selection_seed),
        )
        if (
            theory_payload["split"] != self.split
            or theory_payload["source_manifest_sha256"]
            != file_sha256(self.manifest_path)
            or base_provenance != theory_provenance
            or base_provenance != scaler_provenance
        ):
            raise ValueError(
                "SV theory dataset provenance does not match the base cache"
            )
        if self.split == "train" and (
            theory_scaler.train_feature_cache_sha256
            != file_sha256(theory_path)
            or theory_scaler.train_manifest_sha256
            != file_sha256(self.manifest_path)
        ):
            raise ValueError(
                "SV theory train scaler does not match the train cache"
            )
        by_key = {
            str(record["sample_key"]): record
            for record in theory_payload["records"]
        }
        manifest_keys = {
            str(record["sample_key"])
            for record in self.manifest["records"]
        }
        if manifest_keys != set(by_key):
            raise ValueError(
                "SV theory cache keys do not equal base-dataset keys"
            )
        augmented = []
        for sample in self.samples:
            record = by_key[sample.sample_key]
            if int(record["label"]) != int(sample.label):
                raise ValueError("SV theory cache label mismatch")
            augmented.append(
                replace(
                    sample,
                    spectral_direction=(
                        theory_scaler.standardize_spectral_direction(
                            record["spectral_direction"].to(
                                torch.float32
                            )
                        )
                        .detach()
                        .clone()
                    ),
                    diffusion_geometry=(
                        theory_scaler.standardize_diffusion_geometry(
                            record["diffusion_geometry"].to(
                                torch.float32
                            )
                        )
                        .detach()
                        .clone()
                    ),
                )
            )
        self.samples = augmented
        self.theory_feature_cache_path = theory_path
        self.theory_scaler_path = Path(theory_scaler_path).resolve()
        self.theory_payload = theory_payload
        self.theory_scaler = theory_scaler
