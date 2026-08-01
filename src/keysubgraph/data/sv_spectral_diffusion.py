"""Immutable exact spectral-diffusion sidecars for SVG-v2 experiments."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn

from keysubgraph.data.data_split import file_sha256
from keysubgraph.features.sv_spectral_diffusion import (
    SV_HKS_DIM,
    SV_HKS_TIME_SCALES,
    SV_SPECTRAL_STATE_DIM,
    SVSpectralDiffusionExtractor,
    SVSpectralDiffusionWindowFeatures,
)
from .sv_signed_gin_artifact import SVSignedGINRecord
from .sv_signed_gin_dataset import SVSignedGINDataset


SV_SPECTRAL_DIFFUSION_RECORD_SCHEMA_VERSION = 1
SV_SPECTRAL_DIFFUSION_MANIFEST_SCHEMA_VERSION = 1
SV_SPECTRAL_DIFFUSION_SCALER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SVSpectralDiffusionRecord:
    sample_key: str
    label: int
    split: str
    windows: Tuple[object, ...]
    window_mask: torch.Tensor
    transition_mask: torch.Tensor
    source_feature_sha256: str
    source_manifest_sha256: str
    protocol_sha256: str
    selector_checkpoint_sha256: str
    selection_mode: str
    selection_seed: int
    laplacian_eta: float
    hks_time_scales: Tuple[float, ...]


def _trusted_load(path: Path):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location="cpu")


def sv_spectral_diffusion_filename(sample_key: str) -> str:
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def build_sv_spectral_diffusion_record(
    source: SVSignedGINRecord,
    source_feature_sha256: str,
    source_manifest_sha256: str,
    extractor: SVSpectralDiffusionExtractor,
) -> SVSpectralDiffusionRecord:
    if not source_feature_sha256 or not source_manifest_sha256:
        raise ValueError("spectral-diffusion source hashes are required")
    features = extractor.build(source.windows)
    if not torch.equal(features.window_mask.cpu(), source.window_mask.cpu()):
        raise ValueError("spectral-diffusion window mask mismatch")
    if not torch.equal(
        features.transition_mask.cpu(), source.transition_mask.cpu()
    ):
        raise ValueError("spectral-diffusion transition mask mismatch")
    record = SVSpectralDiffusionRecord(
        sample_key=source.sample_key,
        label=int(source.label),
        split=source.split,
        windows=features.windows,
        window_mask=features.window_mask.cpu(),
        transition_mask=features.transition_mask.cpu(),
        source_feature_sha256=str(source_feature_sha256),
        source_manifest_sha256=str(source_manifest_sha256),
        protocol_sha256=source.protocol_sha256,
        selector_checkpoint_sha256=source.selector_checkpoint_sha256,
        selection_mode=source.selection_mode,
        selection_seed=int(source.selection_seed),
        laplacian_eta=extractor.laplacian_eta,
        hks_time_scales=tuple(extractor.hks_time_scales),
    )
    validate_sv_spectral_diffusion_record(record)
    return record


def validate_sv_spectral_diffusion_record(
    record: SVSpectralDiffusionRecord,
) -> None:
    if not isinstance(record, SVSpectralDiffusionRecord):
        raise ValueError("invalid spectral-diffusion record")
    if (
        not record.sample_key
        or int(record.label) not in (0, 1)
        or record.split not in ("train", "validation", "test")
        or not record.windows
    ):
        raise ValueError("spectral-diffusion identity is invalid")
    if tuple(record.window_mask.shape) != (len(record.windows),) or tuple(
        record.transition_mask.shape
    ) != (max(0, len(record.windows) - 1),):
        raise ValueError("spectral-diffusion masks are misaligned")
    if record.window_mask.dtype != torch.bool or record.transition_mask.dtype != torch.bool:
        raise ValueError("spectral-diffusion masks must be boolean")
    expected_transition = torch.tensor(
        [
            bool(record.window_mask[index])
            and bool(record.window_mask[index + 1])
            for index in range(max(0, len(record.windows) - 1))
        ],
        dtype=torch.bool,
    )
    if not torch.equal(record.transition_mask.cpu(), expected_transition):
        raise ValueError("spectral-diffusion transition mask is invalid")
    if tuple(record.hks_time_scales) != tuple(SV_HKS_TIME_SCALES):
        raise ValueError("spectral-diffusion HKS grid is invalid")
    for index, window in enumerate(record.windows):
        if (window is not None) != bool(record.window_mask[index]):
            raise ValueError("spectral-diffusion window payload/mask mismatch")
        if window is None:
            continue
        if not isinstance(window, SVSpectralDiffusionWindowFeatures):
            raise ValueError("invalid spectral-diffusion window type")
        count = int(window.eigenvalues.numel())
        tensors = (
            window.eigenvalues,
            window.eigenvectors,
            window.hks,
            window.spectral_quantiles,
        )
        if (
            count < 1
            or tuple(window.eigenvalues.shape) != (count,)
            or tuple(window.eigenvectors.shape) != (count, count)
            or tuple(window.hks.shape) != (count, SV_HKS_DIM)
            or tuple(window.spectral_quantiles.shape)
            != (SV_SPECTRAL_STATE_DIM,)
            or any(not bool(torch.isfinite(value).all()) for value in tensors)
        ):
            raise ValueError("spectral-diffusion tensors are invalid")
        identity = window.eigenvectors.transpose(0, 1).matmul(
            window.eigenvectors
        )
        if not torch.allclose(
            identity,
            torch.eye(
                count,
                dtype=identity.dtype,
                device=identity.device,
            ),
            atol=2.0e-4,
            rtol=0.0,
        ):
            raise ValueError("spectral-diffusion eigenvectors are not orthogonal")
    required = (
        record.source_feature_sha256,
        record.source_manifest_sha256,
        record.protocol_sha256,
        record.selector_checkpoint_sha256,
        record.selection_mode,
    )
    if any(not str(value) for value in required) or record.laplacian_eta <= 0.0:
        raise ValueError("spectral-diffusion provenance is incomplete")


def save_sv_spectral_diffusion_record(
    record: SVSpectralDiffusionRecord,
    path: Path,
    overwrite: bool = False,
) -> Path:
    validate_sv_spectral_diffusion_record(record)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("spectral-diffusion record already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": SV_SPECTRAL_DIFFUSION_RECORD_SCHEMA_VERSION,
            "artifact_type": "sv_spectral_diffusion_record",
            "record": record,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_sv_spectral_diffusion_record(path: Path) -> SVSpectralDiffusionRecord:
    payload = _trusted_load(path)
    if (
        payload.get("schema_version")
        != SV_SPECTRAL_DIFFUSION_RECORD_SCHEMA_VERSION
        or payload.get("artifact_type") != "sv_spectral_diffusion_record"
    ):
        raise ValueError("unsupported spectral-diffusion record")
    record = payload.get("record")
    validate_sv_spectral_diffusion_record(record)
    return record


def _record_provenance(record: SVSpectralDiffusionRecord) -> Dict:
    return {
        "source_manifest_sha256": record.source_manifest_sha256,
        "protocol_sha256": record.protocol_sha256,
        "selector_checkpoint_sha256": record.selector_checkpoint_sha256,
        "selection_mode": record.selection_mode,
        "selection_seed": int(record.selection_seed),
        "laplacian_eta": float(record.laplacian_eta),
        "hks_time_scales": list(record.hks_time_scales),
    }


def write_sv_spectral_diffusion_manifest(
    records: Sequence[Tuple[SVSpectralDiffusionRecord, Path]],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("spectral-diffusion manifest already exists")
    if not records:
        raise ValueError("spectral-diffusion manifest cannot be empty")
    splits = {record.split for record, _ in records}
    provenance = {
        json.dumps(_record_provenance(record), sort_keys=True)
        for record, _ in records
    }
    keys = [record.sample_key for record, _ in records]
    if len(splits) != 1 or len(provenance) != 1 or len(set(keys)) != len(keys):
        raise ValueError("spectral-diffusion manifest mixes provenance")
    rows = []
    for record, record_path in sorted(records, key=lambda item: item[0].sample_key):
        resolved = Path(record_path).resolve()
        try:
            relative = resolved.relative_to(output_path.parent).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        rows.append(
            {
                "sample_key": record.sample_key,
                "label": int(record.label),
                "split": record.split,
                "window_count": len(record.windows),
                "valid_window_count": int(record.window_mask.sum()),
                "valid_transition_count": int(record.transition_mask.sum()),
                "source_feature_sha256": record.source_feature_sha256,
                "feature_path": relative,
                "feature_sha256": file_sha256(resolved),
            }
        )
    payload = {
        "schema_version": SV_SPECTRAL_DIFFUSION_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "sv_spectral_diffusion_manifest",
        "split": next(iter(splits)),
        "sample_count": len(rows),
        "records": rows,
        **_record_provenance(records[0][0])
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def read_sv_spectral_diffusion_manifest(
    path: Path,
) -> Tuple[Dict, List[SVSpectralDiffusionRecord]]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version")
        != SV_SPECTRAL_DIFFUSION_MANIFEST_SCHEMA_VERSION
        or payload.get("artifact_type") != "sv_spectral_diffusion_manifest"
    ):
        raise ValueError("unsupported spectral-diffusion manifest")
    rows = payload.get("records", [])
    if len(rows) != int(payload.get("sample_count", -1)):
        raise ValueError("spectral-diffusion manifest count mismatch")
    records = []
    seen = set()
    expected_provenance = {
        name: payload[name]
        for name in (
            "source_manifest_sha256",
            "protocol_sha256",
            "selector_checkpoint_sha256",
            "selection_mode",
            "selection_seed",
            "laplacian_eta",
            "hks_time_scales",
        )
    }
    for row in rows:
        record_path = Path(row["feature_path"])
        if not record_path.is_absolute():
            record_path = path.parent / record_path
        if file_sha256(record_path) != row["feature_sha256"]:
            raise ValueError("spectral-diffusion artifact hash mismatch")
        record = load_sv_spectral_diffusion_record(record_path)
        checks = (
            record.sample_key == row["sample_key"],
            int(record.label) == int(row["label"]),
            record.split == row["split"] == payload["split"],
            len(record.windows) == int(row["window_count"]),
            int(record.window_mask.sum()) == int(row["valid_window_count"]),
            int(record.transition_mask.sum())
            == int(row["valid_transition_count"]),
            record.source_feature_sha256 == row["source_feature_sha256"],
            _record_provenance(record) == expected_provenance,
            record.sample_key not in seen,
        )
        if not all(checks):
            raise ValueError("spectral-diffusion manifest record mismatch")
        seen.add(record.sample_key)
        records.append(record)
    return payload, records


class SVSpectralDiffusionStandardizer(nn.Module):
    def __init__(
        self,
        hks_mean: torch.Tensor,
        hks_scale: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_scale: torch.Tensor,
        train_sample_count: int,
        train_manifest_sha256: str,
        source_manifest_sha256: str,
        protocol_sha256: str,
        selector_checkpoint_sha256: str,
        selection_mode: str,
        selection_seed: int,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if (
            tuple(hks_mean.shape) != (SV_HKS_DIM,)
            or tuple(hks_scale.shape) != (SV_HKS_DIM,)
            or tuple(delta_mean.shape) != (SV_SPECTRAL_STATE_DIM,)
            or tuple(delta_scale.shape) != (SV_SPECTRAL_STATE_DIM,)
            or any(
                not bool(torch.isfinite(value).all())
                for value in (hks_mean, hks_scale, delta_mean, delta_scale)
            )
            or bool((hks_scale <= 0.0).any())
            or bool((delta_scale <= 0.0).any())
        ):
            raise ValueError("spectral-diffusion scaler tensors are invalid")
        if train_sample_count < 1 or epsilon <= 0.0:
            raise ValueError("spectral-diffusion scaler metadata is invalid")
        self.register_buffer("hks_mean", hks_mean.to(torch.float32))
        self.register_buffer("hks_scale", hks_scale.to(torch.float32))
        self.register_buffer("delta_mean", delta_mean.to(torch.float32))
        self.register_buffer("delta_scale", delta_scale.to(torch.float32))
        self.train_sample_count = int(train_sample_count)
        self.train_manifest_sha256 = str(train_manifest_sha256)
        self.source_manifest_sha256 = str(source_manifest_sha256)
        self.protocol_sha256 = str(protocol_sha256)
        self.selector_checkpoint_sha256 = str(selector_checkpoint_sha256)
        self.selection_mode = str(selection_mode)
        self.selection_seed = int(selection_seed)
        self.epsilon = float(epsilon)

    def standardize_hks(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != SV_HKS_DIM:
            raise ValueError("HKS must end in dimension 6")
        return (value - self.hks_mean.to(value)) / self.hks_scale.to(value)

    def standardize_delta(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != SV_SPECTRAL_STATE_DIM:
            raise ValueError("spectral delta must end in dimension 16")
        return (value - self.delta_mean.to(value)) / self.delta_scale.to(value)


def _mean_scale(value: torch.Tensor, epsilon: float):
    value = value.to(torch.float64)
    mean = value.mean(dim=0)
    scale = torch.sqrt((value - mean).square().mean(dim=0) + float(epsilon))
    return mean, scale


def fit_sv_spectral_diffusion_standardizer(
    train_manifest: Mapping,
    records: Sequence[SVSpectralDiffusionRecord],
    train_manifest_sha256: str,
    epsilon: float = 1.0e-8,
) -> SVSpectralDiffusionStandardizer:
    if (
        train_manifest.get("split") != "train"
        or not records
        or not train_manifest_sha256
    ):
        raise ValueError("spectral-diffusion scaler requires train records")
    hks_values = []
    delta_values = []
    for record in records:
        if record.split != "train":
            raise ValueError("spectral-diffusion scaler saw a non-train record")
        for window in record.windows:
            if window is not None:
                hks_values.append(window.hks)
        for left, right in zip(record.windows[:-1], record.windows[1:]):
            if left is not None and right is not None:
                delta_values.append(
                    right.spectral_quantiles - left.spectral_quantiles
                )
    if not hks_values or not delta_values:
        raise ValueError("spectral-diffusion train cache lacks valid states")
    hks_mean, hks_scale = _mean_scale(torch.cat(hks_values, dim=0), epsilon)
    delta_mean, delta_scale = _mean_scale(torch.stack(delta_values), epsilon)
    return SVSpectralDiffusionStandardizer(
        hks_mean,
        hks_scale,
        delta_mean,
        delta_scale,
        len(records),
        train_manifest_sha256,
        train_manifest["source_manifest_sha256"],
        train_manifest["protocol_sha256"],
        train_manifest["selector_checkpoint_sha256"],
        train_manifest["selection_mode"],
        int(train_manifest["selection_seed"]),
        epsilon,
    )


def save_sv_spectral_diffusion_standardizer(
    scaler: SVSpectralDiffusionStandardizer,
    path: Path,
    overwrite: bool = False,
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("spectral-diffusion scaler already exists")
    payload = {
        "schema_version": SV_SPECTRAL_DIFFUSION_SCALER_SCHEMA_VERSION,
        "artifact_type": "sv_spectral_diffusion_train_only_scaler",
        "fit_split": "train",
        "hks_time_scales": list(SV_HKS_TIME_SCALES),
        "train_sample_count": scaler.train_sample_count,
        "train_manifest_sha256": scaler.train_manifest_sha256,
        "source_manifest_sha256": scaler.source_manifest_sha256,
        "protocol_sha256": scaler.protocol_sha256,
        "selector_checkpoint_sha256": scaler.selector_checkpoint_sha256,
        "selection_mode": scaler.selection_mode,
        "selection_seed": scaler.selection_seed,
        "epsilon": scaler.epsilon,
        "hks_mean": scaler.hks_mean.detach().cpu().tolist(),
        "hks_scale": scaler.hks_scale.detach().cpu().tolist(),
        "delta_mean": scaler.delta_mean.detach().cpu().tolist(),
        "delta_scale": scaler.delta_scale.detach().cpu().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))
    return path


def load_sv_spectral_diffusion_standardizer(
    path: Path,
) -> SVSpectralDiffusionStandardizer:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version")
        != SV_SPECTRAL_DIFFUSION_SCALER_SCHEMA_VERSION
        or payload.get("artifact_type")
        != "sv_spectral_diffusion_train_only_scaler"
        or payload.get("fit_split") != "train"
        or tuple(payload.get("hks_time_scales", ()))
        != tuple(SV_HKS_TIME_SCALES)
    ):
        raise ValueError("unsupported spectral-diffusion scaler")
    return SVSpectralDiffusionStandardizer(
        torch.tensor(payload["hks_mean"]),
        torch.tensor(payload["hks_scale"]),
        torch.tensor(payload["delta_mean"]),
        torch.tensor(payload["delta_scale"]),
        int(payload["train_sample_count"]),
        payload["train_manifest_sha256"],
        payload["source_manifest_sha256"],
        payload["protocol_sha256"],
        payload["selector_checkpoint_sha256"],
        payload["selection_mode"],
        int(payload["selection_seed"]),
        float(payload["epsilon"]),
    )


class SVSpectralDiffusionAugmentedDataset(SVSignedGINDataset):
    """Join cached HKS/eigenbases to the unchanged SVG hard-graph cache."""

    def __init__(
        self,
        manifest_path: Path,
        scaler_path: Path,
        spectral_manifest_path: Path,
        spectral_scaler_path: Path,
        max_samples=None,
        include_windows: bool = True,
    ) -> None:
        super().__init__(
            manifest_path,
            scaler_path,
            max_samples=max_samples,
            include_windows=include_windows,
        )
        spectral_path = Path(spectral_manifest_path).resolve()
        spectral_manifest, records = read_sv_spectral_diffusion_manifest(
            spectral_path
        )
        spectral_scaler = load_sv_spectral_diffusion_standardizer(
            spectral_scaler_path
        )
        base_provenance = (
            self.manifest["protocol_sha256"],
            self.manifest["selector_checkpoint_sha256"],
            self.manifest["selection_mode"],
            int(self.manifest["selection_seed"]),
        )
        spectral_provenance = (
            spectral_manifest["protocol_sha256"],
            spectral_manifest["selector_checkpoint_sha256"],
            spectral_manifest["selection_mode"],
            int(spectral_manifest["selection_seed"]),
        )
        scaler_provenance = (
            spectral_scaler.protocol_sha256,
            spectral_scaler.selector_checkpoint_sha256,
            spectral_scaler.selection_mode,
            int(spectral_scaler.selection_seed),
        )
        if (
            spectral_manifest["split"] != self.split
            or spectral_manifest["source_manifest_sha256"]
            != file_sha256(self.manifest_path)
            or base_provenance != spectral_provenance
            or base_provenance != scaler_provenance
        ):
            raise ValueError("spectral-diffusion dataset provenance mismatch")
        if self.split == "train" and (
            spectral_scaler.train_manifest_sha256
            != file_sha256(spectral_path)
            or spectral_scaler.source_manifest_sha256
            != file_sha256(self.manifest_path)
        ):
            raise ValueError("spectral-diffusion train scaler mismatch")
        base_rows = {
            row["sample_key"]: row
            for row in self.manifest["records"]
        }
        spectral_rows = {
            row["sample_key"]: row
            for row in spectral_manifest["records"]
        }
        expected_keys = set(self.sample_keys)
        if (
            set(base_rows) != set(spectral_rows)
            or not expected_keys.issubset(set(spectral_rows))
        ):
            raise ValueError("spectral-diffusion manifest sample set mismatch")
        for sample_key in expected_keys:
            if spectral_rows[sample_key]["source_feature_sha256"] != (
                base_rows[sample_key]["feature_sha256"]
            ):
                raise ValueError("spectral-diffusion source artifact mismatch")
        by_key = {record.sample_key: record for record in records}
        augmented = []
        for sample in self.samples:
            record = by_key.get(sample.sample_key)
            if record is None or int(record.label) != int(sample.label):
                raise ValueError("spectral-diffusion sample join mismatch")
            windows = []
            for window in sample.windows:
                position = int(window.time_position)
                sidecar = record.windows[position]
                if sidecar is None or sidecar.hks.shape[0] != window.node_features.shape[0]:
                    raise ValueError("spectral-diffusion window join mismatch")
                delta = None
                if position < len(record.windows) - 1:
                    following = record.windows[position + 1]
                    if following is not None:
                        delta = spectral_scaler.standardize_delta(
                            following.spectral_quantiles
                            - sidecar.spectral_quantiles
                        ).to(torch.float32)
                windows.append(
                    replace(
                        window,
                        hks=spectral_scaler.standardize_hks(
                            sidecar.hks
                        ).to(torch.float32),
                        diffusion_eigenvalues=sidecar.eigenvalues,
                        diffusion_eigenvectors=sidecar.eigenvectors,
                        spectral_delta_to_next=delta,
                    )
                )
            augmented.append(replace(sample, windows=tuple(windows)))
        self.samples = augmented
        self.spectral_manifest = spectral_manifest
        self.spectral_scaler = spectral_scaler
        self.spectral_manifest_path = spectral_path
        self.spectral_scaler_path = Path(spectral_scaler_path).resolve()
