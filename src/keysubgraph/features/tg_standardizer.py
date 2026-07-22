"""Training-fold-only standardization for 34-D TG-SGW theory features."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class TGTheoryFeatureStandardizer:
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    fit_split: str = "train"
    standard_deviation_floor: float = 1.0e-6
    data_protocol_sha256: Optional[str] = None
    teacher_checkpoint_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.mean) != 34 or len(self.scale) != 34:
            raise ValueError("TG theory scaler requires 34 coordinates")
        if self.fit_split != "train":
            raise ValueError("TG theory scaler must be fitted on train only")
        if self.standard_deviation_floor <= 0.0:
            raise ValueError("TG theory standard deviation floor must be positive")
        if any(value < self.standard_deviation_floor for value in self.scale):
            raise ValueError("TG theory scaler contains a sub-floor scale")

    @classmethod
    def fit(
        cls,
        features: Sequence[torch.Tensor],
        fit_split: str = "train",
        standard_deviation_floor: float = 1.0e-6,
        data_protocol_sha256: Optional[str] = None,
        teacher_checkpoint_sha256: Optional[str] = None,
    ) -> "TGTheoryFeatureStandardizer":
        if fit_split != "train":
            raise ValueError("TG theory scaler cannot fit validation or test")
        if not features:
            raise ValueError("TG theory scaler requires training features")
        matrix = torch.stack(
            [item.detach().to(dtype=torch.float64, device="cpu") for item in features]
        )
        if matrix.ndim != 2 or matrix.shape[1] != 34:
            raise ValueError("TG theory training features must have shape [B, 34]")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("TG theory features contain non-finite values")
        mean = matrix.mean(dim=0)
        scale = matrix.std(dim=0, unbiased=False).clamp_min(
            float(standard_deviation_floor)
        )
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            fit_split=fit_split,
            standard_deviation_floor=float(standard_deviation_floor),
            data_protocol_sha256=data_protocol_sha256,
            teacher_checkpoint_sha256=teacher_checkpoint_sha256,
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 34:
            raise ValueError("TG theory scaler expects a 34-D final axis")
        mean = features.new_tensor(self.mean)
        scale = features.new_tensor(self.scale)
        return (features - mean) / scale

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        payload["artifact_type"] = "tg_sgw_theory_feature_scaler"
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TGTheoryFeatureStandardizer":
        if payload.get("schema_version") != 1 or payload.get("artifact_type") != "tg_sgw_theory_feature_scaler":
            raise ValueError("unsupported TG theory scaler artifact")
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            fit_split=str(payload["fit_split"]),
            standard_deviation_floor=float(payload["standard_deviation_floor"]),
            data_protocol_sha256=payload.get("data_protocol_sha256"),
            teacher_checkpoint_sha256=payload.get("teacher_checkpoint_sha256"),
        )

    def save(self, path: Path, overwrite: bool = False) -> Path:
        path = Path(path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError("TG theory scaler already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
        return path

    @classmethod
    def load(cls, path: Path) -> "TGTheoryFeatureStandardizer":
        with Path(path).resolve().open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
