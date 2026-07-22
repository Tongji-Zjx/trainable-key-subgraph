"""Strictly paired Stage-C hard-student artifacts and teacher targets."""

from __future__ import absolute_import, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.features.hard_graph_cache import (
    HardGraphSampleCache,
    load_hard_graph_cache,
)
from keysubgraph.features.tg_standardizer import TGTheoryFeatureStandardizer
from keysubgraph.theory.tg_features import (
    TGSGWFeatureArtifact,
    load_tg_sgw_feature_artifact,
)


TG_TEACHER_TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TGTeacherTarget:
    sample_key: str
    sample_id: str
    label: int
    split: str
    logits: torch.Tensor
    representation: torch.Tensor
    data_protocol_sha256: str
    teacher_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if tuple(self.logits.shape) != (2,):
            raise ValueError("teacher target logits must have shape [2]")
        if tuple(self.representation.shape) != (192,):
            raise ValueError("teacher target representation must have shape [192]")
        if self.label not in (0, 1):
            raise ValueError("teacher target label must be binary")
        if not bool(torch.isfinite(self.logits).all()):
            raise ValueError("teacher target logits contain non-finite values")
        if not bool(torch.isfinite(self.representation).all()):
            raise ValueError("teacher target representation contains non-finite values")


@dataclass(frozen=True)
class TGHardStudentSample:
    hard_cache: HardGraphSampleCache
    theory_artifact: TGSGWFeatureArtifact
    teacher_target: TGTeacherTarget
    standardized_theory_features: torch.Tensor

    @property
    def sample_key(self) -> str:
        return self.hard_cache.sample_key

    @property
    def label(self) -> int:
        return self.hard_cache.label

    @property
    def split(self) -> str:
        return self.hard_cache.split


def save_tg_teacher_target(
    target: TGTeacherTarget, path: Path, overwrite: bool = False
) -> Path:
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("TG teacher target already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": TG_TEACHER_TARGET_SCHEMA_VERSION,
            "artifact_type": "tg_sgw_teacher_target",
            "target": target,
        },
        str(temporary),
    )
    os.replace(str(temporary), str(path))
    return path


def load_tg_teacher_target(path: Path) -> TGTeacherTarget:
    try:
        payload = torch.load(
            str(Path(path).resolve()), map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    if payload.get("schema_version") != TG_TEACHER_TARGET_SCHEMA_VERSION:
        raise ValueError("unsupported TG teacher-target schema")
    if payload.get("artifact_type") != "tg_sgw_teacher_target":
        raise ValueError("unexpected TG teacher-target artifact")
    target = payload.get("target")
    if not isinstance(target, TGTeacherTarget):
        raise ValueError("invalid TG teacher-target payload")
    return target


def _load_unique(directory: Path, loader, artifact_name: str) -> Dict[str, object]:
    paths = sorted(Path(directory).resolve().glob("*.pt"))
    if not paths:
        raise ValueError("{} directory contains no .pt files".format(artifact_name))
    values = {}
    for path in paths:
        value = loader(path)
        key = str(value.sample_key)
        if key in values:
            raise ValueError("duplicate {} sample_key: {}".format(artifact_name, key))
        values[key] = value
    return values


def _validate_pair(
    hard: HardGraphSampleCache,
    theory: TGSGWFeatureArtifact,
    teacher: TGTeacherTarget,
    scaler: TGTheoryFeatureStandardizer,
    expected_split: Optional[str],
) -> None:
    values = (hard, theory, teacher)
    for name in ("sample_key", "sample_id", "label", "split"):
        if len(set(getattr(item, name) for item in values)) != 1:
            raise ValueError("Stage-C paired artifact {} mismatch".format(name))
    if expected_split is not None and hard.split != expected_split:
        raise ValueError("Stage-C sample belongs to the wrong split")
    if hard.eligible_for_stage_c != theory.eligible_for_stage_c:
        raise ValueError("hard/theory Stage-C eligibility mismatch")
    protocol_hashes = {item.data_protocol_sha256 for item in values}
    teacher_hashes = {item.teacher_checkpoint_sha256 for item in values}
    if len(protocol_hashes) != 1 or len(teacher_hashes) != 1:
        raise ValueError("Stage-C paired artifact hash mismatch")
    if scaler.data_protocol_sha256 != hard.data_protocol_sha256:
        raise ValueError("TG theory scaler protocol mismatch")
    if scaler.teacher_checkpoint_sha256 != hard.teacher_checkpoint_sha256:
        raise ValueError("TG theory scaler teacher mismatch")
    if tuple(theory.features.h_classification.shape) != (34,):
        raise ValueError("TG theory artifact must contain 34-D classification features")


class TGHardStudentDataset(Dataset):
    """Load exactly aligned hard graphs, SGW features and frozen teacher targets."""

    def __init__(
        self,
        hard_cache_dir: Path,
        theory_feature_dir: Path,
        teacher_target_dir: Path,
        theory_standardizer: TGTheoryFeatureStandardizer,
        expected_split: Optional[str] = None,
    ) -> None:
        hard = _load_unique(hard_cache_dir, load_hard_graph_cache, "hard cache")
        theory = _load_unique(
            theory_feature_dir, load_tg_sgw_feature_artifact, "theory feature"
        )
        teacher = _load_unique(
            teacher_target_dir, load_tg_teacher_target, "teacher target"
        )
        sets = (set(hard), set(theory), set(teacher))
        if sets[0] != sets[1] or sets[0] != sets[2]:
            raise ValueError("Stage-C artifact directories contain different sample sets")
        self.theory_standardizer = theory_standardizer
        samples = []
        for key in sorted(sets[0]):
            hard_item = hard[key]
            theory_item = theory[key]
            teacher_item = teacher[key]
            _validate_pair(
                hard_item,
                theory_item,
                teacher_item,
                theory_standardizer,
                expected_split,
            )
            if not hard_item.eligible_for_stage_c:
                continue
            standardized = theory_standardizer.transform(
                theory_item.features.h_classification
            ).to(dtype=torch.float32)
            samples.append(
                TGHardStudentSample(
                    hard_item, theory_item, teacher_item, standardized
                )
            )
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("Stage-C dataset contains no eligible samples")
        self.split = expected_split or self.samples[0].split
        if len(set(item.split for item in self.samples)) != 1:
            raise ValueError("a Stage-C dataset cannot mix data splits")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TGHardStudentSample:
        return self.samples[index]

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(item.label for item in self.samples)


def tg_hard_student_collate(
    samples: Sequence[TGHardStudentSample],
) -> Tuple[TGHardStudentSample, ...]:
    if not samples:
        raise ValueError("cannot collate an empty Stage-C batch")
    return tuple(samples)


def create_tg_hard_student_loader(
    dataset: TGHardStudentDataset,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid Stage-C DataLoader configuration")
    if shuffle is None:
        shuffle = dataset.split == "train"
    if shuffle and dataset.split != "train":
        raise ValueError("Stage-C validation/test loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tg_hard_student_collate,
        generator=generator,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
