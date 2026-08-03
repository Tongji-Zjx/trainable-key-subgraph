"""Immutable frozen-G2 feature artifacts for the SafeQ residual stage."""

from __future__ import absolute_import, division, print_function

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from keysubgraph.data.data_split import file_sha256


G2_SAFEQ_MANIFEST_SCHEMA_VERSION = 1
G2_SAFEQ_FEATURE_SCHEMA_VERSION = 1


def _trusted_load(path: Path):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location="cpu")


@dataclass(frozen=True)
class G2SafeQItem:
    sample_key: str
    site: str
    subject_id: str
    label: int
    base_logit: torch.Tensor
    static_logit: torch.Tensor
    transition_summary: torch.Tensor
    has_valid_transition: torch.Tensor


@dataclass(frozen=True)
class G2SafeQBatch:
    sample_keys: Tuple[str, ...]
    sites: Tuple[str, ...]
    subject_ids: Tuple[str, ...]
    labels: torch.Tensor
    base_logits: torch.Tensor
    static_logits: torch.Tensor
    transition_summaries: torch.Tensor
    has_valid_transition: torch.Tensor

    def to(self, device, non_blocking: bool = False):
        return G2SafeQBatch(
            sample_keys=self.sample_keys,
            sites=self.sites,
            subject_ids=self.subject_ids,
            labels=self.labels.to(device, non_blocking=non_blocking),
            base_logits=self.base_logits.to(device, non_blocking=non_blocking),
            static_logits=self.static_logits.to(
                device, non_blocking=non_blocking
            ),
            transition_summaries=self.transition_summaries.to(
                device, non_blocking=non_blocking
            ),
            has_valid_transition=self.has_valid_transition.to(
                device, non_blocking=non_blocking
            ),
        )


def collate_g2_safeq(items):
    if not items:
        raise ValueError("cannot collate an empty SafeQ batch")
    return G2SafeQBatch(
        sample_keys=tuple(item.sample_key for item in items),
        sites=tuple(item.site for item in items),
        subject_ids=tuple(item.subject_id for item in items),
        labels=torch.tensor([item.label for item in items], dtype=torch.long),
        base_logits=torch.stack([item.base_logit for item in items]),
        static_logits=torch.stack([item.static_logit for item in items]),
        transition_summaries=torch.stack(
            [item.transition_summary for item in items]
        ),
        has_valid_transition=torch.stack(
            [item.has_valid_transition for item in items]
        ).to(dtype=torch.bool),
    )


class G2SafeQDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("artifact_type") != "g2_safeq_manifest"
            or int(manifest.get("schema_version", 0))
            != G2_SAFEQ_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("not a SafeQ manifest")
        feature_path = (
            self.manifest_path.parent / manifest["feature_file"]
        ).resolve()
        if (
            not feature_path.is_file()
            or file_sha256(feature_path) != manifest.get("feature_sha256")
        ):
            raise ValueError("SafeQ feature hash mismatch")
        payload = _trusted_load(feature_path)
        if (
            payload.get("artifact_type") != "g2_safeq_features"
            or int(payload.get("schema_version", 0))
            != G2_SAFEQ_FEATURE_SCHEMA_VERSION
            or payload.get("split") != manifest.get("split")
        ):
            raise ValueError("SafeQ feature schema mismatch")

        keys = tuple(str(value) for value in payload["sample_keys"])
        sites = tuple(str(value) for value in payload["sites"])
        subjects = tuple(str(value) for value in payload["subject_ids"])
        labels = payload["labels"].to(dtype=torch.long)
        base = payload["base_logits"].to(dtype=torch.float32)
        static = payload["static_logits"].to(dtype=torch.float32)
        summary = payload["transition_summaries"].to(dtype=torch.float32)
        valid_transition = payload["has_valid_transition"].to(
            dtype=torch.bool
        )
        count = len(keys)
        summary_dim = int(manifest.get("summary_dim", -1))
        valid = (
            count > 0
            and len(set(keys)) == count
            and len(sites) == count
            and len(subjects) == count
            and tuple(labels.shape) == (count,)
            and tuple(base.shape) == (count,)
            and tuple(static.shape) == (count,)
            and tuple(summary.shape) == (count, summary_dim)
            and tuple(valid_transition.shape) == (count,)
            and bool(torch.isfinite(base).all())
            and bool(torch.isfinite(static).all())
            and bool(torch.isfinite(summary).all())
            and set(int(value) for value in labels.tolist()).issubset({0, 1})
        )
        if not valid:
            raise ValueError("SafeQ feature tensors are invalid")
        if int(manifest.get("sample_count", -1)) != count:
            raise ValueError("SafeQ sample count mismatch")
        if bool((summary[~valid_transition] != 0.0).any()):
            raise ValueError("SafeQ invalid-transition summaries must be zero")

        self.manifest = manifest
        self.feature_path = feature_path
        self._keys = keys
        self._sites = sites
        self._subjects = subjects
        self._labels = labels
        self._base = base
        self._static = static
        self._summary = summary
        self._valid_transition = valid_transition

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def labels(self):
        return tuple(int(value) for value in self._labels.tolist())

    @property
    def sample_keys(self):
        return self._keys

    @property
    def sites(self):
        return self._sites

    @property
    def summary_dim(self) -> int:
        return int(self._summary.shape[1])

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> G2SafeQItem:
        return G2SafeQItem(
            sample_key=self._keys[index],
            site=self._sites[index],
            subject_id=self._subjects[index],
            label=int(self._labels[index]),
            base_logit=self._base[index],
            static_logit=self._static[index],
            transition_summary=self._summary[index],
            has_valid_transition=self._valid_transition[index],
        )
