"""Immutable frozen-representation artifacts for promoted F2 fusion."""

from __future__ import absolute_import, division, print_function

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from keysubgraph.data.data_split import file_sha256


SVG_SHORT_TERM_REPRESENTATION_F2_MANIFEST_SCHEMA_VERSION = 1
SVG_SHORT_TERM_REPRESENTATION_F2_FEATURE_SCHEMA_VERSION = 1


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
class SVGShortTermRepresentationF2Item:
    sample_key: str
    site: str
    subject_id: str
    label: int
    g2_anchor_logit: torch.Tensor
    g2_representation: torch.Tensor
    short_term_representation: torch.Tensor


@dataclass(frozen=True)
class SVGShortTermRepresentationF2Batch:
    sample_keys: Tuple[str, ...]
    sites: Tuple[str, ...]
    subject_ids: Tuple[str, ...]
    labels: torch.Tensor
    g2_anchor_logits: torch.Tensor
    g2_representations: torch.Tensor
    short_term_representations: torch.Tensor

    def to(self, device, non_blocking: bool = False):
        return SVGShortTermRepresentationF2Batch(
            sample_keys=self.sample_keys,
            sites=self.sites,
            subject_ids=self.subject_ids,
            labels=self.labels.to(device, non_blocking=non_blocking),
            g2_anchor_logits=self.g2_anchor_logits.to(
                device, non_blocking=non_blocking
            ),
            g2_representations=self.g2_representations.to(
                device, non_blocking=non_blocking
            ),
            short_term_representations=self.short_term_representations.to(
                device, non_blocking=non_blocking
            ),
        )


def collate_svg_short_term_representation_f2(items):
    if not items:
        raise ValueError("cannot collate an empty representation F2 batch")
    return SVGShortTermRepresentationF2Batch(
        sample_keys=tuple(item.sample_key for item in items),
        sites=tuple(item.site for item in items),
        subject_ids=tuple(item.subject_id for item in items),
        labels=torch.tensor([item.label for item in items], dtype=torch.long),
        g2_anchor_logits=torch.stack(
            [item.g2_anchor_logit for item in items], dim=0
        ),
        g2_representations=torch.stack(
            [item.g2_representation for item in items], dim=0
        ),
        short_term_representations=torch.stack(
            [item.short_term_representation for item in items], dim=0
        ),
    )


class SVGShortTermRepresentationF2Dataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("artifact_type")
            != "svg_short_term_representation_f2_manifest"
            or int(manifest.get("schema_version", 0))
            != SVG_SHORT_TERM_REPRESENTATION_F2_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("not a representation F2 manifest")
        feature_path = (
            self.manifest_path.parent / manifest["feature_file"]
        ).resolve()
        if not feature_path.is_file() or file_sha256(feature_path) != manifest.get(
            "feature_sha256"
        ):
            raise ValueError("representation F2 feature hash mismatch")
        payload = _trusted_load(feature_path)
        if (
            payload.get("artifact_type")
            != "svg_short_term_representation_f2_features"
            or int(payload.get("schema_version", 0))
            != SVG_SHORT_TERM_REPRESENTATION_F2_FEATURE_SCHEMA_VERSION
            or payload.get("split") != manifest.get("split")
        ):
            raise ValueError("representation F2 feature schema mismatch")
        keys = tuple(str(value) for value in payload["sample_keys"])
        sites = tuple(str(value) for value in payload["sites"])
        subjects = tuple(str(value) for value in payload["subject_ids"])
        labels = payload["labels"].to(dtype=torch.long)
        anchor = payload["g2_anchor_logits"].to(dtype=torch.float32)
        g2 = payload["g2_representations"].to(dtype=torch.float32)
        short = payload["short_term_representations"].to(dtype=torch.float32)
        count = len(keys)
        valid = (
            count > 0
            and len(set(keys)) == count
            and len(sites) == count
            and len(subjects) == count
            and tuple(labels.shape) == (count,)
            and tuple(anchor.shape) == (count,)
            and g2.ndim == 2
            and short.ndim == 2
            and int(g2.shape[0]) == count
            and int(short.shape[0]) == count
            and bool(torch.isfinite(anchor).all())
            and bool(torch.isfinite(g2).all())
            and bool(torch.isfinite(short).all())
            and set(int(value) for value in labels.tolist()).issubset({0, 1})
        )
        if not valid:
            raise ValueError("representation F2 feature tensors are invalid")
        if int(manifest.get("sample_count", -1)) != count:
            raise ValueError("representation F2 sample count mismatch")
        if int(manifest.get("g2_representation_dim", -1)) != int(g2.shape[1]):
            raise ValueError("representation F2 G2 dimension mismatch")
        if int(manifest.get("short_term_representation_dim", -1)) != int(
            short.shape[1]
        ):
            raise ValueError("representation F2 short-term dimension mismatch")
        self.manifest = manifest
        self.feature_path = feature_path
        self._keys = keys
        self._sites = sites
        self._subjects = subjects
        self._labels = labels
        self._anchor = anchor
        self._g2 = g2
        self._short = short

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
    def g2_representation_dim(self) -> int:
        return int(self._g2.shape[1])

    @property
    def short_term_representation_dim(self) -> int:
        return int(self._short.shape[1])

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int):
        return SVGShortTermRepresentationF2Item(
            sample_key=self._keys[index],
            site=self._sites[index],
            subject_id=self._subjects[index],
            label=int(self._labels[index]),
            g2_anchor_logit=self._anchor[index],
            g2_representation=self._g2[index],
            short_term_representation=self._short[index],
        )

