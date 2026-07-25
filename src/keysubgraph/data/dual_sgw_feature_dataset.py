"""Feature-only datasets for cached exact Dual-STSE SGW records."""

from __future__ import absolute_import, division, print_function

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dual_sgw_manifest import read_dual_sgw_manifest
from .dual_sgw_scaler import load_dual_sgw_standardizer


class DualSGWFeatureDataset(Dataset):
    """Load immutable 34-D records without reopening the original graphs."""

    def __init__(self, manifest_path: Path, scaler_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        self.manifest, records, _ = read_dual_sgw_manifest(
            self.manifest_path
        )
        self.scaler = load_dual_sgw_standardizer(self.scaler_path)
        if (
            self.scaler.protocol_sha256
            != self.manifest["protocol_sha256"]
            or self.scaler.selector_checkpoint_sha256
            != self.manifest["selector_checkpoint_sha256"]
            or self.scaler.selection_mode
            != self.manifest["selection_mode"]
            or int(self.scaler.selection_seed)
            != int(self.manifest["selection_seed"])
        ):
            raise ValueError(
                "dual SGW feature dataset has mismatched scaler provenance"
            )
        self.samples = tuple(
            {
                "sample_key": record.sample_key,
                "label": int(record.label),
                "features": self.scaler(
                    record.representation.detach().to(torch.float32)
                )
                .detach()
                .clone(),
            }
            for record in records
        )
        if not self.samples:
            raise ValueError("dual SGW feature dataset cannot be empty")
        if any(
            tuple(sample["features"].shape) != (34,)
            for sample in self.samples
        ):
            raise ValueError("dual SGW feature dataset requires 34-D values")
        keys = [sample["sample_key"] for sample in self.samples]
        if len(set(keys)) != len(keys):
            raise ValueError("dual SGW feature dataset contains duplicate keys")

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def labels(self):
        return tuple(int(sample["label"]) for sample in self.samples)

    @property
    def sample_keys(self):
        return tuple(str(sample["sample_key"]) for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_key": sample["sample_key"],
            "label": sample["label"],
            "features": sample["features"].clone(),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dual_sgw_feature_loader(
    dataset: DualSGWFeatureDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid dual SGW feature loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("validation and test feature loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )

