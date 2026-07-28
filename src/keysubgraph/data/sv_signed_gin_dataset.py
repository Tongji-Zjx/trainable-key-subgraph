"""List-batched frozen hard-graph datasets for SV Signed-GIN."""

from __future__ import absolute_import, division, print_function

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.data_split import file_sha256
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINBatch,
    SVSignedGINSampleInput,
    SVSignedGINWindowInput,
)
from .sv_signed_gin_manifest import read_sv_signed_gin_manifest
from .sv_signed_gin_scaler import (
    load_sv_signed_gin_standardizers,
)


class SVSignedGINDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        scaler_path: Path,
        max_samples=None,
    ) -> None:
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("SV dataset max_samples must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        self.manifest, records = read_sv_signed_gin_manifest(
            self.manifest_path
        )
        self.scaler = load_sv_signed_gin_standardizers(
            self.scaler_path
        )
        checks = (
            self.scaler.protocol_sha256
            == self.manifest["protocol_sha256"],
            self.scaler.selector_checkpoint_sha256
            == self.manifest["selector_checkpoint_sha256"],
            self.scaler.selection_mode
            == self.manifest["selection_mode"],
            int(self.scaler.selection_seed)
            == int(self.manifest["selection_seed"]),
        )
        if not all(checks):
            raise ValueError("SV dataset/scaler provenance mismatch")
        if self.manifest["split"] == "train" and (
            self.scaler.train_manifest_sha256
            != file_sha256(self.manifest_path)
        ):
            raise ValueError("SV train scaler manifest mismatch")

        self.samples = []
        self.sites = []
        self.subject_ids = []
        selected_records = (
            records[: int(max_samples)]
            if max_samples is not None
            else records
        )
        for record in selected_records:
            windows = tuple(
                SVSignedGINWindowInput(
                    node_features=self.scaler.standardize_nodes(
                        window.node_features.to(torch.float32)
                    ).detach().clone(),
                    adjacency=window.adjacency.to(
                        torch.float32
                    ).detach().clone(),
                )
                for window in record.windows
                if window is not None
            )
            self.samples.append(
                SVSignedGINSampleInput(
                    sample_key=record.sample_key,
                    label=int(record.label),
                    windows=windows,
                    static_features=self.scaler.standardize_static(
                        record.static_features.to(torch.float32)
                    ).detach().clone(),
                    variation=self.scaler.standardize_variation(
                        record.variation.to(torch.float32)
                    ).detach().clone(),
                )
            )
            self.sites.append(record.site)
            self.subject_ids.append(record.subject_id)
        if not self.samples:
            raise ValueError("SV Signed-GIN dataset cannot be empty")

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def labels(self):
        return tuple(sample.label for sample in self.samples)

    @property
    def sample_keys(self):
        return tuple(sample.sample_key for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SVSignedGINSampleInput:
        return self.samples[index]


def collate_sv_signed_gin(samples) -> SVSignedGINBatch:
    if not samples:
        raise ValueError("cannot collate an empty SV Signed-GIN batch")
    return SVSignedGINBatch(tuple(samples))


def _seed_worker(worker_id):
    del worker_id
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_sv_signed_gin_loader(
    dataset: SVSignedGINDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid SV Signed-GIN loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("SV validation/test loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_sv_signed_gin,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
