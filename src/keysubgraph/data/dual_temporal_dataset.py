"""Variable-length cached datasets for D3-B temporal residual models."""

from __future__ import absolute_import, division, print_function

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.data_split import file_sha256
from .dual_temporal_manifest import read_dual_temporal_manifest
from .dual_temporal_scaler import load_dual_temporal_standardizer


@dataclass(frozen=True)
class DualTemporalBatch:
    sample_keys: tuple
    labels: torch.Tensor
    transition_values: torch.Tensor
    time_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    base_logits: torch.Tensor

    def __len__(self):
        return len(self.sample_keys)

    def to(self, device):
        return DualTemporalBatch(
            sample_keys=self.sample_keys,
            labels=self.labels.to(device),
            transition_values=self.transition_values.to(device),
            time_mask=self.time_mask.to(device),
            sequence_lengths=self.sequence_lengths.to(device),
            base_logits=self.base_logits.to(device),
        )


class DualTemporalDataset(Dataset):
    def __init__(self, manifest_path: Path, scaler_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        self.manifest, records = read_dual_temporal_manifest(
            self.manifest_path
        )
        self.scaler = load_dual_temporal_standardizer(self.scaler_path)
        checks = (
            (
                self.scaler.protocol_sha256,
                self.manifest["protocol_sha256"],
            ),
            (
                self.scaler.selector_checkpoint_sha256,
                self.manifest["selector_checkpoint_sha256"],
            ),
            (
                self.scaler.exact_head_checkpoint_sha256,
                self.manifest["exact_head_checkpoint_sha256"],
            ),
            (
                self.scaler.sgw_scaler_sha256,
                self.manifest["sgw_scaler_sha256"],
            ),
            (
                self.scaler.exact_manifest_sha256,
                self.manifest["exact_manifest_sha256"],
            ),
            (
                self.scaler.selection_mode,
                self.manifest["selection_mode"],
            ),
            (
                int(self.scaler.selection_seed),
                int(self.manifest["selection_seed"]),
            ),
        )
        if any(left != right for left, right in checks):
            raise ValueError("temporal dataset/scaler provenance mismatch")
        if self.manifest["split"] == "train" and (
            self.scaler.train_manifest_sha256
            != file_sha256(self.manifest_path)
        ):
            raise ValueError("temporal train scaler manifest mismatch")
        self.samples = []
        for record in records:
            compact = record.transition_values[
                record.transition_mask
            ].to(torch.float32)
            standardized = self.scaler(compact).detach().clone()
            self.samples.append(
                {
                    "sample_key": record.sample_key,
                    "label": int(record.label),
                    "transition_values": standardized,
                    "base_logits": record.base_logits.to(
                        torch.float32
                    ).detach().clone(),
                    "window_count": int(record.window_count),
                    "original_transition_count": int(
                        record.transition_mask.numel()
                    ),
                }
            )
        if not self.samples:
            raise ValueError("dual temporal dataset cannot be empty")

    @property
    def split(self):
        return str(self.manifest["split"])

    @property
    def labels(self):
        return tuple(sample["label"] for sample in self.samples)

    @property
    def sample_keys(self):
        return tuple(sample["sample_key"] for sample in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }


def collate_dual_temporal(samples):
    if not samples:
        raise ValueError("cannot collate an empty temporal batch")
    lengths = torch.tensor(
        [sample["transition_values"].shape[0] for sample in samples],
        dtype=torch.long,
    )
    maximum = max(1, int(lengths.max().item()))
    values = torch.zeros(
        (len(samples), maximum, 16), dtype=torch.float32
    )
    mask = torch.zeros(
        (len(samples), maximum), dtype=torch.bool
    )
    for index, sample in enumerate(samples):
        length = int(lengths[index].item())
        if length:
            values[index, :length] = sample["transition_values"]
            mask[index, :length] = True
    return DualTemporalBatch(
        sample_keys=tuple(sample["sample_key"] for sample in samples),
        labels=torch.tensor(
            [sample["label"] for sample in samples], dtype=torch.long
        ),
        transition_values=values,
        time_mask=mask,
        sequence_lengths=lengths,
        base_logits=torch.stack(
            [sample["base_logits"] for sample in samples], dim=0
        ),
    )


def shuffle_dual_temporal_batch(
    batch: DualTemporalBatch, seed: int
) -> DualTemporalBatch:
    """Deterministically permute valid transitions within each sample."""
    values = batch.transition_values.clone()
    for index, (key, length) in enumerate(
        zip(batch.sample_keys, batch.sequence_lengths.tolist())
    ):
        length = int(length)
        if length < 2:
            continue
        digest = hashlib.sha256(
            "{}|{}".format(int(seed), key).encode("utf-8")
        ).digest()
        generator = torch.Generator()
        generator.manual_seed(
            int.from_bytes(digest[:8], byteorder="little") % (2 ** 63)
        )
        order = torch.randperm(length, generator=generator)
        values[index, :length] = values[index, :length][order]
    return DualTemporalBatch(
        sample_keys=batch.sample_keys,
        labels=batch.labels,
        transition_values=values,
        time_mask=batch.time_mask,
        sequence_lengths=batch.sequence_lengths,
        base_logits=batch.base_logits,
    )


def _seed_worker(worker_id):
    del worker_id
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dual_temporal_loader(
    dataset: DualTemporalDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid temporal loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("validation/test temporal loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_dual_temporal,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
