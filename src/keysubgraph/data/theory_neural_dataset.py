"""Variable-length list batching for Stage-1 theory-guided models."""

from __future__ import absolute_import, division, print_function

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.theory_neural_manifest import read_theory_neural_manifest
from keysubgraph.data.theory_neural_scaler import load_theory_neural_scaler
from keysubgraph.models.theory_guided_neural import (
    TheoryNeuralBatch,
    TheoryNeuralSampleInput,
    TheoryNeuralWindowInput,
)


class TheoryNeuralDataset(Dataset):
    def __init__(self, project_root, manifest_path, scaler_path, max_samples=None):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        self.manifest, records = read_theory_neural_manifest(
            self.manifest_path, self.project_root
        )
        self.scaler = load_theory_neural_scaler(self.scaler_path)
        checks = (
            self.scaler.protocol_sha256 == self.manifest["protocol_sha256"],
            self.scaler.selector_checkpoint_sha256
            == self.manifest["selector_checkpoint_sha256"],
            self.scaler.feature_schema_sha256
            == self.manifest["feature_schema_sha256"],
        )
        if not all(checks):
            raise ValueError("Stage-1 dataset/scaler provenance mismatch")
        if self.manifest["split"] == "train" and (
            self.scaler.train_manifest_sha256 != file_sha256(self.manifest_path)
        ):
            raise ValueError("Stage-1 train scaler manifest mismatch")
        selected = records if max_samples is None else records[: int(max_samples)]
        if not selected:
            raise ValueError("Stage-1 dataset cannot be empty")
        self.samples = []
        self.sites = []
        self.subject_ids = []
        for record in selected:
            windows = tuple(
                (
                    TheoryNeuralWindowInput(
                        node_features=self.scaler.standardize_nodes(
                            window.node_features.to(torch.float32)
                        ),
                        adjacency=window.adjacency.to(torch.float32),
                        edge_features=window.edge_features.to(torch.float32),
                        spectral_quantiles=self.scaler.standardize_quantiles(
                            window.spectral_quantiles.to(torch.float32)
                        ),
                    )
                    if window is not None
                    else None
                )
                for window in record.windows
            )
            self.samples.append(
                TheoryNeuralSampleInput(
                    sample_key=record.sample_key,
                    label=int(record.label),
                    windows=windows,
                    window_mask=record.window_mask.clone(),
                    transition_targets=self.scaler.standardize_transitions(
                        record.transition_features.to(torch.float32)
                    ),
                    transition_mask=record.transition_mask.clone(),
                )
            )
            self.sites.append(record.site)
            self.subject_ids.append(record.subject_id)

    @property
    def split(self):
        return self.manifest["split"]

    @property
    def labels(self):
        return tuple(sample.label for sample in self.samples)

    @property
    def sample_keys(self):
        return tuple(sample.sample_key for sample in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_theory_neural(samples):
    if not samples:
        raise ValueError("cannot collate an empty Stage-1 batch")
    return TheoryNeuralBatch(tuple(samples))


def _seed_worker(worker_id):
    del worker_id
    seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(seed)
    np.random.seed(seed)


def create_theory_neural_loader(
    dataset, batch_size, seed, shuffle, num_workers=0, pin_memory=False
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid Stage-1 loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("Stage-1 evaluation loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_theory_neural,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
