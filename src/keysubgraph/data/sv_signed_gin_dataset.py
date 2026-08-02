"""List-batched frozen hard-graph datasets for SV Signed-GIN."""

from __future__ import absolute_import, division, print_function

import random
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

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
        include_windows: bool = True,
    ) -> None:
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("SV dataset max_samples must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        self.scaler_path = Path(scaler_path).resolve()
        self.include_windows = bool(include_windows)
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
            windows = (
                tuple(
                    SVSignedGINWindowInput(
                        node_features=self.scaler.standardize_nodes(
                            window.node_features.to(torch.float32)
                        ).detach().clone(),
                        adjacency=window.adjacency.to(
                            torch.float32
                        ).detach().clone(),
                        time_position=int(position),
                        communities=(
                            window.communities.to(
                                dtype=torch.long
                            ).detach().clone()
                            if getattr(window, "communities", None)
                            is not None
                            else None
                        ),
                    )
                    for position, window in enumerate(record.windows)
                    if window is not None
                )
                if self.include_windows
                else ()
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


class SVMultiBudgetDataset(Dataset):
    """Three aligned hard-budget views with independent train scalers."""

    expected_budgets = ((0.35, 0.20), (0.50, 0.30), (0.65, 0.40))

    def __init__(
        self,
        manifest_paths,
        scaler_paths,
        max_samples=None,
        include_windows: bool = True,
    ) -> None:
        if len(manifest_paths) != 3 or len(scaler_paths) != 3:
            raise ValueError("E1 requires exactly three budget views")
        datasets = [
            SVSignedGINDataset(
                manifest,
                scaler,
                max_samples=max_samples,
                include_windows=include_windows,
            )
            for manifest, scaler in zip(manifest_paths, scaler_paths)
        ]
        budgets = [
            (
                float(dataset.manifest.get("node_ratio", 0.50)),
                float(dataset.manifest.get("edge_ratio", 0.30)),
            )
            for dataset in datasets
        ]
        order = sorted(range(3), key=lambda index: budgets[index])
        datasets = [datasets[index] for index in order]
        budgets = [budgets[index] for index in order]
        for observed, expected in zip(budgets, self.expected_budgets):
            if any(
                abs(left - right) > 1.0e-8
                for left, right in zip(observed, expected)
            ):
                raise ValueError("E1 hard-budget grid is not frozen")
        anchor = datasets[1]
        for dataset in datasets:
            checks = (
                dataset.split == anchor.split,
                dataset.sample_keys == anchor.sample_keys,
                dataset.labels == anchor.labels,
                tuple(dataset.sites) == tuple(anchor.sites),
                tuple(dataset.subject_ids) == tuple(anchor.subject_ids),
                dataset.manifest["protocol_sha256"]
                == anchor.manifest["protocol_sha256"],
                dataset.manifest["selector_checkpoint_sha256"]
                == anchor.manifest["selector_checkpoint_sha256"],
                dataset.manifest["selection_mode"]
                == anchor.manifest["selection_mode"],
                int(dataset.manifest["selection_seed"])
                == int(anchor.manifest["selection_seed"]),
            )
            if not all(checks):
                raise ValueError("E1 budget datasets are not aligned")
        self.datasets = tuple(datasets)
        self.manifest_paths = tuple(
            Path(datasets[index].manifest_path) for index in range(3)
        )
        self.scaler_paths = tuple(
            Path(datasets[index].scaler_path) for index in range(3)
        )
        self.manifest = dict(anchor.manifest)
        self.manifest["multi_budget_grid"] = [
            [float(node), float(edge)] for node, edge in budgets
        ]
        self.samples = []
        for index in range(len(anchor)):
            views = tuple(dataset[index] for dataset in datasets)
            middle = views[1]
            self.samples.append(
                SVSignedGINSampleInput(
                    sample_key=middle.sample_key,
                    label=middle.label,
                    windows=middle.windows,
                    static_features=middle.static_features,
                    variation=middle.variation,
                    spectral_direction=middle.spectral_direction,
                    diffusion_geometry=middle.diffusion_geometry,
                    budget_views=views,
                )
            )
        self.sites = list(anchor.sites)
        self.subject_ids = list(anchor.subject_ids)

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


class SVBalancedBatchSampler(Sampler):
    """Deterministic balanced batches without also weighting the loss."""

    def __init__(self, labels, batch_size: int, seed: int) -> None:
        self.labels = [int(value) for value in labels]
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("balanced SV batches require batch size >= 2")
        self.by_class = {
            label: [
                index
                for index, value in enumerate(self.labels)
                if value == label
            ]
            for label in (0, 1)
        }
        if not self.by_class[0] or not self.by_class[1]:
            raise ValueError("balanced SV sampler requires both classes")
        self.num_batches = int(
            math.ceil(len(self.labels) / float(self.batch_size))
        )
        self.random = random.Random(int(seed))

    def __iter__(self):
        for _ in range(self.num_batches):
            positive_count = self.batch_size // 2
            negative_count = self.batch_size - positive_count
            batch = self._draw(1, positive_count) + self._draw(
                0, negative_count
            )
            self.random.shuffle(batch)
            yield batch

    def _draw(self, label: int, count: int):
        population = self.by_class[int(label)]
        if len(population) >= int(count):
            return self.random.sample(population, int(count))
        return [self.random.choice(population) for _ in range(int(count))]

    def __len__(self):
        return self.num_batches


class SVSiteClassBalancedBatchSampler(Sampler):
    """Deterministically balance class and sites within each class."""

    def __init__(self, labels, sites, batch_size: int, seed: int) -> None:
        self.labels = [int(value) for value in labels]
        self.sites = [str(value) for value in sites]
        self.batch_size = int(batch_size)
        if len(self.labels) != len(self.sites) or not self.labels:
            raise ValueError("site/class sampler inputs are misaligned")
        if self.batch_size < 2:
            raise ValueError(
                "site/class balanced SV batches require batch size >= 2"
            )
        self.by_class_site = {0: {}, 1: {}}
        for index, (label, site) in enumerate(
            zip(self.labels, self.sites)
        ):
            if label not in (0, 1) or not site:
                raise ValueError("site/class sampler metadata is invalid")
            self.by_class_site[label].setdefault(site, []).append(index)
        if not self.by_class_site[0] or not self.by_class_site[1]:
            raise ValueError(
                "site/class balanced sampler requires both classes"
            )
        self.num_batches = int(
            math.ceil(len(self.labels) / float(self.batch_size))
        )
        self.random = random.Random(int(seed))
        self.site_offsets = {0: 0, 1: 0}

    def _draw_class(self, label: int, count: int):
        sites = sorted(self.by_class_site[int(label)])
        self.random.shuffle(sites)
        offset = int(self.site_offsets[int(label)])
        selected = []
        for item in range(int(count)):
            site = sites[(offset + item) % len(sites)]
            selected.append(
                self.random.choice(self.by_class_site[int(label)][site])
            )
        self.site_offsets[int(label)] = (
            offset + int(count)
        ) % len(sites)
        return selected

    def __iter__(self):
        for _ in range(self.num_batches):
            positive_count = self.batch_size // 2
            negative_count = self.batch_size - positive_count
            batch = self._draw_class(1, positive_count)
            batch += self._draw_class(0, negative_count)
            self.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def create_sv_signed_gin_loader(
    dataset: SVSignedGINDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    balanced_batch_sampler: bool = False,
    site_class_balanced_sampler: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid SV Signed-GIN loader configuration")
    if dataset.split != "train" and shuffle:
        raise ValueError("SV validation/test loaders cannot shuffle")
    if balanced_batch_sampler and dataset.split != "train":
        raise ValueError("balanced SV batches are train-only")
    if balanced_batch_sampler and shuffle:
        raise ValueError("balanced SV sampler owns the training order")
    if site_class_balanced_sampler and dataset.split != "train":
        raise ValueError("site/class balanced SV batches are train-only")
    if site_class_balanced_sampler and shuffle:
        raise ValueError(
            "site/class balanced sampler owns the training order"
        )
    if balanced_batch_sampler and site_class_balanced_sampler:
        raise ValueError("SV balanced samplers are mutually exclusive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    common = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_sv_signed_gin,
        "worker_init_fn": _seed_worker if num_workers else None,
    }
    if balanced_batch_sampler:
        return DataLoader(
            batch_sampler=SVBalancedBatchSampler(
                dataset.labels, int(batch_size), int(seed)
            ),
            **common
        )
    if site_class_balanced_sampler:
        return DataLoader(
            batch_sampler=SVSiteClassBalancedBatchSampler(
                dataset.labels,
                dataset.sites,
                int(batch_size),
                int(seed),
            ),
            **common
        )
    return DataLoader(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        **common
    )
