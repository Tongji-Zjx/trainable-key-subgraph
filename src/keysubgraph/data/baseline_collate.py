"""Mask-aware dense batching for variable-length hard-subgraph sequences."""

from __future__ import absolute_import, division, print_function

import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_dataset import BaselineHardSubgraphDataset, BaselineSequenceSample


@dataclass(frozen=True)
class BaselineBatch:
    """Flattened subgraphs plus explicit window/sample mappings and masks."""

    node_features: torch.Tensor
    adjacency: torch.Tensor
    edge_mask: torch.Tensor
    node_mask: torch.Tensor
    subgraph_to_window: torch.Tensor
    window_to_sample: torch.Tensor
    window_time_index: torch.Tensor
    window_subgraph_count: torch.Tensor
    window_index: torch.Tensor
    time_mask: torch.Tensor
    labels: torch.Tensor
    sample_keys: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    subject_ids: Tuple[str, ...]
    sites: Tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.labels.numel())

    @property
    def subgraph_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def window_count(self) -> int:
        return int(self.window_to_sample.numel())

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features.shape[-1])

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "BaselineBatch":
        tensor_names = (
            "node_features",
            "adjacency",
            "edge_mask",
            "node_mask",
            "subgraph_to_window",
            "window_to_sample",
            "window_time_index",
            "window_subgraph_count",
            "window_index",
            "time_mask",
            "labels",
        )
        moved = {
            name: getattr(self, name).to(
                device=device, non_blocking=non_blocking
            )
            for name in tensor_names
        }
        return BaselineBatch(
            sample_keys=self.sample_keys,
            sample_ids=self.sample_ids,
            subject_ids=self.subject_ids,
            sites=self.sites,
            **moved
        )

    def __len__(self) -> int:
        return self.batch_size


def baseline_padded_collate(
    samples: Sequence[BaselineSequenceSample],
) -> BaselineBatch:
    """Pad only to batch maxima; never truncate nodes, subgraphs, or windows."""

    if not samples:
        raise ValueError("cannot collate an empty baseline batch")
    subgraphs = []
    subgraph_to_window = []
    window_to_sample = []
    window_time_index = []
    window_subgraph_count = []
    max_timepoints = max(sample.num_timepoints for sample in samples)
    window_index = torch.full(
        (len(samples), max_timepoints), -1, dtype=torch.long
    )
    time_mask = torch.zeros(len(samples), max_timepoints, dtype=torch.bool)
    for sample_index, sample in enumerate(samples):
        if sample.num_timepoints < 1:
            raise ValueError("baseline sample contains no timepoints")
        for expected_time, window in enumerate(sample.windows):
            if window.time_index != expected_time:
                raise ValueError("baseline windows must be contiguous and ordered")
            if not window.subgraphs:
                raise ValueError("effective baseline window contains no subgraphs")
            flat_window_index = len(window_to_sample)
            window_index[sample_index, expected_time] = flat_window_index
            time_mask[sample_index, expected_time] = True
            window_to_sample.append(sample_index)
            window_time_index.append(expected_time)
            window_subgraph_count.append(len(window.subgraphs))
            for subgraph in window.subgraphs:
                subgraphs.append(subgraph)
                subgraph_to_window.append(flat_window_index)
    if not subgraphs:
        raise ValueError("baseline batch contains no subgraphs")
    feature_dim = subgraphs[0].node_features.shape[-1]
    if any(item.node_features.shape[-1] != feature_dim for item in subgraphs):
        raise ValueError("node feature dimensions differ across subgraphs")
    max_nodes = max(item.node_count for item in subgraphs)
    if max_nodes < 1:
        raise ValueError("baseline batch contains an empty subgraph")
    dtype = subgraphs[0].node_features.dtype
    node_features = torch.zeros(
        len(subgraphs), max_nodes, feature_dim, dtype=dtype
    )
    adjacency = torch.zeros(len(subgraphs), max_nodes, max_nodes, dtype=dtype)
    edge_mask = torch.zeros(
        len(subgraphs), max_nodes, max_nodes, dtype=torch.bool
    )
    node_mask = torch.zeros(len(subgraphs), max_nodes, dtype=torch.bool)
    for index, subgraph in enumerate(subgraphs):
        node_count = subgraph.node_count
        node_features[index, :node_count] = subgraph.node_features
        adjacency[index, :node_count, :node_count] = subgraph.adjacency
        edge_mask[index, :node_count, :node_count] = subgraph.edge_mask
        node_mask[index, :node_count] = True
    return BaselineBatch(
        node_features=node_features,
        adjacency=adjacency,
        edge_mask=edge_mask,
        node_mask=node_mask,
        subgraph_to_window=torch.tensor(subgraph_to_window, dtype=torch.long),
        window_to_sample=torch.tensor(window_to_sample, dtype=torch.long),
        window_time_index=torch.tensor(window_time_index, dtype=torch.long),
        window_subgraph_count=torch.tensor(window_subgraph_count, dtype=torch.long),
        window_index=window_index,
        time_mask=time_mask,
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        sample_keys=tuple(sample.sample_key for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
        subject_ids=tuple(sample.subject_id for sample in samples),
        sites=tuple(sample.site for sample in samples),
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_baseline_loader(
    dataset: BaselineHardSubgraphDataset,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if shuffle is None:
        shuffle = dataset.split in ("train", "all")
    if shuffle and dataset.split not in ("train", "all"):
        raise ValueError("validation and test baseline loaders must not shuffle")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=baseline_padded_collate,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def iter_valid_window_indices(batch: BaselineBatch) -> Iterator[Tuple[int, int]]:
    for sample_index in range(batch.batch_size):
        for time_index in range(batch.time_mask.shape[1]):
            if bool(batch.time_mask[sample_index, time_index]):
                yield sample_index, time_index
