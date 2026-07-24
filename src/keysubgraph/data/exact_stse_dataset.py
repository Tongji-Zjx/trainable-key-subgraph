"""Coordinate-preserving dataset isolated to the Exact-STSE reproduction."""

from __future__ import absolute_import, division, print_function

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .graph_dataset import (
    GraphSequenceDataset,
    GraphSequenceSample,
    _adapt_payload,
)


def _coordinate_sequence(
    value: Any, node_counts: Sequence[int]
) -> Tuple[torch.Tensor, ...]:
    raw: List[torch.Tensor]
    if torch.is_tensor(value):
        if value.dim() == 2:
            raw = [value for _ in node_counts]
        elif value.dim() == 3 and value.shape[0] == len(node_counts):
            raw = [value[index] for index in range(value.shape[0])]
        else:
            raise ValueError("coords tensor must have shape [N,3] or [M,N,3]")
    elif isinstance(value, (list, tuple)) and len(value) == len(node_counts):
        if not all(torch.is_tensor(item) and item.dim() == 2 for item in value):
            raise ValueError("time-aligned coords must contain 2-D tensors")
        raw = list(value)
    else:
        raise ValueError("coords are neither shared nor time-aligned")

    result = []
    for time_index, (coordinates, node_count) in enumerate(
        zip(raw, node_counts)
    ):
        coordinates = coordinates.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if tuple(coordinates.shape) != (int(node_count), 3):
            raise ValueError(
                "time {} coordinates must have shape [{},3]".format(
                    time_index, node_count
                )
            )
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError(
                "time {} coordinates contain non-finite values".format(
                    time_index
                )
            )
        result.append(coordinates)
    if not any(bool((item != 0.0).any()) for item in result):
        raise ValueError("Exact-STSE coordinate sample is all zero")
    return tuple(result)


def _coordinates_for_mode(
    payload: Any,
    node_counts: Sequence[int],
    require_coordinates: bool,
) -> Tuple[torch.Tensor, ...]:
    """Load real coordinates or create inert placeholders for NoCoord."""

    if require_coordinates:
        if "coords" not in payload:
            raise ValueError("sample is missing coords")
        return _coordinate_sequence(payload["coords"], node_counts)
    # Deliberately do not inspect payload["coords"]. NoCoord shares the sample
    # interface with Coord, but its model input remains the original 18-D
    # coordinate-free feature vector.
    return tuple(
        torch.zeros((int(node_count), 3), dtype=torch.float32)
        for node_count in node_counts
    )


@dataclass(frozen=True)
class ExactSTSESample:
    graph: GraphSequenceSample
    coordinates: Tuple[torch.Tensor, ...]

    @property
    def sample_key(self) -> str:
        return self.graph.sample_key

    @property
    def label(self) -> int:
        return int(self.graph.label)

    @property
    def split(self) -> str:
        return self.graph.split

    @property
    def num_timepoints(self) -> int:
        return self.graph.num_timepoints

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "ExactSTSESample":
        return ExactSTSESample(
            graph=self.graph.to(device, non_blocking=non_blocking),
            coordinates=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.coordinates
            ),
        )


@dataclass(frozen=True)
class ExactSTSEBatch:
    samples: Tuple[ExactSTSESample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ExactSTSESample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> ExactSTSESample:
        return self.samples[index]

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(
            [sample.label for sample in self.samples], dtype=torch.long
        )

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(sample.sample_key for sample in self.samples)

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "ExactSTSEBatch":
        return ExactSTSEBatch(
            tuple(
                sample.to(device=device, non_blocking=non_blocking)
                for sample in self.samples
            )
        )


def exact_stse_collate(
    samples: Sequence[ExactSTSESample],
) -> ExactSTSEBatch:
    if not samples:
        raise ValueError("cannot collate an empty Exact-STSE batch")
    return ExactSTSEBatch(tuple(samples))


class ExactSTSEDataset(GraphSequenceDataset):
    """Load isolated Exact-STSE samples with an explicit coordinate contract."""

    def __init__(
        self,
        dataset_root: Path,
        sample_index_csv: Path,
        splits_csv: Path,
        split: str,
        edge_presence_threshold: float = 0.0,
        require_coordinates: bool = True,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            sample_index_csv=sample_index_csv,
            splits_csv=splits_csv,
            split=split,
            edge_presence_threshold=edge_presence_threshold,
        )
        self.require_coordinates = bool(require_coordinates)

    def __getitem__(self, index: int) -> ExactSTSESample:
        assignment = self.assignments[index]
        path = (self.dataset_root / assignment.relative_path).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError:
            raise ValueError("sample path escapes the dataset root")
        if not path.is_file():
            raise FileNotFoundError(str(path))
        try:
            payload = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")
        except Exception as error:
            raise RuntimeError(
                "failed to load {}: {}".format(
                    assignment.sample_key, error
                )
            )
        if not isinstance(payload, dict):
            raise ValueError(
                "{} payload is not a dict".format(assignment.sample_key)
            )
        try:
            graph = _adapt_payload(
                payload, assignment, self.edge_presence_threshold
            )
            coordinates = _coordinates_for_mode(
                payload,
                graph.node_counts,
                self.require_coordinates,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("{}: {}".format(assignment.sample_key, error))
        return ExactSTSESample(graph=graph, coordinates=coordinates)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_exact_stse_loader(
    dataset: ExactSTSEDataset,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle=None,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if shuffle is None:
        shuffle = dataset.split == "train"
    if shuffle and dataset.split != "train":
        raise ValueError("validation and test loaders cannot shuffle")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        collate_fn=exact_stse_collate,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
