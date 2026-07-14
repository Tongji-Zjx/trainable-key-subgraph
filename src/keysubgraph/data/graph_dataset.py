"""Variable-length graph-sequence Dataset and list-based DataLoader."""

from __future__ import absolute_import, division, print_function

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_split import (
    SPLIT_NAMES,
    IndexSample,
    SplitAssignment,
    read_sample_index,
    read_split_assignments,
)


@dataclass(frozen=True)
class GraphSequenceSample:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    split: str
    relative_path: str
    adjacency: Tuple[torch.Tensor, ...]
    edge_mask: Tuple[torch.Tensor, ...]
    coordinates: Tuple[torch.Tensor, ...]
    node_names: Tuple[Tuple[str, ...], ...]
    communities: Tuple[torch.Tensor, ...]
    window_starts: torch.Tensor
    source_global_threshold: Optional[float]
    repetition_time: Optional[float]
    edge_presence_threshold: float

    @property
    def num_timepoints(self) -> int:
        return len(self.adjacency)

    @property
    def node_counts(self) -> Tuple[int, ...]:
        return tuple(int(graph.shape[0]) for graph in self.adjacency)

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "GraphSequenceSample":
        return GraphSequenceSample(
            sample_key=self.sample_key,
            sample_id=self.sample_id,
            site=self.site,
            subject_id=self.subject_id,
            session_id=self.session_id,
            label=self.label,
            split=self.split,
            relative_path=self.relative_path,
            adjacency=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.adjacency
            ),
            edge_mask=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.edge_mask
            ),
            coordinates=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.coordinates
            ),
            node_names=self.node_names,
            communities=tuple(
                item.to(device=device, non_blocking=non_blocking)
                for item in self.communities
            ),
            window_starts=self.window_starts.to(
                device=device, non_blocking=non_blocking
            ),
            source_global_threshold=self.source_global_threshold,
            repetition_time=self.repetition_time,
            edge_presence_threshold=self.edge_presence_threshold,
        )


@dataclass(frozen=True)
class GraphSequenceBatch:
    """A typed list batch; tensors remain unpadded inside each sample."""

    samples: Tuple[GraphSequenceSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[GraphSequenceSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> GraphSequenceSample:
        return self.samples[index]

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor([sample.label for sample in self.samples], dtype=torch.long)

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(sample.sample_key for sample in self.samples)

    def to(
        self, device: Union[str, torch.device], non_blocking: bool = False
    ) -> "GraphSequenceBatch":
        return GraphSequenceBatch(
            tuple(
                sample.to(device=device, non_blocking=non_blocking)
                for sample in self.samples
            )
        )


def list_batch_collate(samples: Sequence[GraphSequenceSample]) -> GraphSequenceBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return GraphSequenceBatch(tuple(samples))


def _tensor_sequence(value: Any, field_name: str, item_dim: int) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.dim() == item_dim:
            return [value]
        if value.dim() == item_dim + 1:
            return [value[index] for index in range(value.shape[0])]
    if isinstance(value, (list, tuple)) and value:
        if all(torch.is_tensor(item) and item.dim() == item_dim for item in value):
            return list(value)
    raise ValueError("{} is not a valid tensor sequence".format(field_name))


def _coordinate_sequence(value: Any, node_counts: Sequence[int]) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.dim() == 2 and len(set(node_counts)) == 1:
            return [value for _ in node_counts]
        if value.dim() == 3 and value.shape[0] == len(node_counts):
            return [value[index] for index in range(value.shape[0])]
    if isinstance(value, (list, tuple)) and len(value) == len(node_counts):
        if all(torch.is_tensor(item) and item.dim() == 2 for item in value):
            return list(value)
    raise ValueError("coords do not align with the graph sequence")


def _name_sequence(value: Any, node_counts: Sequence[int]) -> List[Tuple[str, ...]]:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, str) for item in value
    ):
        if len(set(node_counts)) != 1 or len(value) != node_counts[0]:
            raise ValueError("shared node_names do not align with node counts")
        names = tuple(value)
        return [names for _ in node_counts]
    if isinstance(value, (list, tuple)) and len(value) == len(node_counts):
        result = []
        for item in value:
            if not isinstance(item, (list, tuple)) or not all(
                isinstance(name, str) for name in item
            ):
                raise ValueError("time-aligned node_names must contain strings")
            result.append(tuple(item))
        return result
    raise ValueError("node_names are neither shared nor time-aligned")


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _adapt_payload(
    payload: Dict[str, Any],
    assignment: SplitAssignment,
    edge_presence_threshold: float,
) -> GraphSequenceSample:
    required = {
        "adjacency",
        "coords",
        "node_names",
        "community_sequence",
        "window_starts",
        "global_threshold",
        "t_r",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("missing .pt fields: {}".format(", ".join(missing)))

    adjacency = _tensor_sequence(payload["adjacency"], "adjacency", 2)
    if not adjacency:
        raise ValueError("graph sequence is empty")
    node_counts = []
    adapted_adjacency = []
    edge_masks = []
    for time_index, graph in enumerate(adjacency):
        if graph.shape[0] == 0 or graph.shape[0] != graph.shape[1]:
            raise ValueError("time {} adjacency must be non-empty and square".format(time_index))
        graph = graph.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(graph).all()):
            raise ValueError("time {} adjacency contains non-finite values".format(time_index))
        if not torch.allclose(graph, graph.transpose(0, 1), atol=1e-6, rtol=0.0):
            raise ValueError("time {} adjacency is not symmetric".format(time_index))
        if float(graph.diagonal().abs().max().item()) > 1e-8:
            raise ValueError("time {} adjacency contains self loops".format(time_index))
        mask = graph.abs() > edge_presence_threshold
        mask.fill_diagonal_(False)
        adapted_adjacency.append(graph)
        edge_masks.append(mask)
        node_counts.append(int(graph.shape[0]))

    communities = _tensor_sequence(
        payload["community_sequence"], "community_sequence", 1
    )
    if len(communities) != len(adapted_adjacency):
        raise ValueError("community sequence length does not match adjacency")
    adapted_communities = []
    for time_index, (values, node_count) in enumerate(zip(communities, node_counts)):
        if values.numel() != node_count:
            raise ValueError("time {} community labels do not match nodes".format(time_index))
        if values.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("community labels must use an integer dtype")
        values = values.detach().to(device="cpu", dtype=torch.long).contiguous()
        if bool((values < 0).any()):
            raise ValueError("community labels must be non-negative")
        adapted_communities.append(values)

    coordinates = _coordinate_sequence(payload["coords"], node_counts)
    adapted_coordinates = []
    spatial_dims = set()
    for time_index, (coords, node_count) in enumerate(zip(coordinates, node_counts)):
        if coords.shape[0] != node_count:
            raise ValueError("time {} coordinates do not match nodes".format(time_index))
        coords = coords.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(coords).all()):
            raise ValueError("coordinates contain non-finite values")
        if not bool((coords != 0).any()):
            raise ValueError("all-zero coordinates are excluded by the data protocol")
        spatial_dims.add(int(coords.shape[1]))
        adapted_coordinates.append(coords)
    if len(spatial_dims) != 1:
        raise ValueError("coordinate dimensions vary across time")

    node_names = _name_sequence(payload["node_names"], node_counts)
    for time_index, (names, node_count) in enumerate(zip(node_names, node_counts)):
        if len(names) != node_count or len(set(names)) != len(names):
            raise ValueError("time {} node names are invalid".format(time_index))

    starts = payload["window_starts"]
    if not torch.is_tensor(starts) or starts.dim() != 1:
        raise ValueError("window_starts must be a 1-D tensor")
    if starts.numel() != len(adapted_adjacency):
        raise ValueError("window_starts length does not match adjacency")
    starts = starts.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(starts).all()):
        raise ValueError("window_starts contains non-finite values")
    if starts.numel() > 1 and not bool((starts[1:] > starts[:-1]).all()):
        raise ValueError("window_starts must be strictly increasing")

    return GraphSequenceSample(
        sample_key=assignment.sample_key,
        sample_id=assignment.sample_id,
        site=assignment.site,
        subject_id=assignment.subject_id,
        session_id=assignment.session_id,
        label=assignment.label,
        split=assignment.split,
        relative_path=assignment.relative_path,
        adjacency=tuple(adapted_adjacency),
        edge_mask=tuple(edge_masks),
        coordinates=tuple(adapted_coordinates),
        node_names=tuple(node_names),
        communities=tuple(adapted_communities),
        window_starts=starts,
        source_global_threshold=_optional_float(payload["global_threshold"]),
        repetition_time=_optional_float(payload["t_r"]),
        edge_presence_threshold=float(edge_presence_threshold),
    )


class GraphSequenceDataset(Dataset):
    """Lazily load one unpadded graph sequence at a time."""

    def __init__(
        self,
        dataset_root: Path,
        sample_index_csv: Path,
        splits_csv: Path,
        split: str,
        edge_presence_threshold: float = 0.0,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError("split must be one of {}".format(SPLIT_NAMES))
        if edge_presence_threshold < 0.0:
            raise ValueError("edge_presence_threshold must be non-negative")
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.edge_presence_threshold = float(edge_presence_threshold)

        index_samples = read_sample_index(sample_index_csv)
        assignments = read_split_assignments(splits_csv)
        index_by_key = {sample.sample_key: sample for sample in index_samples}
        assignment_keys = {item.sample_key for item in assignments}
        if assignment_keys != set(index_by_key):
            raise ValueError("sample index and splits.csv sample sets differ")
        for assignment in assignments:
            indexed = index_by_key[assignment.sample_key]
            self._validate_metadata(indexed, assignment)
        self.assignments = tuple(
            item for item in assignments if item.split == self.split
        )
        if not self.assignments:
            raise ValueError("requested split is empty")

    @staticmethod
    def _validate_metadata(indexed: IndexSample, assignment: SplitAssignment) -> None:
        pairs = (
            ("sample_id", indexed.sample_id, assignment.sample_id),
            ("site", indexed.site, assignment.site),
            ("subject_id", indexed.subject_id, assignment.subject_id),
            ("session_id", indexed.session_id, assignment.session_id),
            ("label", indexed.label, assignment.label),
            ("relative_path", indexed.relative_path, assignment.relative_path),
            ("group_id", indexed.group_id, assignment.group_id),
        )
        for name, index_value, split_value in pairs:
            if index_value != split_value:
                raise ValueError(
                    "index/split metadata mismatch for {}: {}".format(
                        assignment.sample_key, name
                    )
                )

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int) -> GraphSequenceSample:
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
        except Exception as error:
            raise RuntimeError(
                "failed to load {}: {}".format(assignment.sample_key, error)
            )
        if not isinstance(payload, dict):
            raise ValueError("{} payload is not a dict".format(assignment.sample_key))
        try:
            return _adapt_payload(payload, assignment, self.edge_presence_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("{}: {}".format(assignment.sample_key, error))


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    dataset: GraphSequenceDataset,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic DataLoader that never pads or truncates graphs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if shuffle is None:
        shuffle = dataset.split == "train"
    if shuffle and dataset.split != "train":
        raise ValueError("validation and test DataLoaders must not shuffle")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_batch_collate,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
