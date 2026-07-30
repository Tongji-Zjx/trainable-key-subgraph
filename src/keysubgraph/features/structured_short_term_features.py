"""Coordinate-free, community-structured features for the short-term branch.

Community identifiers are deliberately used only as within-window equivalence
labels.  They are never embedded or compared across windows/samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample


NODE_FEATURE_NAMES: Tuple[str, ...] = (
    "absolute_degree",
    "positive_degree",
    "negative_degree_magnitude",
    "positive_ratio",
    "negative_ratio",
    "delta_absolute_degree",
    "delta_positive_degree",
    "delta_negative_degree_magnitude",
    "community_fraction",
    "intra_positive_mean_strength",
    "intra_negative_mean_strength",
    "inter_positive_mean_strength",
    "inter_negative_mean_strength",
    "intra_edge_density",
    "inter_edge_density",
)

COMMUNITY_SUMMARY_NAMES: Tuple[str, ...] = (
    "community_count_ratio",
    "community_size_entropy",
    "maximum_community_fraction",
    "mean_intra_positive_strength",
    "mean_intra_negative_strength",
    "mean_intra_edge_density",
)


@dataclass(frozen=True)
class StructuredWindowFeatures:
    """Features and invariant summaries for one graph window."""

    node_features: torch.Tensor
    community_summary: torch.Tensor
    mean_absolute_degree: torch.Tensor


@dataclass(frozen=True)
class StructuredShortTermStandardizer:
    """Train-only statistics used by the coordinate-free short-term branch."""

    node_mean: Tuple[float, ...]
    node_std: Tuple[float, ...]
    community_mean: Tuple[float, ...]
    community_std: Tuple[float, ...]
    train_sample_count: int
    train_window_count: int
    train_node_count: int
    protocol_sha256: str
    edge_presence_threshold: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.node_mean) != len(NODE_FEATURE_NAMES):
            raise ValueError("node standardizer dimension mismatch")
        if len(self.node_std) != len(NODE_FEATURE_NAMES):
            raise ValueError("node standardizer dimension mismatch")
        if len(self.community_mean) != len(COMMUNITY_SUMMARY_NAMES):
            raise ValueError("community standardizer dimension mismatch")
        if len(self.community_std) != len(COMMUNITY_SUMMARY_NAMES):
            raise ValueError("community standardizer dimension mismatch")
        if min(self.node_std) <= 0.0 or min(self.community_std) <= 0.0:
            raise ValueError("standardizer standard deviations must be positive")
        if self.train_sample_count <= 0:
            raise ValueError("standardizer requires at least one training sample")

    def normalize_nodes(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.node_mean)
        std = values.new_tensor(self.node_std)
        return (values - mean) / std

    def community_anomaly(self, summary: torch.Tensor) -> torch.Tensor:
        """Return a label-invariant structural anomaly score.

        The score is the mean absolute train-standardized deviation of the
        window-level community summary.  It replaces the paper's raw
        community-label frequency, whose identifiers are not aligned.
        """

        mean = summary.new_tensor(self.community_mean)
        std = summary.new_tensor(self.community_std)
        return ((summary - mean) / std).abs().mean()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": "structured_short_term_standardizer",
            "schema_version": self.schema_version,
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "community_summary_names": list(COMMUNITY_SUMMARY_NAMES),
            "node_mean": list(self.node_mean),
            "node_std": list(self.node_std),
            "community_mean": list(self.community_mean),
            "community_std": list(self.community_std),
            "train_sample_count": self.train_sample_count,
            "train_window_count": self.train_window_count,
            "train_node_count": self.train_node_count,
            "protocol_sha256": self.protocol_sha256,
            "edge_presence_threshold": self.edge_presence_threshold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredShortTermStandardizer":
        if payload.get("artifact_type") != "structured_short_term_standardizer":
            raise ValueError("unexpected short-term standardizer artifact type")
        if tuple(payload.get("node_feature_names", ())) != NODE_FEATURE_NAMES:
            raise ValueError("short-term node feature schema mismatch")
        if tuple(payload.get("community_summary_names", ())) != COMMUNITY_SUMMARY_NAMES:
            raise ValueError("short-term community summary schema mismatch")
        return cls(
            node_mean=tuple(float(value) for value in payload["node_mean"]),
            node_std=tuple(float(value) for value in payload["node_std"]),
            community_mean=tuple(float(value) for value in payload["community_mean"]),
            community_std=tuple(float(value) for value in payload["community_std"]),
            train_sample_count=int(payload["train_sample_count"]),
            train_window_count=int(payload["train_window_count"]),
            train_node_count=int(payload["train_node_count"]),
            protocol_sha256=str(payload["protocol_sha256"]),
            edge_presence_threshold=float(payload["edge_presence_threshold"]),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def save(self, path: Path, overwrite: bool = False) -> Path:
        path = Path(path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "StructuredShortTermStandardizer":
        with Path(path).resolve().open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class StructuredShortTermFeatureBuilder:
    """Build signed, coordinate-free and community-label-invariant features."""

    def __init__(
        self,
        edge_presence_threshold: float = 0.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        if edge_presence_threshold < 0.0:
            raise ValueError("edge_presence_threshold must be non-negative")
        self.edge_presence_threshold = float(edge_presence_threshold)
        self.epsilon = float(epsilon)

    @staticmethod
    def _node_names(
        adjacency: torch.Tensor,
        node_names: Sequence[str],
    ) -> Tuple[str, ...]:
        names = tuple(str(name) for name in node_names)
        if len(names) != int(adjacency.shape[0]):
            raise ValueError("node-name count does not match adjacency")
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique within a timepoint")
        return names

    @staticmethod
    def _degrees(adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive = adjacency.clamp_min(0.0).sum(dim=1)
        negative = (-adjacency.clamp_max(0.0)).sum(dim=1)
        return positive + negative, positive, negative

    def _aligned_previous_degrees(
        self,
        current_adjacency: torch.Tensor,
        current_names: Sequence[str],
        previous_adjacency: Optional[torch.Tensor],
        previous_names: Optional[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency = current_adjacency
        node_count = int(adjacency.shape[0])
        if previous_adjacency is None or previous_names is None:
            # There is no fabricated pre-sequence graph.  The first temporal
            # difference is defined as zero by using the current degrees as
            # the aligned reference.
            return self._degrees(adjacency)

        previous_degree, previous_positive, previous_negative = self._degrees(
            previous_adjacency
        )
        checked_previous_names = self._node_names(
            previous_adjacency,
            previous_names,
        )
        previous_lookup = {
            name: index for index, name in enumerate(checked_previous_names)
        }
        checked_current_names = self._node_names(adjacency, current_names)
        # A node absent from the previous window receives a zero temporal
        # difference, not a fabricated jump from an all-zero graph.
        current_degree, current_positive, current_negative = self._degrees(
            adjacency
        )
        aligned_degree = current_degree.clone()
        aligned_positive = current_positive.clone()
        aligned_negative = current_negative.clone()
        for current_index, name in enumerate(checked_current_names):
            previous_index = previous_lookup.get(name)
            if previous_index is None:
                continue
            aligned_degree[current_index] = previous_degree[previous_index]
            aligned_positive[current_index] = previous_positive[previous_index]
            aligned_negative[current_index] = previous_negative[previous_index]
        return aligned_degree, aligned_positive, aligned_negative

    def _community_features(
        self,
        adjacency: torch.Tensor,
        community: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_count = int(adjacency.shape[0])
        device = adjacency.device
        dtype = adjacency.dtype
        output = adjacency.new_zeros((node_count, 7))
        valid = community >= 0
        valid_labels = torch.unique(community[valid], sorted=True)
        if node_count == 0 or int(valid_labels.numel()) == 0:
            return output, adjacency.new_zeros((len(COMMUNITY_SUMMARY_NAMES),))

        positive = adjacency.clamp_min(0.0)
        negative = -adjacency.clamp_max(0.0)
        present = adjacency.abs() > self.edge_presence_threshold
        eye = torch.eye(node_count, dtype=torch.bool, device=device)
        present = present & ~eye

        community_sizes: List[float] = []
        community_positive_means: List[torch.Tensor] = []
        community_negative_means: List[torch.Tensor] = []
        community_densities: List[torch.Tensor] = []

        all_indices = torch.arange(node_count, device=device)
        for label in valid_labels:
            member_mask = valid & (community == label)
            member_indices = all_indices[member_mask]
            community_size = int(member_indices.numel())
            if community_size == 0:
                continue
            outside_mask = ~member_mask
            outside_count = node_count - community_size
            intra_denominator = max(community_size - 1, 1)
            inter_denominator = max(outside_count, 1)

            intra_positive = positive[member_mask][:, member_mask].sum(dim=1)
            intra_negative = negative[member_mask][:, member_mask].sum(dim=1)
            inter_positive = positive[member_mask][:, outside_mask].sum(dim=1)
            inter_negative = negative[member_mask][:, outside_mask].sum(dim=1)
            intra_density = (
                present[member_mask][:, member_mask].to(dtype).sum(dim=1)
                / float(intra_denominator)
            )
            inter_density = (
                present[member_mask][:, outside_mask].to(dtype).sum(dim=1)
                / float(inter_denominator)
            )

            output[member_mask, 0] = float(community_size) / float(node_count)
            output[member_mask, 1] = intra_positive / float(intra_denominator)
            output[member_mask, 2] = intra_negative / float(intra_denominator)
            output[member_mask, 3] = inter_positive / float(inter_denominator)
            output[member_mask, 4] = inter_negative / float(inter_denominator)
            output[member_mask, 5] = intra_density
            output[member_mask, 6] = inter_density

            community_sizes.append(float(community_size))
            community_positive_means.append(output[member_mask, 1].mean())
            community_negative_means.append(output[member_mask, 2].mean())
            community_densities.append(output[member_mask, 5].mean())

        size_tensor = torch.tensor(community_sizes, device=device, dtype=dtype)
        fractions = size_tensor / size_tensor.sum().clamp_min(self.epsilon)
        size_entropy = -(fractions * fractions.clamp_min(self.epsilon).log()).sum()
        if len(community_sizes) > 1:
            size_entropy = size_entropy / math.log(float(len(community_sizes)))
        else:
            size_entropy = size_entropy * 0.0
        summary = torch.stack(
            (
                adjacency.new_tensor(float(len(community_sizes)) / float(node_count)),
                size_entropy,
                fractions.max(),
                torch.stack(community_positive_means).mean(),
                torch.stack(community_negative_means).mean(),
                torch.stack(community_densities).mean(),
            )
        )
        return output, summary

    def build_window(
        self,
        current_adjacency: torch.Tensor,
        current_names: Sequence[str],
        community: torch.Tensor,
        previous_adjacency: Optional[torch.Tensor] = None,
        previous_names: Optional[Sequence[str]] = None,
    ) -> StructuredWindowFeatures:
        adjacency = current_adjacency
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not torch.isfinite(adjacency).all():
            raise ValueError("adjacency contains non-finite values")
        self._node_names(adjacency, current_names)
        community = community.to(device=adjacency.device, dtype=torch.long)
        if community.ndim != 1 or community.shape[0] != adjacency.shape[0]:
            raise ValueError("community labels do not match adjacency")

        degree, positive, negative = self._degrees(adjacency)
        previous_degree, previous_positive, previous_negative = (
            self._aligned_previous_degrees(
                adjacency,
                current_names,
                previous_adjacency,
                previous_names,
            )
        )
        denominator = (positive + negative).clamp_min(self.epsilon)
        community_features, community_summary = self._community_features(
            adjacency,
            community,
        )
        node_features = torch.cat(
            (
                degree.unsqueeze(1),
                positive.unsqueeze(1),
                negative.unsqueeze(1),
                (positive / denominator).unsqueeze(1),
                (negative / denominator).unsqueeze(1),
                (degree - previous_degree).unsqueeze(1),
                (positive - previous_positive).unsqueeze(1),
                (negative - previous_negative).unsqueeze(1),
                community_features,
            ),
            dim=1,
        )
        if node_features.shape[1] != len(NODE_FEATURE_NAMES):
            raise RuntimeError("internal short-term feature dimension mismatch")
        return StructuredWindowFeatures(
            node_features=node_features,
            community_summary=community_summary,
            mean_absolute_degree=degree.mean(),
        )

    def build_sample(self, sample: GraphSequenceSample) -> List[StructuredWindowFeatures]:
        output: List[StructuredWindowFeatures] = []
        previous_adjacency: Optional[torch.Tensor] = None
        previous_names: Optional[Sequence[str]] = None
        for adjacency, names, community in zip(
            sample.adjacency,
            sample.node_names,
            sample.communities,
        ):
            output.append(
                self.build_window(
                    current_adjacency=adjacency,
                    current_names=names,
                    community=community,
                    previous_adjacency=previous_adjacency,
                    previous_names=previous_names,
                )
            )
            previous_adjacency = adjacency
            previous_names = names
        if not output:
            raise ValueError("short-term sample contains no graph windows")
        return output


class _VectorMoments:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.total = torch.zeros(dimension, dtype=torch.float64)
        self.total_square = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.total.numel():
            raise ValueError("moment input dimension mismatch")
        self.count += int(values.shape[0])
        self.total += values.sum(dim=0)
        self.total_square += values.square().sum(dim=0)

    def finish(self, minimum_std: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        if self.count <= 0:
            raise ValueError("cannot finish empty moments")
        mean = self.total / float(self.count)
        variance = self.total_square / float(self.count) - mean.square()
        std = variance.clamp_min(0.0).sqrt().clamp_min(float(minimum_std))
        return tuple(mean.tolist()), tuple(std.tolist())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fit_structured_short_term_standardizer(
    samples: Iterable[GraphSequenceSample],
    protocol_path: Path,
    edge_presence_threshold: float,
    minimum_std: float = 1.0e-6,
) -> StructuredShortTermStandardizer:
    """Fit all normalization state from a training-sample iterable only."""

    builder = StructuredShortTermFeatureBuilder(
        edge_presence_threshold=edge_presence_threshold
    )
    node_moments = _VectorMoments(len(NODE_FEATURE_NAMES))
    community_moments = _VectorMoments(len(COMMUNITY_SUMMARY_NAMES))
    sample_count = 0
    window_count = 0
    for sample in samples:
        split = str(getattr(sample, "split", ""))
        if split and split != "train":
            raise ValueError("short-term standardizer may only consume train samples")
        sample_count += 1
        windows = builder.build_sample(sample)
        for window in windows:
            node_moments.update(window.node_features)
            community_moments.update(window.community_summary)
            window_count += 1
    node_mean, node_std = node_moments.finish(minimum_std)
    community_mean, community_std = community_moments.finish(minimum_std)
    return StructuredShortTermStandardizer(
        node_mean=node_mean,
        node_std=node_std,
        community_mean=community_mean,
        community_std=community_std,
        train_sample_count=sample_count,
        train_window_count=window_count,
        train_node_count=node_moments.count,
        protocol_sha256=sha256_file(protocol_path),
        edge_presence_threshold=float(edge_presence_threshold),
    )
