"""Train-only community frequencies and paper-aligned p_ST statistics."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.graph_dataset import GraphSequenceSample


PAPER_SHORT_TERM_COMMUNITY_FREQUENCY_SCHEMA = (
    "paper_short_term_community_frequency_v1"
)
PAPER_SHORT_TERM_PST_FEATURE_SCHEMA = "paper_short_term_pst_raw_v1"
PST_RAW_STATISTIC_NAMES: Tuple[str, ...] = (
    "degree_mean",
    "degree_std",
    "degree_max",
    "anomaly_mean",
    "anomaly_std",
    "num_valid_windows",
)


def _sample_keys_sha256(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in keys)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperShortTermCommunityFrequency:
    """Community-label probabilities fitted only on the active train split."""

    counts: Tuple[Tuple[int, int], ...]
    total_count: int
    train_sample_count: int
    train_window_count: int
    protocol_sha256: str
    train_manifest_sha256: str
    train_sample_keys_sha256: str
    epsilon: float = 1.0e-12
    outer_fold: Optional[int] = None

    def __post_init__(self) -> None:
        labels = [int(label) for label, _ in self.counts]
        counts = [int(count) for _, count in self.counts]
        if labels != sorted(labels) or len(labels) != len(set(labels)):
            raise ValueError("community-frequency labels must be unique and sorted")
        if any(label < 0 for label in labels):
            raise ValueError("invalid community labels cannot enter frequencies")
        if any(count <= 0 for count in counts):
            raise ValueError("community-frequency counts must be positive")
        if self.total_count <= 0 or sum(counts) != self.total_count:
            raise ValueError("community-frequency total count mismatch")
        if self.train_sample_count <= 0 or self.train_window_count <= 0:
            raise ValueError("community frequencies require non-empty train data")
        if self.epsilon <= 0.0:
            raise ValueError("community-frequency epsilon must be positive")
        if self.outer_fold is not None and self.outer_fold < 0:
            raise ValueError("outer fold cannot be negative")

    @property
    def count_mapping(self) -> Dict[int, int]:
        return {int(label): int(count) for label, count in self.counts}

    @property
    def probability_mapping(self) -> Dict[int, float]:
        denominator = float(self.total_count)
        return {
            int(label): float(count) / denominator
            for label, count in self.counts
        }

    def probability_tensor(
        self,
        size: int,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if size <= 0:
            raise ValueError("community probability tensor size must be positive")
        if reference is None:
            output = torch.zeros(size, dtype=torch.float32)
        else:
            output = reference.new_zeros((size,))
        for label, probability in self.probability_mapping.items():
            if label < size:
                output[label] = probability
        return output

    def to_dict(self) -> Dict[str, Any]:
        probabilities = self.probability_mapping
        return {
            "artifact_type": PAPER_SHORT_TERM_COMMUNITY_FREQUENCY_SCHEMA,
            "schema_version": 1,
            "epsilon": float(self.epsilon),
            "counts": {
                str(label): int(count) for label, count in self.counts
            },
            "probabilities": {
                str(label): float(probabilities[label])
                for label, _ in self.counts
            },
            "total_count": int(self.total_count),
            "train_sample_count": int(self.train_sample_count),
            "train_window_count": int(self.train_window_count),
            "protocol_sha256": str(self.protocol_sha256),
            "train_manifest_sha256": str(self.train_manifest_sha256),
            "train_sample_keys_sha256": str(self.train_sample_keys_sha256),
            "outer_fold": self.outer_fold,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PaperShortTermCommunityFrequency":
        if (
            payload.get("artifact_type")
            != PAPER_SHORT_TERM_COMMUNITY_FREQUENCY_SCHEMA
            or int(payload.get("schema_version", 0)) != 1
        ):
            raise ValueError("unsupported paper short-term frequency artifact")
        counts = tuple(
            sorted(
                (
                    (int(label), int(count))
                    for label, count in payload["counts"].items()
                ),
                key=lambda item: item[0],
            )
        )
        artifact = cls(
            counts=counts,
            total_count=int(payload["total_count"]),
            train_sample_count=int(payload["train_sample_count"]),
            train_window_count=int(payload["train_window_count"]),
            protocol_sha256=str(payload["protocol_sha256"]),
            train_manifest_sha256=str(payload["train_manifest_sha256"]),
            train_sample_keys_sha256=str(payload["train_sample_keys_sha256"]),
            epsilon=float(payload.get("epsilon", 1.0e-12)),
            outer_fold=(
                None
                if payload.get("outer_fold") is None
                else int(payload["outer_fold"])
            ),
        )
        recorded = {
            int(label): float(value)
            for label, value in payload.get("probabilities", {}).items()
        }
        expected = artifact.probability_mapping
        if set(recorded) != set(expected) or any(
            abs(recorded[label] - expected[label]) > 1.0e-12
            for label in expected
        ):
            raise ValueError("community-frequency probability mismatch")
        return artifact

    def save(self, path: Path, overwrite: bool = False) -> Path:
        path = Path(path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
        return path

    @classmethod
    def load(cls, path: Path) -> "PaperShortTermCommunityFrequency":
        with Path(path).resolve().open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def fit_paper_short_term_community_frequency(
    samples: Iterable[GraphSequenceSample],
    protocol_path: Path,
    train_manifest_path: Path,
    epsilon: float = 1.0e-12,
    outer_fold: Optional[int] = None,
) -> PaperShortTermCommunityFrequency:
    """Fit raw community-label frequencies from train samples only."""

    if epsilon <= 0.0:
        raise ValueError("community-frequency epsilon must be positive")
    counter: Counter = Counter()
    sample_keys = []
    sample_count = 0
    window_count = 0
    for sample in samples:
        if str(sample.split) != "train":
            raise ValueError("community frequency can only consume train samples")
        sample_count += 1
        sample_keys.append(str(sample.sample_key))
        for communities in sample.communities:
            window_count += 1
            valid = communities.detach().to(device="cpu", dtype=torch.long)
            valid = valid[valid >= 0]
            counter.update(int(value) for value in valid.tolist())
    total_count = int(sum(counter.values()))
    if sample_count <= 0 or window_count <= 0 or total_count <= 0:
        raise ValueError("community frequency requires valid train communities")
    return PaperShortTermCommunityFrequency(
        counts=tuple(sorted((int(key), int(value)) for key, value in counter.items())),
        total_count=total_count,
        train_sample_count=sample_count,
        train_window_count=window_count,
        protocol_sha256=file_sha256(protocol_path),
        train_manifest_sha256=file_sha256(train_manifest_path),
        train_sample_keys_sha256=_sample_keys_sha256(sample_keys),
        epsilon=float(epsilon),
        outer_fold=outer_fold,
    )


def compute_paper_short_term_pst_statistics(
    sample: GraphSequenceSample,
    community_frequency: PaperShortTermCommunityFrequency,
    probability_lookup: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the fixed-order six-dimensional raw p_ST vector."""

    if not sample.adjacency:
        raise ValueError("paper short-term sample contains no graph windows")
    reference = sample.adjacency[0]
    if probability_lookup is None:
        largest = max(
            (label for label, _ in community_frequency.counts),
            default=-1,
        )
        probability_lookup = community_frequency.probability_tensor(
            max(largest + 1, 1),
            reference=reference,
        )
    else:
        probability_lookup = probability_lookup.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    all_degrees = []
    anomaly_scores = []
    epsilon = reference.new_tensor(float(community_frequency.epsilon))
    for adjacency, communities in zip(sample.adjacency, sample.communities):
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("p_ST adjacency must be square")
        if communities.ndim != 1 or communities.shape[0] != adjacency.shape[0]:
            raise ValueError("p_ST community labels do not match adjacency")
        degree = adjacency.abs().sum(dim=1)
        if degree.numel() > 0:
            all_degrees.append(degree)
        labels = communities.to(device=adjacency.device, dtype=torch.long)
        valid = labels >= 0
        if bool(valid.any()):
            valid_labels = labels[valid]
            probabilities = adjacency.new_zeros(valid_labels.shape)
            known = valid_labels < int(probability_lookup.numel())
            if bool(known.any()):
                probabilities[known] = probability_lookup[
                    valid_labels[known]
                ]
            anomaly_scores.append(-(probabilities + epsilon).log().mean())

    if all_degrees:
        degrees = torch.cat(tuple(all_degrees), dim=0)
        degree_mean = degrees.mean()
        degree_std = degrees.std(unbiased=False)
        degree_max = degrees.max()
    else:
        degree_mean = reference.new_zeros(())
        degree_std = reference.new_zeros(())
        degree_max = reference.new_zeros(())
    if anomaly_scores:
        anomalies = torch.stack(tuple(anomaly_scores))
        anomaly_mean = anomalies.mean()
        anomaly_std = anomalies.std(unbiased=False)
    else:
        anomaly_mean = reference.new_zeros(())
        anomaly_std = reference.new_zeros(())
    return torch.stack(
        (
            degree_mean,
            degree_std,
            degree_max,
            anomaly_mean,
            anomaly_std,
            reference.new_tensor(float(len(sample.adjacency))),
        )
    )


def paper_short_term_pst_feature_schema(projection_dim: int) -> Dict[str, Any]:
    if projection_dim <= 0:
        raise ValueError("p_ST projection dimension must be positive")
    return {
        "name": PAPER_SHORT_TERM_PST_FEATURE_SCHEMA,
        "raw_dim": len(PST_RAW_STATISTIC_NAMES),
        "order": list(PST_RAW_STATISTIC_NAMES),
        "projection_dim": int(projection_dim),
        "community_frequency_scope": "inner_train_only",
        "degree_std_unbiased": False,
        "anomaly_std_unbiased": False,
    }

