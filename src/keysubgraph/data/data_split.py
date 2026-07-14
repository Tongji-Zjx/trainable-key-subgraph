"""Reproducible stratified, group-aware train/validation/test splits."""

from __future__ import absolute_import, division, print_function

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Parameters that fully determine a split."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    search_attempts: int = 256
    max_class_ratio_deviation: float = 0.05

    def __post_init__(self) -> None:
        ratios = self.ratios
        if any(ratio <= 0.0 for ratio in ratios):
            raise ValueError("all split ratios must be positive")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")
        if self.search_attempts < 1:
            raise ValueError("search_attempts must be at least 1")
        if self.max_class_ratio_deviation < 0.0:
            raise ValueError("max_class_ratio_deviation must be non-negative")

    @property
    def ratios(self) -> Tuple[float, float, float]:
        return (self.train_ratio, self.validation_ratio, self.test_ratio)


@dataclass(frozen=True)
class IndexSample:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    relative_path: str
    group_id: str


@dataclass(frozen=True)
class SplitAssignment:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    group_id: str
    label: int
    relative_path: str
    split: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Group:
    group_id: str
    samples: Tuple[IndexSample, ...]
    class_counts: Tuple[int, int]

    @property
    def size(self) -> int:
        return len(self.samples)


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y")


def _group_id(row: Mapping[str, str]) -> str:
    subject_id = row.get("subject_id", "").strip()
    site = row.get("site", "").strip()
    if subject_id:
        return "{}::{}".format(site, subject_id)
    return "sample::{}".format(row["sample_key"].strip())


def read_sample_index(path: Path) -> List[IndexSample]:
    """Read and strictly validate an included-sample CSV index."""

    path = Path(path).resolve()
    required = {
        "sample_key",
        "sample_id",
        "site",
        "subject_id",
        "session_id",
        "label",
        "relative_path",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError("sample index is missing fields: {}".format(", ".join(missing)))
        rows = list(reader)

    samples: List[IndexSample] = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        if "included" in row and not _truthy(row["included"]):
            raise ValueError(
                "row {} is excluded; splits must use sample_index.csv only".format(row_number)
            )
        sample_key = row["sample_key"].strip()
        if not sample_key:
            raise ValueError("row {} has an empty sample_key".format(row_number))
        if sample_key in seen:
            raise ValueError("duplicate sample_key: {}".format(sample_key))
        seen.add(sample_key)
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            raise ValueError("row {} has a non-integer label".format(row_number))
        if label not in (0, 1):
            raise ValueError("row {} label must be 0 or 1".format(row_number))

        samples.append(
            IndexSample(
                sample_key=sample_key,
                sample_id=row["sample_id"].strip(),
                site=row["site"].strip(),
                subject_id=row["subject_id"].strip(),
                session_id=row["session_id"].strip(),
                label=label,
                relative_path=row["relative_path"].strip(),
                group_id=_group_id(row),
            )
        )
    if not samples:
        raise ValueError("sample index is empty")
    if set(sample.label for sample in samples) != {0, 1}:
        raise ValueError("binary stratification requires both labels 0 and 1")
    return sorted(samples, key=lambda sample: sample.sample_key)


def read_split_assignments(path: Path) -> List[SplitAssignment]:
    """Load an existing splits.csv without creating a new random split."""

    path = Path(path).resolve()
    required = set(SplitAssignment.__dataclass_fields__)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("splits CSV is missing fields: {}".format(", ".join(missing)))
        rows = list(reader)
    assignments = []
    for row_number, row in enumerate(rows, start=2):
        try:
            label = int(row["label"])
            seed = int(row["seed"])
        except (TypeError, ValueError):
            raise ValueError("row {} has an invalid label or seed".format(row_number))
        if label not in (0, 1):
            raise ValueError("row {} label must be 0 or 1".format(row_number))
        assignments.append(
            SplitAssignment(
                sample_key=row["sample_key"].strip(),
                sample_id=row["sample_id"].strip(),
                site=row["site"].strip(),
                subject_id=row["subject_id"].strip(),
                session_id=row["session_id"].strip(),
                group_id=row["group_id"].strip(),
                label=label,
                relative_path=row["relative_path"].strip(),
                split=row["split"].strip(),
                seed=seed,
            )
        )
    if len(set(item.seed for item in assignments)) > 1:
        raise ValueError("splits CSV contains multiple random seeds")
    return assignments


def _largest_remainder_targets(total: int, ratios: Sequence[float]) -> List[int]:
    raw = [total * ratio for ratio in ratios]
    targets = [int(value) for value in raw]
    remaining = total - sum(targets)
    order = sorted(range(len(ratios)), key=lambda index: (-(raw[index] - targets[index]), index))
    for index in order[:remaining]:
        targets[index] += 1
    return targets


def _make_groups(samples: Sequence[IndexSample]) -> List[_Group]:
    members: Dict[str, List[IndexSample]] = defaultdict(list)
    for sample in samples:
        members[sample.group_id].append(sample)
    groups = []
    for group_id, group_samples in sorted(members.items()):
        counts = Counter(sample.label for sample in group_samples)
        groups.append(
            _Group(
                group_id=group_id,
                samples=tuple(sorted(group_samples, key=lambda item: item.sample_key)),
                class_counts=(counts[0], counts[1]),
            )
        )
    return groups


def _allocation_score(
    totals: Sequence[int],
    classes: Sequence[Sequence[int]],
    target_totals: Sequence[int],
    target_classes: Sequence[Sequence[int]],
) -> float:
    score = 0.0
    for split_index in range(3):
        total_scale = max(target_totals[split_index], 1)
        total_error = (totals[split_index] - target_totals[split_index]) / total_scale
        score += total_error * total_error
        if totals[split_index] > target_totals[split_index]:
            score += total_error * total_error
        for label in (0, 1):
            class_scale = max(target_classes[label][split_index], 1)
            class_error = (
                classes[split_index][label] - target_classes[label][split_index]
            ) / class_scale
            score += class_error * class_error
            if classes[split_index][label] > target_classes[label][split_index]:
                score += class_error * class_error
    return score


def _one_allocation(
    groups: Sequence[_Group],
    config: SplitConfig,
    attempt: int,
    target_totals: Sequence[int],
    target_classes: Sequence[Sequence[int]],
) -> Tuple[Dict[str, int], float]:
    rng = random.Random(config.seed + attempt * 104729)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: -group.size)
    totals = [0, 0, 0]
    classes = [[0, 0], [0, 0], [0, 0]]
    allocation: Dict[str, int] = {}

    for group in ordered:
        candidates = []
        split_order = list(range(3))
        rng.shuffle(split_order)
        for split_index in split_order:
            next_totals = list(totals)
            next_classes = [list(item) for item in classes]
            next_totals[split_index] += group.size
            next_classes[split_index][0] += group.class_counts[0]
            next_classes[split_index][1] += group.class_counts[1]
            candidates.append(
                (
                    _allocation_score(
                        next_totals, next_classes, target_totals, target_classes
                    ),
                    split_index,
                    next_totals,
                    next_classes,
                )
            )
        _, selected, totals, classes = min(candidates, key=lambda item: item[0])
        allocation[group.group_id] = selected

    final_score = _allocation_score(totals, classes, target_totals, target_classes)
    for split_index in range(3):
        if totals[split_index] == 0:
            final_score += 1000.0
        for label in (0, 1):
            if classes[split_index][label] == 0:
                final_score += 100.0
    return allocation, final_score


def create_data_splits(
    samples: Sequence[IndexSample], config: Optional[SplitConfig] = None
) -> List[SplitAssignment]:
    """Create a deterministic stratified allocation without splitting groups."""

    config = config or SplitConfig()
    groups = _make_groups(samples)
    if len(groups) < 3:
        raise ValueError("at least three subject groups are required")
    class_totals = [sum(sample.label == label for sample in samples) for label in (0, 1)]
    if any(total < 3 for total in class_totals):
        raise ValueError("each class needs at least three samples")

    target_totals = _largest_remainder_targets(len(samples), config.ratios)
    target_classes = [
        _largest_remainder_targets(class_total, config.ratios)
        for class_total in class_totals
    ]
    best_allocation = None
    best_score = float("inf")
    for attempt in range(config.search_attempts):
        allocation, score = _one_allocation(
            groups, config, attempt, target_totals, target_classes
        )
        if score < best_score:
            best_allocation = allocation
            best_score = score
    assert best_allocation is not None

    assignments = []
    for sample in samples:
        assignments.append(
            SplitAssignment(
                sample_key=sample.sample_key,
                sample_id=sample.sample_id,
                site=sample.site,
                subject_id=sample.subject_id,
                session_id=sample.session_id,
                group_id=sample.group_id,
                label=sample.label,
                relative_path=sample.relative_path,
                split=SPLIT_NAMES[best_allocation[sample.group_id]],
                seed=config.seed,
            )
        )
    assignments.sort(key=lambda item: (SPLIT_NAMES.index(item.split), item.sample_key))
    validate_assignments(assignments, config)
    return assignments


def summarize_assignments(
    assignments: Sequence[SplitAssignment], config: SplitConfig
) -> Dict[str, Any]:
    overall_counts = Counter(item.label for item in assignments)
    overall_ratios = {
        str(label): overall_counts[label] / len(assignments) for label in (0, 1)
    }
    split_summaries = {}
    maximum_deviation = 0.0
    for split_name in SPLIT_NAMES:
        rows = [item for item in assignments if item.split == split_name]
        class_counts = Counter(item.label for item in rows)
        class_ratios = {
            str(label): class_counts[label] / len(rows) if rows else 0.0
            for label in (0, 1)
        }
        deviations = {
            str(label): abs(class_ratios[str(label)] - overall_ratios[str(label)])
            for label in (0, 1)
        }
        maximum_deviation = max(maximum_deviation, max(deviations.values()))
        split_summaries[split_name] = {
            "sample_count": len(rows),
            "sample_ratio": len(rows) / len(assignments),
            "class_counts": {str(label): class_counts[label] for label in (0, 1)},
            "class_ratios": class_ratios,
            "class_ratio_deviation_from_overall": deviations,
            "group_count": len(set(item.group_id for item in rows)),
        }

    sample_sets = {
        name: {item.sample_key for item in assignments if item.split == name}
        for name in SPLIT_NAMES
    }
    group_sets = {
        name: {item.group_id for item in assignments if item.split == name}
        for name in SPLIT_NAMES
    }
    sample_overlap = any(
        sample_sets[left] & sample_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    group_overlap = any(
        group_sets[left] & group_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    return {
        "total_samples": len(assignments),
        "total_groups": len(set(item.group_id for item in assignments)),
        "overall_class_counts": {
            str(label): overall_counts[label] for label in (0, 1)
        },
        "overall_class_ratios": overall_ratios,
        "splits": split_summaries,
        "checks": {
            "sample_overlap": bool(sample_overlap),
            "group_overlap": bool(group_overlap),
            "maximum_class_ratio_deviation": maximum_deviation,
            "allowed_class_ratio_deviation": config.max_class_ratio_deviation,
            "class_ratios_reasonable": maximum_deviation
            <= config.max_class_ratio_deviation,
        },
    }


def validate_assignments(
    assignments: Sequence[SplitAssignment], config: SplitConfig
) -> Dict[str, Any]:
    if not assignments:
        raise ValueError("split assignments are empty")
    keys = [item.sample_key for item in assignments]
    if len(keys) != len(set(keys)):
        raise ValueError("split assignments contain duplicate sample keys")
    if any(item.split not in SPLIT_NAMES for item in assignments):
        raise ValueError("unknown split name")
    summary = summarize_assignments(assignments, config)
    checks = summary["checks"]
    if checks["sample_overlap"]:
        raise ValueError("samples overlap across splits")
    if checks["group_overlap"]:
        raise ValueError("subject groups overlap across splits")
    if any(summary["splits"][name]["sample_count"] == 0 for name in SPLIT_NAMES):
        raise ValueError("every split must contain samples")
    if not checks["class_ratios_reasonable"]:
        raise ValueError(
            "class ratio deviation {:.6f} exceeds allowed {:.6f}".format(
                checks["maximum_class_ratio_deviation"],
                checks["allowed_class_ratio_deviation"],
            )
        )
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(path: Path, assignments: Sequence[SplitAssignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SplitAssignment.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(item.to_dict() for item in assignments)
    os.replace(str(temporary), str(path))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def write_split_artifacts(
    assignments: Sequence[SplitAssignment],
    output_dir: Path,
    source_index: Path,
    config: SplitConfig,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Write immutable-by-default split artifacts and their validation report."""

    output_dir = Path(output_dir).resolve()
    csv_path = output_dir / "splits.csv"
    json_path = output_dir / "splits.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "split artifacts already exist; reuse them or pass overwrite=True explicitly"
        )
    summary = validate_assignments(assignments, config)
    payload = {
        "schema_version": 1,
        "source_index": Path(source_index).name,
        "source_index_sha256": file_sha256(source_index),
        "seed": config.seed,
        "ratios": dict(zip(SPLIT_NAMES, config.ratios)),
        "group_aware": True,
        "group_key": "site::subject_id (sample_key fallback when subject_id is empty)",
        "summary": summary,
        "assignments": [item.to_dict() for item in assignments],
    }
    _atomic_write_csv(csv_path, assignments)
    _atomic_write_json(json_path, payload)
    return {"csv": csv_path, "json": json_path}
