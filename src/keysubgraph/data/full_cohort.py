"""Immutable assignments for exploratory training on every indexed sample."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .data_split import IndexSample, SplitAssignment, file_sha256


FULL_COHORT_MODE = "all_samples_exploratory"


def create_full_cohort_assignments(
    samples: Sequence[IndexSample], seed: int = 42
) -> List[SplitAssignment]:
    """Assign every indexed sample to the explicit ``all`` partition."""

    if not samples:
        raise ValueError("sample index is empty")
    if len({sample.sample_key for sample in samples}) != len(samples):
        raise ValueError("sample index contains duplicate sample keys")
    assignments = [
        SplitAssignment(
            sample_key=sample.sample_key,
            sample_id=sample.sample_id,
            site=sample.site,
            subject_id=sample.subject_id,
            session_id=sample.session_id,
            group_id=sample.group_id,
            label=sample.label,
            relative_path=sample.relative_path,
            split="all",
            seed=int(seed),
        )
        for sample in samples
    ]
    assignments.sort(key=lambda item: item.sample_key)
    return assignments


def _atomic_csv(path: Path, assignments: Sequence[SplitAssignment]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(SplitAssignment.__dataclass_fields__.keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in assignments)
    os.replace(str(temporary), str(path))


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def write_full_cohort_artifacts(
    assignments: Sequence[SplitAssignment],
    source_index: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Write immutable CSV/JSON artifacts describing the complete cohort."""

    assignments = list(assignments)
    if not assignments or any(item.split != "all" for item in assignments):
        raise ValueError("full-cohort assignments must all use split='all'")
    if len({item.sample_key for item in assignments}) != len(assignments):
        raise ValueError("full-cohort assignments contain duplicate samples")
    if len({item.seed for item in assignments}) != 1:
        raise ValueError("full-cohort assignments must use one seed")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "splits.csv"
    json_path = output_dir / "splits.json"
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(
            "full-cohort artifacts already exist; reuse them or explicitly overwrite"
        )

    class_counts = Counter(item.label for item in assignments)
    payload = {
        "assignment_mode": FULL_COHORT_MODE,
        "source_index_sha256": file_sha256(source_index),
        "seed": assignments[0].seed,
        "ratios": {"all": 1.0},
        "group_key": "not_applicable_all_samples_are_in_one_cohort",
        "sample_count": len(assignments),
        "class_counts": {str(label): class_counts[label] for label in (0, 1)},
        "assignments": [item.to_dict() for item in assignments],
    }
    _atomic_csv(csv_path, assignments)
    _atomic_json(json_path, payload)
    return {"csv": csv_path, "json": json_path}
