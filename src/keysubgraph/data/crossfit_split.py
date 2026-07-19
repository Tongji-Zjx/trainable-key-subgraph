"""Immutable subject-grouped outer folds and per-fold inner partitions."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_split import IndexSample, file_sha256


CROSSFIT_SCHEMA_VERSION = 1


def _portable_reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass(frozen=True)
class OuterFoldAssignment:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    outer_test_fold: int


@dataclass(frozen=True)
class CrossfitFoldAssignment:
    outer_fold: int
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    role: str


@dataclass(frozen=True)
class _SubjectGroup:
    subject_id: str
    samples: Tuple[IndexSample, ...]
    class_counts: Mapping[int, int]

    @property
    def size(self) -> int:
        return len(self.samples)


def _subject_groups(samples: Sequence[IndexSample]) -> List[_SubjectGroup]:
    members = defaultdict(list)
    seen_keys = set()
    for sample in samples:
        if not sample.subject_id.strip():
            raise ValueError("cross-fitting requires a non-empty subject_id")
        if sample.sample_key in seen_keys:
            raise ValueError("cross-fitting samples contain duplicate sample keys")
        seen_keys.add(sample.sample_key)
        members[sample.subject_id.strip()].append(sample)
    groups = []
    for subject_id, current in sorted(members.items()):
        current = tuple(sorted(current, key=lambda item: item.sample_key))
        groups.append(
            _SubjectGroup(
                subject_id=subject_id,
                samples=current,
                class_counts=Counter(item.label for item in current),
            )
        )
    return groups


def _fold_score(
    totals: Sequence[int],
    classes: Sequence[Counter],
    target_total: float,
    target_classes: Mapping[int, float],
) -> float:
    score = 0.0
    for fold_index, total in enumerate(totals):
        total_error = (float(total) - target_total) / max(target_total, 1.0)
        score += total_error * total_error
        if total > target_total:
            score += 0.5 * total_error * total_error
        for label in (0, 1):
            target = max(float(target_classes[label]), 1.0)
            error = (float(classes[fold_index][label]) - target) / target
            score += 2.0 * error * error
            if classes[fold_index][label] > target_classes[label]:
                score += 0.5 * error * error
    return score


def _allocate_once(
    groups: Sequence[_SubjectGroup], num_folds: int, seed: int, attempt: int
) -> Tuple[Dict[str, int], float]:
    rng = random.Random(int(seed) + 104729 * int(attempt))
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda group: (-group.size, -max(group.class_counts.values()))
    )
    target_total = sum(group.size for group in groups) / float(num_folds)
    overall_classes = Counter()
    for group in groups:
        overall_classes.update(group.class_counts)
    target_classes = {
        label: overall_classes[label] / float(num_folds) for label in (0, 1)
    }
    totals = [0] * num_folds
    classes = [Counter() for _ in range(num_folds)]
    allocation = {}
    for group in ordered:
        candidates = list(range(num_folds))
        rng.shuffle(candidates)
        best_fold = None
        best_score = None
        for fold_index in candidates:
            next_totals = list(totals)
            next_classes = [Counter(item) for item in classes]
            next_totals[fold_index] += group.size
            next_classes[fold_index].update(group.class_counts)
            score = _fold_score(
                next_totals, next_classes, target_total, target_classes
            )
            if best_score is None or score < best_score:
                best_fold = fold_index
                best_score = score
        allocation[group.subject_id] = int(best_fold)
        totals[best_fold] += group.size
        classes[best_fold].update(group.class_counts)
    if any(total == 0 for total in totals):
        return allocation, float("inf")
    if any(classes[index][label] == 0 for index in range(num_folds) for label in (0, 1)):
        return allocation, float("inf")
    return allocation, _fold_score(totals, classes, target_total, target_classes)


def create_outer_folds(
    samples: Sequence[IndexSample],
    num_folds: int = 5,
    seed: int = 202607,
    attempts: int = 256,
) -> List[OuterFoldAssignment]:
    """Create deterministic stratified folds without splitting subjects."""

    if num_folds < 2:
        raise ValueError("cross-fitting requires at least two outer folds")
    if attempts < 1 or seed < 0:
        raise ValueError("invalid outer-fold search configuration")
    groups = _subject_groups(samples)
    if len(groups) < num_folds:
        raise ValueError("fewer subject groups than outer folds")
    overall = Counter(sample.label for sample in samples)
    if set(overall) != {0, 1} or min(overall.values()) < num_folds:
        raise ValueError("each class must support every outer fold")
    best_allocation = None
    best_score = None
    for attempt in range(attempts):
        allocation, score = _allocate_once(groups, num_folds, seed, attempt)
        if best_score is None or score < best_score:
            best_allocation = allocation
            best_score = score
    if best_allocation is None or best_score == float("inf"):
        raise RuntimeError("unable to construct non-empty stratified outer folds")
    assignments = [
        OuterFoldAssignment(
            sample_key=sample.sample_key,
            sample_id=sample.sample_id,
            site=sample.site,
            subject_id=sample.subject_id.strip(),
            session_id=sample.session_id,
            label=int(sample.label),
            outer_test_fold=int(best_allocation[sample.subject_id.strip()]),
        )
        for sample in samples
    ]
    assignments.sort(key=lambda item: (item.outer_test_fold, item.sample_key))
    validate_outer_folds(assignments, num_folds)
    return assignments


def summarize_outer_folds(
    assignments: Sequence[OuterFoldAssignment], num_folds: int
) -> Dict[str, Any]:
    total_classes = Counter(item.label for item in assignments)
    total_count = len(assignments)
    folds = {}
    subject_folds = defaultdict(set)
    seen_keys = set()
    for item in assignments:
        seen_keys.add(item.sample_key)
        subject_folds[item.subject_id].add(item.outer_test_fold)
    for fold_index in range(num_folds):
        rows = [item for item in assignments if item.outer_test_fold == fold_index]
        counts = Counter(item.label for item in rows)
        folds[str(fold_index)] = {
            "sample_count": len(rows),
            "subject_count": len({item.subject_id for item in rows}),
            "class_counts": {str(label): counts[label] for label in (0, 1)},
            "class_ratios": {
                str(label): counts[label] / float(len(rows)) if rows else 0.0
                for label in (0, 1)
            },
        }
    return {
        "total_samples": total_count,
        "total_subjects": len(subject_folds),
        "class_counts": {str(label): total_classes[label] for label in (0, 1)},
        "folds": folds,
        "checks": {
            "duplicate_sample_keys": len(seen_keys) != total_count,
            "subject_overlap": any(len(value) != 1 for value in subject_folds.values()),
            "all_folds_non_empty": all(folds[str(index)]["sample_count"] > 0 for index in range(num_folds)),
            "all_folds_have_both_classes": all(
                min(folds[str(index)]["class_counts"].values()) > 0
                for index in range(num_folds)
            ),
        },
    }


def validate_outer_folds(
    assignments: Sequence[OuterFoldAssignment], num_folds: int
) -> Dict[str, Any]:
    if not assignments:
        raise ValueError("outer-fold assignments are empty")
    if any(item.outer_test_fold < 0 or item.outer_test_fold >= num_folds for item in assignments):
        raise ValueError("outer-fold assignment contains an invalid fold")
    summary = summarize_outer_folds(assignments, num_folds)
    checks = summary["checks"]
    if checks["duplicate_sample_keys"]:
        raise ValueError("outer-fold assignments contain duplicate samples")
    if checks["subject_overlap"]:
        raise ValueError("a subject crosses outer-test folds")
    if not checks["all_folds_non_empty"]:
        raise ValueError("an outer-test fold is empty")
    if not checks["all_folds_have_both_classes"]:
        raise ValueError("an outer-test fold lacks a class")
    return summary


def _inner_allocate_once(
    groups: Sequence[_SubjectGroup], validation_ratio: float, seed: int, attempt: int
) -> Tuple[Dict[str, str], float]:
    rng = random.Random(int(seed) + 130363 * int(attempt))
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: (-group.size, -max(group.class_counts.values())))
    total_count = sum(group.size for group in groups)
    overall_classes = Counter()
    for group in groups:
        overall_classes.update(group.class_counts)
    target_totals = [
        total_count * (1.0 - validation_ratio), total_count * validation_ratio
    ]
    target_classes = {
        label: [
            overall_classes[label] * (1.0 - validation_ratio),
            overall_classes[label] * validation_ratio,
        ]
        for label in (0, 1)
    }
    totals = [0, 0]
    classes = [Counter(), Counter()]
    allocation = {}

    def score(candidate_totals, candidate_classes):
        value = 0.0
        for partition in range(2):
            total_target = max(target_totals[partition], 1.0)
            total_error = (
                candidate_totals[partition] - target_totals[partition]
            ) / total_target
            value += total_error * total_error
            if candidate_totals[partition] > target_totals[partition]:
                value += 0.5 * total_error * total_error
            for label in (0, 1):
                target = max(target_classes[label][partition], 1.0)
                error = (
                    candidate_classes[partition][label]
                    - target_classes[label][partition]
                ) / target
                value += 2.0 * error * error
                if candidate_classes[partition][label] > target_classes[label][partition]:
                    value += 0.5 * error * error
        return value

    for group in ordered:
        candidates = [0, 1]
        rng.shuffle(candidates)
        best_partition = None
        best_score = None
        for partition in candidates:
            next_totals = list(totals)
            next_classes = [Counter(item) for item in classes]
            next_totals[partition] += group.size
            next_classes[partition].update(group.class_counts)
            current_score = score(next_totals, next_classes)
            if best_score is None or current_score < best_score:
                best_partition = partition
                best_score = current_score
        totals[best_partition] += group.size
        classes[best_partition].update(group.class_counts)
        allocation[group.subject_id] = (
            "inner_train" if best_partition == 0 else "inner_validation"
        )
    if any(total == 0 for total in totals):
        return allocation, float("inf")
    if any(classes[index][label] == 0 for index in range(2) for label in (0, 1)):
        return allocation, float("inf")
    return allocation, score(totals, classes)


def create_crossfit_fold_assignments(
    samples: Sequence[IndexSample],
    outer_assignments: Sequence[OuterFoldAssignment],
    inner_validation_ratio: float = 0.1875,
    seed: int = 202608,
    attempts: int = 256,
) -> List[CrossfitFoldAssignment]:
    """Create one fixed inner train/validation split inside every outer-dev."""

    if inner_validation_ratio <= 0.0 or inner_validation_ratio >= 0.5:
        raise ValueError("inner validation ratio must be in (0, 0.5)")
    if attempts < 1 or seed < 0:
        raise ValueError("invalid inner split search configuration")
    sample_by_key = {sample.sample_key: sample for sample in samples}
    if len(sample_by_key) != len(samples):
        raise ValueError("crossfit samples contain duplicate sample keys")
    outer_by_key = {item.sample_key: item for item in outer_assignments}
    if set(sample_by_key) != set(outer_by_key):
        raise ValueError("outer assignments and samples differ")
    num_folds = max(item.outer_test_fold for item in outer_assignments) + 1
    validate_outer_folds(outer_assignments, num_folds)
    output = []
    for outer_fold in range(num_folds):
        dev_samples = [
            sample for sample in samples
            if outer_by_key[sample.sample_key].outer_test_fold != outer_fold
        ]
        groups = _subject_groups(dev_samples)
        best_allocation = None
        best_score = None
        for attempt in range(attempts):
            allocation, score = _inner_allocate_once(
                groups,
                inner_validation_ratio,
                seed + outer_fold * 1009,
                attempt,
            )
            if best_score is None or score < best_score:
                best_allocation = allocation
                best_score = score
        if best_allocation is None or best_score == float("inf"):
            raise RuntimeError("unable to construct an inner partition")
        for sample in samples:
            if outer_by_key[sample.sample_key].outer_test_fold == outer_fold:
                role = "outer_test"
            else:
                role = best_allocation[sample.subject_id.strip()]
            output.append(
                CrossfitFoldAssignment(
                    outer_fold=outer_fold,
                    sample_key=sample.sample_key,
                    sample_id=sample.sample_id,
                    site=sample.site,
                    subject_id=sample.subject_id.strip(),
                    session_id=sample.session_id,
                    label=int(sample.label),
                    role=role,
                )
            )
    output.sort(key=lambda item: (item.outer_fold, item.role, item.sample_key))
    validate_crossfit_fold_assignments(output, num_folds, len(samples))
    return output


def summarize_crossfit_fold_assignments(
    assignments: Sequence[CrossfitFoldAssignment], num_folds: int
) -> Dict[str, Any]:
    folds = {}
    allowed_roles = ("inner_train", "inner_validation", "outer_test")
    for outer_fold in range(num_folds):
        rows = [item for item in assignments if item.outer_fold == outer_fold]
        role_summary = {}
        for role in allowed_roles:
            current = [item for item in rows if item.role == role]
            counts = Counter(item.label for item in current)
            role_summary[role] = {
                "sample_count": len(current),
                "subject_count": len({item.subject_id for item in current}),
                "class_counts": {str(label): counts[label] for label in (0, 1)},
                "class_ratios": {
                    str(label): counts[label] / float(len(current)) if current else 0.0
                    for label in (0, 1)
                },
            }
        subject_roles = defaultdict(set)
        sample_roles = defaultdict(set)
        for item in rows:
            subject_roles[item.subject_id].add(item.role)
            sample_roles[item.sample_key].add(item.role)
        folds[str(outer_fold)] = {
            "roles": role_summary,
            "checks": {
                "sample_role_overlap": any(len(value) != 1 for value in sample_roles.values()),
                "subject_role_overlap": any(len(value) != 1 for value in subject_roles.values()),
                "all_roles_non_empty": all(role_summary[role]["sample_count"] > 0 for role in allowed_roles),
                "all_roles_have_both_classes": all(
                    min(role_summary[role]["class_counts"].values()) > 0
                    for role in allowed_roles
                ),
            },
        }
    return {"num_outer_folds": num_folds, "folds": folds}


def validate_crossfit_fold_assignments(
    assignments: Sequence[CrossfitFoldAssignment],
    num_folds: int,
    expected_sample_count: Optional[int] = None,
) -> Dict[str, Any]:
    if not assignments:
        raise ValueError("crossfit fold assignments are empty")
    allowed_roles = {"inner_train", "inner_validation", "outer_test"}
    if any(item.role not in allowed_roles for item in assignments):
        raise ValueError("crossfit assignment contains an invalid role")
    if any(item.outer_fold < 0 or item.outer_fold >= num_folds for item in assignments):
        raise ValueError("crossfit assignment contains an invalid outer fold")
    summary = summarize_crossfit_fold_assignments(assignments, num_folds)
    all_outer_test_folds = defaultdict(list)
    for item in assignments:
        if item.role == "outer_test":
            all_outer_test_folds[item.sample_key].append(item.outer_fold)
    if expected_sample_count is not None and len(all_outer_test_folds) != expected_sample_count:
        raise ValueError("outer-test assignments do not cover every sample")
    if any(len(value) != 1 for value in all_outer_test_folds.values()):
        raise ValueError("a sample appears in multiple outer-test folds")
    for fold in summary["folds"].values():
        checks = fold["checks"]
        if checks["sample_role_overlap"]:
            raise ValueError("a sample crosses roles within an outer fold")
        if checks["subject_role_overlap"]:
            raise ValueError("a subject crosses roles within an outer fold")
        if not checks["all_roles_non_empty"]:
            raise ValueError("a crossfit role is empty")
        if not checks["all_roles_have_both_classes"]:
            raise ValueError("a crossfit role lacks a class")
    return summary


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, assignments: Sequence[OuterFoldAssignment]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(asdict(assignments[0]).keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in assignments:
            writer.writerow(asdict(item))
    os.replace(str(temporary), str(path))


def _atomic_crossfit_csv(
    path: Path, assignments: Sequence[CrossfitFoldAssignment]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(asdict(assignments[0]).keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in assignments:
            writer.writerow(asdict(item))
    os.replace(str(temporary), str(path))


def write_outer_fold_artifacts(
    assignments: Sequence[OuterFoldAssignment],
    output_dir: Path,
    source_index_path: Path,
    num_folds: int = 5,
    seed: int = 202607,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Write immutable outer_splits.csv/json bound to the sample index."""

    output_dir = Path(output_dir).resolve()
    source_index_path = Path(source_index_path).resolve()
    csv_path = output_dir / "outer_splits.csv"
    json_path = output_dir / "outer_splits.json"
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError("outer split artifacts already exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_outer_folds(assignments, num_folds)
    _atomic_csv(csv_path, assignments)
    payload = {
        "schema_version": CROSSFIT_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "confirmatory_cross_fitted_outer_split",
        "group_key": "subject_id",
        "num_outer_folds": int(num_folds),
        "seed": int(seed),
        "source_index": _portable_reference(source_index_path),
        "source_index_sha256": file_sha256(source_index_path),
        "assignments": [asdict(item) for item in assignments],
        "summary": summary,
    }
    _atomic_json(json_path, payload)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "csv_sha256": file_sha256(csv_path),
        "json_sha256": file_sha256(json_path),
    }


def read_outer_fold_artifacts(
    json_path: Path, source_index_path: Optional[Path] = None
) -> Tuple[Dict[str, Any], List[OuterFoldAssignment]]:
    json_path = Path(json_path).resolve()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version") != CROSSFIT_SCHEMA_VERSION
        or not payload.get("immutable")
        or payload.get("purpose") != "confirmatory_cross_fitted_outer_split"
        or payload.get("group_key") != "subject_id"
    ):
        raise ValueError("unsupported outer split artifact")
    if source_index_path is not None and payload.get("source_index_sha256") != file_sha256(Path(source_index_path)):
        raise ValueError("outer split source index hash mismatch")
    assignments = [OuterFoldAssignment(**item) for item in payload["assignments"]]
    summary = validate_outer_folds(assignments, int(payload["num_outer_folds"]))
    if summary != payload.get("summary"):
        raise ValueError("outer split summary differs from assignments")
    return payload, assignments


def write_crossfit_fold_artifacts(
    assignments: Sequence[CrossfitFoldAssignment],
    output_dir: Path,
    outer_json_path: Path,
    source_index_path: Path,
    inner_validation_ratio: float = 0.1875,
    seed: int = 202608,
    overwrite: bool = False,
) -> Dict[str, str]:
    output_dir = Path(output_dir).resolve()
    outer_json_path = Path(outer_json_path).resolve()
    source_index_path = Path(source_index_path).resolve()
    csv_path = output_dir / "fold_assignments.csv"
    json_path = output_dir / "fold_assignments.json"
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError("crossfit fold artifacts already exist")
    num_folds = max(item.outer_fold for item in assignments) + 1
    summary = validate_crossfit_fold_assignments(assignments, num_folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_crossfit_csv(csv_path, assignments)
    payload = {
        "schema_version": CROSSFIT_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "confirmatory_cross_fitted_fold_roles",
        "group_key": "subject_id",
        "num_outer_folds": num_folds,
        "inner_validation_ratio": float(inner_validation_ratio),
        "seed": int(seed),
        "source_index_sha256": file_sha256(source_index_path),
        "outer_splits_json": _portable_reference(outer_json_path),
        "outer_splits_json_sha256": file_sha256(outer_json_path),
        "assignments": [asdict(item) for item in assignments],
        "summary": summary,
    }
    _atomic_json(json_path, payload)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "csv_sha256": file_sha256(csv_path),
        "json_sha256": file_sha256(json_path),
    }


def read_crossfit_fold_artifacts(
    json_path: Path,
    outer_json_path: Optional[Path] = None,
    source_index_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[CrossfitFoldAssignment]]:
    json_path = Path(json_path).resolve()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version") != CROSSFIT_SCHEMA_VERSION
        or not payload.get("immutable")
        or payload.get("purpose") != "confirmatory_cross_fitted_fold_roles"
        or payload.get("group_key") != "subject_id"
    ):
        raise ValueError("unsupported crossfit fold artifact")
    if outer_json_path is not None and payload.get("outer_splits_json_sha256") != file_sha256(Path(outer_json_path)):
        raise ValueError("crossfit outer split hash mismatch")
    if source_index_path is not None and payload.get("source_index_sha256") != file_sha256(Path(source_index_path)):
        raise ValueError("crossfit source index hash mismatch")
    assignments = [CrossfitFoldAssignment(**item) for item in payload["assignments"]]
    summary = validate_crossfit_fold_assignments(
        assignments, int(payload["num_outer_folds"])
    )
    if summary != payload.get("summary"):
        raise ValueError("crossfit fold summary differs from assignments")
    return payload, assignments
