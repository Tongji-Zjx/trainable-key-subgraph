"""Materialize one frozen cross-fitting fold as a standard train/validation/test protocol."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from collections import Counter
from pathlib import Path

from keysubgraph.data.data_protocol import freeze_data_protocol
from keysubgraph.data.data_split import SplitAssignment, file_sha256, read_sample_index


ROLE_TO_SPLIT = {
    "inner_train": "train",
    "inner_validation": "validation",
    "outer_test": "test",
}


def _atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SplitAssignment.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    os.replace(str(temporary), str(path))


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def prepare_fold_protocol(
    project_root, fold_assignments_path, source_protocol_path, fold, output_root,
    overwrite=False,
):
    """Write a standard immutable protocol whose test partition is exactly one outer fold."""

    project_root = Path(project_root).resolve()
    fold_assignments_path = Path(fold_assignments_path).resolve()
    source_protocol_path = Path(source_protocol_path).resolve()
    output_dir = Path(output_root).resolve() / "fold_{}".format(int(fold)) / "protocol"
    split_csv = output_dir / "splits.csv"
    split_json = output_dir / "splits.json"
    protocol_json = output_dir / "data_protocol.json"
    existing = [path for path in (split_csv, split_json, protocol_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("fold protocol already exists")
    with source_protocol_path.open("r", encoding="utf-8") as handle:
        source_protocol = json.load(handle)
    with fold_assignments_path.open("r", encoding="utf-8") as handle:
        fold_payload = json.load(handle)
    if int(fold) < 0 or int(fold) >= int(fold_payload["num_outer_folds"]):
        raise ValueError("outer fold is out of range")
    source_index = project_root / source_protocol["paths"]["sample_index_csv"]
    samples = {sample.sample_key: sample for sample in read_sample_index(source_index)}
    selected = [
        row for row in fold_payload["assignments"] if int(row["outer_fold"]) == int(fold)
    ]
    if set(row["sample_key"] for row in selected) != set(samples):
        raise ValueError("fold assignment does not cover the source sample index")
    seed = int(fold_payload["seed"])
    assignments = []
    for row in selected:
        sample = samples[row["sample_key"]]
        split = ROLE_TO_SPLIT.get(str(row["role"]))
        if split is None:
            raise ValueError("unknown crossfit role")
        assignments.append(SplitAssignment(
            sample_key=sample.sample_key, sample_id=sample.sample_id,
            site=sample.site, subject_id=sample.subject_id,
            session_id=sample.session_id, group_id=sample.group_id,
            label=sample.label, relative_path=sample.relative_path,
            split=split, seed=seed,
        ))
    assignments.sort(key=lambda row: (("train", "validation", "test").index(row.split), row.sample_key))
    split_sets = {name: {row.sample_key for row in assignments if row.split == name} for name in ("train", "validation", "test")}
    subject_sets = {name: {row.group_id for row in assignments if row.split == name} for name in ("train", "validation", "test")}
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("fold protocol sample leakage")
    if any(subject_sets[left] & subject_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("fold protocol subject leakage")
    summary = {}
    for name in ("train", "validation", "test"):
        rows = [row for row in assignments if row.split == name]
        counts = Counter(row.label for row in rows)
        if not rows or set(counts) != {0, 1}:
            raise ValueError("fold protocol partition is empty or lacks a class")
        summary[name] = {
            "sample_count": len(rows), "subject_count": len({row.group_id for row in rows}),
            "class_counts": {str(label): counts[label] for label in (0, 1)},
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(split_csv, assignments)
    ratios = {name: len(split_sets[name]) / len(assignments) for name in split_sets}
    split_payload = {
        "schema_version": 1, "immutable": True,
        "assignment_mode": "confirmatory_cross_fitted",
        "source_index": source_index.resolve().relative_to(project_root).as_posix(),
        "source_index_sha256": file_sha256(source_index),
        "crossfit_assignments": fold_assignments_path.relative_to(project_root).as_posix(),
        "crossfit_assignments_sha256": file_sha256(fold_assignments_path),
        "outer_fold": int(fold), "seed": seed, "ratios": ratios,
        "group_aware": True, "group_key": "site::subject_id",
        "summary": summary, "assignments": [row.to_dict() for row in assignments],
    }
    _atomic_json(split_json, split_payload)
    protocol = freeze_data_protocol(
        project_root=project_root,
        dataset_root=project_root / source_protocol["paths"]["dataset_root"],
        sample_index_csv=source_index, splits_csv=split_csv, splits_json=split_json,
        output_path=protocol_json,
        edge_presence_threshold=float(source_protocol["edge_presence_threshold"]),
        overwrite=overwrite,
    )
    protocol["crossfit"] = {
        "outer_fold": int(fold), "fold_assignments_sha256": file_sha256(fold_assignments_path),
        "role_mapping": ROLE_TO_SPLIT,
    }
    _atomic_json(protocol_json, protocol)
    return {"output_dir": output_dir, "protocol": protocol_json, "summary": summary}
