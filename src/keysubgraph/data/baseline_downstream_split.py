"""Reproducible downstream partitions derived from an all-sample baseline manifest."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .baseline_manifest import (
    BaselineManifestRecord,
    read_baseline_manifest,
)
from .data_split import (
    SPLIT_NAMES,
    IndexSample,
    SplitAssignment,
    SplitConfig,
    create_data_splits,
    file_sha256,
    summarize_assignments,
)


def _portable_path(path: Path, project_root: Path) -> str:
    absolute = Path(os.path.abspath(str(path)))
    try:
        return absolute.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _group_id(record: BaselineManifestRecord) -> str:
    if record.subject_id:
        return "{}::{}".format(record.site, record.subject_id)
    return "sample::{}".format(record.sample_key)


def _as_index_sample(record: BaselineManifestRecord) -> IndexSample:
    return IndexSample(
        sample_key=record.sample_key,
        sample_id=record.sample_id,
        site=record.site,
        subject_id=record.subject_id,
        session_id=record.session_id,
        label=record.label,
        relative_path=record.relative_path,
        group_id=_group_id(record),
    )


def _derived_record(
    record: BaselineManifestRecord, downstream_split: str
) -> BaselineManifestRecord:
    values = asdict(record)
    values["source_split"] = record.source_split or record.split
    values["split"] = downstream_split
    return BaselineManifestRecord(**values)


def create_baseline_downstream_splits(
    project_root: Path,
    parent_manifest_path: Path,
    output_dir: Path,
    config: SplitConfig = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Freeze group-aware classifier partitions without changing hard subgraphs."""

    project_root = Path(project_root).resolve()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = config or SplitConfig()
    parent_payload, parent_records = read_baseline_manifest(
        parent_manifest_path, project_root, verify_exports=True
    )
    if parent_payload.get("split") != "all":
        raise ValueError("downstream partitions require a parent manifest with split='all'")
    if parent_payload.get("source_split", "all") != "all":
        raise ValueError("downstream parent manifest must originate from split='all'")
    if parent_payload.get("evidence_level") != "exploratory_in_sample":
        raise ValueError("all-sample downstream partitions must remain exploratory_in_sample")

    csv_path = output_dir / "baseline_splits.csv"
    json_path = output_dir / "baseline_splits.json"
    partition_paths = {
        split: output_dir / split / "baseline_manifest.json"
        for split in SPLIT_NAMES
    }
    all_outputs = [csv_path, json_path]
    for split in SPLIT_NAMES:
        all_outputs.extend(
            (
                partition_paths[split],
                output_dir / split / "baseline_manifest.csv",
                output_dir / split / "baseline_manifest_summary.json",
            )
        )
    if not overwrite and any(path.exists() for path in all_outputs):
        raise FileExistsError(
            "downstream split outputs already exist; reuse them or pass overwrite=True"
        )

    assignments = create_data_splits(
        [_as_index_sample(record) for record in parent_records], config
    )
    summary = summarize_assignments(assignments, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows = [assignment.to_dict() for assignment in assignments]
    _atomic_csv(
        csv_path,
        list(SplitAssignment.__dataclass_fields__),
        assignment_rows,
    )
    split_payload = {
        "schema_version": 1,
        "immutable": True,
        "purpose": "baseline_classifier_downstream_split",
        "evidence_level": "exploratory_in_sample",
        "parent_manifest": _portable_path(parent_manifest_path, project_root),
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "seed": config.seed,
        "ratios": dict(zip(SPLIT_NAMES, config.ratios)),
        "group_aware": True,
        "group_key": "site::subject_id (sample_key fallback when subject_id is empty)",
        "summary": summary,
        "assignments": assignment_rows,
    }
    _atomic_json(json_path, split_payload)
    csv_hash = file_sha256(csv_path)
    json_hash = file_sha256(json_path)
    assignment_by_key = {item.sample_key: item for item in assignments}
    parent_hash = file_sha256(parent_manifest_path)
    results = {}
    for split in SPLIT_NAMES:
        partition_dir = output_dir / split
        partition_dir.mkdir(parents=True, exist_ok=True)
        records = [
            _derived_record(record, split)
            for record in parent_records
            if assignment_by_key[record.sample_key].split == split
        ]
        if not records:
            raise RuntimeError("downstream partition is unexpectedly empty")
        manifest_payload = {
            "schema_version": 1,
            "immutable": True,
            "manifest_kind": "derived_downstream_partition",
            "evidence_level": "exploratory_in_sample",
            "project_root": parent_payload.get("project_root", "."),
            "data_protocol": parent_payload["data_protocol"],
            "data_protocol_sha256": parent_payload["data_protocol_sha256"],
            "split": split,
            "source_split": "all",
            "checkpoint_sha256": parent_payload["checkpoint_sha256"],
            "hard_extraction_config": parent_payload.get("hard_extraction_config"),
            "subgraph_source": parent_payload.get("subgraph_source", "key"),
            "matched_control_manifest": parent_payload.get(
                "matched_control_manifest", ""
            ),
            "matched_control_manifest_sha256": parent_payload.get(
                "matched_control_manifest_sha256", ""
            ),
            "parent_manifest": _portable_path(parent_manifest_path, project_root),
            "parent_manifest_sha256": parent_hash,
            "downstream_splits_csv": _portable_path(csv_path, project_root),
            "downstream_splits_csv_sha256": csv_hash,
            "downstream_splits_json": _portable_path(json_path, project_root),
            "downstream_splits_json_sha256": json_hash,
            "downstream_split_seed": config.seed,
            "sample_count": len(records),
            "timepoint_count": sum(record.timepoint_count for record in records),
            "subgraph_count": sum(record.subgraph_count for record in records),
            "records": [asdict(record) for record in records],
        }
        manifest_csv = partition_dir / "baseline_manifest.csv"
        manifest_json = partition_paths[split]
        manifest_summary = partition_dir / "baseline_manifest_summary.json"
        _atomic_csv(
            manifest_csv,
            list(asdict(records[0]).keys()),
            [asdict(record) for record in records],
        )
        _atomic_json(manifest_json, manifest_payload)
        _atomic_json(
            manifest_summary,
            {
                "schema_version": 1,
                "split": split,
                "source_split": "all",
                "sample_count": len(records),
                "timepoint_count": manifest_payload["timepoint_count"],
                "subgraph_count": manifest_payload["subgraph_count"],
                "class_counts": {
                    str(label): Counter(record.label for record in records)[label]
                    for label in (0, 1)
                },
                "group_count": len({_group_id(record) for record in records}),
                "parent_manifest_sha256": parent_hash,
                "downstream_splits_json_sha256": json_hash,
                "subgraph_source": parent_payload.get("subgraph_source", "key"),
                "matched_control_manifest_sha256": parent_payload.get(
                    "matched_control_manifest_sha256", ""
                ),
            },
        )
        results[split] = str(manifest_json)
    return {
        "output_dir": str(output_dir),
        "parent_manifest_sha256": parent_hash,
        "splits_csv": str(csv_path),
        "splits_json": str(json_path),
        "summary": summary,
        "manifests": results,
    }
