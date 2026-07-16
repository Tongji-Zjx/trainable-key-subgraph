"""Immutable inventory for baseline-model hard-subgraph inputs."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .data_protocol import protocol_partitions, validate_data_protocol
from .data_split import file_sha256, read_split_assignments


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BaselineManifestRecord:
    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: int
    split: str
    relative_path: str
    hard_subgraph_json: str
    hard_subgraph_sha256: str
    timepoint_count: int
    subgraph_count: int
    checkpoint_sha256: str
    data_protocol_sha256: str
    edge_presence_threshold: float


def _portable_path(path: Path, project_root: Path) -> str:
    absolute = Path(os.path.abspath(str(path)))
    try:
        return absolute.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def resolve_manifest_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root).resolve() / path
    return path.resolve()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path: Path, records: List[BaselineManifestRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    os.replace(str(temporary), str(path))


def _read_export(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported hard-subgraph export: {}".format(path))
    return payload


def build_baseline_manifest(
    project_root: Path,
    protocol_path: Path,
    export_dir: Path,
    split: str,
    output_dir: Path,
    checkpoint_path: Optional[Path] = None,
    evidence_level: str = "exploratory_in_sample",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate one complete export partition and freeze its inventory."""

    project_root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_data_protocol(protocol_path, project_root)
    if split not in protocol_partitions(protocol):
        raise ValueError("split is not defined by the data protocol")
    protocol_sha256 = file_sha256(protocol_path)
    if evidence_level not in ("exploratory_in_sample", "confirmatory_cross_fitted"):
        raise ValueError("unsupported baseline evidence level")
    export_dir = Path(export_dir).resolve()
    partition_dir = export_dir / split
    if partition_dir.is_dir():
        export_dir = partition_dir
    if not export_dir.is_dir():
        raise FileNotFoundError(str(export_dir))

    output_dir = Path(output_dir).resolve()
    csv_path = output_dir / "baseline_manifest.csv"
    json_path = output_dir / "baseline_manifest.json"
    summary_path = output_dir / "baseline_manifest_summary.json"
    if not overwrite and any(path.exists() for path in (csv_path, json_path, summary_path)):
        raise FileExistsError("baseline manifest already exists")

    assignments = [
        item
        for item in read_split_assignments(
            project_root / protocol["paths"]["splits_csv"]
        )
        if item.split == split
    ]
    assignment_by_key = {item.sample_key: item for item in assignments}
    if not assignment_by_key:
        raise ValueError("requested split has no assignments")

    export_paths = sorted(export_dir.glob("*.json"))
    if not export_paths:
        raise ValueError("hard-subgraph export directory is empty")
    payload_by_key = {}
    extraction_config = None
    checkpoint_sha256 = None
    for path in export_paths:
        payload = _read_export(path)
        sample_key = str(payload.get("sample_key", ""))
        if not sample_key or sample_key in payload_by_key:
            raise ValueError("missing or duplicate sample_key in exports")
        payload_by_key[sample_key] = (path, payload)
        current_config = payload.get("hard_extraction_config")
        if extraction_config is None:
            extraction_config = current_config
        elif current_config != extraction_config:
            raise ValueError("hard extraction configuration differs across exports")
        current_checkpoint = str(payload.get("checkpoint_sha256", ""))
        if checkpoint_sha256 is None:
            checkpoint_sha256 = current_checkpoint
        elif current_checkpoint != checkpoint_sha256:
            raise ValueError("checkpoint hash differs across exports")

    if set(payload_by_key) != set(assignment_by_key):
        missing = sorted(set(assignment_by_key) - set(payload_by_key))
        extra = sorted(set(payload_by_key) - set(assignment_by_key))
        raise ValueError(
            "export and split sample sets differ; missing={}, extra={}".format(
                missing[:1], extra[:1]
            )
        )
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if file_sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError("checkpoint does not match hard-subgraph exports")

    records = []
    timepoint_total = 0
    subgraph_total = 0
    for sample_key in sorted(assignment_by_key):
        assignment = assignment_by_key[sample_key]
        path, payload = payload_by_key[sample_key]
        expected = (
            ("sample_id", assignment.sample_id),
            ("site", assignment.site),
            ("subject_id", assignment.subject_id),
            ("session_id", assignment.session_id),
            ("label", assignment.label),
            ("split", assignment.split),
            ("relative_path", assignment.relative_path),
        )
        for name, value in expected:
            if payload.get(name) != value:
                raise ValueError("export metadata mismatch for {}: {}".format(sample_key, name))
        if payload.get("data_protocol_sha256") != protocol_sha256:
            raise ValueError("export data protocol hash mismatch")
        if float(payload.get("edge_presence_threshold", -1.0)) != float(
            protocol["edge_presence_threshold"]
        ):
            raise ValueError("export edge threshold mismatch")
        timepoints = payload.get("timepoints")
        if not isinstance(timepoints, list) or not timepoints:
            raise ValueError("export contains no timepoints")
        expected_indices = list(range(len(timepoints)))
        if [item.get("time_index") for item in timepoints] != expected_indices:
            raise ValueError("export timepoints are not contiguous and ordered")
        counts = []
        for timepoint in timepoints:
            subgraphs = timepoint.get("subgraphs")
            if not isinstance(subgraphs, list) or not subgraphs:
                raise ValueError("effective timepoint contains no subgraphs")
            if int(timepoint.get("num_valid_subgraphs", -1)) != len(subgraphs):
                raise ValueError("num_valid_subgraphs does not match subgraphs")
            counts.append(len(subgraphs))
        subgraph_count = sum(counts)
        timepoint_total += len(timepoints)
        subgraph_total += subgraph_count
        records.append(
            BaselineManifestRecord(
                sample_key=sample_key,
                sample_id=assignment.sample_id,
                site=assignment.site,
                subject_id=assignment.subject_id,
                session_id=assignment.session_id,
                label=assignment.label,
                split=assignment.split,
                relative_path=assignment.relative_path,
                hard_subgraph_json=_portable_path(path, project_root),
                hard_subgraph_sha256=file_sha256(path),
                timepoint_count=len(timepoints),
                subgraph_count=subgraph_count,
                checkpoint_sha256=str(checkpoint_sha256),
                data_protocol_sha256=protocol_sha256,
                edge_presence_threshold=float(protocol["edge_presence_threshold"]),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "immutable": True,
        "evidence_level": evidence_level,
        "project_root": _portable_path(project_root, project_root),
        "data_protocol": _portable_path(protocol_path, project_root),
        "data_protocol_sha256": protocol_sha256,
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "hard_extraction_config": extraction_config,
        "sample_count": len(records),
        "timepoint_count": timepoint_total,
        "subgraph_count": subgraph_total,
        "records": [asdict(record) for record in records],
    }
    summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split": split,
        "sample_count": len(records),
        "timepoint_count": timepoint_total,
        "subgraph_count": subgraph_total,
        "class_counts": {
            str(label): sum(record.label == label for record in records)
            for label in (0, 1)
        },
        "site_count": len({record.site for record in records}),
        "checkpoint_sha256": checkpoint_sha256,
        "data_protocol_sha256": protocol_sha256,
    }
    _atomic_csv(csv_path, records)
    _atomic_json(json_path, payload)
    _atomic_json(summary_path, summary)
    return payload


def read_baseline_manifest(
    manifest_path: Path, project_root: Path, verify_exports: bool = True
) -> Tuple[Dict[str, Any], Tuple[BaselineManifestRecord, ...]]:
    """Read and optionally hash-check a frozen baseline manifest."""

    manifest_path = Path(manifest_path).resolve()
    project_root = Path(project_root).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION or not payload.get(
        "immutable"
    ):
        raise ValueError("unsupported or mutable baseline manifest")
    protocol_path = resolve_manifest_path(payload["data_protocol"], project_root)
    validate_data_protocol(protocol_path, project_root)
    if file_sha256(protocol_path) != payload.get("data_protocol_sha256"):
        raise ValueError("baseline manifest protocol hash mismatch")
    records = tuple(BaselineManifestRecord(**item) for item in payload.get("records", []))
    if not records or len(records) != int(payload.get("sample_count", -1)):
        raise ValueError("baseline manifest record count mismatch")
    if len({record.sample_key for record in records}) != len(records):
        raise ValueError("baseline manifest contains duplicate samples")
    if verify_exports:
        for record in records:
            path = resolve_manifest_path(record.hard_subgraph_json, project_root)
            if not path.is_file() or file_sha256(path) != record.hard_subgraph_sha256:
                raise ValueError("hard-subgraph export hash mismatch: {}".format(record.sample_key))
    return payload, records
