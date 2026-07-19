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
    source_split: str = ""
    subgraph_source: str = "key"
    matched_control_manifest: str = ""
    matched_control_manifest_sha256: str = ""


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
    matched_control_manifest_path: Optional[Path] = None,
    subgraph_source: str = "key",
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
    matched_manifest_value = ""
    matched_manifest_hash = ""
    matched_source_records = None
    matched_experiment_kind = ""
    if matched_control_manifest_path is not None:
        matched_control_manifest_path = Path(matched_control_manifest_path).resolve()
        with matched_control_manifest_path.open("r", encoding="utf-8") as handle:
            matched_payload = json.load(handle)
        if (
            matched_payload.get("schema_version") != 1
            or matched_payload.get("purpose") != "baseline_matched_subgraph_sources"
            or not matched_payload.get("immutable")
        ):
            raise ValueError("unsupported matched-control manifest")
        partition_inventory = matched_payload.get("partition_inventories", {}).get(split)
        if matched_payload.get("split") != split and partition_inventory is None:
            raise ValueError("matched-control manifest split differs")
        if matched_payload.get("data_protocol_sha256") != protocol_sha256:
            raise ValueError("matched-control manifest protocol differs")
        if subgraph_source not in matched_payload.get("sources", []):
            raise ValueError("subgraph source is absent from matched-control manifest")
        matched_inventory = partition_inventory or matched_payload
        included_keys = set(matched_inventory.get("included_sample_keys", []))
        if not included_keys or not included_keys.issubset(assignment_by_key):
            raise ValueError("matched-control sample cohort is invalid")
        assignment_by_key = {
            key: assignment_by_key[key] for key in included_keys
        }
        matched_source_records = {
            str(item["sample_key"]): item
            for item in matched_inventory["source_records"][subgraph_source]
        }
        if set(matched_source_records) != included_keys:
            raise ValueError("matched-control source inventory differs from cohort")
        matched_manifest_value = _portable_path(
            matched_control_manifest_path, project_root
        )
        matched_manifest_hash = file_sha256(matched_control_manifest_path)
        matched_experiment_kind = str(matched_payload.get("experiment_kind", ""))
    elif subgraph_source != "key":
        raise ValueError("non-key source requires a matched-control manifest")

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
        if str(payload.get("subgraph_source", "key")) != subgraph_source:
            raise ValueError("hard export subgraph source differs")

    if set(payload_by_key) != set(assignment_by_key):
        missing = sorted(set(assignment_by_key) - set(payload_by_key))
        extra = sorted(set(payload_by_key) - set(assignment_by_key))
        raise ValueError(
            "export and split sample sets differ; missing={}, extra={}".format(
                missing[:1], extra[:1]
            )
        )
    if matched_source_records is not None:
        for sample_key, (path, unused_payload) in payload_by_key.items():
            del unused_payload
            expected = matched_source_records[sample_key]
            if file_sha256(path) != expected.get("sha256"):
                raise ValueError("hard export differs from matched-control inventory")
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
                source_split=assignment.split,
                subgraph_source=subgraph_source,
                matched_control_manifest=matched_manifest_value,
                matched_control_manifest_sha256=matched_manifest_hash,
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
        "source_split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "hard_extraction_config": extraction_config,
        "subgraph_source": subgraph_source,
        "matched_control_manifest": matched_manifest_value,
        "matched_control_manifest_sha256": matched_manifest_hash,
        "matched_control_experiment_kind": matched_experiment_kind,
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
        "subgraph_source": subgraph_source,
        "matched_control_manifest_sha256": matched_manifest_hash,
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
    record_payloads = []
    for item in payload.get("records", []):
        current = dict(item)
        current.setdefault("source_split", current.get("split", ""))
        current.setdefault("subgraph_source", payload.get("subgraph_source", "key"))
        current.setdefault(
            "matched_control_manifest", payload.get("matched_control_manifest", "")
        )
        current.setdefault(
            "matched_control_manifest_sha256",
            payload.get("matched_control_manifest_sha256", ""),
        )
        record_payloads.append(current)
    records = tuple(BaselineManifestRecord(**item) for item in record_payloads)
    if not records or len(records) != int(payload.get("sample_count", -1)):
        raise ValueError("baseline manifest record count mismatch")
    if len({record.sample_key for record in records}) != len(records):
        raise ValueError("baseline manifest contains duplicate samples")
    downstream_split = str(payload.get("split", ""))
    source_split = str(payload.get("source_split", downstream_split))
    if not downstream_split or not source_split:
        raise ValueError("baseline manifest split metadata is missing")
    if any(record.split != downstream_split for record in records):
        raise ValueError("baseline manifest record downstream splits differ")
    if any(record.source_split != source_split for record in records):
        raise ValueError("baseline manifest record source splits differ")
    if any(record.label not in (0, 1) for record in records):
        raise ValueError("baseline manifest contains a non-binary label")
    if any(
        record.data_protocol_sha256 != payload.get("data_protocol_sha256")
        for record in records
    ):
        raise ValueError("baseline manifest record protocol hashes differ")
    if any(
        record.checkpoint_sha256 != payload.get("checkpoint_sha256")
        for record in records
    ):
        raise ValueError("baseline manifest record checkpoint hashes differ")
    subgraph_source = str(payload.get("subgraph_source", "key"))
    if any(record.subgraph_source != subgraph_source for record in records):
        raise ValueError("baseline manifest record subgraph sources differ")
    matched_value = str(payload.get("matched_control_manifest", ""))
    matched_hash = str(payload.get("matched_control_manifest_sha256", ""))
    if bool(matched_value) != bool(matched_hash):
        raise ValueError("matched-control manifest metadata is incomplete")
    if any(
        record.matched_control_manifest != matched_value
        or record.matched_control_manifest_sha256 != matched_hash
        for record in records
    ):
        raise ValueError("baseline records use different matched-control manifests")
    if matched_value:
        matched_path = resolve_manifest_path(matched_value, project_root)
        if not matched_path.is_file() or file_sha256(matched_path) != matched_hash:
            raise ValueError("matched-control manifest hash mismatch")
        with matched_path.open("r", encoding="utf-8") as handle:
            matched_payload = json.load(handle)
        if matched_payload.get("purpose") != "baseline_matched_subgraph_sources":
            raise ValueError("matched-control manifest has the wrong purpose")
        if subgraph_source not in matched_payload.get("sources", []):
            raise ValueError("baseline subgraph source is absent from matching artifact")
        matched_inventory = matched_payload.get("partition_inventories", {}).get(
            source_split, matched_payload
        )
        if {record.sample_key for record in records} - set(
            matched_inventory.get("included_sample_keys", [])
        ):
            raise ValueError("baseline samples are absent from matched-control cohort")
    if "timepoint_count" in payload and sum(
        record.timepoint_count for record in records
    ) != int(payload["timepoint_count"]):
        raise ValueError("baseline manifest timepoint count mismatch")
    if "subgraph_count" in payload and sum(
        record.subgraph_count for record in records
    ) != int(payload["subgraph_count"]):
        raise ValueError("baseline manifest subgraph count mismatch")
    parent_value = payload.get("parent_manifest")
    parent_hash = payload.get("parent_manifest_sha256")
    if source_split != downstream_split:
        if not parent_value or not parent_hash:
            raise ValueError("derived baseline manifest parent metadata is missing")
        parent_path = resolve_manifest_path(parent_value, project_root)
        if not parent_path.is_file() or file_sha256(parent_path) != parent_hash:
            raise ValueError("derived baseline parent manifest hash mismatch")
        parent_payload, parent_records = read_baseline_manifest(
            parent_path, project_root, verify_exports=False
        )
        if parent_payload.get("split") != source_split:
            raise ValueError("derived baseline source split differs from parent")
        parent_by_key = {record.sample_key: record for record in parent_records}
        if not set(record.sample_key for record in records).issubset(parent_by_key):
            raise ValueError("derived baseline records are absent from parent")
        for record in records:
            parent_record = parent_by_key[record.sample_key]
            derived_values = asdict(record)
            derived_values["split"] = source_split
            derived_values["source_split"] = source_split
            parent_values = asdict(parent_record)
            parent_values["source_split"] = (
                parent_record.source_split or parent_record.split
            )
            if derived_values != parent_values:
                raise ValueError("derived baseline record differs from parent")
        for path_name, hash_name in (
            ("downstream_splits_csv", "downstream_splits_csv_sha256"),
            ("downstream_splits_json", "downstream_splits_json_sha256"),
        ):
            artifact_value = payload.get(path_name)
            artifact_hash = payload.get(hash_name)
            if not artifact_value or not artifact_hash:
                raise ValueError("derived baseline split artifact metadata is missing")
            artifact_path = resolve_manifest_path(artifact_value, project_root)
            if not artifact_path.is_file() or file_sha256(artifact_path) != artifact_hash:
                raise ValueError("derived baseline split artifact hash mismatch")
        splits_json_path = resolve_manifest_path(
            payload["downstream_splits_json"], project_root
        )
        with splits_json_path.open("r", encoding="utf-8") as handle:
            split_payload = json.load(handle)
        if split_payload.get("purpose") != "baseline_classifier_downstream_split":
            raise ValueError("derived baseline split artifact has the wrong purpose")
        assignments = split_payload.get("assignments", [])
        expected_keys = {
            str(item.get("sample_key"))
            for item in assignments
            if item.get("split") == downstream_split
        }
        if expected_keys != {record.sample_key for record in records}:
            raise ValueError("derived baseline records differ from split assignments")
    if verify_exports:
        for record in records:
            path = resolve_manifest_path(record.hard_subgraph_json, project_root)
            if not path.is_file() or file_sha256(path) != record.hard_subgraph_sha256:
                raise ValueError("hard-subgraph export hash mismatch: {}".format(record.sample_key))
    return payload, records
