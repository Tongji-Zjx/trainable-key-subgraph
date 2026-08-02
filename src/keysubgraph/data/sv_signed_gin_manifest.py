"""Immutable manifests for SV Signed-GIN hard-graph records."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from keysubgraph.data.data_split import file_sha256
from .sv_signed_gin_artifact import (
    SVSignedGINRecord,
    load_sv_signed_gin_record,
)


SV_SIGNED_GIN_MANIFEST_SCHEMA_VERSION = 2
SV_SIGNED_GIN_MANIFEST_SUPPORTED_SCHEMA_VERSIONS = (1, 2)


def sv_signed_gin_filename(sample_key: str) -> str:
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def _provenance(record: SVSignedGINRecord) -> Dict:
    return {
        "protocol_sha256": record.protocol_sha256,
        "selector_checkpoint_sha256": record.selector_checkpoint_sha256,
        "selection_mode": record.selection_mode,
        "selection_seed": int(record.selection_seed),
        "node_ratio": float(getattr(record, "node_ratio", 0.50)),
        "edge_ratio": float(getattr(record, "edge_ratio", 0.30)),
    }


def write_sv_signed_gin_manifest(
    records: List[Tuple[SVSignedGINRecord, Path]],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("SV Signed-GIN manifest already exists")
    if not records:
        raise ValueError("cannot write an empty SV Signed-GIN manifest")
    keys = [record.sample_key for record, _ in records]
    if len(set(keys)) != len(keys):
        raise ValueError("SV Signed-GIN manifest contains duplicate samples")
    splits = {record.split for record, _ in records}
    provenance = {
        json.dumps(_provenance(record), sort_keys=True)
        for record, _ in records
    }
    if len(splits) != 1 or len(provenance) != 1:
        raise ValueError("SV Signed-GIN manifest mixes split/provenance")
    rows = []
    for record, path in sorted(
        records, key=lambda item: item[0].sample_key
    ):
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(
                output_path.parent
            ).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        rows.append(
            {
                "sample_key": record.sample_key,
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "site": record.site,
                "label": int(record.label),
                "split": record.split,
                "valid_window_count": record.valid_window_count,
                "valid_transition_count": record.valid_transition_count,
                "feature_path": relative,
                "feature_sha256": file_sha256(resolved),
            }
        )
    common = _provenance(records[0][0])
    payload = {
        "schema_version": SV_SIGNED_GIN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "sv_hard_sgw_signed_gin_manifest",
        "sample_count": len(rows),
        "split": next(iter(splits)),
        "records": rows,
        **common
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def write_sv_signed_gin_manifest_from_paths(
    feature_paths: Sequence[Path],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Write a manifest while retaining at most one feature record in RAM."""
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("SV Signed-GIN manifest already exists")
    resolved_paths = [Path(path).resolve() for path in feature_paths]
    if not resolved_paths:
        raise ValueError("cannot write an empty SV Signed-GIN manifest")

    rows = []
    keys = set()
    split = None
    common = None
    for resolved in resolved_paths:
        record = load_sv_signed_gin_record(resolved)
        record_provenance = _provenance(record)
        if split is None:
            split = record.split
            common = record_provenance
        elif record.split != split or record_provenance != common:
            raise ValueError("SV Signed-GIN manifest mixes split/provenance")
        if record.sample_key in keys:
            raise ValueError(
                "SV Signed-GIN manifest contains duplicate samples"
            )
        keys.add(record.sample_key)
        try:
            relative = resolved.relative_to(
                output_path.parent
            ).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        rows.append(
            {
                "sample_key": record.sample_key,
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "site": record.site,
                "label": int(record.label),
                "split": record.split,
                "valid_window_count": record.valid_window_count,
                "valid_transition_count": (
                    record.valid_transition_count
                ),
                "feature_path": relative,
                "feature_sha256": file_sha256(resolved),
            }
        )
        del record

    rows.sort(key=lambda row: row["sample_key"])
    payload = {
        "schema_version": SV_SIGNED_GIN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "sv_hard_sgw_signed_gin_manifest",
        "sample_count": len(rows),
        "split": split,
        "records": rows,
        **common
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def read_sv_signed_gin_manifest(
    path: Path,
) -> Tuple[Dict, List[SVSignedGINRecord]]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") not in (
        SV_SIGNED_GIN_MANIFEST_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError("unsupported SV Signed-GIN manifest schema")
    if payload.get("artifact_type") != (
        "sv_hard_sgw_signed_gin_manifest"
    ):
        raise ValueError("unexpected SV Signed-GIN manifest")
    rows = payload.get("records", [])
    if len(rows) != int(payload.get("sample_count", -1)):
        raise ValueError("SV Signed-GIN manifest count mismatch")
    records = []
    seen = set()
    for row in rows:
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if file_sha256(feature_path) != row["feature_sha256"]:
            raise ValueError("SV Signed-GIN artifact hash mismatch")
        record = load_sv_signed_gin_record(feature_path)
        payload_provenance = {
            "protocol_sha256": payload["protocol_sha256"],
            "selector_checkpoint_sha256": payload[
                "selector_checkpoint_sha256"
            ],
            "selection_mode": payload["selection_mode"],
            "selection_seed": int(payload["selection_seed"]),
            "node_ratio": float(payload.get("node_ratio", 0.50)),
            "edge_ratio": float(payload.get("edge_ratio", 0.30)),
        }
        checks = (
            record.sample_key == row["sample_key"],
            record.sample_id == row["sample_id"],
            record.subject_id == row["subject_id"],
            record.site == row["site"],
            int(record.label) == int(row["label"]),
            record.split == row["split"] == payload["split"],
            record.valid_window_count
            == int(row["valid_window_count"]),
            record.valid_transition_count
            == int(row["valid_transition_count"]),
            _provenance(record) == payload_provenance,
        )
        if not all(checks):
            raise ValueError("SV Signed-GIN manifest record mismatch")
        if record.sample_key in seen:
            raise ValueError("SV Signed-GIN manifest duplicate key")
        seen.add(record.sample_key)
        records.append(record)
    return payload, records
