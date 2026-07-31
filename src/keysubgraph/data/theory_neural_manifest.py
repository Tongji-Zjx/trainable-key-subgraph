"""Immutable manifests for Stage-1 theory-guided neural records."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from pathlib import Path

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.theory_neural_artifact import load_theory_neural_record


def theory_neural_filename(sample_key):
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def write_theory_neural_manifest(paths, output_path, project_root, overwrite=False):
    output_path = Path(output_path).resolve()
    project_root = Path(project_root).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("Stage-1 manifest already exists")
    rows = []
    provenance = None
    split = None
    for path in paths:
        path = Path(path).resolve()
        record = load_theory_neural_record(path)
        current = (
            record.protocol_sha256,
            record.selector_checkpoint_sha256,
            record.selection_mode,
            int(record.selection_seed),
            record.feature_schema_sha256,
        )
        if provenance is None:
            provenance = current
            split = record.split
        if current != provenance or record.split != split:
            raise ValueError("Stage-1 records have incompatible provenance")
        rows.append(
            {
                "sample_key": record.sample_key,
                "sample_id": record.sample_id,
                "subject_id": record.subject_id,
                "site": record.site,
                "label": int(record.label),
                "feature_path": path.relative_to(project_root).as_posix(),
                "feature_sha256": file_sha256(path),
            }
        )
    if not rows or len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("Stage-1 manifest is empty or duplicated")
    rows.sort(key=lambda row: row["sample_key"])
    payload = {
        "schema_version": 1,
        "artifact_type": "svg_theory_guided_neural_manifest",
        "split": split,
        "sample_count": len(rows),
        "protocol_sha256": provenance[0],
        "selector_checkpoint_sha256": provenance[1],
        "selection_mode": provenance[2],
        "selection_seed": provenance[3],
        "feature_schema_sha256": provenance[4],
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def read_theory_neural_manifest(path, project_root):
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != "svg_theory_guided_neural_manifest":
        raise ValueError("unsupported Stage-1 manifest")
    records = []
    for row in payload.get("records", []):
        feature_path = project_root / row["feature_path"]
        if file_sha256(feature_path) != row["feature_sha256"]:
            raise ValueError("Stage-1 feature hash mismatch")
        record = load_theory_neural_record(feature_path)
        checks = (
            record.sample_key == row["sample_key"],
            int(record.label) == int(row["label"]),
            record.split == payload["split"],
            record.protocol_sha256 == payload["protocol_sha256"],
            record.selector_checkpoint_sha256
            == payload["selector_checkpoint_sha256"],
            record.feature_schema_sha256 == payload["feature_schema_sha256"],
        )
        if not all(checks):
            raise ValueError("Stage-1 manifest record mismatch")
        records.append(record)
    if len(records) != int(payload.get("sample_count", -1)):
        raise ValueError("Stage-1 manifest count mismatch")
    return payload, records
