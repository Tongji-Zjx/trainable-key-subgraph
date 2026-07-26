"""Immutable manifests for D3-B temporal variation records."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from keysubgraph.data.data_split import file_sha256
from .dual_temporal_artifact import (
    DualTemporalVariationRecord,
    load_dual_temporal_record,
)


DUAL_TEMPORAL_MANIFEST_SCHEMA_VERSION = 1


def dual_temporal_filename(sample_key: str) -> str:
    return hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest() + ".pt"


def _provenance(record: DualTemporalVariationRecord) -> Dict:
    return {
        "protocol_sha256": record.protocol_sha256,
        "selector_checkpoint_sha256": (
            record.selector_checkpoint_sha256
        ),
        "exact_head_checkpoint_sha256": (
            record.exact_head_checkpoint_sha256
        ),
        "sgw_scaler_sha256": record.sgw_scaler_sha256,
        "exact_manifest_sha256": record.exact_manifest_sha256,
        "selection_mode": record.selection_mode,
        "selection_seed": int(record.selection_seed),
    }


def write_dual_temporal_manifest(
    records: List[Tuple[DualTemporalVariationRecord, Path]],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("dual temporal manifest already exists")
    if not records:
        raise ValueError("cannot write an empty dual temporal manifest")
    keys = [record.sample_key for record, _ in records]
    if len(set(keys)) != len(keys):
        raise ValueError("dual temporal manifest contains duplicate samples")
    splits = {record.split for record, _ in records}
    provenance = {
        json.dumps(_provenance(record), sort_keys=True)
        for record, _ in records
    }
    if len(splits) != 1 or len(provenance) != 1:
        raise ValueError("dual temporal manifest mixes split/provenance")
    common = _provenance(records[0][0])
    rows = []
    for record, path in sorted(
        records, key=lambda item: item[0].sample_key
    ):
        path = Path(path).resolve()
        try:
            relative = path.relative_to(output_path.parent).as_posix()
        except ValueError:
            relative = path.as_posix()
        rows.append(
            {
                "sample_key": record.sample_key,
                "label": int(record.label),
                "split": record.split,
                "window_count": int(record.window_count),
                "valid_transition_count": (
                    record.valid_transition_count
                ),
                "feature_path": relative,
                "feature_sha256": file_sha256(path),
            }
        )
    payload = {
        "schema_version": DUAL_TEMPORAL_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dual_d3b_temporal_variation_manifest",
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


def read_dual_temporal_manifest(
    path: Path,
) -> Tuple[Dict, List[DualTemporalVariationRecord]]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != (
        DUAL_TEMPORAL_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported dual temporal manifest schema")
    if payload.get("artifact_type") != (
        "dual_d3b_temporal_variation_manifest"
    ):
        raise ValueError("unexpected dual temporal manifest")
    rows = payload.get("records", [])
    if len(rows) != int(payload.get("sample_count", -1)):
        raise ValueError("dual temporal manifest count mismatch")
    records = []
    seen = set()
    provenance_keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "exact_head_checkpoint_sha256",
        "sgw_scaler_sha256",
        "exact_manifest_sha256",
        "selection_mode",
        "selection_seed",
    )
    for row in rows:
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if file_sha256(feature_path) != row["feature_sha256"]:
            raise ValueError("dual temporal artifact hash mismatch")
        record = load_dual_temporal_record(feature_path)
        if (
            record.sample_key != row["sample_key"]
            or int(record.label) != int(row["label"])
            or record.split != row["split"]
            or int(record.window_count) != int(row["window_count"])
            or record.valid_transition_count
            != int(row["valid_transition_count"])
        ):
            raise ValueError("dual temporal manifest record mismatch")
        record_provenance = _provenance(record)
        if any(
            record_provenance[key] != payload[key]
            for key in provenance_keys
        ):
            raise ValueError("dual temporal manifest provenance mismatch")
        if record.sample_key in seen:
            raise ValueError("dual temporal manifest duplicate key")
        seen.add(record.sample_key)
        records.append(record)
    return payload, records
