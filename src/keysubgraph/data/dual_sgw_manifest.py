"""Manifest utilities for cached exact Dual-STSE SGW features."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from keysubgraph.models.dual_exact_sgw import (
    DualSGWFeatureRecord,
    load_dual_sgw_feature_record,
)


DUAL_SGW_MANIFEST_SCHEMA_VERSION = 1


def dual_feature_filename(sample_key: str) -> str:
    digest = hashlib.sha256(
        str(sample_key).encode("utf-8")
    ).hexdigest()
    return digest + ".pt"


def write_dual_sgw_manifest(
    records: List[Tuple[DualSGWFeatureRecord, Path]],
    output_path: Path,
    protocol_sha256: str,
    selector_checkpoint_sha256: str,
    selection_mode: str,
    selection_seed: int,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError("dual SGW manifest already exists")
    if not records:
        raise ValueError("cannot write an empty dual SGW manifest")
    keys = [record.sample_key for record, _ in records]
    if len(set(keys)) != len(keys):
        raise ValueError("dual SGW manifest contains duplicate samples")
    split_values = {record.split for record, _ in records}
    if len(split_values) != 1:
        raise ValueError("one dual SGW manifest must contain one split")
    rows = []
    for record, path in sorted(records, key=lambda item: item[0].sample_key):
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
                "feature_path": relative,
            }
        )
    payload = {
        "schema_version": DUAL_SGW_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dual_stse_exact_sgw_manifest",
        "sample_count": len(rows),
        "split": next(iter(split_values)),
        "protocol_sha256": str(protocol_sha256),
        "selector_checkpoint_sha256": str(
            selector_checkpoint_sha256
        ),
        "selection_mode": str(selection_mode),
        "selection_seed": int(selection_seed),
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def read_dual_sgw_manifest(
    path: Path,
) -> Tuple[Dict, List[DualSGWFeatureRecord], Dict[str, torch.Tensor]]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != DUAL_SGW_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported dual SGW manifest schema")
    if payload.get("artifact_type") != "dual_stse_exact_sgw_manifest":
        raise ValueError("unexpected dual SGW manifest")
    rows = payload.get("records", [])
    if len(rows) != int(payload.get("sample_count", -1)):
        raise ValueError("dual SGW manifest sample count mismatch")
    records = []
    lookup = {}
    for row in rows:
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        record = load_dual_sgw_feature_record(feature_path)
        if (
            record.sample_key != row["sample_key"]
            or record.label != int(row["label"])
            or record.split != row["split"]
        ):
            raise ValueError("dual SGW manifest record mismatch")
        if (
            record.protocol_sha256 != payload["protocol_sha256"]
            or record.selector_checkpoint_sha256
            != payload["selector_checkpoint_sha256"]
            or record.selection_mode != payload["selection_mode"]
            or int(record.selection_seed)
            != int(payload["selection_seed"])
        ):
            raise ValueError("dual SGW manifest provenance mismatch")
        records.append(record)
        lookup[record.sample_key] = record.representation
    if len(lookup) != len(records):
        raise ValueError("dual SGW manifest contains duplicate keys")
    return payload, records, lookup
