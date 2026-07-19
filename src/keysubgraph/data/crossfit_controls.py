"""Frozen Key/Random controls for the confirmatory cross-fitted experiment."""

from __future__ import absolute_import, division, print_function

import copy
import hashlib
import json
import os
from pathlib import Path

from keysubgraph.analysis.controls import generate_random_controls
from keysubgraph.data.data_protocol import validate_data_protocol
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.graph_dataset import GraphSequenceDataset


KEY_RANDOM_CONTROL_SCHEMA_VERSION = 1


def _subgraph_signature(record):
    return (
        tuple(int(node) for node in record["node_ids"]),
        tuple(tuple(int(value) for value in edge) for edge in record["edge_index"]),
    )


def _key_map(payload):
    result = {}
    for timepoint in payload.get("timepoints", []):
        time_index = int(timepoint["time_index"])
        for subgraph_index, record in enumerate(timepoint.get("subgraphs", [])):
            key = (time_index, subgraph_index)
            if key in result:
                raise ValueError("duplicate Key tuple")
            result[key] = record
    if not result:
        raise ValueError("Key payload contains no subgraphs")
    return result


def build_key_random_payloads(sample, key_payload, random_seed=42, repeat_index=0):
    """Build one deterministic Random control with the exact Key tuple budgets."""

    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    keys = _key_map(key_payload)
    random_records = generate_random_controls(
        sample, key_payload, repeats=repeat_index + 1, seed=random_seed
    )
    random_map = {}
    rejected_identical = 0
    for record in random_records:
        if int(record.get("repeat_index", -1)) != repeat_index:
            continue
        tuple_key = (int(record["time_index"]), int(record["subgraph_index"]))
        if tuple_key in random_map:
            raise ValueError("duplicate Random tuple")
        if tuple_key in keys and _subgraph_signature(record) == _subgraph_signature(keys[tuple_key]):
            rejected_identical += 1
            continue
        random_map[tuple_key] = record
    common = sorted(set(keys) & set(random_map))
    by_time = {}
    inventory = []
    for tuple_key in common:
        key_record = keys[tuple_key]
        random_record = random_map[tuple_key]
        node_count = len(key_record["node_ids"])
        edge_count = len(key_record["edge_index"])
        if len(random_record["node_ids"]) != node_count:
            raise ValueError("Random node budget differs from Key")
        if len(random_record["edge_index"]) != edge_count:
            raise ValueError("Random edge budget differs from Key")
        by_time.setdefault(tuple_key[0], []).append(tuple_key)
        inventory.append({
            "time_index": tuple_key[0],
            "subgraph_index": tuple_key[1],
            "node_count": node_count,
            "edge_count": edge_count,
        })
    expected_times = [int(item["time_index"]) for item in key_payload["timepoints"]]
    empty_timepoints = [value for value in expected_times if value not in by_time]
    audit = {
        "included": bool(common) and not empty_timepoints,
        "key_tuple_count": len(keys),
        "matched_tuple_count": len(common),
        "dropped_tuple_count": len(keys) - len(common),
        "empty_timepoints": empty_timepoints,
        "random_identical_to_key_rejected": rejected_identical,
        "tuple_inventory": inventory,
    }
    if not audit["included"]:
        return {}, audit

    payloads = {}
    for source, source_map in (("key", keys), ("random", random_map)):
        payload = copy.deepcopy(key_payload)
        payload["subgraph_source"] = source
        payload["key_random_control"] = {
            "random_seed": int(random_seed),
            "repeat_index": int(repeat_index),
            "matched_tuple_count": len(common),
        }
        timepoints = []
        for original in key_payload["timepoints"]:
            time_index = int(original["time_index"])
            current = {
                key: copy.deepcopy(value)
                for key, value in original.items()
                if key not in ("subgraphs", "candidate_pool", "num_valid_subgraphs")
            }
            current["subgraphs"] = []
            for tuple_key in by_time[time_index]:
                record = copy.deepcopy(source_map[tuple_key])
                record["source"] = source
                record["subgraph_index"] = tuple_key[1]
                current["subgraphs"].append(record)
            current["num_valid_subgraphs"] = len(current["subgraphs"])
            timepoints.append(current)
        payload["timepoints"] = timepoints
        payloads[source] = payload
    return payloads, audit


def freeze_key_random_inventory(sample_audits, output_path, fold, random_seed=42):
    """Freeze a label-free cohort/tuple inventory shared by all model variants."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    records = []
    for item in sample_audits:
        if set(item) != {"sample_key", "audit"}:
            raise ValueError("control inventory accepts sample identity and audit only")
        audit = item["audit"]
        if not audit.get("included"):
            continue
        records.append({
            "sample_key": str(item["sample_key"]),
            "tuple_inventory": copy.deepcopy(audit["tuple_inventory"]),
        })
    records.sort(key=lambda item: item["sample_key"])
    if not records or len({item["sample_key"] for item in records}) != len(records):
        raise ValueError("control inventory needs unique included samples")
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    payload = {
        "schema_version": KEY_RANDOM_CONTROL_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "crossfit_key_random_common_cohort",
        "fold": int(fold),
        "random_seed": int(random_seed),
        "sources": ["key", "random"],
        "included_sample_keys": [item["sample_key"] for item in records],
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return payload


def _portable_path(path, project_root):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def build_crossfit_key_random_exports(
    project_root, protocol_path, key_export_root, output_root,
    random_seed=42, repeat_index=0,
):
    """Materialize one fold-wide Key/Random control manifest shared by all partitions."""

    project_root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_data_protocol(protocol_path, project_root)
    protocol_hash = file_sha256(protocol_path)
    key_export_root = Path(key_export_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "key_random_control_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(str(manifest_path))
    partitions = {}
    checkpoint_hash = None
    extraction_config = None
    for split in ("train", "validation", "test"):
        dataset = GraphSequenceDataset(
            project_root / protocol["paths"]["dataset_root"],
            project_root / protocol["paths"]["sample_index_csv"],
            project_root / protocol["paths"]["splits_csv"], split,
            float(protocol["edge_presence_threshold"]),
        )
        dataset_by_key = {
            assignment.sample_key: dataset[index]
            for index, assignment in enumerate(dataset.assignments)
        }
        export_dir = key_export_root / split
        export_paths = sorted(export_dir.glob("*.json"))
        if not export_paths:
            raise ValueError("Key export partition is empty: {}".format(split))
        records = {"key": [], "random": []}
        included = []
        excluded = []
        audits = []
        observed = set()
        for export_path in export_paths:
            with export_path.open("r", encoding="utf-8") as handle:
                key_payload = json.load(handle)
            sample_key = str(key_payload.get("sample_key", ""))
            if sample_key in observed or sample_key not in dataset_by_key:
                raise ValueError("duplicate or unknown Key export sample")
            observed.add(sample_key)
            if key_payload.get("data_protocol_sha256") != protocol_hash:
                raise ValueError("Key export protocol hash differs")
            current_checkpoint = str(key_payload.get("checkpoint_sha256", ""))
            current_config = key_payload.get("hard_extraction_config")
            if checkpoint_hash is None:
                checkpoint_hash, extraction_config = current_checkpoint, current_config
            elif checkpoint_hash != current_checkpoint or extraction_config != current_config:
                raise ValueError("Key export extraction provenance differs")
            payloads, audit = build_key_random_payloads(
                dataset_by_key[sample_key], key_payload, random_seed, repeat_index
            )
            audit = dict(audit)
            audit["sample_key"] = sample_key
            audits.append(audit)
            if not audit["included"]:
                excluded.append({"sample_key": sample_key, "reason": "unmatched_random_control"})
                continue
            included.append(sample_key)
            for source in ("key", "random"):
                output_path = output_root / source / split / export_path.name
                _write_json(output_path, payloads[source])
                records[source].append({
                    "sample_key": sample_key,
                    "export_json": _portable_path(output_path, project_root),
                    "sha256": file_sha256(output_path),
                    "timepoint_count": len(payloads[source]["timepoints"]),
                    "subgraph_count": sum(len(item["subgraphs"]) for item in payloads[source]["timepoints"]),
                })
        if observed != set(dataset_by_key):
            raise ValueError("Key exports do not cover fold partition")
        if not included:
            raise RuntimeError("Random matching excluded an entire partition")
        partitions[split] = {
            "included_sample_keys": sorted(included), "excluded_samples": excluded,
            "sample_audits": audits, "source_records": records,
        }
    manifest = {
        "schema_version": 1, "immutable": True,
        "purpose": "baseline_matched_subgraph_sources",
        "experiment_kind": "crossfit_key_random_common_cohort",
        "split": "crossfit_fold", "partitions": ["train", "validation", "test"],
        "data_protocol": _portable_path(protocol_path, project_root),
        "data_protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "hard_extraction_config": extraction_config,
        "random_seed": int(random_seed), "random_repeat_index": int(repeat_index),
        "sources": ["key", "random"], "partition_inventories": partitions,
    }
    _write_json(manifest_path, manifest)
    return manifest
