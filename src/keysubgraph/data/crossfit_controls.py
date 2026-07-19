"""Frozen Key/Random controls for the confirmatory cross-fitted experiment."""

from __future__ import absolute_import, division, print_function

import copy
import hashlib
import json
import os
from pathlib import Path

from keysubgraph.analysis.controls import generate_random_controls


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
