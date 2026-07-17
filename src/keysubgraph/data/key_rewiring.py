"""Deterministic signed endpoint rewiring for exported Key subgraphs."""

from __future__ import absolute_import, division, print_function

import copy
import hashlib
import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data_protocol import validate_data_protocol
from .data_split import file_sha256
from .graph_dataset import GraphSequenceDataset


KEY_REWIRED_SOURCES = ("key", "key_rewired")
KEY_REWIRING_SCHEMA_VERSION = 1


def _portable_path(path: Path, project_root: Path) -> str:
    absolute = Path(os.path.abspath(str(path)))
    try:
        return absolute.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _stable_seed(seed: int, *parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return (int(seed) + int.from_bytes(digest[:8], "big")) % (2 ** 32)


def _canonical_edge(edge: Sequence[int]) -> Tuple[int, int]:
    if not isinstance(edge, (list, tuple)) or len(edge) != 2:
        raise ValueError("edge must contain two endpoints")
    left, right = int(edge[0]), int(edge[1])
    if left == right:
        raise ValueError("self loops cannot be rewired")
    return (left, right) if left < right else (right, left)


def rewire_key_subgraph(
    key_subgraph: Dict[str, Any],
    sample_key: str,
    time_index: int,
    subgraph_index: int,
    seed: int = 2026,
    max_attempts: int = 256,
) -> Optional[Dict[str, Any]]:
    """Reassign signed Key weights to a different simple topology.

    Node identity, edge count, positive/negative counts, and the exact signed
    weight multiset are retained. Returning ``None`` means that no different
    simple edge set exists (for example, a complete subgraph).
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    nodes = [int(node) for node in key_subgraph.get("node_ids", [])]
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise ValueError("key subgraph has invalid node_ids")
    source_edges = [_canonical_edge(edge) for edge in key_subgraph.get("edge_index", [])]
    weights = [float(value) for value in key_subgraph.get("original_edge_weights", [])]
    if not source_edges or len(source_edges) != len(weights):
        raise ValueError("key edge and weight counts differ")
    if len(set(source_edges)) != len(source_edges):
        raise ValueError("key subgraph contains duplicate edges")
    if any(weight == 0.0 for weight in weights):
        raise ValueError("signed rewiring does not accept zero-weight edges")
    node_set = set(nodes)
    if any(left not in node_set or right not in node_set for left, right in source_edges):
        raise ValueError("key edge endpoint is absent from node_ids")

    possible_edges = list(itertools.combinations(sorted(nodes), 2))
    edge_count = len(source_edges)
    if edge_count >= len(possible_edges):
        return None
    original_set = set(source_edges)
    rng = random.Random(
        _stable_seed(seed, sample_key, time_index, subgraph_index, "key_rewired")
    )
    selected = None
    for _ in range(max_attempts):
        candidate = sorted(rng.sample(possible_edges, edge_count))
        if set(candidate) != original_set:
            selected = candidate
            break
    if selected is None:
        return None

    positive_indices = [index for index, weight in enumerate(weights) if weight > 0.0]
    negative_indices = [index for index, weight in enumerate(weights) if weight < 0.0]
    rng.shuffle(selected)
    positive_edges = selected[:len(positive_indices)]
    negative_edges = selected[len(positive_indices):]
    rng.shuffle(positive_edges)
    rng.shuffle(negative_edges)
    new_edges = positive_edges + negative_edges
    source_order = list(positive_indices)
    rng.shuffle(source_order)
    negative_order = list(negative_indices)
    rng.shuffle(negative_order)
    source_order.extend(negative_order)

    rewired = copy.deepcopy(key_subgraph)
    rewired["edge_index"] = [list(edge) for edge in new_edges]
    rewired["original_edge_weights"] = [weights[index] for index in source_order]
    for field in ("delta_edge_weight", "delta_edge_mask"):
        values = key_subgraph.get(field)
        if isinstance(values, list) and len(values) == edge_count:
            rewired[field] = [values[index] for index in source_order]
    rewired["source"] = "key_rewired"
    rewired["repeat_index"] = None
    rewired["rewiring"] = {
        "schema_version": KEY_REWIRING_SCHEMA_VERSION,
        "method": "signed_endpoint_resampling_without_replacement",
        "seed": int(seed),
        "stable_seed": int(
            _stable_seed(seed, sample_key, time_index, subgraph_index, "key_rewired")
        ),
        "source_edge_index": [list(source_edges[index]) for index in source_order],
        "source_subgraph_index": int(subgraph_index),
        "positive_edge_count": len(positive_indices),
        "negative_edge_count": len(negative_indices),
        "changed_edge_count": len(set(new_edges).symmetric_difference(original_set)) // 2,
    }
    return rewired


def build_key_rewired_payloads(
    key_payload: Dict[str, Any],
    seed: int = 2026,
    max_attempts: int = 256,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Restrict Key and Key-rewired to the same perturbable tuple inventory."""

    sample_key = str(key_payload.get("sample_key", ""))
    key_timepoints = []
    rewired_timepoints = []
    total = 0
    retained = 0
    empty_timepoints = []
    changed_edges = 0
    retained_edges = 0
    for timepoint in key_payload.get("timepoints", []):
        time_index = int(timepoint["time_index"])
        key_records = []
        rewired_records = []
        for subgraph_index, key_subgraph in enumerate(timepoint.get("subgraphs", [])):
            total += 1
            rewired = rewire_key_subgraph(
                key_subgraph,
                sample_key,
                time_index,
                subgraph_index,
                seed=seed,
                max_attempts=max_attempts,
            )
            if rewired is None:
                continue
            key_record = copy.deepcopy(key_subgraph)
            key_record["source"] = "key"
            key_record["subgraph_index"] = subgraph_index
            rewired["subgraph_index"] = subgraph_index
            key_records.append(key_record)
            rewired_records.append(rewired)
            retained += 1
            retained_edges += len(rewired["edge_index"])
            changed_edges += int(rewired["rewiring"]["changed_edge_count"])
        if not key_records:
            empty_timepoints.append(time_index)
        def timepoint_payload(records):
            current = {
                name: copy.deepcopy(value)
                for name, value in timepoint.items()
                if name not in ("subgraphs", "candidate_pool", "num_valid_subgraphs")
            }
            current["subgraphs"] = records
            current["num_valid_subgraphs"] = len(records)
            return current
        key_timepoints.append(timepoint_payload(key_records))
        rewired_timepoints.append(timepoint_payload(rewired_records))

    audit = {
        "key_tuple_count": total,
        "matched_tuple_count": retained,
        "dropped_unrewirable_tuple_count": total - retained,
        "empty_timepoints": empty_timepoints,
        "retained_edge_count": retained_edges,
        "changed_edge_count": changed_edges,
        "changed_edge_ratio": (
            float(changed_edges) / retained_edges if retained_edges else 0.0
        ),
        "included": bool(retained) and not empty_timepoints,
    }
    if not audit["included"]:
        return {}, audit
    payloads = {}
    for source, timepoints in (
        ("key", key_timepoints), ("key_rewired", rewired_timepoints)
    ):
        payload = copy.deepcopy(key_payload)
        payload["subgraph_source"] = source
        payload["key_rewiring"] = {
            "seed": int(seed),
            "method": "signed_endpoint_resampling_without_replacement",
            "key_tuple_count": total,
            "matched_tuple_count": retained,
        }
        payload["timepoints"] = timepoints
        payloads[source] = payload
    return payloads, audit


def build_key_rewired_exports(
    project_root: Path,
    protocol_path: Path,
    key_export_dir: Path,
    split: str,
    output_root: Path,
    rewiring_seed: int = 2026,
    max_attempts: int = 256,
) -> Dict[str, Any]:
    """Materialize an immutable matched Key versus Key-rewired experiment."""

    project_root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_data_protocol(protocol_path, project_root)
    protocol_hash = file_sha256(protocol_path)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("key-rewired output root already exists")
    key_export_dir = Path(key_export_dir).resolve()
    if (key_export_dir / split).is_dir():
        key_export_dir = key_export_dir / split
    export_paths = sorted(key_export_dir.glob("*.json"))
    if not export_paths:
        raise ValueError("key export directory contains no JSON files")

    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        project_root / paths["dataset_root"],
        project_root / paths["sample_index_csv"],
        project_root / paths["splits_csv"],
        split=split,
        edge_presence_threshold=float(protocol["edge_presence_threshold"]),
    )
    dataset_keys = {assignment.sample_key for assignment in dataset.assignments}
    observed_keys = set()
    source_records = {source: [] for source in KEY_REWIRED_SOURCES}
    included = []
    excluded = []
    audits = []
    checkpoint_hash = None
    extraction_config = None

    for index, export_path in enumerate(export_paths, 1):
        with export_path.open("r", encoding="utf-8") as handle:
            key_payload = json.load(handle)
        sample_key = str(key_payload.get("sample_key", ""))
        if sample_key in observed_keys:
            raise ValueError("duplicate sample_key in key exports")
        observed_keys.add(sample_key)
        if sample_key not in dataset_keys:
            raise ValueError("key export sample is absent from frozen Dataset")
        if key_payload.get("data_protocol_sha256") != protocol_hash:
            raise ValueError("key export protocol hash mismatch")
        current_checkpoint = str(key_payload.get("checkpoint_sha256", ""))
        current_config = key_payload.get("hard_extraction_config")
        if checkpoint_hash is None:
            checkpoint_hash = current_checkpoint
            extraction_config = current_config
        elif current_checkpoint != checkpoint_hash or current_config != extraction_config:
            raise ValueError("key export checkpoint or extraction config differs")

        payloads, audit = build_key_rewired_payloads(
            key_payload, seed=rewiring_seed, max_attempts=max_attempts
        )
        audit["sample_key"] = sample_key
        audits.append(audit)
        if not audit["included"]:
            excluded.append({
                "sample_key": sample_key,
                "reason": "empty_timepoint_after_rewiring_match",
                "empty_timepoints": audit["empty_timepoints"],
            })
        else:
            included.append(sample_key)
            for source in KEY_REWIRED_SOURCES:
                output_path = output_root / source / split / export_path.name
                _write_json(output_path, payloads[source])
                source_records[source].append({
                    "sample_key": sample_key,
                    "export_json": _portable_path(output_path, project_root),
                    "sha256": file_sha256(output_path),
                    "timepoint_count": len(payloads[source]["timepoints"]),
                    "subgraph_count": sum(
                        len(item["subgraphs"]) for item in payloads[source]["timepoints"]
                    ),
                })
        if index % 25 == 0 or index == len(export_paths):
            print(
                "key rewiring: {}/{} processed; included={} excluded={}".format(
                    index, len(export_paths), len(included), len(excluded)
                ),
                flush=True,
            )
    if observed_keys != dataset_keys:
        raise ValueError("key exports do not cover the frozen split")
    if not included:
        raise RuntimeError("rewiring excluded every sample")

    retained_edges = sum(item["retained_edge_count"] for item in audits if item["included"])
    changed_edges = sum(item["changed_edge_count"] for item in audits if item["included"])
    manifest = {
        "schema_version": 1,
        "immutable": True,
        "purpose": "baseline_matched_subgraph_sources",
        "experiment_kind": "key_signed_endpoint_rewiring",
        "evidence_level": "exploratory_in_sample",
        "split": split,
        "data_protocol": _portable_path(protocol_path, project_root),
        "data_protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "hard_extraction_config": extraction_config,
        "rewiring_seed": int(rewiring_seed),
        "rewiring_method": "signed_endpoint_resampling_without_replacement",
        "sources": list(KEY_REWIRED_SOURCES),
        "included_sample_keys": sorted(included),
        "excluded_samples": excluded,
        "sample_audits": audits,
        "source_records": source_records,
        "rewiring_summary": {
            "retained_edge_count": retained_edges,
            "changed_edge_count": changed_edges,
            "changed_edge_ratio": float(changed_edges) / retained_edges,
        },
    }
    _write_json(output_root / "matched_control_manifest.json", manifest)
    return manifest
