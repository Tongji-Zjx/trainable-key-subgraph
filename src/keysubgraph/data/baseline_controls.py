"""Freeze tuple-matched Key, Low-score, Top-degree, and Random exports."""

from __future__ import absolute_import, division, print_function

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from keysubgraph.analysis.controls import (
    generate_random_controls,
    generate_top_degree_controls,
    select_low_score_controls,
)

from .data_protocol import validate_data_protocol
from .data_split import file_sha256
from .graph_dataset import GraphSequenceDataset, GraphSequenceSample


MATCHED_CONTROL_SCHEMA_VERSION = 1
MATCHED_SOURCES = ("key", "low_score", "top_degree", "random")


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


def _control_map(records: List[Dict[str, Any]], source: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    result = {}
    for record in records:
        key = (int(record["time_index"]), int(record["subgraph_index"]))
        if key in result:
            raise ValueError("duplicate {} control tuple: {}".format(source, key))
        result[key] = record
    return result


def build_matched_source_payloads(
    sample: GraphSequenceSample,
    key_payload: Dict[str, Any],
    random_seed: int = 42,
    random_repeat_index: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Return four payloads restricted to the exact common tuple inventory."""

    if random_repeat_index < 0:
        raise ValueError("random_repeat_index must be non-negative")
    key_map = {}
    for timepoint in key_payload.get("timepoints", []):
        time_index = int(timepoint["time_index"])
        for subgraph_index, subgraph in enumerate(timepoint.get("subgraphs", [])):
            key_map[(time_index, subgraph_index)] = subgraph
    if not key_map:
        raise ValueError("key export contains no subgraphs")
    low_map = _control_map(select_low_score_controls(key_payload), "low_score")
    top_map = _control_map(
        generate_top_degree_controls(sample, key_payload), "top_degree"
    )
    random_records = generate_random_controls(
        sample,
        key_payload,
        repeats=random_repeat_index + 1,
        seed=random_seed,
    )
    random_map_unfiltered = _control_map(
        [
            record
            for record in random_records
            if int(record.get("repeat_index", -1)) == random_repeat_index
        ],
        "random",
    )
    def signature(record):
        return (
            tuple(record["node_ids"]),
            tuple(tuple(edge) for edge in record["edge_index"]),
        )

    random_identical_count = sum(
        tuple_key in key_map
        and signature(record) == signature(key_map[tuple_key])
        for tuple_key, record in random_map_unfiltered.items()
    )
    random_map = {
        tuple_key: record
        for tuple_key, record in random_map_unfiltered.items()
        if tuple_key not in key_map
        or signature(record) != signature(key_map[tuple_key])
    }
    maps = {
        "key": key_map,
        "low_score": low_map,
        "top_degree": top_map,
        "random": random_map,
    }
    common = set(key_map)
    for source in MATCHED_SOURCES[1:]:
        common &= set(maps[source])
    common = sorted(common)

    failure_counts = {
        source: len(set(key_map) - set(maps[source]))
        for source in MATCHED_SOURCES[1:]
    }
    by_time = {}
    for tuple_key in common:
        by_time.setdefault(tuple_key[0], []).append(tuple_key)
        key_nodes = len(key_map[tuple_key]["node_ids"])
        key_edges = len(key_map[tuple_key]["edge_index"])
        for source in MATCHED_SOURCES[1:]:
            control = maps[source][tuple_key]
            if len(control["node_ids"]) != key_nodes or len(control["edge_index"]) != key_edges:
                raise ValueError("{} control is not size matched".format(source))

    expected_times = [int(item["time_index"]) for item in key_payload["timepoints"]]
    empty_timepoints = [time_index for time_index in expected_times if not by_time.get(time_index)]
    audit = {
        "key_tuple_count": len(key_map),
        "matched_tuple_count": len(common),
        "dropped_tuple_count": len(key_map) - len(common),
        "source_missing_tuple_counts": failure_counts,
        "empty_timepoints": empty_timepoints,
        "random_identical_to_key_rejected": int(random_identical_count),
        "top_degree_identical_to_key_count": int(sum(
            tuple_key in top_map
            and signature(record) == signature(key_map[tuple_key])
            for tuple_key, record in top_map.items()
        )),
        "included": not empty_timepoints and bool(common),
    }
    if not audit["included"]:
        return {}, audit

    payloads = {}
    for source in MATCHED_SOURCES:
        payload = copy.deepcopy(key_payload)
        payload["subgraph_source"] = source
        payload["matched_control"] = {
            "random_seed": int(random_seed),
            "random_repeat_index": int(random_repeat_index),
            "key_tuple_count": len(key_map),
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
            selected = []
            for tuple_key in by_time[time_index]:
                subgraph = copy.deepcopy(maps[source][tuple_key])
                subgraph["source"] = source
                subgraph["subgraph_index"] = tuple_key[1]
                selected.append(subgraph)
            current["subgraphs"] = selected
            current["num_valid_subgraphs"] = len(selected)
            timepoints.append(current)
        payload["timepoints"] = timepoints
        payloads[source] = payload
    return payloads, audit


def build_matched_control_exports(
    project_root: Path,
    protocol_path: Path,
    key_export_dir: Path,
    split: str,
    output_root: Path,
    random_seed: int = 42,
    random_repeat_index: int = 0,
) -> Dict[str, Any]:
    """Materialize a common tuple inventory for all four subgraph sources."""

    project_root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_data_protocol(protocol_path, project_root)
    protocol_hash = file_sha256(protocol_path)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("matched-control output root already exists")
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
    dataset_indices = {
        assignment.sample_key: index
        for index, assignment in enumerate(dataset.assignments)
    }
    source_records = {source: [] for source in MATCHED_SOURCES}
    sample_audits = []
    checkpoint_hash = None
    extraction_config = None
    included_keys = []
    excluded = []
    observed_export_keys = set()
    output_filenames = set()

    def report_progress():
        if len(sample_audits) % 25 == 0 or len(sample_audits) == len(export_paths):
            print(
                "matched controls: {}/{} samples processed; included={} excluded={}".format(
                    len(sample_audits),
                    len(export_paths),
                    len(included_keys),
                    len(excluded),
                ),
                flush=True,
            )

    for export_path in export_paths:
        with export_path.open("r", encoding="utf-8") as handle:
            key_payload = json.load(handle)
        sample_key = str(key_payload.get("sample_key", ""))
        if sample_key in observed_export_keys:
            raise ValueError("duplicate sample_key in key exports")
        observed_export_keys.add(sample_key)
        if export_path.name in output_filenames:
            raise ValueError("duplicate key export filename")
        output_filenames.add(export_path.name)
        if sample_key not in dataset_indices:
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
        sample = dataset[dataset_indices[sample_key]]
        payloads, audit = build_matched_source_payloads(
            sample,
            key_payload,
            random_seed=random_seed,
            random_repeat_index=random_repeat_index,
        )
        audit.update({"sample_key": sample_key, "sample_id": sample.sample_id})
        sample_audits.append(audit)
        if not audit["included"]:
            excluded.append({
                "sample_key": sample_key,
                "reason": "empty_timepoint_after_common_tuple_matching",
                "empty_timepoints": audit["empty_timepoints"],
            })
            report_progress()
            continue
        included_keys.append(sample_key)
        for source in MATCHED_SOURCES:
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
        report_progress()
    if not included_keys:
        raise RuntimeError("matching excluded every sample")
    expected_dataset_keys = set(dataset_indices)
    if expected_dataset_keys != observed_export_keys:
        raise ValueError("key exports do not cover the frozen split")
    manifest = {
        "schema_version": MATCHED_CONTROL_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "baseline_matched_subgraph_sources",
        "evidence_level": "exploratory_in_sample",
        "split": split,
        "data_protocol": _portable_path(protocol_path, project_root),
        "data_protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "hard_extraction_config": extraction_config,
        "random_seed": int(random_seed),
        "random_repeat_index": int(random_repeat_index),
        "sources": list(MATCHED_SOURCES),
        "included_sample_keys": sorted(included_keys),
        "excluded_samples": excluded,
        "sample_audits": sample_audits,
        "source_records": source_records,
    }
    manifest_path = output_root / "matched_control_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest
