"""Matched dose-response deletion of high-score versus random Key edges."""

from __future__ import absolute_import, division, print_function

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .data_protocol import validate_data_protocol
from .data_split import file_sha256
from .graph_dataset import GraphSequenceDataset


EDGE_PERTURBATION_SCHEMA_VERSION = 1
EDGE_PERTURBATION_RATIOS = (0.0, 0.10, 0.25, 0.50)
EDGE_PERTURBATION_MODES = ("targeted", "random")
EDGE_ALIGNED_FIELDS = (
    "edge_index",
    "original_edge_weights",
    "edge_scores",
    "delta_edge_weight",
    "delta_edge_mask",
)


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
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return (int(seed) + int.from_bytes(digest[:8], "big")) % (2 ** 32)


def _ratio_code(ratio: float) -> str:
    return "{:03d}".format(int(round(float(ratio) * 100.0)))


def perturbation_source(mode: str, ratio: float) -> str:
    if ratio == 0.0:
        return "key_edge_000"
    if mode not in EDGE_PERTURBATION_MODES:
        raise ValueError("unsupported edge perturbation mode")
    return "key_edge_{}_{}".format(mode, _ratio_code(ratio))


def perturbation_sources(ratios: Sequence[float] = EDGE_PERTURBATION_RATIOS) -> Tuple[str, ...]:
    sources = [perturbation_source("targeted", 0.0)]
    for ratio in ratios:
        if float(ratio) == 0.0:
            continue
        sources.extend(
            perturbation_source(mode, float(ratio))
            for mode in EDGE_PERTURBATION_MODES
        )
    return tuple(sources)


def _validate_ratios(ratios: Sequence[float]) -> Tuple[float, ...]:
    normalized = tuple(float(value) for value in ratios)
    if not normalized or normalized[0] != 0.0:
        raise ValueError("edge perturbation ratios must begin with zero")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("edge perturbation ratios must be unique and increasing")
    if any(value < 0.0 or value >= 1.0 for value in normalized):
        raise ValueError("edge perturbation ratios must be in [0, 1)")
    return normalized


def _canonical_edge(edge: Sequence[int]) -> Tuple[int, int]:
    if not isinstance(edge, (list, tuple)) or len(edge) != 2:
        raise ValueError("edge must contain two endpoints")
    left, right = int(edge[0]), int(edge[1])
    if left == right:
        raise ValueError("self loops cannot be perturbed")
    return (left, right) if left < right else (right, left)


def _edge_inventory(key_subgraph: Dict[str, Any]):
    edges = [_canonical_edge(edge) for edge in key_subgraph.get("edge_index", [])]
    weights = [float(value) for value in key_subgraph.get("original_edge_weights", [])]
    scores = [float(value) for value in key_subgraph.get("edge_scores", [])]
    if len(edges) < 2:
        raise ValueError("edge perturbation requires at least two edges")
    if len(set(edges)) != len(edges):
        raise ValueError("Key subgraph contains duplicate edges")
    if len(weights) != len(edges) or len(scores) != len(edges):
        raise ValueError("Key edge weights or scores do not align")
    if any(not math.isfinite(value) for value in weights + scores):
        raise ValueError("Key edge values must be finite")
    if any(value == 0.0 for value in weights):
        raise ValueError("Key perturbation does not accept zero-weight edges")
    nodes = {int(node) for node in key_subgraph.get("node_ids", [])}
    if any(left not in nodes or right not in nodes for left, right in edges):
        raise ValueError("Key edge endpoint is absent from node_ids")
    for field in EDGE_ALIGNED_FIELDS:
        values = key_subgraph.get(field)
        if values is not None and (not isinstance(values, list) or len(values) != len(edges)):
            raise ValueError("edge-aligned field differs in length: {}".format(field))
    return edges, weights, scores


def _deletion_count(edge_count: int, ratio: float) -> int:
    if ratio == 0.0:
        return 0
    requested = int(math.floor(edge_count * ratio + 0.5))
    return min(edge_count - 1, max(1, requested))


def edge_deletion_order(
    key_subgraph: Dict[str, Any],
    mode: str,
    sample_key: str,
    time_index: int,
    subgraph_index: int,
    seed: int,
) -> List[int]:
    """Return one frozen full edge order; prefixes define nested doses."""

    edges, unused_weights, scores = _edge_inventory(key_subgraph)
    del unused_weights
    if mode == "targeted":
        return sorted(
            range(len(edges)),
            key=lambda index: (-scores[index], edges[index], index),
        )
    if mode != "random":
        raise ValueError("unsupported edge perturbation mode")
    order = list(range(len(edges)))
    stable_seed = _stable_seed(
        seed, sample_key, time_index, subgraph_index, "random_edge_deletion"
    )
    random.Random(stable_seed).shuffle(order)
    return order


def perturb_key_subgraph(
    key_subgraph: Dict[str, Any],
    mode: str,
    ratio: float,
    sample_key: str,
    time_index: int,
    subgraph_index: int,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Delete a nested prefix of high-score or frozen-random edges."""

    if ratio < 0.0 or ratio >= 1.0 or seed < 0:
        raise ValueError("invalid edge perturbation configuration")
    edges, weights, scores = _edge_inventory(key_subgraph)
    order = edge_deletion_order(
        key_subgraph, mode, sample_key, time_index, subgraph_index, seed
    )
    delete_count = _deletion_count(len(edges), ratio)
    deleted = set(order[:delete_count])
    retained = [index for index in range(len(edges)) if index not in deleted]
    output = copy.deepcopy(key_subgraph)
    for field in EDGE_ALIGNED_FIELDS:
        values = key_subgraph.get(field)
        if values is not None:
            output[field] = [copy.deepcopy(values[index]) for index in retained]
    source = perturbation_source(mode, ratio)
    output["source"] = source
    output["subgraph_index"] = int(subgraph_index)
    output["edge_perturbation"] = {
        "schema_version": EDGE_PERTURBATION_SCHEMA_VERSION,
        "method": "edge_deletion",
        "mode": "none" if ratio == 0.0 else mode,
        "requested_ratio": float(ratio),
        "realized_ratio": float(delete_count) / len(edges),
        "seed": int(seed),
        "stable_seed": int(_stable_seed(
            seed, sample_key, time_index, subgraph_index, "random_edge_deletion"
        )) if mode == "random" else None,
        "source_subgraph_index": int(subgraph_index),
        "original_edge_count": len(edges),
        "deleted_edge_count": delete_count,
        "retained_edge_count": len(retained),
        "deletion_order": order,
        "deleted_source_positions": order[:delete_count],
        "deleted_edges": [list(edges[index]) for index in order[:delete_count]],
        "deleted_weights": [weights[index] for index in order[:delete_count]],
        "deleted_scores": [scores[index] for index in order[:delete_count]],
        "deleted_positive_count": sum(weights[index] > 0.0 for index in deleted),
        "deleted_negative_count": sum(weights[index] < 0.0 for index in deleted),
    }
    return output


def build_edge_perturbation_payloads(
    key_payload: Dict[str, Any],
    ratios: Sequence[float] = EDGE_PERTURBATION_RATIOS,
    seed: int = 2026,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Create all matched doses after one common perturbability filter."""

    ratios = _validate_ratios(ratios)
    sample_key = str(key_payload.get("sample_key", ""))
    sources = perturbation_sources(ratios)
    timepoints = {source: [] for source in sources}
    tuple_count = 0
    retained_tuple_count = 0
    dropped_small_tuple_count = 0
    empty_timepoints = []
    source_totals = {
        source: {"original_edge_count": 0, "deleted_edge_count": 0}
        for source in sources
    }
    for timepoint in key_payload.get("timepoints", []):
        time_index = int(timepoint["time_index"])
        records = {source: [] for source in sources}
        for subgraph_index, subgraph in enumerate(timepoint.get("subgraphs", [])):
            tuple_count += 1
            if len(subgraph.get("edge_index", [])) < 2:
                dropped_small_tuple_count += 1
                continue
            retained_tuple_count += 1
            for ratio in ratios:
                modes = ("targeted",) if ratio == 0.0 else EDGE_PERTURBATION_MODES
                for mode in modes:
                    perturbed = perturb_key_subgraph(
                        subgraph, mode, ratio, sample_key, time_index,
                        subgraph_index, seed=seed,
                    )
                    source = perturbation_source(mode, ratio)
                    records[source].append(perturbed)
                    provenance = perturbed["edge_perturbation"]
                    source_totals[source]["original_edge_count"] += int(
                        provenance["original_edge_count"]
                    )
                    source_totals[source]["deleted_edge_count"] += int(
                        provenance["deleted_edge_count"]
                    )
        if not records[sources[0]]:
            empty_timepoints.append(time_index)
        for source in sources:
            current = {
                name: copy.deepcopy(value) for name, value in timepoint.items()
                if name not in ("subgraphs", "candidate_pool", "num_valid_subgraphs")
            }
            current["subgraphs"] = records[source]
            current["num_valid_subgraphs"] = len(records[source])
            timepoints[source].append(current)
    included = retained_tuple_count > 0 and not empty_timepoints
    audit = {
        "sample_key": sample_key,
        "key_tuple_count": tuple_count,
        "matched_tuple_count": retained_tuple_count,
        "dropped_lt_two_edge_tuple_count": dropped_small_tuple_count,
        "empty_timepoints": empty_timepoints,
        "included": included,
        "source_totals": source_totals,
    }
    if not included:
        return {}, audit
    payloads = {}
    for source in sources:
        payload = copy.deepcopy(key_payload)
        payload["subgraph_source"] = source
        payload["edge_perturbation_experiment"] = {
            "schema_version": EDGE_PERTURBATION_SCHEMA_VERSION,
            "seed": int(seed),
            "ratios": list(ratios),
            "common_tuple_filter": "edge_count_at_least_two",
            "source": source,
        }
        payload["timepoints"] = timepoints[source]
        payloads[source] = payload
    return payloads, audit


def build_edge_perturbation_exports(
    project_root: Path,
    protocol_path: Path,
    key_export_dir: Path,
    split: str,
    output_root: Path,
    ratios: Sequence[float] = EDGE_PERTURBATION_RATIOS,
    perturbation_seed: int = 2026,
) -> Dict[str, Any]:
    """Materialize immutable matched Key edge-deletion dose exports."""

    ratios = _validate_ratios(ratios)
    project_root = Path(project_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_data_protocol(protocol_path, project_root)
    protocol_hash = file_sha256(protocol_path)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("edge perturbation output root already exists")
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
    observed = set()
    sources = perturbation_sources(ratios)
    source_records = {source: [] for source in sources}
    included = []
    excluded = []
    audits = []
    checkpoint_hash = None
    extraction_config = None
    for index, export_path in enumerate(export_paths, 1):
        with export_path.open("r", encoding="utf-8") as handle:
            key_payload = json.load(handle)
        sample_key = str(key_payload.get("sample_key", ""))
        if sample_key in observed or sample_key not in dataset_keys:
            raise ValueError("duplicate or unknown key export sample")
        observed.add(sample_key)
        if key_payload.get("data_protocol_sha256") != protocol_hash:
            raise ValueError("key export protocol hash mismatch")
        current_checkpoint = str(key_payload.get("checkpoint_sha256", ""))
        current_config = key_payload.get("hard_extraction_config")
        if checkpoint_hash is None:
            checkpoint_hash = current_checkpoint
            extraction_config = current_config
        elif checkpoint_hash != current_checkpoint or extraction_config != current_config:
            raise ValueError("key export checkpoint or extraction config differs")
        payloads, audit = build_edge_perturbation_payloads(
            key_payload, ratios=ratios, seed=perturbation_seed
        )
        audits.append(audit)
        if not audit["included"]:
            excluded.append({
                "sample_key": sample_key,
                "reason": "empty_timepoint_after_common_perturbability_filter",
                "empty_timepoints": audit["empty_timepoints"],
            })
        else:
            included.append(sample_key)
            for source in sources:
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
                "edge perturbation: {}/{} processed; included={} excluded={}".format(
                    index, len(export_paths), len(included), len(excluded)
                ), flush=True,
            )
    if observed != dataset_keys:
        raise ValueError("key exports do not cover the frozen split")
    if not included:
        raise RuntimeError("edge perturbation excluded every sample")
    summaries = {}
    for source in sources:
        original = sum(
            audit["source_totals"][source]["original_edge_count"]
            for audit in audits if audit["included"]
        )
        deleted = sum(
            audit["source_totals"][source]["deleted_edge_count"]
            for audit in audits if audit["included"]
        )
        summaries[source] = {
            "original_edge_count": original,
            "deleted_edge_count": deleted,
            "retained_edge_count": original - deleted,
            "realized_deleted_ratio": float(deleted) / original if original else 0.0,
        }
    manifest = {
        "schema_version": EDGE_PERTURBATION_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "baseline_matched_subgraph_sources",
        "experiment_kind": "key_edge_deletion_dose_response",
        "evidence_level": "exploratory_in_sample",
        "split": split,
        "data_protocol": _portable_path(protocol_path, project_root),
        "data_protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "hard_extraction_config": extraction_config,
        "perturbation_seed": int(perturbation_seed),
        "ratios": list(ratios),
        "sources": list(sources),
        "common_tuple_filter": "edge_count_at_least_two",
        "included_sample_keys": sorted(included),
        "excluded_samples": excluded,
        "sample_audits": audits,
        "source_records": source_records,
        "perturbation_summary": summaries,
    }
    _write_json(output_root / "matched_control_manifest.json", manifest)
    return manifest
