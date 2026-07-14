"""Freeze and validate the immutable data contract used by experiments."""

from __future__ import absolute_import, division, print_function

import json
import os
from pathlib import Path
from typing import Any, Dict

from .data_split import file_sha256, read_sample_index, read_split_assignments
from .full_cohort import FULL_COHORT_MODE


def protocol_partitions(protocol: Dict[str, Any]):
    """Return the dataset partitions defined by a validated protocol."""

    return ("all",) if protocol.get("experiment_mode") == FULL_COHORT_MODE else (
        "train",
        "validation",
        "test",
    )


def _portable_path(path: Path, project_root: Path) -> str:
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def freeze_data_protocol(
    project_root: Path,
    dataset_root: Path,
    sample_index_csv: Path,
    splits_csv: Path,
    splits_json: Path,
    output_path: Path,
    edge_presence_threshold: float = 0.0,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate data artifacts and write their reproducible contract."""

    project_root = Path(project_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    sample_index_csv = Path(sample_index_csv).resolve()
    splits_csv = Path(splits_csv).resolve()
    splits_json = Path(splits_json).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "data protocol already exists; reuse it or explicitly request overwrite"
        )
    if edge_presence_threshold < 0.0:
        raise ValueError("edge_presence_threshold must be non-negative")

    samples = read_sample_index(sample_index_csv)
    assignments = read_split_assignments(splits_csv)
    with splits_json.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    index_digest = file_sha256(sample_index_csv)
    if split_payload.get("source_index_sha256") != index_digest:
        raise ValueError("splits.json was not generated from the current sample index")
    csv_rows = [item.to_dict() for item in assignments]
    if csv_rows != split_payload.get("assignments"):
        raise ValueError("splits.csv and splits.json assignments do not match")
    index_keys = {sample.sample_key for sample in samples}
    split_keys = {assignment.sample_key for assignment in assignments}
    if index_keys != split_keys:
        raise ValueError("sample index and split assignments do not contain the same samples")

    experiment_mode = split_payload.get("assignment_mode", "partitioned_evaluation")
    if experiment_mode == FULL_COHORT_MODE:
        if any(assignment.split != "all" for assignment in assignments):
            raise ValueError("all-sample protocol contains a non-'all' assignment")
        if split_payload.get("ratios") != {"all": 1.0}:
            raise ValueError("all-sample protocol must declare ratios={'all': 1.0}")
    elif any(assignment.split == "all" for assignment in assignments):
        raise ValueError("'all' assignments require all_samples_exploratory mode")

    missing_files = [
        sample.relative_path
        for sample in samples
        if not (dataset_root / sample.relative_path).is_file()
    ]
    if missing_files:
        raise ValueError(
            "dataset files referenced by the index are missing (first: {})".format(
                missing_files[0]
            )
        )

    payload = {
        "schema_version": 1,
        "immutable": True,
        "paths": {
            "dataset_root": _portable_path(dataset_root, project_root),
            "sample_index_csv": _portable_path(sample_index_csv, project_root),
            "splits_csv": _portable_path(splits_csv, project_root),
            "splits_json": _portable_path(splits_json, project_root),
        },
        "sha256": {
            "sample_index_csv": index_digest,
            "splits_csv": file_sha256(splits_csv),
            "splits_json": file_sha256(splits_json),
        },
        "sample_count": len(samples),
        "experiment_mode": experiment_mode,
        "label_source": "sample_index.csv and splits.csv only; .pt labels are ignored",
        "split_policy": "reuse splits.csv; training code must not randomly re-split",
        "model_selection_policy": (
            "lowest full-cohort inference loss; not validation performance"
            if experiment_mode == FULL_COHORT_MODE
            else "validation-only model selection"
        ),
        "batching": "list_based_variable_length_no_padding_no_truncation",
        "edge_presence_rule": "abs(A_ij) > edge_presence_threshold",
        "edge_presence_threshold": float(edge_presence_threshold),
        "source_global_threshold_policy": ".pt global_threshold is retained as metadata only",
        "signed_edge_policy": "positive and negative nonzero edges are both valid",
        "community_policy": (
            "community ids are local grouping labels and must never be passed to nn.Embedding"
        ),
        "split_seed": int(split_payload["seed"]),
        "split_ratios": split_payload["ratios"],
        "group_key": split_payload["group_key"],
    }
    _atomic_write_json(output_path, payload)
    return payload


def validate_data_protocol(protocol_path: Path, project_root: Path) -> Dict[str, Any]:
    """Verify that all frozen artifacts still match the protocol hashes."""

    protocol_path = Path(protocol_path).resolve()
    project_root = Path(project_root).resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not payload.get("immutable"):
        raise ValueError("unsupported or mutable data protocol")
    for name in ("sample_index_csv", "splits_csv", "splits_json"):
        artifact = project_root / payload["paths"][name]
        if not artifact.is_file():
            raise ValueError("protocol artifact is missing: {}".format(name))
        if file_sha256(artifact) != payload["sha256"][name]:
            raise ValueError("protocol artifact hash mismatch: {}".format(name))
    dataset_root = project_root / payload["paths"]["dataset_root"]
    if not dataset_root.is_dir():
        raise ValueError("protocol dataset root is missing")
    return payload
