"""Deterministic sample discovery and metadata validation.

The indexer intentionally does not alter, truncate, pad, or repair graph data.
It records every discovered file in an inventory, then separates currently
usable samples from explicitly excluded samples.
"""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REQUIRED_FIELDS = (
    "adjacency",
    "node_names",
    "community_sequence",
    "window_starts",
    "global_threshold",
    "t_r",
)


@dataclass(frozen=True)
class IndexBuildConfig:
    """Validation settings used while building a sample index."""

    dataset_root: Path
    require_contiguous_communities: bool = True
    edge_presence_threshold: float = 0.0
    symmetry_tolerance: float = 1e-6
    diagonal_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).resolve())
        if self.edge_presence_threshold < 0.0:
            raise ValueError("edge_presence_threshold must be non-negative")


@dataclass(frozen=True)
class SampleRecord:
    """One portable, serializable sample-index row."""

    sample_key: str
    sample_id: str
    site: str
    subject_id: str
    session_id: str
    label: Optional[int]
    relative_path: str
    num_timepoints: Optional[int]
    min_num_nodes: Optional[int]
    max_num_nodes: Optional[int]
    spatial_dim: Optional[int]
    coords_valid: bool
    community_valid: bool
    adjacency_valid: bool
    node_names_valid: bool
    window_starts_valid: bool
    has_positive_edges: bool
    has_negative_edges: bool
    empty_timepoints: Optional[int]
    edge_presence_threshold: float
    source_global_threshold: Optional[float]
    repetition_time: Optional[float]
    included: bool
    exclusion_reasons: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discover_sample_files(dataset_root: Path) -> List[Path]:
    """Return files in deterministic site/label/path order."""

    root = Path(dataset_root).resolve()
    paths = [path for path in root.glob("*/*/*.pt") if path.is_file()]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _portable_relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse_identity(path: Path, root: Path) -> Tuple[str, Optional[int], str, str, str]:
    relative = path.resolve().relative_to(root.resolve())
    if len(relative.parts) != 3:
        raise ValueError("expected <site>/<label>/<sample>.pt layout")

    site, label_text, filename = relative.parts
    if label_text not in ("0", "1"):
        raise ValueError("label directory must be 0 or 1")

    sample_id = Path(filename).stem
    prefix = site + "_"
    identity_text = sample_id[len(prefix) :] if sample_id.startswith(prefix) else sample_id
    identity_parts = identity_text.rsplit("_", 1)
    if len(identity_parts) == 2 and identity_parts[1].isdigit():
        subject_id, session_id = identity_parts
    else:
        subject_id, session_id = identity_text, ""

    sample_key = site + "/" + sample_id
    return site, int(label_text), sample_id, subject_id, session_id


def _adjacency_sequence(value: Any) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.dim() == 2:
            return [value]
        if value.dim() == 3:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError("adjacency tensor must have shape [N,N] or [M,N,N]")
    if isinstance(value, (list, tuple)) and value:
        if not all(torch.is_tensor(item) and item.dim() == 2 for item in value):
            raise ValueError("adjacency list items must be 2-D tensors")
        return list(value)
    raise ValueError("adjacency must be a non-empty tensor or tensor sequence")


def _vector_sequence(value: Any, expected_length: int, field_name: str) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.dim() == 1 and expected_length == 1:
            return [value]
        if value.dim() == 2 and value.shape[0] == expected_length:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError("{} does not match the graph sequence".format(field_name))
    if isinstance(value, (list, tuple)) and len(value) == expected_length:
        if not all(torch.is_tensor(item) and item.dim() == 1 for item in value):
            raise ValueError("{} items must be 1-D tensors".format(field_name))
        return list(value)
    raise ValueError("{} must be aligned to timepoints".format(field_name))


def _coordinate_sequence(
    value: Any, node_counts: Sequence[int]
) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        if value.dim() == 2 and len(set(node_counts)) == 1 and value.shape[0] == node_counts[0]:
            return [value for _ in node_counts]
        if value.dim() == 3 and value.shape[0] == len(node_counts):
            return [value[index] for index in range(value.shape[0])]
        raise ValueError("coords do not align with node counts")
    if isinstance(value, (list, tuple)) and len(value) == len(node_counts):
        if not all(torch.is_tensor(item) and item.dim() == 2 for item in value):
            raise ValueError("coordinate list items must be 2-D tensors")
        return list(value)
    raise ValueError("coords must be a tensor or time-aligned tensor sequence")


def _node_name_sequence(value: Any, node_counts: Sequence[int]) -> List[List[str]]:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, str) for item in value
    ):
        names = list(value)
        if len(set(node_counts)) != 1 or len(names) != node_counts[0]:
            raise ValueError("shared node_names do not align with node counts")
        return [names for _ in node_counts]

    if isinstance(value, (list, tuple)) and len(value) == len(node_counts):
        sequences = [list(item) for item in value]
        if not all(all(isinstance(name, str) for name in names) for names in sequences):
            raise ValueError("node_names entries must be strings")
        return sequences
    raise ValueError("node_names must be shared or time-aligned string sequences")


def _communities_are_contiguous(values: torch.Tensor) -> bool:
    unique = sorted(int(item) for item in torch.unique(values).tolist())
    return bool(unique) and unique == list(range(unique[-1] + 1))


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _excluded_record(
    path: Path,
    config: IndexBuildConfig,
    reasons: Iterable[str],
    label: Optional[int] = None,
) -> SampleRecord:
    try:
        site, parsed_label, sample_id, subject_id, session_id = _parse_identity(
            path, config.dataset_root
        )
        if label is None:
            label = parsed_label
    except ValueError:
        relative = _portable_relative_path(path, config.dataset_root)
        site = relative.split("/", 1)[0] if "/" in relative else ""
        sample_id = path.stem
        subject_id = sample_id
        session_id = ""

    return SampleRecord(
        sample_key=site + "/" + sample_id,
        sample_id=sample_id,
        site=site,
        subject_id=subject_id,
        session_id=session_id,
        label=label,
        relative_path=_portable_relative_path(path, config.dataset_root),
        num_timepoints=None,
        min_num_nodes=None,
        max_num_nodes=None,
        spatial_dim=None,
        coords_valid=False,
        community_valid=False,
        adjacency_valid=False,
        node_names_valid=False,
        window_starts_valid=False,
        has_positive_edges=False,
        has_negative_edges=False,
        empty_timepoints=None,
        edge_presence_threshold=config.edge_presence_threshold,
        source_global_threshold=None,
        repetition_time=None,
        included=False,
        exclusion_reasons="|".join(sorted(set(reasons))),
    )


def inspect_sample(path: Path, config: IndexBuildConfig) -> SampleRecord:
    """Load one sample on CPU and return validation metadata."""

    path = Path(path).resolve()
    reasons: List[str] = []
    try:
        site, label, sample_id, subject_id, session_id = _parse_identity(
            path, config.dataset_root
        )
    except ValueError as error:
        return _excluded_record(path, config, ["invalid_layout:" + str(error)])

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exact torch errors vary
        return _excluded_record(
            path,
            config,
            ["load_error:{}".format(type(error).__name__)],
            label=label,
        )

    if not isinstance(payload, dict):
        return _excluded_record(path, config, ["payload_not_dict"], label=label)

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return _excluded_record(
            path,
            config,
            ["missing_fields:" + ",".join(sorted(missing))],
            label=label,
        )

    try:
        graphs = _adjacency_sequence(payload["adjacency"])
    except ValueError as error:
        return _excluded_record(path, config, ["invalid_adjacency:" + str(error)], label)

    num_timepoints = len(graphs)
    node_counts = [int(graph.shape[0]) for graph in graphs]
    adjacency_valid = True
    has_positive_edges = False
    has_negative_edges = False
    empty_timepoints = 0
    threshold = config.edge_presence_threshold

    for graph in graphs:
        if graph.shape[0] == 0:
            adjacency_valid = False
            empty_timepoints += 1
            reasons.append("adjacency_empty_graph")
            continue
        if graph.shape[0] != graph.shape[1]:
            adjacency_valid = False
            reasons.append("adjacency_not_square")
            continue
        if not bool(torch.isfinite(graph).all()):
            adjacency_valid = False
            reasons.append("adjacency_nonfinite")
        if float((graph - graph.transpose(-1, -2)).abs().max().item()) > config.symmetry_tolerance:
            adjacency_valid = False
            reasons.append("adjacency_not_symmetric")
        if float(graph.diagonal().abs().max().item()) > config.diagonal_tolerance:
            adjacency_valid = False
            reasons.append("adjacency_has_self_loops")

        edge_mask = graph.abs() > threshold
        edge_mask = edge_mask & ~torch.eye(
            graph.shape[0], dtype=torch.bool, device=graph.device
        )
        if not bool(edge_mask.any()):
            empty_timepoints += 1
        has_positive_edges = has_positive_edges or bool((edge_mask & (graph > 0)).any())
        has_negative_edges = has_negative_edges or bool((edge_mask & (graph < 0)).any())

    if not adjacency_valid:
        reasons.append("invalid_adjacency")
    if empty_timepoints:
        reasons.append("empty_timepoints:{}".format(empty_timepoints))

    try:
        communities = _vector_sequence(
            payload["community_sequence"], num_timepoints, "community_sequence"
        )
        community_valid = True
        for community, node_count in zip(communities, node_counts):
            if community.numel() != node_count:
                community_valid = False
                reasons.append("community_node_count_mismatch")
                continue
            if community.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                community_valid = False
                reasons.append("community_not_integer")
            if bool((community < 0).any()):
                community_valid = False
                reasons.append("community_has_negative_label")
            if (
                config.require_contiguous_communities
                and not bool((community < 0).any())
                and not _communities_are_contiguous(community)
            ):
                community_valid = False
                reasons.append("community_labels_not_contiguous")
    except ValueError as error:
        communities = []
        community_valid = False
        reasons.append("invalid_community:" + str(error))

    if not community_valid:
        reasons.append("invalid_community")

    coords_valid = False
    spatial_dim = None
    if "coords" in payload:
        try:
            coordinates = _coordinate_sequence(payload["coords"], node_counts)
            spatial_dims = {int(coords.shape[1]) for coords in coordinates}
            coords_valid = len(spatial_dims) == 1
            for coords, node_count in zip(coordinates, node_counts):
                if coords.shape[0] != node_count or not bool(torch.isfinite(coords).all()):
                    coords_valid = False
            if not any(bool((coords != 0).any()) for coords in coordinates):
                coords_valid = False
            spatial_dim = next(iter(spatial_dims)) if len(spatial_dims) == 1 else None
        except ValueError:
            coords_valid = False
            spatial_dim = None

    try:
        name_sequences = _node_name_sequence(payload["node_names"], node_counts)
        node_names_valid = all(
            len(names) == node_count and len(set(names)) == len(names)
            for names, node_count in zip(name_sequences, node_counts)
        )
    except (TypeError, ValueError) as error:
        node_names_valid = False
        reasons.append("invalid_node_names:" + str(error))
    if not node_names_valid:
        reasons.append("invalid_node_names")

    window_starts = payload["window_starts"]
    window_starts_valid = bool(
        torch.is_tensor(window_starts)
        and window_starts.dim() == 1
        and window_starts.numel() == num_timepoints
        and (
            window_starts.numel() <= 1
            or bool((window_starts[1:] > window_starts[:-1]).all())
        )
    )
    if not window_starts_valid:
        reasons.append("invalid_window_starts")

    blocking_reasons = sorted(set(reasons))

    return SampleRecord(
        sample_key=site + "/" + sample_id,
        sample_id=sample_id,
        site=site,
        subject_id=subject_id,
        session_id=session_id,
        label=label,
        relative_path=_portable_relative_path(path, config.dataset_root),
        num_timepoints=num_timepoints,
        min_num_nodes=min(node_counts),
        max_num_nodes=max(node_counts),
        spatial_dim=spatial_dim,
        coords_valid=coords_valid,
        community_valid=community_valid,
        adjacency_valid=adjacency_valid,
        node_names_valid=node_names_valid,
        window_starts_valid=window_starts_valid,
        has_positive_edges=has_positive_edges,
        has_negative_edges=has_negative_edges,
        empty_timepoints=empty_timepoints,
        edge_presence_threshold=threshold,
        source_global_threshold=_safe_float(payload["global_threshold"]),
        repetition_time=_safe_float(payload["t_r"]),
        included=not blocking_reasons,
        exclusion_reasons="|".join(blocking_reasons),
    )


def build_sample_index(config: IndexBuildConfig) -> List[SampleRecord]:
    """Inspect all samples and enforce globally unique sample keys."""

    records = [inspect_sample(path, config) for path in discover_sample_files(config.dataset_root)]
    counts = Counter(record.sample_key for record in records)
    duplicates = {key for key, count in counts.items() if count > 1}
    if duplicates:
        updated: List[SampleRecord] = []
        for record in records:
            if record.sample_key not in duplicates:
                updated.append(record)
                continue
            row = record.to_dict()
            reasons = [item for item in record.exclusion_reasons.split("|") if item]
            reasons.append("duplicate_sample_key")
            row["included"] = False
            row["exclusion_reasons"] = "|".join(sorted(set(reasons)))
            updated.append(SampleRecord(**row))
        records = updated
    return records


def _atomic_write_csv(path: Path, records: Sequence[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(SampleRecord.__dataclass_fields__.keys())
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    os.replace(str(temporary), str(path))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def summarize_records(records: Sequence[SampleRecord]) -> Dict[str, Any]:
    included = [record for record in records if record.included]
    excluded = [record for record in records if not record.included]
    reason_counts: Counter = Counter()
    for record in excluded:
        reason_counts.update(item for item in record.exclusion_reasons.split("|") if item)

    return {
        "total_samples": len(records),
        "included_samples": len(included),
        "excluded_samples": len(excluded),
        "class_counts": {
            str(label): sum(record.label == label for record in included)
            for label in (0, 1)
        },
        "site_class_counts": {
            "{}/{}".format(site, label): count
            for (site, label), count in sorted(
                Counter((record.site, record.label) for record in included).items()
            )
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
    }


def write_index_artifacts(
    records: Sequence[SampleRecord], output_dir: Path
) -> Dict[str, Path]:
    """Write deterministic inventory, included index, exclusions, and summary."""

    output_dir = Path(output_dir).resolve()
    inventory_path = output_dir / "sample_inventory.csv"
    index_path = output_dir / "sample_index.csv"
    exclusion_path = output_dir / "exclusion_manifest.csv"
    summary_path = output_dir / "sample_index_summary.json"

    _atomic_write_csv(inventory_path, records)
    _atomic_write_csv(index_path, [record for record in records if record.included])
    _atomic_write_csv(exclusion_path, [record for record in records if not record.included])
    _atomic_write_json(summary_path, summarize_records(records))

    return {
        "inventory": inventory_path,
        "index": index_path,
        "exclusions": exclusion_path,
        "summary": summary_path,
    }
