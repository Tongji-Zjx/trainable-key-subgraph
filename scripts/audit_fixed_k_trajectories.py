"""Audit fixed-K hard objects and dynamic birth/death trajectories.

This command is read-only.  It validates immutable Stage-0 artifacts and
summarizes whether every valid window contains exactly K objects while the
global number and duration of trajectories remain sample dependent.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.multiview_critical import read_multiview_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _summary(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _overlap(left, right):
    denominator = min(len(left), len(right))
    return len(left & right) / float(denominator) if denominator else 0.0


def _object_sets(item):
    nodes = tuple(int(value) for value in item.union_node_indices.tolist())
    node_set = set(nodes)
    edge_set = set()
    adjacency = item.adjacency
    for left, right in (adjacency != 0.0).triu(diagonal=1).nonzero().tolist():
        edge_set.add(tuple(sorted((nodes[int(left)], nodes[int(right)]))))
    return node_set, edge_set


def build_report(manifest_path, project_root=PROJECT_ROOT):
    payload, records = read_multiview_manifest(manifest_path, project_root)
    if payload.get("feature_config", {}).get("object_decomposition") not in (
        "selector_fixed_k_v1",
        "selector_fixed_k_diverse_v2",
        "selector_multi_object_v3",
    ):
        raise ValueError("trajectory audit requires selector fixed-K artifacts")
    window_counts = []
    trajectory_counts = []
    birth_counts = []
    death_counts = []
    track_lengths = []
    continuation_confidences = []
    pairwise_node_overlaps = []
    pairwise_edge_overlaps = []
    unique_node_fractions = []
    union_efficiencies = []
    k_values = set()
    failures = []
    for record in records:
        sample = record.features
        trajectory_set = getattr(sample, "trajectory_set", None)
        if trajectory_set is None:
            failures.append("{}:missing_trajectory_set".format(sample.sample_key))
            continue
        k = int(trajectory_set.active_subgraphs_per_valid_window)
        k_values.add(k)
        valid_windows = int(sample.window_mask.sum())
        window_counts.append(valid_windows)
        trajectory_counts.append(trajectory_set.trajectory_count)
        birth_counts.append(trajectory_set.total_birth_count)
        death_counts.append(
            sum(len(item.death_track_ids) for item in trajectory_set.assignments)
        )
        lengths = [item.length for item in trajectory_set.trajectories]
        track_lengths.extend(lengths)
        expected_observations = valid_windows * k
        if sum(lengths) != expected_observations:
            failures.append("{}:trajectory_coverage".format(sample.sample_key))
        for window_index, assignment in enumerate(trajectory_set.assignments):
            expected = k if bool(sample.window_mask[window_index]) else 0
            if sample.hard_windows[window_index] is not None:
                objects = sample.hard_windows[window_index].objects
                if len(objects) != expected:
                    failures.append(
                        "{}:window_{}_object_count".format(
                            sample.sample_key, window_index
                        )
                    )
                object_sets = [_object_sets(item) for item in objects]
                union_nodes = set()
                total_membership = 0
                for object_index, (nodes, edges) in enumerate(object_sets):
                    total_membership += len(nodes)
                    unique_node_fractions.append(
                        len(nodes - union_nodes) / float(max(1, len(nodes)))
                    )
                    union_nodes.update(nodes)
                    for prior in range(object_index):
                        pairwise_node_overlaps.append(
                            _overlap(nodes, object_sets[prior][0])
                        )
                        pairwise_edge_overlaps.append(
                            _overlap(edges, object_sets[prior][1])
                        )
                union_efficiencies.append(
                    len(union_nodes) / float(max(1, total_membership))
                )
            active = assignment.track_ids[:expected]
            if (
                bool((active < 0).any())
                or len(set(int(value) for value in active.tolist())) != expected
            ):
                failures.append(
                    "{}:window_{}_track_slots".format(
                        sample.sample_key, window_index
                    )
                )
            continued = assignment.continuation_from >= 0
            continuation_confidences.extend(
                assignment.match_confidence[continued].tolist()
            )
    if len(k_values) != 1:
        failures.append("manifest:inconsistent_k={}".format(sorted(k_values)))
    report = {
        "artifact": "fixed_k_dynamic_trajectory_audit",
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "split": payload["split"],
        "sample_count": len(records),
        "critical_subgraph_count": next(iter(k_values)) if len(k_values) == 1 else None,
        "checks": {
            "all_valid_windows_have_exactly_k_objects": not failures,
            "trajectory_observation_coverage_complete": not failures,
            "active_track_ids_unique_within_window": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
        "valid_window_count_per_sample": _summary(window_counts),
        "global_trajectory_count_per_sample": _summary(trajectory_counts),
        "new_birth_count_per_sample": _summary(birth_counts),
        "death_count_per_sample": _summary(death_counts),
        "trajectory_length": _summary(track_lengths),
        "continuation_confidence": _summary(continuation_confidences),
        "pairwise_node_overlap": _summary(pairwise_node_overlaps),
        "pairwise_edge_overlap": _summary(pairwise_edge_overlaps),
        "unique_node_fraction": _summary(unique_node_fractions),
        "union_efficiency": _summary(union_efficiencies),
    }
    return report


def _write_json(path, report, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("trajectory audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _write_markdown(path, report, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("trajectory audit report already exists")
    trajectory = report["global_trajectory_count_per_sample"]
    length = report["trajectory_length"]
    births = report["new_birth_count_per_sample"]
    node_overlap = report["pairwise_node_overlap"]
    edge_overlap = report["pairwise_edge_overlap"]
    unique = report["unique_node_fraction"]
    efficiency = report["union_efficiency"]

    def value(item, key, digits=4):
        current = item[key]
        if current is None:
            return "N/A"
        if digits == 0:
            return "{:.0f}".format(current)
        return "{:.4f}".format(current)

    lines = [
        "# Fixed-K 动态关键子图轨迹审计",
        "",
        "- Split：`{}`".format(report["split"]),
        "- 样本数：{}".format(report["sample_count"]),
        "- 每个有效窗口 K：{}".format(report["critical_subgraph_count"]),
        "- 审计通过：{}".format(
            "是" if report["checks"]["failure_count"] == 0 else "否"
        ),
        "",
        "| 统计量 | 均值 | 中位数 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
        "| 每样本动态全局轨迹数 | {} | {} | {} | {} |".format(
            value(trajectory, "mean"), value(trajectory, "median"),
            value(trajectory, "minimum", 0), value(trajectory, "maximum", 0),
        ),
        "| 每样本新增轨迹数 n | {} | {} | {} | {} |".format(
            value(births, "mean"), value(births, "median"),
            value(births, "minimum", 0), value(births, "maximum", 0),
        ),
        "| 轨迹长度 | {} | {} | {} | {} |".format(
            value(length, "mean"), value(length, "median"),
            value(length, "minimum", 0), value(length, "maximum", 0),
        ),
        "| 子图两两节点重叠 | {} | {} | {} | {} |".format(
            value(node_overlap, "mean"), value(node_overlap, "median"),
            value(node_overlap, "minimum"), value(node_overlap, "maximum"),
        ),
        "| 子图两两边重叠 | {} | {} | {} | {} |".format(
            value(edge_overlap, "mean"), value(edge_overlap, "median"),
            value(edge_overlap, "minimum"), value(edge_overlap, "maximum"),
        ),
        "| 每个子图新增节点比例 | {} | {} | {} | {} |".format(
            value(unique, "mean"), value(unique, "median"),
            value(unique, "minimum"), value(unique, "maximum"),
        ),
        "| 硬并图唯一节点效率 | {} | {} | {} | {} |".format(
            value(efficiency, "mean"), value(efficiency, "median"),
            value(efficiency, "minimum"), value(efficiency, "maximum"),
        ),
        "",
        "> 每个有效窗口固定保留 K 个对象；全局轨迹数为 K+n，轨迹长度允许小于样本窗口数。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    report = build_report(args.manifest)
    output_dir = Path(args.output_dir).resolve()
    _write_json(output_dir / "trajectory_audit.json", report, args.overwrite)
    _write_markdown(output_dir / "trajectory_audit.md", report, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["checks"]["failure_count"]:
        raise ValueError("fixed-K trajectory audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
