"""Read-only Stage-0 audit of cached multi-object S/V/G artifacts."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.multiview_critical import read_multiview_manifest  # noqa: E402


def _summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": values[0],
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": values[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, records = read_multiview_manifest(args.manifest, PROJECT_ROOT)
    object_counts, object_nodes, object_edges = [], [], []
    positive_edges, negative_edges, k_differences = [], [], []
    transported_mass, unmatched_source, unmatched_target = [], [], []
    matched_cost, transition_cost, convergence = [], [], []
    singleton_count, object_total = 0, 0
    for record in records:
        valid_windows = [item for item in record.features.hard_windows if item is not None]
        for window in valid_windows:
            object_counts.append(len(window.objects))
            object_total += len(window.objects)
            for item in window.objects:
                nodes = int(item.adjacency.shape[0])
                mask = torch.triu(item.adjacency.abs() > 0.0, diagonal=1)
                values = item.adjacency[mask]
                object_nodes.append(nodes)
                object_edges.append(int(values.numel()))
                positive_edges.append(int((values > 0.0).sum()))
                negative_edges.append(int((values < 0.0).sum()))
                singleton_count += int(nodes == 1)
        for left, right in zip(valid_windows[:-1], valid_windows[1:]):
            k_differences.append(abs(len(right.objects) - len(left.objects)))
        for item in record.features.transitions:
            if item is None:
                continue
            plan = item.transport_plan
            total = float(plan.sum())
            transported_mass.append(total)
            source_window = record.features.hard_windows[item.source_index]
            target_window = record.features.hard_windows[item.target_index]
            source = plan.new_tensor([obj.mass for obj in source_window.objects])
            target = plan.new_tensor([obj.mass for obj in target_window.objects])
            source = source / source.sum().clamp_min(1.0e-8)
            target = target / target.sum().clamp_min(1.0e-8)
            unmatched_source.append(float((source - plan.sum(dim=1)).clamp_min(0.0).sum()))
            unmatched_target.append(float((target - plan.sum(dim=0)).clamp_min(0.0).sum()))
            matched_cost.append(float((plan * item.object_cost).sum() / plan.sum().clamp_min(1.0e-8)))
            transition_cost.extend(float(value) for value in item.object_cost.flatten().tolist())
            convergence.append(bool(item.solver_converged))
    elapsed = [float(row.get("precompute_seconds", 0.0)) for row in manifest["records"]]
    peak = [float(row.get("peak_memory_mib", 0.0)) for row in manifest["records"]]
    byte_sizes = [
        (PROJECT_ROOT / row["feature_path"]).stat().st_size
        for row in manifest["records"]
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "multiview_critical_stage0_audit",
        "split": manifest["split"],
        "sample_count": len(records),
        "window_object_count": _summary(object_counts),
        "object_node_count": _summary(object_nodes),
        "object_edge_count": _summary(object_edges),
        "positive_edge_count_per_object": _summary(positive_edges),
        "negative_edge_count_per_object": _summary(negative_edges),
        "singleton_object_fraction": float(singleton_count) / float(max(1, object_total)),
        "adjacent_window_object_count_difference": _summary(k_differences),
        "uot_transported_mass": _summary(transported_mass),
        "uot_unmatched_source_mass": _summary(unmatched_source),
        "uot_unmatched_target_mass": _summary(unmatched_target),
        "transport_weighted_cost": _summary(matched_cost),
        "all_object_pair_cost": _summary(transition_cost),
        "fgw_convergence_fraction": float(sum(convergence)) / float(max(1, len(convergence))),
        "precompute_seconds_per_sample": _summary(elapsed),
        "peak_memory_mib_per_sample": _summary(peak),
        "artifact_mebibytes_per_sample": _summary([value / (1024.0 ** 2) for value in byte_sizes]),
        "total_artifact_mebibytes": sum(byte_sizes) / (1024.0 ** 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
