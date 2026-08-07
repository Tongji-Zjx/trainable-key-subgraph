"""Run a lightweight, read-only audit of learned multi-object trajectories.

Unlike the Stage-0 multi-view precomputation, this command audits the selector's
own fixed-K objects and ROI/coordinate-aware trajectories directly.  It does
not compute downstream Exact-SGW/UOT features and never updates model weights.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_types import (  # noqa: E402
    DualSTSEHardSGWConfig,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


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


def _ratio_overlap(left, right):
    denominator = min(len(left), len(right))
    return len(left & right) / float(denominator) if denominator else 0.0


def _jaccard(left, right):
    union = left | right
    return len(left & right) / float(len(union)) if union else 1.0


def _sets(output):
    nodes = set(
        int(value)
        for value in torch.nonzero(
            output.hard_node_mask.detach().cpu(), as_tuple=False
        ).flatten().tolist()
    )
    edges = set()
    edge_mask = output.hard_edge_mask.detach().cpu().to(torch.bool)
    for left, right in torch.nonzero(
        torch.triu(edge_mask, diagonal=1), as_tuple=False
    ).tolist():
        edges.add((int(left), int(right)))
    return nodes, edges


def _write_json(path, payload, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("trajectory audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _format(summary, key="mean"):
    value = summary.get(key)
    return "N/A" if value is None else "{:.4f}".format(float(value))


def _write_markdown(path, report, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError("trajectory audit report already exists")
    metrics = report["metrics"]
    lines = [
        "# Theory Multi-Object Selector 轨迹初步审计",
        "",
        "- Split：`{}`".format(report["split"]),
        "- 样本数：{}".format(report["sample_count"]),
        "- 每窗子图数 K：{}".format(report["critical_subgraph_count"]),
        "- checkpoint epoch：{}".format(report["checkpoint_epoch"]),
        "- 结构校验：{}".format("通过" if report["checks"]["passed"] else "失败"),
        "",
        "| 指标 | 均值 | 中位数 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = (
        ("窗内两两节点重叠率", "within_window_pairwise_node_overlap"),
        ("窗内两两边重叠率", "within_window_pairwise_edge_overlap"),
        ("硬并图唯一节点效率", "union_unique_node_efficiency"),
        ("相邻窗口硬并图节点 Jaccard", "adjacent_union_node_jaccard"),
        ("相邻窗口硬并图边 Jaccard", "adjacent_union_edge_jaccard"),
        ("同轨迹相邻节点 Jaccard", "matched_track_node_jaccard"),
        ("同轨迹相邻边 Jaccard", "matched_track_edge_jaccard"),
        ("全局轨迹数/样本", "trajectory_count_per_sample"),
        ("新出生数/样本", "new_birth_count_per_sample"),
        ("轨迹长度", "trajectory_length"),
        ("单窗口轨迹比例/样本", "singleton_trajectory_fraction_per_sample"),
        ("最长轨迹覆盖率/样本", "longest_track_window_fraction_per_sample"),
        ("轨迹延续率/样本", "continuation_rate_per_sample"),
    )
    for label, key in rows:
        item = metrics[key]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                label,
                _format(item, "mean"),
                _format(item, "median"),
                _format(item, "minimum"),
                _format(item, "maximum"),
            )
        )
    lines.extend(
        [
            "",
            "## 初筛标记",
            "",
        ]
    )
    if report["screening_flags"]:
        lines.extend("- `{}`".format(item) for item in report["screening_flags"])
    else:
        lines.append("- 未触发预定义高重叠或高破碎风险标记。")
    lines.extend(
        [
            "",
            "> 本报告直接审计冻结 selector 的输出；不计算下游 Exact-SGW/UOT，未更新任何参数，也未使用 test。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("trajectory audit max-samples must be positive")
    protocol_path = Path(args.protocol).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    protocol = validate_data_protocol(protocol_path, PROJECT_ROOT)
    protocol_sha256 = file_sha256(protocol_path)
    checkpoint_payload = _trusted_load(checkpoint_path, torch.device("cpu"))
    config = DualSTSEHardSGWConfig(**checkpoint_payload["model_config"])
    if config.selector_architecture != "theory_multi_object":
        raise ValueError("trajectory audit requires the theory multi-object selector")
    if not config.selector_object_temporal_state:
        raise ValueError("trajectory audit requires object temporal state")

    paths = protocol["paths"]
    dataset = ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
        require_coordinates=True,
        node_name_policy=protocol_node_name_policy(protocol),
    )
    loader = create_exact_stse_loader(
        dataset,
        batch_size=1,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    device = torch.device(args.device)
    model = DualSTSEHardSGWClassifier(config).to(device)
    load_dual_checkpoint(
        checkpoint_path,
        model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha256,
    )
    model.eval()

    within_node = []
    within_edge = []
    union_efficiency = []
    adjacent_union_node = []
    adjacent_union_edge = []
    track_node = []
    track_edge = []
    trajectory_counts = []
    birth_counts = []
    track_lengths = []
    singleton_fractions = []
    longest_fractions = []
    continuation_rates = []
    confidence_values = []
    failures = []
    sample_reports = []
    k = int(config.critical_subgraph_count)

    for sample_index, cpu_batch in enumerate(loader):
        if sample_index >= args.max_samples:
            break
        with torch.no_grad():
            selected = model.selector(
                cpu_batch.to(device),
                selection_mode="learned",
                random_seed=args.seed,
                track_subgraphs=True,
            )
        objects_by_window = selected.hard_subgraphs[0]
        unions_by_window = selected.hard_windows[0]
        trajectories = selected.trajectory_sets[0]
        sample_key = cpu_batch[0].sample_key
        if trajectories is None:
            failures.append(sample_key + ":missing_trajectory_set")
            continue

        object_sets_by_window = []
        union_sets_by_window = []
        valid_windows = 0
        for window_index, (objects, union) in enumerate(
            zip(objects_by_window, unions_by_window)
        ):
            valid = [item for item in objects if item is not None and item.window_valid]
            if union is None or not union.window_valid:
                object_sets_by_window.append(())
                union_sets_by_window.append(None)
                if valid:
                    failures.append(
                        "{}:window_{}_objects_without_union".format(sample_key, window_index)
                    )
                continue
            valid_windows += 1
            if len(valid) != k:
                failures.append(
                    "{}:window_{}_object_count={}".format(
                        sample_key, window_index, len(valid)
                    )
                )
            sets = tuple(_sets(item) for item in valid)
            object_sets_by_window.append(sets)
            union_sets = _sets(union)
            union_sets_by_window.append(union_sets)
            total_membership = sum(len(item[0]) for item in sets)
            unique_nodes = set().union(*(item[0] for item in sets)) if sets else set()
            union_efficiency.append(
                len(unique_nodes) / float(max(1, total_membership))
            )
            if unique_nodes != union_sets[0]:
                failures.append(
                    "{}:window_{}_union_node_mismatch".format(sample_key, window_index)
                )
            for left in range(len(sets)):
                for right in range(left + 1, len(sets)):
                    within_node.append(_ratio_overlap(sets[left][0], sets[right][0]))
                    within_edge.append(_ratio_overlap(sets[left][1], sets[right][1]))

        for left, right in zip(union_sets_by_window[:-1], union_sets_by_window[1:]):
            if left is None or right is None:
                continue
            adjacent_union_node.append(_jaccard(left[0], right[0]))
            adjacent_union_edge.append(_jaccard(left[1], right[1]))

        lengths = [int(item.length) for item in trajectories.trajectories]
        trajectory_counts.append(int(trajectories.trajectory_count))
        birth_counts.append(int(trajectories.total_birth_count))
        track_lengths.extend(lengths)
        singleton_fractions.append(
            sum(value == 1 for value in lengths) / float(max(1, len(lengths)))
        )
        longest_fractions.append(
            max(lengths, default=0) / float(max(1, valid_windows))
        )
        observation_count = sum(lengths)
        continuation_count = max(0, observation_count - len(lengths))
        possible_continuations = max(0, (valid_windows - 1) * k)
        continuation_rates.append(
            continuation_count / float(max(1, possible_continuations))
        )
        for assignment in trajectories.assignments:
            continued = assignment.continuation_from >= 0
            confidence_values.extend(
                float(value)
                for value in assignment.match_confidence[continued].tolist()
            )
        for trajectory in trajectories.trajectories:
            observations = list(
                zip(trajectory.window_indices, trajectory.object_indices)
            )
            for (left_window, left_object), (right_window, right_object) in zip(
                observations[:-1], observations[1:]
            ):
                if right_window != left_window + 1:
                    failures.append(sample_key + ":trajectory_gap")
                    continue
                left_sets = object_sets_by_window[left_window][left_object]
                right_sets = object_sets_by_window[right_window][right_object]
                track_node.append(_jaccard(left_sets[0], right_sets[0]))
                track_edge.append(_jaccard(left_sets[1], right_sets[1]))
        expected_observations = valid_windows * k
        if observation_count != expected_observations:
            failures.append(sample_key + ":trajectory_coverage")
        sample_reports.append(
            {
                "sample_key": sample_key,
                "valid_window_count": valid_windows,
                "trajectory_count": int(trajectories.trajectory_count),
                "new_birth_count": int(trajectories.total_birth_count),
                "trajectory_lengths": lengths,
                "singleton_fraction": singleton_fractions[-1],
                "longest_track_window_fraction": longest_fractions[-1],
                "continuation_rate": continuation_rates[-1],
            }
        )
        print(
            "audited {}/{} {} windows={} trajectories={} births={}".format(
                len(sample_reports),
                min(len(dataset), args.max_samples),
                sample_key,
                valid_windows,
                trajectories.trajectory_count,
                trajectories.total_birth_count,
            ),
            flush=True,
        )

    metrics = {
        "within_window_pairwise_node_overlap": _summary(within_node),
        "within_window_pairwise_edge_overlap": _summary(within_edge),
        "union_unique_node_efficiency": _summary(union_efficiency),
        "adjacent_union_node_jaccard": _summary(adjacent_union_node),
        "adjacent_union_edge_jaccard": _summary(adjacent_union_edge),
        "matched_track_node_jaccard": _summary(track_node),
        "matched_track_edge_jaccard": _summary(track_edge),
        "trajectory_count_per_sample": _summary(trajectory_counts),
        "new_birth_count_per_sample": _summary(birth_counts),
        "trajectory_length": _summary(track_lengths),
        "singleton_trajectory_fraction_per_sample": _summary(singleton_fractions),
        "longest_track_window_fraction_per_sample": _summary(longest_fractions),
        "continuation_rate_per_sample": _summary(continuation_rates),
        "match_confidence": _summary(confidence_values),
    }
    flags = []
    node_overlap_mean = metrics["within_window_pairwise_node_overlap"]["mean"]
    track_node_mean = metrics["matched_track_node_jaccard"]["mean"]
    singleton_mean = metrics["singleton_trajectory_fraction_per_sample"]["mean"]
    if node_overlap_mean is not None and node_overlap_mean > 0.40:
        flags.append("high_within_window_object_overlap")
    if track_node_mean is not None and track_node_mean < 0.30:
        flags.append("low_matched_track_node_continuity")
    if singleton_mean is not None and singleton_mean > 0.25:
        flags.append("high_singleton_trajectory_fraction")

    report = {
        "artifact": "theory_multi_object_selector_trajectory_audit",
        "split": args.split,
        "sample_count": len(sample_reports),
        "critical_subgraph_count": k,
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "checkpoint_best_epoch": int(checkpoint_payload.get("best_epoch", -1)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "protocol_sha256": protocol_sha256,
        "device": str(device),
        "checks": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
        "metrics": metrics,
        "screening_flags": flags,
        "samples": sample_reports,
    }
    output_dir = Path(args.output_dir).resolve()
    _write_json(output_dir / "trajectory_audit.json", report, args.overwrite)
    _write_markdown(output_dir / "trajectory_audit.md", report, args.overwrite)
    print("audit output: {}".format(output_dir), flush=True)
    if failures:
        raise ValueError("trajectory audit structural checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
