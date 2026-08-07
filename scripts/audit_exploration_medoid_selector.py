"""Read-only formal audit of real-medoid exploration and hard continuity."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import os
import statistics
import sys
from itertools import combinations
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
from keysubgraph.diagnostics.retrospective_consensus import (  # noqa: E402
    best_object_assignment,
    jaccard,
    transition_phase,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_types import (  # noqa: E402
    DualSTSEHardSGWConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-train-samples", type=int, default=24)
    parser.add_argument("--max-validation-samples", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _sets(output):
    nodes = set(
        int(value)
        for value in torch.nonzero(
            output.hard_node_mask.detach().cpu(), as_tuple=False
        ).flatten().tolist()
    )
    mask = output.hard_edge_mask.detach().cpu().to(torch.bool)
    edges = {
        (int(left), int(right))
        for left, right in torch.nonzero(
            torch.triu(mask, diagonal=1), as_tuple=False
        ).tolist()
    }
    return nodes, edges


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _summary(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / float(len(finite)),
        "median": statistics.median(finite),
        "standard_deviation": (
            statistics.pstdev(finite) if len(finite) > 1 else 0.0
        ),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def _aggregate(rows):
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in ("window",)
        }
    )
    return {key: _summary([row[key] for row in rows if key in row]) for key in keys}


def _mean_pairwise(objects):
    node_values = []
    edge_values = []
    for left, right in combinations(objects, 2):
        node_values.append(jaccard(left[0], right[0]))
        edge_values.append(jaccard(left[1], right[1]))
    return (
        sum(node_values) / float(len(node_values)) if node_values else 0.0,
        sum(edge_values) / float(len(edge_values)) if edge_values else 0.0,
    )


def _mean_same_slot(left, right):
    count = min(len(left), len(right))
    if count < 1:
        return 0.0, 0.0
    return (
        sum(jaccard(left[index][0], right[index][0]) for index in range(count))
        / float(count),
        sum(jaccard(left[index][1], right[index][1]) for index in range(count))
        / float(count),
    )


def _metric(diagnostics, name):
    value = diagnostics.get(name, 0.0)
    return _scalar(value)


def _audit_sample(model, batch, split, config):
    output = model.selector(
        batch,
        selection_mode="learned",
        track_subgraphs=False,
    )
    sample_key = batch.sample_keys[0]
    windows = []
    unions = []
    within_rows = []
    for window_index, (items, union) in enumerate(
        zip(output.hard_subgraphs[0], output.hard_windows[0])
    ):
        objects = tuple(
            _sets(item)
            for item in items
            if item is not None and item.window_valid
        )
        windows.append(objects)
        unions.append(_sets(union) if union.window_valid else None)
        pair_node, pair_edge = _mean_pairwise(objects)
        total_nodes = sum(len(item[0]) for item in objects)
        union_nodes = len(set().union(*(item[0] for item in objects))) if objects else 0
        within_rows.append(
            {
                "split": split,
                "sample_key": sample_key,
                "window": window_index,
                "pairwise_object_node_jaccard": pair_node,
                "pairwise_object_edge_jaccard": pair_edge,
                "node_redundancy": (
                    1.0 - union_nodes / float(total_nodes)
                    if total_nodes else 0.0
                ),
            }
        )
    exploration_windows = int(output.diagnostics["exploration_window_count"][0])
    transition_rows = []
    for right in range(1, len(windows)):
        if unions[right - 1] is None or unions[right] is None:
            continue
        same_node, same_edge = _mean_same_slot(
            windows[right - 1], windows[right]
        )
        best = best_object_assignment(windows[right - 1], windows[right])
        transition_rows.append(
            {
                "split": split,
                "sample_key": sample_key,
                "window": right,
                "phase": transition_phase(
                    right,
                    exploration_windows,
                    config.selector_exploration_history_ramp_windows,
                ),
                "union_node_jaccard": jaccard(
                    unions[right - 1][0], unions[right][0]
                ),
                "union_edge_jaccard": jaccard(
                    unions[right - 1][1], unions[right][1]
                ),
                "same_slot_node_jaccard": same_node,
                "same_slot_edge_jaccard": same_edge,
                "best_possible_node_jaccard": float(
                    best["mean_node_jaccard"]
                ),
                "best_possible_edge_jaccard": float(
                    best["mean_edge_jaccard"]
                ),
                "slot_assignment_gap": float(best["mean_node_jaccard"])
                - same_node,
            }
        )
    diagnostics = output.diagnostics
    sample_row = {
        "split": split,
        "sample_key": sample_key,
        "window_count": len(windows),
        "exploration_window_count": exploration_windows,
        "candidate_pool_size": _metric(
            diagnostics, "mean_exploration_candidate_pool_size"
        ),
        "shortlist_size": _metric(
            diagnostics, "mean_exploration_shortlist_size"
        ),
        "anchor_support_rate": _metric(
            diagnostics, "mean_exploration_anchor_support"
        ),
        "unsupported_anchor_rate": _metric(
            diagnostics, "mean_exploration_unsupported_anchor_rate"
        ),
        "anchor_pair_similarity": _metric(
            diagnostics, "mean_exploration_anchor_similarity"
        ),
        "cluster_similarity_including_self": _metric(
            diagnostics, "mean_exploration_cluster_similarity"
        ),
        "cross_window_cluster_similarity": _metric(
            diagnostics,
            "mean_exploration_cross_window_cluster_similarity",
        ),
        "nearest_cross_window_candidate_similarity": _metric(
            diagnostics,
            "mean_exploration_nearest_cross_window_similarity",
        ),
        "medoid_objective": _metric(
            diagnostics, "mean_exploration_medoid_objective"
        ),
        "legacy_soft_consensus_weight": _metric(
            diagnostics, "mean_exploration_consensus_confidence"
        ),
    }
    return sample_row, within_rows, transition_rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _fmt(value):
    return "N/A" if value is None else "{:.4f}".format(float(value))


def _write_report(path, report):
    overall = report["overall"]
    sample = overall["sample_metrics"]
    transition = overall["transition_metrics"]
    within = overall["within_window_metrics"]
    lines = [
        "# 探索期真实 Medoid Selector 正式审计",
        "",
        "- 参数更新量：0（冻结短程训练最佳 checkpoint）",
        "- 使用 split：train、validation；test 未使用",
        "- 探索初始化：真实候选池 → 多样短名单 → K 个真实 medoid",
        "- 旧软平均共识：关闭",
        "- 样本数：{}".format(report["sample_count"]),
        "",
        "## 探索期初始化",
        "",
        "| 指标 | 均值 | 中位数 |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("候选池大小", "candidate_pool_size"),
        ("代表短名单大小", "shortlist_size"),
        ("跨窗口支持率", "anchor_support_rate"),
        ("无跨窗口支持锚点率", "unsupported_anchor_rate"),
        ("Medoid 间相似度", "anchor_pair_similarity"),
        ("跨窗口簇内相似度（排除自身）", "cross_window_cluster_similarity"),
        ("候选最近跨窗口相似度", "nearest_cross_window_candidate_similarity"),
        ("Medoid 目标值", "medoid_objective"),
        ("旧软共识权重", "legacy_soft_consensus_weight"),
    ):
        values = sample[key]
        lines.append(
            "| {} | {} | {} |".format(
                label, _fmt(values["mean"]), _fmt(values["median"])
            )
        )
    lines.extend(
        [
            "",
            "## 子图多样性与相邻窗口连续性",
            "",
            "| 指标 | 均值 | 中位数 |",
            "|---|---:|---:|",
        ]
    )
    for label, values in (
        ("同窗口对象节点 Jaccard", within["pairwise_object_node_jaccard"]),
        ("同窗口节点冗余率", within["node_redundancy"]),
        ("硬并图相邻窗口节点 Jaccard", transition["union_node_jaccard"]),
        ("同槽位对象节点 Jaccard", transition["same_slot_node_jaccard"]),
        ("最佳可能对象节点 Jaccard", transition["best_possible_node_jaccard"]),
        ("槽位分配差距", transition["slot_assignment_gap"]),
        ("同槽位对象边 Jaccard", transition["same_slot_edge_jaccard"]),
    ):
        lines.append(
            "| {} | {} | {} |".format(
                label, _fmt(values["mean"]), _fmt(values["median"])
            )
        )
    lines.extend(
        [
            "",
            "> 跨窗口簇内相似度已排除 medoid 与自身的 1.0 相似度。",
            "> 本报告只审计结构与连续性，不使用 test，也不重新训练或调阈值。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_train_samples < 1 or args.max_validation_samples < 1:
        raise ValueError("formal audit sample limits must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("medoid audit output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    payload = _trusted_load(args.checkpoint.resolve(), torch.device("cpu"))
    config = DualSTSEHardSGWConfig(**payload["model_config"])
    if not config.selector_structural_temporal_memory:
        raise ValueError("audit requires structural temporal memory")
    device = torch.device(args.device)
    model = DualSTSEHardSGWClassifier(config).to(device).eval()
    model.load_state_dict(payload["model_state_dict"])
    paths = protocol["paths"]
    common = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    sample_rows = []
    within_rows = []
    transition_rows = []
    split_limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
    }
    with torch.no_grad():
        for split, limit in split_limits.items():
            dataset = ExactSTSEDataset(
                *common,
                split,
                protocol["edge_presence_threshold"],
                require_coordinates=True,
                node_name_policy=protocol_node_name_policy(protocol),
            )
            loader = create_exact_stse_loader(
                dataset,
                1,
                seed=args.seed,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            for index, batch in enumerate(loader):
                if index >= limit:
                    break
                sample, local_within, local_transitions = _audit_sample(
                    model, batch.to(device), split, config
                )
                sample_rows.append(sample)
                within_rows.extend(local_within)
                transition_rows.extend(local_transitions)
                print(
                    "audited {} {}/{} {}".format(
                        split, index + 1, min(limit, len(dataset)), batch.sample_keys[0]
                    ),
                    flush=True,
                )
    def grouped(rows, split):
        return _aggregate([row for row in rows if row["split"] == split])

    report = {
        "artifact": "exploration_real_medoid_formal_audit",
        "protocol_sha256": file_sha256(args.protocol.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "best_validation_roc_auc": payload.get("best_validation_roc_auc"),
        "sample_count": len(sample_rows),
        "test_used": False,
        "config": {
            "critical_subgraph_count": config.critical_subgraph_count,
            "candidate_similarity_threshold": (
                config.selector_exploration_candidate_similarity_threshold
            ),
            "shortlist_multiplier": (
                config.selector_exploration_shortlist_multiplier
            ),
        },
        "overall": {
            "sample_metrics": _aggregate(sample_rows),
            "within_window_metrics": _aggregate(within_rows),
            "transition_metrics": _aggregate(transition_rows),
        },
        "by_split": {
            split: {
                "sample_metrics": grouped(sample_rows, split),
                "within_window_metrics": grouped(within_rows, split),
                "transition_metrics": grouped(transition_rows, split),
            }
            for split in split_limits
        },
    }
    _write_csv(output_dir / "sample_metrics.csv", sample_rows)
    _write_csv(output_dir / "within_window_metrics.csv", within_rows)
    _write_csv(output_dir / "transition_metrics.csv", transition_rows)
    _write_json(output_dir / "summary.json", report)
    _write_report(output_dir / "report.md", report)
    print("audit output: {}".format(output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
