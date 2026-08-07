"""Read-only diagnosis of retrospective consensus and temporal continuity.

The command replays one frozen selector checkpoint under six configuration
conditions.  It never updates weights and never uses the test split.  The
diagnostic separates selector instability from tracker rejection and checks
whether the exploration consensus remains realizable in actual windows.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.diagnostics.retrospective_consensus import (  # noqa: E402
    accepted_assignment_metrics,
    aggregate_records,
    best_object_assignment,
    jaccard,
    summarize,
    transition_phase,
)
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


CONDITIONS = (
    (
        "independent",
        {
            "selector_structural_temporal_memory": False,
            "selector_object_temporal_state": False,
            "selector_exploration_consensus_enabled": False,
            "selector_confidence_gated_history": False,
        },
        "Each window is selected independently; no temporal memory.",
    ),
    (
        "A_recursive_legacy",
        {
            "selector_exploration_consensus_enabled": False,
            "selector_confidence_gated_history": False,
        },
        "Pre-consensus recursive structural memory.",
    ),
    (
        "B_consensus_immediate",
        {
            "selector_exploration_consensus_enabled": True,
            "selector_exploration_retrospective_strength": 0.30,
            "selector_exploration_history_ramp_windows": 1,
            "selector_confidence_gated_history": False,
        },
        "Consensus with immediate full post-exploration history.",
    ),
    (
        "C_consensus_ramp",
        {
            "selector_exploration_consensus_enabled": True,
            "selector_exploration_retrospective_strength": 0.30,
            "selector_exploration_history_ramp_windows": 4,
            "selector_confidence_gated_history": False,
        },
        "Consensus with four-window ramp and no confidence gate.",
    ),
    (
        "D_current_default",
        {
            "selector_exploration_consensus_enabled": True,
            "selector_exploration_retrospective_strength": 0.30,
            "selector_exploration_history_ramp_windows": 4,
            "selector_confidence_gated_history": True,
        },
        "Current consensus, ramp, and confidence-gated history.",
    ),
    (
        "E_stronger_retrospective",
        {
            "selector_exploration_consensus_enabled": True,
            "selector_exploration_retrospective_strength": 0.60,
            "selector_exploration_history_ramp_windows": 4,
            "selector_confidence_gated_history": True,
        },
        "Current flow with stronger retrospective history.",
    ),
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


def _write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("retrospective diagnostic output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _sets(output) -> Tuple[set, set]:
    nodes = set(
        int(value)
        for value in torch.nonzero(
            output.hard_node_mask.detach().cpu(), as_tuple=False
        ).flatten().tolist()
    )
    edges = set()
    mask = output.hard_edge_mask.detach().cpu().to(torch.bool)
    for left, right in torch.nonzero(
        torch.triu(mask, diagonal=1), as_tuple=False
    ).tolist():
        edges.add((int(left), int(right)))
    return nodes, edges


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _exploration_count(config: DualSTSEHardSGWConfig, windows: int) -> int:
    requested = int(__import__("math").ceil(
        float(config.selector_exploration_fraction) * int(windows)
    ))
    return min(
        int(windows),
        max(
            int(config.selector_exploration_min_windows),
            min(int(config.selector_exploration_max_windows), requested),
        ),
    )


def _capture_sample(output, sample_key: str, config) -> Dict[str, Any]:
    objects = []
    unions = []
    for current_objects, union in zip(
        output.hard_subgraphs[0], output.hard_windows[0]
    ):
        objects.append(tuple(
            _sets(item)
            for item in current_objects
            if item is not None and item.window_valid
        ))
        unions.append(_sets(union) if union is not None and union.window_valid else None)
    trajectory = output.trajectory_sets[0]
    assignments = []
    for item in trajectory.assignments:
        assignments.append({
            "continuation_from": tuple(int(v) for v in item.continuation_from.tolist()),
            "birth_mask": tuple(bool(v) for v in item.birth_mask.tolist()),
            "match_confidence": tuple(float(v) for v in item.match_confidence.tolist()),
        })
    diagnostics = output.diagnostics
    return {
        "sample_key": sample_key,
        "objects": tuple(objects),
        "unions": tuple(unions),
        "assignments": tuple(assignments),
        "exploration_windows": _exploration_count(config, len(objects)),
        "trajectory_count": int(trajectory.trajectory_count),
        "birth_count": int(trajectory.total_birth_count),
        "selector_diagnostics": {
            "mean_history_strength": _scalar(diagnostics["mean_history_strength"]),
            "mean_slot_alignment_confidence": _scalar(
                diagnostics["mean_slot_alignment_confidence"]
            ),
            "mean_memory_update_gate": _scalar(
                diagnostics["mean_memory_update_gate"]
            ),
            "mean_exploration_consensus_confidence": _scalar(
                diagnostics["mean_exploration_consensus_confidence"]
            ),
        },
    }


def _condition_metrics(samples: Sequence[Mapping[str, Any]], config) -> Dict[str, Any]:
    all_transitions: List[Dict[str, float]] = []
    phases: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in (
            "exploration_internal", "exploration_boundary", "history_ramp", "steady_state"
        )
    }
    sample_dynamics = []
    selector_diagnostics: Dict[str, List[float]] = {}
    for sample in samples:
        objects = sample["objects"]
        unions = sample["unions"]
        assignments = sample["assignments"]
        for name, value in sample["selector_diagnostics"].items():
            selector_diagnostics.setdefault(name, []).append(float(value))
        for right in range(1, len(objects)):
            if unions[right - 1] is None or unions[right] is None:
                continue
            best = best_object_assignment(objects[right - 1], objects[right])
            accepted = accepted_assignment_metrics(
                objects[right - 1],
                objects[right],
                assignments[right]["continuation_from"],
            )
            accepted_confidence = [
                confidence
                for previous, confidence in zip(
                    assignments[right]["continuation_from"],
                    assignments[right]["match_confidence"],
                )
                if int(previous) >= 0
            ]
            record = {
                "union_node_jaccard": jaccard(unions[right - 1][0], unions[right][0]),
                "union_edge_jaccard": jaccard(unions[right - 1][1], unions[right][1]),
                "best_possible_node_jaccard": float(best["mean_node_jaccard"]),
                "best_possible_edge_jaccard": float(best["mean_edge_jaccard"]),
                **accepted,
                "tracker_gap_node_jaccard": (
                    float(best["mean_node_jaccard"])
                    - float(accepted["coverage_adjusted_node_jaccard"])
                ),
                "birth_rate": sum(assignments[right]["birth_mask"])
                / float(max(1, len(objects[right]))),
                "accepted_match_confidence": (
                    sum(accepted_confidence) / float(len(accepted_confidence))
                    if accepted_confidence else 0.0
                ),
            }
            all_transitions.append(record)
            phase = transition_phase(
                right,
                sample["exploration_windows"],
                config.selector_exploration_history_ramp_windows,
            )
            phases[phase].append(record)
        sample_dynamics.append({
            "trajectory_count": float(sample["trajectory_count"]),
            "birth_count": float(sample["birth_count"]),
        })
    return {
        "overall_transitions": aggregate_records(all_transitions),
        "phase_transitions": {
            name: aggregate_records(records) for name, records in phases.items()
        },
        "sample_dynamics": aggregate_records(sample_dynamics),
        "selector_diagnostics": {
            name: summarize(values) for name, values in selector_diagnostics.items()
        },
        "transition_count": len(all_transitions),
    }


def _realizability(
    condition_samples: Sequence[Mapping[str, Any]],
    independent_samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    independent_lookup = {item["sample_key"]: item for item in independent_samples}
    records = []
    per_sample = []
    for sample in condition_samples:
        reference = independent_lookup[sample["sample_key"]]
        local = []
        for index in range(min(sample["exploration_windows"], len(sample["objects"]))):
            if sample["unions"][index] is None or reference["unions"][index] is None:
                continue
            best = best_object_assignment(reference["objects"][index], sample["objects"][index])
            record = {
                "object_node_jaccard_to_independent": float(best["mean_node_jaccard"]),
                "object_edge_jaccard_to_independent": float(best["mean_edge_jaccard"]),
                "union_node_jaccard_to_independent": jaccard(
                    reference["unions"][index][0], sample["unions"][index][0]
                ),
                "union_edge_jaccard_to_independent": jaccard(
                    reference["unions"][index][1], sample["unions"][index][1]
                ),
            }
            records.append(record)
            local.append(record)
        if local:
            per_sample.append({
                key: sum(item[key] for item in local) / float(len(local))
                for key in local[0]
            })
    return {
        "window_level": aggregate_records(records),
        "sample_level": aggregate_records(per_sample),
        "window_count": len(records),
    }


def _screen(report: Dict[str, Any]) -> List[Dict[str, str]]:
    flags = []
    current = report["conditions"]["D_current_default"]
    overall = current["metrics"]["overall_transitions"]
    best = float(overall["best_possible_node_jaccard"]["mean"] or 0.0)
    acceptance = float(overall["acceptance_rate"]["mean"] or 0.0)
    gap = float(overall["tracker_gap_node_jaccard"]["mean"] or 0.0)
    realizability = float(
        current["exploration_realizability"]["window_level"]
        ["object_node_jaccard_to_independent"]["mean"] or 0.0
    )
    if best < 0.55:
        flags.append({
            "code": "selector_instability",
            "evidence": "best possible adjacent-object node Jaccard is {:.3f}".format(best),
        })
    if acceptance < 0.75 and gap > 0.15:
        flags.append({
            "code": "tracker_rejection",
            "evidence": "acceptance {:.3f}, best-to-accepted gap {:.3f}".format(acceptance, gap),
        })
    if realizability < 0.55:
        flags.append({
            "code": "consensus_low_realizability",
            "evidence": "exploration objects agree with independent windows by only {:.3f}".format(realizability),
        })
    default_union = float(overall["union_node_jaccard"]["mean"] or 0.0)
    immediate_union = float(
        report["conditions"]["B_consensus_immediate"]["metrics"]
        ["overall_transitions"]["union_node_jaccard"]["mean"] or 0.0
    )
    stronger_union = float(
        report["conditions"]["E_stronger_retrospective"]["metrics"]
        ["overall_transitions"]["union_node_jaccard"]["mean"] or 0.0
    )
    if max(immediate_union, stronger_union) > default_union + 0.03:
        flags.append({
            "code": "history_influence_too_weak",
            "evidence": "stronger/immediate history improves union Jaccard by {:.3f}".format(
                max(immediate_union, stronger_union) - default_union
            ),
        })
    return flags


def _fmt(value) -> str:
    return "N/A" if value is None else "{:.4f}".format(float(value))


def _write_markdown(path: Path, report: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("retrospective diagnostic report already exists")
    lines = [
        "# Retrospective Exploration Consensus 只读诊断",
        "",
        "- 样本：{} 个 `{}` 样本".format(report["sample_count"], report["split"]),
        "- checkpoint 参数更新：0",
        "- test 使用：否",
        "",
        "## 冻结流程消融",
        "",
        "| 条件 | Union node Jaccard | Best object Jaccard | 接受率 | 出生率 | 轨迹数/样本 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _, description in CONDITIONS:
        current = report["conditions"][name]
        metrics = current["metrics"]["overall_transitions"]
        dynamics = current["metrics"]["sample_dynamics"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                name,
                _fmt(metrics["union_node_jaccard"]["mean"]),
                _fmt(metrics["best_possible_node_jaccard"]["mean"]),
                _fmt(metrics["acceptance_rate"]["mean"]),
                _fmt(metrics["birth_rate"]["mean"]),
                _fmt(dynamics["trajectory_count"]["mean"]),
            )
        )
    lines.extend(["", "## 当前默认流程的分阶段结果", ""])
    current = report["conditions"]["D_current_default"]["metrics"]
    lines.extend([
        "| 阶段 | N | Union node Jaccard | Best object Jaccard | 接受率 | Tracker gap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for phase, metrics in current["phase_transitions"].items():
        count = metrics.get("union_node_jaccard", {}).get("count", 0)
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                phase,
                count,
                _fmt(metrics.get("union_node_jaccard", {}).get("mean")),
                _fmt(metrics.get("best_possible_node_jaccard", {}).get("mean")),
                _fmt(metrics.get("acceptance_rate", {}).get("mean")),
                _fmt(metrics.get("tracker_gap_node_jaccard", {}).get("mean")),
            )
        )
    lines.extend(["", "## 探索期共识可实现性", ""])
    lines.extend([
        "| 条件 | 对独立对象 node Jaccard | 对独立对象 edge Jaccard | Union node Jaccard |",
        "|---|---:|---:|---:|",
    ])
    for name, _, _ in CONDITIONS:
        metrics = report["conditions"][name]["exploration_realizability"]["window_level"]
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                name,
                _fmt(metrics["object_node_jaccard_to_independent"]["mean"]),
                _fmt(metrics["object_edge_jaccard_to_independent"]["mean"]),
                _fmt(metrics["union_node_jaccard_to_independent"]["mean"]),
            )
        )
    lines.extend(["", "## 自动筛查结论", ""])
    if report["screening_flags"]:
        for item in report["screening_flags"]:
            lines.append("- `{}`：{}".format(item["code"], item["evidence"]))
    else:
        lines.append("- 未触发预定义瓶颈阈值；需结合完整 JSON 检查分阶段分布。")
    lines.extend([
        "",
        "> Best object Jaccard 是忽略出生/死亡阈值后的最优二分匹配上限；若它本身很低，问题在 selector 输出，而非后续匹配器。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main() -> int:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    protocol_path = Path(args.protocol).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("bottleneck diagnostic output exists")
    protocol = validate_data_protocol(protocol_path, PROJECT_ROOT)
    protocol_sha256 = file_sha256(protocol_path)
    payload = _trusted_load(checkpoint_path, torch.device("cpu"))
    base_config = DualSTSEHardSGWConfig(**payload["model_config"])
    if base_config.selector_architecture != "theory_multi_object":
        raise ValueError("diagnostic requires theory_multi_object selector")
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
    cpu_batches = []
    for index, batch in enumerate(loader):
        if index >= args.max_samples:
            break
        cpu_batches.append(batch)
    device = torch.device(args.device)
    condition_samples: Dict[str, List[Dict[str, Any]]] = {}
    condition_configs: Dict[str, DualSTSEHardSGWConfig] = {}
    for condition_index, (name, overrides, description) in enumerate(CONDITIONS):
        config = replace(base_config, **overrides)
        condition_configs[name] = config
        # Always construct the exact checkpoint architecture.  In particular,
        # disabling structural memory for the independent-forward condition
        # must not remove the trained memory-gate tensors from the state dict.
        model = DualSTSEHardSGWClassifier(base_config).to(device)
        load_dual_checkpoint(
            checkpoint_path,
            model,
            device,
            expected_stage="selector_proxy",
            expected_protocol_sha256=protocol_sha256,
        )
        # These switches alter only forward-time information flow; no module
        # or learned tensor is added, removed, or reinitialized.
        model.selector.config = config
        if hasattr(model.selector.scorer, "confidence_gated_history"):
            model.selector.scorer.confidence_gated_history = bool(
                config.selector_confidence_gated_history
            )
        model.eval()
        samples = []
        for index, cpu_batch in enumerate(cpu_batches):
            with torch.no_grad():
                selected = model.selector(
                    cpu_batch.to(device),
                    selection_mode="learned",
                    random_seed=args.seed,
                    track_subgraphs=True,
                )
            samples.append(_capture_sample(selected, cpu_batch[0].sample_key, config))
            print(
                "condition {}/{} {} sample {}/{} {}".format(
                    condition_index + 1,
                    len(CONDITIONS),
                    name,
                    index + 1,
                    len(cpu_batches),
                    cpu_batch[0].sample_key,
                ),
                flush=True,
            )
        condition_samples[name] = samples
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    independent = condition_samples["independent"]
    report: Dict[str, Any] = {
        "artifact": "retrospective_exploration_consensus_diagnostic",
        "split": args.split,
        "sample_count": len(cpu_batches),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "protocol_sha256": protocol_sha256,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "device": str(device),
        "parameter_updates": 0,
        "test_used": False,
        "base_model_config": asdict(base_config),
        "conditions": {},
    }
    for name, overrides, description in CONDITIONS:
        report["conditions"][name] = {
            "description": description,
            "overrides": overrides,
            "metrics": _condition_metrics(condition_samples[name], condition_configs[name]),
            "exploration_realizability": _realizability(
                condition_samples[name], independent
            ),
        }
    report["screening_flags"] = _screen(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "diagnostic.json", report, args.overwrite)
    _write_markdown(output_dir / "report.md", report, args.overwrite)
    print("diagnostic output: {}".format(output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
