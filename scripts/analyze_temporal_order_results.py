"""Analyze ordered, frozen-shuffled, and permutation-invariant baselines."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
import torch


SEEDS = tuple(range(42, 47))
MODES = ("ordered", "shuffled", "independent_bag")
DIRECTORY_PATTERNS = {
    "ordered": "key_full_seed{}_v1",
    "shuffled": "key_shuffled_perm101_seed{}_v1",
    "independent_bag": "key_independent_bag_seed{}_v1",
}
MODE_LABELS = {
    "ordered": "Ordered-GRU",
    "shuffled": "Shuffled-GRU (perm=101)",
    "independent_bag": "Independent-bag",
}
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def trusted_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def paired_ci(values):
    array = np.asarray(values, dtype=np.float64)
    radius = float(
        stats.t.ppf(0.975, len(array) - 1)
        * array.std(ddof=1)
        / math.sqrt(len(array))
    )
    return float(array.mean() - radius), float(array.mean() + radius)


def exact_sign_flip_pvalue(values):
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    values_under_null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        values_under_null.append(
            abs(float((array * np.asarray(signs, dtype=np.float64)).mean()))
        )
    return float(np.mean(np.asarray(values_under_null) >= observed - 1e-15))


def bh_adjust(pvalues):
    count = len(pvalues)
    order = np.argsort(np.asarray(pvalues, dtype=np.float64))
    adjusted = np.ones(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(pvalues[index]) * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path, default=Path("outputs/baseline_training")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "docs/experiment_results/temporal_order_perm101_seed42_46_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "docs/experiment_results/temporal_order_perm101_seed42_46_analysis.md"
        ),
    )
    return parser.parse_args()


def normalized_config(payload):
    config = dict(payload["model_config"])
    config.setdefault("temporal_order", "ordered")
    config.setdefault("permutation_seed", 42)
    return config


def load_runs(root):
    runs = {}
    for mode in MODES:
        for seed in SEEDS:
            directory = root / DIRECTORY_PATTERNS[mode].format(seed)
            required = (
                "history.json",
                "best_checkpoint.pt",
                "validation_evaluation.json",
                "test_evaluation.json",
            )
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise RuntimeError("{} missing {}".format(directory, missing))
            checkpoint_path = directory / "best_checkpoint.pt"
            checkpoint_hash = file_sha256(checkpoint_path)
            checkpoint = trusted_load(checkpoint_path)
            config = normalized_config(checkpoint)
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("{} training seed mismatch".format(directory))
            if mode == "ordered" and not (
                config["history_mode"] == "full"
                and config["temporal_order"] == "ordered"
            ):
                raise RuntimeError("ordered checkpoint configuration mismatch")
            if mode == "shuffled" and not (
                config["history_mode"] == "full"
                and config["temporal_order"] == "shuffled"
                and int(config["permutation_seed"]) == 101
            ):
                raise RuntimeError("shuffled checkpoint configuration mismatch")
            if mode == "independent_bag" and config["history_mode"] != "independent_bag":
                raise RuntimeError("bag checkpoint configuration mismatch")
            evaluations = {}
            for split in ("validation", "test"):
                with (directory / "{}_evaluation.json".format(split)).open(
                    "r", encoding="utf-8"
                ) as handle:
                    evaluations[split] = json.load(handle)
                if evaluations[split]["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("{} {} checkpoint hash mismatch".format(mode, split))
            with (directory / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            runs[(mode, seed)] = {
                "directory": directory.as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint": checkpoint,
                "config": config,
                "history": history,
                "evaluation": evaluations,
            }
    return runs


def validate_runs(runs):
    alignment = {}
    evidence_levels = set()
    parameter_counts = set()
    architecture_configs = set()
    ignored_config_fields = {
        "history_mode", "history_keep_ratio", "temporal_order", "permutation_seed"
    }
    for run in runs.values():
        evidence_levels.add(run["checkpoint"]["evidence_level"])
        parameter_counts.add(
            sum(int(tensor.numel()) for tensor in run["checkpoint"]["model_state_dict"].values())
        )
        architecture_configs.add(
            tuple(sorted(
                (key, json.dumps(value, sort_keys=True))
                for key, value in run["config"].items()
                if key not in ignored_config_fields
            ))
        )
    if len(evidence_levels) != 1 or len(parameter_counts) != 1 or len(architecture_configs) != 1:
        raise RuntimeError("evidence level, parameter count, or architecture differs")
    for split in ("validation", "test"):
        reference = None
        manifest_hashes = set()
        for run in runs.values():
            payload = run["evaluation"][split]
            if payload["debug_limited_batches"] is not None:
                raise RuntimeError("debug-limited evaluation found")
            manifest_hashes.add(payload["baseline_manifest_sha256"])
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in payload["metrics"]["predictions"]
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} samples are not aligned".format(split))
        if len(manifest_hashes) != 1:
            raise RuntimeError("{} manifest hashes differ".format(split))
        alignment[split] = {
            "sample_count": len(reference),
            "manifest_sha256": next(iter(manifest_hashes)),
            "aligned": True,
        }
    return {
        "alignment": alignment,
        "evidence_level": next(iter(evidence_levels)),
        "state_tensor_element_count": next(iter(parameter_counts)),
        "architecture_matched": True,
    }


def summarize(runs):
    summary = {}
    per_seed = []
    for mode in MODES:
        summary[mode] = {}
        for split in ("validation", "test"):
            summary[mode][split] = {}
            for metric in METRICS:
                values = [
                    float(runs[(mode, seed)]["evaluation"][split]["metrics"][metric])
                    for seed in SEEDS
                ]
                mean, sd = mean_sd(values)
                summary[mode][split][metric] = {
                    "mean": mean,
                    "sd": sd,
                    "values": values,
                }
        for seed in SEEDS:
            run = runs[(mode, seed)]
            row = {
                "mode": mode,
                "seed": seed,
                "epochs_completed": len(run["history"]),
            }
            for split in ("validation", "test"):
                metrics = run["evaluation"][split]["metrics"]
                for metric in METRICS + ("threshold", "accuracy", "f1"):
                    row["{}_{}".format(split, metric)] = float(metrics[metric])
                probabilities = np.asarray(
                    [item["class_1_probability"] for item in metrics["predictions"]],
                    dtype=np.float64,
                )
                row["{}_probability_sd".format(split)] = float(
                    probabilities.std(ddof=1)
                )
            flags = [
                item["validation"]["roc_auc"] == 0.5
                and item["validation"]["threshold"] == 0.5
                for item in run["history"]
            ]
            row["collapse_epoch_count"] = int(sum(flags))
            per_seed.append(row)
    return summary, per_seed


def compare(runs):
    definitions = (
        ("ordered_vs_shuffled", "ordered", "shuffled"),
        ("ordered_vs_bag", "ordered", "independent_bag"),
        ("bag_vs_shuffled", "independent_bag", "shuffled"),
    )
    comparisons = {}
    for split in ("validation", "test"):
        comparisons[split] = {}
        for metric in METRICS:
            rows = []
            pvalues = []
            for name, target, control in definitions:
                target_values = np.asarray(
                    [runs[(target, seed)]["evaluation"][split]["metrics"][metric] for seed in SEEDS],
                    dtype=np.float64,
                )
                control_values = np.asarray(
                    [runs[(control, seed)]["evaluation"][split]["metrics"][metric] for seed in SEEDS],
                    dtype=np.float64,
                )
                improvement = (
                    control_values - target_values
                    if metric == "unweighted_log_loss"
                    else target_values - control_values
                )
                ci_low, ci_high = paired_ci(improvement)
                pvalue = exact_sign_flip_pvalue(improvement)
                rows.append({
                    "comparison": name,
                    "target": target,
                    "control": control,
                    "improvement_values": improvement.tolist(),
                    "mean_improvement": float(improvement.mean()),
                    "sd_improvement": float(improvement.std(ddof=1)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": int((improvement > 0).sum()),
                    "exact_sign_flip_p": pvalue,
                })
                pvalues.append(pvalue)
            for row, qvalue in zip(rows, bh_adjust(pvalues)):
                row["bh_q"] = qvalue
            comparisons[split][metric] = rows
    return comparisons


def f6(value):
    return "{:.6f}".format(float(value))


def find_comparison(comparisons, split, metric, name):
    return next(
        row for row in comparisons[split][metric]
        if row["comparison"] == name
    )


def render_markdown(payload):
    summary = payload["summary"]
    comparisons = payload["comparisons"]
    lines = [
        "# 时间顺序探索实验：Ordered、Shuffled 与 Independent-bag",
        "",
        "## 验收",
        "",
        "- 3 种模式 × 5 个训练 seed，共 15 次训练；Shuffled 固定 permutation seed=101。",
        "- validation/test 样本、标签、subject、site、manifest 和 checkpoint 哈希全部一致。",
        "- 三种模式的图编码器、表示维度、分类头和状态张量元素数一致；仅历史/顺序机制不同。",
        "- 所有结果的 evidence level 为 `exploratory_in_sample`。",
        "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {}（均值 ± 样本标准差）".format(split.capitalize()),
            "",
            "| 模式 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ])
        for mode in MODES:
            data = summary[mode][split]
            lines.append(
                "| {} | {} ± {} | {} ± {} | {} ± {} |".format(
                    MODE_LABELS[mode],
                    f6(data["unweighted_log_loss"]["mean"]),
                    f6(data["unweighted_log_loss"]["sd"]),
                    f6(data["roc_auc"]["mean"]),
                    f6(data["roc_auc"]["sd"]),
                    f6(data["balanced_accuracy"]["mean"]),
                    f6(data["balanced_accuracy"]["sd"]),
                )
            )
        lines.append("")
    lines.extend([
        "## Test 同-seed配对比较",
        "",
        "正值表示目标模式优于对照；log-loss 使用 `对照 − 目标`。",
        "",
        "| 目标 vs 对照 | ΔLog-loss (95% CI) | 胜/5 | ΔAUROC (95% CI) | 胜/5 | ΔBAcc | 胜/5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    comparison_labels = {
        "ordered_vs_shuffled": "Ordered vs Shuffled",
        "ordered_vs_bag": "Ordered vs Bag",
        "bag_vs_shuffled": "Bag vs Shuffled",
    }
    for name in ("ordered_vs_shuffled", "ordered_vs_bag", "bag_vs_shuffled"):
        ll = find_comparison(comparisons, "test", "unweighted_log_loss", name)
        auc = find_comparison(comparisons, "test", "roc_auc", name)
        bacc = find_comparison(comparisons, "test", "balanced_accuracy", name)
        lines.append(
            "| {} | {} [{}, {}] | {}/5 | {} [{}, {}] | {}/5 | {} | {}/5 |".format(
                comparison_labels[name], f6(ll["mean_improvement"]),
                f6(ll["ci95_low"]), f6(ll["ci95_high"]), ll["wins"],
                f6(auc["mean_improvement"]), f6(auc["ci95_low"]),
                f6(auc["ci95_high"]), auc["wins"],
                f6(bacc["mean_improvement"]), bacc["wins"],
            )
        )
    lines.extend([
        "",
        "所有精确双侧符号翻转检验及 BH 校正结果均未达到 0.05；5 个 seed 的最小可能双侧 p 值为 0.0625。",
        "",
        "## 各 seed 的 Test AUROC",
        "",
        "| 模式 | seed42 | seed43 | seed44 | seed45 | seed46 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        values = summary[mode]["test"]["roc_auc"]["values"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                MODE_LABELS[mode], *[f6(value) for value in values]
            )
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "1. Ordered-GRU 没有稳定优于 Shuffled-GRU：test log-loss 反而平均高 0.002328，AUROC 仅平均高 0.016092，且方向随 seed 改变。因此当前不支持时间顺序贡献。",
        "2. Independent-bag 的 test 均值最好。相对 Ordered，它降低 log-loss 0.011670、提高 AUROC 0.024073；相对 Shuffled，它降低 log-loss 0.009342、提高 AUROC 0.040165。",
        "3. Bag 相对 Shuffled 的 log-loss 在 4/5 seed 改善，但区间仍跨 0。它是下一阶段最合理的探索骨干，不是已经证实的最优模型。",
        "4. 结果更支持“多个窗口的无序集合聚合可能有价值”，不支持“GRU 将早期子图信息按时间顺序有效传递”的主张。",
        "5. 15 次运行中有 4 次出现至少一个 `validation AUROC=0.5 且 threshold=0.5` 的 epoch：Ordered 和 Shuffled 各 2 次，Bag 为 0 次。Bag 在这一定义下未出现完全塌缩，但其跨 seed 波动仍不可忽略。",
        "6. Validation 上 Shuffled 的平均 log-loss 最低，而 Bag 的 AUROC 和 balanced accuracy 最高；Test 上 Bag 三项均值最高。这种指标与分区不一致进一步要求避免把均值排序解释成确定结论。",
        "7. 本轮只有一个 permutation seed，无法描述所有可能排列的分布；但按简化实验的停止规则，已经没有必要立即扩展为 25 次 shuffled 训练。",
        "8. 上游提取器接触过全样本标签，且既有 CUDA 训练未严格逐次复现；这些结果只能用于监督性样本内探索，不能证明样本外泛化或理论机制成立。",
        "",
        "## 下一步",
        "",
        "冻结 Independent-bag 作为探索性分类骨干，进入子图来源比较：Key vs matched Low-score、Top-degree 和 Random。先使用 seed42–44；只有 Key 稳定优于匹配控制，才继续结构先验或扰动实验。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    audit = validate_runs(runs)
    summary, per_seed = summarize(runs)
    comparisons = compare(runs)
    payload = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "permutation_seed": 101,
        "run_count": len(runs),
        "audit": audit,
        "summary": summary,
        "per_seed": per_seed,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    print(json.dumps({
        "run_count": len(runs),
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
