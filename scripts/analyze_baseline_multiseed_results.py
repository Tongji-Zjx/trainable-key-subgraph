"""Summarize paired multi-seed baseline history-mode experiments."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats


MODES = (
    "full",
    "current_only",
    "truncate_025",
    "truncate_050",
    "truncate_075",
    "independent_bag",
)
MODE_LABELS = {
    "full": "Full",
    "current_only": "Current-only",
    "truncate_025": "Truncate-25%",
    "truncate_050": "Truncate-50%",
    "truncate_075": "Truncate-75%",
    "independent_bag": "Independent-bag",
}
METRICS = (
    "unweighted_log_loss",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
    "f1",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def paired_ci(values, confidence=0.95):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean
    radius = float(
        stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1)
        * array.std(ddof=1)
        / math.sqrt(len(array))
    )
    return mean - radius, mean + radius


def exact_sign_flip_pvalue(values):
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        permuted.append(abs(float((array * np.asarray(signs)).mean())))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


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


def longest_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path, default=Path("outputs/baseline_training")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "docs/experiment_results/history_modes_seed42_46_multiseed_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "docs/experiment_results/history_modes_seed42_46_multiseed_analysis.md"
        ),
    )
    return parser.parse_args()


def load_runs(root):
    pattern = re.compile(
        r"^key_(full|current_only|truncate_025|truncate_050|truncate_075|independent_bag)_seed(42|43|44|45|46)_v1$"
    )
    runs = {}
    for directory in root.iterdir():
        match = pattern.match(directory.name) if directory.is_dir() else None
        if match is None:
            continue
        mode, seed_text = match.groups()
        seed = int(seed_text)
        required = (
            "history.json",
            "best_checkpoint.pt",
            "validation_evaluation.json",
            "test_evaluation.json",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise RuntimeError("{} missing {}".format(directory, missing))
        evaluations = {}
        for split in ("validation", "test"):
            path = directory / "{}_evaluation.json".format(split)
            with path.open("r", encoding="utf-8") as handle:
                evaluations[split] = json.load(handle)
        with (directory / "history.json").open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        actual_checkpoint_hash = file_sha256(directory / "best_checkpoint.pt")
        for split, payload in evaluations.items():
            if payload["checkpoint_sha256"] != actual_checkpoint_hash:
                raise RuntimeError("{} {} checkpoint hash mismatch".format(mode, split))
        runs[(mode, seed)] = {
            "directory": directory.as_posix(),
            "history": history,
            "evaluation": evaluations,
            "checkpoint_sha256": actual_checkpoint_hash,
        }
    expected = set((mode, seed) for mode in MODES for seed in range(42, 47))
    if set(runs) != expected:
        raise RuntimeError(
            "experiment inventory mismatch: missing={} extra={}".format(
                sorted(expected - set(runs)), sorted(set(runs) - expected)
            )
        )
    return runs


def validate_alignment(runs):
    result = {}
    for split in ("validation", "test"):
        reference = None
        manifest_hashes = set()
        for run in runs.values():
            payload = run["evaluation"][split]
            manifest_hashes.add(payload["baseline_manifest_sha256"])
            predictions = payload["metrics"]["predictions"]
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in predictions
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} prediction rows are not aligned".format(split))
        if len(manifest_hashes) != 1:
            raise RuntimeError("{} manifest hashes differ".format(split))
        result[split] = {
            "sample_count": len(reference),
            "manifest_sha256": next(iter(manifest_hashes)),
            "aligned_across_all_runs": True,
        }
    return result


def analyze(runs):
    summary = {}
    per_seed = []
    collapse = []
    for mode in MODES:
        summary[mode] = {}
        for split in ("validation", "test"):
            summary[mode][split] = {}
            for metric in METRICS:
                values = [
                    float(runs[(mode, seed)]["evaluation"][split]["metrics"][metric])
                    for seed in range(42, 47)
                ]
                mean, sd = mean_sd(values)
                summary[mode][split][metric] = {
                    "mean": mean,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }
        for seed in range(42, 47):
            run = runs[(mode, seed)]
            row = {"mode": mode, "seed": seed, "epochs_completed": len(run["history"])}
            for split in ("validation", "test"):
                metrics = run["evaluation"][split]["metrics"]
                for metric in METRICS + ("threshold",):
                    row["{}_{}".format(split, metric)] = float(metrics[metric])
                probabilities = np.asarray(
                    [item["class_1_probability"] for item in metrics["predictions"]],
                    dtype=np.float64,
                )
                row["{}_probability_sd".format(split)] = float(
                    probabilities.std(ddof=1)
                )
            per_seed.append(row)
            flags = [
                item["validation"]["roc_auc"] == 0.5
                and item["validation"]["threshold"] == 0.5
                for item in run["history"]
            ]
            collapse.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "epochs_completed": len(flags),
                    "collapse_epoch_count": int(sum(flags)),
                    "longest_consecutive_collapse": longest_true_run(flags),
                }
            )

    paired = {}
    controls = [mode for mode in MODES if mode != "full"]
    for split in ("validation", "test"):
        paired[split] = {}
        for metric in ("unweighted_log_loss", "roc_auc", "balanced_accuracy"):
            comparisons = []
            pvalues = []
            for mode in controls:
                control = np.asarray(
                    [
                        runs[(mode, seed)]["evaluation"][split]["metrics"][metric]
                        for seed in range(42, 47)
                    ],
                    dtype=np.float64,
                )
                full = np.asarray(
                    [
                        runs[("full", seed)]["evaluation"][split]["metrics"][metric]
                        for seed in range(42, 47)
                    ],
                    dtype=np.float64,
                )
                improvement = full - control if metric == "unweighted_log_loss" else control - full
                ci_low, ci_high = paired_ci(improvement)
                pvalue = exact_sign_flip_pvalue(improvement)
                comparisons.append(
                    {
                        "mode": mode,
                        "improvement_values": improvement.tolist(),
                        "mean_improvement": float(improvement.mean()),
                        "sd_improvement": float(improvement.std(ddof=1)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "exact_sign_flip_p": pvalue,
                        "wins": int((improvement > 0).sum()),
                        "ties": int((improvement == 0).sum()),
                    }
                )
                pvalues.append(pvalue)
            for comparison, qvalue in zip(comparisons, bh_adjust(pvalues)):
                comparison["bh_q"] = qvalue
            paired[split][metric] = comparisons
    return summary, per_seed, paired, collapse


def f6(value):
    return "{:.6f}".format(float(value))


def render_markdown(payload):
    summary = payload["summary"]
    paired = payload["paired_comparisons_vs_full"]
    diagnostics = payload["descriptive_diagnostics"]
    lines = [
        "# 基线历史模式多随机种子对比（seed 42–46）",
        "",
        "## 验收与口径",
        "",
        "- 共验收 6 种模式 × 5 个 seed = 30 次训练。",
        "- 所有运行的 validation/test 样本顺序、标签、subject、site 和 manifest 哈希一致。",
        "- 同一 seed 内各模式与 Full 配对；log-loss 改善定义为 `Full − 对照`，其余指标改善定义为 `对照 − Full`，因此正值均表示对照模式更好。",
        "- checkpoint 按 validation unweighted log-loss 选择；test 未参与模型或阈值选择。",
        "",
        "## 跨 seed 表现（均值 ± 样本标准差）",
        "",
        "### Validation",
        "",
        "| 模式 | Log-loss | AUROC | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        data = summary[mode]["validation"]
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
    lines.extend(
        [
            "",
            "### Test",
            "",
            "| 模式 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        data = summary[mode]["test"]
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
    lines.extend(
        [
            "",
            "## Test：相对 Full 的同-seed配对改善",
            "",
            "| 模式 | ΔLog-loss | 胜/5 | 95% CI | p / BH-q | ΔAUROC | 胜/5 | 95% CI | p / BH-q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ll_by_mode = {item["mode"]: item for item in paired["test"]["unweighted_log_loss"]}
    auc_by_mode = {item["mode"]: item for item in paired["test"]["roc_auc"]}
    for mode in MODES[1:]:
        ll = ll_by_mode[mode]
        auc = auc_by_mode[mode]
        lines.append(
            "| {} | {} | {}/5 | [{}, {}] | {} / {} | {} | {}/5 | [{}, {}] | {} / {} |".format(
                MODE_LABELS[mode],
                f6(ll["mean_improvement"]), ll["wins"], f6(ll["ci95_low"]),
                f6(ll["ci95_high"]), f6(ll["exact_sign_flip_p"]), f6(ll["bh_q"]),
                f6(auc["mean_improvement"]), auc["wins"], f6(auc["ci95_low"]),
                f6(auc["ci95_high"]), f6(auc["exact_sign_flip_p"]), f6(auc["bh_q"]),
            )
        )
    lines.extend(
        [
            "",
            "注：只有 5 个配对 seed，精确双侧符号翻转检验的最小可能 p 值为 0.0625；因此本轮不能仅凭 p 值确认显著性。",
            "",
            "## 主要发现",
            "",
            "1. **Independent-bag 的跨 seed 平均表现最好。** 相对 Full，它在 validation 上平均降低 log-loss {}（相对降低 {:.2f}%）、提高 AUROC {}；在 test 上平均降低 log-loss {}（相对降低 {:.2f}%）、提高 AUROC {}、提高 balanced accuracy {}。它的 validation AUROC 与 balanced accuracy 均为 5/5 seed 改善，test AUROC 和 balanced accuracy 均为 4/5 seed 改善。".format(
                f6(paired["validation"]["unweighted_log_loss"][-1]["mean_improvement"]),
                100.0 * paired["validation"]["unweighted_log_loss"][-1]["mean_improvement"] / summary["full"]["validation"]["unweighted_log_loss"]["mean"],
                f6(paired["validation"]["roc_auc"][-1]["mean_improvement"]),
                f6(paired["test"]["unweighted_log_loss"][-1]["mean_improvement"]),
                100.0 * paired["test"]["unweighted_log_loss"][-1]["mean_improvement"] / summary["full"]["test"]["unweighted_log_loss"]["mean"],
                f6(paired["test"]["roc_auc"][-1]["mean_improvement"]),
                f6(paired["test"]["balanced_accuracy"][-1]["mean_improvement"]),
            ),
            "2. **但统计证据仍不足。** Independent-bag 相对 Full 的 test log-loss 95% CI 为 [{}, {}]，test AUROC 95% CI 为 [{}, {}]，均跨越 0；所有 BH-q 也均未达到 0.05。".format(
                f6(paired["test"]["unweighted_log_loss"][-1]["ci95_low"]),
                f6(paired["test"]["unweighted_log_loss"][-1]["ci95_high"]),
                f6(paired["test"]["roc_auc"][-1]["ci95_low"]),
                f6(paired["test"]["roc_auc"][-1]["ci95_high"]),
            ),
            "3. **seed42 的排序不可推广。** 多 seed 后，Full 的平均 test AUROC 高于 Current-only 和三个截断模式；只有 Independent-bag 仍高于 Full。历史保留比例从 25%→50%→75%→100% 没有形成单调剂量关系。",
            "4. **没有得到递归历史传递有效的稳定证据。** Full 在 test AUROC 上平均优于 Current-only，提示历史信息可能有用；但 validation 方向相反，而且 Independent-bag（使用全部窗口但不传递递归状态）整体最好。因此当前结果更支持“跨窗口聚合可能有用”，尚不支持“递归状态传递本身必要或有效”。",
            "5. **概率校准和排序均较弱。** test 类别先验恒定预测的 log-loss 为 {}；Full 均值为 {}，Independent-bag 为 {}。只有 Independent-bag 在模式均值上明显优于该朴素参照。30 次运行中 validation AUROC 与 test AUROC 的描述性 Pearson 相关为 {}，说明 validation 排序表现没有稳定映射到 test。".format(
                f6(diagnostics["test_constant_prevalence_log_loss"]),
                f6(summary["full"]["test"]["unweighted_log_loss"]["mean"]),
                f6(summary["independent_bag"]["test"]["unweighted_log_loss"]["mean"]),
                f6(diagnostics["validation_test_auc_pearson_r"]),
            ),
            "",
            "## 各 seed 的 Test AUROC",
            "",
            "| 模式 | seed42 | seed43 | seed44 | seed45 | seed46 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        values = summary[mode]["test"]["roc_auc"]["values"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                MODE_LABELS[mode], *[f6(value) for value in values]
            )
        )
    collapsed = [
        item for item in payload["collapse_diagnostics"]
        if item["collapse_epoch_count"] > 0
    ]
    lines.extend(
        [
            "",
            "## 稳定性诊断",
            "",
            "出现 `validation AUROC=0.5` 且 `threshold=0.5` 的运行：",
            "",
            "| 模式 | seed | 塌缩 epoch 数 | 最长连续数 | 总 epoch 数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if collapsed:
        for item in collapsed:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    MODE_LABELS[item["mode"]], item["seed"],
                    item["collapse_epoch_count"], item["longest_consecutive_collapse"],
                    item["epochs_completed"],
                )
            )
    else:
        lines.append("| 无 | — | 0 | 0 | — |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 当前最合理的探索性结论是：Independent-bag 值得作为下一阶段候选，但不能宣称已经显著优于 Full。",
            "2. 现有结果不能验证递归子图传递机制；它至多提示使用多个时间窗口可能优于只使用当前窗口，而最佳聚合方式可能是非递归的。",
            "3. 本表只有 5 个 seed，且当前 CUDA 训练尚未达到逐次严格复现；6/30 次运行还出现连续塌缩，因此需要先修复确定性和稳定性，再做确认性复验。",
            "4. 所有评估文件的 evidence level 均为 `exploratory_in_sample`：下游 validation/test 未参与基线训练，但上游提取器使用过全样本及其真实标签。因此这些集合不是端到端独立测试集，不能用于声称样本外泛化。",
            "5. test 已在本轮被查看，只能用于本轮最终探索性比较；后续若据此选择 Independent-bag，必须在新的独立数据或预先冻结的新划分上确认。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    alignment = validate_alignment(runs)
    summary, per_seed, paired, collapse = analyze(runs)
    test_predictions = runs[("full", 42)]["evaluation"]["test"]["metrics"][
        "predictions"
    ]
    test_labels = np.asarray(
        [row["label"] for row in test_predictions], dtype=np.float64
    )
    prevalence = float(test_labels.mean())
    constant_log_loss = float(
        -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence))
    )
    validation_auc = np.asarray(
        [row["validation_roc_auc"] for row in per_seed], dtype=np.float64
    )
    test_auc = np.asarray(
        [row["test_roc_auc"] for row in per_seed], dtype=np.float64
    )
    descriptive_diagnostics = {
        "test_positive_prevalence": prevalence,
        "test_constant_prevalence_log_loss": constant_log_loss,
        "validation_test_auc_pearson_r": float(
            stats.pearsonr(validation_auc, test_auc)[0]
        ),
        "validation_test_auc_spearman_r": float(
            stats.spearmanr(validation_auc, test_auc)[0]
        ),
        "collapse_run_count": sum(
            item["collapse_epoch_count"] > 0 for item in collapse
        ),
    }
    payload = {
        "schema_version": 1,
        "seeds": list(range(42, 47)),
        "modes": list(MODES),
        "run_count": len(runs),
        "alignment": alignment,
        "summary": summary,
        "per_seed": per_seed,
        "paired_comparisons_vs_full": paired,
        "collapse_diagnostics": collapse,
        "descriptive_diagnostics": descriptive_diagnostics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    print(json.dumps({
        "run_count": len(runs),
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
