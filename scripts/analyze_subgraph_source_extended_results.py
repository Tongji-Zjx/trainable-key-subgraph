"""Analyze the extended tuple-matched subgraph-source experiments.

Key, Top-degree, and Random use seeds 42--47. Low-score remains a
supplementary seeds 42--44 comparison. Positive paired improvements always
mean that Key performed better than the named control.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import torch


SOURCE_SEEDS = {
    "key": (42, 43, 44, 45, 46, 47),
    "top_degree": (42, 43, 44, 45, 46, 47),
    "random": (42, 43, 44, 45, 46, 47),
    "low_score": (42, 43, 44),
}
SOURCE_LABELS = {
    "key": "Key",
    "top_degree": "Top-degree",
    "random": "Random",
    "low_score": "Low-score",
}
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("outputs/baseline_source_training/key_controls_seed42_r0_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "docs/experiment_results/subgraph_sources_primary_seed42_47_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "docs/experiment_results/subgraph_sources_primary_seed42_47_analysis.md"
        ),
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trusted_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def run_dir(root, source, seed):
    return root / "{}_bag_seed{}_v1".format(source, seed)


def load_and_audit(root):
    runs = {}
    common_values = defaultdict(set)
    identities = {"validation": None, "test": None}
    manifest_hashes = defaultdict(set)

    for source, seeds in SOURCE_SEEDS.items():
        for seed in seeds:
            directory = run_dir(root, source, seed)
            required = (
                "history.json", "best_checkpoint.pt",
                "validation_evaluation.json", "test_evaluation.json",
            )
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise RuntimeError("{} missing {}".format(directory, missing))

            checkpoint_path = directory / "best_checkpoint.pt"
            checkpoint_hash = sha256(checkpoint_path)
            checkpoint = trusted_load(checkpoint_path)
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("seed mismatch: {}".format(directory))
            if checkpoint.get("subgraph_source") != source:
                raise RuntimeError("source mismatch: {}".format(directory))
            if checkpoint["model_config"].get("history_mode") != "independent_bag":
                raise RuntimeError("non-independent-bag run: {}".format(directory))

            common_values["matched_control_manifest_sha256"].add(
                checkpoint.get("matched_control_manifest_sha256")
            )
            common_values["data_protocol_sha256"].add(
                checkpoint.get("data_protocol_sha256")
            )
            common_values["extractor_checkpoint_sha256"].add(
                checkpoint.get("extractor_checkpoint_sha256")
            )
            common_values["evidence_level"].add(checkpoint.get("evidence_level"))
            common_values["model_config"].add(
                json.dumps(checkpoint["model_config"], sort_keys=True)
            )

            with (directory / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            evaluations = {}
            for split in ("validation", "test"):
                with (directory / "{}_evaluation.json".format(split)).open(
                    "r", encoding="utf-8"
                ) as handle:
                    evaluation = json.load(handle)
                if evaluation["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("checkpoint hash mismatch: {} {}".format(directory, split))
                if evaluation.get("subgraph_source") != source:
                    raise RuntimeError("evaluation source mismatch")
                if evaluation.get("debug_limited_batches") is not None:
                    raise RuntimeError("debug-limited evaluation found")
                identity = tuple(
                    (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                    for row in evaluation["metrics"]["predictions"]
                )
                if identities[split] is None:
                    identities[split] = identity
                elif identities[split] != identity:
                    raise RuntimeError("{} samples are not aligned".format(split))
                manifest_hashes[(source, split)].add(
                    evaluation["baseline_manifest_sha256"]
                )
                evaluations[split] = evaluation
            runs[(source, seed)] = {
                "checkpoint_sha256": checkpoint_hash,
                "history": history,
                "evaluation": evaluations,
            }

    for name, values in common_values.items():
        if len(values) != 1:
            raise RuntimeError("runs differ on {}: {}".format(name, values))
    for key, values in manifest_hashes.items():
        if len(values) != 1:
            raise RuntimeError("manifest changed across seeds: {}".format(key))

    first_history = runs[("key", 42)]["history"][0]
    counts = {
        "train": int(first_history["train"]["sample_count"]),
        "validation": len(identities["validation"]),
        "test": len(identities["test"]),
    }
    counts["total"] = sum(counts.values())
    audit = {
        "run_count": len(runs),
        "source_seeds": {key: list(value) for key, value in SOURCE_SEEDS.items()},
        "shared_artifacts": {
            name: next(iter(values)) for name, values in common_values.items()
        },
        "partition_sample_counts": counts,
        "aligned_across_sources_and_seeds": True,
        "source_manifest_sha256": {
            "{}_{}".format(source, split): next(iter(values))
            for (source, split), values in manifest_hashes.items()
        },
    }
    return runs, audit


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def t_ci(values):
    array = np.asarray(values, dtype=np.float64)
    radius = float(
        stats.t.ppf(0.975, len(array) - 1)
        * array.std(ddof=1) / math.sqrt(len(array))
    )
    return float(array.mean() - radius), float(array.mean() + radius)


def exact_sign_flip_p(values):
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        null.append(abs(float(np.mean(array * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bh_adjust(values):
    count = len(values)
    order = np.argsort(values)
    adjusted = np.ones(count)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def metric_value(runs, source, seed, split, metric):
    return float(runs[(source, seed)]["evaluation"][split]["metrics"][metric])


def summarize(runs):
    output = {}
    per_seed = []
    collapse = []
    for source, seeds in SOURCE_SEEDS.items():
        output[source] = {"seed_count": len(seeds), "seeds": list(seeds)}
        for split in ("validation", "test"):
            output[source][split] = {}
            for metric in METRICS + ("accuracy", "f1"):
                values = [metric_value(runs, source, seed, split, metric) for seed in seeds]
                mean, sd = mean_sd(values)
                output[source][split][metric] = {
                    "mean": mean, "sd": sd, "values": values,
                }
        for seed in seeds:
            history = runs[(source, seed)]["history"]
            flags = [
                float(row["validation"]["roc_auc"]) == 0.5
                and float(row["validation"]["threshold"]) == 0.5
                for row in history
            ]
            item = {
                "source": source,
                "seed": seed,
                "epochs_completed": len(history),
                "collapse_epoch_count": int(sum(flags)),
            }
            for split in ("validation", "test"):
                for metric in METRICS + ("accuracy", "f1", "threshold"):
                    item["{}_{}".format(split, metric)] = metric_value(
                        runs, source, seed, split, metric
                    )
            per_seed.append(item)
            if any(flags):
                collapse.append({
                    "source": source, "seed": seed,
                    "collapse_epoch_count": int(sum(flags)),
                    "epochs_completed": len(history),
                })
    return output, per_seed, collapse


def compare(runs):
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for metric in METRICS:
            rows = []
            for control in ("top_degree", "random", "low_score"):
                seeds = tuple(sorted(set(SOURCE_SEEDS["key"]) & set(SOURCE_SEEDS[control])))
                key = np.asarray([
                    metric_value(runs, "key", seed, split, metric) for seed in seeds
                ])
                other = np.asarray([
                    metric_value(runs, control, seed, split, metric) for seed in seeds
                ])
                improvement = other - key if metric == "unweighted_log_loss" else key - other
                low, high = t_ci(improvement)
                rows.append({
                    "control": control,
                    "seeds": list(seeds),
                    "n": len(seeds),
                    "improvement_values": improvement.tolist(),
                    "mean_improvement": float(improvement.mean()),
                    "sd_improvement": float(improvement.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(improvement > 0.0)),
                    "exact_sign_flip_p": exact_sign_flip_p(improvement),
                })
            primary = [row for row in rows if row["control"] in ("top_degree", "random")]
            adjusted = bh_adjust([row["exact_sign_flip_p"] for row in primary])
            for row, qvalue in zip(primary, adjusted):
                row["primary_bh_q_within_metric"] = qvalue
            result[split][metric] = rows
    return result


def prediction_rows(runs, source, seed):
    return runs[(source, seed)]["evaluation"]["test"]["metrics"]["predictions"]


def sample_losses(rows):
    output = []
    for row in rows:
        probability = min(max(float(row["class_1_probability"]), 1e-12), 1.0 - 1e-12)
        output.append(-math.log(probability) if int(row["label"]) else -math.log(1.0 - probability))
    return np.asarray(output)


def subject_bootstrap(runs, repeats, random_seed):
    reference = prediction_rows(runs, "key", 42)
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[(row["site"], row["subject_id"])].append(index)
    group_indices = list(groups.values())
    rng = np.random.RandomState(random_seed)
    output = {}
    for control in ("top_degree", "random", "low_score"):
        seeds = tuple(sorted(set(SOURCE_SEEDS["key"]) & set(SOURCE_SEEDS[control])))
        key_losses = np.mean(np.stack([
            sample_losses(prediction_rows(runs, "key", seed)) for seed in seeds
        ]), axis=0)
        control_losses = np.mean(np.stack([
            sample_losses(prediction_rows(runs, control, seed)) for seed in seeds
        ]), axis=0)
        difference = control_losses - key_losses
        draws = np.empty(repeats)
        for repeat in range(repeats):
            sampled = rng.randint(0, len(group_indices), len(group_indices))
            indices = [item for group in sampled for item in group_indices[group]]
            draws[repeat] = float(np.mean(difference[indices]))
        output[control] = {
            "seeds": list(seeds),
            "mean_log_loss_improvement": float(np.mean(difference)),
            "ci95_low": float(np.percentile(draws, 2.5)),
            "ci95_high": float(np.percentile(draws, 97.5)),
            "two_sided_bootstrap_p": float(min(
                1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))
            )),
            "subject_count": len(group_indices),
            "sample_count": len(reference),
            "conditioning": "predictions averaged over paired fitted seeds",
        }
    return output


def lookup(comparisons, split, metric, control):
    return next(
        row for row in comparisons[split][metric] if row["control"] == control
    )


def fmt(value):
    return "{:.6f}".format(float(value))


def render_markdown(payload):
    summary = payload["summary"]
    comparisons = payload["paired_comparisons"]
    audit = payload["audit"]
    bootstrap = payload["conditional_subject_bootstrap"]
    counts = audit["partition_sample_counts"]
    lines = [
        "# 子图来源扩展实验分析（主分析 seed 42–47）",
        "",
        "## 验收范围",
        "",
        "- 共验收 21 次 Independent-bag 训练：Key、Top-degree、Random 各 6 个 seed；Low-score 保留 3 个 seed 作为补充。",
        "- checkpoint 来源、训练 seed、模型配置、评估 checkpoint 哈希均已核对；validation/test 的样本顺序、标签、subject 和 site 完全对齐。",
        "- 匹配队列 train/validation/test = {}/{}/{}，共 {} 个样本。".format(
            counts["train"], counts["validation"], counts["test"], counts["total"]
        ),
        "- 证据等级仍是 `exploratory_in_sample`：上游 Key 提取器曾使用全样本标签训练，结果不能解释为样本外泛化。",
        "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {} 结果（均值 ± seed 标准差）".format(split.capitalize()),
            "",
            "| 来源 | seed数 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|---:|",
        ])
        for source in ("key", "top_degree", "random", "low_score"):
            row = summary[source][split]
            lines.append("| {} | {} | {} ± {} | {} ± {} | {} ± {} |".format(
                SOURCE_LABELS[source], summary[source]["seed_count"],
                fmt(row["unweighted_log_loss"]["mean"]), fmt(row["unweighted_log_loss"]["sd"]),
                fmt(row["roc_auc"]["mean"]), fmt(row["roc_auc"]["sd"]),
                fmt(row["balanced_accuracy"]["mean"]), fmt(row["balanced_accuracy"]["sd"]),
            ))
        lines.append("")

    lines.extend([
        "## 主检验：Test 上 Key 相对控制的同-seed配对改善",
        "",
        "正值表示 Key 更好；Log-loss 定义为 `控制 − Key`，其余为 `Key − 控制`。",
        "",
        "| 控制 | n | ΔLog-loss [95% CI] | 胜/总 | p；BH-q | ΔAUROC [95% CI] | 胜/总 | p；BH-q | ΔBAcc | 胜/总 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for control in ("top_degree", "random"):
        ll = lookup(comparisons, "test", "unweighted_log_loss", control)
        auc = lookup(comparisons, "test", "roc_auc", control)
        bacc = lookup(comparisons, "test", "balanced_accuracy", control)
        lines.append("| {} | {} | {} [{}, {}] | {}/{} | {}; {} | {} [{}, {}] | {}/{} | {}; {} | {} | {}/{} |".format(
            SOURCE_LABELS[control], ll["n"],
            fmt(ll["mean_improvement"]), fmt(ll["ci95_low"]), fmt(ll["ci95_high"]), ll["wins"], ll["n"],
            fmt(ll["exact_sign_flip_p"]), fmt(ll["primary_bh_q_within_metric"]),
            fmt(auc["mean_improvement"]), fmt(auc["ci95_low"]), fmt(auc["ci95_high"]), auc["wins"], auc["n"],
            fmt(auc["exact_sign_flip_p"]), fmt(auc["primary_bh_q_within_metric"]),
            fmt(bacc["mean_improvement"]), bacc["wins"], bacc["n"],
        ))

    lines.extend([
        "",
        "精确双侧符号翻转检验枚举全部 2^6 种符号；n=6 时可能达到的最小双侧 p 值为 0.03125。BH-q 仅在同一指标的两个预设主控制间校正。t 区间只描述 seed 差值的均值，不应在小样本下单独作为正态性证据。",
        "",
        "## 补充比较：Low-score（仅 seed 42–44）",
        "",
    ])
    for metric in METRICS:
        row = lookup(comparisons, "test", metric, "low_score")
        lines.append("- {}：平均改善 {}，95% seed CI [{}, {}]，Key 胜 {}/3，精确 p={}。".format(
            metric, fmt(row["mean_improvement"]), fmt(row["ci95_low"]),
            fmt(row["ci95_high"]), row["wins"], fmt(row["exact_sign_flip_p"])
        ))

    lines.extend([
        "",
        "## 条件于已训练模型的 subject bootstrap（Test log-loss）",
        "",
        "| 控制 | 配对seed数 | 平均改善 | 95% CI | p |",
        "|---|---:|---:|---:|---:|",
    ])
    for control in ("top_degree", "random", "low_score"):
        row = bootstrap[control]
        lines.append("| {} | {} | {} | [{}, {}] | {} |".format(
            SOURCE_LABELS[control], len(row["seeds"]), fmt(row["mean_log_loss_improvement"]),
            fmt(row["ci95_low"]), fmt(row["ci95_high"]), fmt(row["two_sided_bootstrap_p"])
        ))
    lines.extend([
        "",
        "该 bootstrap 先在同一来源的配对 seed 间平均每个样本损失，再重采样 subject；它反映现有拟合模型下的样本不确定性，不包含重新训练产生的 seed 不确定性。",
        "",
        "## 各 seed Test AUROC",
        "",
        "| 来源 | seed42 | seed43 | seed44 | seed45 | seed46 | seed47 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for source in ("key", "top_degree", "random", "low_score"):
        values = summary[source]["test"]["roc_auc"]["values"]
        cells = [fmt(value) for value in values] + ["—"] * (6 - len(values))
        lines.append("| {} | {} |".format(SOURCE_LABELS[source], " | ".join(cells)))

    key_test = summary["key"]["test"]
    top_auc = lookup(comparisons, "test", "roc_auc", "top_degree")
    random_auc = lookup(comparisons, "test", "roc_auc", "random")
    top_ll = lookup(comparisons, "test", "unweighted_log_loss", "top_degree")
    random_ll = lookup(comparisons, "test", "unweighted_log_loss", "random")
    lines.extend([
        "",
        "## 结论",
        "",
        "1. 六 seed 下，Key 的 Test 均值为 log-loss {}、AUROC {}、balanced accuracy {}。".format(
            fmt(key_test["unweighted_log_loss"]["mean"]), fmt(key_test["roc_auc"]["mean"]),
            fmt(key_test["balanced_accuracy"]["mean"])
        ),
        "2. Key 相对 Top-degree 的 Test AUROC 平均提高 {}，{}/6 seed 胜出；相对 Random 平均提高 {}，{}/6 seed 胜出。".format(
            fmt(top_auc["mean_improvement"]), top_auc["wins"],
            fmt(random_auc["mean_improvement"]), random_auc["wins"]
        ),
        "3. 主要概率指标 log-loss 上，Key 相对 Top-degree 平均改善 {}（{}/6 胜），相对 Random 改善 {}（{}/6 胜）。".format(
            fmt(top_ll["mean_improvement"]), top_ll["wins"],
            fmt(random_ll["mean_improvement"]), random_ll["wins"]
        ),
        "4. 若六个配对差值方向全部一致，精确检验 p=0.03125 可作为跨 seed 稳定性的初步证据；仍需结合效应区间、校正后 q 值和实际数值判断，不能只看 p 值。",
        "5. Low-score 仍只有三个 seed，只能作为方向性补充，不能与六-seed主比较赋予同等证据权重。",
        "6. 本实验支持判断“Key 不是简单的高连接强度选择，也不是尺寸匹配的随机压缩”；但由于上游提取器的全样本监督暴露，它尚不能证明样本外特异性或因果有效性。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs, audit = load_and_audit(args.training_root)
    summary, per_seed, collapse = summarize(runs)
    comparisons = compare(runs)
    bootstrap = subject_bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    labels = [int(row["label"]) for row in prediction_rows(runs, "key", 42)]
    prevalence = float(np.mean(labels))
    payload = {
        "schema_version": 1,
        "analysis_role": {
            "primary": {"sources": ["key", "top_degree", "random"], "seeds": list(range(42, 48))},
            "supplementary": {"sources": ["key", "low_score"], "seeds": [42, 43, 44]},
        },
        "audit": audit,
        "summary": summary,
        "per_seed": per_seed,
        "collapse_diagnostics": collapse,
        "paired_comparisons": comparisons,
        "conditional_subject_bootstrap": bootstrap,
        "test_constant_prevalence_log_loss": float(
            -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence))
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
        handle.write("\n")
    print(json.dumps({
        "run_count": audit["run_count"],
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
