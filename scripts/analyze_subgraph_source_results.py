"""Analyze tuple-matched Key, Low-score, Top-degree, and Random baselines."""

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


SOURCES = ("key", "low_score", "top_degree", "random")
SOURCE_LABELS = {
    "key": "Key",
    "low_score": "Low-score",
    "top_degree": "Top-degree",
    "random": "Random",
}
SEEDS = (42, 43, 44)
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(
            "outputs/baseline_source_training/key_controls_seed42_r0_v1"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "docs/experiment_results/subgraph_sources_seed42_44_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "docs/experiment_results/subgraph_sources_seed42_44_analysis.md"
        ),
    )
    return parser.parse_args()


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
    null_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        null_values.append(
            abs(float((array * np.asarray(signs, dtype=np.float64)).mean()))
        )
    return float(np.mean(np.asarray(null_values) >= observed - 1e-15))


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


def experiment_dir(root, source, seed):
    return root / "{}_bag_seed{}_v1".format(source, seed)


def load_runs(root):
    runs = {}
    for source in SOURCES:
        for seed in SEEDS:
            directory = experiment_dir(root, source, seed)
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
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("checkpoint training seed mismatch")
            if checkpoint.get("subgraph_source") != source:
                raise RuntimeError("checkpoint source mismatch")
            config = dict(checkpoint["model_config"])
            if config.get("history_mode") != "independent_bag":
                raise RuntimeError("source experiment is not Independent-bag")
            evaluations = {}
            for split in ("validation", "test"):
                path = directory / "{}_evaluation.json".format(split)
                with path.open("r", encoding="utf-8") as handle:
                    evaluations[split] = json.load(handle)
                if evaluations[split]["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("evaluation checkpoint hash mismatch")
                if evaluations[split].get("subgraph_source") != source:
                    raise RuntimeError("evaluation source mismatch")
            with (directory / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            runs[(source, seed)] = {
                "directory": directory.as_posix(),
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "evaluation": evaluations,
                "history": history,
            }
    return runs


def validate_runs(runs):
    matched_hashes = {
        run["checkpoint"].get("matched_control_manifest_sha256")
        for run in runs.values()
    }
    protocol_hashes = {
        run["checkpoint"].get("data_protocol_sha256") for run in runs.values()
    }
    extractor_hashes = {
        run["checkpoint"].get("extractor_checkpoint_sha256")
        for run in runs.values()
    }
    evidence_levels = {
        run["checkpoint"].get("evidence_level") for run in runs.values()
    }
    state_counts = {
        sum(int(tensor.numel()) for tensor in run["checkpoint"]["model_state_dict"].values())
        for run in runs.values()
    }
    if any(len(values) != 1 for values in (
        matched_hashes, protocol_hashes, extractor_hashes, evidence_levels, state_counts
    )):
        raise RuntimeError("runs do not share matching, protocol, extractor, or architecture")
    alignment = {}
    for split in ("validation", "test"):
        reference = None
        source_manifest_hashes = defaultdict(set)
        for (source, unused_seed), run in runs.items():
            del unused_seed
            payload = run["evaluation"][split]
            if payload.get("debug_limited_batches") is not None:
                raise RuntimeError("debug-limited evaluation found")
            source_manifest_hashes[source].add(payload["baseline_manifest_sha256"])
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in payload["metrics"]["predictions"]
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} samples are not aligned".format(split))
        if any(len(values) != 1 for values in source_manifest_hashes.values()):
            raise RuntimeError("manifest changed across seeds")
        alignment[split] = {
            "sample_count": len(reference),
            "class_counts": {
                str(label): sum(item[1] == label for item in reference)
                for label in (0, 1)
            },
            "aligned_across_sources_and_seeds": True,
            "source_manifest_sha256": {
                source: next(iter(values))
                for source, values in source_manifest_hashes.items()
            },
        }
    first_history = runs[("key", 42)]["history"][0]
    train_count = int(first_history["train"]["sample_count"])
    validation_count = alignment["validation"]["sample_count"]
    test_count = alignment["test"]["sample_count"]
    return {
        "matched_control_manifest_sha256": next(iter(matched_hashes)),
        "data_protocol_sha256": next(iter(protocol_hashes)),
        "extractor_checkpoint_sha256": next(iter(extractor_hashes)),
        "evidence_level": next(iter(evidence_levels)),
        "state_tensor_element_count": next(iter(state_counts)),
        "alignment": alignment,
        "partition_sample_counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
            "total": train_count + validation_count + test_count,
        },
    }


def summarize(runs):
    summary = {}
    per_seed = []
    collapse = []
    for source in SOURCES:
        summary[source] = {}
        for split in ("validation", "test"):
            summary[source][split] = {}
            for metric in METRICS + ("accuracy", "f1"):
                values = [
                    float(runs[(source, seed)]["evaluation"][split]["metrics"][metric])
                    for seed in SEEDS
                ]
                mean, sd = mean_sd(values)
                summary[source][split][metric] = {
                    "mean": mean,
                    "sd": sd,
                    "values": values,
                }
        for seed in SEEDS:
            run = runs[(source, seed)]
            row = {"source": source, "seed": seed, "epochs_completed": len(run["history"])}
            for split in ("validation", "test"):
                metrics = run["evaluation"][split]["metrics"]
                for metric in METRICS + ("accuracy", "f1", "threshold"):
                    row["{}_{}".format(split, metric)] = float(metrics[metric])
            flags = [
                item["validation"]["roc_auc"] == 0.5
                and item["validation"]["threshold"] == 0.5
                for item in run["history"]
            ]
            row["collapse_epoch_count"] = int(sum(flags))
            per_seed.append(row)
            if any(flags):
                collapse.append({
                    "source": source,
                    "seed": seed,
                    "collapse_epoch_count": int(sum(flags)),
                    "epochs_completed": len(flags),
                })
    return summary, per_seed, collapse


def compare_across_seeds(runs):
    result = {}
    for split in ("validation", "test"):
        result[split] = {}
        for metric in METRICS:
            comparisons = []
            pvalues = []
            key_values = np.asarray([
                runs[("key", seed)]["evaluation"][split]["metrics"][metric]
                for seed in SEEDS
            ], dtype=np.float64)
            for control in SOURCES[1:]:
                control_values = np.asarray([
                    runs[(control, seed)]["evaluation"][split]["metrics"][metric]
                    for seed in SEEDS
                ], dtype=np.float64)
                improvement = (
                    control_values - key_values
                    if metric == "unweighted_log_loss"
                    else key_values - control_values
                )
                ci_low, ci_high = paired_ci(improvement)
                pvalue = exact_sign_flip_pvalue(improvement)
                comparisons.append({
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
            for comparison, qvalue in zip(comparisons, bh_adjust(pvalues)):
                comparison["bh_q"] = qvalue
            result[split][metric] = comparisons
    return result


def prediction_rows(runs, source, seed, split="test"):
    return runs[(source, seed)]["evaluation"][split]["metrics"]["predictions"]


def sample_log_losses(rows):
    values = []
    for row in rows:
        probability = min(max(float(row["class_1_probability"]), 1e-12), 1.0 - 1e-12)
        values.append(
            -math.log(probability) if int(row["label"]) == 1
            else -math.log(1.0 - probability)
        )
    return np.asarray(values, dtype=np.float64)


def subject_bootstrap(runs, repeats, seed):
    reference = prediction_rows(runs, "key", SEEDS[0])
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[(row["site"], row["subject_id"])].append(index)
    group_indices = list(groups.values())
    rng = np.random.RandomState(seed)
    mean_losses = {}
    for source in SOURCES:
        mean_losses[source] = np.mean(
            np.stack([
                sample_log_losses(prediction_rows(runs, source, current_seed))
                for current_seed in SEEDS
            ], axis=0),
            axis=0,
        )
    output = {}
    for control in SOURCES[1:]:
        differences = mean_losses[control] - mean_losses["key"]
        bootstrap_values = np.empty(repeats, dtype=np.float64)
        for repeat in range(repeats):
            selected_groups = rng.randint(0, len(group_indices), len(group_indices))
            selected_samples = [
                sample_index
                for group_index in selected_groups
                for sample_index in group_indices[group_index]
            ]
            bootstrap_values[repeat] = differences[selected_samples].mean()
        output[control] = {
            "mean_log_loss_improvement": float(differences.mean()),
            "ci95_low": float(np.percentile(bootstrap_values, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_values, 97.5)),
            "two_sided_bootstrap_p": float(
                min(
                    1.0,
                    2.0 * min(
                        np.mean(bootstrap_values <= 0.0),
                        np.mean(bootstrap_values >= 0.0),
                    ),
                )
            ),
            "subject_count": len(group_indices),
            "sample_count": len(reference),
            "conditioning": "averaged predictions from the three fitted seeds",
        }
    return output


def f6(value):
    return "{:.6f}".format(float(value))


def comparison_lookup(payload, split, metric, control):
    return next(
        row for row in payload[split][metric] if row["control"] == control
    )


def render_markdown(payload):
    summary = payload["summary"]
    comparisons = payload["across_seed_comparisons"]
    bootstrap = payload["conditional_subject_bootstrap"]
    audit = payload["audit"]
    lines = [
        "# 匹配子图来源实验：Key、Low-score、Top-degree 与 Random",
        "",
        "## 验收与分析范围",
        "",
        "- 4 种来源 × 3 个训练 seed，共 12 次 Independent-bag 训练。",
        "- 四种来源共享同一个 matched-control hash，validation/test 样本、标签、subject 和 site 完全对齐。",
        "- 匹配后队列为 train/validation/test = {}/{}/{}，共 {} 个样本；约占原938样本的 {:.1f}%。".format(
            audit["partition_sample_counts"]["train"],
            audit["partition_sample_counts"]["validation"],
            audit["partition_sample_counts"]["test"],
            audit["partition_sample_counts"]["total"],
            100.0 * audit["partition_sample_counts"]["total"] / 938.0,
        ),
        "- 所有结果均为 `exploratory_in_sample`，且只适用于该严格匹配子队列。",
        "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {}（均值 ± 样本标准差）".format(split.capitalize()),
            "",
            "| 来源 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ])
        for source in SOURCES:
            data = summary[source][split]
            lines.append(
                "| {} | {} ± {} | {} ± {} | {} ± {} |".format(
                    SOURCE_LABELS[source],
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
        "## Test：Key相对控制的同-seed配对改善",
        "",
        "正值表示 Key 更好；log-loss 使用 `控制 − Key`。",
        "",
        "| 控制 | ΔLog-loss (95% seed CI) | 胜/3 | ΔAUROC (95% seed CI) | 胜/3 | ΔBAcc | 胜/3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for control in SOURCES[1:]:
        ll = comparison_lookup(comparisons, "test", "unweighted_log_loss", control)
        auc = comparison_lookup(comparisons, "test", "roc_auc", control)
        bacc = comparison_lookup(comparisons, "test", "balanced_accuracy", control)
        lines.append(
            "| {} | {} [{}, {}] | {}/3 | {} [{}, {}] | {}/3 | {} | {}/3 |".format(
                SOURCE_LABELS[control],
                f6(ll["mean_improvement"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]), ll["wins"],
                f6(auc["mean_improvement"]), f6(auc["ci95_low"]), f6(auc["ci95_high"]), auc["wins"],
                f6(bacc["mean_improvement"]), bacc["wins"],
            )
        )
    lines.extend([
        "",
        "只有3个seed，精确双侧符号翻转检验最小 p=0.25；所有基于seed的log-loss区间均跨0，不能据此确认统计显著。",
        "",
        "## 条件于三个已训练模型的样本级Bootstrap",
        "",
        "| 控制 | 平均ΔLog-loss | 95% subject-bootstrap CI | p |",
        "|---|---:|---:|---:|",
    ])
    for control in SOURCES[1:]:
        item = bootstrap[control]
        lines.append(
            "| {} | {} | [{}, {}] | {} |".format(
                SOURCE_LABELS[control], f6(item["mean_log_loss_improvement"]),
                f6(item["ci95_low"]), f6(item["ci95_high"]),
                f6(item["two_sided_bootstrap_p"]),
            )
        )
    lines.extend([
        "",
        "该Bootstrap把三个seed的每样本损失先求平均，再重采样subject；它反映匹配test样本的不确定性，但不包含新训练seed的不确定性，因此只能作为条件性补充证据。",
        "",
        "## 各seed的Test AUROC",
        "",
        "| 来源 | seed42 | seed43 | seed44 |",
        "|---|---:|---:|---:|",
    ])
    for source in SOURCES:
        values = summary[source]["test"]["roc_auc"]["values"]
        lines.append(
            "| {} | {} | {} | {} |".format(
                SOURCE_LABELS[source], *[f6(value) for value in values]
            )
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "1. Key 在 validation 和 test 的跨seed平均 log-loss、AUROC、balanced accuracy 上均为四来源最佳。",
        "2. Key 的 test AUROC 在3/3 seed中均优于三个控制：相对 Low-score +0.093286、Top-degree +0.218111、Random +0.125839。",
        "3. Key 相对 Top-degree 的优势最大，说明结果不像是单纯选择高连接强度节点；Top-degree 的平均test AUROC仅0.462763。",
        "4. Key 相对 Random 的稳定AUROC优势说明训练提取器捕捉到的结构不等同于尺寸匹配的随机压缩。",
        "5. Key 相对 Low-score 的优势较小但方向明确，提示提取器分数包含一定排序信息；不过seed44的Low-score最佳checkpoint为恒定预测，会放大该seed的AUROC差异。",
        "6. seed44的四种来源训练历史都出现过连续塌缩，说明这是共同训练稳定性问题而非Low-score独有问题。Key、Top-degree和Random的最佳checkpoint没有停在完全塌缩状态，但seed44不应被视为强独立复现。",
        "7. 主要指标log-loss仅在2/3 seed改善，seed44中Key略差于三个控制；跨seed区间仍跨0。条件样本bootstrap区间高于0，只说明现有三个模型在这96个test样本上的平均损失差，不覆盖重新训练模型的不确定性。",
        "8. Key平均test log-loss为0.651321，略优于类别先验恒定预测的0.656010；优势并不大，当前更强的信号来自AUROC排序，而不是概率校准。",
        "9. 严格匹配仅保留640/938个样本。匹配提高了来源比较的内部公平性，但限制了结论对完整队列的代表性；在获得匹配清单前不能判断298个样本的具体排除结构。",
        "10. 上游Key提取器使用过全样本真实标签，而Top-degree和Random不依赖这些标签。因此Key优势可能同时包含真实结构信号与监督性样本内偏倚，不能解释为样本外特异性证明。",
        "11. 综合判断：Key关键子图表现出超越Random和Top-degree的初步特异性，值得继续；但无需立即扩展复杂模型，应先增加训练seed或进行一个最小扰动验证。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    audit = validate_runs(runs)
    summary, per_seed, collapse = summarize(runs)
    comparisons = compare_across_seeds(runs)
    bootstrap = subject_bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    labels = [
        int(row["label"]) for row in prediction_rows(runs, "key", SEEDS[0])
    ]
    prevalence = float(np.mean(labels))
    payload = {
        "schema_version": 1,
        "sources": list(SOURCES),
        "seeds": list(SEEDS),
        "run_count": len(runs),
        "audit": audit,
        "summary": summary,
        "per_seed": per_seed,
        "collapse_diagnostics": collapse,
        "across_seed_comparisons": comparisons,
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
    print(json.dumps({
        "run_count": len(runs),
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
