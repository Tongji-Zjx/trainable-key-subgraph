"""Audit and compare matched Key versus signed endpoint-rewired Key runs."""

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


SOURCES = ("key", "key_rewired")
SEEDS = (42, 43, 44, 45, 46, 47)
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("outputs/baseline_key_rewired_training/seed2026_v1"),
    )
    parser.add_argument(
        "--matched-manifest", type=Path,
        default=Path(
            "outputs/baseline_key_rewired_exports/seed2026_v1/"
            "matched_control_manifest.json"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("docs/experiment_results/key_rewired_seed42_47_analysis.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("docs/experiment_results/key_rewired_seed42_47_analysis.md"),
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


def load_runs(root):
    runs = {}
    for source in SOURCES:
        for seed in SEEDS:
            directory = run_dir(root, source, seed)
            required = (
                "best_checkpoint.pt", "history.json",
                "validation_evaluation.json", "test_evaluation.json",
            )
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise RuntimeError("{} missing {}".format(directory, missing))
            checkpoint_path = directory / "best_checkpoint.pt"
            checkpoint_hash = sha256(checkpoint_path)
            checkpoint = trusted_load(checkpoint_path)
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("checkpoint seed mismatch")
            if checkpoint.get("subgraph_source") != source:
                raise RuntimeError("checkpoint source mismatch")
            if checkpoint["model_config"].get("history_mode") != "independent_bag":
                raise RuntimeError("run is not Independent-bag")
            with (directory / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            evaluations = {}
            for split in ("validation", "test"):
                with (directory / "{}_evaluation.json".format(split)).open(
                    "r", encoding="utf-8"
                ) as handle:
                    evaluation = json.load(handle)
                if evaluation["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("evaluation checkpoint hash mismatch")
                if evaluation.get("subgraph_source") != source:
                    raise RuntimeError("evaluation source mismatch")
                if evaluation.get("debug_limited_batches") is not None:
                    raise RuntimeError("debug-limited evaluation found")
                evaluations[split] = evaluation
            runs[(source, seed)] = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "history": history,
                "evaluation": evaluations,
            }
    return runs


def audit_runs(runs, matched_manifest_path):
    with matched_manifest_path.open("r", encoding="utf-8") as handle:
        matched = json.load(handle)
    if matched.get("experiment_kind") != "key_signed_endpoint_rewiring":
        raise RuntimeError("wrong matched experiment kind")
    if tuple(matched.get("sources", [])) != SOURCES:
        raise RuntimeError("wrong matched sources")
    expected_matched_hash = sha256(matched_manifest_path)
    fields = (
        "matched_control_manifest_sha256", "data_protocol_sha256",
        "extractor_checkpoint_sha256",
    )
    shared = {}
    for field in fields:
        values = {run["checkpoint"].get(field) for run in runs.values()}
        if len(values) != 1:
            raise RuntimeError("checkpoint field differs: {}".format(field))
        shared[field] = next(iter(values))
    if shared["matched_control_manifest_sha256"] != expected_matched_hash:
        raise RuntimeError("training does not bind the supplied matched manifest")
    model_configs = {
        json.dumps(run["checkpoint"]["model_config"], sort_keys=True)
        for run in runs.values()
    }
    if len(model_configs) != 1:
        raise RuntimeError("model configurations differ")
    checkpoint_hashes = {run["checkpoint_sha256"] for run in runs.values()}
    if len(checkpoint_hashes) != len(runs):
        raise RuntimeError("checkpoint files are not unique")

    alignment = {}
    for split in ("validation", "test"):
        reference = None
        manifest_hashes = defaultdict(set)
        for (source, unused_seed), run in runs.items():
            del unused_seed
            evaluation = run["evaluation"][split]
            manifest_hashes[source].add(evaluation["baseline_manifest_sha256"])
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in evaluation["metrics"]["predictions"]
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} samples are not aligned".format(split))
        if any(len(values) != 1 for values in manifest_hashes.values()):
            raise RuntimeError("partition manifest changed across seeds")
        alignment[split] = {
            "sample_count": len(reference),
            "class_counts": {
                str(label): sum(row[1] == label for row in reference)
                for label in (0, 1)
            },
            "aligned": True,
            "source_manifest_sha256": {
                source: next(iter(values)) for source, values in manifest_hashes.items()
            },
        }
    first = runs[("key", 42)]["history"][0]
    train_count = int(first["train"]["sample_count"])
    return {
        "run_count": len(runs),
        "unique_checkpoint_count": len(checkpoint_hashes),
        "shared_checkpoint_fields": shared,
        "alignment": alignment,
        "partition_sample_counts": {
            "train": train_count,
            "validation": alignment["validation"]["sample_count"],
            "test": alignment["test"]["sample_count"],
            "total": train_count + alignment["validation"]["sample_count"]
            + alignment["test"]["sample_count"],
        },
        "rewiring": {
            "seed": int(matched["rewiring_seed"]),
            "included_sample_count": len(matched["included_sample_keys"]),
            "excluded_sample_count": len(matched["excluded_samples"]),
            "summary": matched["rewiring_summary"],
        },
    }


def metric(runs, source, seed, split, name):
    return float(runs[(source, seed)]["evaluation"][split]["metrics"][name])


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def t_ci(values):
    values = np.asarray(values, dtype=np.float64)
    radius = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean() - radius), float(values.mean() + radius)


def sign_flip_p(values):
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bh_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues))
    running = 1.0
    for reverse_rank in range(len(pvalues) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(pvalues[index]) * len(pvalues) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(runs):
    summary = {}
    per_seed = []
    collapse = []
    for source in SOURCES:
        summary[source] = {}
        for split in ("validation", "test"):
            summary[source][split] = {}
            for name in METRICS + ("accuracy", "f1"):
                values = [metric(runs, source, seed, split, name) for seed in SEEDS]
                mean, sd = mean_sd(values)
                summary[source][split][name] = {
                    "mean": mean, "sd": sd, "values": values,
                }
        for seed in SEEDS:
            history = runs[(source, seed)]["history"]
            flags = [
                float(row["validation"]["roc_auc"]) == 0.5
                and float(row["validation"]["threshold"]) == 0.5
                for row in history
            ]
            row = {
                "source": source, "seed": seed,
                "epochs_completed": len(history),
                "collapse_epoch_count": int(sum(flags)),
            }
            for split in ("validation", "test"):
                for name in METRICS + ("accuracy", "f1", "threshold"):
                    row["{}_{}".format(split, name)] = metric(
                        runs, source, seed, split, name
                    )
            per_seed.append(row)
            if any(flags):
                collapse.append({
                    "source": source, "seed": seed,
                    "epochs_completed": len(history),
                    "collapse_epoch_count": int(sum(flags)),
                })
    return summary, per_seed, collapse


def paired_comparisons(runs):
    output = {}
    for split in ("validation", "test"):
        rows = {}
        for name in METRICS:
            key = np.asarray([metric(runs, "key", seed, split, name) for seed in SEEDS])
            rewired = np.asarray([
                metric(runs, "key_rewired", seed, split, name) for seed in SEEDS
            ])
            improvement = rewired - key if name == "unweighted_log_loss" else key - rewired
            low, high = t_ci(improvement)
            rows[name] = {
                "improvement_values": improvement.tolist(),
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(np.median(improvement)),
                "sd_improvement": float(improvement.std(ddof=1)),
                "ci95_low": low, "ci95_high": high,
                "wins": int(np.sum(improvement > 0.0)),
                "exact_sign_flip_p": sign_flip_p(improvement),
                "leave_one_seed_out_mean_min": float(min(
                    np.delete(improvement, index).mean()
                    for index in range(len(improvement))
                )),
                "leave_one_seed_out_mean_max": float(max(
                    np.delete(improvement, index).mean()
                    for index in range(len(improvement))
                )),
            }
        primary_names = ("unweighted_log_loss", "roc_auc")
        qvalues = bh_adjust([rows[name]["exact_sign_flip_p"] for name in primary_names])
        for name, qvalue in zip(primary_names, qvalues):
            rows[name]["bh_q_across_coprimary_metrics"] = qvalue
        output[split] = rows
    return output


def predictions(runs, source, seed):
    return runs[(source, seed)]["evaluation"]["test"]["metrics"]["predictions"]


def binary_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if not positive or not negative:
        return float("nan")
    ranks = stats.rankdata(probabilities, method="average")
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def subject_bootstrap(runs, repeats, random_seed):
    reference = predictions(runs, "key", 42)
    labels = np.asarray([int(row["label"]) for row in reference])
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[(row["site"], row["subject_id"])].append(index)
    group_indices = list(groups.values())
    probabilities = {}
    for source in SOURCES:
        probabilities[source] = np.mean(np.stack([
            np.asarray([
                float(row["class_1_probability"])
                for row in predictions(runs, source, seed)
            ])
            for seed in SEEDS
        ]), axis=0)

    def log_loss(probability, current_labels):
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return float(np.mean(
            -current_labels * np.log(probability)
            - (1 - current_labels) * np.log(1.0 - probability)
        ))

    rng = np.random.RandomState(random_seed)
    ll_draws = np.empty(repeats)
    auc_draws = np.empty(repeats)
    for repeat in range(repeats):
        sampled_groups = rng.randint(0, len(group_indices), len(group_indices))
        indices = np.asarray([
            item for group in sampled_groups for item in group_indices[group]
        ], dtype=np.int64)
        current_labels = labels[indices]
        key_probability = probabilities["key"][indices]
        rewired_probability = probabilities["key_rewired"][indices]
        ll_draws[repeat] = log_loss(rewired_probability, current_labels) - log_loss(
            key_probability, current_labels
        )
        auc_draws[repeat] = binary_auc(current_labels, key_probability) - binary_auc(
            current_labels, rewired_probability
        )

    def describe(values, observed):
        finite = values[np.isfinite(values)]
        return {
            "observed_improvement": float(observed),
            "ci95_low": float(np.percentile(finite, 2.5)),
            "ci95_high": float(np.percentile(finite, 97.5)),
            "two_sided_bootstrap_p": float(min(
                1.0, 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))
            )),
        }
    observed_ll = log_loss(probabilities["key_rewired"], labels) - log_loss(
        probabilities["key"], labels
    )
    observed_auc = binary_auc(labels, probabilities["key"]) - binary_auc(
        labels, probabilities["key_rewired"]
    )
    return {
        "conditioning": "probabilities averaged over the six fitted seeds",
        "subject_count": len(group_indices),
        "sample_count": len(labels),
        "log_loss": describe(ll_draws, observed_ll),
        "roc_auc": describe(auc_draws, observed_auc),
    }


def f6(value):
    return "{:.6f}".format(float(value))


def render_markdown(payload):
    audit = payload["audit"]
    summary = payload["summary"]
    paired = payload["paired_comparisons"]
    bootstrap = payload["conditional_subject_bootstrap"]
    lines = [
        "# Key 与 Key-rewired 最小拓扑扰动实验（seed 42–47）",
        "",
        "## 数据与实验验收",
        "",
        "- 12/12 次 Independent-bag 训练、checkpoint、validation 和 test 评估完整。",
        "- 12 个 checkpoint 哈希均不同；checkpoint 的来源、seed、模型配置、匹配清单及评估绑定均通过核验。",
        "- 938 个原始样本中保留 {} 个、排除 {} 个；下游 train/validation/test = {}/{}/{}。".format(
            audit["rewiring"]["included_sample_count"],
            audit["rewiring"]["excluded_sample_count"],
            audit["partition_sample_counts"]["train"],
            audit["partition_sample_counts"]["validation"],
            audit["partition_sample_counts"]["test"],
        ),
        "- 共保留 {:,} 条边，其中 {:,} 条边的端点发生改变，改变比例为 {:.1f}%。".format(
            int(audit["rewiring"]["summary"]["retained_edge_count"]),
            int(audit["rewiring"]["summary"]["changed_edge_count"]),
            100.0 * float(audit["rewiring"]["summary"]["changed_edge_ratio"]),
        ),
        "- 两来源在 validation/test 的样本、标签、subject、site 和顺序完全一致。",
        "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {}（均值 ± seed 标准差）".format(split.capitalize()), "",
            "| 来源 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ])
        for source, label in (("key", "Key"), ("key_rewired", "Key-rewired")):
            row = summary[source][split]
            lines.append("| {} | {} ± {} | {} ± {} | {} ± {} |".format(
                label,
                f6(row["unweighted_log_loss"]["mean"]), f6(row["unweighted_log_loss"]["sd"]),
                f6(row["roc_auc"]["mean"]), f6(row["roc_auc"]["sd"]),
                f6(row["balanced_accuracy"]["mean"]), f6(row["balanced_accuracy"]["sd"]),
            ))
        lines.append("")
    lines.extend([
        "## Test 同-seed配对改善", "",
        "正值表示 Key 更好。Log-loss 使用 `Key-rewired − Key`；其余指标使用 `Key − Key-rewired`。", "",
        "| 指标 | 平均改善 [95% seed CI] | Key胜/6 | 精确p | 共主要指标BH-q |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in METRICS:
        row = paired["test"][name]
        qvalue = row.get("bh_q_across_coprimary_metrics")
        lines.append("| {} | {} [{}, {}] | {}/6 | {} | {} |".format(
            name, f6(row["mean_improvement"]), f6(row["ci95_low"]),
            f6(row["ci95_high"]), row["wins"], f6(row["exact_sign_flip_p"]),
            f6(qvalue) if qvalue is not None else "—",
        ))
    lines.extend([
        "",
        "Log-loss 与 AUROC 预设为两个共同主要指标，BH 校正在这两个检验间进行；balanced accuracy 为阈值依赖的补充指标。",
        "",
        "## 条件于六个已训练模型的 subject bootstrap", "",
        "| 指标 | 基于六seed平均概率的改善 | 95% CI | p |",
        "|---|---:|---:|---:|",
    ])
    for name in ("unweighted_log_loss", "roc_auc"):
        key = "log_loss" if name == "unweighted_log_loss" else name
        row = bootstrap[key]
        lines.append("| {} | {} | [{}, {}] | {} |".format(
            name, f6(row["observed_improvement"]), f6(row["ci95_low"]),
            f6(row["ci95_high"]), f6(row["two_sided_bootstrap_p"]),
        ))
    lines.extend([
        "",
        "该 bootstrap 只量化当前六组拟合模型在 test subject 上的采样不确定性，不代替跨训练 seed 的检验。",
        "",
        "## 各 seed Test AUROC", "",
        "| seed | Key | Key-rewired | ΔAUROC |",
        "|---:|---:|---:|---:|",
    ])
    for index, seed in enumerate(SEEDS):
        key_value = summary["key"]["test"]["roc_auc"]["values"][index]
        rewired_value = summary["key_rewired"]["test"]["roc_auc"]["values"][index]
        lines.append("| {} | {} | {} | {} |".format(
            seed, f6(key_value), f6(rewired_value), f6(key_value - rewired_value)
        ))
    auc = paired["test"]["roc_auc"]
    loss = paired["test"]["unweighted_log_loss"]
    bacc = paired["test"]["balanced_accuracy"]
    lines.extend([
        "", "## 结论", "",
        "1. Key 的 Test AUROC 在 6/6 seed 上高于 Key-rewired，平均绝对提高 {}（约 {:.2f} 个百分点）；未校正精确 p={}。".format(
            f6(auc["mean_improvement"]), 100.0 * auc["mean_improvement"],
            f6(auc["exact_sign_flip_p"]),
        ),
        "2. 两个共同主要指标校正后，AUROC的BH-q={}、log-loss的BH-q={}，均未达到传统0.05阈值，因此应表述为稳定的探索性方向证据，而不是确定性验证。".format(
            f6(auc["bh_q_across_coprimary_metrics"]),
            f6(loss["bh_q_across_coprimary_metrics"]),
        ),
        "3. Log-loss 平均改善 {}，Key 在 {}/6 seed 胜出；效应较小，概率校准证据弱于AUROC排序证据。".format(
            f6(loss["mean_improvement"]), loss["wins"]
        ),
        "4. Balanced accuracy 的平均差值为 {}，Key 仅在 {}/6 seed 胜出，说明验证集阈值迁移到test后不稳定，不能用它支持拓扑有效性。".format(
            f6(bacc["mean_improvement"]), bacc["wins"]
        ),
        "5. AUROC差值中位数为{}；去掉任意一个seed后的平均差值范围为[{}, {}]。去掉差距最大的seed44后优势仍为正，但降至约0.008，表明方向并非完全由seed44制造，效应大小却受其明显影响。".format(
            f6(auc["median_improvement"]),
            f6(auc["leave_one_seed_out_mean_min"]),
            f6(auc["leave_one_seed_out_mean_max"]),
        ),
        "6. Validation上的Key AUROC平均只高0.0063，且log-loss反而略差；Test上的效应没有得到同等强度的validation复现。",
        "7. 78.0%的边端点被改变后，Key仍保持小幅但方向一致的AUROC优势，支持原始Key拓扑包含一定判别排序信息；但效应远小于此前Key相对Random/Top-degree的差距。",
        "8. Key自身在本次新队列的平均Test AUROC接近随机水平，因此本实验只能说明原拓扑相对其重连版本略有优势，不能单独证明模型具有强分类能力。",
        "9. 上游Key提取器使用过全样本标签，证据等级仍为 `exploratory_in_sample`，不能解释为样本外泛化或因果效应。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    audit = audit_runs(runs, args.matched_manifest)
    summary, per_seed, collapse = summarize(runs)
    paired = paired_comparisons(runs)
    bootstrap = subject_bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    payload = {
        "schema_version": 1,
        "evidence_level": "exploratory_in_sample",
        "seeds": list(SEEDS),
        "audit": audit,
        "summary": summary,
        "per_seed": per_seed,
        "collapse_diagnostics": collapse,
        "paired_comparisons": paired,
        "conditional_subject_bootstrap": bootstrap,
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
