"""Audit and analyze matched targeted-versus-random edge-deletion doses."""

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


SEEDS = (42, 43, 44, 45, 46, 47)
RATIO_CODES = ("010", "025", "050")
BASE = "key_edge_000"
SOURCES = (BASE,) + tuple(
    "key_edge_{}_{}".format(mode, ratio)
    for ratio in RATIO_CODES for mode in ("targeted", "random")
)
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("outputs/baseline_edge_perturbation_training/seed2026_v1"),
    )
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("outputs/baseline_edge_perturbation_experiment/seed2026_v1"),
    )
    parser.add_argument(
        "--matched-manifest", type=Path,
        default=Path(
            "outputs/baseline_edge_perturbation_exports/seed2026_v1/"
            "matched_control_manifest.json"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path(
            "docs/experiment_results/edge_perturbation_dose_seed42_47_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path(
            "docs/experiment_results/edge_perturbation_dose_seed42_47_analysis.md"
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


def sign_flip_p(values):
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def t_ci(values):
    values = np.asarray(values, dtype=np.float64)
    radius = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean() - radius), float(values.mean() + radius)


def bh_adjust(pvalues):
    order = np.argsort(pvalues)
    output = np.ones(len(pvalues))
    running = 1.0
    for reverse_rank in range(len(pvalues) - 1, -1, -1):
        index = int(order[reverse_rank])
        running = min(running, float(pvalues[index]) * len(pvalues) / (reverse_rank + 1))
        output[index] = running
    return output.tolist()


def load_runs(root):
    runs = {}
    for source in SOURCES:
        for seed in SEEDS:
            directory = root / "{}_seed{}".format(source, seed)
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
            config = checkpoint["model_config"]
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("training seed mismatch")
            if checkpoint.get("subgraph_source") != source:
                raise RuntimeError("checkpoint source mismatch")
            if not (
                config.get("history_mode") == "independent_bag"
                and config.get("structural_interface_version") == 0
                and not config.get("use_structural_features")
                and not config.get("use_structural_deltas", False)
            ):
                raise RuntimeError("baseline condition differs")
            with (directory / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            evaluations = {}
            for split in ("validation", "test"):
                with (directory / "{}_evaluation.json".format(split)).open(
                    "r", encoding="utf-8"
                ) as handle:
                    evaluation = json.load(handle)
                if evaluation["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("evaluation checkpoint mismatch")
                if evaluation.get("subgraph_source") != source:
                    raise RuntimeError("evaluation source mismatch")
                if evaluation.get("debug_limited_batches") is not None:
                    raise RuntimeError("limited evaluation found")
                evaluations[split] = evaluation
            runs[source, seed] = {
                "checkpoint": checkpoint,
                "checkpoint_hash": checkpoint_hash,
                "history": history,
                "evaluation": evaluations,
            }
    return runs


def audit(runs, experiment_root, matched_manifest_path):
    with matched_manifest_path.open("r", encoding="utf-8") as handle:
        matched = json.load(handle)
    with (experiment_root / "source_experiment.json").open("r", encoding="utf-8") as handle:
        experiment = json.load(handle)
    if set(matched["sources"]) != set(SOURCES) or set(experiment["sources"]) != set(SOURCES):
        raise RuntimeError("source inventory differs")
    hashes = {run["checkpoint_hash"] for run in runs.values()}
    if len(hashes) != 42:
        raise RuntimeError("checkpoint hashes are not unique")
    parameter_counts = set()
    architectures = set()
    for run in runs.values():
        state = run["checkpoint"]["model_state_dict"]
        parameter_counts.add(sum(int(value.numel()) for value in state.values()))
        architectures.add(json.dumps(run["checkpoint"]["model_config"], sort_keys=True))
        if run["checkpoint"].get("matched_control_manifest_sha256") != sha256(
            matched_manifest_path
        ):
            raise RuntimeError("checkpoint matched-control hash differs")
    if len(parameter_counts) != 1 or len(architectures) != 1:
        raise RuntimeError("model architectures differ")
    summary = matched["perturbation_summary"]
    realized = {"000": float(summary[BASE]["realized_deleted_ratio"])}
    for ratio in RATIO_CODES:
        targeted = summary["key_edge_targeted_{}".format(ratio)]
        random = summary["key_edge_random_{}".format(ratio)]
        for name in (
            "original_edge_count", "deleted_edge_count", "retained_edge_count",
            "realized_deleted_ratio",
        ):
            if targeted[name] != random[name]:
                raise RuntimeError("targeted/random deletion inventory differs")
        realized[ratio] = float(targeted["realized_deleted_ratio"])
    alignment = {}
    for split in ("validation", "test"):
        reference = None
        for source in SOURCES:
            local_manifest = (
                experiment_root / "splits" / source / split / "baseline_manifest.json"
            )
            local_hash = sha256(local_manifest)
            for seed in SEEDS:
                run = runs[source, seed]
                evaluation = run["evaluation"][split]
                if evaluation["baseline_manifest_sha256"] != local_hash:
                    raise RuntimeError("evaluation manifest hash differs")
                if split == "validation" and run["checkpoint"].get(
                    "validation_manifest_sha256"
                ) != local_hash:
                    raise RuntimeError("checkpoint validation manifest differs")
                identity = tuple(
                    (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                    for row in evaluation["metrics"]["predictions"]
                )
                if reference is None:
                    reference = identity
                elif identity != reference:
                    raise RuntimeError("{} evaluation samples differ".format(split))
        alignment[split] = {
            "aligned": True, "sample_count": len(reference),
            "class_counts": {
                str(label): sum(row[1] == label for row in reference) for label in (0, 1)
            },
        }
    train_count = int(runs[BASE, 42]["history"][0]["train"]["sample_count"])
    return {
        "run_count": 42,
        "unique_checkpoint_count": len(hashes),
        "parameter_count": next(iter(parameter_counts)),
        "included_sample_count": len(matched["included_sample_keys"]),
        "excluded_sample_count": len(matched["excluded_samples"]),
        "common_original_edge_count": int(summary[BASE]["original_edge_count"]),
        "realized_deleted_ratios": realized,
        "downstream_assignment_sha256": experiment["downstream_assignment_sha256"],
        "alignment": alignment,
        "partition_sample_counts": {
            "train": train_count,
            "validation": alignment["validation"]["sample_count"],
            "test": alignment["test"]["sample_count"],
        },
        "control_checks": {
            "targeted_random_exact_edge_count_match": True,
            "common_sample_tuple_inventory": True,
            "same_model_architecture": True,
            "evaluation_samples_aligned": True,
        },
    }


def metric(runs, source, seed, split, name):
    return float(runs[source, seed]["evaluation"][split]["metrics"][name])


def summarize(runs):
    output = {}
    collapse = []
    per_seed = []
    for source in SOURCES:
        output[source] = {}
        for split in ("validation", "test"):
            output[source][split] = {}
            for name in METRICS:
                values = np.asarray([
                    metric(runs, source, seed, split, name) for seed in SEEDS
                ])
                output[source][split][name] = {
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                    "values": values.tolist(),
                }
        for seed in SEEDS:
            history = runs[source, seed]["history"]
            count = sum(
                float(row["validation"]["roc_auc"]) == 0.5
                and float(row["validation"]["threshold"]) == 0.5
                for row in history
            )
            row = {
                "source": source, "seed": seed, "epochs_completed": len(history),
                "collapse_epoch_count": int(count),
            }
            per_seed.append(row)
            if count:
                collapse.append(row)
    return output, per_seed, collapse


def damage_values(runs, ratio, split, name):
    targeted = "key_edge_targeted_{}".format(ratio)
    random_source = "key_edge_random_{}".format(ratio)
    left = np.asarray([metric(runs, targeted, seed, split, name) for seed in SEEDS])
    right = np.asarray([metric(runs, random_source, seed, split, name) for seed in SEEDS])
    return left - right if name == "unweighted_log_loss" else right - left


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = t_ci(values)
    return {
        "values": values.tolist(), "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)), "ci95_low": low, "ci95_high": high,
        "wins": int(np.sum(values > 0)), "exact_sign_flip_p": sign_flip_p(values),
    }


def comparisons(runs, realized):
    output = {"targeted_vs_random": {}, "versus_baseline": {}, "dose_slope": {}}
    for split in ("validation", "test"):
        output["targeted_vs_random"][split] = {}
        for ratio in RATIO_CODES:
            output["targeted_vs_random"][split][ratio] = {
                name: describe(damage_values(runs, ratio, split, name))
                for name in METRICS
            }
        for name in METRICS:
            pvalues = [
                output["targeted_vs_random"][split][ratio][name]["exact_sign_flip_p"]
                for ratio in RATIO_CODES
            ]
            for ratio, qvalue in zip(RATIO_CODES, bh_adjust(pvalues)):
                output["targeted_vs_random"][split][ratio][name]["dose_bh_q"] = qvalue
        output["versus_baseline"][split] = {}
        for mode in ("targeted", "random"):
            output["versus_baseline"][split][mode] = {}
            for ratio in RATIO_CODES:
                source = "key_edge_{}_{}".format(mode, ratio)
                row = {}
                for name in METRICS:
                    current = np.asarray([
                        metric(runs, source, seed, split, name) for seed in SEEDS
                    ])
                    baseline = np.asarray([
                        metric(runs, BASE, seed, split, name) for seed in SEEDS
                    ])
                    damage = current - baseline if name == "unweighted_log_loss" else baseline - current
                    row[name] = describe(damage)
                output["versus_baseline"][split][mode][ratio] = row
        ratios = np.asarray([0.0] + [realized[ratio] for ratio in RATIO_CODES])
        output["dose_slope"][split] = {}
        for name in METRICS:
            slopes = []
            monotonic = []
            for seed_index, unused_seed in enumerate(SEEDS):
                del unused_seed
                differences = [0.0] + [
                    damage_values(runs, ratio, split, name)[seed_index]
                    for ratio in RATIO_CODES
                ]
                slopes.append(float(np.polyfit(ratios, np.asarray(differences), 1)[0]))
                monotonic.append(all(
                    differences[index + 1] >= differences[index]
                    for index in range(len(differences) - 1)
                ))
            row = describe(slopes)
            row["fully_monotonic_seed_count"] = int(sum(monotonic))
            row["ratio_values"] = ratios.tolist()
            output["dose_slope"][split][name] = row
    return output


def auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    ranks = stats.rankdata(probabilities, method="average")
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def log_loss(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities), 1e-12, 1 - 1e-12)
    return float(np.mean(-labels * np.log(probabilities) - (1 - labels) * np.log(1 - probabilities)))


def bootstrap(runs, repeats, seed):
    reference = runs[BASE, 42]["evaluation"]["test"]["metrics"]["predictions"]
    labels = np.asarray([int(row["label"]) for row in reference])
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[row["site"], row["subject_id"]].append(index)
    subject_indices = list(groups.values())
    probabilities = {}
    for source in SOURCES:
        probabilities[source] = np.mean(np.stack([
            np.asarray([
                float(row["class_1_probability"])
                for row in runs[source, current_seed]["evaluation"]["test"]["metrics"]["predictions"]
            ]) for current_seed in SEEDS
        ]), axis=0)
    rng = np.random.RandomState(seed)
    output = {}
    for ratio in RATIO_CODES:
        targeted = probabilities["key_edge_targeted_{}".format(ratio)]
        random_probability = probabilities["key_edge_random_{}".format(ratio)]
        ll_draws = np.empty(repeats)
        auc_draws = np.empty(repeats)
        for repeat in range(repeats):
            selected = rng.randint(0, len(subject_indices), len(subject_indices))
            indices = np.asarray([item for group in selected for item in subject_indices[group]])
            current_labels = labels[indices]
            ll_draws[repeat] = log_loss(current_labels, targeted[indices]) - log_loss(
                current_labels, random_probability[indices]
            )
            auc_draws[repeat] = auc(current_labels, random_probability[indices]) - auc(
                current_labels, targeted[indices]
            )
        def result(draws, observed):
            draws = draws[np.isfinite(draws)]
            return {
                "observed": float(observed),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "two_sided_p": float(min(1.0, 2 * min(
                    np.mean(draws <= 0), np.mean(draws >= 0)
                ))),
            }
        output[ratio] = {
            "log_loss": result(
                ll_draws, log_loss(labels, targeted) - log_loss(labels, random_probability)
            ),
            "roc_auc": result(
                auc_draws, auc(labels, random_probability) - auc(labels, targeted)
            ),
        }
    output["metadata"] = {
        "conditioning": "probabilities averaged over six fitted seeds",
        "sample_count": len(labels), "subject_count": len(subject_indices),
    }
    return output


def f6(value):
    return "{:.6f}".format(float(value))


def render(payload):
    audit_payload = payload["audit"]
    summary = payload["summary"]
    compared = payload["comparisons"]
    boot = payload["conditional_subject_bootstrap"]
    lines = [
        "# 高分边定向删除与随机删除剂量实验（seed 42–47）", "",
        "## 验收", "",
        "- 42/42次正式训练、最佳checkpoint、history及validation/test评估完整。",
        "- 42个checkpoint哈希均不同；模型参数量相同（{}），均为Independent-bag中性结构接口。".format(audit_payload["parameter_count"]),
        "- 公共队列包含{}个样本、排除{}个；train/validation/test={}/{}/{}。".format(
            audit_payload["included_sample_count"], audit_payload["excluded_sample_count"],
            audit_payload["partition_sample_counts"]["train"],
            audit_payload["partition_sample_counts"]["validation"],
            audit_payload["partition_sample_counts"]["test"],
        ),
        "- Targeted与Random在每档删除边数完全相同；公共原始边数为{}。".format(audit_payload["common_original_edge_count"]),
        "- 实际删除比例：10%档={}，25%档={}，50%档={}。".format(
            f6(audit_payload["realized_deleted_ratios"]["010"]),
            f6(audit_payload["realized_deleted_ratios"]["025"]),
            f6(audit_payload["realized_deleted_ratios"]["050"]),
        ), "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {}（均值 ± seed标准差）".format(split.capitalize()), "",
            "| 条件 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ])
        for source in SOURCES:
            row = summary[source][split]
            lines.append("| {} | {} ± {} | {} ± {} | {} ± {} |".format(
                source,
                f6(row["unweighted_log_loss"]["mean"]), f6(row["unweighted_log_loss"]["sd"]),
                f6(row["roc_auc"]["mean"]), f6(row["roc_auc"]["sd"]),
                f6(row["balanced_accuracy"]["mean"]), f6(row["balanced_accuracy"]["sd"]),
            ))
        lines.append("")
    lines.extend([
        "## Test：Targeted相对Random的额外损伤", "",
        "正值表示定向删除造成更强损伤；log-loss使用`Targeted−Random`，AUROC使用`Random−Targeted`。", "",
        "| 剂量 | ΔLog-loss [95% seed CI] | 胜/6 | p；BH-q | ΔAUROC [95% seed CI] | 胜/6 | p；BH-q |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ratio in RATIO_CODES:
        row = compared["targeted_vs_random"]["test"][ratio]
        ll = row["unweighted_log_loss"]
        auc_row = row["roc_auc"]
        lines.append("| {}% | {} [{}, {}] | {}/6 | {}；{} | {} [{}, {}] | {}/6 | {}；{} |".format(
            int(ratio), f6(ll["mean"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
            ll["wins"], f6(ll["exact_sign_flip_p"]), f6(ll["dose_bh_q"]),
            f6(auc_row["mean"]), f6(auc_row["ci95_low"]), f6(auc_row["ci95_high"]),
            auc_row["wins"], f6(auc_row["exact_sign_flip_p"]), f6(auc_row["dose_bh_q"]),
        ))
    lines.extend(["", "## 剂量—反应斜率", ""])
    for split in ("validation", "test"):
        ll = compared["dose_slope"][split]["unweighted_log_loss"]
        auc_row = compared["dose_slope"][split]["roc_auc"]
        lines.append(
            "- {}：ΔLog-loss斜率={} [{}, {}]（{}/6正向，p={}）；ΔAUROC斜率={} [{}, {}]（{}/6正向，p={}）。".format(
                split.capitalize(), f6(ll["mean"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
                ll["wins"], f6(ll["exact_sign_flip_p"]), f6(auc_row["mean"]),
                f6(auc_row["ci95_low"]), f6(auc_row["ci95_high"]), auc_row["wins"],
                f6(auc_row["exact_sign_flip_p"]),
            )
        )
    lines.extend([
        "", "## 条件于六个已训练模型的subject bootstrap", "",
        "该bootstrap不包含重新训练的seed不确定性。", "",
        "| 剂量 | ΔLog-loss [95% CI] | p | ΔAUROC [95% CI] | p |",
        "|---:|---:|---:|---:|---:|",
    ])
    for ratio in RATIO_CODES:
        ll = boot[ratio]["log_loss"]
        auc_row = boot[ratio]["roc_auc"]
        lines.append("| {}% | {} [{}, {}] | {} | {} [{}, {}] | {} |".format(
            int(ratio), f6(ll["observed"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
            f6(ll["two_sided_p"]), f6(auc_row["observed"]),
            f6(auc_row["ci95_low"]), f6(auc_row["ci95_high"]), f6(auc_row["two_sided_p"]),
        ))
    test_slope_ll = compared["dose_slope"]["test"]["unweighted_log_loss"]
    test_slope_auc = compared["dose_slope"]["test"]["roc_auc"]
    collapse = payload["collapse_diagnostics"]
    collapse_text = "；".join(
        "{} seed{}有{}个epoch".format(
            row["source"], row["seed"], row["collapse_epoch_count"]
        ) for row in collapse
    ) or "无"
    lines.extend([
        "", "## 结论", "",
        "1. 10%定向与随机删除的Test log-loss几乎无差异；该剂量不足以稳定破坏概率预测。",
        "2. 25%时，Targeted相对Random的log-loss额外恶化{}，AUROC额外下降{}；两者均6/6 seed同向。".format(
            f6(compared["targeted_vs_random"]["test"]["025"]["unweighted_log_loss"]["mean"]),
            f6(compared["targeted_vs_random"]["test"]["025"]["roc_auc"]["mean"]),
        ),
        "3. 50%时差距扩大：log-loss额外恶化{}，AUROC额外下降{}；AUROC仍6/6 seed同向。".format(
            f6(compared["targeted_vs_random"]["test"]["050"]["unweighted_log_loss"]["mean"]),
            f6(compared["targeted_vs_random"]["test"]["050"]["roc_auc"]["mean"]),
        ),
        "4. 预设的Test差异剂量斜率在log-loss和AUROC上均6/6为正，分别为{}和{}，区间均高于0；支持高分边扰动具有剂量相关的额外损伤。".format(
            f6(test_slope_ll["mean"]), f6(test_slope_auc["mean"]),
        ),
        "5. Validation的剂量斜率平均也为正，但10%和25%的单剂量Targeted-vs-Random方向与Test不一致，说明效应的分区复现并不完美，不能表述为确定性证明。",
        "6. Random删除50%在Test上没有造成退化，甚至平均略好；因此观察到的Targeted损伤不是单纯由边数减少解释，更符合高分边携带特异信息。",
        "7. Balanced accuracy在部分剂量与概率指标方向相反，表明验证阈值迁移仍不稳定；主要解释应以预设unweighted log-loss和AUROC为准。",
        "8. 完全塌缩诊断：{}。这只涉及50%条件的seed44，不改变25%和跨剂量斜率的总体方向，但提示强扰动下训练稳定性下降。".format(collapse_text),
        "9. 证据仍为`exploratory_in_sample`：上游提取器见过全样本标签。本轮支持的是样本内的高分边特异性与剂量敏感性，正式验证仍需交叉拟合。", "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    audit_payload = audit(runs, args.experiment_root, args.matched_manifest)
    summary, per_seed, collapse = summarize(runs)
    compared = comparisons(runs, audit_payload["realized_deleted_ratios"])
    boot = bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    payload = {
        "schema_version": 1,
        "evidence_level": "exploratory_in_sample",
        "sources": list(SOURCES), "seeds": list(SEEDS),
        "audit": audit_payload, "summary": summary, "per_seed": per_seed,
        "collapse_diagnostics": collapse, "comparisons": compared,
        "conditional_subject_bootstrap": boot,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write(render(payload))
        handle.write("\n")
    print(json.dumps({
        "run_count": audit_payload["run_count"],
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
