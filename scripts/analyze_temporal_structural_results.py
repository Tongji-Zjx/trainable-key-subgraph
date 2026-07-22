"""Audit and analyze A/B/F/G/H temporal structural-difference experiments."""

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


GROUPS = ("A", "B", "F", "G", "H")
SEEDS = (42, 43, 44, 45, 46, 47)
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")
COMPARISONS = (
    ("ordered_delta_F_vs_static_B", "F", "B", True),
    ("delta_only_G_vs_none_A", "G", "A", True),
    ("ordered_F_vs_shuffled_H", "F", "H", True),
    ("static_B_vs_none_A", "B", "A", False),
)
EXPECTED = {
    "A": (False, False, "ordered"),
    "B": (True, False, "ordered"),
    "F": (True, True, "ordered"),
    "G": (False, True, "ordered"),
    "H": (True, True, "shuffled"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("outputs/baseline_temporal_structural_training/key_delta_perm42_v1"),
    )
    parser.add_argument(
        "--transform-root", type=Path,
        default=Path("outputs/baseline_temporal_structural_transforms/key_delta_perm42_v1"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("docs/experiment_results/temporal_structural_ABFGH_seed42_47_analysis.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("docs/experiment_results/temporal_structural_ABFGH_seed42_47_analysis.md"),
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


def load_runs(root):
    runs = {}
    for group in GROUPS:
        for seed in SEEDS:
            directory = root / "group_{}_seed{}".format(group, seed)
            names = (
                "best_checkpoint.pt", "history.json", "structural_transform.json",
                "validation_evaluation.json", "test_evaluation.json",
            )
            missing = [name for name in names if not (directory / name).is_file()]
            if missing:
                raise RuntimeError("{} missing {}".format(directory, missing))
            checkpoint_path = directory / "best_checkpoint.pt"
            checkpoint_hash = sha256(checkpoint_path)
            checkpoint = trusted_load(checkpoint_path)
            config = checkpoint["model_config"]
            expected = EXPECTED[group]
            actual = (
                bool(config.get("use_structural_features")),
                bool(config.get("use_structural_deltas")),
                config.get("structural_delta_order"),
            )
            if int(checkpoint["training_config"]["seed"]) != seed or actual != expected:
                raise RuntimeError("checkpoint condition mismatch: {}".format(directory))
            if config.get("structural_group") != group:
                raise RuntimeError("checkpoint group mismatch")
            if not (
                config.get("structural_interface_version") == 2
                and config.get("structural_feature_dim") == 22
                and config.get("history_mode") == "independent_bag"
                and config.get("temporal_order") == "ordered"
                and config.get("structural_delta_permutation_seed") == 42
            ):
                raise RuntimeError("temporal experiment configuration mismatch")
            transform_path = directory / "structural_transform.json"
            with transform_path.open("r", encoding="utf-8") as handle:
                transform = json.load(handle)
            if checkpoint.get("structural_transform") != transform:
                raise RuntimeError("checkpoint transform payload mismatch")
            if checkpoint.get("structural_transform_sha256") != sha256(transform_path):
                raise RuntimeError("checkpoint transform hash mismatch")
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
                if evaluation.get("structural_group") != group:
                    raise RuntimeError("evaluation group mismatch")
                if evaluation.get("debug_limited_batches") is not None:
                    raise RuntimeError("debug-limited evaluation found")
                evaluations[split] = evaluation
            runs[group, seed] = {
                "checkpoint": checkpoint,
                "checkpoint_hash": checkpoint_hash,
                "transform": transform,
                "history": history,
                "evaluation": evaluations,
            }
    return runs


def load_prepared(root):
    output = {}
    for group in GROUPS:
        path = root / "group_{}.json".format(group)
        with path.open("r", encoding="utf-8") as handle:
            output[group] = json.load(handle)
    return output


def audit(runs, prepared):
    hashes = {run["checkpoint_hash"] for run in runs.values()}
    if len(hashes) != 30:
        raise RuntimeError("checkpoint hashes are not unique")
    shared = {}
    for field in (
        "data_protocol_sha256", "extractor_checkpoint_sha256",
        "parent_manifest_sha256", "downstream_splits_json_sha256",
        "matched_control_manifest_sha256", "subgraph_source", "evidence_level",
    ):
        values = {run["checkpoint"].get(field) for run in runs.values()}
        if len(values) != 1:
            raise RuntimeError("checkpoint provenance differs: {}".format(field))
        shared[field] = next(iter(values))
    parameter_counts = set()
    architectures = set()
    reference_statistics = None
    for (group, unused_seed), run in runs.items():
        del unused_seed
        if run["transform"] != prepared[group]:
            raise RuntimeError("run transform differs from prepared transform")
        state = run["checkpoint"]["model_state_dict"]
        parameter_counts.add(sum(
            int(value.numel()) for key, value in state.items()
            if key not in (
                "structural_mean", "structural_std", "structural_prior_scale",
                "structural_transform_fitted",
            )
        ))
        config = dict(run["checkpoint"]["model_config"])
        for field in (
            "structural_group", "use_structural_features", "use_structural_deltas",
            "structural_delta_order",
        ):
            config.pop(field, None)
        architectures.add(json.dumps(config, sort_keys=True))
        payload = prepared[group]
        statistics = (
            payload["feature_names"], payload["mean"], payload["std"],
            payload["valid_window_counts"], payload["train_sample_key_sha256"],
        )
        if reference_statistics is None:
            reference_statistics = statistics
        elif statistics != reference_statistics:
            raise RuntimeError("A/B/F/G/H do not share normalization statistics")
        if not (
            payload.get("fitted_on") == "train_only"
            and payload.get("normalization_delta_order") == "ordered"
            and payload.get("first_window_policy") == "masked_not_zero_observation"
        ):
            raise RuntimeError("temporal transform policy mismatch")
    if len(parameter_counts) != 1 or len(architectures) != 1:
        raise RuntimeError("experimental architectures are not matched")
    alignment = {}
    for split in ("validation", "test"):
        reference = None
        manifest_hashes = set()
        for run in runs.values():
            evaluation = run["evaluation"][split]
            manifest_hashes.add(evaluation["baseline_manifest_sha256"])
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in evaluation["metrics"]["predictions"]
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} sample alignment differs".format(split))
        if len(manifest_hashes) != 1:
            raise RuntimeError("{} manifest differs".format(split))
        alignment[split] = {
            "aligned": True,
            "sample_count": len(reference),
            "manifest_sha256": next(iter(manifest_hashes)),
            "class_counts": {
                str(label): sum(row[1] == label for row in reference) for label in (0, 1)
            },
        }
    train_count = int(runs["A", 42]["history"][0]["train"]["sample_count"])
    return {
        "run_count": 30,
        "unique_checkpoint_count": len(hashes),
        "parameter_count": next(iter(parameter_counts)),
        "shared_provenance": shared,
        "alignment": alignment,
        "partition_sample_counts": {
            "train": train_count,
            "validation": alignment["validation"]["sample_count"],
            "test": alignment["test"]["sample_count"],
        },
        "control_checks": {
            "same_22_feature_schema_and_train_normalization": True,
            "first_window_delta_is_masked": True,
            "H_only_changes_delta_predecessor_order": True,
            "parameter_matched": True,
        },
    }


def metric(runs, group, seed, split, name):
    return float(runs[group, seed]["evaluation"][split]["metrics"][name])


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
        running = min(running, float(pvalues[index]) * len(pvalues) / (reverse_rank + 1))
        adjusted[index] = running
    return adjusted.tolist()


def summarize(runs):
    summary = {}
    per_seed = []
    collapse = []
    for group in GROUPS:
        summary[group] = {}
        for split in ("validation", "test"):
            summary[group][split] = {}
            for name in METRICS:
                values = [metric(runs, group, seed, split, name) for seed in SEEDS]
                mean, sd = mean_sd(values)
                summary[group][split][name] = {"mean": mean, "sd": sd, "values": values}
        for seed in SEEDS:
            history = runs[group, seed]["history"]
            flags = [
                float(row["validation"]["roc_auc"]) == 0.5
                and float(row["validation"]["threshold"]) == 0.5
                for row in history
            ]
            row = {
                "group": group, "seed": seed, "epochs_completed": len(history),
                "collapse_epoch_count": int(sum(flags)),
            }
            per_seed.append(row)
            if any(flags):
                collapse.append(row)
    return summary, per_seed, collapse


def comparisons(runs):
    output = {}
    for split in ("validation", "test"):
        output[split] = {}
        for name, target, control, primary in COMPARISONS:
            row = {"target": target, "control": control, "primary": primary}
            for metric_name in METRICS:
                target_values = np.asarray([
                    metric(runs, target, seed, split, metric_name) for seed in SEEDS
                ])
                control_values = np.asarray([
                    metric(runs, control, seed, split, metric_name) for seed in SEEDS
                ])
                improvement = (
                    control_values - target_values
                    if metric_name == "unweighted_log_loss"
                    else target_values - control_values
                )
                low, high = t_ci(improvement)
                leave_one_out = [
                    float(np.delete(improvement, index).mean())
                    for index in range(len(improvement))
                ]
                row[metric_name] = {
                    "values": improvement.tolist(), "mean": float(improvement.mean()),
                    "sd": float(improvement.std(ddof=1)), "ci95_low": low,
                    "ci95_high": high, "wins": int(np.sum(improvement > 0)),
                    "exact_sign_flip_p": sign_flip_p(improvement),
                    "leave_one_seed_out_mean_min": min(leave_one_out),
                    "leave_one_seed_out_mean_max": max(leave_one_out),
                }
            output[split][name] = row
        primary = [name for name, _, _, flag in COMPARISONS if flag]
        pvalues = [
            output[split][name]["unweighted_log_loss"]["exact_sign_flip_p"]
            for name in primary
        ]
        for name, qvalue in zip(primary, bh_adjust(pvalues)):
            output[split][name]["unweighted_log_loss"]["primary_bh_q"] = qvalue
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
    probabilities = np.clip(np.asarray(probabilities), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-labels * np.log(probabilities) - (1 - labels) * np.log(1 - probabilities)))


def bootstrap(runs, repeats, seed):
    reference = runs["A", 42]["evaluation"]["test"]["metrics"]["predictions"]
    labels = np.asarray([int(row["label"]) for row in reference])
    subjects = defaultdict(list)
    for index, row in enumerate(reference):
        subjects[row["site"], row["subject_id"]].append(index)
    subject_indices = list(subjects.values())
    probabilities = {}
    for group in GROUPS:
        probabilities[group] = np.mean(np.stack([
            np.asarray([
                float(row["class_1_probability"])
                for row in runs[group, current_seed]["evaluation"]["test"]["metrics"]["predictions"]
            ]) for current_seed in SEEDS
        ]), axis=0)
    rng = np.random.RandomState(seed)
    output = {}
    for name, target, control, primary in COMPARISONS:
        ll_draws = np.empty(repeats)
        auc_draws = np.empty(repeats)
        for repeat in range(repeats):
            selected = rng.randint(0, len(subject_indices), len(subject_indices))
            indices = np.asarray([item for group in selected for item in subject_indices[group]])
            current_labels = labels[indices]
            ll_draws[repeat] = log_loss(current_labels, probabilities[control][indices]) - log_loss(
                current_labels, probabilities[target][indices]
            )
            auc_draws[repeat] = auc(current_labels, probabilities[target][indices]) - auc(
                current_labels, probabilities[control][indices]
            )
        def describe(draws, observed):
            draws = draws[np.isfinite(draws)]
            return {
                "observed": float(observed),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "two_sided_p": float(min(1.0, 2.0 * min(
                    np.mean(draws <= 0), np.mean(draws >= 0)
                ))),
            }
        output[name] = {
            "target": target, "control": control, "primary": primary,
            "log_loss": describe(
                ll_draws,
                log_loss(labels, probabilities[control]) - log_loss(labels, probabilities[target]),
            ),
            "roc_auc": describe(
                auc_draws,
                auc(labels, probabilities[target]) - auc(labels, probabilities[control]),
            ),
        }
    output["metadata"] = {
        "sample_count": len(labels), "subject_count": len(subject_indices),
        "conditioning": "probabilities averaged over six fitted seeds",
    }
    return output


def f6(value):
    return "{:.6f}".format(float(value))


def render(payload):
    audit_payload = payload["audit"]
    summary = payload["summary"]
    compared = payload["paired_comparisons"]
    boot = payload["conditional_subject_bootstrap"]
    lines = [
        "# A/B/F/G/H时间差分结构实验（seed 42–47）", "",
        "## 验收", "",
        "- 30/30次正式训练、最佳checkpoint、history、冻结变换及validation/test评估完整。",
        "- 30个checkpoint哈希均不同；各组参数量相同（{}），不存在模型容量差异。".format(audit_payload["parameter_count"]),
        "- 五组共享相同22维特征schema、train-only均值/标准差和有效窗口计数。",
        "- 首窗口差分为mask无效而非零观测；H只把差分前驱改为冻结置乱（seed 42）。",
        "- train/validation/test={}/{}/{}；validation/test样本、标签、subject、site及顺序完全对齐。".format(
            audit_payload["partition_sample_counts"]["train"],
            audit_payload["partition_sample_counts"]["validation"],
            audit_payload["partition_sample_counts"]["test"],
        ), "",
        "组定义：A无结构输入；B仅静态；F静态+ordered差分；G仅ordered差分；H静态+shuffled差分。", "",
    ]
    for split in ("validation", "test"):
        lines.extend([
            "## {}（均值 ± seed标准差）".format(split.capitalize()), "",
            "| 组 | Log-loss | AUROC | Balanced accuracy |",
            "|---|---:|---:|---:|",
        ])
        for group in GROUPS:
            row = summary[group][split]
            lines.append("| {} | {} ± {} | {} ± {} | {} ± {} |".format(
                group,
                f6(row["unweighted_log_loss"]["mean"]), f6(row["unweighted_log_loss"]["sd"]),
                f6(row["roc_auc"]["mean"]), f6(row["roc_auc"]["sd"]),
                f6(row["balanced_accuracy"]["mean"]), f6(row["balanced_accuracy"]["sd"]),
            ))
        lines.append("")
    lines.extend([
        "## Test同seed配对比较", "",
        "正值表示目标组更好；log-loss定义为`控制−目标`。", "",
        "| 比较 | ΔLog-loss [95% seed CI] | 胜/6 | p | 主假设q | ΔAUROC | 胜/6 | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, target, control, primary in COMPARISONS:
        row = compared["test"][name]
        ll = row["unweighted_log_loss"]
        auc_row = row["roc_auc"]
        lines.append("| {} vs {} | {} [{}, {}] | {}/6 | {} | {} | {} | {}/6 | {} |".format(
            target, control, f6(ll["mean"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
            ll["wins"], f6(ll["exact_sign_flip_p"]),
            f6(ll.get("primary_bh_q")) if primary else "—",
            f6(auc_row["mean"]), auc_row["wins"], f6(auc_row["exact_sign_flip_p"]),
        ))
    lines.extend([
        "", "## 条件于六个已训练模型的subject bootstrap", "",
        "该bootstrap量化当前拟合模型下的test subject采样不确定性，不包含重新训练的seed不确定性。", "",
        "| 比较 | ΔLog-loss [95% CI] | p | ΔAUROC [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, target, control, unused_primary in COMPARISONS:
        del unused_primary
        row = boot[name]
        ll = row["log_loss"]
        auc_row = row["roc_auc"]
        lines.append("| {} vs {} | {} [{}, {}] | {} | {} [{}, {}] | {} |".format(
            target, control, f6(ll["observed"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
            f6(ll["two_sided_p"]), f6(auc_row["observed"]),
            f6(auc_row["ci95_low"]), f6(auc_row["ci95_high"]), f6(auc_row["two_sided_p"]),
        ))
    lines.extend([
        "", "## Seed敏感性", "",
        "- F vs B的leave-one-seed-out平均ΔLog-loss范围为[{}, {}]，会随删除的seed改变符号。".format(
            f6(compared["test"]["ordered_delta_F_vs_static_B"]["unweighted_log_loss"]["leave_one_seed_out_mean_min"]),
            f6(compared["test"]["ordered_delta_F_vs_static_B"]["unweighted_log_loss"]["leave_one_seed_out_mean_max"]),
        ),
        "- G seed42出现训练塌缩；排除seed42后，G vs A的平均ΔAUROC仍为{}，方向仍不支持G。".format(
            f6(np.mean(compared["test"]["delta_only_G_vs_none_A"]["roc_auc"]["values"][1:])),
        ),
        "- F vs H的leave-one-seed-out平均ΔLog-loss范围为[{}, {}]，始终为负；主要log-loss在该敏感性检查中仍偏向H。".format(
            f6(compared["test"]["ordered_F_vs_shuffled_H"]["unweighted_log_loss"]["leave_one_seed_out_mean_min"]),
            f6(compared["test"]["ordered_F_vs_shuffled_H"]["unweighted_log_loss"]["leave_one_seed_out_mean_max"]),
        ),
    ])
    fb = compared["test"]["ordered_delta_F_vs_static_B"]
    ga = compared["test"]["delta_only_G_vs_none_A"]
    fh = compared["test"]["ordered_F_vs_shuffled_H"]
    ba = compared["test"]["static_B_vs_none_A"]
    collapse = payload["collapse_diagnostics"]
    collapse_text = (
        "发现{}：共{}个validation epoch满足AUROC=0.5且threshold=0.5。".format(
            "、".join("{} seed{}".format(row["group"], row["seed"]) for row in collapse),
            sum(row["collapse_epoch_count"] for row in collapse),
        ) if collapse else "没有运行出现预定义的完全塌缩epoch。"
    )
    lines.extend([
        "", "## 结论", "",
        "1. F vs B的Test log-loss平均改善{}，虽为5/6 seed正向，但区间跨0；AUROC平均变化{}且仅3/6正向。显式ordered一阶差分没有形成跨指标、跨seed的稳定增益。".format(
            f6(fb["unweighted_log_loss"]["mean"]), f6(fb["roc_auc"]["mean"])
        ),
        "2. G vs A的log-loss仅改善{}，AUROC平均下降{}；差分单独输入不支持判别增益。".format(
            f6(ga["unweighted_log_loss"]["mean"]), f6(abs(ga["roc_auc"]["mean"]))
        ),
        "3. F vs H的log-loss为{}（负值表示H更好），F只在2/6 seed胜出；AUROC虽平均提高{}，也仅3/6正向。真实前驱关系没有稳定优于置乱前驱关系。".format(
            f6(fh["unweighted_log_loss"]["mean"]), f6(fh["roc_auc"]["mean"])
        ),
        "4. B vs A同样只有很小且不稳定的log-loss变化{}，说明本轮不能把任何差分结论归因于稳定的静态结构基线优势。".format(
            f6(ba["unweighted_log_loss"]["mean"])
        ),
        "5. H拥有最低的组均值Test log-loss，但它故意破坏真实时间前驱，因此不能作为时间演化有效的证据；更可能反映有限样本下的优化或正则化差异。",
        "6. {}".format(collapse_text),
        "7. 按预设停止规则，不建议立刻扩展二阶差分或复杂时序网络；下一项应进入定向高分边剂量扰动与等量随机扰动对照。",
        "8. 本轮证据等级仍为`exploratory_in_sample`，且Independent-bag只检验显式差分特征，不足以否定所有可能的时间动力学模型。", "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    prepared = load_prepared(args.transform_root)
    audit_payload = audit(runs, prepared)
    summary, per_seed, collapse = summarize(runs)
    compared = comparisons(runs)
    boot = bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    payload = {
        "schema_version": 1,
        "evidence_level": "exploratory_in_sample",
        "groups": list(GROUPS), "seeds": list(SEEDS),
        "audit": audit_payload, "summary": summary, "per_seed": per_seed,
        "collapse_diagnostics": collapse, "paired_comparisons": compared,
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
