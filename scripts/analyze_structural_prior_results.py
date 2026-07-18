"""Audit and analyze A-E structural feature/statistical prior experiments."""

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


GROUPS = ("A", "B", "C", "D", "E")
SEEDS = (42, 43, 44, 45, 46, 47)
METRICS = ("unweighted_log_loss", "roc_auc", "balanced_accuracy")
COMPARISONS = (
    ("features_B_vs_A", "B", "A", True),
    ("real_D_vs_permuted_E", "D", "E", True),
    ("uniform_C_vs_none_B", "C", "B", False),
    ("real_D_vs_none_B", "D", "B", False),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path,
        default=Path(
            "outputs/baseline_structural_training/key_controls_beta1_perm42_v1"
        ),
    )
    parser.add_argument(
        "--transform-root", type=Path,
        default=Path(
            "outputs/baseline_structural_transforms/key_controls_beta1_perm42_v1"
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path(
            "docs/experiment_results/structural_prior_ABCDE_seed42_47_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path(
            "docs/experiment_results/structural_prior_ABCDE_seed42_47_analysis.md"
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


def directory(root, group, seed):
    return root / "group_{}_bag_seed{}_v1".format(group, seed)


def load_runs(root):
    runs = {}
    for group in GROUPS:
        for seed in SEEDS:
            current = directory(root, group, seed)
            required = (
                "best_checkpoint.pt", "history.json", "structural_transform.json",
                "validation_evaluation.json", "test_evaluation.json",
            )
            missing = [name for name in required if not (current / name).is_file()]
            if missing:
                raise RuntimeError("{} missing {}".format(current, missing))
            checkpoint_path = current / "best_checkpoint.pt"
            checkpoint_hash = sha256(checkpoint_path)
            checkpoint = trusted_load(checkpoint_path)
            config = checkpoint["model_config"]
            if int(checkpoint["training_config"]["seed"]) != seed:
                raise RuntimeError("checkpoint seed mismatch")
            if config.get("structural_group") != group:
                raise RuntimeError("checkpoint structural group mismatch")
            if config.get("history_mode") != "independent_bag":
                raise RuntimeError("run is not Independent-bag")
            transform_path = current / "structural_transform.json"
            with transform_path.open("r", encoding="utf-8") as handle:
                transform = json.load(handle)
            if checkpoint.get("structural_transform") != transform:
                raise RuntimeError("checkpoint transform payload mismatch")
            if checkpoint.get("structural_transform_sha256") != sha256(transform_path):
                raise RuntimeError("checkpoint transform hash mismatch")
            with (current / "history.json").open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            evaluations = {}
            for split in ("validation", "test"):
                with (current / "{}_evaluation.json".format(split)).open(
                    "r", encoding="utf-8"
                ) as handle:
                    evaluation = json.load(handle)
                if evaluation["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("evaluation checkpoint mismatch")
                if evaluation.get("structural_group") != group:
                    raise RuntimeError("evaluation structural group mismatch")
                if evaluation.get("debug_limited_batches") is not None:
                    raise RuntimeError("debug-limited evaluation found")
                evaluations[split] = evaluation
            runs[(group, seed)] = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "transform": transform,
                "history": history,
                "evaluation": evaluations,
            }
    return runs


def load_prepared_transforms(root):
    output = {}
    for group in GROUPS:
        path = root / "group_{}.json".format(group)
        if not path.is_file():
            raise RuntimeError("prepared transform missing: {}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            output[group] = {"payload": json.load(handle), "sha256": sha256(path)}
    return output


def audit(runs, prepared):
    checkpoint_hashes = {run["checkpoint_sha256"] for run in runs.values()}
    if len(checkpoint_hashes) != len(runs):
        raise RuntimeError("checkpoint hashes are not unique")
    shared_fields = {}
    for field in (
        "data_protocol_sha256", "extractor_checkpoint_sha256",
        "parent_manifest_sha256", "downstream_splits_json_sha256",
        "matched_control_manifest_sha256", "subgraph_source", "evidence_level",
    ):
        values = {run["checkpoint"].get(field) for run in runs.values()}
        if len(values) != 1:
            raise RuntimeError("checkpoint field differs: {}".format(field))
        shared_fields[field] = next(iter(values))

    parameter_counts = {}
    architecture_configs = set()
    for (group, seed), run in runs.items():
        parameter_counts[(group, seed)] = sum(
            int(value.numel())
            for key, value in run["checkpoint"]["model_state_dict"].items()
            if key not in (
                "structural_mean", "structural_std", "structural_prior_scale",
                "structural_transform_fitted",
            )
        )
        config = dict(run["checkpoint"]["model_config"])
        for name in (
            "structural_group", "use_structural_features", "prior_mode"
        ):
            config.pop(name, None)
        architecture_configs.add(json.dumps(config, sort_keys=True))
        prepared_payload = prepared[group]["payload"]
        if run["transform"] != prepared_payload:
            raise RuntimeError("run transform differs from prepared group {}".format(group))
    if len(set(parameter_counts.values())) != 1 or len(architecture_configs) != 1:
        raise RuntimeError("A-E architectures or parameter counts differ")

    reference_stats = None
    for group in ("B", "C", "D", "E"):
        payload = prepared[group]["payload"]
        current = (
            payload["feature_names"], payload["mean"], payload["std"],
            payload["normalized_importance"], payload["train_sample_key_sha256"],
        )
        if reference_stats is None:
            reference_stats = current
        elif current != reference_stats:
            raise RuntimeError("B-E do not share training statistics")
        if payload.get("fitted_on") != "train_only":
            raise RuntimeError("transform was not fit on train only")
    if len(set(round(value, 12) for value in prepared["C"]["payload"]["prior_scale"])) != 1:
        raise RuntimeError("C prior is not uniform")
    d_scale = prepared["D"]["payload"]["prior_scale"]
    e_scale = prepared["E"]["payload"]["prior_scale"]
    if sorted(d_scale) != sorted(e_scale) or d_scale == e_scale:
        raise RuntimeError("D/E prior distribution control is invalid")

    alignment = {}
    for split in ("validation", "test"):
        reference = None
        manifest_hashes = defaultdict(set)
        for (group, unused_seed), run in runs.items():
            del unused_seed
            evaluation = run["evaluation"][split]
            manifest_hashes[group].add(evaluation["baseline_manifest_sha256"])
            identity = tuple(
                (row["sample_key"], int(row["label"]), row["subject_id"], row["site"])
                for row in evaluation["metrics"]["predictions"]
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise RuntimeError("{} samples are not aligned".format(split))
        if any(len(values) != 1 for values in manifest_hashes.values()):
            raise RuntimeError("evaluation manifest changed across seeds")
        alignment[split] = {
            "sample_count": len(reference),
            "class_counts": {
                str(label): sum(row[1] == label for row in reference)
                for label in (0, 1)
            },
            "aligned": True,
            "manifest_sha256": {
                group: next(iter(values)) for group, values in manifest_hashes.items()
            },
        }
    train_count = int(runs[("A", 42)]["history"][0]["train"]["sample_count"])
    return {
        "run_count": len(runs),
        "unique_checkpoint_count": len(checkpoint_hashes),
        "parameter_count": next(iter(parameter_counts.values())),
        "shared_fields": shared_fields,
        "alignment": alignment,
        "partition_sample_counts": {
            "train": train_count,
            "validation": alignment["validation"]["sample_count"],
            "test": alignment["test"]["sample_count"],
            "total": train_count + alignment["validation"]["sample_count"]
            + alignment["test"]["sample_count"],
        },
        "transform_control_checks": {
            "B_to_E_same_standardization_and_importance": True,
            "C_uniform": True,
            "D_E_same_scale_multiset_different_mapping": True,
        },
    }


def value(runs, group, seed, split, metric):
    return float(runs[(group, seed)]["evaluation"][split]["metrics"][metric])


def mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def t_ci(values):
    array = np.asarray(values, dtype=np.float64)
    radius = stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array))
    return float(array.mean() - radius), float(array.mean() + radius)


def sign_flip_p(values):
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    null = [
        abs(float(np.mean(array * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
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
    for group in GROUPS:
        summary[group] = {}
        for split in ("validation", "test"):
            summary[group][split] = {}
            for metric in METRICS + ("accuracy", "f1"):
                values = [value(runs, group, seed, split, metric) for seed in SEEDS]
                mean, sd = mean_sd(values)
                summary[group][split][metric] = {
                    "mean": mean, "sd": sd, "values": values,
                }
        for seed in SEEDS:
            history = runs[(group, seed)]["history"]
            flags = [
                float(row["validation"]["roc_auc"]) == 0.5
                and float(row["validation"]["threshold"]) == 0.5
                for row in history
            ]
            row = {
                "group": group, "seed": seed, "epochs_completed": len(history),
                "collapse_epoch_count": int(sum(flags)),
            }
            for split in ("validation", "test"):
                for metric in METRICS + ("accuracy", "f1", "threshold"):
                    row["{}_{}".format(split, metric)] = value(
                        runs, group, seed, split, metric
                    )
            per_seed.append(row)
            if any(flags):
                collapse.append({
                    "group": group, "seed": seed,
                    "epochs_completed": len(history),
                    "collapse_epoch_count": int(sum(flags)),
                })
    return summary, per_seed, collapse


def compare(runs):
    output = {}
    for split in ("validation", "test"):
        output[split] = {}
        for name, target, control, primary in COMPARISONS:
            output[split][name] = {"target": target, "control": control, "primary": primary}
            for metric in METRICS:
                target_values = np.asarray([
                    value(runs, target, seed, split, metric) for seed in SEEDS
                ])
                control_values = np.asarray([
                    value(runs, control, seed, split, metric) for seed in SEEDS
                ])
                improvement = (
                    control_values - target_values
                    if metric == "unweighted_log_loss"
                    else target_values - control_values
                )
                low, high = t_ci(improvement)
                output[split][name][metric] = {
                    "values": improvement.tolist(),
                    "mean": float(improvement.mean()),
                    "median": float(np.median(improvement)),
                    "sd": float(improvement.std(ddof=1)),
                    "ci95_low": low, "ci95_high": high,
                    "wins": int(np.sum(improvement > 0.0)),
                    "exact_sign_flip_p": sign_flip_p(improvement),
                }
        primary_names = [name for name, _, _, primary in COMPARISONS if primary]
        pvalues = [
            output[split][name]["unweighted_log_loss"]["exact_sign_flip_p"]
            for name in primary_names
        ]
        for name, qvalue in zip(primary_names, bh_adjust(pvalues)):
            output[split][name]["unweighted_log_loss"]["primary_bh_q"] = qvalue
    return output


def prediction_rows(runs, group, seed):
    return runs[(group, seed)]["evaluation"]["test"]["metrics"]["predictions"]


def auc(labels, probabilities):
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


def log_loss(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities), 1e-12, 1.0 - 1e-12)
    return float(np.mean(-labels * np.log(probabilities) - (1 - labels) * np.log(1 - probabilities)))


def bootstrap(runs, repeats, seed):
    reference = prediction_rows(runs, "A", 42)
    labels = np.asarray([int(row["label"]) for row in reference])
    groups = defaultdict(list)
    for index, row in enumerate(reference):
        groups[(row["site"], row["subject_id"])].append(index)
    group_indices = list(groups.values())
    probabilities = {}
    for group in GROUPS:
        probabilities[group] = np.mean(np.stack([
            np.asarray([
                float(row["class_1_probability"])
                for row in prediction_rows(runs, group, current_seed)
            ])
            for current_seed in SEEDS
        ]), axis=0)
    rng = np.random.RandomState(seed)
    output = {}
    for name, target, control, primary in COMPARISONS:
        ll_draws = np.empty(repeats)
        auc_draws = np.empty(repeats)
        for repeat in range(repeats):
            selected_groups = rng.randint(0, len(group_indices), len(group_indices))
            indices = np.asarray([
                item for group_index in selected_groups for item in group_indices[group_index]
            ])
            current_labels = labels[indices]
            target_probability = probabilities[target][indices]
            control_probability = probabilities[control][indices]
            ll_draws[repeat] = log_loss(current_labels, control_probability) - log_loss(
                current_labels, target_probability
            )
            auc_draws[repeat] = auc(current_labels, target_probability) - auc(
                current_labels, control_probability
            )
        def describe(draws, observed):
            draws = draws[np.isfinite(draws)]
            return {
                "observed": float(observed),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "two_sided_p": float(min(
                    1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))
                )),
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
        "conditioning": "probabilities averaged over six fitted seeds",
        "sample_count": len(labels), "subject_count": len(group_indices),
    }
    return output


def prior_ranking(prepared):
    payload = prepared["D"]["payload"]
    rows = [
        {
            "feature": feature,
            "effect_size": float(effect),
            "normalized_importance": float(importance),
            "prior_scale": float(scale),
        }
        for feature, effect, importance, scale in zip(
            payload["feature_names"], payload["effect_size"],
            payload["normalized_importance"], payload["prior_scale"],
        )
    ]
    return sorted(rows, key=lambda row: (-row["normalized_importance"], row["feature"]))


def f6(value):
    return "{:.6f}".format(float(value))


def render(payload):
    audit_payload = payload["audit"]
    summary = payload["summary"]
    comparisons = payload["paired_comparisons"]
    boot = payload["conditional_subject_bootstrap"]
    lines = [
        "# A–E结构特征与统计先验实验（seed 42–47）", "",
        "## 验收", "",
        "- 30/30次训练、checkpoint、结构artifact、validation和test评估完整。",
        "- 30个checkpoint哈希均不同；A–E参数量相同（{}），模型容量匹配。".format(
            audit_payload["parameter_count"]
        ),
        "- B–E共享完全相同的train标准化与效应量；C为统一缩放；D/E权重多重集合相同但映射不同。",
        "- train/validation/test={}/{}/{}，validation/test样本、标签、subject、site及顺序完全对齐。".format(
            audit_payload["partition_sample_counts"]["train"],
            audit_payload["partition_sample_counts"]["validation"],
            audit_payload["partition_sample_counts"]["test"],
        ), "",
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
        "## Test配对比较", "",
        "正值表示目标组更好；log-loss使用`控制−目标`。", "",
        "| 比较 | ΔLog-loss [95% seed CI] | 胜/6 | p | 主假设q | ΔAUROC | 胜/6 | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, target, control, primary in COMPARISONS:
        row = comparisons["test"][name]
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
        "| 比较 | ΔLog-loss [95% CI] | p | ΔAUROC [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, target, control, primary in COMPARISONS:
        row = boot[name]
        ll = row["log_loss"]
        auc_row = row["roc_auc"]
        lines.append("| {} vs {} | {} [{}, {}] | {} | {} [{}, {}] | {} |".format(
            target, control, f6(ll["observed"]), f6(ll["ci95_low"]), f6(ll["ci95_high"]),
            f6(ll["two_sided_p"]), f6(auc_row["observed"]),
            f6(auc_row["ci95_low"]), f6(auc_row["ci95_high"]), f6(auc_row["two_sided_p"]),
        ))
    lines.extend([
        "", "## Train折真实先验权重", "",
        "| 排名 | 结构指标 | 标准化效应量 | 归一化重要性 | 缩放 |",
        "|---:|---|---:|---:|---:|",
    ])
    for index, row in enumerate(payload["real_prior_ranking"], 1):
        lines.append("| {} | {} | {} | {} | {} |".format(
            index, row["feature"], f6(row["effect_size"]),
            f6(row["normalized_importance"]), f6(row["prior_scale"]),
        ))
    ba = comparisons["test"]["features_B_vs_A"]
    de = comparisons["test"]["real_D_vs_permuted_E"]
    cb = comparisons["test"]["uniform_C_vs_none_B"]
    db = comparisons["test"]["real_D_vs_none_B"]
    lines.extend([
        "", "## 结论", "",
        "1. B vs A的主要log-loss平均改善为{}，仅{}/6 seed胜出，95% seed区间跨0；真实结构特征本身未显示稳定增益。".format(
            f6(ba["unweighted_log_loss"]["mean"]), ba["unweighted_log_loss"]["wins"]
        ),
        "2. D vs E的主要log-loss差值为{}，仅{}/6 seed支持D；负值表示真实映射反而逊于置乱映射。D的AUROC也平均低{}。".format(
            f6(de["unweighted_log_loss"]["mean"]), de["unweighted_log_loss"]["wins"],
            f6(abs(de["roc_auc"]["mean"])),
        ),
        "3. 因此当前实验不支持“train折真实统计先验映射有效”这一假设；E表现最好更符合一般缩放/正则化或优化效应，而不是指标语义映射带来的增益。",
        "4. C vs B的log-loss平均改善{}，D vs B仅{}；真实先验没有优于无先验，而统一缩放有轻微但不稳定的方向。".format(
            f6(cb["unweighted_log_loss"]["mean"]), f6(db["unweighted_log_loss"]["mean"])
        ),
        "5. 任何单组均值优势都不能替代预设比较：即使E的平均Test表现最佳，也不能称为统计先验成功，因为E故意破坏了真实维度对应关系。",
        "6. 结果提示创新方向不应建立在当前静态效应量先验上；更值得研究的是跨窗口结构分布、稳定性或关键子图相对控制的特异性权重，而非单次train折类别效应量。",
        "7. 证据等级仍为`exploratory_in_sample`，上游提取器使用过全样本标签，不能解释为样本外确认。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = load_runs(args.training_root)
    prepared = load_prepared_transforms(args.transform_root)
    audit_payload = audit(runs, prepared)
    summary, per_seed, collapse = summarize(runs)
    comparisons = compare(runs)
    bootstrap_payload = bootstrap(runs, args.bootstrap_repeats, args.bootstrap_seed)
    payload = {
        "schema_version": 1,
        "evidence_level": "exploratory_in_sample",
        "groups": list(GROUPS), "seeds": list(SEEDS),
        "audit": audit_payload,
        "summary": summary,
        "per_seed": per_seed,
        "collapse_diagnostics": collapse,
        "paired_comparisons": comparisons,
        "conditional_subject_bootstrap": bootstrap_payload,
        "real_prior_ranking": prior_ranking(prepared),
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
