#!/usr/bin/env python3
"""Read-only WMRC development-validation versus fixed-test distribution audit."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.data import (  # noqa: E402
    SIGNED_CONNECTIVITY_PROFILE_NAMES,
    STATIC_FEATURE_NAMES,
    build_global_static_record,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir", action="append", type=Path, required=True,
        help="four frozen source folds containing cache/{validation,test}/manifest.json",
    )
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--safe-static-root", type=Path, required=True)
    parser.add_argument("--frozen-subgraph-prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spectral-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_records(path, expected_split):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records are missing: {}".format(path))
    seen = set()
    for row in records:
        key = str(row.get("sample_key", ""))
        if not key or key in seen:
            raise ValueError("manifest sample keys are invalid")
        if str(row.get("split")) != expected_split:
            raise ValueError("manifest split mismatch")
        seen.add(key)
    return records


def validate_cohorts(fold_dirs):
    validation = []
    validation_keys = set()
    test_reference = None
    for index, fold_dir in enumerate(fold_dirs):
        current = read_records(
            Path(fold_dir) / "cache" / "validation" / "manifest.json",
            "validation",
        )
        current_keys = {str(row["sample_key"]) for row in current}
        if validation_keys.intersection(current_keys):
            raise ValueError("validation rotations overlap")
        validation_keys.update(current_keys)
        validation.extend(current)
        test = read_records(
            Path(fold_dir) / "cache" / "test" / "manifest.json", "test"
        )
        signature = sorted(
            (str(row["sample_key"]), int(row["label"]), str(row["site"]))
            for row in test
        )
        if test_reference is None:
            test_reference = (test, signature)
        elif signature != test_reference[1]:
            raise ValueError("fixed test cohorts differ across rotations")
    test = test_reference[0]
    test_keys = {str(row["sample_key"]) for row in test}
    if validation_keys.intersection(test_keys):
        raise ValueError("development validation and fixed test overlap")
    return validation, test


def proportions(values, vocabulary):
    counts = Counter(str(value) for value in values)
    total = float(sum(counts.values()))
    return {
        key: {"count": int(counts.get(key, 0)), "proportion": counts.get(key, 0) / total}
        for key in vocabulary
    }


def jensen_shannon(first, second, epsilon=1.0e-12):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / first.sum()
    second = second / second.sum()
    middle = 0.5 * (first + second)

    def divergence(values):
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log((values[mask] + epsilon) / middle[mask])))

    return 0.5 * divergence(first) + 0.5 * divergence(second)


def composition_audit(development, test):
    result = {}
    definitions = {
        "label": lambda row: str(int(row["label"])),
        "site": lambda row: str(row["site"]),
        "site_label": lambda row: "{}|{}".format(row["site"], int(row["label"])),
    }
    for name, function in definitions.items():
        development_values = [function(row) for row in development]
        test_values = [function(row) for row in test]
        vocabulary = sorted(set(development_values).union(test_values))
        first = proportions(development_values, vocabulary)
        second = proportions(test_values, vocabulary)
        result[name] = {
            "development": first,
            "test": second,
            "maximum_absolute_proportion_difference": max(
                abs(first[key]["proportion"] - second[key]["proportion"])
                for key in vocabulary
            ),
            "jensen_shannon_divergence": jensen_shannon(
                [first[key]["proportion"] for key in vocabulary],
                [second[key]["proportion"] for key in vocabulary],
            ),
        }
    return result


def graph_vector(record):
    positive = record.raw_positive_adjacency.numpy()
    negative = record.raw_negative_adjacency.numpy()
    count = int(record.node_count)
    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    positive_values = positive[upper]
    negative_values = negative[upper]
    absolute_values = positive_values + negative_values
    present = absolute_values > 0.0
    positive_present = positive_values > 0.0
    negative_present = negative_values > 0.0
    possible = float(upper.sum())
    present_count = int(present.sum())
    communities = record.community_labels.numpy()
    community_sizes = np.asarray(list(Counter(communities.tolist()).values()), dtype=np.float64)

    def selected_mean(values, mask):
        return float(np.mean(values[mask])) if bool(mask.any()) else 0.0

    values = [
        float(count),
        present_count / possible,
        float(positive_present.sum()) / possible,
        float(negative_present.sum()) / possible,
        float(positive_present.sum()) / max(float(present_count), 1.0),
        selected_mean(absolute_values, present),
        float(np.std(absolute_values[present])) if bool(present.any()) else 0.0,
        selected_mean(positive_values, positive_present),
        selected_mean(negative_values, negative_present),
        float(community_sizes.size),
        float(np.mean(community_sizes)),
        float(np.std(community_sizes)),
    ]
    values.extend(record.eigenvalues.numpy().astype(np.float64).tolist())
    if record.eigenvalues.numel() > 1:
        values.append(float(record.eigenvalues[1] - record.eigenvalues[0]))
    else:
        values.append(0.0)
    node_features = record.node_features.numpy().astype(np.float64)
    non_spectral_indices = list(range(len(STATIC_FEATURE_NAMES))) + list(
        range(
            len(STATIC_FEATURE_NAMES) + int(record.eigenvalues.numel()),
            node_features.shape[1],
        )
    )
    selected = node_features[:, non_spectral_indices]
    values.extend(np.mean(selected, axis=0).tolist())
    values.extend(np.std(selected, axis=0).tolist())
    return np.asarray(values, dtype=np.float64)


def graph_feature_names(spectral_dim):
    names = [
        "node_count", "edge_density", "positive_edge_density",
        "negative_edge_density", "positive_fraction_among_present_edges",
        "mean_absolute_edge_weight", "std_absolute_edge_weight",
        "mean_positive_edge_weight", "mean_negative_edge_magnitude",
        "community_count", "community_size_mean", "community_size_std",
    ]
    names.extend("signed_laplacian_eigenvalue_{}".format(i) for i in range(spectral_dim))
    names.append("signed_laplacian_first_gap")
    aggregate = list(STATIC_FEATURE_NAMES) + list(SIGNED_CONNECTIVITY_PROFILE_NAMES)
    names.extend("node_mean_{}".format(name) for name in aggregate)
    names.extend("node_std_{}".format(name) for name in aggregate)
    return names


def ks_statistic(first, second):
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    values = np.unique(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, values, side="right") / float(first.size)
    second_cdf = np.searchsorted(second, values, side="right") / float(second.size)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def population_stability_index(reference, target, bins=10, epsilon=1.0e-6):
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cuts = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))[1:-1]
    reference_counts = np.histogram(reference, bins=np.concatenate(([-np.inf], cuts, [np.inf])))[0]
    target_counts = np.histogram(target, bins=np.concatenate(([-np.inf], cuts, [np.inf])))[0]
    reference_probability = reference_counts / float(reference.size)
    target_probability = target_counts / float(target.size)
    reference_probability = np.clip(reference_probability, epsilon, None)
    target_probability = np.clip(target_probability, epsilon, None)
    return float(
        np.sum(
            (target_probability - reference_probability)
            * np.log(target_probability / reference_probability)
        )
    )


def continuous_shift(development, test, names):
    rows = []
    for index, name in enumerate(names):
        first = development[:, index]
        second = test[:, index]
        variance = 0.5 * (float(np.var(first)) + float(np.var(second)))
        smd = (float(np.mean(second)) - float(np.mean(first))) / math.sqrt(
            max(variance, 1.0e-12)
        )
        row = {
            "feature": name,
            "development_mean": float(np.mean(first)),
            "development_standard_deviation": float(np.std(first)),
            "test_mean": float(np.mean(second)),
            "test_standard_deviation": float(np.std(second)),
            "standardized_mean_difference_test_minus_development": float(smd),
            "absolute_standardized_mean_difference": abs(float(smd)),
            "ks_statistic": ks_statistic(first, second),
            "population_stability_index": population_stability_index(first, second),
        }
        row["screening_score"] = max(
            row["absolute_standardized_mean_difference"], row["ks_statistic"]
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row["screening_score"], reverse=True)


def label_association_audit(development_features, test_features, development_rows, test_rows, names):
    from sklearn.metrics import roc_auc_score

    development_labels = np.asarray(
        [int(row["label"]) for row in development_rows], dtype=np.int64
    )
    test_labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    rows = []
    for index, name in enumerate(names):
        development_values = development_features[:, index]
        test_values = test_features[:, index]
        development_auc = (
            float(roc_auc_score(development_labels, development_values))
            if np.std(development_values) > 0.0 else 0.5
        )
        test_auc = (
            float(roc_auc_score(test_labels, test_values))
            if np.std(test_values) > 0.0 else 0.5
        )

        def effect(values, labels):
            negative = values[labels == 0]
            positive = values[labels == 1]
            pooled = math.sqrt(
                max(0.5 * (float(np.var(negative)) + float(np.var(positive))), 1.0e-12)
            )
            return (float(np.mean(positive)) - float(np.mean(negative))) / pooled

        development_effect = effect(development_values, development_labels)
        test_effect = effect(test_values, test_labels)
        rows.append(
            {
                "feature": name,
                "development_label_auc": development_auc,
                "test_label_auc": test_auc,
                "absolute_auc_change": abs(test_auc - development_auc),
                "development_class_effect": development_effect,
                "test_class_effect": test_effect,
                "direction_agreement": bool(
                    np.sign(development_effect) == np.sign(test_effect)
                ),
            }
        )
    development_auc_effects = np.asarray(
        [row["development_label_auc"] - 0.5 for row in rows], dtype=np.float64
    )
    test_auc_effects = np.asarray(
        [row["test_label_auc"] - 0.5 for row in rows], dtype=np.float64
    )
    strong = [
        row for row in rows
        if abs(row["development_label_auc"] - 0.5) >= 0.05
    ]
    return {
        "feature_count": len(rows),
        "all_feature_direction_agreement_rate": float(
            np.mean([row["direction_agreement"] for row in rows])
        ),
        "strong_development_feature_count": len(strong),
        "strong_development_feature_direction_agreement_rate": (
            float(np.mean([row["direction_agreement"] for row in strong]))
            if strong else None
        ),
        "development_test_auc_effect_pearson": float(
            np.corrcoef(development_auc_effects, test_auc_effects)[0, 1]
        ),
        "features": sorted(
            rows, key=lambda row: row["absolute_auc_change"], reverse=True
        ),
    }


def label_probe(development_features, test_features, development_rows, test_rows, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    development_labels = np.asarray(
        [int(row["label"]) for row in development_rows], dtype=np.int64
    )
    test_labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    validation_aucs = []
    for train_indices, validation_indices in splitter.split(
        development_features, development_labels
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed),
        )
        model.fit(development_features[train_indices], development_labels[train_indices])
        probability = model.predict_proba(development_features[validation_indices])[:, 1]
        validation_aucs.append(
            float(roc_auc_score(development_labels[validation_indices], probability))
        )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed),
    )
    model.fit(development_features, development_labels)
    test_probability = model.predict_proba(test_features)[:, 1]
    return {
        "purpose": "fixed in-memory audit probe; not a production model",
        "hyperparameter_selection_performed": False,
        "development_repeated_cv_auc_mean": float(np.mean(validation_aucs)),
        "development_repeated_cv_auc_standard_deviation": float(np.std(validation_aucs)),
        "fixed_test_auc": float(roc_auc_score(test_labels, test_probability)),
        "fixed_test_accuracy_at_0_5": float(
            accuracy_score(test_labels, test_probability >= 0.5)
        ),
        "fixed_test_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(test_labels, test_probability >= 0.5)
        ),
    }


def domain_probe(development_features, test_features, development_rows, test_rows, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.concatenate(
        (np.zeros(len(development_rows), dtype=np.int64), np.ones(len(test_rows), dtype=np.int64))
    )
    sites = sorted(
        {str(row["site"]) for row in development_rows}.union(
            str(row["site"]) for row in test_rows
        )
    )
    site_position = {site: index for index, site in enumerate(sites)}
    all_rows = list(development_rows) + list(test_rows)
    composition = np.zeros((len(all_rows), len(sites) + 1), dtype=np.float64)
    for index, row in enumerate(all_rows):
        composition[index, site_position[str(row["site"])]] = 1.0
        composition[index, -1] = float(row["label"])
    structure = np.concatenate((development_features, test_features), axis=0)
    feature_sets = {
        "site_and_label_only": composition,
        "graph_structure_only": structure,
        "graph_structure_plus_site_and_label": np.concatenate(
            (structure, composition), axis=1
        ),
    }
    folds = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=seed
    )
    result = {}
    for name, features in feature_sets.items():
        aucs = []
        balanced_accuracies = []
        for train_indices, test_indices in folds.split(features, labels):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, class_weight="balanced", solver="liblinear",
                    max_iter=2000, random_state=seed,
                ),
            )
            model.fit(features[train_indices], labels[train_indices])
            probability = model.predict_proba(features[test_indices])[:, 1]
            aucs.append(float(roc_auc_score(labels[test_indices], probability)))
            balanced_accuracies.append(
                float(
                    balanced_accuracy_score(
                        labels[test_indices], probability >= 0.5
                    )
                )
            )
        result[name] = {
            "repeated_cv_roc_auc_mean": float(np.mean(aucs)),
            "repeated_cv_roc_auc_standard_deviation": float(np.std(aucs)),
            "repeated_cv_balanced_accuracy_at_0_5_mean": float(
                np.mean(balanced_accuracies)
            ),
            "repeated_cv_balanced_accuracy_at_0_5_standard_deviation": float(
                np.std(balanced_accuracies)
            ),
            "fold_count": len(aucs),
            "feature_dimension": int(features.shape[1]),
        }
    return result


def load_csv_logits(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "sample_key": str(row["sample_key"]),
                    "label": int(row["label"]),
                    "site": str(row["site"]),
                    "logit": float(row["final_logit"]),
                }
            )
    return rows


def logit_distribution(rows):
    logits = np.asarray([row["logit"] for row in rows], dtype=np.float64)
    output = {
        "sample_count": int(logits.size),
        "mean": float(np.mean(logits)),
        "standard_deviation": float(np.std(logits)),
        "positive_prediction_rate": float(np.mean(logits >= 0.0)),
    }
    for label in (0, 1):
        selected = np.asarray(
            [row["logit"] for row in rows if int(row["label"]) == label],
            dtype=np.float64,
        )
        output["class_{}".format(label)] = {
            "count": int(selected.size),
            "mean": float(np.mean(selected)),
            "standard_deviation": float(np.std(selected)),
        }
    return output


def model_output_audit(static_root, subgraph_root):
    output = {"subgraph": {"rotations": []}, "static": {}}
    for fold in range(4):
        validation = load_csv_logits(
            Path(subgraph_root) / "fold_{}".format(fold) / "validation_predictions.csv"
        )
        test = load_csv_logits(
            Path(subgraph_root) / "fold_{}".format(fold) / "test_predictions.csv"
        )
        output["subgraph"]["rotations"].append(
            {
                "rotation": fold,
                "validation": logit_distribution(validation),
                "test": logit_distribution(test),
            }
        )
    for stage in ("s1", "s2", "s3"):
        output["static"][stage] = {"rotations": []}
        for fold in range(4):
            entries = {}
            for split in ("validation", "test"):
                path = Path(static_root) / "fold_{}".format(fold) / stage / (
                    split + "_features.npz"
                )
                payload = np.load(str(path), allow_pickle=False)
                rows = [
                    {"logit": float(logit), "label": int(label)}
                    for logit, label in zip(
                        payload["background_logits"], payload["labels"]
                    )
                ]
                entries[split] = logit_distribution(rows)
            output["static"][stage]["rotations"].append(
                {"rotation": fold, **entries}
            )
    for branch in [output["subgraph"]] + [output["static"][s] for s in ("s1", "s2", "s3")]:
        branch["mean_shift"] = {
            "test_minus_validation_logit_mean": float(
                np.mean(
                    [
                        row["test"]["mean"] - row["validation"]["mean"]
                        for row in branch["rotations"]
                    ]
                )
            ),
            "test_minus_validation_positive_prediction_rate": float(
                np.mean(
                    [
                        row["test"]["positive_prediction_rate"]
                        - row["validation"]["positive_prediction_rate"]
                        for row in branch["rotations"]
                    ]
                )
            ),
        }
    return output


def render_markdown(report):
    composition = report["composition"]
    lines = [
        "# WMRC Development-Validation 与固定 Test 只读分布审计",
        "",
        "- 参数更新量：0",
        "- 划分修改：否",
        "- Test用于模型/融合选择：否",
        "- Development样本数：{}".format(report["cohorts"]["development_count"]),
        "- 固定Test样本数：{}".format(report["cohorts"]["test_count"]),
        "- Development/Test样本重叠：0",
        "",
        "## 组成差异",
        "",
        "| 组成 | 最大比例差 | Jensen–Shannon散度 |",
        "|---|---:|---:|",
    ]
    for name in ("label", "site", "site_label"):
        row = composition[name]
        lines.append(
            "| {} | {:.6f} | {:.6f} |".format(
                name,
                row["maximum_absolute_proportion_difference"],
                row["jensen_shannon_divergence"],
            )
        )
    lines.extend(("", "### Site × Label", "", "| 单元 | Development | Test | 比例差(Test−Dev) |", "|---|---:|---:|---:|"))
    first = composition["site_label"]["development"]
    second = composition["site_label"]["test"]
    for key in sorted(first):
        lines.append(
            "| {} | {} ({:.3f}) | {} ({:.3f}) | {:+.3f} |".format(
                key,
                first[key]["count"], first[key]["proportion"],
                second[key]["count"], second[key]["proportion"],
                second[key]["proportion"] - first[key]["proportion"],
            )
        )
    lines.extend(("", "## 图结构与谱特征差异（前15项）", "", "| 特征 | Dev均值 | Test均值 | SMD | KS | PSI |", "|---|---:|---:|---:|---:|---:|"))
    for row in report["continuous_feature_shift"][:15]:
        lines.append(
            "| {} | {:.6g} | {:.6g} | {:+.4f} | {:.4f} | {:.4f} |".format(
                row["feature"], row["development_mean"], row["test_mean"],
                row["standardized_mean_difference_test_minus_development"],
                row["ks_statistic"], row["population_stability_index"],
            )
        )
    lines.extend(("", "## Development/Test域分类探针", "", "| 输入 | 5折×10次 AUC | BA@0.5 | 维度 |", "|---|---:|---:|---:|"))
    for name, row in report["domain_probe"].items():
        lines.append(
            "| {} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {} |".format(
                name,
                row["repeated_cv_roc_auc_mean"],
                row["repeated_cv_roc_auc_standard_deviation"],
                row["repeated_cv_balanced_accuracy_at_0_5_mean"],
                row["repeated_cv_balanced_accuracy_at_0_5_standard_deviation"],
                row["feature_dimension"],
            )
        )
    association = report["label_association"]
    lines.extend(
        (
            "",
            "## 特征—类别关系稳定性",
            "",
            "- 全部特征类别效应方向一致率：{:.2%}".format(
                association["all_feature_direction_agreement_rate"]
            ),
            "- Development中 `|单变量AUC−0.5|≥0.05` 的特征数：{}".format(
                association["strong_development_feature_count"]
            ),
            "- 上述较强特征方向一致率：{}".format(
                "N/A" if association["strong_development_feature_direction_agreement_rate"] is None
                else "{:.2%}".format(
                    association["strong_development_feature_direction_agreement_rate"]
                )
            ),
            "- Development/Test单变量AUC效应Pearson：{:.6f}".format(
                association["development_test_auc_effect_pearson"]
            ),
            "",
            "| 变化最大的特征 | Dev类别AUC | Test类别AUC | Dev效应 | Test效应 | 方向一致 |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in association["features"][:12]:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:+.4f} | {:+.4f} | {} |".format(
                row["feature"], row["development_label_auc"], row["test_label_auc"],
                row["development_class_effect"], row["test_class_effect"],
                "是" if row["direction_agreement"] else "否",
            )
        )
    probe = report["fixed_structure_label_probe"]
    lines.extend(
        (
            "",
            "### 固定L2结构类别探针",
            "",
            "| Dev 5折×10次 AUC | Fixed-test AUC | Test ACC@0.5 | Test BA@0.5 |",
            "|---:|---:|---:|---:|",
            "| {:.6f} ± {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                probe["development_repeated_cv_auc_mean"],
                probe["development_repeated_cv_auc_standard_deviation"],
                probe["fixed_test_auc"], probe["fixed_test_accuracy_at_0_5"],
                probe["fixed_test_balanced_accuracy_at_0_5"],
            ),
            "",
            "> 该探针固定为train-only标准化与 `C=1` 的L2 Logistic Regression；不做超参数选择、不保存权重，仅用于诊断类别关系是否能从Development迁移到Test。",
        )
    )
    lines.extend(("", "## 模型Logit漂移", "", "| 分支 | Test−Validation logit均值变化 | Test−Validation阳性预测率变化 |", "|---|---:|---:|"))
    model = report["model_output_shift"]
    branches = [("frozen_subgraph", model["subgraph"])] + [
        (stage.upper(), model["static"][stage]) for stage in ("s1", "s2", "s3")
    ]
    for name, branch in branches:
        shift = branch["mean_shift"]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} |".format(
                name, shift["test_minus_validation_logit_mean"],
                shift["test_minus_validation_positive_prediction_rate"],
            )
        )
    flagged = [
        row for row in report["continuous_feature_shift"]
        if row["absolute_standardized_mean_difference"] >= 0.25
        or row["ks_statistic"] >= 0.20
    ]
    lines.extend(
        (
            "",
            "## 自动判读",
            "",
            "- 达到 `|SMD|≥0.25` 或 `KS≥0.20` 的结构/谱特征数：{} / {}。".format(
                len(flagged), len(report["continuous_feature_shift"])
            ),
            "- 域探针AUC越高于0.5，表示仅凭对应输入越容易区分Development与Test。",
            "- 该审计描述分布差异，不使用Test反向修改模型、权重或checkpoint。",
        )
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if len(args.fold_dir) != 4:
        raise ValueError("exactly four fold directories are required")
    fold_dirs = [path.resolve() for path in args.fold_dir]
    development_rows, test_rows = validate_cohorts(fold_dirs)
    cache = args.feature_cache.resolve()
    development_records = [
        build_global_static_record(
            args.global_root.resolve(), row, args.spectral_dim, cache,
            include_signed_profile=True,
        )
        for row in development_rows
    ]
    test_records = [
        build_global_static_record(
            args.global_root.resolve(), row, args.spectral_dim, cache,
            include_signed_profile=True,
        )
        for row in test_rows
    ]
    development_features = np.stack([graph_vector(record) for record in development_records])
    test_features = np.stack([graph_vector(record) for record in test_records])
    names = graph_feature_names(args.spectral_dim)
    if development_features.shape[1] != len(names):
        raise RuntimeError("graph audit feature-name mismatch")
    report = {
        "artifact_type": "mokse_background_distribution_shift_audit_v1",
        "read_only": True,
        "parameter_updates": 0,
        "fixed_test_used_for_model_or_fusion_selection": False,
        "cohorts": {
            "development_count": len(development_rows),
            "test_count": len(test_rows),
            "development_unique_count": len({row["sample_key"] for row in development_rows}),
            "test_unique_count": len({row["sample_key"] for row in test_rows}),
            "overlap_count": 0,
        },
        "composition": composition_audit(development_rows, test_rows),
        "continuous_feature_shift": continuous_shift(
            development_features, test_features, names
        ),
        "domain_probe": domain_probe(
            development_features, test_features, development_rows, test_rows, args.seed
        ),
        "label_association": label_association_audit(
            development_features,
            test_features,
            development_rows,
            test_rows,
            names,
        ),
        "fixed_structure_label_probe": label_probe(
            development_features,
            test_features,
            development_rows,
            test_rows,
            args.seed,
        ),
        "model_output_shift": model_output_audit(
            args.safe_static_root.resolve(),
            args.frozen_subgraph_prediction_root.resolve(),
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "audit.json", report)
    atomic_text(args.output_dir / "report.md", render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
