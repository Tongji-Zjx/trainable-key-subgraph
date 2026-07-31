"""Summarize immutable theory-guided upgrade artifacts (Stage 0 first)."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.theory.class_margin_diagnostics import (  # noqa: E402
    apply_standardizer,
    class_margin_metrics,
    component_margin_metrics,
    stratified_paired_bootstrap,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage0",), default="stage0")
    parser.add_argument("--fold-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _load_json(path):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_core(path):
    with np.load(str(Path(path).resolve()), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _fold_payload(path):
    path = Path(path).resolve()
    manifest = _load_json(path / "manifest.json")
    if manifest.get("artifact_type") != "svg_stage0_theory_diagnostics":
        raise ValueError("not a Stage-0 fold artifact")
    for name, expected in manifest.get("files", {}).items():
        if file_sha256(path / name) != expected:
            raise ValueError("Stage-0 fold file hash mismatch: {}".format(name))
    full = _load_core(path / "full_core_features.npz")
    hard = _load_core(path / "hard_core_features.npz")
    if not np.array_equal(full["sample_keys"], hard["sample_keys"]):
        raise ValueError("Stage-0 full/hard fold samples disagree")
    if not np.array_equal(full["labels"], hard["labels"]):
        raise ValueError("Stage-0 full/hard fold labels disagree")
    scaler = _load_json(path / "train_only_core_scaler.json")
    standardized_full = apply_standardizer(full["core"], scaler)
    standardized_hard = apply_standardizer(hard["core"], scaler)
    return {
        "path": path,
        "manifest": manifest,
        "sample_keys": full["sample_keys"],
        "sites": full["sites"],
        "labels": full["labels"].astype(np.int64),
        "full": full["core"].astype(np.float64),
        "hard": hard["core"].astype(np.float64),
        "standardized_full": standardized_full,
        "standardized_hard": standardized_hard,
    }


def _site_metrics(full, hard, labels, sites):
    output = {}
    for site in sorted(set(str(value) for value in sites)):
        indices = np.flatnonzero(np.asarray(sites, dtype=str) == site)
        current_labels = labels[indices]
        if set(current_labels.tolist()) != {0, 1}:
            output[site] = {
                "sample_count": int(indices.size),
                "eligible": False,
            }
            continue
        output[site] = {
            "sample_count": int(indices.size),
            "eligible": True,
            "metrics": class_margin_metrics(
                full[indices], hard[indices], current_labels
            ),
        }
    return output


def _markdown(payload):
    raw = payload["raw_primary"]
    standardized = payload["train_only_standardized_sensitivity"]
    intervals = payload["bootstrap"]["intervals"]
    lines = [
        "# SVG Stage 0 理论条件配对OOF汇总",
        "",
        "- 外折数：{}".format(payload["fold_count"]),
        "- OOF样本数：{}".format(payload["sample_count"]),
        "- 每个样本仅出现一次：{}".format(
            "是" if payload["checks"]["unique_oof_samples"] else "否"
        ),
        "- 主定义：未标准化18维Exact谱–GW core上的欧氏Wasserstein-1",
        "- Bootstrap：{}次".format(payload["bootstrap"]["repeats"]),
        "",
        "| 指标 | Pooled raw | 95% CI | Fold-standardized敏感性 |",
        "|---|---:|---:|---:|",
    ]
    names = (
        ("delta_full", "完整图类别间隔"),
        ("eta_0_pair", "类别0配对半径"),
        ("eta_1_pair", "类别1配对半径"),
        ("eta_0_ot", "类别0 OT半径"),
        ("eta_1_ot", "类别1 OT半径"),
        ("lower_bound_pair", "配对理论下界"),
        ("lower_bound_ot", "OT理论下界"),
        ("delta_hard", "硬图类别间隔"),
    )
    for name, label in names:
        interval = intervals[name]
        lines.append(
            "| {} | {:.6f} | [{:.6f}, {:.6f}] | {:.6f} |".format(
                label,
                raw[name],
                interval["lower_95"],
                interval["upper_95"],
                standardized[name],
            )
        )
    pair_positive = intervals["lower_bound_pair"]["lower_95"] > 0.0
    ot_positive = intervals["lower_bound_ot"]["lower_95"] > 0.0
    if pair_positive:
        conclusion = "当前数据支持保守配对充分条件。"
    elif raw["lower_bound_pair"] > 0.0:
        conclusion = "保守配对下界点估计为正，但置信区间跨0，仅支持非平凡趋势。"
    else:
        conclusion = "当前数据未验证保守配对理论充分条件成立。"
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- {}".format(conclusion),
            "- OT下界95% CI下界为正：{}。".format(
                "是" if ot_positive else "否"
            ),
            "- OT半径不大于配对上界：{}。".format(
                "通过"
                if raw["checks"]["eta_ot_not_above_pair"]
                else "失败"
            ),
            "- 硬图类别间隔不低于OT下界：{}。".format(
                "通过"
                if raw["checks"]["hard_margin_not_below_ot_lower_bound"]
                else "失败"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    output_dir = Path(args.output_dir).resolve()
    outputs = (output_dir / "pooled_metrics.json", output_dir / "pooled_report.md")
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Stage-0 pooled outputs already exist")
    folds = [_fold_payload(path) for path in args.fold_dirs]
    fold_ids = [int(item["manifest"]["fold"]) for item in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Stage-0 pooled folds are duplicated")
    assignment_hashes = {
        item["manifest"]["test_provenance"]["fold_assignments_sha256"]
        for item in folds
    }
    schema_hashes = {
        item["manifest"]["test_provenance"]["feature_schema_sha256"]
        for item in folds
    }
    if len(assignment_hashes) != 1 or len(schema_hashes) != 1:
        raise ValueError("Stage-0 folds have incompatible provenance")
    sample_keys = np.concatenate([item["sample_keys"] for item in folds])
    if len(set(sample_keys.tolist())) != int(sample_keys.size):
        raise ValueError("Stage-0 pooled OOF samples are not unique")
    sites = np.concatenate([item["sites"] for item in folds])
    labels = np.concatenate([item["labels"] for item in folds])
    full = np.concatenate([item["full"] for item in folds], axis=0)
    hard = np.concatenate([item["hard"] for item in folds], axis=0)
    standardized_full = np.concatenate(
        [item["standardized_full"] for item in folds], axis=0
    )
    standardized_hard = np.concatenate(
        [item["standardized_hard"] for item in folds], axis=0
    )
    raw = class_margin_metrics(full, hard, labels)
    standardized = class_margin_metrics(
        standardized_full, standardized_hard, labels
    )
    fold_metrics = {
        str(item["manifest"]["fold"]): class_margin_metrics(
            item["full"], item["hard"], item["labels"]
        )
        for item in folds
    }
    print(
        "START pooled exact bootstrap repeats={}".format(
            args.bootstrap_repeats
        ),
        flush=True,
    )
    bootstrap = stratified_paired_bootstrap(
        full,
        hard,
        labels,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    print("FINISH pooled exact bootstrap", flush=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "svg_stage0_theory_diagnostics_pooled_oof",
        "fold_count": len(folds),
        "folds": sorted(fold_ids),
        "sample_count": int(sample_keys.size),
        "class_counts": {
            str(value): int((labels == value).sum()) for value in (0, 1)
        },
        "primary_definition": {
            "feature": "raw_exact_sgw_core_18d",
            "ground_metric": "euclidean",
        },
        "raw_primary": raw,
        "train_only_standardized_sensitivity": standardized,
        "component_metrics": component_margin_metrics(full, hard, labels),
        "fold_metrics": fold_metrics,
        "site_metrics": _site_metrics(full, hard, labels, sites),
        "bootstrap": bootstrap,
        "provenance": {
            "fold_assignments_sha256": next(iter(assignment_hashes)),
            "feature_schema_sha256": next(iter(schema_hashes)),
            "fold_manifest_sha256": {
                str(item["manifest"]["fold"]): file_sha256(
                    item["path"] / "manifest.json"
                )
                for item in folds
            },
        },
        "checks": {
            "unique_oof_samples": True,
            "compatible_fold_assignments": True,
            "compatible_feature_schema": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(outputs[0], payload)
    outputs[1].write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "pooled_metrics": str(outputs[0]),
                "pooled_report": str(outputs[1]),
                "raw_primary": raw,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
