#!/usr/bin/env python3
"""Summarize E0--E4 MoKSE global-background four-fold experiments."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.tge.trainer import classification_metrics, site_stratified_roc_auc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, action="append", required=True)
    parser.add_argument("--e0-xgb", type=Path)
    parser.add_argument("--e4-xgb", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-weight", type=float, default=0.5)
    return parser.parse_args()


def score(labels, probabilities, sites):
    result = classification_metrics(labels, probabilities, 0.5)
    result["site_stratified_roc_auc"] = site_stratified_roc_auc(
        labels.tolist(), probabilities.tolist(), sites.tolist()
    )
    return result


def load_fold(path, probability_weight):
    payload = np.load(str(path / "fusion" / "test_features.npz"), allow_pickle=False)
    labels = payload["labels"].astype(np.int64)
    sites = payload["sites"].astype(str)
    evolution = 1.0 / (1.0 + np.exp(-np.clip(payload["evolution_logits"], -50.0, 50.0)))
    background = 1.0 / (1.0 + np.exp(-np.clip(payload["background_logits"], -50.0, 50.0)))
    fusion = 1.0 / (1.0 + np.exp(-np.clip(payload["fused_logits"], -50.0, 50.0)))
    probability = probability_weight * evolution + (1.0 - probability_weight) * background
    return {
        "E0_neural_evolution": score(labels, evolution, sites),
        "E1_global_gcn_only": score(labels, background, sites),
        "E2_equal_probability_fusion": score(labels, probability, sites),
        "E3_controlled_logit_fusion": score(labels, fusion, sites),
    }


def mean_metrics(rows):
    keys = ("roc_auc", "accuracy", "balanced_accuracy", "auprc", "f1", "site_stratified_roc_auc")
    return {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in keys if any(row.get(key) is not None for row in rows)
    }


def load_xgb(path):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "selection_rule": payload["selection_rule"],
        "test_used_for_parameter_selection": payload["test_used_for_parameter_selection"],
        "unbiased_generalization_estimate": payload["unbiased_generalization_estimate"],
        "candidate": payload["best"]["candidate"],
        "folds": [row["metrics"]["test"] for row in payload["best"]["folds"]],
        "mean": payload["best"]["summary"],
    }


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if len(args.fold_dir) != 4:
        raise ValueError("exactly four fold directories are required")
    if not 0.0 <= args.probability_weight <= 1.0:
        raise ValueError("probability weight must be in [0, 1]")
    fold_rows = [load_fold(path.resolve(), args.probability_weight) for path in args.fold_dir]
    experiments = {}
    for name in fold_rows[0]:
        rows = [fold[name] for fold in fold_rows]
        experiments[name] = {"folds": rows, "mean": mean_metrics(rows)}
    e0 = load_xgb(args.e0_xgb.resolve()) if args.e0_xgb else None
    e4 = load_xgb(args.e4_xgb.resolve()) if args.e4_xgb else None
    if e0 is not None:
        experiments["E0_formal_evolution_plus_xgb"] = e0
    if e4 is not None:
        experiments["E4_controlled_fusion_plus_xgb"] = e4
    report = {
        "artifact_type": "mokse_background_fourfold_summary_v1",
        "fold_directories": [str(path.resolve()) for path in args.fold_dir],
        "decision_threshold": 0.5,
        "probability_fusion_evolution_weight": args.probability_weight,
        "experiments": experiments,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output_dir / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = ["# MoKSE-Net 全局静态背景分支四折汇总", "", "阈值固定为 0.5。E2 使用等权概率融合。", "", "| 实验 | Mean Test AUROC | Mean Test ACC | Mean Test BA | Mean Test AUPRC |", "|---|---:|---:|---:|---:|"]
    for name, result in experiments.items():
        mean = result["mean"]
        def value(key):
            candidate = mean.get(key)
            return "N/A" if candidate is None else "{:.6f}".format(float(candidate))
        lines.append("| {} | {} | {} | {} | {} |".format(
            name, value("mean_test_roc_auc" if "xgb" in name else "roc_auc"),
            value("mean_test_accuracy" if "xgb" in name else "accuracy"),
            value("mean_test_balanced_accuracy" if "xgb" in name else "balanced_accuracy"),
            value("mean_test_auprc" if "xgb" in name else "auprc"),
        ))
    if e0 is not None or e4 is not None:
        lines.extend(["", "> 含 XGB 的 E0/E4 依用户要求按四折 mean test AUROC 选择共享参数，属于 test-guided 探索性上限，不是无偏泛化估计。"])
    atomic_write(args.output_dir / "summary.md", "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
