#!/usr/bin/env python3
"""Fit BG-Safe fusion on four development-OOF rotations, then test once."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.safe_fusion import (  # noqa: E402
    SafeFusionConfig,
    apply_safe_fusion,
    score_logits,
    select_safe_fusion,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir", action="append", type=Path, required=True,
        help="four selected static-stage directories containing *_features.npz",
    )
    parser.add_argument(
        "--subgraph-prediction-dir", action="append", type=Path, default=[],
        help="optional matching directories with validation/test_predictions.csv final_logit",
    )
    parser.add_argument("--dataset", choices=("adhd", "wmrc"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument("--stability-penalty", type=float, default=0.50)
    parser.add_argument("--near-best-tolerance", type=float, default=0.002)
    parser.add_argument("--minimum-mean-auc-gain", type=float, default=0.005)
    parser.add_argument("--maximum-worst-rotation-drop", type=float, default=0.01)
    parser.add_argument("--maximum-site-auc-drop", type=float, default=0.01)
    parser.add_argument("--minimum-nondecreasing-rotations", type=int, default=3)
    return parser.parse_args()


def load_npz(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    required = ("sample_keys", "sites", "labels", "background_logits")
    if any(name not in payload for name in required):
        raise ValueError("background feature export is incomplete: {}".format(path))
    return {
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "background_logits": payload["background_logits"].astype(np.float64),
        "embedded_subgraph_logits": payload["evolution_logits"].astype(np.float64),
    }


def load_external_logits(directory, split, reference):
    path = Path(directory) / (split + "_predictions.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    by_key = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("sample_key", ""))
            if not key or key in by_key:
                raise ValueError("external subgraph prediction keys are invalid")
            if "final_logit" not in row:
                raise ValueError("external predictions do not contain final_logit")
            by_key[key] = row
    expected = reference["sample_keys"].tolist()
    if set(by_key) != set(expected):
        raise ValueError("external subgraph predictions do not match background samples")
    logits = []
    for index, key in enumerate(expected):
        row = by_key[key]
        if row.get("label") not in (None, "") and int(row["label"]) != int(
            reference["labels"][index]
        ):
            raise ValueError("external subgraph label mismatch")
        if row.get("site") not in (None, "") and str(row["site"]) != str(
            reference["sites"][index]
        ):
            raise ValueError("external subgraph site mismatch")
        logits.append(float(row["final_logit"]))
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("external subgraph logits contain non-finite values")
    return values, str(path.resolve())


def load_fold(fold_dir, split, subgraph_dir=None):
    background_path = Path(fold_dir) / (split + "_features.npz")
    background = load_npz(background_path)
    if subgraph_dir is None:
        subgraph = background["embedded_subgraph_logits"]
        source = str(background_path.resolve())
        source_kind = "embedded_evolution_logits"
    else:
        subgraph, source = load_external_logits(subgraph_dir, split, background)
        source_kind = "external_final_subgraph_logits"
    return {
        "sample_keys": background["sample_keys"],
        "sites": background["sites"],
        "labels": background["labels"],
        "subgraph_logits": subgraph,
        "background_logits": background["background_logits"],
        "subgraph_source": source,
        "subgraph_source_kind": source_kind,
        "background_source": str(background_path.resolve()),
        "subgraph_source_sha256": file_sha256(source),
        "background_source_sha256": file_sha256(
            Path(fold_dir) / (split + "_features.npz")
        ),
    }


def reorder_fold(fold, sample_order):
    position = {key: index for index, key in enumerate(fold["sample_keys"].tolist())}
    if set(position) != set(sample_order):
        raise ValueError("fixed test cohorts differ across rotations")
    indices = np.asarray([position[key] for key in sample_order], dtype=np.int64)
    output = dict(fold)
    for name in ("sample_keys", "sites", "labels", "subgraph_logits", "background_logits"):
        output[name] = fold[name][indices]
    return output


def mean_metrics(rows):
    names = (
        "roc_auc", "auprc", "accuracy", "balanced_accuracy", "sensitivity",
        "specificity", "f1", "site_stratified_roc_auc",
    )
    output = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        output[name] = float(np.mean(values)) if values else None
        output[name + "_std"] = float(np.std(values)) if values else None
    return output


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


def write_predictions(path, test, subgraph, background, fused):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "sample_key", "site", "label", "subgraph_logit",
                "background_logit", "final_logit", "probability", "prediction",
            )
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(fused, -50.0, 50.0)))
        for index, key in enumerate(test["sample_keys"]):
            writer.writerow(
                (
                    key,
                    test["sites"][index],
                    int(test["labels"][index]),
                    float(subgraph[index]),
                    float(background[index]),
                    float(fused[index]),
                    float(probability[index]),
                    int(probability[index] >= 0.5),
                )
            )


def main():
    args = parse_args()
    if len(args.fold_dir) != 4:
        raise ValueError("exactly four validation rotations are required")
    if args.subgraph_prediction_dir and len(args.subgraph_prediction_dir) != 4:
        raise ValueError("external subgraph prediction directories must match four rotations")
    subgraph_dirs = (
        args.subgraph_prediction_dir if args.subgraph_prediction_dir else [None] * 4
    )
    # Selection phase deliberately loads validation only.  Fixed test is not
    # opened until the shared decision has been frozen below.
    validation_folds = [
        load_fold(fold_dir, "validation", subgraph_dir)
        for fold_dir, subgraph_dir in zip(args.fold_dir, subgraph_dirs)
    ]
    config = SafeFusionConfig(
        dataset=args.dataset,
        weight_grid_step=args.weight_grid_step,
        stability_penalty=args.stability_penalty,
        near_best_tolerance=args.near_best_tolerance,
        minimum_mean_auc_gain=args.minimum_mean_auc_gain,
        maximum_worst_rotation_drop=args.maximum_worst_rotation_drop,
        maximum_site_auc_drop=args.maximum_site_auc_drop,
        minimum_non_decreasing_rotations=args.minimum_nondecreasing_rotations,
    )
    selection = select_safe_fusion(validation_folds, config)

    test_folds = [
        load_fold(fold_dir, "test", subgraph_dir)
        for fold_dir, subgraph_dir in zip(args.fold_dir, subgraph_dirs)
    ]
    sample_order = test_folds[0]["sample_keys"].tolist()
    test_folds = [reorder_fold(fold, sample_order) for fold in test_folds]
    reference = test_folds[0]
    for fold in test_folds[1:]:
        if not np.array_equal(fold["labels"], reference["labels"]):
            raise ValueError("fixed test labels differ across rotations")
        if not np.array_equal(fold["sites"], reference["sites"]):
            raise ValueError("fixed test sites differ across rotations")

    rotation_rows = []
    subgraph_logits = []
    background_logits = []
    final_logits = []
    for rotation, fold in enumerate(test_folds):
        final = apply_safe_fusion(
            selection, fold["subgraph_logits"], fold["background_logits"]
        )
        subgraph_metrics = score_logits(
            fold["labels"], fold["sites"], fold["subgraph_logits"]
        )
        background_metrics = score_logits(
            fold["labels"], fold["sites"], fold["background_logits"]
        )
        final_metrics = score_logits(fold["labels"], fold["sites"], final)
        rotation_rows.append(
            {
                "rotation": rotation,
                "subgraph": subgraph_metrics,
                "background": background_metrics,
                "final": final_metrics,
                "auc_difference_vs_subgraph": (
                    float(final_metrics["roc_auc"])
                    - float(subgraph_metrics["roc_auc"])
                ),
            }
        )
        subgraph_logits.append(fold["subgraph_logits"])
        background_logits.append(fold["background_logits"])
        final_logits.append(final)
    subgraph_ensemble = np.mean(np.stack(subgraph_logits), axis=0)
    background_ensemble = np.mean(np.stack(background_logits), axis=0)
    final_ensemble = np.mean(np.stack(final_logits), axis=0)
    ensemble = {
        "subgraph": score_logits(
            reference["labels"], reference["sites"], subgraph_ensemble
        ),
        "background": score_logits(
            reference["labels"], reference["sites"], background_ensemble
        ),
        "final": score_logits(
            reference["labels"], reference["sites"], final_ensemble
        ),
    }
    report = {
        "artifact_type": "mokse_background_safe_fixed_test_evaluation_v1",
        "dataset": args.dataset,
        "selection": selection,
        "fixed_test_sample_count": int(reference["labels"].size),
        "fixed_test_is_shared_across_rotations": True,
        "mean_rotation_metrics": {
            "subgraph": mean_metrics([row["subgraph"] for row in rotation_rows]),
            "background": mean_metrics([row["background"] for row in rotation_rows]),
            "final": mean_metrics([row["final"] for row in rotation_rows]),
        },
        "ensemble_metrics": ensemble,
        "rotations": rotation_rows,
        "fixed_test_used_for_selection": False,
        "decision_threshold": 0.5,
        "development_subgraph_sources": [
            fold["subgraph_source"] for fold in validation_folds
        ],
        "development_subgraph_source_kinds": [
            fold["subgraph_source_kind"] for fold in validation_folds
        ],
        "development_subgraph_source_sha256": [
            fold["subgraph_source_sha256"] for fold in validation_folds
        ],
        "development_background_sources": [
            fold["background_source"] for fold in validation_folds
        ],
        "development_background_source_sha256": [
            fold["background_source_sha256"] for fold in validation_folds
        ],
        "subgraph_sources": [fold["subgraph_source"] for fold in test_folds],
        "subgraph_source_kinds": [
            fold["subgraph_source_kind"] for fold in test_folds
        ],
        "background_sources": [fold["background_source"] for fold in test_folds],
        "subgraph_source_sha256": [
            fold["subgraph_source_sha256"] for fold in test_folds
        ],
        "background_source_sha256": [
            fold["background_source_sha256"] for fold in test_folds
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "evaluation.json", report)
    write_predictions(
        args.output_dir / "fixed_test_ensemble_predictions.csv",
        reference,
        subgraph_ensemble,
        background_ensemble,
        final_ensemble,
    )
    lines = [
        "# MoKSE-Net-BG-Safe 固定测试集评估",
        "",
        "- 数据集：`{}`".format(args.dataset),
        "- Development validation rotations：4",
        "- 融合来源：`{}`".format(selection["selected_source"]),
        "- 子图权重：{:.6f}".format(selection["selected_subgraph_weight"]),
        "- 固定test参与选择：否",
        "- 分类阈值：0.5",
        "",
        "| 路径 | Mean rotation AUROC | Mean ACC | Ensemble AUROC | Ensemble ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("subgraph", "background", "final"):
        mean = report["mean_rotation_metrics"][name]
        combined = ensemble[name]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                name,
                float(mean["roc_auc"]),
                float(mean["accuracy"]),
                float(combined["roc_auc"]),
                float(combined["accuracy"]),
            )
        )
    if selection["fallback_reason"]:
        lines.extend(("", "> 回退原因：{}".format(selection["fallback_reason"])))
    atomic_text(args.output_dir / "summary.md", "\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
