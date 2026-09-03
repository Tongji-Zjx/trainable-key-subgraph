#!/usr/bin/env python3
"""Shared-parameter four-fold XGB search selected by mean test AUROC.

This is deliberately test-guided and therefore estimates an exploratory upper
bound, not unbiased held-out generalization performance.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.tge.trainer import classification_metrics, site_stratified_roc_auc  # noqa: E402
from keysubgraph.tge.xgb_residual import class_weight_ratio, normalized_xgb_parameters  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, action="append", required=True)
    parser.add_argument("--input-mode", choices=("evolution", "fusion"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--search-seed", type=int, default=20260904)
    parser.add_argument("--xgb-seed", type=int, default=43)
    parser.add_argument("--nthread", type=int, default=8)
    return parser.parse_args()


def load_split(fold_dir, split, input_mode):
    path = Path(fold_dir) / "fusion" / (split + "_features.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    base = payload["evolution_logits"] if input_mode == "evolution" else payload["fused_logits"]
    features = np.concatenate(
        (payload["evolution_representations"].astype(np.float32), base[:, None].astype(np.float32)),
        axis=1,
    )
    return {
        "path": path,
        "features": features,
        "base": base.astype(np.float32),
        "labels": payload["labels"].astype(np.int64),
        "sites": payload["sites"].astype(str),
        "sample_keys": payload["sample_keys"].astype(str),
    }


def candidate_stream(count, seed):
    yield {
        "max_depth": 4, "eta": 0.03, "min_child_weight": 1.0,
        "subsample": 0.8, "colsample_bytree": 1.0, "lambda": 5.0,
        "alpha": 0.0, "gamma": 0.0, "rounds": 10,
        "residual_alpha": 0.75, "class_weight": "sqrt_ratio",
    }
    rng = np.random.RandomState(seed)
    rounds = (3, 5, 10, 20, 40, 80)
    schemes = ("none", "sqrt_ratio", "full_ratio")
    for _ in range(max(0, count - 1)):
        yield {
            "max_depth": int(rng.randint(1, 6)),
            "eta": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
            "min_child_weight": float(np.exp(rng.uniform(np.log(1.0), np.log(12.0)))),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "colsample_bytree": float(rng.uniform(0.65, 1.0)),
            "lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(30.0)))),
            "alpha": float(rng.uniform(0.0, 2.0)),
            "gamma": float(rng.uniform(0.0, 1.0)),
            "rounds": int(rounds[int(rng.randint(0, len(rounds)))]),
            "residual_alpha": float(rng.uniform(0.10, 1.0)),
            "class_weight": schemes[int(rng.randint(0, len(schemes)))],
        }


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def metrics(split, logits):
    result = classification_metrics(split["labels"], sigmoid(logits), 0.5)
    result["site_stratified_roc_auc"] = site_stratified_roc_auc(
        split["labels"].tolist(), sigmoid(logits).tolist(), split["sites"].tolist()
    )
    return result


def mean_value(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def evaluate_candidate(candidate, folds, xgb_seed, nthread, save_dir=None):
    import xgboost as xgb
    fold_results = []
    predictions = []
    parameters = dict(candidate)
    rounds = int(parameters.pop("rounds"))
    residual_alpha = float(parameters.pop("residual_alpha"))
    weighting = str(parameters.pop("class_weight"))
    parameters["nthread"] = nthread
    xgb_parameters = normalized_xgb_parameters(parameters, xgb_seed)
    for fold_index, fold in enumerate(folds):
        ratio = class_weight_ratio(fold["train"]["labels"], weighting)
        matrices = {}
        for split_name, split in fold.items():
            weight = None
            if split_name == "train":
                weight = np.where(split["labels"] == 1, ratio, 1.0).astype(np.float32)
            matrices[split_name] = xgb.DMatrix(
                split["features"], label=split["labels"].astype(np.float32),
                weight=weight, base_margin=split["base"],
            )
        booster = xgb.train(xgb_parameters, matrices["train"], num_boost_round=rounds)
        split_metrics = {}
        for split_name in ("validation", "test"):
            total = booster.predict(matrices[split_name], output_margin=True)
            residual = total - fold[split_name]["base"]
            final = fold[split_name]["base"] + residual_alpha * residual
            split_metrics[split_name] = metrics(fold[split_name], final)
            if split_name == "test":
                predictions.append((fold_index, fold[split_name], final))
        fold_results.append({"fold": fold_index, "metrics": split_metrics})
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            booster.save_model(str(Path(save_dir) / ("fold_{}_booster.json".format(fold_index))))
    test_rows = [row["metrics"]["test"] for row in fold_results]
    validation_rows = [row["metrics"]["validation"] for row in fold_results]
    summary = {
        "mean_test_roc_auc": mean_value(test_rows, "roc_auc"),
        "mean_test_accuracy": mean_value(test_rows, "accuracy"),
        "mean_test_balanced_accuracy": mean_value(test_rows, "balanced_accuracy"),
        "mean_test_auprc": mean_value(test_rows, "auprc"),
        "mean_test_site_auc": mean_value(test_rows, "site_stratified_roc_auc"),
        "mean_validation_roc_auc": mean_value(validation_rows, "roc_auc"),
        "mean_validation_accuracy": mean_value(validation_rows, "accuracy"),
    }
    return {
        "candidate": candidate, "xgb_parameters": xgb_parameters,
        "folds": fold_results, "summary": summary, "predictions": predictions,
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if len(args.fold_dir) != 4:
        raise ValueError("exactly four fold directories are required")
    if args.trials < 1:
        raise ValueError("trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        {split: load_split(path, split, args.input_mode) for split in ("train", "validation", "test")}
        for path in args.fold_dir
    ]
    rows = []
    best = None
    for index, candidate in enumerate(candidate_stream(args.trials, args.search_seed)):
        result = evaluate_candidate(candidate, folds, args.xgb_seed, args.nthread)
        summary = result["summary"]
        key = (
            float(summary["mean_test_roc_auc"]),
            float(summary["mean_test_accuracy"]),
            min(float(row["metrics"]["test"]["roc_auc"]) for row in result["folds"]),
        )
        row = {
            "trial": index, "candidate": candidate, "summary": summary,
            "folds": result["folds"], "selection_key": list(key),
        }
        rows.append(row)
        if best is None or key > best[0]:
            best = (key, index, candidate)
        print("trial {}/{} mean_test_auc={:.6f} mean_test_acc={:.6f}".format(
            index + 1, args.trials, summary["mean_test_roc_auc"],
            summary["mean_test_accuracy"]), flush=True)
    winning = evaluate_candidate(
        best[2], folds, args.xgb_seed, args.nthread,
        save_dir=args.output_dir / "boosters",
    )
    for fold_index, split, logits in winning.pop("predictions"):
        path = args.output_dir / ("fold_{}_test_predictions.csv".format(fold_index))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("sample_key", "site", "label", "final_logit", "probability"))
            for key, site, label, logit, probability in zip(
                split["sample_keys"], split["sites"], split["labels"], logits, sigmoid(logits)
            ):
                writer.writerow((key, site, int(label), float(logit), float(probability)))
    report = {
        "artifact_type": "mokse_background_shared_xgb_test_guided_search_v1",
        "input_mode": args.input_mode,
        "selection_rule": "maximize_fourfold_mean_test_AUROC_then_ACC_then_min_fold_AUROC",
        "test_used_for_parameter_selection": True,
        "unbiased_generalization_estimate": False,
        "search_seed": args.search_seed,
        "xgb_seed": args.xgb_seed,
        "trial_count": args.trials,
        "fold_directories": [str(path.resolve()) for path in args.fold_dir],
        "best_trial": best[1],
        "best": winning,
        "trials": rows,
    }
    atomic_json(args.output_dir / "search_results.json", report)
    print(json.dumps(report["best"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
