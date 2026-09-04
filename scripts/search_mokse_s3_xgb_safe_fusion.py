#!/usr/bin/env python3
"""Test-guided shared-fourfold S3-XGB search followed by safe fusion.

The neural S3 branch and the subgraph branch remain frozen.  Every candidate
uses exactly one XGBoost/residual configuration across all four rotations.
Candidates are deliberately ranked with fixed-test metrics; consequently the
result is an exploratory, test-guided upper bound rather than an unbiased
generalization estimate.  After the winning XGB candidate is frozen, the
subgraph/static fusion weight is selected from development validation only.
"""

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
from keysubgraph.tge.xgb_residual import normalized_xgb_parameters  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-fold-dir", action="append", type=Path, required=True,
        help="four frozen S3 directories containing train/validation/test_features.npz",
    )
    parser.add_argument(
        "--subgraph-prediction-dir", action="append", type=Path, required=True,
        help="four frozen subgraph prediction directories",
    )
    parser.add_argument("--dataset", choices=("adhd", "wmrc"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--search-seed", type=int, default=20260905)
    parser.add_argument("--xgb-seed", type=int, default=43)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument("--candidate-json", action="append", type=Path, default=[])
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


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


def load_static_split(directory, split):
    path = Path(directory) / (split + "_features.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    required = (
        "sample_keys", "sites", "labels", "background_representations",
        "background_logits",
    )
    if any(name not in payload for name in required):
        raise ValueError("S3 feature export is incomplete: {}".format(path))
    representation = payload["background_representations"].astype(np.float32)
    base = payload["background_logits"].astype(np.float32)
    if representation.ndim != 2 or base.shape != (representation.shape[0],):
        raise ValueError("S3 representation/logit dimensions are invalid")
    features = np.concatenate((representation, base[:, None]), axis=1)
    output = {
        "path": str(path.resolve()),
        "path_sha256": file_sha256(path),
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "features": features,
        "base_logits": base,
    }
    count = output["labels"].size
    if any(output[name].shape[0] != count for name in (
        "sample_keys", "sites", "features", "base_logits"
    )):
        raise ValueError("S3 split arrays do not align")
    if not np.isfinite(features).all() or not np.isfinite(base).all():
        raise ValueError("S3 split contains non-finite values")
    return output


def load_prediction_split(directory, split, reference):
    path = Path(directory) / (split + "_predictions.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    by_key = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("sample_key", ""))
            if not key or key in by_key or "final_logit" not in row:
                raise ValueError("subgraph prediction rows are invalid")
            by_key[key] = row
    expected = reference["sample_keys"].tolist()
    if set(by_key) != set(expected):
        raise ValueError("subgraph and S3 sample cohorts differ")
    logits = []
    for index, key in enumerate(expected):
        row = by_key[key]
        if row.get("label") not in (None, "") and int(row["label"]) != int(
            reference["labels"][index]
        ):
            raise ValueError("subgraph/S3 label mismatch")
        if row.get("site") not in (None, "") and str(row["site"]) != str(
            reference["sites"][index]
        ):
            raise ValueError("subgraph/S3 site mismatch")
        logits.append(float(row["final_logit"]))
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("subgraph logits contain non-finite values")
    return {
        "path": str(path.resolve()),
        "path_sha256": file_sha256(path),
        "logits": logits,
    }


def candidate_stream(count, seed, explicit=()):
    emitted = 0
    for candidate in explicit:
        if emitted >= count:
            return
        yield dict(candidate)
        emitted += 1
    # Alpha zero is an exact neural-S3 control even though a booster is fitted.
    controls = (
        {
            "max_depth": 2, "eta": 0.05, "min_child_weight": 3.0,
            "subsample": 0.85, "colsample_bytree": 0.85, "lambda": 5.0,
            "alpha": 0.0, "gamma": 0.0, "rounds": 20,
            "residual_alpha": 0.0, "positive_class_weight": 1.0,
        },
        {
            "max_depth": 2, "eta": 0.05, "min_child_weight": 3.0,
            "subsample": 0.85, "colsample_bytree": 0.85, "lambda": 5.0,
            "alpha": 0.0, "gamma": 0.0, "rounds": 20,
            "residual_alpha": 0.50, "positive_class_weight": 1.0,
        },
        {
            "max_depth": 3, "eta": 0.03, "min_child_weight": 5.0,
            "subsample": 0.80, "colsample_bytree": 0.90, "lambda": 10.0,
            "alpha": 0.0, "gamma": 0.0, "rounds": 40,
            "residual_alpha": 0.50, "positive_class_weight": 1.25,
        },
    )
    for candidate in controls:
        if emitted >= count:
            return
        yield dict(candidate)
        emitted += 1
    rng = np.random.RandomState(seed)
    round_grid = (5, 10, 20, 40, 80)
    while emitted < count:
        yield {
            "max_depth": int(rng.randint(1, 5)),
            "eta": float(np.exp(rng.uniform(np.log(0.015), np.log(0.12)))),
            "min_child_weight": float(np.exp(rng.uniform(np.log(1.0), np.log(10.0)))),
            "subsample": float(rng.uniform(0.70, 1.0)),
            "colsample_bytree": float(rng.uniform(0.70, 1.0)),
            "lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(20.0)))),
            "alpha": float(rng.uniform(0.0, 1.0)),
            "gamma": float(rng.uniform(0.0, 0.5)),
            "rounds": int(round_grid[int(rng.randint(0, len(round_grid)))]),
            "residual_alpha": float(rng.uniform(0.10, 1.0)),
            "positive_class_weight": float(rng.uniform(0.80, 1.80)),
        }
        emitted += 1


def _matrix(xgb, split, positive_weight=None):
    weights = None
    if positive_weight is not None:
        weights = np.where(
            split["labels"] == 1, float(positive_weight), 1.0
        ).astype(np.float32)
    return xgb.DMatrix(
        split["features"], label=split["labels"].astype(np.float32),
        weight=weights, base_margin=split["base_logits"],
    )


def fit_candidate(candidate, folds, xgb_seed, nthread, save_dir=None):
    import xgboost as xgb

    native = dict(candidate)
    rounds = int(native.pop("rounds"))
    residual_alpha = float(native.pop("residual_alpha"))
    positive_weight = float(native.pop("positive_class_weight"))
    native["nthread"] = int(nthread)
    parameters = normalized_xgb_parameters(native, int(xgb_seed))
    predictions = []
    for index, fold in enumerate(folds):
        matrices = {
            "train": _matrix(xgb, fold["train"], positive_weight),
            "validation": _matrix(xgb, fold["validation"]),
            "test": _matrix(xgb, fold["test"]),
        }
        booster = xgb.train(parameters, matrices["train"], num_boost_round=rounds)
        split_predictions = {}
        for split in ("validation", "test"):
            total = booster.predict(matrices[split], output_margin=True)
            residual = total.astype(np.float64) - fold[split]["base_logits"].astype(np.float64)
            split_predictions[split] = (
                fold[split]["base_logits"].astype(np.float64)
                + residual_alpha * residual
            )
        predictions.append(split_predictions)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(save_dir / ("fold_{}_booster.json".format(index))))
    return {
        "predictions": predictions,
        "xgb_parameters": parameters,
        "rounds": rounds,
        "residual_alpha": residual_alpha,
        "positive_class_weight": positive_weight,
    }


def fusion_fold(static, subgraph, background_logits):
    return {
        "sample_keys": static["sample_keys"],
        "sites": static["sites"],
        "labels": static["labels"],
        "subgraph_logits": subgraph["logits"],
        "background_logits": np.asarray(background_logits, dtype=np.float64),
    }


def evaluate_candidate(candidate, folds, dataset, xgb_seed, nthread, save_dir=None):
    fitted = fit_candidate(candidate, folds, xgb_seed, nthread, save_dir)
    validation_folds = [
        fusion_fold(
            fold["validation"], fold["subgraph_validation"], prediction["validation"]
        )
        for fold, prediction in zip(folds, fitted["predictions"])
    ]
    selection = select_safe_fusion(validation_folds, SafeFusionConfig(dataset=dataset))
    rows = []
    for index, (fold, prediction) in enumerate(zip(folds, fitted["predictions"])):
        static = fold["test"]
        subgraph = fold["subgraph_test"]
        background_logits = prediction["test"]
        final_logits = apply_safe_fusion(
            selection, subgraph["logits"], background_logits
        )
        rows.append({
            "rotation": index,
            "subgraph": score_logits(
                static["labels"], static["sites"], subgraph["logits"]
            ),
            "s3_neural": score_logits(
                static["labels"], static["sites"], static["base_logits"]
            ),
            "s3_xgb": score_logits(
                static["labels"], static["sites"], background_logits
            ),
            "final": score_logits(
                static["labels"], static["sites"], final_logits
            ),
            "final_logits": final_logits,
            "s3_xgb_logits": background_logits,
        })
    summary = {
        name: mean_metrics([row[name] for row in rows])
        for name in ("subgraph", "s3_neural", "s3_xgb", "final")
    }
    return {
        "candidate": dict(candidate),
        "xgb_parameters": fitted["xgb_parameters"],
        "rounds": fitted["rounds"],
        "residual_alpha": fitted["residual_alpha"],
        "positive_class_weight": fitted["positive_class_weight"],
        "safe_fusion_selection": selection,
        "rotations": rows,
        "mean_test_metrics": summary,
        "predictions": fitted["predictions"],
    }


def compact_result(result):
    return {
        "candidate": result["candidate"],
        "xgb_parameters": result["xgb_parameters"],
        "safe_fusion_selection": result["safe_fusion_selection"],
        "mean_test_metrics": result["mean_test_metrics"],
        "rotations": [
            {
                "rotation": row["rotation"],
                "subgraph": row["subgraph"],
                "s3_neural": row["s3_neural"],
                "s3_xgb": row["s3_xgb"],
                "final": row["final"],
            }
            for row in result["rotations"]
        ],
    }


def reorder(values, keys, order):
    position = {key: index for index, key in enumerate(keys.tolist())}
    if set(position) != set(order):
        raise ValueError("fixed test cohorts differ across rotations")
    return np.asarray(values)[[position[key] for key in order]]


def write_predictions(path, static, subgraph, s3_xgb, final):
    path.parent.mkdir(parents=True, exist_ok=True)
    probability = sigmoid(final)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_key", "site", "label", "subgraph_logit", "s3_neural_logit",
            "s3_xgb_logit", "final_logit", "probability", "prediction",
        ))
        for index, key in enumerate(static["sample_keys"]):
            writer.writerow((
                key, static["sites"][index], int(static["labels"][index]),
                float(subgraph["logits"][index]), float(static["base_logits"][index]),
                float(s3_xgb[index]), float(final[index]), float(probability[index]),
                int(probability[index] >= 0.5),
            ))


def main():
    args = parse_args()
    if len(args.static_fold_dir) != 4 or len(args.subgraph_prediction_dir) != 4:
        raise ValueError("exactly four aligned S3 and subgraph directories are required")
    if args.trials < 1:
        raise ValueError("trials must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for static_dir, subgraph_dir in zip(
        args.static_fold_dir, args.subgraph_prediction_dir
    ):
        fold = {
            split: load_static_split(static_dir, split)
            for split in ("train", "validation", "test")
        }
        fold["subgraph_validation"] = load_prediction_split(
            subgraph_dir, "validation", fold["validation"]
        )
        fold["subgraph_test"] = load_prediction_split(
            subgraph_dir, "test", fold["test"]
        )
        folds.append(fold)

    explicit = []
    for path in args.candidate_json:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        explicit.append(dict(payload.get("candidate", payload)))
    trials = []
    best = None
    for trial, candidate in enumerate(candidate_stream(
        args.trials, args.search_seed, explicit
    )):
        result = evaluate_candidate(
            candidate, folds, args.dataset, args.xgb_seed, args.nthread
        )
        metrics = result["mean_test_metrics"]["final"]
        key = (
            float(metrics["roc_auc"]),
            float(metrics["accuracy"]),
            min(float(row["final"]["roc_auc"]) for row in result["rotations"]),
        )
        row = compact_result(result)
        row["trial"] = trial
        row["selection_key"] = list(key)
        trials.append(row)
        if best is None or key > best[0]:
            best = (key, trial, candidate)
        print(
            "trial {}/{} final_test_auc={:.6f} final_test_acc={:.6f} "
            "s3_xgb_auc={:.6f} weight={:.2f} source={}".format(
                trial + 1, args.trials, metrics["roc_auc"], metrics["accuracy"],
                result["mean_test_metrics"]["s3_xgb"]["roc_auc"],
                result["safe_fusion_selection"]["selected_subgraph_weight"],
                result["safe_fusion_selection"]["selected_source"],
            ), flush=True,
        )

    winning = evaluate_candidate(
        best[2], folds, args.dataset, args.xgb_seed, args.nthread,
        save_dir=output_dir / "boosters",
    )
    compact_winning = compact_result(winning)
    for fold_index, (fold, row, prediction) in enumerate(zip(
        folds, winning["rotations"], winning["predictions"]
    )):
        prediction_dir = output_dir / "predictions" / ("fold_{}".format(fold_index))
        write_predictions(
            prediction_dir / "test_predictions.csv", fold["test"],
            fold["subgraph_test"], prediction["test"], row["final_logits"],
        )
        # Persist the winning S3-XGB validation logits for reproducible refits.
        validation_final = apply_safe_fusion(
            winning["safe_fusion_selection"],
            fold["subgraph_validation"]["logits"], prediction["validation"],
        )
        write_predictions(
            prediction_dir / "validation_predictions.csv", fold["validation"],
            fold["subgraph_validation"], prediction["validation"], validation_final,
        )

    # The four rotations share one fixed test cohort; provide a secondary logit ensemble.
    reference = folds[0]["test"]
    order = reference["sample_keys"].tolist()
    ensemble = {}
    for name in ("subgraph", "s3_neural", "s3_xgb", "final"):
        values = []
        for fold, row in zip(folds, winning["rotations"]):
            if name == "subgraph":
                raw = fold["subgraph_test"]["logits"]
            elif name == "s3_neural":
                raw = fold["test"]["base_logits"]
            elif name == "s3_xgb":
                raw = row["s3_xgb_logits"]
            else:
                raw = row["final_logits"]
            values.append(reorder(raw, fold["test"]["sample_keys"], order))
        logits = np.mean(np.stack(values, axis=0), axis=0)
        ensemble[name] = score_logits(reference["labels"], reference["sites"], logits)

    report = {
        "artifact_type": "mokse_s3_xgb_safe_fusion_test_guided_fourfold_v1",
        "dataset": args.dataset,
        "selection_rule": (
            "maximize shared-parameter four-rotation mean Test AUROC; then mean "
            "Test ACC@0.5; then minimum rotation Test AUROC"
        ),
        "test_used_for_xgb_parameter_selection": True,
        "test_used_for_fusion_weight_selection": False,
        "unbiased_generalization_estimate": False,
        "decision_threshold": 0.5,
        "fourfold_shared_xgb_candidate": True,
        "search_seed": args.search_seed,
        "xgb_seed": args.xgb_seed,
        "trial_count": args.trials,
        "static_fold_directories": [str(path.resolve()) for path in args.static_fold_dir],
        "subgraph_prediction_directories": [
            str(path.resolve()) for path in args.subgraph_prediction_dir
        ],
        "best_trial": best[1],
        "best": compact_winning,
        "ensemble_metrics": ensemble,
        "trials": trials,
    }
    atomic_json(output_dir / "search_results.json", report)
    mean = compact_winning["mean_test_metrics"]
    selection = compact_winning["safe_fusion_selection"]
    lines = [
        "# {} S3-XGB与冻结子图安全融合".format(args.dataset.upper()), "",
        "- 明确test-guided XGB参数搜索：是", "- 四折共享XGB配置：是",
        "- test参与融合权重选择：否", "- 决策阈值：0.5",
        "- 最佳trial：{}".format(best[1]),
        "- 子图融合权重：{:.6f}".format(selection["selected_subgraph_weight"]),
        "- 融合来源：`{}`".format(selection["selected_source"]), "",
        "| 路径 | Mean Test AUROC | Mean Test ACC | BA | AUPRC | Site-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("subgraph", "s3_neural", "s3_xgb", "final"):
        metric = mean[name]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                name, metric["roc_auc"], metric["accuracy"],
                metric["balanced_accuracy"], metric["auprc"],
                metric["site_stratified_roc_auc"],
            )
        )
    lines.extend([
        "", "> 本结果按用户指定由固定test指导XGB参数选择，仅表示探索性上限，",
        "> 不能作为无偏泛化性能估计。", "",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "best_trial": best[1], "candidate": best[2],
        "fusion": selection, "mean_test_metrics": mean,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
