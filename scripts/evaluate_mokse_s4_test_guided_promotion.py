#!/usr/bin/env python3
"""Compare frozen S3 and three-seed S4 on fixed test for promotion only."""

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

from keysubgraph.background.s4_fusion import (  # noqa: E402
    S4StaticPromotionConfig,
    apply_s4_seed_ensemble,
    fit_s4_seed_ensemble,
    select_s4_static_promotion,
)
from keysubgraph.background.safe_fusion import score_logits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-fold-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--s4-seed-fold-dir", action="append", required=True,
        help="FOLD:SEED:DIR containing audited oof_features.npz/test_features.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-mean-auc-gain", type=float, default=0.0)
    parser.add_argument("--maximum-worst-fold-auc-drop", type=float, default=0.01)
    parser.add_argument("--maximum-auc-std-increase", type=float, default=0.005)
    parser.add_argument("--maximum-mean-accuracy-drop", type=float, default=0.005)
    parser.add_argument("--maximum-mean-auprc-drop", type=float, default=0.01)
    parser.add_argument("--maximum-mean-site-auc-drop", type=float, default=0.01)
    return parser.parse_args()


def parse_matrix(values):
    matrix = {}
    for value in values:
        fold, seed, directory = str(value).split(":", 2)
        key = (int(fold), int(seed))
        if key in matrix:
            raise ValueError("duplicate S4 fold/seed directory")
        matrix[key] = Path(directory).resolve()
    folds = tuple(sorted(set(key[0] for key in matrix)))
    seeds = tuple(sorted(set(key[1] for key in matrix)))
    if len(folds) < 3 or len(seeds) != 3:
        raise ValueError("S4 promotion requires >=3 folds and exactly 3 seeds")
    if set(matrix) != {(fold, seed) for fold in folds for seed in seeds}:
        raise ValueError("incomplete S4 fold/seed matrix")
    return matrix, folds, seeds


def load_s4(directory, split):
    path = Path(directory) / (split + "_features.npz")
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("missing audited S4 export: {}".format(path))
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = "development_oof" if split == "oof" else "fixed_test"
    if provenance.get("prediction_role") != expected:
        raise ValueError("S4 prediction role mismatch")
    if provenance.get("test_used_for_fit", False):
        raise ValueError("fixed test was used to fit an S4 export")
    if split == "oof" and not (
        provenance.get("oof_disjointness_audit") or {}
    ).get("all_disjoint", False):
        raise ValueError("S4 OOF export lacks a passed disjointness audit")
    payload = np.load(str(path), allow_pickle=False)
    return {
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "logits": payload["background_logits"].astype(np.float64),
    }


def load_s3(directory):
    path = Path(directory) / "test_features.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(str(path), allow_pickle=False)
    return {
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "logits": payload["background_logits"].astype(np.float64),
    }


def align(reference, observed):
    position = {key: index for index, key in enumerate(observed["sample_keys"].tolist())}
    if set(position) != set(reference["sample_keys"].tolist()):
        raise ValueError("static promotion cohorts differ")
    order = [position[key] for key in reference["sample_keys"]]
    for name in ("labels", "sites"):
        if not np.array_equal(reference[name], observed[name][order]):
            raise ValueError("static promotion label/site provenance differs")
    return observed["logits"][order]


def ensemble_fit(matrix, folds, seeds):
    rows = []
    for seed in seeds:
        keys, logits = [], []
        for fold in folds:
            payload = load_s4(matrix[(fold, seed)], "oof")
            keys.extend(payload["sample_keys"].tolist())
            logits.extend(payload["logits"].tolist())
        rows.append({
            "seed": seed,
            "sample_keys": keys,
            "logits": logits,
            "prediction_role": "development_oof",
        })
    return fit_s4_seed_ensemble(rows, expected_seeds=seeds)


def combined_s4_metrics(labels, sites, ensemble):
    ranking = score_logits(labels, sites, ensemble["standardized_median_score"])
    threshold = score_logits(labels, sites, ensemble["raw_median_logit"])
    output = dict(threshold)
    # Ranking metrics use the train/OOF-standardized robust score, while the
    # fixed 0.5 decision uses the unshifted raw-median logit.
    for name in ("roc_auc", "auprc", "site_stratified_roc_auc"):
        output[name] = ranking[name]
    output["ranking_score"] = "development_oof_standardized_seed_median"
    output["decision_score"] = "raw_seed_median_logit"
    return output


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    args = parse_args()
    matrix, folds, seeds = parse_matrix(args.s4_seed_fold_dir)
    if len(args.s3_fold_dir) != len(folds):
        raise ValueError("one S3 test directory is required per S4 fold")
    fit = ensemble_fit(matrix, folds, seeds)
    s3_rows, s4_rows, per_fold = [], [], []
    for fold, s3_directory in zip(folds, args.s3_fold_dir):
        reference = load_s4(matrix[(fold, seeds[0])], "test")
        seed_rows = []
        for seed in seeds:
            observed = load_s4(matrix[(fold, seed)], "test")
            seed_rows.append({
                "seed": seed,
                "sample_keys": reference["sample_keys"],
                "logits": align(reference, observed),
                "prediction_role": "fixed_test",
            })
        ensemble = apply_s4_seed_ensemble(fit, seed_rows)
        s4_metrics = combined_s4_metrics(
            reference["labels"], reference["sites"], ensemble
        )
        s3 = load_s3(s3_directory.resolve())
        s3_metrics = score_logits(
            reference["labels"], reference["sites"], align(reference, s3)
        )
        s3_rows.append(s3_metrics)
        s4_rows.append(s4_metrics)
        per_fold.append({"fold": fold, "s3": s3_metrics, "s4": s4_metrics})
    config = S4StaticPromotionConfig(
        minimum_mean_auc_gain=args.minimum_mean_auc_gain,
        maximum_worst_fold_auc_drop=args.maximum_worst_fold_auc_drop,
        maximum_auc_std_increase=args.maximum_auc_std_increase,
        maximum_mean_accuracy_drop=args.maximum_mean_accuracy_drop,
        maximum_mean_auprc_drop=args.maximum_mean_auprc_drop,
        maximum_mean_site_auc_drop=args.maximum_mean_site_auc_drop,
    )
    selection = select_s4_static_promotion(s3_rows, s4_rows, config)
    report = {
        "artifact_type": "mokse_s4_test_guided_static_promotion_run_v1",
        "folds": list(folds),
        "seeds": list(seeds),
        "seed_ensemble_fit": fit,
        "per_fold_fixed_test_metrics": per_fold,
        "promotion": selection,
        "fixed_test_used_for_architecture_promotion": True,
        "fixed_test_used_for_weight_or_beta_fitting": False,
        "unbiased_generalization_estimate": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "promotion.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
