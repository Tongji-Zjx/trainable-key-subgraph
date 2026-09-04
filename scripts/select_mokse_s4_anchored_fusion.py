#!/usr/bin/env python3
"""Build S4 OOF seed scores and select an anchored static complement."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.s4_fusion import (  # noqa: E402
    S4AnchoredFusionConfig,
    apply_s4_anchored_fusion,
    apply_s4_seed_ensemble,
    fit_s4_seed_ensemble,
    select_s4_anchored_fusion,
)
from keysubgraph.background.safe_fusion import score_logits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("adhd", "wmrc"), required=True)
    parser.add_argument(
        "--seed-fold-dir",
        action="append",
        required=True,
        help="FOLD:SEED:RUN_DIR; provide the same three seeds for every fold",
    )
    parser.add_argument(
        "--subgraph-prediction-dir",
        action="append",
        type=Path,
        required=True,
        help="one directory per fold containing oof_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-role",
        choices=("development_oof", "checkpoint_selection_validation"),
        default="development_oof",
        help=(
            "Use strict development OOF by default. The validation option is "
            "an explicitly biased legacy-screening mode."
        ),
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--minimum-mean-auc-gain", type=float, default=0.003)
    parser.add_argument("--maximum-mean-accuracy-drop", type=float, default=0.005)
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_seed_fold_specs(values):
    result = {}
    for value in values:
        parts = str(value).split(":", 2)
        if len(parts) != 3:
            raise ValueError("seed-fold spec must be FOLD:SEED:RUN_DIR")
        fold, seed, directory = int(parts[0]), int(parts[1]), Path(parts[2]).resolve()
        if (fold, seed) in result:
            raise ValueError("duplicate S4 fold/seed run")
        result[(fold, seed)] = directory
    folds = tuple(sorted(set(fold for fold, _ in result)))
    seeds = tuple(sorted(set(seed for _, seed in result)))
    if len(folds) < 3 or len(seeds) != 3:
        raise ValueError("S4 selection requires at least three folds and exactly three seeds")
    if set(result) != {(fold, seed) for fold in folds for seed in seeds}:
        raise ValueError("S4 fold/seed run matrix is incomplete")
    return result, folds, seeds


def load_static(directory, split, expected_role=None):
    path = Path(directory) / (split + "_features.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    provenance_path = path.with_suffix(path.suffix + ".json")
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "S4 strict selection requires prediction provenance: {}".format(
                provenance_path
            )
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if expected_role is None:
        expected_role = "development_oof" if split == "oof" else "fixed_test"
    if provenance.get("prediction_role") != expected_role:
        raise ValueError("S4 static prediction role mismatch")
    if provenance.get("test_used_for_fit", False):
        raise ValueError("S4 static predictions were fitted with fixed-test data")
    if provenance.get("cross_seed_representation_averaging", False):
        raise ValueError("S4 forbids cross-seed representation averaging")
    if expected_role == "development_oof" and not (
        provenance.get("oof_disjointness_audit") or {}
    ).get("all_disjoint", False):
        raise ValueError("S4 static OOF disjointness was not verified")
    payload = np.load(str(path), allow_pickle=False)
    required = ("sample_keys", "sites", "labels", "background_logits")
    if any(name not in payload for name in required):
        raise ValueError("S4 static export is incomplete: {}".format(path))
    return {
        "sample_keys": payload["sample_keys"].astype(str),
        "sites": payload["sites"].astype(str),
        "labels": payload["labels"].astype(np.int64),
        "logits": payload["background_logits"].astype(np.float64),
        "source": str(path),
        "provenance": str(provenance_path),
        "conditional_on_frozen_selector_and_trajectory_cache": bool(
            provenance.get(
                "conditional_on_frozen_selector_and_trajectory_cache", False
            )
        ),
        "end_to_end_selector_oof": bool(
            provenance.get("end_to_end_selector_oof", True)
        ),
        "historical_fixed_test_guided_hyperparameters": bool(
            provenance.get("historical_fixed_test_guided_hyperparameters", False)
        ),
    }


def align_static(reference, observed):
    position = {key: index for index, key in enumerate(observed["sample_keys"].tolist())}
    keys = reference["sample_keys"].tolist()
    if set(position) != set(keys):
        raise ValueError("S4 seed cohorts differ")
    order = [position[key] for key in keys]
    for name in ("labels", "sites"):
        if not np.array_equal(reference[name], observed[name][order]):
            raise ValueError("S4 seed label/site provenance differs")
    return observed["logits"][order]


def load_subgraph(directory, split, reference, expected_role=None):
    path = Path(directory) / (split + "_predictions.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    provenance_path = path.with_suffix(path.suffix + ".json")
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "S4 strict selection requires subgraph prediction provenance: {}".format(
                provenance_path
            )
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if expected_role is None:
        expected_role = "development_oof" if split == "oof" else "fixed_test"
    if provenance.get("prediction_role") != expected_role:
        raise ValueError("S4 subgraph prediction role mismatch")
    if expected_role == "development_oof" and not (
        provenance.get("oof_disjointness_audit") or {}
    ).get("all_disjoint", False):
        raise ValueError("S4 subgraph OOF disjointness was not verified")
    if provenance.get("test_used_for_fit", False):
        raise ValueError("S4 subgraph predictions were fitted with fixed-test data")
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("sample_key", ""))
            if not key or key in rows or "final_logit" not in row:
                raise ValueError("invalid frozen subgraph predictions")
            rows[key] = row
    keys = reference["sample_keys"].tolist()
    if set(rows) != set(keys):
        raise ValueError("S4 and frozen subgraph cohorts differ")
    logits = []
    for index, key in enumerate(keys):
        row = rows[key]
        if row.get("label") not in (None, "") and int(row["label"]) != int(
            reference["labels"][index]
        ):
            raise ValueError("S4/subgraph label mismatch")
        if row.get("site") not in (None, "") and str(row["site"]) != str(
            reference["sites"][index]
        ):
            raise ValueError("S4/subgraph site mismatch")
        logits.append(float(row["final_logit"]))
    return {
        "logits": np.asarray(logits, dtype=np.float64),
        "source": str(path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "conditional_on_frozen_selector_and_trajectory_cache": bool(
            provenance.get(
                "conditional_on_frozen_selector_and_trajectory_cache", False
            )
        ),
        "end_to_end_selector_oof": bool(
            provenance.get("end_to_end_selector_oof", True)
        ),
        "historical_fixed_test_guided_hyperparameters": bool(
            provenance.get("historical_fixed_test_guided_hyperparameters", False)
        ),
    }


def write_prediction_diagnostics(path, rows):
    fieldnames = (
        "fold", "sample_key", "site", "label", "subgraph_logit",
        "static_seed_1_logit", "static_seed_2_logit", "static_seed_3_logit",
        "static_raw_median_logit", "static_standardized_median_score",
        "static_uncertainty", "static_residual", "reliability_q", "beta",
        "correction", "fused_logit", "fallback_flag",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows):
    output = {}
    for name in (
        "roc_auc", "accuracy", "balanced_accuracy", "auprc", "f1",
        "site_stratified_roc_auc",
    ):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        output[name] = None if not values else float(np.mean(values))
        output[name + "_std"] = None if not values else float(np.std(values))
    return output


def main():
    args = parse_args()
    matrix, folds, seeds = parse_seed_fold_specs(args.seed_fold_dir)
    if len(args.subgraph_prediction_dir) != len(folds):
        raise ValueError("one frozen subgraph directory is required per fold")

    selection_split = (
        "oof" if args.selection_role == "development_oof" else "validation"
    )
    validation = {}
    conditional_flags = []
    end_to_end_flags = []
    historical_test_guided_flags = []
    for fold in folds:
        # oof_features must be exported by a checkpoint selected on a separate
        # inner-validation split; ordinary validation_features are rejected.
        reference = load_static(
            matrix[(fold, seeds[0])], selection_split,
            expected_role=args.selection_role,
        )
        seed_logits = {seeds[0]: reference["logits"]}
        for seed in seeds[1:]:
            seed_logits[seed] = align_static(
                reference,
                load_static(
                    matrix[(fold, seed)], selection_split,
                    expected_role=args.selection_role,
                ),
            )
        validation[fold] = {
            "reference": reference,
            "seed_logits": seed_logits,
        }
        conditional_flags.append(
            bool(reference["conditional_on_frozen_selector_and_trajectory_cache"])
        )
        end_to_end_flags.append(bool(reference["end_to_end_selector_oof"]))

    seen = set()
    for fold in folds:
        keys = validation[fold]["reference"]["sample_keys"].tolist()
        if seen.intersection(keys):
            raise ValueError("S4 development-OOF folds are not disjoint")
        seen.update(keys)

    def ensemble_fit_from_folds(included_folds):
        by_seed = {seed: {"sample_keys": [], "logits": []} for seed in seeds}
        for included_fold in included_folds:
            keys = validation[included_fold]["reference"]["sample_keys"].tolist()
            for seed in seeds:
                by_seed[seed]["sample_keys"].extend(keys)
                by_seed[seed]["logits"].extend(
                    validation[included_fold]["seed_logits"][seed].tolist()
                )
        return fit_s4_seed_ensemble([
            {
                "seed": seed,
                "sample_keys": by_seed[seed]["sample_keys"],
                "logits": by_seed[seed]["logits"],
                "prediction_role": args.selection_role,
            }
            for seed in seeds
        ], expected_seeds=seeds, fit_role=args.selection_role)

    # The all-OOF fit is frozen only for later fixed-test application.  Each
    # development fold below is scored with a scale/tau fit from other folds.
    ensemble_fit = ensemble_fit_from_folds(folds)

    fusion_folds = []
    fold_ensembles = {}
    fold_ensemble_fits = {}
    for fold, subgraph_dir in zip(folds, args.subgraph_prediction_dir):
        reference = validation[fold]["reference"]
        cross_fit = ensemble_fit_from_folds(
            [other for other in folds if other != fold]
        )
        ensemble = apply_s4_seed_ensemble(cross_fit, [
            {
                "seed": seed,
                "sample_keys": reference["sample_keys"],
                "logits": validation[fold]["seed_logits"][seed],
                "prediction_role": args.selection_role,
            }
            for seed in seeds
        ])
        subgraph_export = load_subgraph(
            subgraph_dir, selection_split, reference,
            expected_role=args.selection_role,
        )
        subgraph = subgraph_export["logits"]
        conditional_flags.append(
            bool(
                subgraph_export[
                    "conditional_on_frozen_selector_and_trajectory_cache"
                ]
            )
        )
        end_to_end_flags.append(bool(subgraph_export["end_to_end_selector_oof"]))
        historical_test_guided_flags.append(
            bool(subgraph_export["historical_fixed_test_guided_hyperparameters"])
        )
        fusion_folds.append({
            "sample_keys": reference["sample_keys"],
            "sites": reference["sites"],
            "labels": reference["labels"],
            "subgraph_logits": subgraph,
            "static_scores": ensemble["standardized_median_score"],
            "static_uncertainty": ensemble["uncertainty"],
            "prediction_role": args.selection_role,
        })
        fold_ensembles[fold] = ensemble
        fold_ensemble_fits[fold] = cross_fit

    config = S4AnchoredFusionConfig(
        dataset=args.dataset,
        minimum_mean_auc_gain=args.minimum_mean_auc_gain,
        maximum_mean_accuracy_drop=args.maximum_mean_accuracy_drop,
    )
    selection = select_s4_anchored_fusion(
        fusion_folds, config, selection_role=args.selection_role
    )
    report = {
        "artifact_type": "mokse_s4_oof_selection_run_v1",
        "dataset": args.dataset,
        "folds": list(folds),
        "seeds": list(seeds),
        "seed_ensemble_fit": ensemble_fit,
        "seed_ensemble_cross_fitted_for_development_oof": True,
        "seed_ensemble_fold_fits": {
            str(fold): fold_ensemble_fits[fold] for fold in folds
        },
        "anchored_fusion_selection": selection,
        "fixed_test_evaluated": bool(args.evaluate_test),
        "fixed_test_used_for_selection": False,
        "selection_role": args.selection_role,
        "conditional_on_frozen_selector_and_trajectory_cache": bool(
            conditional_flags and all(conditional_flags)
        ),
        "end_to_end_selector_oof": bool(
            end_to_end_flags and all(end_to_end_flags)
        ),
        "historical_fixed_test_guided_hyperparameters": bool(
            historical_test_guided_flags and any(historical_test_guided_flags)
        ),
        "unbiased_fusion_given_frozen_upstream": (
            args.selection_role == "development_oof"
        ),
        "unbiased_generalization_estimate": (
            args.selection_role == "development_oof"
            and bool(end_to_end_flags and all(end_to_end_flags))
        ),
    }

    development_rows = []
    for fold_index, fold in enumerate(folds):
        payload = fusion_folds[fold_index]
        ensemble = fold_ensembles[fold]
        development_rows.append({
            "fold": fold,
            "subgraph": score_logits(
                payload["labels"], payload["sites"], payload["subgraph_logits"]
            ),
            "static_raw_median": score_logits(
                payload["labels"], payload["sites"], ensemble["raw_median_logit"]
            ),
        })
    report["development_oof_fold_metrics"] = development_rows
    report["development_oof_mean"] = {
        name: mean_metrics([row[name] for row in development_rows])
        for name in ("subgraph", "static_raw_median")
    }

    diagnostic_rows = []
    beta = float(selection["selected_beta"])
    fallback = selection["selected_source"] == "subgraph_exact_fallback"
    for fold_index, fold in enumerate(folds):
        payload = fusion_folds[fold_index]
        ensemble = fold_ensembles[fold]
        fold_selection = dict(selection)
        fold_selection["final_calibration"] = selection[
            "leave_one_fold_out_calibrations"
        ][fold_index]
        fused = apply_s4_anchored_fusion(
            fold_selection,
            payload["subgraph_logits"],
            payload["static_scores"],
            payload["static_uncertainty"],
        )
        raw_seed_logits = np.stack(
            [validation[fold]["seed_logits"][seed] for seed in seeds], axis=1
        )
        for index, sample_key in enumerate(payload["sample_keys"]):
            diagnostic_rows.append({
                "fold": fold,
                "sample_key": str(sample_key),
                "site": str(payload["sites"][index]),
                "label": int(payload["labels"][index]),
                "subgraph_logit": float(payload["subgraph_logits"][index]),
                "static_seed_1_logit": float(raw_seed_logits[index, 0]),
                "static_seed_2_logit": float(raw_seed_logits[index, 1]),
                "static_seed_3_logit": float(raw_seed_logits[index, 2]),
                "static_raw_median_logit": float(
                    ensemble["raw_median_logit"][index]
                ),
                "static_standardized_median_score": float(
                    ensemble["standardized_median_score"][index]
                ),
                "static_uncertainty": float(ensemble["uncertainty"][index]),
                "static_residual": float(fused["static_residual"][index]),
                "reliability_q": float(fused["reliability"][index]),
                "beta": beta,
                "correction": float(fused["correction"][index]),
                "fused_logit": float(fused["fused_logits"][index]),
                "fallback_flag": int(fallback),
            })

    if args.evaluate_test:
        test_rows = []
        for fold, subgraph_dir in zip(folds, args.subgraph_prediction_dir):
            reference = load_static(
                matrix[(fold, seeds[0])], "test", expected_role="fixed_test"
            )
            payloads = [{
                "seed": seeds[0], "sample_keys": reference["sample_keys"],
                "logits": reference["logits"], "prediction_role": "fixed_test",
            }]
            for seed in seeds[1:]:
                observed = load_static(
                    matrix[(fold, seed)], "test", expected_role="fixed_test"
                )
                payloads.append({
                    "seed": seed,
                    "sample_keys": reference["sample_keys"],
                    "logits": align_static(reference, observed),
                    "prediction_role": "fixed_test",
                })
            ensemble = apply_s4_seed_ensemble(ensemble_fit, payloads)
            subgraph = load_subgraph(
                subgraph_dir, "test", reference, expected_role="fixed_test"
            )["logits"]
            fused = apply_s4_anchored_fusion(
                selection,
                subgraph,
                ensemble["standardized_median_score"],
                ensemble["uncertainty"],
            )
            test_rows.append({
                "fold": fold,
                "subgraph": score_logits(reference["labels"], reference["sites"], subgraph),
                "static_raw_median": score_logits(
                    reference["labels"], reference["sites"], ensemble["raw_median_logit"]
                ),
                "fused": score_logits(
                    reference["labels"], reference["sites"], fused["fused_logits"]
                ),
            })
        report["fixed_test_rows"] = test_rows
        report["fixed_test_mean"] = {
            name: mean_metrics([row[name] for row in test_rows])
            for name in ("subgraph", "static_raw_median", "fused")
        }
        report["fixed_test_interpretation"] = (
            "shared fixed-test repeated across rotation-trained models; exploratory audit only"
        )
        if args.selection_role != "development_oof":
            report["methodological_limitation"] = (
                "fusion beta was screened on checkpoint-selection validation "
                "predictions; this is not a strict OOF or unbiased estimate"
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "seed_ensemble_fit.json", ensemble_fit)
    atomic_json(output_dir / "anchored_fusion_selection.json", selection)
    atomic_json(output_dir / "evaluation.json", report)
    write_prediction_diagnostics(
        output_dir / "development_oof_diagnostics.csv", diagnostic_rows
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
