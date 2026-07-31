"""Summarize Stage-1 N0--N4 OOF results and within-dataset gates."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)
from keysubgraph.models.theory_guided_neural import (  # noqa: E402
    THEORY_NEURAL_VARIANTS,
)


def _atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _read_predictions(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_bootstrap(reference, candidate, repeats, seed):
    reference_by_key = {row["sample_key"]: row for row in reference}
    candidate_by_key = {row["sample_key"]: row for row in candidate}
    if set(reference_by_key) != set(candidate_by_key):
        raise ValueError("Stage-1 paired predictions are not aligned")
    rows = [reference_by_key[key] for key in sorted(reference_by_key)]
    labels = np.asarray([int(row["label"]) for row in rows])
    reference_scores = np.asarray([
        float(reference_by_key[row["sample_key"]]["positive_probability"])
        for row in rows
    ])
    candidate_scores = np.asarray([
        float(candidate_by_key[row["sample_key"]]["positive_probability"])
        for row in rows
    ])
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["site"], int(row["label"]))].append(index)
    rng = np.random.RandomState(int(seed))
    values = []
    for _ in range(int(repeats)):
        selected = []
        for indices in strata.values():
            selected.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        selected = np.asarray(selected, dtype=np.int64)
        current_labels = labels[selected]
        if len(set(current_labels.tolist())) < 2:
            continue
        values.append(
            float(roc_auc_score(current_labels, candidate_scores[selected]))
            - float(roc_auc_score(current_labels, reference_scores[selected]))
        )
    if not values:
        raise ValueError("Stage-1 paired bootstrap produced no valid repeat")
    values = np.asarray(values)
    return {
        "repeats": int(len(values)),
        "mean": float(values.mean()),
        "confidence_interval_95": [
            float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
        ],
        "probability_positive": float(np.mean(values > 0.0)),
    }


def _diagnostic_means(output_root, variant, seed, fold_count):
    payloads = []
    for fold in range(fold_count):
        path = (Path(output_root) / "fold_{}".format(fold) / "models"
                / "{}_seed{}".format(variant, seed) / "diagnostics.json")
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    representation = {}
    for name in (
        "normalized_effective_rank", "fisher_ratio", "mean_pairwise_cosine"
    ):
        representation[name] = float(np.mean([
            item["representation"][name] for item in payloads
        ]))
    site_probe = float(np.mean([
        item["site_probes"]["representations"] for item in payloads
        if item["site_probes"]["representations"] is not None
    ]))
    return {
        "representation": representation,
        "site_probe_balanced_accuracy": site_probe,
        "fold_diagnostics": payloads,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.fold_assignments.open("r", encoding="utf-8") as handle:
        fold_count = int(json.load(handle)["num_outer_folds"])
    summaries = {}
    predictions = {}
    diagnostics = {}
    for variant in THEORY_NEURAL_VARIANTS:
        directory = args.output_root / "oof_summary" / "{}_seed{}".format(
            variant, args.seed
        )
        summary_path = directory / "summary.json"
        if args.overwrite or not summary_path.is_file():
            summarize_sv_signed_gin_crossfit(
                args.output_root, args.fold_assignments, variant=variant,
                seed=args.seed, output_dir=directory,
                overwrite=args.overwrite
            )
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries[variant] = json.load(handle)
        predictions[variant] = _read_predictions(directory / "oof_predictions.csv")
        diagnostics[variant] = _diagnostic_means(
            args.output_root, variant, args.seed, fold_count
        )
    baseline = THEORY_NEURAL_VARIANTS[0]
    baseline_metrics = summaries[baseline]["metrics"]
    candidates = {}
    for offset, variant in enumerate(THEORY_NEURAL_VARIANTS[1:], start=1):
        metrics = summaries[variant]["metrics"]
        auc_delta = float(metrics["pooled_oof_roc_auc"]) - float(
            baseline_metrics["pooled_oof_roc_auc"]
        )
        site_delta = float(metrics["pooled_oof_site_stratified_roc_auc"]) - float(
            baseline_metrics["pooled_oof_site_stratified_roc_auc"]
        )
        fold_deltas = [
            float(current["roc_auc"]) - float(reference["roc_auc"])
            for current, reference in zip(
                summaries[variant]["folds"], summaries[baseline]["folds"]
            )
        ]
        rank_delta = (
            diagnostics[variant]["representation"]["normalized_effective_rank"]
            - diagnostics[baseline]["representation"]["normalized_effective_rank"]
        )
        fisher_delta = (
            diagnostics[variant]["representation"]["fisher_ratio"]
            - diagnostics[baseline]["representation"]["fisher_ratio"]
        )
        site_probe_delta = (
            diagnostics[variant]["site_probe_balanced_accuracy"]
            - diagnostics[baseline]["site_probe_balanced_accuracy"]
        )
        candidates[variant] = {
            "pooled_auc_delta": auc_delta,
            "site_stratified_auc_delta": site_delta,
            "fold_auc_deltas": fold_deltas,
            "paired_bootstrap": _paired_bootstrap(
                predictions[baseline], predictions[variant],
                args.bootstrap_repeats, args.bootstrap_seed + offset
            ),
            "normalized_effective_rank_delta": rank_delta,
            "fisher_ratio_delta": fisher_delta,
            "site_probe_balanced_accuracy_delta": site_probe_delta,
            "within_dataset_checks": {
                "positive_pooled_delta": auc_delta > 0.0,
                "pooled_not_clear_drop": auc_delta >= -0.01,
                "site_not_clear_drop": site_delta >= -0.01,
                "not_single_fold_only": sum(value > 0.0 for value in fold_deltas) >= 2,
                "rank_and_fisher_consistent": not (
                    rank_delta > 0.0 and fisher_delta <= 0.0
                ),
                "site_probe_not_obviously_stronger": site_probe_delta <= 0.05,
            },
        }
    result = {
        "artifact_type": "theory_guided_neural_stage1_dataset_summary",
        "dataset": args.dataset,
        "baseline_variant": baseline,
        "primary_candidate": "N4_ema_center",
        "selection_note": (
            "N4 is the preregistered primary; outer-test is not used to choose N1-N4"
        ),
        "summaries": {
            name: {
                "metrics": payload["metrics"], "folds": payload["folds"]
            } for name, payload in summaries.items()
        },
        "diagnostics": diagnostics,
        "comparisons_to_n0": candidates,
    }
    target = args.output_root / "stage1_dataset_summary.json"
    _atomic_json(target, result)
    print("Stage-1 dataset summary:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
