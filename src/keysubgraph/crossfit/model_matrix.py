"""Frozen A-D model matrix for the minimal OOF confirmatory experiment."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from keysubgraph.data.data_split import file_sha256


OOF_RUN_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelVariant:
    name: str
    source: str
    encoder_type: str
    description: str


MODEL_VARIANTS = (
    ModelVariant("A", "key", "signed", "Key Signed Bag"),
    ModelVariant("B", "random", "signed", "Random Signed Bag"),
    ModelVariant("C", "key", "node_only", "Key No-edge-message Bag"),
    ModelVariant("D", "random", "node_only", "Random No-edge-message Bag"),
)


def _manifest_path(fold, source, role):
    return (
        "outputs/crossfit/manifests/fold_{}/{}_{}/baseline_manifest.json"
        .format(fold, source, role)
    )


def build_oof_run_plan(fold_assignments_path, folds=5, seeds=(42, 43, 44)):
    """Return the immutable 4 x fold x seed downstream training plan."""

    fold_assignments_path = Path(fold_assignments_path).resolve()
    if not fold_assignments_path.is_file():
        raise FileNotFoundError(str(fold_assignments_path))
    if folds < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid OOF folds or seeds")
    if any(int(seed) < 0 for seed in seeds):
        raise ValueError("model seeds must be non-negative")
    records = []
    for fold in range(folds):
        for variant in MODEL_VARIANTS:
            for seed in seeds:
                run_id = "fold{}_{}_seed{}".format(fold, variant.name, seed)
                output_dir = "outputs/crossfit/downstream/{}".format(run_id)
                records.append({
                    "run_id": run_id,
                    "outer_fold": fold,
                    "variant": variant.name,
                    "description": variant.description,
                    "source": variant.source,
                    "encoder_type": variant.encoder_type,
                    "history_mode": "independent_bag",
                    "seed": int(seed),
                    "train_manifest": _manifest_path(fold, variant.source, "inner_train"),
                    "validation_manifest": _manifest_path(fold, variant.source, "inner_validation"),
                    "test_manifest": _manifest_path(fold, variant.source, "outer_test"),
                    "control_inventory": (
                        "outputs/crossfit/controls/fold_{}/key_random_inventory.json"
                        .format(fold)
                    ),
                    "extractor_checkpoint": (
                        "outputs/crossfit/extractor/fold_{}/best_checkpoint.pt".format(fold)
                    ),
                    "output_dir": output_dir,
                    "checkpoint": output_dir + "/best_checkpoint.pt",
                    "oof_predictions": output_dir + "/outer_test_predictions.json",
                })
    run_ids = [item["run_id"] for item in records]
    checkpoints = [item["checkpoint"] for item in records]
    if len(set(run_ids)) != len(run_ids) or len(set(checkpoints)) != len(checkpoints):
        raise AssertionError("OOF run identities or checkpoints collide")
    return {
        "schema_version": OOF_RUN_PLAN_SCHEMA_VERSION,
        "immutable": True,
        "purpose": "minimal_cross_fitted_key_structure_confirmation",
        "fold_assignments": "configs/crossfit/{}".format(fold_assignments_path.name),
        "fold_assignments_sha256": file_sha256(fold_assignments_path),
        "fold_count": int(folds),
        "seeds": [int(seed) for seed in seeds],
        "variants": [asdict(item) for item in MODEL_VARIANTS],
        "shared_training_configuration": {
            "history_mode": "independent_bag",
            "selection_metric": "unweighted_log_loss",
            "epochs": 100,
            "early_stopping_patience": 15,
        },
        "expected_run_count": len(MODEL_VARIANTS) * folds * len(seeds),
        "runs": records,
    }


def write_oof_run_plan(payload, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path
