"""Fail-closed audit of one completed fold dry run."""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256, read_split_assignments  # noqa: E402
from keysubgraph.training.baseline_trainer import read_baseline_checkpoint_payload  # noqa: E402


EXPECTED_VARIANTS = {
    "A": ("key", "signed"), "B": ("random", "signed"),
    "C": ("key", "node_only"), "D": ("random", "node_only"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--downstream-root", type=Path, default=PROJECT_ROOT / "outputs/crossfit/downstream")
    parser.add_argument("--perturbation-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    assignments = read_split_assignments(PROJECT_ROOT / protocol["paths"]["splits_csv"])
    groups = {name: {row.group_id for row in assignments if row.split == name} for name in ("train", "validation", "test")}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("subject leakage in fold protocol")
    protocol_test = {row.sample_key for row in assignments if row.split == "test"}
    key_test_manifest = PROJECT_ROOT / "outputs/crossfit/manifests/fold_{}/key_outer_test/baseline_manifest.json".format(args.fold)
    with key_test_manifest.open("r", encoding="utf-8") as handle:
        key_manifest_payload = json.load(handle)
    expected_test = {row["sample_key"] for row in key_manifest_payload["records"]}
    if not expected_test or not expected_test.issubset(protocol_test):
        raise ValueError("common control cohort is not a valid outer-test subset")
    checkpoint_hashes = {}
    prediction_sets = {}
    for variant, (source, encoder) in EXPECTED_VARIANTS.items():
        run_dir = args.downstream_root / "fold{}_{}_seed{}".format(args.fold, variant, args.model_seed)
        checkpoint_path = run_dir / "best_checkpoint.pt"
        prediction_path = run_dir / "outer_test_predictions.json"
        checkpoint = read_baseline_checkpoint_payload(checkpoint_path)
        config = checkpoint["model_config"]
        if config.get("encoder_type", "signed") != encoder or config.get("history_mode") != "independent_bag":
            raise ValueError("variant model configuration differs: {}".format(variant))
        if checkpoint.get("subgraph_source", "key") != source:
            raise ValueError("variant source differs: {}".format(variant))
        if int(checkpoint["training_config"]["seed"]) != args.model_seed:
            raise ValueError("variant model seed differs")
        checkpoint_hashes[variant] = file_sha256(checkpoint_path)
        with prediction_path.open("r", encoding="utf-8") as handle:
            prediction = json.load(handle)
        if prediction.get("checkpoint_sha256") != checkpoint_hashes[variant]:
            raise ValueError("prediction checkpoint hash differs")
        keys = [row["sample_key"] for row in prediction.get("predictions", [])]
        if len(keys) != len(set(keys)) or set(keys) != expected_test:
            raise ValueError("variant OOF coverage differs")
        prediction_sets[variant] = set(keys)
    if len(set(checkpoint_hashes.values())) != 4:
        raise ValueError("variants incorrectly share a downstream checkpoint")
    with (args.perturbation_root / "perturbation_run_summary.json").open("r", encoding="utf-8") as handle:
        perturbations = json.load(handle)
    if perturbations.get("condition_count") != 13:
        raise ValueError("perturbation condition count differs")
    if Path(perturbations["checkpoint"]).resolve() != (args.downstream_root / "fold{}_A_seed{}".format(args.fold, args.model_seed) / "best_checkpoint.pt").resolve():
        raise ValueError("perturbations do not reuse the A checkpoint")
    for condition in perturbations["conditions"]:
        path = Path(condition["predictions"])
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("checkpoint_sha256") != checkpoint_hashes["A"]:
            raise ValueError("perturbation prediction uses a non-A checkpoint")
    with args.analysis.open("r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    if not analysis.get("coverage_audit", {}).get("valid"):
        raise ValueError("fold analysis coverage audit failed")
    result = {
        "schema_version": 1, "valid": True, "outer_fold": args.fold,
        "model_seed": args.model_seed, "test_sample_count": len(expected_test),
        "protocol_test_sample_count": len(protocol_test),
        "control_matching_excluded_test_count": len(protocol_test - expected_test),
        "unique_variant_checkpoints": 4, "perturbation_condition_count": 13,
        "checks": {
            "subject_leakage": False, "variant_oof_coverage_identical": True,
            "checkpoint_reuse_across_variants": False,
            "perturbations_reuse_A_checkpoint": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
