#!/usr/bin/env python3
"""Export a frozen S4 checkpoint ensemble on a disjoint OOF or fixed-test split."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.data import (  # noqa: E402
    BackgroundFeatureScaler,
    build_global_static_record,
)
from keysubgraph.background.model import (  # noqa: E402
    MoKSEBackgroundModel,
    StaticBackgroundConfig,
)
from keysubgraph.background.training import (  # noqa: E402
    BackgroundFusionDataset,
    evaluate_background_ensemble,
)
from keysubgraph.tge.trainer import load_tge_checkpoint  # noqa: E402
from keysubgraph.tge.xgb_residual import collect_frozen_c4_split  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tge-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-split", required=True)
    parser.add_argument(
        "--prediction-role",
        choices=("development_oof", "fixed_test"),
        required=True,
    )
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--checkpoint-validation-manifest", type=Path)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def safe_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def manifest_keys(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = [str(row.get("sample_key", "")) for row in payload.get("records", ())]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("manifest sample keys are empty or duplicated: {}".format(path))
    return set(keys)


def audit_oof_disjointness(args):
    if args.prediction_role != "development_oof":
        return None
    if args.train_manifest is None or args.checkpoint_validation_manifest is None:
        raise ValueError(
            "development_oof export requires train and checkpoint-validation manifests"
        )
    train = manifest_keys(args.train_manifest)
    checkpoint_validation = manifest_keys(args.checkpoint_validation_manifest)
    oof = manifest_keys(args.manifest)
    if train.intersection(checkpoint_validation):
        raise ValueError("inner train and checkpoint-validation manifests overlap")
    if oof.intersection(train) or oof.intersection(checkpoint_validation):
        raise ValueError("OOF manifest overlaps model fitting or checkpoint selection")
    return {
        "train_sample_count": len(train),
        "checkpoint_validation_sample_count": len(checkpoint_validation),
        "oof_sample_count": len(oof),
        "all_disjoint": True,
    }


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
    audit = audit_oof_disjointness(args)
    run_dir = args.run_dir.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    relative_profile_epsilon = float(
        run_manifest.get("relative_profile_epsilon", 1.0e-3)
    )
    relative_profile_log_clip = float(
        run_manifest.get("relative_profile_log_clip", 4.0)
    )
    checkpoints = sorted(run_dir.glob("checkpoint_top_*.pt"))
    if not checkpoints:
        checkpoints = [run_dir / "best_checkpoint.pt"]
    if any(not path.is_file() for path in checkpoints):
        raise FileNotFoundError("S4 checkpoint ensemble is incomplete")
    payloads = [safe_load(path) for path in checkpoints]
    model_config = StaticBackgroundConfig(**payloads[0]["model_config"])
    if model_config.encoder_variant != "s4_robust":
        raise ValueError("export requires an S4 robust background checkpoint")
    scaler = BackgroundFeatureScaler.from_dict(payloads[0]["feature_scaler"])
    for payload in payloads[1:]:
        if payload["model_config"] != payloads[0]["model_config"]:
            raise ValueError("S4 checkpoint ensemble model configs differ")
        if payload["feature_scaler"] != payloads[0]["feature_scaler"]:
            raise ValueError("S4 checkpoint ensemble feature scalers differ")

    device = torch.device(args.device)
    _, tge_model = load_tge_checkpoint(args.tge_checkpoint.resolve(), device)
    evolution = collect_frozen_c4_split(
        tge_model,
        args.manifest.resolve(),
        args.manifest_split,
        device,
        args.batch_size,
    )
    records = tuple(
        build_global_static_record(
            args.global_root.resolve(),
            row,
            spectral_dimensions=model_config.base_input_dim - 12,
            cache_dir=args.cache_dir.resolve() / "profile10_relative",
            include_signed_profile=True,
            signed_profile_mode="relative",
            relative_profile_epsilon=relative_profile_epsilon,
            relative_profile_log_clip=relative_profile_log_clip,
        )
        for row in evolution["rows"]
    )
    dataset = BackgroundFusionDataset(records, evolution, scaler)
    models = []
    for payload in payloads:
        model = MoKSEBackgroundModel(model_config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    result = evaluate_background_ensemble(
        models, dataset, device, args.batch_size, "background_only"
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        sample_keys=np.asarray(result["sample_keys"], dtype=str),
        sites=np.asarray(result["sites"], dtype=str),
        labels=result["labels"].astype(np.int64),
        background_representations=result["background_representations"].astype(np.float32),
        background_logits=result["background_logits"].astype(np.float32),
    )
    provenance = {
        "artifact_type": "mokse_s4_frozen_prediction_export_v1",
        "prediction_role": args.prediction_role,
        "output": str(output),
        "run_dir": str(run_dir),
        "checkpoints": [str(path.resolve()) for path in checkpoints],
        "manifest": str(args.manifest.resolve()),
        "manifest_split": args.manifest_split,
        "sample_count": len(dataset),
        "relative_profile_epsilon": relative_profile_epsilon,
        "relative_profile_log_clip": relative_profile_log_clip,
        "oof_disjointness_audit": audit,
        "representation_averaging_scope": "same_seed_training_trajectory_only",
        "cross_seed_representation_averaging": False,
        "test_used_for_fit": False,
    }
    atomic_json(output.with_suffix(output.suffix + ".json"), provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
