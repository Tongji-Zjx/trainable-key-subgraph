#!/usr/bin/env python3
"""Train E1 and E3 MoKSE-Net-BG models on one frozen fold."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# This must be set before importing/initializing torch CUDA.  Setting it from
# the trainer after a frozen TGE forward is too late for strict CuBLAS mode.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.data import (  # noqa: E402
    build_global_static_record,
    fit_background_feature_scaler,
)
from keysubgraph.background.model import StaticBackgroundConfig  # noqa: E402
from keysubgraph.background.training import (  # noqa: E402
    BackgroundFusionDataset,
    BackgroundTrainingConfig,
    train_background_model,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.tge.trainer import load_tge_checkpoint  # noqa: E402
from keysubgraph.tge.xgb_residual import collect_frozen_c4_split  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", nargs="+", choices=("background_only", "fusion"), default=("background_only", "fusion"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--spectral-dim", type=int, default=8)
    parser.add_argument("--lambda-background", type=float, default=0.20)
    parser.add_argument("--lambda-rank", type=float, default=0.05)
    parser.add_argument("--lambda-gate", type=float, default=1.0e-3)
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint, evolution_model = load_tge_checkpoint(checkpoint_path, device)
    evolution_model.eval()
    manifests = {
        "train": args.train_manifest.resolve(),
        "validation": args.validation_manifest.resolve(),
        "test": args.test_manifest.resolve(),
    }
    evolution = {
        split: collect_frozen_c4_split(
            evolution_model, path, split, device, args.batch_size
        )
        for split, path in manifests.items()
    }
    background = {}
    for split in ("train", "validation", "test"):
        background[split] = tuple(
            build_global_static_record(
                args.global_root.resolve(), row,
                spectral_dimensions=args.spectral_dim,
                cache_dir=args.cache_dir.resolve(),
            )
            for row in evolution[split]["rows"]
        )
    scaler = fit_background_feature_scaler(background["train"])
    datasets = {
        split: BackgroundFusionDataset(background[split], evolution[split], scaler)
        for split in ("train", "validation", "test")
    }
    model_config = StaticBackgroundConfig(
        input_dim=12 + args.spectral_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    training_config = BackgroundTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        lambda_background=args.lambda_background,
        lambda_rank=args.lambda_rank,
        lambda_gate=args.lambda_gate,
        strict_deterministic=not args.non_strict,
    )
    reports = {}
    for mode in args.modes:
        reports[mode] = train_background_model(
            datasets["train"], datasets["validation"], datasets["test"],
            scaler, args.output_dir / mode, device, mode,
            model_config=model_config, training_config=training_config,
        )
    provenance = {
        "artifact_type": "mokse_background_fold_run_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "global_root": str(args.global_root.resolve()),
        "manifest_sha256": {split: file_sha256(path) for split, path in manifests.items()},
        "sample_count": {split: len(dataset) for split, dataset in datasets.items()},
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "reports": reports,
    }
    atomic_json(args.output_dir / "run_manifest.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
