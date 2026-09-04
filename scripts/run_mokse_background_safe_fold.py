#!/usr/bin/env python3
"""Train one S0-S3 MoKSE-BG-Safe static condition on a frozen rotation."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.background.data import (  # noqa: E402
    SIGNED_CONNECTIVITY_PROFILE_NAMES,
    STATIC_FEATURE_NAMES,
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


STAGES = {
    "s0": {
        "profile": False, "community": False,
        "dropedge": 0.0, "consistency": 0.0, "top_k": 1,
    },
    "s1": {
        "profile": True, "community": False,
        "dropedge": 0.0, "consistency": 0.0, "top_k": 1,
    },
    "s2": {
        "profile": True, "community": True,
        "dropedge": 0.0, "consistency": 0.0, "top_k": 1,
    },
    "s3": {
        "profile": True, "community": True,
        "dropedge": 0.05, "consistency": 0.05, "top_k": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="evaluate the fixed test only after the static stage is frozen",
    )
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--spectral-dim", type=int, default=8)
    parser.add_argument("--lambda-rank", type=float, default=0.05)
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(first, second):
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _site_eta_squared(values, sites):
    values = np.asarray(values, dtype=np.float64)
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= 0.0:
        return 0.0
    between = 0.0
    sites = np.asarray(sites, dtype=str)
    for site in np.unique(sites):
        group = values[sites == site]
        between += float(group.size) * (float(np.mean(group)) - overall) ** 2
    return float(between / total)


def feature_audit(records, feature_names):
    node_values = torch.cat([record.node_features for record in records], dim=0).numpy()
    subject_means = np.stack(
        [record.node_features.mean(dim=0).numpy() for record in records], axis=0
    )
    node_counts = np.asarray([record.node_count for record in records], dtype=np.float64)
    sites = np.asarray([record.site for record in records], dtype=str)
    rows = []
    for index, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "node_level_variance": float(np.var(node_values[:, index])),
                "subject_mean_variance": float(np.var(subject_means[:, index])),
                "subject_mean_node_count_spearman": _spearman(
                    subject_means[:, index], node_counts
                ),
                "subject_mean_site_eta_squared": _site_eta_squared(
                    subject_means[:, index], sites
                ),
            }
        )
    return {
        "artifact_type": "mokse_background_train_feature_audit_v1",
        "sample_count": len(records),
        "node_count_minimum": int(node_counts.min()),
        "node_count_maximum": int(node_counts.max()),
        "features": rows,
        "validation_or_test_used": False,
    }


def main():
    args = parse_args()
    if args.evaluate_test and args.test_manifest is None:
        raise ValueError("--evaluate-test requires --test-manifest")
    stage = dict(STAGES[args.stage])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint, evolution_model = load_tge_checkpoint(checkpoint_path, device)
    evolution_model.eval()
    manifests = {
        "train": args.train_manifest.resolve(),
        "validation": args.validation_manifest.resolve(),
    }
    if args.evaluate_test:
        manifests["test"] = args.test_manifest.resolve()
    evolution = {
        split: collect_frozen_c4_split(
            evolution_model, manifest, split, device, args.batch_size
        )
        for split, manifest in manifests.items()
    }
    feature_cache = args.cache_dir.resolve() / (
        "profile10" if stage["profile"] else "base20"
    )
    records = {}
    for split in manifests:
        records[split] = tuple(
            build_global_static_record(
                args.global_root.resolve(),
                row,
                spectral_dimensions=args.spectral_dim,
                cache_dir=feature_cache,
                include_signed_profile=stage["profile"],
            )
            for row in evolution[split]["rows"]
        )
    scaler = fit_background_feature_scaler(records["train"])
    datasets = {
        split: BackgroundFusionDataset(records[split], evolution[split], scaler)
        for split in manifests
    }
    input_dim = len(STATIC_FEATURE_NAMES) + args.spectral_dim
    if stage["profile"]:
        input_dim += len(SIGNED_CONNECTIVITY_PROFILE_NAMES)
    feature_names = list(STATIC_FEATURE_NAMES) + [
        "signed_laplacian_eigenvector_{}".format(i)
        for i in range(args.spectral_dim)
    ] + (list(SIGNED_CONNECTIVITY_PROFILE_NAMES) if stage["profile"] else [])
    audit = feature_audit(records["train"], feature_names)
    atomic_json(output_dir / "train_feature_audit.json", audit)
    model_config = StaticBackgroundConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        enable_community_residual=stage["community"],
        community_gate_max=0.25,
        community_gate_initial=0.05,
    )
    training_config = BackgroundTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        lambda_rank=args.lambda_rank,
        signed_dropedge_probability=stage["dropedge"],
        lambda_consistency=stage["consistency"],
        checkpoint_ensemble_top_k=stage["top_k"],
        strict_deterministic=not args.non_strict,
    )
    report = train_background_model(
        datasets["train"],
        datasets["validation"],
        datasets.get("test"),
        scaler,
        output_dir,
        device,
        "background_only",
        model_config=model_config,
        training_config=training_config,
    )
    checkpoint_files = [output_dir / "best_checkpoint.pt"] + sorted(
        output_dir.glob("checkpoint_top_*.pt")
    )
    provenance = {
        "artifact_type": "mokse_background_safe_fold_run_v1",
        "stage": args.stage,
        "stage_definition": stage,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "global_root": str(args.global_root.resolve()),
        "feature_cache": str(feature_cache),
        "manifest_sha256": {
            split: file_sha256(path) for split, path in manifests.items()
        },
        "sample_count": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "feature_names": feature_names,
        "train_feature_audit": str((output_dir / "train_feature_audit.json").resolve()),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "static_checkpoint_sha256": {
            path.name: file_sha256(path) for path in checkpoint_files
        },
        "report": report,
        "test_used_for_selection": False,
    }
    atomic_json(output_dir / "run_manifest.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
