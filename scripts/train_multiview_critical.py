"""Train the complete revised S/V/G critical-subgraph channel."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.multiview_critical import (  # noqa: E402
    MultiViewCriticalDataset,
    create_multiview_loader,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.models.multiview_critical import (  # noqa: E402
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
)
from keysubgraph.training.multiview_critical_trainer import (  # noqa: E402
    MultiViewTrainingConfig,
    train_multiview_critical,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def _artifact_dimensions(dataset):
    sample = dataset.samples[0]
    window = next((item for item in sample.hard_windows if item is not None), None)
    if window is None:
        raise ValueError("multi-view artifact cannot establish feature dimensions")
    return {
        "node_feature_dim": int(window.node_features.shape[-1]),
        "edge_feature_dim": int(window.edge_features.shape[-1]),
        "spectral_feature_dim": int(window.spectral_features.shape[-1]),
        "stable_static_dim": int(sample.stable_static.numel()),
        "q_dim": int(dataset.scaler.q_mean.numel()),
        "delta_q_dim": int(dataset.scaler.delta_mean.numel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--lambda-q", type=float, default=0.1)
    parser.add_argument("--lambda-delta-q", type=float, default=0.1)
    parser.add_argument("--static-mode", choices=("stable", "neural", "residual"), default="residual")
    parser.add_argument("--disable-static-attention", action="store_true")
    parser.add_argument("--disable-v", action="store_true")
    parser.add_argument("--disable-g", action="store_true")
    parser.add_argument("--shuffle-correspondence", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Seed before model construction so paired stage ablations share the same
    # initialization for every common parameter, not merely loader/dropout RNG.
    set_reproducible_seed(args.seed)

    max_train = 4 if args.smoke else args.max_train_samples
    max_validation = 4 if args.smoke else args.max_validation_samples
    if max_train is not None and max_train < 1:
        raise ValueError("max train samples must be positive")
    if max_validation is not None and max_validation < 1:
        raise ValueError("max validation samples must be positive")
    train = MultiViewCriticalDataset(
        PROJECT_ROOT, args.train_manifest, args.scaler,
        max_samples=max_train,
    )
    validation = MultiViewCriticalDataset(
        PROJECT_ROOT, args.validation_manifest, args.scaler,
        max_samples=max_validation,
    )
    if train.manifest["protocol_sha256"] != validation.manifest["protocol_sha256"] or train.manifest["selector_checkpoint_sha256"] != validation.manifest["selector_checkpoint_sha256"] or train.manifest["feature_schema_sha256"] != validation.manifest["feature_schema_sha256"]:
        raise ValueError("multi-view train/validation provenance mismatch")
    dimensions = _artifact_dimensions(train)
    if dimensions != _artifact_dimensions(validation):
        raise ValueError("multi-view train/validation feature dimensions differ")
    train_loader = create_multiview_loader(
        train, args.batch_size, args.seed, True, args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = create_multiview_loader(
        validation, args.batch_size, args.seed, False, args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model_config = MultiViewCriticalConfig(
        node_feature_dim=dimensions["node_feature_dim"],
        edge_feature_dim=dimensions["edge_feature_dim"],
        spectral_feature_dim=dimensions["spectral_feature_dim"],
        stable_static_dim=dimensions["stable_static_dim"],
        q_dim=dimensions["q_dim"],
        delta_q_dim=dimensions["delta_q_dim"],
        static_mode=args.static_mode,
        enable_static_attention=not args.disable_static_attention,
        enable_v=not args.disable_v,
        enable_g=not args.disable_g,
        correspondence_mode="shuffled" if args.shuffle_correspondence else "uot",
    )
    model = MultiViewCriticalClassifier(model_config).to(torch.device(args.device))
    training = MultiViewTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=0 if args.smoke else args.early_stopping_patience,
        lambda_q=args.lambda_q,
        lambda_delta_q=args.lambda_delta_q,
        seed=args.seed,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    result = train_multiview_critical(
        model, train_loader, validation_loader, torch.device(args.device),
        args.output_dir, training,
        checkpoint_metadata={
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "validation_manifest_sha256": file_sha256(args.validation_manifest),
            "scaler_sha256": file_sha256(args.scaler),
            "protocol_sha256": train.manifest["protocol_sha256"],
            "selector_checkpoint_sha256": train.manifest["selector_checkpoint_sha256"],
            "feature_schema_sha256": train.manifest["feature_schema_sha256"],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
