"""Train SG0/SG1/SG2 classifiers on frozen SV hard-graph caches."""

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

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.sv_signed_gin_dataset import (  # noqa: E402
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SV_SIGNED_GIN_MESSAGE_MODES,
    SV_SIGNED_GIN_POOLING_MODES,
    SV_SIGNED_GIN_VARIANTS,
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    SVSignedGINTrainingConfig,
    train_sv_signed_gin_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=SV_SIGNED_GIN_VARIANTS, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=15
    )
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "composite_auc"),
        default="composite_auc",
    )
    parser.add_argument("--gin-hidden-dim", type=int, default=64)
    parser.add_argument("--gin-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--message-mode",
        choices=SV_SIGNED_GIN_MESSAGE_MODES,
        default="signed_weighted",
    )
    parser.add_argument(
        "--pooling",
        choices=SV_SIGNED_GIN_POOLING_MODES,
        default="attention",
    )
    parser.add_argument("--gin-residual", action="store_true")
    parser.add_argument(
        "--gin-jumping-knowledge", action="store_true"
    )
    parser.add_argument(
        "--gin-compact-readout", action="store_true"
    )
    parser.add_argument(
        "--gin-batch-normalization", action="store_true"
    )
    parser.add_argument(
        "--auxiliary-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def _balanced_limit(dataset, count):
    if count is None:
        return
    if count < 2:
        raise ValueError("SV overfit sample count must be at least two")
    by_class = {
        label: [
            index
            for index, value in enumerate(dataset.labels)
            if value == label
        ]
        for label in (0, 1)
    }
    left = count // 2
    right = count - left
    if len(by_class[0]) < left or len(by_class[1]) < right:
        raise ValueError("SV overfit subset cannot contain both classes")
    indices = by_class[0][:left] + by_class[1][:right]
    dataset.samples = [dataset.samples[index] for index in indices]
    dataset.sites = [dataset.sites[index] for index in indices]
    dataset.subject_ids = [
        dataset.subject_ids[index] for index in indices
    ]


def main():
    args = parse_args()
    if args.smoke and args.overfit_samples is not None:
        raise ValueError("SV smoke and overfit modes are mutually exclusive")
    validation_manifest = (
        args.train_manifest
        if args.overfit_samples is not None
        else args.validation_manifest
    )
    train = SVSignedGINDataset(args.train_manifest, args.scaler)
    validation = SVSignedGINDataset(
        validation_manifest, args.scaler
    )
    if train.split != "train":
        raise ValueError("SV training manifest must be train")
    if args.overfit_samples is None and validation.split != "validation":
        raise ValueError("SV validation manifest must be validation")
    if args.overfit_samples is not None:
        _balanced_limit(train, args.overfit_samples)
        _balanced_limit(validation, args.overfit_samples)
    elif args.smoke:
        _balanced_limit(train, min(4, len(train)))

    train_loader = create_sv_signed_gin_loader(
        train,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = create_sv_signed_gin_loader(
        validation,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SVSignedGINClassifier(
        SVSignedGINConfig(
            variant=args.variant,
            gin_hidden_dim=args.gin_hidden_dim,
            gin_layers=args.gin_layers,
            dropout=args.dropout,
            message_mode=args.message_mode,
            pooling=args.pooling,
            gin_residual=args.gin_residual,
            gin_jumping_knowledge=args.gin_jumping_knowledge,
            gin_compact_readout=args.gin_compact_readout,
            gin_batch_normalization=args.gin_batch_normalization,
        )
    )
    epochs = 1 if args.smoke else args.epochs
    patience = (
        0
        if args.disable_early_stopping
        else args.early_stopping_patience
    )
    selection_metric = (
        "roc_auc"
        if args.smoke or args.overfit_samples is not None
        else args.selection_metric
    )
    training_config = SVSignedGINTrainingConfig(
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        early_stopping_patience=patience,
        selection_metric=selection_metric,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    manifest = train.manifest
    provenance = {
        "protocol_sha256": manifest["protocol_sha256"],
        "selector_checkpoint_sha256": manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": manifest["selection_mode"],
        "selection_seed": int(manifest["selection_seed"]),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(
            validation_manifest
        ),
        "scaler_sha256": file_sha256(args.scaler),
    }
    result = train_sv_signed_gin_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=train.labels,
        device=torch.device(args.device),
        config=training_config,
        output_dir=args.output_dir,
        provenance=provenance,
    )
    result.update(
        {
            "device": args.device,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "smoke": bool(args.smoke),
            "overfit_samples": args.overfit_samples,
            "auxiliary_loss_weight": args.auxiliary_loss_weight,
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
