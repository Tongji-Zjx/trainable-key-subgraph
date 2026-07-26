"""Train one frozen-base D3-B temporal variant (T1--T4)."""

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
from keysubgraph.data.dual_temporal_dataset import (  # noqa: E402
    DualTemporalDataset,
    create_dual_temporal_loader,
)
from keysubgraph.models.dual_variation_temporal import (  # noqa: E402
    DUAL_TEMPORAL_VARIANTS,
    DualVariationTemporalClassifier,
    DualVariationTemporalConfig,
)
from keysubgraph.training.dual_variation_temporal_trainer import (  # noqa: E402
    DualTemporalTrainingConfig,
    train_dual_temporal_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--temporal-scaler", type=Path, required=True)
    parser.add_argument("--variant", choices=DUAL_TEMPORAL_VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--temporal-auxiliary-weight", type=float, default=0.30)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    train_dataset = DualTemporalDataset(
        args.train_manifest, args.temporal_scaler
    )
    validation_dataset = DualTemporalDataset(
        args.validation_manifest, args.temporal_scaler
    )
    if train_dataset.split != "train" or validation_dataset.split != "validation":
        raise ValueError("temporal train/validation manifests use wrong splits")
    provenance_keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "exact_head_checkpoint_sha256",
        "sgw_scaler_sha256",
        "selection_mode",
        "selection_seed",
    )
    if any(
        train_dataset.manifest[key] != validation_dataset.manifest[key]
        for key in provenance_keys
    ):
        raise ValueError("temporal train/validation provenance mismatch")
    train_loader = create_dual_temporal_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=_device(args.device).type == "cuda",
    )
    validation_loader = create_dual_temporal_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=_device(args.device).type == "cuda",
    )
    model = DualVariationTemporalClassifier(
        DualVariationTemporalConfig(
            variant=args.variant,
            dropout=args.dropout,
        )
    )
    config = DualTemporalTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=(
            0 if args.smoke else args.early_stopping_patience
        ),
        temporal_auxiliary_weight=args.temporal_auxiliary_weight,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    provenance = {
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
        "temporal_scaler_sha256": file_sha256(args.temporal_scaler),
        **{
            key: train_dataset.manifest[key]
            for key in provenance_keys
        },
        "train_exact_manifest_sha256": (
            train_dataset.manifest["exact_manifest_sha256"]
        ),
        "validation_exact_manifest_sha256": (
            validation_dataset.manifest["exact_manifest_sha256"]
        ),
    }
    result = train_dual_temporal_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=train_dataset.labels,
        device=_device(args.device),
        config=config,
        output_dir=args.output_dir,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
