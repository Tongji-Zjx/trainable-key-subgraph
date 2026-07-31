"""Train one Stage-1 N0--N4 theory-guided neural variant."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.theory_neural_dataset import (  # noqa: E402
    TheoryNeuralDataset,
    create_theory_neural_loader,
)
from keysubgraph.models.theory_guided_neural import (  # noqa: E402
    THEORY_NEURAL_VARIANTS,
    TheoryGuidedNeuralClassifier,
    TheoryNeuralConfig,
)
from keysubgraph.training.theory_guided_neural_trainer import (  # noqa: E402
    TheoryNeuralTrainingConfig,
    train_theory_neural_classifier,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--variant", choices=THEORY_NEURAL_VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--selection-metric", choices=("roc_auc", "composite_auc"),
        default="composite_auc"
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--quantile-loss-weight", type=float, default=0.05)
    parser.add_argument("--transition-loss-weight", type=float, default=0.05)
    parser.add_argument("--center-loss-weight", type=float, default=0.02)
    parser.add_argument("--auxiliary-warmup-epochs", type=int, default=5)
    parser.add_argument("--auxiliary-ramp-epochs", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    # Seed before model construction so nested N0--N4 comparisons share the
    # same initialization for every common parameter tensor.
    set_reproducible_seed(args.seed)
    if args.batch_size * args.gradient_accumulation_steps < 8 and not args.smoke:
        raise ValueError("formal Stage-1 training requires effective batch >= 8")
    train = TheoryNeuralDataset(
        PROJECT_ROOT, args.train_manifest, args.scaler,
        max_samples=8 if args.smoke else None
    )
    validation = TheoryNeuralDataset(
        PROJECT_ROOT, args.validation_manifest, args.scaler,
        max_samples=8 if args.smoke else None
    )
    loaders = (
        create_theory_neural_loader(
            train, args.batch_size, args.seed, True, args.num_workers,
            pin_memory=args.device.startswith("cuda")
        ),
        create_theory_neural_loader(
            validation, args.batch_size, args.seed, False, args.num_workers,
            pin_memory=args.device.startswith("cuda")
        ),
    )
    provenance = {
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.validation_manifest),
        "scaler_sha256": file_sha256(args.scaler),
        "protocol_sha256": train.manifest["protocol_sha256"],
        "selector_checkpoint_sha256": train.manifest[
            "selector_checkpoint_sha256"
        ],
        "feature_schema_sha256": train.manifest["feature_schema_sha256"],
    }
    if any(
        train.manifest[name] != validation.manifest[name]
        for name in (
            "protocol_sha256", "selector_checkpoint_sha256",
            "feature_schema_sha256"
        )
    ):
        raise ValueError("Stage-1 train/validation provenance mismatch")
    model = TheoryGuidedNeuralClassifier(
        TheoryNeuralConfig(variant=args.variant, dropout=args.dropout)
    )
    config = TheoryNeuralTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=0 if args.smoke else args.early_stopping_patience,
        selection_metric=args.selection_metric,
        quantile_loss_weight=args.quantile_loss_weight,
        transition_loss_weight=args.transition_loss_weight,
        center_loss_weight=args.center_loss_weight,
        auxiliary_warmup_epochs=args.auxiliary_warmup_epochs,
        auxiliary_ramp_epochs=args.auxiliary_ramp_epochs,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    result = train_theory_neural_classifier(
        model, loaders[0], loaders[1], train.labels,
        torch.device(args.device), config, args.output_dir, provenance
    )
    print("best epoch:", result["best_epoch"])
    print("validation thresholds:", result["validation_thresholds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
