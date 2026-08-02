"""Train the promoted representation-level F2 residual head."""

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
from keysubgraph.data.svg_short_term_representation_f2 import (  # noqa: E402
    SVGShortTermRepresentationF2Dataset,
)
from keysubgraph.models.svg_short_term_representation_f2 import (  # noqa: E402
    SVGShortTermRepresentationF2,
    SVGShortTermRepresentationF2Config,
)
from keysubgraph.training.svg_short_term_representation_f2_trainer import (  # noqa: E402
    SVGShortTermRepresentationF2TrainingConfig,
    create_svg_short_term_representation_f2_loader,
    train_svg_short_term_representation_f2,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--residual-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--initial-gate", type=float, default=0.01)
    parser.add_argument("--residual-auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--gate-penalty-weight", type=float, default=1.0e-3)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train = SVGShortTermRepresentationF2Dataset(args.train_manifest)
    validation = SVGShortTermRepresentationF2Dataset(
        args.validation_manifest
    )
    if train.split != "train" or validation.split != "validation":
        raise ValueError("representation F2 training split mismatch")
    provenance_fields = (
        "protocol_sha256",
        "short_term_checkpoint_sha256",
        "g2_checkpoint_sha256",
        "g2_scaler_sha256",
        "g2_spectral_scaler_sha256",
        "g2_variant",
    )
    if any(
        train.manifest.get(name) != validation.manifest.get(name)
        for name in provenance_fields
    ):
        raise ValueError("representation F2 train/validation provenance mismatch")
    if (
        train.g2_representation_dim != validation.g2_representation_dim
        or train.short_term_representation_dim
        != validation.short_term_representation_dim
    ):
        raise ValueError("representation F2 train/validation dimension mismatch")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed)
    model = SVGShortTermRepresentationF2(
        SVGShortTermRepresentationF2Config(
            short_term_representation_dim=train.short_term_representation_dim,
            g2_representation_dim=train.g2_representation_dim,
            residual_hidden_dim=args.residual_hidden_dim,
            dropout=args.dropout,
            initial_gate=args.initial_gate,
        )
    )
    config = SVGShortTermRepresentationF2TrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=(0 if args.smoke else args.early_stopping_patience),
        minimum_epochs=(0 if args.smoke else args.minimum_epochs),
        residual_auxiliary_weight=args.residual_auxiliary_weight,
        gate_penalty_weight=args.gate_penalty_weight,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    train_loader = create_svg_short_term_representation_f2_loader(
        train,
        args.batch_size,
        args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_svg_short_term_representation_f2_loader(
        validation,
        args.batch_size,
        args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    provenance = {
        name: train.manifest[name] for name in provenance_fields
    }
    provenance.update(
        {
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "validation_manifest_sha256": file_sha256(
                args.validation_manifest
            ),
        }
    )
    result = train_svg_short_term_representation_f2(
        model,
        train_loader,
        validation_loader,
        train.labels,
        device,
        config,
        args.output_dir,
        provenance,
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
