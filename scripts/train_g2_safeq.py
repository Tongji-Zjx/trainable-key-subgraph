"""Train the cheap residual head and fit leakage-safe SafeQ mixing."""

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
from keysubgraph.data.g2_safeq import G2SafeQDataset  # noqa: E402
from keysubgraph.models.g2_safeq import (  # noqa: E402
    G2SafeQConfig,
    G2SafeQResidual,
)
from keysubgraph.training.g2_safeq_trainer import (  # noqa: E402
    G2SafeQTrainingConfig,
    create_g2_safeq_loader,
    train_g2_safeq,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--residual-hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.005)
    parser.add_argument("--maximum-component-drop", type=float, default=0.002)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train = G2SafeQDataset(args.train_manifest)
    validation = G2SafeQDataset(args.validation_manifest)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("SafeQ training requires train/validation manifests")
    provenance_fields = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
        "g2_checkpoint_sha256",
        "g2_scaler_sha256",
        "g2_spectral_scaler_sha256",
        "g2_variant",
        "transition_hidden_dim",
        "summary_dim",
        "frozen_g2",
        "train_only_scalers",
    )
    if any(
        train.manifest.get(name) != validation.manifest.get(name)
        for name in provenance_fields
    ):
        raise ValueError("SafeQ train/validation provenance mismatch")
    if train.summary_dim != validation.summary_dim:
        raise ValueError("SafeQ train/validation dimensions differ")
    transition_dim = int(train.manifest["transition_hidden_dim"])
    if train.summary_dim != 2 * transition_dim:
        raise ValueError("SafeQ summary is not mean-plus-std")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed)
    model = G2SafeQResidual(
        G2SafeQConfig(
            transition_hidden_dim=transition_dim,
            residual_hidden_dim=args.residual_hidden_dim,
            dropout=args.dropout,
        )
    )
    config = G2SafeQTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=(
            0 if args.smoke else args.early_stopping_patience
        ),
        minimum_epochs=0 if args.smoke else args.minimum_epochs,
        seed=args.seed,
        minimum_composite_gain=args.minimum_composite_gain,
        maximum_component_drop=args.maximum_component_drop,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    train_loader = create_g2_safeq_loader(
        train,
        args.batch_size,
        args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_g2_safeq_loader(
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
            "optimizer_fit_split": "train",
            "checkpoint_selection_split": "validation",
            "outer_test_used": False,
        }
    )
    result = train_g2_safeq(
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
