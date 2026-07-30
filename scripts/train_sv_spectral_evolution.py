"""Train the neural spectral-evolution residual on a frozen S anchor."""

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
from keysubgraph.data.sv_spectral_evolution import (  # noqa: E402
    SVSpectralEvolutionDataset,
    create_sv_spectral_evolution_loader,
)
from keysubgraph.models.sv_signed_gin import (  # noqa: E402
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.models.sv_spectral_evolution import (  # noqa: E402
    SVSpectralEvolutionClassifier,
    SVSpectralEvolutionConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (  # noqa: E402
    load_sv_signed_gin_checkpoint,
)
from keysubgraph.training.sv_spectral_evolution_trainer import (  # noqa: E402
    SVSpectralEvolutionTrainingConfig,
    train_sv_spectral_evolution_classifier,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--static-scaler", type=Path, required=True)
    parser.add_argument("--transition-scaler", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "composite_auc"),
        default="composite_auc",
    )
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _trusted_load(path, device):
    try:
        return torch.load(
            str(Path(path).resolve()),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def _load_anchor(path, device):
    raw = _trusted_load(path, device)
    anchor = SVSignedGINClassifier(
        SVSignedGINConfig(**raw["model_config"])
    ).to(device)
    load_sv_signed_gin_checkpoint(path, anchor, device)
    if anchor.config.variant != "static_spectral_only":
        raise ValueError("anchor checkpoint is not static_spectral_only")
    return anchor, raw


def main():
    args = parse_args()
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    anchor, anchor_payload = _load_anchor(
        args.anchor_checkpoint, device
    )
    train = SVSpectralEvolutionDataset(
        args.train_manifest,
        args.static_scaler,
        args.transition_scaler,
        max_samples=4 if args.smoke else None,
    )
    validation = SVSpectralEvolutionDataset(
        args.validation_manifest,
        args.static_scaler,
        args.transition_scaler,
        max_samples=4 if args.smoke else None,
    )
    if train.split != "train" or validation.split != "validation":
        raise ValueError("spectral evolution split mismatch")
    train_loader = create_sv_spectral_evolution_loader(
        train,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = create_sv_spectral_evolution_loader(
        validation,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = SVSpectralEvolutionClassifier(
        anchor,
        SVSpectralEvolutionConfig(dropout=args.dropout),
    )
    config = SVSpectralEvolutionTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=(
            0 if args.smoke else args.early_stopping_patience
        ),
        selection_metric=(
            "roc_auc" if args.smoke else args.selection_metric
        ),
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        seed=args.seed,
        max_train_batches=2 if args.smoke else None,
        max_validation_batches=2 if args.smoke else None,
    )
    provenance = {
        "protocol_sha256": train.manifest["protocol_sha256"],
        "selector_checkpoint_sha256": train.manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": train.manifest["selection_mode"],
        "selection_seed": int(train.manifest["selection_seed"]),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
        "static_scaler_sha256": file_sha256(args.static_scaler),
        "transition_scaler_sha256": file_sha256(
            args.transition_scaler
        ),
        "anchor_checkpoint_sha256": file_sha256(
            args.anchor_checkpoint
        ),
        "anchor_best_epoch": int(anchor_payload["best_epoch"]),
    }
    result = train_sv_spectral_evolution_classifier(
        model,
        train_loader,
        validation_loader,
        train.labels,
        device,
        config,
        args.output_dir,
        provenance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
