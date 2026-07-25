"""Train a linear or 34->16->2 classifier on frozen exact-SGW features."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_feature_dataset import (  # noqa: E402
    DualSGWFeatureDataset,
    create_dual_sgw_feature_loader,
)
from keysubgraph.models.dual_sgw_feature_classifier import (  # noqa: E402
    DUAL_SGW_FEATURE_CLASSIFIERS,
    DualSGWFeatureClassifier,
    DualSGWFeatureClassifierConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (  # noqa: E402
    DualSGWFeatureTrainingConfig,
    train_dual_sgw_feature_classifier,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument(
        "--classifier-type",
        choices=DUAL_SGW_FEATURE_CLASSIFIERS,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _aligned_provenance(train_dataset, validation_dataset, protocol_sha):
    if train_dataset.split != "train":
        raise ValueError("training manifest is not the train partition")
    if validation_dataset.split != "validation":
        raise ValueError(
            "validation manifest is not the validation partition"
        )
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if train_dataset.manifest[key] != validation_dataset.manifest[key]:
            raise ValueError("feature manifests are not provenance-aligned")
    if train_dataset.manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("feature manifest protocol hash mismatch")
    if (
        set(train_dataset.sample_keys)
        & set(validation_dataset.sample_keys)
    ):
        raise ValueError("feature train and validation samples overlap")


def main():
    args = parse_args()
    validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    train_dataset = DualSGWFeatureDataset(
        args.train_manifest, args.scaler
    )
    validation_dataset = DualSGWFeatureDataset(
        args.validation_manifest, args.scaler
    )
    _aligned_provenance(train_dataset, validation_dataset, protocol_sha)
    device = _device(args.device)
    set_reproducible_seed(args.seed)
    model = DualSGWFeatureClassifier(
        DualSGWFeatureClassifierConfig(
            classifier_type=args.classifier_type
        )
    )
    train_loader = create_dual_sgw_feature_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_dual_sgw_feature_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    provenance = {
        "protocol_sha256": protocol_sha,
        "selector_checkpoint_sha256": train_dataset.manifest[
            "selector_checkpoint_sha256"
        ],
        "selection_mode": train_dataset.manifest["selection_mode"],
        "selection_seed": int(train_dataset.manifest["selection_seed"]),
        "sgw_scaler_sha256": file_sha256(args.scaler),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
    }
    result = train_dual_sgw_feature_classifier(
        model,
        train_loader,
        validation_loader,
        train_dataset.labels,
        device,
        DualSGWFeatureTrainingConfig(
            epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            max_train_batches=1 if args.smoke else None,
            max_validation_batches=1 if args.smoke else None,
        ),
        args.output_dir,
        provenance,
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update(
        {
            "device": str(device),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    )
    print(
        json.dumps(
            printable, ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
