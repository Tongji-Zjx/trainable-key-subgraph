"""Train D1/D2/D3 exact-SGW auxiliary classifier from cached features."""

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
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (  # noqa: E402
    load_dual_sgw_standardizer,
)
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (  # noqa: E402
    DualSTSEHardSGWLossConfig,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    DualTrainingConfig,
    load_dual_checkpoint,
    train_dual_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    train_payload, train_records, train_lookup = read_dual_sgw_manifest(
        args.train_manifest
    )
    validation_payload, _, validation_lookup = read_dual_sgw_manifest(
        args.validation_manifest
    )
    if train_payload["split"] != "train" or validation_payload["split"] != (
        "validation"
    ):
        raise ValueError("dual SGW manifests use the wrong partitions")
    for key in (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    ):
        if train_payload[key] != validation_payload[key]:
            raise ValueError("dual SGW manifests are not provenance-aligned")
    if train_payload["protocol_sha256"] != protocol_sha:
        raise ValueError("dual SGW manifest protocol mismatch")
    scaler = load_dual_sgw_standardizer(args.scaler)
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256
        != train_payload["selector_checkpoint_sha256"]
        or scaler.selection_mode != train_payload["selection_mode"]
        or scaler.selection_seed != train_payload["selection_seed"]
    ):
        raise ValueError("dual SGW scaler provenance mismatch")
    paths = protocol["paths"]
    common = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    train_dataset = ExactSTSEDataset(
        *common,
        "train",
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )
    validation_dataset = ExactSTSEDataset(
        *common,
        "validation",
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )
    if {item.sample_key for item in train_dataset.assignments} != set(
        train_lookup
    ):
        raise ValueError("train SGW cache does not cover the frozen split")
    if {
        item.sample_key for item in validation_dataset.assignments
    } != set(validation_lookup):
        raise ValueError("validation SGW cache does not cover the frozen split")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = DualSTSEHardSGWClassifier().to(device)
    if args.selector_checkpoint is not None:
        selector_sha = file_sha256(args.selector_checkpoint)
        if selector_sha != train_payload["selector_checkpoint_sha256"]:
            raise ValueError("selector checkpoint does not match SGW cache")
        load_dual_checkpoint(
            args.selector_checkpoint,
            model,
            device,
            expected_stage="selector_proxy",
            expected_protocol_sha256=protocol_sha,
        )
    model.set_sgw_standardizer(scaler)
    train_loader = create_exact_stse_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_exact_stse_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    result = train_dual_stage(
        model,
        train_loader,
        validation_loader,
        [item.label for item in train_dataset.assignments],
        device,
        DualTrainingConfig(
            stage="sgw_classifier",
            epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_train_batches=1 if args.smoke else None,
            max_validation_batches=1 if args.smoke else None,
        ),
        DualSTSEHardSGWLossConfig(),
        args.output_dir,
        protocol_sha,
        {
            "stse_checkpoint_sha256": "not_used_in_sgw_stage",
            "selector_checkpoint_sha256": train_payload[
                "selector_checkpoint_sha256"
            ],
            "sgw_scaler_sha256": file_sha256(args.scaler),
        },
        train_lookup,
        validation_lookup,
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
