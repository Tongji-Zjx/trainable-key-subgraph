"""Train D4 fusion from validated STSE and cached learned-hard SGW."""

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
from keysubgraph.data.dual_sgw_manifest import read_dual_sgw_manifest  # noqa: E402
from keysubgraph.data.dual_sgw_scaler import load_dual_sgw_standardizer  # noqa: E402
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
from keysubgraph.training.exact_stse_trainer import (  # noqa: E402
    load_exact_stse_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stse-checkpoint", type=Path, required=True)
    parser.add_argument("--sgw-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--fine-tune-stse", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    train_payload, _, train_lookup = read_dual_sgw_manifest(
        args.train_manifest
    )
    validation_payload, _, validation_lookup = read_dual_sgw_manifest(
        args.validation_manifest
    )
    if train_payload["split"] != "train" or validation_payload["split"] != (
        "validation"
    ):
        raise ValueError("fusion manifests use the wrong partitions")
    for key in (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    ):
        if train_payload[key] != validation_payload[key]:
            raise ValueError("fusion manifests are not aligned")
    if train_payload["selection_mode"] != "learned":
        raise ValueError("D4 requires learned-hard SGW features")
    scaler = load_dual_sgw_standardizer(args.scaler)
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256
        != train_payload["selector_checkpoint_sha256"]
        or scaler.selection_mode != "learned"
        or scaler.selection_seed != train_payload["selection_seed"]
    ):
        raise ValueError("D4 SGW scaler provenance mismatch")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = DualSTSEHardSGWClassifier().to(device)
    load_dual_checkpoint(
        args.sgw_checkpoint,
        model,
        device,
        expected_stage="sgw_classifier",
        expected_protocol_sha256=protocol_sha,
    )
    load_exact_stse_checkpoint(
        args.stse_checkpoint,
        model.stse_channel.model,
        device,
        expected_protocol_sha256=protocol_sha,
    )
    model.set_sgw_standardizer(scaler)
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
    provenance = {
        "stse_checkpoint_sha256": file_sha256(args.stse_checkpoint),
        "selector_checkpoint_sha256": train_payload[
            "selector_checkpoint_sha256"
        ],
        "sgw_scaler_sha256": file_sha256(args.scaler),
        "sgw_classifier_checkpoint_sha256": file_sha256(
            args.sgw_checkpoint
        ),
    }
    result = train_dual_stage(
        model,
        train_loader,
        validation_loader,
        [item.label for item in train_dataset.assignments],
        device,
        DualTrainingConfig(
            stage="fusion",
            epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            fine_tune_stse=args.fine_tune_stse,
            max_train_batches=1 if args.smoke else None,
            max_validation_batches=1 if args.smoke else None,
        ),
        DualSTSEHardSGWLossConfig(),
        args.output_dir,
        protocol_sha,
        provenance,
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
