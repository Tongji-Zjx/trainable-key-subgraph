"""Train the differentiable selector-proxy stage of Dual-STSE-HardSGW."""

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

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
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
    train_dual_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_no_coord_full.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "exact_stse_no_coord_full_cohort":
        raise ValueError("dual selector requires the frozen 938-sample protocol")
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
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
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
    result = train_dual_stage(
        model=DualSTSEHardSGWClassifier(),
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=DualTrainingConfig(
            stage="selector_proxy",
            epochs=1 if args.smoke else args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            max_train_batches=1 if args.smoke else None,
            max_validation_batches=1 if args.smoke else None,
        ),
        loss_config=DualSTSEHardSGWLossConfig(),
        output_dir=args.output_dir,
        protocol_sha256=file_sha256(args.protocol),
        provenance={
            "stse_checkpoint_sha256": "not_used_in_selector_stage",
            "selector_checkpoint_sha256": "trained_by_this_stage",
            "sgw_scaler_sha256": "not_applicable",
        },
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

