"""Train one Hard-STSE-Temporal-SGW variant on a frozen data protocol."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.models.hard_stse_loss import HardSTSELossConfig  # noqa: E402
from keysubgraph.models.hard_stse_temporal_sgw import (  # noqa: E402
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import (  # noqa: E402
    HARD_STSE_VARIANTS,
    HardSelectionSchedule,
    HardSTSEConfig,
)
from keysubgraph.training import (  # noqa: E402
    HardSTSETrainingConfig,
    set_reproducible_seed,
    train_hard_stse,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument("--variant", choices=HARD_STSE_VARIANTS, required=True)
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
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--start-node-ratio", type=float, default=0.90)
    parser.add_argument("--start-edge-ratio", type=float, default=0.80)
    parser.add_argument("--target-node-ratio", type=float, default=0.50)
    parser.add_argument("--target-edge-ratio", type=float, default=0.30)
    parser.add_argument("--budget-weight-max", type=float, default=0.10)
    parser.add_argument("--laplacian-weight-max", type=float, default=0.05)
    parser.add_argument("--gw-proxy-weight-max", type=float, default=0.02)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _variant_settings(variant):
    return {
        "M0": ("full", False),
        "M1": ("random", False),
        "M2": ("learned", False),
        "M3": ("learned", True),
    }[variant]


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("Hard-STSE classification requires strict_theory data")
    if protocol.get("experiment_mode") == "all_samples_exploratory":
        raise ValueError("exploratory all-sample data cannot select a classifier")
    paths = protocol["paths"]
    dataset_arguments = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    train_dataset = GraphSequenceDataset(
        *dataset_arguments,
        "train",
        protocol["edge_presence_threshold"]
    )
    validation_dataset = GraphSequenceDataset(
        *dataset_arguments,
        "validation",
        protocol["edge_presence_threshold"]
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_loader = create_data_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_data_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    schedule = HardSelectionSchedule(
        start_node_ratio=args.start_node_ratio,
        start_edge_ratio=args.start_edge_ratio,
        target_node_ratio=args.target_node_ratio,
        target_edge_ratio=args.target_edge_ratio,
    )
    selection_mode, use_sgw = _variant_settings(args.variant)
    set_reproducible_seed(args.seed)
    model = HardSTSETemporalSGWClassifier(
        HardSTSEConfig(
            variant=args.variant,
            selection_mode=selection_mode,
            use_sgw=use_sgw,
            selection_schedule=schedule,
        )
    )
    loss_config = HardSTSELossConfig(
        budget_weight_max=args.budget_weight_max,
        laplacian_weight_max=args.laplacian_weight_max,
        gw_proxy_weight_max=args.gw_proxy_weight_max,
    )
    training_config = HardSTSETrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        minimum_learning_rate=args.minimum_learning_rate,
        seed=args.seed,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_hard_stse(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=training_config,
        loss_config=loss_config,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol_sha256=file_sha256(args.protocol),
        resume_checkpoint=args.resume,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update(
        {
            "variant": args.variant,
            "device": str(device),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else None
            ),
            "debug_smoke": bool(args.smoke),
        }
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
