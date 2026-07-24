"""Train one document-specified Exact-STSE coordinate-ablation variant."""

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
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.exact_stse import (  # noqa: E402
    ExactSTSEClassifier,
    ExactSTSEConfig,
)
from keysubgraph.training import (  # noqa: E402
    ExactSTSETrainingConfig,
    set_reproducible_seed,
    train_exact_stse,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_coords.json",
    )
    parser.add_argument(
        "--variant",
        choices=("exact_stse", "exact_stse_no_coord"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _validate_protocol_contract(protocol, variant):
    protocol_name = protocol.get("protocol_name")
    if variant == "exact_stse":
        if protocol_name != "exact_stse_coordinate_ablation":
            raise ValueError(
                "coordinate Exact-STSE requires its frozen 307-sample protocol"
            )
        return True
    if protocol_name not in (
        "exact_stse_coordinate_ablation",
        "exact_stse_no_coord_full_cohort",
    ):
        raise ValueError(
            "NoCoord requires an explicit Exact-STSE reproduction protocol"
        )
    return False


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    use_coordinates = _validate_protocol_contract(protocol, args.variant)
    paths = protocol["paths"]
    dataset_args = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    train_dataset = ExactSTSEDataset(
        *dataset_args,
        "train",
        protocol["edge_presence_threshold"],
        require_coordinates=use_coordinates,
    )
    validation_dataset = ExactSTSEDataset(
        *dataset_args,
        "validation",
        protocol["edge_presence_threshold"],
        require_coordinates=use_coordinates,
    )
    device = _device(args.device)
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
    set_reproducible_seed(args.seed)
    model = ExactSTSEClassifier(
        ExactSTSEConfig(
            use_coordinates=use_coordinates,
        )
    )
    model.reset_parameters_with_seed(args.seed)
    training_config = ExactSTSETrainingConfig(
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
    result = train_exact_stse(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=training_config,
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
            "variant": model.config.model_variant,
            "input_dim": model.config.input_dim,
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
            "protocol_name": protocol["protocol_name"],
            "cohort_sample_count": int(protocol["sample_count"]),
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "coordinates_loaded": bool(use_coordinates),
        }
    )
    summary_path = args.output_dir.resolve() / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            printable, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
    printable["run_summary"] = str(summary_path)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
