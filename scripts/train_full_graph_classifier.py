"""Train one controlled full-graph encoder on the frozen strict-theory split."""

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
from keysubgraph.models import (  # noqa: E402
    FULL_GRAPH_ENCODERS,
    FullGraphClassifierConfig,
    FullGraphSequenceClassifier,
)
from keysubgraph.training import (  # noqa: E402
    FullGraphTrainingConfig,
    set_reproducible_seed,
    train_full_graph_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument("--encoder-type", choices=FULL_GRAPH_ENCODERS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--baseline-dropout", type=float, default=0.2)
    parser.add_argument("--gated-gnn-dropout", type=float, default=0.15)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("full-graph comparison requires the strict_theory protocol")
    if protocol.get("experiment_mode") == "all_samples_exploratory":
        raise ValueError("all-sample exploratory data cannot estimate validation performance")
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
        pin_memory=device.type == "cuda",
    )
    set_reproducible_seed(args.seed)
    model = FullGraphSequenceClassifier(
        FullGraphClassifierConfig(
            encoder_type=args.encoder_type,
            baseline_dropout=args.baseline_dropout,
            gated_gnn_dropout=args.gated_gnn_dropout,
            classifier_dropout=args.classifier_dropout,
        )
    )
    training_config = FullGraphTrainingConfig(
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
    result = train_full_graph_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=training_config,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol_sha256=file_sha256(args.protocol),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update(
        {
            "encoder_type": args.encoder_type,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds_total": time.perf_counter() - started,
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
