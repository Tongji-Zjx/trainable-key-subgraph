"""Train the differentiable soft_graph baseline using the frozen split."""

from __future__ import absolute_import, print_function

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
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)
from keysubgraph.training.trainer import (  # noqa: E402
    TrainingConfig,
    set_reproducible_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "training" / "soft_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-node-ratio", type=float, default=0.30)
    parser.add_argument("--target-edge-ratio", type=float, default=0.30)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--selection-metric", choices=("roc_auc", "balanced_accuracy", "loss"), default="roc_auc")
    parser.add_argument("--node-score-hidden", type=int, default=32)
    parser.add_argument("--edge-score-hidden", type=int, default=32)
    parser.add_argument("--graph-hidden", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--classifier-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process only one train and validation batch; never use for formal results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = args.output_dir.resolve() / "history.json"
    if history_path.exists() and args.resume is None:
        raise FileExistsError(
            "training output already exists; pass --resume instead of silently overwriting"
        )
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    train_dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "train",
        protocol["edge_presence_threshold"],
    )
    validation_dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "validation",
        protocol["edge_presence_threshold"],
    )
    train_loader = create_data_loader(
        train_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    validation_loader = create_data_loader(
        validation_dataset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    set_reproducible_seed(args.seed)
    model = SoftGraphClassifier(
        SoftExtractorConfig(
            node_score_hidden_dim=args.node_score_hidden,
            edge_score_hidden_dim=args.edge_score_hidden,
            graph_hidden_dim=args.graph_hidden,
            graph_layers=args.graph_layers,
            classifier_hidden_dim=args.classifier_hidden,
            dropout=args.dropout,
        )
    )
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_node_ratio=args.target_node_ratio,
        target_edge_ratio=args.target_edge_ratio,
        budget_weight=args.budget_weight,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
        selection_metric=args.selection_metric,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        config=config,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol=protocol,
        resume_checkpoint=args.resume,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    printable = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in result.items()
    }
    printable["device"] = str(device)
    printable["debug_smoke"] = args.smoke
    printable["elapsed_seconds"] = elapsed_seconds
    printable["cuda_peak_memory_mib"] = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else None
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
