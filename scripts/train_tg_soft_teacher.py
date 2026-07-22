"""Train the versioned TG-SGW Stage-A soft teacher on a frozen protocol."""

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
from keysubgraph.data.graph_dataset import GraphSequenceDataset, create_data_loader  # noqa: E402
from keysubgraph.models import (  # noqa: E402
    TG_SOFT_TEACHER_ABLATIONS,
    TGSoftTeacher,
    TGSoftTeacherConfig,
    TGSoftTeacherLossConfig,
    tg_soft_teacher_ablation_weights,
)
from keysubgraph.training import (  # noqa: E402
    TGSoftTeacherTrainingConfig,
    set_reproducible_seed,
    train_tg_soft_teacher,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tg_sgw" / "soft_teacher_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--selection-metric", choices=("balanced_accuracy", "roc_auc", "loss"), default="balanced_accuracy")
    parser.add_argument("--target-node-ratio", type=float, default=0.30)
    parser.add_argument("--target-edge-ratio", type=float, default=0.20)
    parser.add_argument("--lambda-budget", type=float, default=0.10)
    parser.add_argument("--lambda-laplacian", type=float, default=0.50)
    parser.add_argument("--lambda-gw-identity", type=float, default=0.10)
    parser.add_argument("--lambda-supcon", type=float, default=0.0)
    parser.add_argument(
        "--ablation",
        choices=("custom",) + TG_SOFT_TEACHER_ABLATIONS,
        default="custom",
        help="fixed nested loss ablation; custom keeps the explicit lambda arguments",
    )
    parser.add_argument("--theory-warmup-epochs", type=int, default=15)
    parser.add_argument("--laplacian-eta", type=float, default=1.0e-3)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("TG-SGW classification requires a strict_theory protocol")
    if protocol.get("experiment_mode") == "all_samples_exploratory":
        raise ValueError("all-sample exploratory data cannot report TG-SGW classification performance")
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
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
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
    # The trainer receives an already constructed module, so seed before
    # construction; seeding only inside the trainer is too late for weights.
    set_reproducible_seed(args.seed)
    model = TGSoftTeacher(
        TGSoftTeacherConfig(
            laplacian_eta=args.laplacian_eta,
            diffusion_time=args.diffusion_time,
        )
    )
    budget_weight = args.lambda_budget
    laplacian_weight = args.lambda_laplacian
    gw_weight = args.lambda_gw_identity
    if args.ablation != "custom":
        budget_weight, laplacian_weight, gw_weight = (
            tg_soft_teacher_ablation_weights(args.ablation)
        )
    loss_config = TGSoftTeacherLossConfig(
        budget_weight=budget_weight,
        laplacian_max_weight=laplacian_weight,
        gw_identity_max_weight=gw_weight,
        supervised_contrastive_weight=args.lambda_supcon,
        target_node_ratio=args.target_node_ratio,
        target_edge_ratio=args.target_edge_ratio,
        theory_warmup_epochs=args.theory_warmup_epochs,
    )
    training_config = TGSoftTeacherTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        selection_metric=args.selection_metric,
        seed=args.seed,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_tg_soft_teacher(
        model,
        train_loader,
        validation_loader,
        [item.label for item in train_dataset.assignments],
        device,
        loss_config,
        training_config,
        args.output_dir,
        args.protocol,
        file_sha256(args.protocol),
        resume_checkpoint=args.resume,
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update(
        {
            "device": str(device),
            "debug_smoke": bool(args.smoke),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda" else None
            ),
            "ablation": args.ablation,
            "loss_weights": {
                "classification": loss_config.classification_weight,
                "budget": loss_config.budget_weight,
                "laplacian": loss_config.laplacian_max_weight,
                "gw_identity": loss_config.gw_identity_max_weight,
                "supervised_contrastive": loss_config.supervised_contrastive_weight,
            },
        }
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
