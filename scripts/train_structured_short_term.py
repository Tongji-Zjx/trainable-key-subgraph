"""Train the complete coordinate-free, community-structured short-term branch."""

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

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.features.structured_short_term_features import (  # noqa: E402
    StructuredShortTermStandardizer,
)
from keysubgraph.features.paper_short_term_pst import (  # noqa: E402
    PaperShortTermCommunityFrequency,
)
from keysubgraph.models.structured_short_term import (  # noqa: E402
    PAPER_ALIGNED_VARIANT,
    PAPER_ALIGNED_PST_VARIANT,
    STRUCTURED_SAFE_VARIANT,
    StructuredShortTermClassifier,
    StructuredShortTermConfig,
    is_paper_aligned_variant,
    variant_uses_pst,
)
from keysubgraph.training import (  # noqa: E402
    StructuredShortTermTrainingConfig,
    set_reproducible_seed,
    train_structured_short_term,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--standardizer", type=Path, required=True)
    parser.add_argument("--community-frequency", type=Path)
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
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "balanced_accuracy", "loss"),
        default="roc_auc",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--node-ffn-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument(
        "--model-variant",
        choices=(
            STRUCTURED_SAFE_VARIANT,
            PAPER_ALIGNED_VARIANT,
            PAPER_ALIGNED_PST_VARIANT,
        ),
        default=STRUCTURED_SAFE_VARIANT,
    )
    parser.add_argument("--community-vocab-size", type=int, default=128)
    parser.add_argument("--community-embedding-dim", type=int, default=16)
    parser.add_argument("--statistics-embedding-dim", type=int, default=16)
    parser.add_argument("--pst-anomaly-epsilon", type=float, default=1.0e-12)
    parser.add_argument("--classifier-hidden-1", type=int, default=64)
    parser.add_argument("--classifier-hidden-2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    standardizer = StructuredShortTermStandardizer.load(args.standardizer)
    protocol_hash = file_sha256(args.protocol)
    standardizer_hash = file_sha256(args.standardizer)
    uses_pst = variant_uses_pst(args.model_variant)
    if uses_pst and args.community_frequency is None:
        raise ValueError("p_ST variant requires --community-frequency")
    if not uses_pst and args.community_frequency is not None:
        raise ValueError(
            "--community-frequency is only valid for the p_ST variant"
        )
    community_frequency = (
        PaperShortTermCommunityFrequency.load(args.community_frequency)
        if uses_pst
        else None
    )
    community_frequency_hash = (
        file_sha256(args.community_frequency) if uses_pst else None
    )
    if standardizer.protocol_sha256 != protocol_hash:
        raise ValueError("standardizer was fitted for a different protocol")
    if (
        standardizer.edge_presence_threshold
        != float(protocol["edge_presence_threshold"])
    ):
        raise ValueError("standardizer edge threshold does not match protocol")

    paths = protocol["paths"]
    dataset_args = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    node_name_policy = protocol_node_name_policy(protocol)
    train_dataset = GraphSequenceDataset(
        *dataset_args,
        "train",
        protocol["edge_presence_threshold"],
        node_name_policy=node_name_policy,
    )
    validation_dataset = GraphSequenceDataset(
        *dataset_args,
        "validation",
        protocol["edge_presence_threshold"],
        node_name_policy=node_name_policy,
    )
    if standardizer.train_sample_count != len(train_dataset):
        raise ValueError(
            "standardizer does not cover the complete frozen training split"
        )
    if community_frequency is not None:
        splits_hash = file_sha256(PROJECT_ROOT / paths["splits_csv"])
        if community_frequency.protocol_sha256 != protocol_hash:
            raise ValueError("community frequency protocol hash mismatch")
        if community_frequency.train_manifest_sha256 != splits_hash:
            raise ValueError("community frequency split-manifest mismatch")
        if community_frequency.train_sample_count != len(train_dataset):
            raise ValueError("community frequency train sample mismatch")
    device = _device(args.device)
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
    set_reproducible_seed(args.seed)
    model = StructuredShortTermClassifier(
        StructuredShortTermConfig(
            hidden_dim=args.hidden_dim,
            node_ffn_dim=args.node_ffn_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_ffn_dim=args.transformer_ffn_dim,
            memory_slots=args.memory_slots,
            statistics_embedding_dim=args.statistics_embedding_dim,
            classifier_hidden_dims=(
                args.classifier_hidden_1,
                args.classifier_hidden_2,
            ),
            dropout=args.dropout,
            variant=args.model_variant,
            community_vocab_size=args.community_vocab_size,
            community_embedding_dim=args.community_embedding_dim,
            pst_anomaly_epsilon=args.pst_anomaly_epsilon,
        ),
        standardizer,
        community_frequency=community_frequency,
    )
    config = StructuredShortTermTrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        minimum_learning_rate=args.minimum_learning_rate,
        selection_metric=args.selection_metric,
        seed=args.seed,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_structured_short_term(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=[item.label for item in train_dataset.assignments],
        device=device,
        training_config=config,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol_sha256=protocol_hash,
        standardizer_path=args.standardizer,
        standardizer_sha256=standardizer_hash,
        community_frequency_path=args.community_frequency,
        community_frequency_sha256=community_frequency_hash,
        resume_checkpoint=args.resume,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    summary.update(
        {
            "device": str(device),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "uses_coordinates": False,
            "uses_community_embedding": (
                is_paper_aligned_variant(args.model_variant)
            ),
            "uses_sequence_statistics": (
                args.model_variant == STRUCTURED_SAFE_VARIANT or uses_pst
            ),
            "uses_pst": uses_pst,
            "community_frequency": (
                None
                if args.community_frequency is None
                else str(args.community_frequency.resolve())
            ),
            "model_variant": args.model_variant,
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else None
            ),
            "debug_smoke": bool(args.smoke),
        }
    )
    summary_path = args.output_dir.resolve() / "run_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
