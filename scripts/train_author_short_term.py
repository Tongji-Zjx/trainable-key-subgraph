"""Train the standalone coordinate-free reproduction of the author branch."""

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
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.features.paper_short_term_pst import (  # noqa: E402
    PaperShortTermCommunityFrequency,
)
from keysubgraph.models.author_short_term import (  # noqa: E402
    AUTHOR_SHORT_TERM_PROFILES,
    AuthorNoCoordinateShortTermClassifier,
    author_short_term_config,
)
from keysubgraph.training.author_short_term_trainer import (  # noqa: E402
    author_short_term_training_config,
    create_author_short_term_evaluation_loader,
    create_author_short_term_train_loader,
    train_author_short_term,
)
from keysubgraph.training.trainer import set_reproducible_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--community-frequency", type=Path, required=True)
    parser.add_argument("--profile", choices=AUTHOR_SHORT_TERM_PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
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
    protocol_hash = file_sha256(args.protocol)
    frequency = PaperShortTermCommunityFrequency.load(args.community_frequency)
    frequency_hash = file_sha256(args.community_frequency)
    paths = protocol["paths"]
    split_path = PROJECT_ROOT / paths["splits_csv"]
    if frequency.protocol_sha256 != protocol_hash:
        raise ValueError("community frequency protocol hash mismatch")
    if frequency.train_manifest_sha256 != file_sha256(split_path):
        raise ValueError("community frequency split hash mismatch")
    dataset_args = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        split_path,
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
    if frequency.train_sample_count != len(train_dataset):
        raise ValueError("community frequency train sample mismatch")
    device = _device(args.device)
    training_config = author_short_term_training_config(
        args.profile,
        epochs=1 if args.smoke else args.epochs,
        seed=args.seed,
    )
    if args.smoke:
        training_config = type(training_config)(
            **dict(
                training_config.__dict__,
                early_stopping_minimum_epochs=0,
                max_train_batches=1,
                max_validation_batches=1,
            )
        )
    set_reproducible_seed(training_config.seed)
    model = AuthorNoCoordinateShortTermClassifier(
        author_short_term_config(args.profile),
        frequency,
        initial_positive_probability=(
            training_config.initial_positive_probability
        ),
    )
    train_loader = create_author_short_term_train_loader(
        train_dataset,
        args.batch_size,
        training_config.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_author_short_term_evaluation_loader(
        validation_dataset,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = train_author_short_term(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_labels=(item.label for item in train_dataset.assignments),
        device=device,
        training_config=training_config,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        protocol_sha256=protocol_hash,
        community_frequency_path=args.community_frequency,
        community_frequency_sha256=frequency_hash,
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
            "model_name": model.model_name,
            "profile": args.profile,
            "device": str(device),
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(validation_dataset),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "uses_coordinates": False,
            "author_source_architecture": True,
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

