"""Overfit a fixed balanced 16-sample training cohort as a pipeline diagnostic."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


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


class _FixedGraphSubset(Dataset):
    def __init__(self, source, indices, split):
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.split = split
        self.assignments = tuple(source.assignments[index] for index in self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    source = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "train",
        protocol["edge_presence_threshold"],
    )
    by_label = {0: [], 1: []}
    for index, assignment in enumerate(source.assignments):
        by_label[int(assignment.label)].append((assignment.sample_key, index))
    if min(len(by_label[0]), len(by_label[1])) < 8:
        raise ValueError("overfit diagnostic requires at least eight samples per class")
    selected = []
    for label in (0, 1):
        selected.extend(index for _, index in sorted(by_label[label])[:8])
    selected = tuple(sorted(selected, key=lambda index: source.assignments[index].sample_key))
    train_subset = _FixedGraphSubset(source, selected, "train")
    evaluation_subset = _FixedGraphSubset(source, selected, "validation")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_loader = create_data_loader(
        train_subset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    evaluation_loader = create_data_loader(
        evaluation_subset,
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = [
        {
            "sample_key": source.assignments[index].sample_key,
            "label": int(source.assignments[index].label),
        }
        for index in selected
    ]
    with (output_dir / "overfit_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, ensure_ascii=False, indent=2)
    set_reproducible_seed(args.seed)
    model = FullGraphSequenceClassifier(
        FullGraphClassifierConfig(
            encoder_type=args.encoder_type,
            baseline_dropout=0.0,
            gated_gnn_dropout=0.0,
            classifier_dropout=0.0,
        )
    )
    result = train_full_graph_classifier(
        model,
        train_loader,
        evaluation_loader,
        [item.label for item in train_subset.assignments],
        device,
        FullGraphTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            gradient_clip_norm=5.0,
            early_stopping_patience=0,
            scheduler_patience=max(args.epochs, 1),
            seed=args.seed,
        ),
        output_dir,
        args.protocol,
        file_sha256(args.protocol),
    )
    evaluation = json.load(
        (output_dir / "best_evaluation.json").open("r", encoding="utf-8")
    )
    metrics = evaluation["train"]
    accepted = (
        metrics["accuracy"] >= 0.95
        and metrics["roc_auc"] is not None
        and metrics["roc_auc"] >= 0.99
    )
    print(json.dumps(
        {
            "accepted": accepted,
            "criteria": {"accuracy": 0.95, "roc_auc": 0.99},
            "metrics": metrics,
            "result": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    if not accepted:
        raise RuntimeError("tiny-cohort overfit acceptance criterion was not met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
