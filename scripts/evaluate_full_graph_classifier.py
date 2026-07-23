"""Evaluate a frozen full-graph encoder checkpoint on one protocol split."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
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
    FullGraphClassifierConfig,
    FullGraphSequenceClassifier,
)
from keysubgraph.training import (  # noqa: E402
    load_full_graph_classifier_checkpoint,
    run_full_graph_classifier_epoch,
)


def _load(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    header = _load(args.checkpoint.resolve(), torch.device("cpu"))
    model = FullGraphSequenceClassifier(
        FullGraphClassifierConfig(**header["model_config"])
    )
    payload = load_full_graph_classifier_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=file_sha256(args.protocol),
    )
    model.to(device)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    loader = create_data_loader(
        dataset,
        args.batch_size,
        seed=int(payload["training_config"]["seed"]),
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    metrics = run_full_graph_classifier_epoch(
        model,
        loader,
        device,
        payload["class_weights"],
        optimizer=None,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": int(payload["best_epoch"]),
        "encoder_type": model.config.encoder_type,
        "split": args.split,
        "metrics": metrics,
    }
    output = (
        args.output
        if args.output is not None
        else args.checkpoint.resolve().parent / (
            "{}_evaluation.json".format(args.split)
        )
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(str(temporary), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
