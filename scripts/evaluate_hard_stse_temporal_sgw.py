"""Evaluate a Hard-STSE-Temporal-SGW checkpoint on one frozen partition."""

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
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)
from keysubgraph.models.hard_stse_loss import (  # noqa: E402
    HardSTSECriterion,
    HardSTSELossConfig,
)
from keysubgraph.models.hard_stse_temporal_sgw import (  # noqa: E402
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.training import (  # noqa: E402
    hard_stse_config_from_dict,
    load_hard_stse_checkpoint,
    run_hard_stse_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_strict_theory.json",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_payload(path, device):
    try:
        return torch.load(str(path.resolve()), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path.resolve()), map_location=device)


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
    payload = _load_payload(args.checkpoint, device)
    model = HardSTSETemporalSGWClassifier(
        hard_stse_config_from_dict(payload["model_config"])
    ).to(device)
    payload = load_hard_stse_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_protocol_sha256=file_sha256(args.protocol),
    )
    loss_config = HardSTSELossConfig(**payload["loss_config"])
    criterion = HardSTSECriterion(model.config, loss_config)
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
        seed=payload["training_config"]["seed"],
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    metrics = run_hard_stse_epoch(
        model,
        loader,
        device,
        criterion,
        payload["class_weights"],
        epoch=int(payload["best_epoch"]),
        optimizer=None,
        selection_seed=int(payload["training_config"]["seed"]),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol_sha256": payload["protocol_sha256"],
        "variant": model.config.variant,
        "split": args.split,
        "best_epoch": int(payload["best_epoch"]),
        "metrics": metrics,
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
