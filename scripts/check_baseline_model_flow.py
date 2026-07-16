"""Run baseline forward, weighted loss, backward, and gradient checks."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_collate import create_baseline_loader  # noqa: E402
from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.training.baseline_trainer import baseline_class_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.manifest)
    loader = create_baseline_loader(
        dataset,
        args.batch_size,
        seed=42,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader)).to(device)
    model = SignedSequenceBaseline(BaselineModelConfig()).to(device)
    labels_for_weights = [record.label for record in dataset.records]
    if len(set(labels_for_weights)) == 2:
        class_weights = baseline_class_weights(labels_for_weights).to(device)
    else:
        class_weights = torch.ones(2, dtype=torch.float32, device=device)
    output = model(batch)
    loss = F.cross_entropy(output.logits, batch.labels, weight=class_weights)
    loss.backward()
    gradient_report = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError("missing gradient: {}".format(name))
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("non-finite gradient: {}".format(name))
        gradient_report[name] = float(parameter.grad.abs().sum().detach().cpu())
    payload = {
        "device": str(device),
        "batch_size": batch.batch_size,
        "subgraph_count": batch.subgraph_count,
        "window_count": batch.window_count,
        "logits_shape": list(output.logits.shape),
        "hidden_states_shape": list(output.hidden_states.shape),
        "loss": float(loss.detach().cpu()),
        "all_outputs_finite": bool(torch.isfinite(output.logits).all()),
        "positive_branch_gradient": gradient_report[
            "subgraph_encoder.layers.0.positive_projection.weight"
        ],
        "negative_branch_gradient": gradient_report[
            "subgraph_encoder.layers.0.negative_projection.weight"
        ],
        "gru_gradient": gradient_report["gru_cell.weight_hh"],
        "classifier_gradient": gradient_report["classifier.3.weight"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
