"""Overfit a tiny balanced real-data cohort as an implementation diagnostic."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_collate import baseline_padded_collate  # noqa: E402
from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--target-accuracy", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_balanced_indices(dataset, samples_per_class):
    if samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    by_label = {0: [], 1: []}
    for index, record in enumerate(dataset.records):
        if record.label in by_label:
            by_label[record.label].append(index)
    if any(len(values) < samples_per_class for values in by_label.values()):
        raise ValueError("manifest does not contain enough samples from both classes")
    indices = by_label[0][:samples_per_class] + by_label[1][:samples_per_class]
    return sorted(indices)


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.manifest)
    indices = choose_balanced_indices(dataset, args.samples_per_class)
    samples = [dataset[index] for index in indices]
    batch = baseline_padded_collate(samples).to(device)
    model = SignedSequenceBaseline(
        BaselineModelConfig(
            node_hidden_dim=16,
            signed_gnn_layers=1,
            signed_gnn_dropout=0.0,
            fusion_dim=32,
            gru_hidden_dim=32,
            classifier_hidden_dim=16,
            classifier_dropout=0.0,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    initial_loss = None
    final_loss = None
    final_accuracy = 0.0
    completed_steps = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()
        output = model(batch)
        loss = F.cross_entropy(output.logits, batch.labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite tiny-cohort loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = float(
                (output.logits.argmax(dim=1) == batch.labels).float().mean().cpu()
            )
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
        final_accuracy = accuracy
        completed_steps = step
        if step == 1 or step % args.log_every == 0:
            print(
                "step {}/{} loss={:.6f} accuracy={:.3f}".format(
                    step, args.max_steps, final_loss, final_accuracy
                ),
                flush=True,
            )
        if final_accuracy >= args.target_accuracy and final_loss < 0.05:
            break

    model.eval()
    with torch.no_grad():
        final_output = model(batch)
        final_loss = float(F.cross_entropy(final_output.logits, batch.labels).cpu())
        final_accuracy = float(
            (final_output.logits.argmax(dim=1) == batch.labels).float().mean().cpu()
        )
    passed = bool(final_accuracy >= args.target_accuracy and final_loss < 0.05)
    payload = {
        "passed": passed,
        "device": str(device),
        "sample_count": batch.batch_size,
        "class_counts": {
            "0": int((batch.labels == 0).sum().cpu()),
            "1": int((batch.labels == 1).sum().cpu()),
        },
        "window_count": batch.window_count,
        "subgraph_count": batch.subgraph_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "steps": completed_steps,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("tiny-cohort overfit acceptance criterion was not met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
