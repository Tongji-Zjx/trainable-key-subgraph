"""Run a real-data forward, loss, backward, and gradient smoke check."""

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
from keysubgraph.models.losses import compute_soft_graph_loss  # noqa: E402
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol.json",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "train",
        protocol["edge_presence_threshold"],
    )
    batch = next(iter(create_data_loader(dataset, batch_size=1, shuffle=False)))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    batch = batch.to(device)
    model = SoftGraphClassifier(
        SoftExtractorConfig(
            node_score_hidden_dim=8,
            edge_score_hidden_dim=8,
            graph_hidden_dim=16,
            graph_layers=2,
            classifier_hidden_dim=8,
            dropout=0.0,
        )
    ).to(device)
    output = model(batch)
    loss = compute_soft_graph_loss(output, batch.labels, budget_weight=0.5)
    loss.total.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    node_gradient = model.node_scorer.network[0].weight.grad
    edge_gradient = model.edge_scorer.network[0].weight.grad
    report = {
        "device": str(device),
        "sample_key": batch[0].sample_key,
        "timepoints": batch[0].num_timepoints,
        "node_counts": list(batch[0].node_counts),
        "logits_shape": list(output.logits.shape),
        "classification_loss": float(loss.classification.detach().cpu()),
        "budget_loss": float(loss.budget.detach().cpu()),
        "total_loss": float(loss.total.detach().cpu()),
        "node_scorer_gradient_l1": float(node_gradient.abs().sum().detach().cpu()),
        "edge_scorer_gradient_l1": float(edge_gradient.abs().sum().detach().cpu()),
        "elapsed_seconds": elapsed_seconds,
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda"
            else None
        ),
    }
    if not all(torch.isfinite(tensor).all() for tensor in (output.logits, loss.total)):
        raise RuntimeError("model smoke check produced non-finite values")
    if report["node_scorer_gradient_l1"] <= 0.0 or report["edge_scorer_gradient_l1"] <= 0.0:
        raise RuntimeError("classification path did not reach both scorers")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
