"""Check a baseline manifest, reconstructed subgraphs, masks, and batches."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_collate import create_baseline_loader  # noqa: E402
from keysubgraph.data.baseline_dataset import BaselineHardSubgraphDataset  # noqa: E402
from keysubgraph.features.structural_prior import (  # noqa: E402
    STATIC_WINDOW_STRUCTURAL_FEATURES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-scan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = BaselineHardSubgraphDataset(PROJECT_ROOT, args.manifest)
    loader = create_baseline_loader(
        dataset,
        args.batch_size,
        seed=42,
        num_workers=args.num_workers,
        shuffle=False,
    )
    sample_count = 0
    timepoint_count = 0
    subgraph_count = 0
    node_counts = []
    positive_edges = 0
    negative_edges = 0
    structural_valid_counts = torch.zeros(
        len(STATIC_WINDOW_STRUCTURAL_FEATURES), dtype=torch.long
    )
    for batch_index, batch in enumerate(loader):
        if not bool(torch.isfinite(batch.node_features).all()):
            raise RuntimeError("baseline node features are non-finite")
        if not bool(torch.isfinite(batch.adjacency).all()):
            raise RuntimeError("baseline adjacency is non-finite")
        if bool((batch.node_features[~batch.node_mask] != 0).any()):
            raise RuntimeError("node padding contains nonzero features")
        if bool((batch.adjacency < 0).any()):
            negative_edges += int(torch.triu(batch.adjacency < 0, diagonal=1).sum())
        if bool((batch.adjacency > 0).any()):
            positive_edges += int(torch.triu(batch.adjacency > 0, diagonal=1).sum())
        if not bool(torch.isfinite(batch.window_structural_features).all()):
            raise RuntimeError("window structural features are non-finite")
        if bool((batch.window_structural_features[~batch.window_structural_mask] != 0).any()):
            raise RuntimeError("missing structural features are not zero-filled")
        structural_valid_counts += batch.window_structural_mask.sum(dim=0)
        sample_count += batch.batch_size
        timepoint_count += batch.window_count
        subgraph_count += batch.subgraph_count
        node_counts.extend(int(value) for value in batch.node_mask.sum(dim=1).tolist())
        if not args.full_scan and batch_index == 0:
            break
    payload = {
        "manifest": str(args.manifest.resolve()),
        "full_scan": bool(args.full_scan),
        "dataset_sample_count": len(dataset),
        "scanned_sample_count": sample_count,
        "timepoint_count": timepoint_count,
        "subgraph_count": subgraph_count,
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "positive_edge_count": positive_edges,
        "negative_edge_count": negative_edges,
        "signed_edges_present": positive_edges > 0 and negative_edges > 0,
        "structural_feature_names": list(STATIC_WINDOW_STRUCTURAL_FEATURES),
        "structural_valid_window_counts": {
            name: int(structural_valid_counts[index])
            for index, name in enumerate(STATIC_WINDOW_STRUCTURAL_FEATURES)
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
