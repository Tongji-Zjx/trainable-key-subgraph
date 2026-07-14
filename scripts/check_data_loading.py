"""Smoke-check the frozen protocol, Dataset, and one batch from every split."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import protocol_partitions, validate_data_protocol  # noqa: E402
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceDataset,
    create_data_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Load and validate every indexed sample instead of only the first batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    report = {"protocol_valid": True, "splits": {}}
    for split in protocol_partitions(protocol):
        dataset = GraphSequenceDataset(
            dataset_root=PROJECT_ROOT / paths["dataset_root"],
            sample_index_csv=PROJECT_ROOT / paths["sample_index_csv"],
            splits_csv=PROJECT_ROOT / paths["splits_csv"],
            split=split,
            edge_presence_threshold=protocol["edge_presence_threshold"],
        )
        loader = create_data_loader(
            dataset,
            batch_size=args.batch_size,
            seed=protocol["split_seed"],
            num_workers=args.num_workers,
            shuffle=False,
        )
        loaded = 0
        first_keys = []
        timepoint_counts = []
        node_counts = []
        all_have_positive_edges = True
        all_have_negative_edges = True
        for batch_index, batch in enumerate(loader):
            if batch_index == 0:
                first_keys = list(batch.sample_keys)
            loaded += len(batch)
            for sample in batch:
                timepoint_counts.append(sample.num_timepoints)
                node_counts.extend(sample.node_counts)
                all_have_positive_edges = all_have_positive_edges and any(
                    bool((graph[mask] > 0).any())
                    for graph, mask in zip(sample.adjacency, sample.edge_mask)
                )
                all_have_negative_edges = all_have_negative_edges and any(
                    bool((graph[mask] < 0).any())
                    for graph, mask in zip(sample.adjacency, sample.edge_mask)
                )
            if not args.full_scan:
                break
        report["splits"][split] = {
            "dataset_size": len(dataset),
            "loaded_samples": loaded,
            "first_batch_sample_keys": first_keys,
            "min_timepoints": min(timepoint_counts),
            "max_timepoints": max(timepoint_counts),
            "min_nodes": min(node_counts),
            "max_nodes": max(node_counts),
            "all_samples_have_positive_edges": all_have_positive_edges,
            "all_samples_have_negative_edges": all_have_negative_edges,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
