"""Build features for real samples without retaining all dense timepoints."""

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

from keysubgraph.data.data_protocol import protocol_partitions, validate_data_protocol  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.features.graph_features import GraphFeatureBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol.json",
    )
    parser.add_argument("--samples-per-split", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_split < 1:
        raise ValueError("samples-per-split must be positive")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    builder = GraphFeatureBuilder()
    report = {}
    with torch.no_grad():
        for split in protocol_partitions(protocol):
            dataset = GraphSequenceDataset(
                PROJECT_ROOT / paths["dataset_root"],
                PROJECT_ROOT / paths["sample_index_csv"],
                PROJECT_ROOT / paths["splits_csv"],
                split,
                protocol["edge_presence_threshold"],
            )
            rows = []
            for sample_index in range(min(args.samples_per_split, len(dataset))):
                sample = dataset[sample_index]
                timepoint_count = 0
                aligned_node_values = 0
                aligned_edge_values = 0
                node_feature_dim = None
                edge_feature_dim = None
                for features in builder.iter_sample(sample):
                    if not bool(torch.isfinite(features.node_features).all()):
                        raise RuntimeError("non-finite node features")
                    if not bool(torch.isfinite(features.edge_features).all()):
                        raise RuntimeError("non-finite edge features")
                    node_feature_dim = features.node_feature_dim
                    edge_feature_dim = features.edge_feature_dim
                    aligned_node_values += int(features.delta_degree_mask.sum())
                    aligned_edge_values += int(features.delta_edge_mask.sum())
                    timepoint_count += 1
                rows.append(
                    {
                        "sample_key": sample.sample_key,
                        "timepoints": timepoint_count,
                        "node_feature_dim": node_feature_dim,
                        "edge_feature_dim": edge_feature_dim,
                        "aligned_delta_nodes": aligned_node_values,
                        "aligned_delta_edges": aligned_edge_values,
                    }
                )
            report[split] = rows
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
