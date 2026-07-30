"""Fit train-only normalization for the coordinate-free short-term branch."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import (  # noqa: E402
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402
from keysubgraph.features.structured_short_term_features import (  # noqa: E402
    fit_structured_short_term_standardizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-std", type=float, default=1.0e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        "train",
        edge_presence_threshold=protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )

    def samples():
        for index in range(len(dataset)):
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dataset):
                print(
                    "standardizer sample {}/{}".format(index + 1, len(dataset)),
                    flush=True,
                )
            yield dataset[index]

    standardizer = fit_structured_short_term_standardizer(
        samples(),
        args.protocol,
        edge_presence_threshold=protocol["edge_presence_threshold"],
        minimum_std=args.minimum_std,
    )
    output = standardizer.save(args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "train_sample_count": standardizer.train_sample_count,
                "train_window_count": standardizer.train_window_count,
                "train_node_count": standardizer.train_node_count,
                "protocol_sha256": standardizer.protocol_sha256,
                "edge_presence_threshold": (
                    standardizer.edge_presence_threshold
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
