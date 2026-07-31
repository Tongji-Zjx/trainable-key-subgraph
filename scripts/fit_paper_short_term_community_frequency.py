"""Fit fold-local train-only community frequencies for paper p_ST."""

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
from keysubgraph.features.paper_short_term_pst import (  # noqa: E402
    fit_paper_short_term_community_frequency,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    paths = protocol["paths"]
    splits_path = PROJECT_ROOT / paths["splits_csv"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        splits_path,
        "train",
        edge_presence_threshold=protocol["edge_presence_threshold"],
        node_name_policy=protocol_node_name_policy(protocol),
    )

    def samples():
        for index in range(len(dataset)):
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dataset):
                print(
                    "community-frequency sample {}/{}".format(
                        index + 1, len(dataset)
                    ),
                    flush=True,
                )
            yield dataset[index]

    artifact = fit_paper_short_term_community_frequency(
        samples(),
        protocol_path=args.protocol,
        train_manifest_path=splits_path,
        epsilon=args.epsilon,
        outer_fold=args.outer_fold,
    )
    output = artifact.save(args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "train_sample_count": artifact.train_sample_count,
                "train_window_count": artifact.train_window_count,
                "community_label_count": len(artifact.counts),
                "valid_community_node_count": artifact.total_count,
                "protocol_sha256": artifact.protocol_sha256,
                "train_manifest_sha256": artifact.train_manifest_sha256,
                "outer_fold": artifact.outer_fold,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

