"""Load and adapt every sample referenced by a sample index."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import SplitAssignment, read_sample_index  # noqa: E402
from keysubgraph.data.graph_dataset import _adapt_payload  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check that every indexed .pt file passes the Dataset adapter."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--edge-presence-threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    samples = read_sample_index(args.sample_index)
    loaded = 0
    total_timepoints = 0

    for indexed in samples:
        path = (data_root / indexed.relative_path).resolve()
        path.relative_to(data_root)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        assignment = SplitAssignment(
            sample_key=indexed.sample_key,
            sample_id=indexed.sample_id,
            site=indexed.site,
            subject_id=indexed.subject_id,
            session_id=indexed.session_id,
            group_id=indexed.group_id,
            label=indexed.label,
            relative_path=indexed.relative_path,
            split="train",
            seed=0,
        )
        sample = _adapt_payload(
            payload, assignment, args.edge_presence_threshold
        )
        loaded += 1
        total_timepoints += sample.num_timepoints

    print("adapted_samples: {}".format(loaded))
    print("total_timepoints: {}".format(total_timepoints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
