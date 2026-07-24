"""Audit one frozen Exact-STSE coordinate or NoCoord cohort."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.exact_stse_dataset import ExactSTSEDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("exact_stse", "exact_stse_no_coord"),
        default="exact_stse",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_exact_stse_coords.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "exact_stse"
        / "input_audit.json",
    )
    return parser.parse_args()


def _summary(values):
    mean = sum(values) / float(len(values))
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean,
        "standard_deviation": math.sqrt(
            sum((value - mean) ** 2 for value in values) / float(len(values))
        ),
    }


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    use_coordinates = args.variant == "exact_stse"
    if use_coordinates and int(protocol["sample_count"]) != 307:
        raise ValueError(
            "coordinate Exact-STSE audit requires the 307-sample cohort"
        )
    paths = protocol["paths"]
    common = (
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
    )
    split_counts = {}
    class_counts = {}
    node_counts = []
    timepoint_counts = []
    coordinate_hashes = Counter()
    maximum_community_id = -1
    sample_keys = []
    failures = []
    for split in ("train", "validation", "test"):
        dataset = ExactSTSEDataset(
            *common,
            split,
            protocol["edge_presence_threshold"],
            require_coordinates=use_coordinates,
        )
        split_counts[split] = len(dataset)
        labels = Counter()
        for index in range(len(dataset)):
            try:
                sample = dataset[index]
                sample_keys.append(sample.sample_key)
                labels[str(sample.label)] += 1
                timepoint_counts.append(sample.num_timepoints)
                for coordinates, communities, count in zip(
                    sample.coordinates,
                    sample.graph.communities,
                    sample.graph.node_counts,
                ):
                    node_counts.append(int(count))
                    if use_coordinates:
                        coordinate_hashes[
                            hashlib.sha256(
                                coordinates.contiguous().numpy().tobytes()
                            ).hexdigest()
                        ] += 1
                    maximum_community_id = max(
                        maximum_community_id,
                        int(communities.max().item()),
                    )
            except Exception as error:
                failures.append(
                    {
                        "split": split,
                        "index": index,
                        "error": str(error),
                    }
                )
        class_counts[split] = dict(labels)
    result = {
        "passed": (
            not failures
            and len(sample_keys) == int(protocol["sample_count"])
            and len(set(sample_keys)) == len(sample_keys)
            and sum(split_counts.values()) == int(protocol["sample_count"])
        ),
        "protocol": str(args.protocol.resolve()),
        "variant": args.variant,
        "source_coordinates_loaded": use_coordinates,
        "sample_count": len(sample_keys),
        "unique_sample_count": len(set(sample_keys)),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "timepoints_per_sample": _summary(timepoint_counts),
        "nodes_per_timepoint": _summary(node_counts),
        "coordinate_array_hash_count": (
            len(coordinate_hashes) if use_coordinates else None
        ),
        "coordinate_array_hash_usage": (
            dict(coordinate_hashes) if use_coordinates else None
        ),
        "maximum_community_id": maximum_community_id,
        "community_vocab_size_required": maximum_community_id + 2,
        "model_input_dimensions": {
            "exact_stse": 24,
            "exact_stse_no_coord": 18,
        },
        "failures": failures,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("Exact-STSE input audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
