"""Create and verify the frozen WMRC NoCoord training protocol."""

from __future__ import absolute_import, print_function

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_protocol import (  # noqa: E402
    freeze_data_protocol,
    protocol_node_name_policy,
    validate_data_protocol,
)
from keysubgraph.data.data_split import (  # noqa: E402
    SplitConfig,
    create_data_splits,
    read_sample_index,
    summarize_assignments,
    write_split_artifacts,
)
from keysubgraph.data.sample_index import (  # noqa: E402
    IndexBuildConfig,
    NODE_NAME_POLICY_ROW_INDEX_FALLBACK,
    build_sample_index,
    summarize_records,
    write_index_artifacts,
)


EXPECTED_SAMPLE_COUNT = 546
PROTOCOL_NAME = "wmrc_no_coord"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "wmrc_5_0.5",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "wmrc",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "data_protocol_wmrc_no_coord.json",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-sample-count", type=int, default=546)
    parser.add_argument("--edge-presence-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _existing_artifacts(artifact_root, protocol_path):
    paths = (
        artifact_root / "index" / "sample_inventory.csv",
        artifact_root / "index" / "sample_index.csv",
        artifact_root / "index" / "exclusion_manifest.csv",
        artifact_root / "index" / "sample_index_summary.json",
        artifact_root / "splits" / "splits.csv",
        artifact_root / "splits" / "splits.json",
        protocol_path,
    )
    return tuple(path for path in paths if path.exists())


def _validate_existing(args):
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError("existing WMRC protocol has the wrong name")
    if int(protocol.get("sample_count", -1)) != args.expected_sample_count:
        raise ValueError("existing WMRC protocol has the wrong sample count")
    if int(protocol.get("split_seed", -1)) != args.seed:
        raise ValueError("existing WMRC protocol has the wrong split seed")
    if (
        protocol_node_name_policy(protocol)
        != NODE_NAME_POLICY_ROW_INDEX_FALLBACK
    ):
        raise ValueError("existing WMRC protocol has the wrong node-name policy")
    return protocol


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.protocol = args.protocol.resolve()
    if args.expected_sample_count < 1:
        raise ValueError("expected-sample-count must be positive")

    existing = _existing_artifacts(args.artifact_root, args.protocol)
    if existing and not args.overwrite:
        if args.protocol.is_file():
            protocol = _validate_existing(args)
            print("Existing immutable WMRC protocol is valid; reusing it.")
            print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        raise RuntimeError(
            "partial WMRC artifacts exist; inspect them and use --overwrite "
            "only when replacement is intended"
        )

    records = build_sample_index(
        IndexBuildConfig(
            dataset_root=args.data_root,
            edge_presence_threshold=args.edge_presence_threshold,
            node_name_policy=NODE_NAME_POLICY_ROW_INDEX_FALLBACK,
        )
    )
    index_summary = summarize_records(records)
    if int(index_summary["total_samples"]) != args.expected_sample_count:
        raise ValueError(
            "expected {} WMRC samples, discovered {}".format(
                args.expected_sample_count,
                index_summary["total_samples"],
            )
        )
    if int(index_summary["included_samples"]) != args.expected_sample_count:
        raise ValueError(
            "WMRC index excluded samples: {}".format(
                index_summary["exclusion_reason_counts"]
            )
        )
    if (
        int(index_summary["node_name_fallback_samples"])
        != args.expected_sample_count
    ):
        raise ValueError(
            "WMRC node-name fallback was not applied to every sample"
        )

    index_paths = write_index_artifacts(
        records,
        args.artifact_root / "index",
    )
    samples = read_sample_index(index_paths["index"])
    split_config = SplitConfig(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    assignments = create_data_splits(samples, split_config)
    split_paths = write_split_artifacts(
        assignments,
        args.artifact_root / "splits",
        index_paths["index"],
        split_config,
        overwrite=args.overwrite,
    )
    split_summary = summarize_assignments(assignments, split_config)
    protocol = freeze_data_protocol(
        project_root=PROJECT_ROOT,
        dataset_root=args.data_root,
        sample_index_csv=index_paths["index"],
        splits_csv=split_paths["csv"],
        splits_json=split_paths["json"],
        output_path=args.protocol,
        edge_presence_threshold=args.edge_presence_threshold,
        protocol_name=PROTOCOL_NAME,
        node_name_policy=NODE_NAME_POLICY_ROW_INDEX_FALLBACK,
        overwrite=args.overwrite,
    )
    result = {
        "passed": True,
        "protocol": str(args.protocol),
        "sample_count": len(assignments),
        "node_name_policy": protocol_node_name_policy(protocol),
        "node_name_fallback_samples": index_summary[
            "node_name_fallback_samples"
        ],
        "split_seed": args.seed,
        "split_counts": {
            name: values["sample_count"]
            for name, values in split_summary["splits"].items()
        },
        "class_counts": {
            name: values["class_counts"]
            for name, values in split_summary["splits"].items()
        },
        "checks": split_summary["checks"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
