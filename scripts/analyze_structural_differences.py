"""Build matched controls and run signed structural difference analysis."""

from __future__ import absolute_import, print_function

import argparse
import csv
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.controls import (  # noqa: E402
    generate_random_controls,
    generate_top_degree_controls,
    select_low_score_controls,
)
from keysubgraph.analysis.statistics import (  # noqa: E402
    run_structural_analysis,
    run_univariate_tests,
)
from keysubgraph.analysis.structural_metrics import (  # noqa: E402
    aggregate_sample_metrics,
    compute_subgraph_metrics,
)
from keysubgraph.analysis.visualization import generate_analysis_figures  # noqa: E402
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "data_protocol.json")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "structural_analysis")
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-controls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_digest = file_sha256(args.protocol)
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    dataset_lookup = {
        assignment.sample_id: index for index, assignment in enumerate(dataset.assignments)
    }
    records = []
    export_dir = args.export_dir.resolve()
    export_paths = sorted(export_dir.glob("*.json"))
    if not export_paths and (export_dir / args.split).is_dir():
        export_paths = sorted((export_dir / args.split).glob("*.json"))
    if not export_paths:
        raise ValueError("no hard export JSON files found")
    for export_path in export_paths:
        with export_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["split"] != args.split:
            raise ValueError("export split mismatch: {}".format(export_path))
        if payload["data_protocol_sha256"] != protocol_digest:
            raise ValueError("export protocol hash mismatch: {}".format(export_path))
        if float(payload["edge_presence_threshold"]) != float(protocol["edge_presence_threshold"]):
            raise ValueError("export edge threshold mismatch: {}".format(export_path))
        if payload["sample_id"] not in dataset_lookup:
            raise ValueError("export sample is absent from the frozen split")
        sample = dataset[dataset_lookup[payload["sample_id"]]]
        for timepoint in payload["timepoints"]:
            for subgraph in timepoint["subgraphs"]:
                row = dict(subgraph)
                row["source"] = "key"
                row["repeat_index"] = None
                records.append(row)
        if not args.no_controls:
            records.extend(
                generate_random_controls(
                    sample, payload, repeats=args.random_repeats, seed=args.seed
                )
            )
            records.extend(generate_top_degree_controls(sample, payload))
            records.extend(select_low_score_controls(payload))

    output_paths = run_structural_analysis(records, args.output_dir)
    metric_rows = [compute_subgraph_metrics(record) for record in records]
    sample_rows = aggregate_sample_metrics(metric_rows)
    test_rows = run_univariate_tests(sample_rows)
    figures = generate_analysis_figures(
        sample_rows, test_rows, args.output_dir / "figures"
    )
    report = {
        "split": args.split,
        "export_count": len(export_paths),
        "subgraph_record_count": len(records),
        "sources": sorted(set(record.get("source", "key") for record in records)),
        "random_repeats": 0 if args.no_controls else args.random_repeats,
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "figures": [str(path) for path in figures],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
