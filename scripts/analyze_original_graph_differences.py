"""Run the signed structural difference analysis on complete original graphs."""

from __future__ import absolute_import, print_function

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.original_graph import (  # noqa: E402
    compute_original_graph_metrics,
)
from keysubgraph.analysis.statistics import run_structural_metric_analysis  # noqa: E402
from keysubgraph.analysis.visualization import generate_analysis_figures  # noqa: E402
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.graph_dataset import GraphSequenceDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_protocol_all_samples.json",
    )
    parser.add_argument(
        "--split", choices=("validation", "test", "all"), default="all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "original_graph_analysis",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    expected = (
        "subgraph_level_metrics.csv",
        "sample_level_metrics.csv",
        "univariate_test_results.csv",
        "control_group_comparison.csv",
        "analysis_summary.json",
    )
    if not args.overwrite and any((args.output_dir / name).exists() for name in expected):
        raise FileExistsError("original-graph analysis output exists; pass --overwrite")

    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if args.split == "all" and protocol.get("experiment_mode") != "all_samples_exploratory":
        raise ValueError("--split all requires an all-sample protocol")
    paths = protocol["paths"]
    dataset = GraphSequenceDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        args.split,
        protocol["edge_presence_threshold"],
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
    state = {"timepoints": 0}

    def metric_rows():
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            for time_index in range(sample.num_timepoints):
                state["timepoints"] += 1
                yield compute_original_graph_metrics(sample, time_index)
            completed = sample_index + 1
            if completed % args.progress_every == 0 or completed == sample_count:
                print(
                    "processed {}/{} samples; {} timepoints".format(
                        completed, sample_count, state["timepoints"]
                    ),
                    flush=True,
                )

    output_paths = run_structural_metric_analysis(metric_rows(), args.output_dir)
    sample_rows = _read_csv(output_paths["sample_metrics"])
    test_rows = _read_csv(output_paths["tests"])
    figures = generate_analysis_figures(
        sample_rows, test_rows, args.output_dir / "figures"
    )
    report = {
        "split": args.split,
        "source": "original",
        "sample_count": sample_count,
        "timepoint_count": state["timepoints"],
        "debug_limited_samples": args.max_samples,
        "exploratory_in_sample_analysis": args.split == "all",
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "figures": [str(path) for path in figures],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
