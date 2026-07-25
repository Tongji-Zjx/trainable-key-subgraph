"""Structural metrics, controls, statistical tests, and reporting."""

from .structural_metrics import METRIC_NAMES, aggregate_sample_metrics, compute_subgraph_metrics
from .statistics import run_structural_analysis, run_structural_metric_analysis
from .original_graph import (
    build_original_graph_record,
    compute_original_graph_metrics,
    iter_original_graph_metrics,
    iter_original_graph_records,
)
from .dual_proxy_exact_alignment import (
    FEATURE_BLOCKS,
    analyze_proxy_exact_alignment,
    write_proxy_exact_alignment_artifacts,
)

__all__ = [
    "METRIC_NAMES",
    "aggregate_sample_metrics",
    "compute_subgraph_metrics",
    "build_original_graph_record",
    "compute_original_graph_metrics",
    "iter_original_graph_metrics",
    "iter_original_graph_records",
    "run_structural_analysis",
    "run_structural_metric_analysis",
    "FEATURE_BLOCKS",
    "analyze_proxy_exact_alignment",
    "write_proxy_exact_alignment_artifacts",
]
