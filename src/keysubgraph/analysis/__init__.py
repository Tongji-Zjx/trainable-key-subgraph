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
from .dual_proxy_input_exact_head import (
    build_proxy_input_exact_head_evaluation,
    write_proxy_input_exact_head_artifacts,
)
from .dual_frozen_logit_ensemble import (
    build_frozen_equal_logit_ensemble,
    write_frozen_equal_logit_ensemble_artifacts,
)
from .dual_classification_bottleneck import (
    analyze_dual_classification_bottleneck,
    write_dual_classification_bottleneck_artifacts,
)
from .dual_transferred_head import (
    build_transferred_head_evaluation,
    write_transferred_head_artifacts,
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
    "build_proxy_input_exact_head_evaluation",
    "write_proxy_input_exact_head_artifacts",
    "build_frozen_equal_logit_ensemble",
    "write_frozen_equal_logit_ensemble_artifacts",
    "analyze_dual_classification_bottleneck",
    "write_dual_classification_bottleneck_artifacts",
    "build_transferred_head_evaluation",
    "write_transferred_head_artifacts",
]
