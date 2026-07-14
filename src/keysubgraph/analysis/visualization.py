"""Deterministic structural-analysis figures from saved metric tables."""

from __future__ import absolute_import, division, print_function

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "keysubgraph_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .structural_metrics import METRIC_NAMES


def generate_analysis_figures(
    sample_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    output_dir = Path(output_dir).resolve()
    boxplot_dir = output_dir / "boxplots"
    boxplot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    sources = sorted(set(row["source"] for row in sample_rows))
    for source in sources:
        rows = [row for row in sample_rows if row["source"] == source]
        figure, axes = plt.subplots(5, 3, figsize=(15, 20))
        for axis, metric in zip(axes.flat, METRIC_NAMES):
            values = []
            labels = []
            for label in (0, 1):
                current = [
                    float(row[metric])
                    for row in rows
                    if int(row["label"]) == label
                    and math.isfinite(float(row[metric]))
                ]
                if current:
                    values.append(current)
                    labels.append(str(label))
            if values:
                axis.boxplot(values, labels=labels, showfliers=False)
            axis.set_title(metric)
            axis.set_xlabel("class")
        figure.suptitle("Sample-level structural metrics: {}".format(source))
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = boxplot_dir / (source + ".png")
        figure.savefig(str(path), dpi=150)
        plt.close(figure)
        paths.append(path)

    if test_rows:
        source_order = sorted(set(row["source"] for row in test_rows))
        lookup = {
            (row["source"], row["metric"]): float(row["standardized_discrepancy"])
            for row in test_rows
        }
        matrix = np.asarray(
            [
                [lookup.get((source, metric), np.nan) for metric in METRIC_NAMES]
                for source in source_order
            ],
            dtype=np.float64,
        )
        figure, axis = plt.subplots(figsize=(16, max(3, len(source_order) * 1.2)))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(METRIC_NAMES)))
        axis.set_xticklabels(METRIC_NAMES, rotation=60, ha="right")
        axis.set_yticks(range(len(source_order)))
        axis.set_yticklabels(source_order)
        axis.set_title("Standardized class discrepancy")
        figure.colorbar(image, ax=axis, label="Delta")
        figure.tight_layout()
        path = output_dir / "discrepancy_heatmap.png"
        figure.savefig(str(path), dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths
