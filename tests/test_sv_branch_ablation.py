from __future__ import absolute_import, division, print_function

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_branch_ablation import (  # noqa: E402
    SV_BRANCH_ABLATION_VARIANTS,
    compare_sv_branch_ablation,
)


def _write_summary(directory, variant, scores):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = (0, 0, 1, 1)
    for index, (label, score) in enumerate(zip(labels, scores)):
        rows.append(
            {
                "fold": index % 2,
                "sample_key": "sample_{}".format(index),
                "site": "site",
                "label": label,
                "positive_probability": score,
                "threshold": 0.5,
                "predicted_label": int(score >= 0.5),
            }
        )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "artifact_type": (
                    "sv_signed_gin_crossfit_oof_summary"
                ),
                "variant": variant,
                "checks": {"every_sample_predicted_once": True},
                "metrics": {"sample_count": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    with (directory / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SVBranchAblationTest(unittest.TestCase):
    def test_paired_comparison_preserves_sample_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = {
                "S": (0.4, 0.3, 0.6, 0.7),
                "SV": (0.2, 0.4, 0.8, 0.6),
                "SVG": (0.1, 0.2, 0.9, 0.8),
            }
            paths = {}
            for name, values in scores.items():
                paths[name] = root / name
                _write_summary(
                    paths[name],
                    SV_BRANCH_ABLATION_VARIANTS[name],
                    values,
                )
            result = compare_sv_branch_ablation(
                [
                    (
                        "dummy",
                        paths["S"],
                        paths["SV"],
                        paths["SVG"],
                    )
                ],
                root / "comparison",
                bootstrap_repeats=20,
                permutation_repeats=20,
                seed=7,
            )
            current = result["datasets"]["dummy"]
            self.assertEqual(current["sample_count"], 4)
            self.assertAlmostEqual(
                current["models"]["SVG"]["pooled_oof_roc_auc"],
                1.0,
            )
            self.assertTrue(result["comparison_json"].is_file())
            self.assertTrue(result["comparison_markdown"].is_file())

    def test_rejects_misaligned_oof_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("S", "SV", "SVG"):
                paths[name] = root / name
                _write_summary(
                    paths[name],
                    SV_BRANCH_ABLATION_VARIANTS[name],
                    (0.4, 0.3, 0.6, 0.7),
                )
            csv_path = paths["SV"] / "oof_predictions.csv"
            text = csv_path.read_text(encoding="utf-8")
            csv_path.write_text(
                text.replace("sample_0", "different"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample sets"):
                compare_sv_branch_ablation(
                    [
                        (
                            "dummy",
                            paths["S"],
                            paths["SV"],
                            paths["SVG"],
                        )
                    ],
                    root / "comparison",
                    bootstrap_repeats=5,
                    permutation_repeats=5,
                )


if __name__ == "__main__":
    unittest.main()
