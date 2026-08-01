from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.svg_v2_f0_fusion import (
    apply_f0_fusion,
    crossfit_oof_f0_fusion,
    fit_f0_fusion,
    read_prediction_artifact,
)


def _branch(prefix, probabilities):
    return {
        "{}-{}".format(prefix, index): {
            "sample_key": "{}-{}".format(prefix, index),
            "site": "site-{}".format(index % 2),
            "label": int(index % 2),
            "positive_probability": float(probability),
        }
        for index, probability in enumerate(probabilities)
    }


class SVGv2F0FusionTest(unittest.TestCase):
    def test_fusion_is_nonnegative_and_uses_frozen_threshold(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9, 0.3, 0.7))
        svg = _branch("fit", (0.3, 0.7, 0.4, 0.6, 0.45, 0.55))
        fitted = fit_f0_fusion(short, svg, optimization_steps=100)
        self.assertGreaterEqual(fitted["weights"]["short_term"], 0.0)
        self.assertGreaterEqual(fitted["weights"]["svg_v2"], 0.0)
        evaluated = apply_f0_fusion(fitted, short, svg)
        thresholds = {
            row["threshold"] for row in evaluated["predictions"]
        }
        self.assertEqual(thresholds, {fitted["threshold"]})
        self.assertGreater(evaluated["metrics"]["roc_auc"], 0.9)

    def test_fusion_rejects_misaligned_branches(self):
        short = _branch("fit", (0.1, 0.8))
        svg = _branch("other", (0.2, 0.7))
        with self.assertRaisesRegex(ValueError, "same samples"):
            fit_f0_fusion(short, svg, optimization_steps=2)

    def test_outer_oof_diagnostic_predicts_each_sample_once(self):
        short = _branch(
            "oof", (0.10, 0.90, 0.20, 0.80, 0.15, 0.85, 0.25, 0.75)
        )
        svg = _branch(
            "oof", (0.30, 0.70, 0.35, 0.65, 0.40, 0.60, 0.45, 0.55)
        )
        for index, key in enumerate(sorted(short)):
            short[key]["fold"] = (index // 2) % 2
            svg[key]["fold"] = (index // 2) % 2
        result = crossfit_oof_f0_fusion(
            short, svg, optimization_steps=30
        )
        self.assertEqual(result["folds"], [0, 1])
        self.assertEqual(len(result["predictions"]), len(short))
        self.assertEqual(
            len({row["sample_key"] for row in result["predictions"]}),
            len(short),
        )
        self.assertTrue(
            all(
                row["fit_and_evaluation_disjoint"]
                for row in result["fold_results"]
            )
        )

    def test_prediction_artifact_reads_evaluation_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            path.write_text(
                json.dumps({"predictions": list(_branch("x", (0.2, 0.8)).values())}),
                encoding="utf-8",
            )
            rows = read_prediction_artifact(path)
        self.assertEqual(set(rows), {"x-0", "x-1"})
        self.assertAlmostEqual(rows["x-1"]["positive_probability"], 0.8)


if __name__ == "__main__":
    unittest.main()
