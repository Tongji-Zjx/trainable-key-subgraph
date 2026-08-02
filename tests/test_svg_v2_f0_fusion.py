from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.svg_v2_f0_fusion import (
    apply_f0_fusion,
    apply_multi_f0_fusion,
    apply_residual_logit_fusion,
    crossfit_oof_f0_fusion,
    fit_f0_fusion,
    fit_multi_f0_fusion,
    fit_residual_logit_fusion,
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

    def test_multi_expert_fusion_is_nonnegative_and_aligned(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9, 0.3, 0.7))
        static = _branch("fit", (0.2, 0.7, 0.3, 0.8, 0.4, 0.6))
        g2 = _branch("fit", (0.3, 0.9, 0.25, 0.85, 0.35, 0.75))
        fitted = fit_multi_f0_fusion(
            {"short_term": short, "s": static, "g2": g2},
            optimization_steps=100,
        )
        self.assertEqual(
            fitted["expert_names"], ["short_term", "s", "g2"]
        )
        self.assertEqual(
            set(fitted["weights"]), {"short_term", "s", "g2"}
        )
        self.assertTrue(
            all(weight >= 0.0 for weight in fitted["weights"].values())
        )
        evaluated = apply_multi_f0_fusion(
            fitted, {"short_term": short, "s": static, "g2": g2}
        )
        self.assertEqual(len(evaluated["predictions"]), len(short))
        self.assertGreater(evaluated["metrics"]["roc_auc"], 0.9)

    def test_multi_expert_fusion_rejects_order_or_sample_changes(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9))
        static = _branch("fit", (0.2, 0.7, 0.3, 0.8))
        fitted = fit_multi_f0_fusion(
            {"short_term": short, "s": static}, optimization_steps=5
        )
        with self.assertRaisesRegex(ValueError, "order"):
            apply_multi_f0_fusion(
                fitted, {"s": static, "short_term": short}
            )
        shifted = _branch("other", (0.2, 0.7, 0.3, 0.8))
        with self.assertRaisesRegex(ValueError, "same samples"):
            fit_multi_f0_fusion(
                {"short_term": short, "s": shifted},
                optimization_steps=2,
            )

    def test_residual_fusion_preserves_anchor_at_zero_gate(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9, 0.3, 0.7))
        g2 = _branch("fit", (0.3, 0.7, 0.4, 0.6, 0.45, 0.55))
        fitted = fit_residual_logit_fusion(
            "short_term",
            short,
            "g2",
            g2,
            optimization_steps=50,
        )
        self.assertGreaterEqual(fitted["gate"], 0.0)
        self.assertLessEqual(fitted["gate"], 1.0)
        evaluated = apply_residual_logit_fusion(fitted, short, g2)
        zero_gate = dict(fitted)
        zero_gate["gate"] = 0.0
        anchor_only = apply_residual_logit_fusion(zero_gate, short, g2)
        changed_residual = dict(zero_gate)
        changed_residual["residual_intercept"] = 100.0
        changed_residual["residual_slope"] = -100.0
        still_anchor_only = apply_residual_logit_fusion(
            changed_residual, short, g2
        )
        self.assertEqual(
            [
                row["positive_probability"]
                for row in anchor_only["predictions"]
            ],
            [
                row["positive_probability"]
                for row in still_anchor_only["predictions"]
            ],
        )
        for row in anchor_only["predictions"]:
            self.assertGreater(row["positive_probability"], 0.0)
            self.assertLess(row["positive_probability"], 1.0)
        self.assertEqual(
            {row["threshold"] for row in evaluated["predictions"]},
            {fitted["threshold"]},
        )

    def test_f1_and_f2_use_opposite_anchors(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9, 0.3, 0.7))
        g2 = _branch("fit", (0.3, 0.7, 0.4, 0.6, 0.45, 0.55))
        f1 = fit_residual_logit_fusion(
            "short_term", short, "g2", g2, optimization_steps=20
        )
        f2 = fit_residual_logit_fusion(
            "g2", g2, "short_term", short, optimization_steps=20
        )
        self.assertEqual(f1["anchor_name"], "short_term")
        self.assertEqual(f1["residual_name"], "g2")
        self.assertEqual(f2["anchor_name"], "g2")
        self.assertEqual(f2["residual_name"], "short_term")

    def test_residual_fusion_rejects_misaligned_samples(self):
        short = _branch("fit", (0.1, 0.8, 0.2, 0.9))
        g2 = _branch("other", (0.3, 0.7, 0.4, 0.6))
        with self.assertRaisesRegex(ValueError, "same samples"):
            fit_residual_logit_fusion(
                "short_term", short, "g2", g2, optimization_steps=2
            )

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
