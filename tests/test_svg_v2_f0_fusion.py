from __future__ import absolute_import, division, print_function

import unittest

from keysubgraph.analysis.svg_v2_f0_fusion import (
    apply_f0_fusion,
    fit_f0_fusion,
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


if __name__ == "__main__":
    unittest.main()
