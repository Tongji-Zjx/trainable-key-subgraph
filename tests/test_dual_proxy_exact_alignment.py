from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from keysubgraph.analysis.dual_proxy_exact_alignment import (
    analyze_proxy_exact_alignment,
    write_proxy_exact_alignment_artifacts,
)


class DualProxyExactAlignmentTest(unittest.TestCase):
    def _inputs(self):
        generator = np.random.RandomState(7)
        exact = generator.normal(size=(6, 34))
        proxy = exact.copy()
        proxy[:, 16] += 1.0
        proxy[:, 17] = exact[::-1, 17]
        labels = [0, 0, 0, 1, 1, 1]
        probabilities = {
            "proxy_proxy": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "exact_proxy": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
            "exact_exact": [0.2, 0.4, 0.3, 0.6, 0.8, 0.7],
            "proxy_exact": [0.3, 0.2, 0.4, 0.7, 0.6, 0.8],
        }
        proxy_masks = [
            np.asarray([True, True, False]) for _ in range(6)
        ]
        exact_masks = [
            np.asarray([True, True, False]) for _ in range(6)
        ]
        exact_masks[-1] = np.asarray([True, False, False])
        return {
            "sample_keys": [
                "sample-{}".format(index) for index in range(6)
            ],
            "labels": labels,
            "proxy_features": proxy,
            "exact_features": exact,
            "probabilities": probabilities,
            "proxy_threshold": 0.5,
            "exact_threshold": 0.5,
            "proxy_transition_masks": proxy_masks,
            "exact_transition_masks": exact_masks,
            "proxy_standardized": proxy,
            "exact_standardized": exact,
        }

    def test_alignment_localizes_the_constructed_feature_mismatch(self):
        result = analyze_proxy_exact_alignment(**self._inputs())
        blocks = result["summary"]["feature_blocks"]
        self.assertAlmostEqual(blocks["spectral_delta"]["rmse"], 0.0)
        self.assertAlmostEqual(blocks["variation"]["rmse"], 0.0)
        self.assertAlmostEqual(blocks["spectral_speed"]["rmse"], 1.0)
        self.assertGreater(blocks["gw_speed"]["rmse"], 0.0)
        self.assertAlmostEqual(
            result["summary"]["transition_masks"][
                "exact_sample_match_rate"
            ],
            5.0 / 6.0,
        )
        paths = result["summary"]["classification_paths"]
        self.assertAlmostEqual(paths["proxy_proxy"]["roc_auc"], 1.0)
        self.assertAlmostEqual(paths["exact_exact"]["roc_auc"], 1.0)
        self.assertEqual(len(result["dimension_rows"]), 34)
        self.assertEqual(
            result["dimension_rows"][17]["block"], "gw_speed"
        )

    def test_artifacts_are_complete_and_immutable(self):
        result = analyze_proxy_exact_alignment(**self._inputs())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_proxy_exact_alignment_artifacts(
                Path(directory) / "alignment",
                result,
                {"read_only": True, "protocol_sha256": "protocol"},
            )
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(
                paths["summary_json"].read_text(encoding="utf-8")
            )
            self.assertTrue(summary["provenance"]["read_only"])
            self.assertEqual(summary["sample_count"], 6)
            self.assertIn(
                "Proxy–Exact",
                paths["summary_markdown"].read_text(encoding="utf-8"),
            )
            with self.assertRaises(FileExistsError):
                write_proxy_exact_alignment_artifacts(
                    Path(directory) / "alignment",
                    result,
                    {"read_only": True},
                )

    def test_invalid_alignment_fails_closed(self):
        inputs = self._inputs()
        inputs["sample_keys"][-1] = inputs["sample_keys"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_proxy_exact_alignment(**inputs)


if __name__ == "__main__":
    unittest.main()
