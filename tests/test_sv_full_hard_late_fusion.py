from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.sv_full_hard_late_fusion import (
    build_sv_full_hard_late_fusion,
    write_sv_full_hard_late_fusion,
)


def _evaluation(split, mode, probabilities, labels=None):
    labels = labels or (0, 0, 1, 1)
    prefix = "v" if split == "validation" else "t"
    selector_sha = "hard-selector" if mode == "learned" else "none"
    return {
        "artifact_type": "sv_hard_sgw_signed_gin_evaluation",
        "split": split,
        "variant": "sv_static_variation",
        "checkpoint_sha256": mode + "-checkpoint",
        "manifest_sha256": mode + "-" + split + "-manifest",
        "scaler_sha256": mode + "-scaler",
        "provenance": {
            "protocol_sha256": "protocol",
            "selector_checkpoint_sha256": selector_sha,
            "selection_mode": mode,
            "selection_seed": 42,
            "training_seed": 42,
        },
        "predictions": [
            {
                "sample_key": "{}{}".format(prefix, index),
                "site": "site-{}".format(index % 2),
                "label": int(label),
                "positive_probability": float(probability),
            }
            for index, (label, probability) in enumerate(
                zip(labels, probabilities)
            )
        ],
    }


class SVFullHardLateFusionTest(unittest.TestCase):
    def _payload(self, test_labels=None):
        return build_sv_full_hard_late_fusion(
            _evaluation(
                "validation",
                "learned",
                (0.10, 0.70, 0.60, 0.90),
            ),
            _evaluation(
                "test",
                "learned",
                (0.20, 0.65, 0.55, 0.80),
                labels=test_labels,
            ),
            _evaluation(
                "validation",
                "full",
                (0.20, 0.30, 0.80, 0.40),
            ),
            _evaluation(
                "test",
                "full",
                (0.25, 0.35, 0.75, 0.45),
                labels=test_labels,
            ),
        )

    def test_weight_and_threshold_are_frozen_on_validation(self):
        payload = self._payload()
        changed_test = self._payload(test_labels=(1, 1, 0, 0))
        self.assertFalse(payload["test_used_for_selection"])
        self.assertEqual(
            payload["selected_hard_weight"],
            changed_test["selected_hard_weight"],
        )
        self.assertEqual(payload["thresholds"], changed_test["thresholds"])
        self.assertIn(payload["selected_hard_weight"], payload["alpha_grid"])
        self.assertEqual(payload["updated_parameter_count"], 0)

    def test_sources_and_samples_must_match(self):
        hard_validation = _evaluation(
            "validation", "full", (0.1, 0.2, 0.8, 0.9)
        )
        with self.assertRaisesRegex(ValueError, "hard branch must use"):
            build_sv_full_hard_late_fusion(
                hard_validation,
                _evaluation(
                    "test", "full", (0.1, 0.2, 0.8, 0.9)
                ),
                _evaluation(
                    "validation", "full", (0.2, 0.3, 0.7, 0.8)
                ),
                _evaluation(
                    "test", "full", (0.2, 0.3, 0.7, 0.8)
                ),
            )

    def test_artifacts_are_complete_and_immutable(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fusion"
            paths = write_sv_full_hard_late_fusion(payload, output)
            self.assertEqual(len(paths), 5)
            self.assertTrue(
                all(Path(path).is_file() for path in paths.values())
            )
            with self.assertRaises(FileExistsError):
                write_sv_full_hard_late_fusion(payload, output)


if __name__ == "__main__":
    unittest.main()
