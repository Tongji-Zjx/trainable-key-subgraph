from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from keysubgraph.analysis.dual_frozen_feature_masking import (
    apply_frozen_feature_mask,
    build_frozen_feature_mask_evaluation,
    write_frozen_feature_mask_artifacts,
)


class DualFrozenFeatureMaskingTest(unittest.TestCase):
    def test_a_to_e_masks_use_only_train_mean(self):
        values = np.arange(68, dtype=np.float64).reshape(2, 34)
        mean = np.linspace(-1.0, 1.0, 34)
        masked = {
            code: apply_frozen_feature_mask(values, mean, code)
            for code in ("A", "B", "C", "D", "E")
        }
        np.testing.assert_array_equal(masked["A"], values)
        np.testing.assert_allclose(
            masked["B"][:, :18],
            np.broadcast_to(mean[:18], (2, 18)),
        )
        np.testing.assert_array_equal(masked["B"][:, 18:], values[:, 18:])
        np.testing.assert_array_equal(masked["C"][:, :16], values[:, :16])
        np.testing.assert_allclose(
            masked["C"][:, 16:18],
            np.broadcast_to(mean[16:18], (2, 2)),
        )
        np.testing.assert_array_equal(masked["C"][:, 18:], values[:, 18:])
        np.testing.assert_array_equal(masked["D"][:, :17], values[:, :17])
        np.testing.assert_allclose(masked["D"][:, 17], mean[17])
        np.testing.assert_array_equal(masked["D"][:, 18:], values[:, 18:])
        np.testing.assert_array_equal(masked["C"], masked["E"])

    def _evaluation(self):
        labels = [0, 1, 0, 1, 0, 1]
        base = np.asarray([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        return build_frozen_feature_mask_evaluation(
            ["sample_{}".format(index) for index in range(6)],
            labels,
            {
                "A": base,
                "B": np.asarray([0.3, 0.7, 0.4, 0.6, 0.5, 0.55]),
                "C": np.asarray([0.25, 0.75, 0.35, 0.65, 0.45, 0.58]),
                "D": np.asarray([0.22, 0.78, 0.32, 0.68, 0.42, 0.59]),
                "E": np.asarray([0.25, 0.75, 0.35, 0.65, 0.45, 0.58]),
            },
        )

    def test_evaluation_is_validation_only_and_uses_shared_threshold(self):
        evaluation = self._evaluation()
        self.assertFalse(evaluation["test_split_used"])
        self.assertEqual(evaluation["updated_parameter_count"], 0)
        self.assertTrue(evaluation["duplicate_condition_check"]["passed"])
        self.assertEqual(len(evaluation["conditions"]), 5)
        thresholds = {
            row["metrics"]["threshold"] for row in evaluation["conditions"]
        }
        self.assertEqual(len(thresholds), 1)
        self.assertIn(
            "both_speeds_incremental_auc_A_minus_E",
            evaluation["contrasts"],
        )

    def test_artifacts_are_complete_and_immutable(self):
        evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "masking"
            paths = write_frozen_feature_mask_artifacts(
                output,
                evaluation,
                {"test_split_used": False},
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(
                paths["evaluation"].read_text(encoding="utf-8")
            )
            self.assertFalse(payload["test_split_used"])
            with self.assertRaises(FileExistsError):
                write_frozen_feature_mask_artifacts(
                    output, evaluation, {}
                )

    def test_c_and_e_mismatch_is_rejected(self):
        labels = [0, 1, 0, 1]
        base = [0.2, 0.8, 0.3, 0.7]
        with self.assertRaisesRegex(ValueError, "C and E"):
            build_frozen_feature_mask_evaluation(
                ["a", "b", "c", "d"],
                labels,
                {
                    "A": base,
                    "B": base,
                    "C": base,
                    "D": base,
                    "E": [0.2, 0.8, 0.3, 0.6],
                },
            )


if __name__ == "__main__":
    unittest.main()
