from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.dual_transferred_head import (
    build_transferred_head_evaluation,
    write_transferred_head_artifacts,
)


class DualTransferredHeadTest(unittest.TestCase):
    def _evaluation(self, test_labels=None):
        return build_transferred_head_evaluation(
            validation_sample_keys=("v0", "v1", "v2", "v3"),
            validation_labels=(0, 0, 1, 1),
            validation_probabilities=(0.2, 0.4, 0.6, 0.8),
            test_sample_keys=("t0", "t1", "t2", "t3"),
            test_labels=(
                tuple(test_labels)
                if test_labels is not None
                else (0, 1, 0, 1)
            ),
            test_probabilities=(0.1, 0.7, 0.3, 0.9),
            original_proxy_threshold=0.375,
        )

    def test_validation_thresholds_are_frozen_for_test(self):
        evaluation = self._evaluation()
        self.assertEqual(
            evaluation["thresholds"]["balanced_accuracy"], 0.5
        )
        self.assertEqual(evaluation["thresholds"]["accuracy"], 0.5)
        self.assertEqual(
            evaluation["thresholds"]["original_proxy"], 0.375
        )
        self.assertEqual(
            evaluation["test"]["metrics"]["balanced_accuracy"][
                "threshold"
            ],
            evaluation["validation"]["metrics"]["balanced_accuracy"][
                "threshold"
            ],
        )
        self.assertEqual(
            evaluation["architecture"]["updated_parameter_count"], 0
        )
        changed_test = self._evaluation(test_labels=(1, 0, 1, 0))
        self.assertEqual(
            evaluation["thresholds"], changed_test["thresholds"]
        )

    def test_artifacts_are_complete_and_immutable(self):
        evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transferred"
            paths = write_transferred_head_artifacts(
                output,
                evaluation,
                {"read_only": True, "selector": "frozen"},
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(
                paths["evaluation"].read_text(encoding="utf-8")
            )
            self.assertTrue(payload["provenance"]["read_only"])
            self.assertFalse(
                payload["architecture"]["uses_exact_sgw_scaler"]
            )
            with self.assertRaises(FileExistsError):
                write_transferred_head_artifacts(
                    output, evaluation, {"read_only": True}
                )

    def test_partition_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_transferred_head_evaluation(
                validation_sample_keys=("same", "v1", "v2", "v3"),
                validation_labels=(0, 0, 1, 1),
                validation_probabilities=(0.2, 0.4, 0.6, 0.8),
                test_sample_keys=("same", "t1", "t2", "t3"),
                test_labels=(0, 1, 0, 1),
                test_probabilities=(0.1, 0.7, 0.3, 0.9),
            )


if __name__ == "__main__":
    unittest.main()
