from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.dual_variation_only_exact_head import (
    build_variation_only_exact_head_evaluation,
    write_variation_only_exact_head_artifacts,
)


class DualVariationOnlyExactHeadTest(unittest.TestCase):
    def _evaluation(self):
        return build_variation_only_exact_head_evaluation(
            validation_sample_keys=("v0", "v1", "v2", "v3", "v4", "v5"),
            validation_labels=(0, 1, 0, 1, 0, 1),
            validation_probabilities=(0.2, 0.8, 0.4, 0.7, 0.5, 0.6),
            test_sample_keys=("t0", "t1", "t2", "t3"),
            test_labels=(0, 1, 0, 1),
            test_probabilities=(0.3, 0.9, 0.45, 0.65),
        )

    def test_validation_thresholds_are_frozen_for_test(self):
        evaluation = self._evaluation()
        self.assertEqual(
            evaluation["architecture"]["path"], "B_variation_only"
        )
        self.assertFalse(
            evaluation["architecture"][
                "all_34_proxy_path_removed_or_modified"
            ]
        )
        for policy, threshold in evaluation["thresholds"].items():
            self.assertEqual(
                evaluation["validation"]["metrics"][policy]["threshold"],
                threshold,
            )
            self.assertEqual(
                evaluation["test"]["metrics"][policy]["threshold"],
                threshold,
            )
        self.assertEqual(
            evaluation["architecture"]["updated_parameter_count"], 0
        )

    def test_artifacts_are_complete_and_immutable(self):
        evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variation_only"
            paths = write_variation_only_exact_head_artifacts(
                output,
                evaluation,
                {"read_only_frozen_models": True},
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            specification = json.loads(
                paths["model_spec"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                specification["architecture"][
                    "retained_raw_dimensions"
                ],
                [18, 34],
            )
            with self.assertRaises(FileExistsError):
                write_variation_only_exact_head_artifacts(
                    output, evaluation, {}
                )

    def test_partition_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_variation_only_exact_head_evaluation(
                validation_sample_keys=("same", "v1"),
                validation_labels=(0, 1),
                validation_probabilities=(0.2, 0.8),
                test_sample_keys=("same", "t1"),
                test_labels=(0, 1),
                test_probabilities=(0.3, 0.7),
            )

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            build_variation_only_exact_head_evaluation(
                validation_sample_keys=("v0", "v1"),
                validation_labels=(0, 1),
                validation_probabilities=(0.2, 1.2),
                test_sample_keys=("t0", "t1"),
                test_labels=(0, 1),
                test_probabilities=(0.3, 0.7),
            )


if __name__ == "__main__":
    unittest.main()
