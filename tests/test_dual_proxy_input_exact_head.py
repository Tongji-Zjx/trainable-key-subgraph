from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.dual_proxy_input_exact_head import (
    build_proxy_input_exact_head_evaluation,
    write_proxy_input_exact_head_artifacts,
)


class DualProxyInputExactHeadTest(unittest.TestCase):
    def _evaluation(self, test_labels=None):
        return build_proxy_input_exact_head_evaluation(
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
        )

    def test_validation_thresholds_are_frozen_for_test(self):
        evaluation = self._evaluation()
        self.assertEqual(
            evaluation["thresholds"]["balanced_accuracy"], 0.5
        )
        self.assertEqual(evaluation["thresholds"]["accuracy"], 0.5)
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
        self.assertEqual(
            evaluation["architecture"]["normalization"],
            "exact_sgw_train_only_standardizer",
        )
        changed_test = self._evaluation(test_labels=(1, 0, 1, 0))
        self.assertEqual(
            evaluation["thresholds"], changed_test["thresholds"]
        )

    def test_artifacts_are_complete_immutable_and_deployable(self):
        evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proxy_input_exact_head"
            paths = write_proxy_input_exact_head_artifacts(
                output,
                evaluation,
                {"read_only_frozen_models": True, "selector": "sha"},
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(
                paths["evaluation"].read_text(encoding="utf-8")
            )
            spec = json.loads(
                paths["model_spec"].read_text(encoding="utf-8")
            )
            self.assertTrue(
                payload["provenance"]["read_only_frozen_models"]
            )
            self.assertEqual(
                spec["frozen_threshold"],
                evaluation["thresholds"]["balanced_accuracy"],
            )
            with self.assertRaises(FileExistsError):
                write_proxy_input_exact_head_artifacts(
                    output, evaluation, {"read_only_frozen_models": True}
                )

    def test_partition_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_proxy_input_exact_head_evaluation(
                validation_sample_keys=("same", "v1", "v2", "v3"),
                validation_labels=(0, 0, 1, 1),
                validation_probabilities=(0.2, 0.4, 0.6, 0.8),
                test_sample_keys=("same", "t1", "t2", "t3"),
                test_labels=(0, 1, 0, 1),
                test_probabilities=(0.1, 0.7, 0.3, 0.9),
            )

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            build_proxy_input_exact_head_evaluation(
                validation_sample_keys=("v0", "v1", "v2", "v3"),
                validation_labels=(0, 0, 1, 1),
                validation_probabilities=(0.2, 0.4, 0.6, 1.1),
                test_sample_keys=("t0", "t1", "t2", "t3"),
                test_labels=(0, 1, 0, 1),
                test_probabilities=(0.1, 0.7, 0.3, 0.9),
            )


if __name__ == "__main__":
    unittest.main()
