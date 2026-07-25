from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.dual_frozen_logit_ensemble import (
    build_frozen_equal_logit_ensemble,
    write_frozen_equal_logit_ensemble_artifacts,
)


def _component(keys, labels, probabilities):
    return {
        "sample_keys": keys,
        "labels": labels,
        "probabilities": probabilities,
    }


class DualFrozenLogitEnsembleTest(unittest.TestCase):
    def _evaluation(self, test_labels=(0, 1, 0, 1)):
        validation = {
            "proxy": _component(
                ("v0", "v1", "v2", "v3"),
                (0, 0, 1, 1),
                (0.1, 0.3, 0.7, 0.9),
            ),
            "exact": _component(
                ("v3", "v2", "v1", "v0"),
                (1, 1, 0, 0),
                (0.8, 0.6, 0.4, 0.2),
            ),
        }
        test = {
            "proxy": _component(
                ("t0", "t1", "t2", "t3"),
                test_labels,
                (0.2, 0.8, 0.4, 0.6),
            ),
            "exact": _component(
                ("t2", "t0", "t3", "t1"),
                (
                    test_labels[2],
                    test_labels[0],
                    test_labels[3],
                    test_labels[1],
                ),
                (0.3, 0.1, 0.7, 0.9),
            ),
        }
        return build_frozen_equal_logit_ensemble(
            validation,
            test,
            ensemble_scope="unit_test",
        )

    def test_equal_logit_ensemble_aligns_keys_and_freezes_validation(self):
        evaluation = self._evaluation()
        self.assertEqual(evaluation["component_count"], 2)
        self.assertEqual(evaluation["weight_per_component"], 0.5)
        self.assertEqual(evaluation["updated_parameter_count"], 0)
        self.assertEqual(
            evaluation["normalization_fit_split"], "validation"
        )
        self.assertEqual(
            evaluation["threshold_fit_split"], "validation"
        )
        self.assertEqual(
            [
                row["sample_key"]
                for row in evaluation["test"]["predictions"]
            ],
            ["t0", "t1", "t2", "t3"],
        )
        changed_test = self._evaluation(test_labels=(1, 0, 1, 0))
        self.assertEqual(
            evaluation["thresholds"], changed_test["thresholds"]
        )
        self.assertEqual(
            evaluation["normalization"],
            changed_test["normalization"],
        )

    def test_artifacts_are_complete_and_immutable(self):
        evaluation = self._evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ensemble"
            paths = write_frozen_equal_logit_ensemble_artifacts(
                output,
                evaluation,
                {"read_only_frozen_predictions": True},
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(
                paths["model_spec"].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["updated_parameter_count"], 0)
            self.assertEqual(payload["weight_per_component"], 0.5)
            with self.assertRaises(FileExistsError):
                write_frozen_equal_logit_ensemble_artifacts(
                    output, evaluation, {}
                )

    def test_component_coverage_mismatch_is_rejected(self):
        validation = {
            "a": _component(
                ("v0", "v1", "v2", "v3"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            ),
            "b": _component(
                ("v0", "v1", "v2", "other"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            ),
        }
        test = {
            "a": _component(
                ("t0", "t1", "t2", "t3"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            ),
            "b": _component(
                ("t0", "t1", "t2", "t3"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            ),
        }
        with self.assertRaisesRegex(ValueError, "different samples"):
            build_frozen_equal_logit_ensemble(
                validation, test, ensemble_scope="invalid"
            )

    def test_zero_variance_component_is_rejected(self):
        validation = {
            "a": _component(
                ("v0", "v1", "v2", "v3"),
                (0, 0, 1, 1),
                (0.5, 0.5, 0.5, 0.5),
            ),
            "b": _component(
                ("v0", "v1", "v2", "v3"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            ),
        }
        test = {
            name: _component(
                ("t0", "t1", "t2", "t3"),
                (0, 0, 1, 1),
                (0.1, 0.2, 0.8, 0.9),
            )
            for name in ("a", "b")
        }
        with self.assertRaisesRegex(ValueError, "zero variance"):
            build_frozen_equal_logit_ensemble(
                validation, test, ensemble_scope="invalid"
            )


if __name__ == "__main__":
    unittest.main()
