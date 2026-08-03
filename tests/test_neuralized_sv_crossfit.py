from __future__ import absolute_import, division, print_function

import unittest
from pathlib import Path

from scripts.summarize_neuralized_sv_short_term import (
    _paired_mean_fold_bootstrap,
)
from keysubgraph.crossfit.neuralized_sv_runner import (
    build_neuralized_sv_fold_commands,
    build_neuralized_sv_short_term_fusion_command,
)


class NeuralizedSVCrossfitTest(unittest.TestCase):
    def test_fold_plan_is_resumable_and_evaluates_validation_and_test(self):
        commands = build_neuralized_sv_fold_commands(
            Path("/project"),
            Path("/source"),
            Path("/output"),
            1,
            variants=("NSV_safe_residual",),
        )
        stages = [item[0] for item in commands]
        self.assertEqual(stages[:4], ["cache_train", "cache_validation", "cache_test", "fit_scaler"])
        self.assertIn("train_NSV_safe_residual", stages)
        self.assertIn("evaluate_NSV_safe_residual_validation", stages)
        self.assertIn("evaluate_NSV_safe_residual_test", stages)
        for _, _, artifact in commands:
            self.assertTrue(artifact.is_absolute())

    def test_fusion_fits_validation_and_only_evaluates_outer_test(self):
        _, command, artifact = build_neuralized_sv_short_term_fusion_command(
            Path("/project"),
            Path("/source"),
            Path("/neural"),
            Path("/fusion"),
            2,
            "NSV_safe_residual",
            short_term_seed=109,
            neural_seed=42,
        )
        joined = " ".join(command).replace("\\", "/")
        self.assertIn("evaluation_seed109/validation_evaluation.json", joined)
        self.assertIn("evaluation_seed109/test_evaluation.json", joined)
        self.assertIn("NSV_safe_residual_seed42/validation_evaluation.json", joined)
        self.assertIn("NSV_safe_residual_seed42/test_evaluation.json", joined)
        self.assertTrue(str(artifact).replace("\\", "/").endswith("evaluation.json"))

    def test_paired_bootstrap_uses_mean_fold_auc(self):
        reference = {}
        candidate = {}
        for fold in range(3):
            for index in range(20):
                label = index % 2
                key = "f{}-{}".format(fold, index)
                common = {
                    "fold": fold,
                    "site": "site{}".format(index % 2),
                    "label": label,
                }
                reference[key] = dict(
                    common, positive_probability=0.45 + 0.1 * (index % 3) / 2.0
                )
                candidate[key] = dict(
                    common, positive_probability=0.1 + 0.8 * label
                )
        result = _paired_mean_fold_bootstrap(
            reference, candidate, repeats=200, seed=7
        )
        self.assertGreater(result["mean"], 0.0)
        self.assertTrue(result["statistically_significant_positive"])


if __name__ == "__main__":
    unittest.main()
