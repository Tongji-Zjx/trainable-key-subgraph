from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

from keysubgraph.crossfit.theory_guided_runner import (
    build_stage0_crossfit_commands,
)


class TheoryGuidedRunnerTest(unittest.TestCase):
    def test_stage0_plan_is_fold_local_ordered_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = build_stage0_crossfit_commands(
                root / "project",
                root / "source",
                root / "output",
                folds=(0, 1, 2),
                fold_bootstrap_repeats=7,
                pooled_bootstrap_repeats=11,
            )
        self.assertEqual(
            [item[0] for item in commands],
            [
                "stage0_fold_0",
                "stage0_fold_1",
                "stage0_fold_2",
                "stage0_pooled_summary",
            ],
        )
        for fold, (_, command, artifact) in enumerate(commands[:3]):
            self.assertIn("fold_{}".format(fold), " ".join(command))
            self.assertIn("--hard-train-manifest", command)
            self.assertIn("--hard-test-manifest", command)
            self.assertEqual(artifact.name, "manifest.json")
        pooled = commands[-1][1]
        self.assertIn("--fold-dirs", pooled)
        self.assertEqual(commands[-1][2].name, "pooled_metrics.json")

    def test_stage0_plan_rejects_duplicate_folds(self):
        with self.assertRaises(ValueError):
            build_stage0_crossfit_commands(
                Path("project"), Path("source"), Path("output"), folds=(0, 0)
            )


if __name__ == "__main__":
    unittest.main()
