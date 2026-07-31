from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

from keysubgraph.crossfit.theory_guided_runner import (
    build_stage0_crossfit_commands,
    build_stage1_fold_commands,
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

    def test_stage1_plan_is_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = build_stage1_fold_commands(
                root, root / "source", root / "output", 1,
                variants=("N0_signed_gin", "N4_ema_center"),
                device="cpu", epochs=2, batch_size=4,
                accumulation_steps=2, num_workers=0,
                gw_max_iter=2, gw_sinkhorn_iter=3,
            )
        names = [item[0] for item in commands]
        self.assertEqual(names[:4], [
            "cache_train", "cache_validation", "cache_test", "fit_scaler"
        ])
        self.assertIn("train_N0_signed_gin", names)
        self.assertIn("evaluate_N4_ema_center", names)
        self.assertIn("diagnose_N4_ema_center", names)
        self.assertEqual(len({str(item[2]) for item in commands}), len(commands))

    def test_formal_effective_batch_is_enforced(self):
        with self.assertRaises(ValueError):
            build_stage1_fold_commands(
                Path("."), Path("source"), Path("output"), 0,
                batch_size=2, accumulation_steps=2,
            )


if __name__ == "__main__":
    unittest.main()
