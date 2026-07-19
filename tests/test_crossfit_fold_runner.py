from __future__ import absolute_import, print_function

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_runner import build_fold_commands  # noqa: E402


class CrossfitFoldRunnerTest(unittest.TestCase):
    def test_fold_zero_plan_contains_four_isolated_models_and_oof_predictions(self):
        commands = build_fold_commands(PROJECT_ROOT, 0, 42, "cuda", False)
        self.assertEqual(len(commands), 14)
        stages = [item[0] for item in commands]
        for variant in ("A", "B", "C", "D"):
            self.assertIn("train_{}".format(variant), stages)
            self.assertIn("evaluate_{}".format(variant), stages)
        checkpoints = [str(item[2]) for item in commands if item[0].startswith("train_")]
        self.assertEqual(len(set(checkpoints)), 4)
        node_only = [item[1] for item in commands if item[0] in ("train_C", "train_D")]
        self.assertTrue(all("node_only" in command for command in node_only))
        self.assertTrue(all("independent_bag" in command for stage, command, _ in commands if stage.startswith("train_")))
        export_markers = [
            str(artifact) for stage, _, artifact in commands
            if stage.startswith("key_export_")
        ]
        self.assertTrue(all("_completion" in marker for marker in export_markers))

    def test_smoke_flag_only_changes_training_commands(self):
        commands = build_fold_commands(PROJECT_ROOT, 0, 42, "cpu", True)
        smoke_stages = [stage for stage, command, _ in commands if "--smoke" in command]
        self.assertEqual(set(smoke_stages), {"extractor", "train_A", "train_B", "train_C", "train_D"})


if __name__ == "__main__":
    unittest.main()
