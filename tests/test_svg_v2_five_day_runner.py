from __future__ import absolute_import, division, print_function

import unittest
from pathlib import Path

from keysubgraph.crossfit.svg_v2_five_day_runner import (
    build_svg_v2_fold_commands,
)


class SVGv2FiveDayRunnerTest(unittest.TestCase):
    def test_screen_reuses_cache_and_never_reads_test(self):
        commands = build_svg_v2_fold_commands(
            Path("project"),
            Path("source"),
            Path("output"),
            fold=1,
            candidates=("A1", "C3", "G2"),
            mode="screen",
        )
        stages = [row[0] for row in commands]
        joined = " ".join(value for _, command, _ in commands for value in command)
        self.assertIn("spectral_cache_train", stages)
        self.assertIn("spectral_cache_validation", stages)
        self.assertNotIn("spectral_cache_test", stages)
        self.assertNotIn("/test/", joined.replace("\\", "/"))
        self.assertNotIn("train_dual_selector.py", joined)
        self.assertNotIn("precompute_sv_signed_gin_cache.py", joined)
        a1 = next(command for stage, command, _ in commands if stage == "train_A1")
        self.assertIn("author_a1", a1)

    def test_confirmatory_adds_test_only_after_training(self):
        commands = build_svg_v2_fold_commands(
            Path("project"),
            Path("source"),
            Path("output"),
            fold=0,
            candidates=("C3_F1",),
            mode="confirmatory",
        )
        stages = [row[0] for row in commands]
        self.assertIn("spectral_cache_test", stages)
        self.assertIn("evaluate_C3_F1", stages)
        train = next(
            command for stage, command, _ in commands if stage == "train_C3_F1"
        )
        self.assertIn("svg_v2_c3_f1_residual", train)


if __name__ == "__main__":
    unittest.main()
