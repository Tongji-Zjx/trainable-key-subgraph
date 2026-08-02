from __future__ import absolute_import, division, print_function

import unittest
from pathlib import Path

from keysubgraph.crossfit.svg_three_day_runner import (
    build_svg_three_day_fold_commands,
)


class SVGThreeDayRunnerTest(unittest.TestCase):
    def test_screen_reuses_selector_and_never_reads_test(self):
        commands = build_svg_three_day_fold_commands(
            Path("project"),
            Path("source"),
            Path("output"),
            fold=1,
            candidates=("D1", "H1", "E1"),
            mode="screen",
        )
        stages = [stage for stage, _, _ in commands]
        joined = " ".join(
            value for _, command, _ in commands for value in command
        ).replace("\\", "/")
        self.assertNotIn("train_dual_selector.py", joined)
        self.assertNotIn("/test/", joined)
        self.assertNotIn("cache_n50_e30_test", stages)
        self.assertIn("cache_n35_e20_train", stages)
        self.assertIn("cache_n50_e30_validation", stages)
        self.assertIn("cache_n65_e40_train", stages)

        h1 = next(
            command for stage, command, _ in commands if stage == "train_H1"
        )
        self.assertIn("--site-class-balanced-sampler", h1)
        self.assertIn("source/fold_1/cache/train/manifest.json", joined)

    def test_e1_freezes_three_budgets_and_equal_mean_variant(self):
        commands = build_svg_three_day_fold_commands(
            Path("project"),
            Path("source"),
            Path("output"),
            fold=0,
            candidates=("E1",),
            mode="confirmatory",
        )
        stages = [stage for stage, _, _ in commands]
        self.assertIn("cache_n35_e20_test", stages)
        self.assertIn("cache_n50_e30_test", stages)
        self.assertIn("cache_n65_e40_test", stages)
        train = next(
            command for stage, command, _ in commands if stage == "train_E1"
        )
        self.assertIn("svg_v2_e1_multi_budget", train)
        self.assertIn("--multi-budget-train-manifests", train)
        self.assertIn("--multi-budget-validation-manifests", train)
        self.assertIn("--multi-budget-scalers", train)
        evaluate = next(
            command
            for stage, command, _ in commands
            if stage == "evaluate_E1"
        )
        self.assertIn("--multi-budget-manifests", evaluate)
        self.assertEqual(
            sum("/test/manifest.json" in value.replace("\\", "/") for value in evaluate),
            4,
        )

    def test_only_frozen_combination_is_exposed(self):
        commands = build_svg_three_day_fold_commands(
            Path("project"),
            Path("source"),
            Path("output"),
            fold=2,
            candidates=("BASELINE", "D1_H1"),
            mode="confirmatory",
            seed=44,
            selection_seed=42,
        )
        stages = [stage for stage, _, _ in commands]
        self.assertIn("train_BASELINE", stages)
        self.assertIn("train_D1_H1", stages)
        combo = next(
            command
            for stage, command, _ in commands
            if stage == "train_D1_H1"
        )
        self.assertIn("svg_v2_d1_community_pooling", combo)
        self.assertIn("--site-class-balanced-sampler", combo)
        cache = next(
            command
            for stage, command, _ in commands
            if stage == "cache_n50_e30_train"
        )
        selection_seed_index = cache.index("--selection-seed") + 1
        self.assertEqual(cache[selection_seed_index], "42")


if __name__ == "__main__":
    unittest.main()
