from __future__ import absolute_import, division, print_function

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.model_matrix import (  # noqa: E402
    MODEL_VARIANTS,
    build_oof_run_plan,
    write_oof_run_plan,
)


class CrossfitModelMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assignments = PROJECT_ROOT / "configs/crossfit/fold_assignments.json"

    def test_minimal_matrix_has_exactly_sixty_isolated_runs(self):
        plan = build_oof_run_plan(self.assignments)
        self.assertEqual(len(MODEL_VARIANTS), 4)
        self.assertEqual(plan["expected_run_count"], 60)
        self.assertEqual(len(plan["runs"]), 60)
        self.assertEqual(len({row["run_id"] for row in plan["runs"]}), 60)
        self.assertEqual(len({row["checkpoint"] for row in plan["runs"]}), 60)

    def test_variants_share_data_and_training_budget_but_not_checkpoints(self):
        plan = build_oof_run_plan(self.assignments)
        fold_seed = [row for row in plan["runs"] if row["outer_fold"] == 2 and row["seed"] == 43]
        self.assertEqual({row["variant"] for row in fold_seed}, {"A", "B", "C", "D"})
        self.assertEqual({row["history_mode"] for row in fold_seed}, {"independent_bag"})
        self.assertEqual(len({row["control_inventory"] for row in fold_seed}), 1)
        self.assertEqual(len({row["extractor_checkpoint"] for row in fold_seed}), 1)
        self.assertEqual(len({row["checkpoint"] for row in fold_seed}), 4)
        for source in ("key", "random"):
            same_source = [row for row in fold_seed if row["source"] == source]
            self.assertEqual(len({row["train_manifest"] for row in same_source}), 1)
            self.assertEqual(len({row["test_manifest"] for row in same_source}), 1)

    def test_run_plan_is_immutable(self):
        plan = build_oof_run_plan(self.assignments)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            write_oof_run_plan(plan, path)
            with self.assertRaises(FileExistsError):
                write_oof_run_plan(plan, path)


if __name__ == "__main__":
    unittest.main()
