from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.model_matrix import build_oof_run_plan  # noqa: E402
from keysubgraph.crossfit.perturbation_plan import (  # noqa: E402
    build_perturbation_inference_plan,
)
from keysubgraph.data.edge_perturbation import (  # noqa: E402
    _deletion_count,
    perturb_key_subgraph,
)
from tests.test_edge_perturbation import _subgraph  # noqa: E402


class CrossfitPerturbationPlanTest(unittest.TestCase):
    def test_confirmatory_plan_is_complete_and_never_retrains(self):
        run_plan = build_oof_run_plan(
            PROJECT_ROOT / "configs/crossfit/fold_assignments.json"
        )
        plan = build_perturbation_inference_plan(run_plan)
        self.assertEqual(plan["expected_inference_count"], 195)
        self.assertEqual(len(plan["inferences"]), 195)
        self.assertTrue(all(not row["retrain"] for row in plan["inferences"]))
        for fold in range(5):
            for seed in (42, 43, 44):
                rows = [row for row in plan["inferences"] if row["outer_fold"] == fold and row["model_seed"] == seed]
                self.assertEqual(len(rows), 13)
                self.assertEqual(len({row["checkpoint"] for row in rows}), 1)
                for dose in (0.25, 0.50):
                    random_rows = [row for row in rows if row["mode"] == "random" and row["dose"] == dose]
                    self.assertEqual({row["repeat_index"] for row in random_rows}, set(range(5)))

    def test_deletion_budget_uses_ceil_and_retains_an_edge(self):
        self.assertEqual(_deletion_count(5, 0.25), 2)
        self.assertEqual(_deletion_count(2, 0.50), 1)
        self.assertEqual(_deletion_count(2, 0.99), 1)

    def test_targeted_and_random_have_identical_dose_budget(self):
        key = _subgraph()
        for dose in (0.25, 0.50):
            targeted = perturb_key_subgraph(key, "targeted", dose, "s", 0, 0, 2026)
            for repeat in range(5):
                random = perturb_key_subgraph(key, "random", dose, "s", 0, 0, 2026 + repeat)
                self.assertEqual(
                    targeted["edge_perturbation"]["deleted_edge_count"],
                    random["edge_perturbation"]["deleted_edge_count"],
                )


if __name__ == "__main__":
    unittest.main()
