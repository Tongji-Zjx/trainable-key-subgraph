from __future__ import absolute_import, division, print_function

import copy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.audit import (  # noqa: E402
    audit_fold_assignments, audit_oof_prediction_coverage,
    audit_perturbation_plan, audit_run_plan,
)


def _load(name):
    return json.loads((PROJECT_ROOT / "configs/crossfit" / name).read_text(encoding="utf-8"))


class CrossfitAuditTest(unittest.TestCase):
    def test_frozen_local_artifacts_pass(self):
        self.assertTrue(audit_fold_assignments(_load("fold_assignments.json"))["valid"])
        self.assertTrue(audit_run_plan(_load("oof_run_plan.json"))["valid"])
        self.assertTrue(audit_perturbation_plan(_load("perturbation_inference_plan.json"))["valid"])

    def test_run_and_perturbation_corruption_fail_closed(self):
        runs = _load("oof_run_plan.json")
        runs["runs"][0]["checkpoint"] = runs["runs"][1]["checkpoint"]
        with self.assertRaises(ValueError):
            audit_run_plan(runs)
        perturbations = _load("perturbation_inference_plan.json")
        perturbations["inferences"][1]["checkpoint"] = "wrong.pt"
        with self.assertRaises(ValueError):
            audit_perturbation_plan(perturbations)

    def test_prediction_coverage_rejects_missing_and_duplicate_rows(self):
        rows = []
        for sample in ("s1", "s2"):
            for seed in (42, 43):
                for variant in ("A", "B", "C", "D"):
                    rows.append({"sample_key": sample, "model_seed": seed, "variant": variant})
        self.assertTrue(audit_oof_prediction_coverage(rows, ("s1", "s2"), (42, 43))["valid"])
        with self.assertRaises(ValueError):
            audit_oof_prediction_coverage(rows[:-1], ("s1", "s2"), (42, 43))
        with self.assertRaises(ValueError):
            audit_oof_prediction_coverage(rows + [copy.deepcopy(rows[0])], ("s1", "s2"), (42, 43))


if __name__ == "__main__":
    unittest.main()
