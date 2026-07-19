from __future__ import absolute_import, print_function

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_analysis import analyze_fold_predictions  # noqa: E402
from tests.test_oof_statistics import _model_rows  # noqa: E402


class CrossfitFoldAnalysisTest(unittest.TestCase):
    def test_fold_summary_uses_subject_bootstrap_and_complete_perturbations(self):
        model_rows = _model_rows("s1", "u1", 42) + _model_rows("s2", "u2", 42)
        perturbations = []
        for sample, subject in (("s1", "u1"), ("s2", "u2")):
            perturbations.append({"outer_fold": 0, "model_seed": 42, "sample_key": sample, "subject_id": subject, "session_id": "1", "label": 1, "dose": 0.0, "mode": "none", "repeat_index": None, "class_1_probability": .8})
            for dose in (.25, .50):
                perturbations.append({"outer_fold": 0, "model_seed": 42, "sample_key": sample, "subject_id": subject, "session_id": "1", "label": 1, "dose": dose, "mode": "targeted", "repeat_index": None, "class_1_probability": .6})
                for repeat in range(5):
                    perturbations.append({"outer_fold": 0, "model_seed": 42, "sample_key": sample, "subject_id": subject, "session_id": "1", "label": 1, "dose": dose, "mode": "random", "repeat_index": repeat, "class_1_probability": .7})
        result = analyze_fold_predictions(model_rows, perturbations, 100, 7)
        self.assertEqual(result["coverage_audit"]["prediction_count"], 8)
        self.assertEqual(result["model_results"]["tpa"]["subject_count"], 2)
        self.assertEqual(result["dose_slope_result"]["subject_count"], 2)


if __name__ == "__main__":
    unittest.main()
