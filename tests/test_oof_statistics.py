from __future__ import absolute_import, division, print_function

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.oof_statistics import (  # noqa: E402
    aggregate_subjects, bootstrap_subject_mean, compute_model_contrasts,
    compute_perturbation_contrasts, dose_slope,
)


def _model_rows(sample="s1", subject="u1", seed=42, probabilities=None):
    probabilities = probabilities or {"A": 0.8, "B": 0.6, "C": 0.7, "D": 0.5}
    return [{
        "outer_fold": 0, "model_seed": seed, "sample_key": sample,
        "subject_id": subject, "session_id": sample, "label": 1,
        "variant": variant, "class_1_probability": probability,
    } for variant, probability in probabilities.items()]


class OofStatisticsTest(unittest.TestCase):
    def test_exact_model_contrasts(self):
        row = compute_model_contrasts(_model_rows())[0]
        losses = {name: -math.log(value) for name, value in {"A": .8, "B": .6, "C": .7, "D": .5}.items()}
        self.assertAlmostEqual(row["dsc"], losses["B"] - losses["A"])
        self.assertAlmostEqual(row["seg"], losses["C"] - losses["A"])
        self.assertAlmostEqual(row["tpa"], (losses["B"] - losses["A"]) - (losses["D"] - losses["C"]))

    def test_incomplete_or_duplicate_pairing_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_model_contrasts(_model_rows()[:-1])
        with self.assertRaises(ValueError):
            compute_model_contrasts(_model_rows() + [_model_rows()[0]])

    def test_sessions_then_seeds_are_averaged_before_bootstrap(self):
        contrasts = []
        for seed in (42, 43):
            contrasts.extend(compute_model_contrasts(_model_rows("session1", "subject", seed)))
            contrasts.extend(compute_model_contrasts(_model_rows("session2", "subject", seed)))
        subjects = aggregate_subjects(contrasts, ("dsc", "seg", "tpa"))
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0]["seed_count"], 2)
        result = bootstrap_subject_mean(subjects, "tpa", repeats=100, seed=9)
        self.assertEqual(result["subject_count"], 1)

    def test_random_repeats_are_averaged_and_dose_slope_is_exact(self):
        rows = []
        for seed in (42, 43):
            rows.append({"outer_fold": 0, "model_seed": seed, "sample_key": "s", "subject_id": "u", "session_id": "1", "label": 1, "dose": 0.0, "mode": "none", "repeat_index": None, "class_1_probability": 0.8})
            for dose, targeted_probability in ((0.25, 0.6), (0.50, 0.4)):
                rows.append({"outer_fold": 0, "model_seed": seed, "sample_key": "s", "subject_id": "u", "session_id": "1", "label": 1, "dose": dose, "mode": "targeted", "repeat_index": None, "class_1_probability": targeted_probability})
                for repeat in range(5):
                    rows.append({"outer_fold": 0, "model_seed": seed, "sample_key": "s", "subject_id": "u", "session_id": "1", "label": 1, "dose": dose, "mode": "random", "repeat_index": repeat, "class_1_probability": 0.7})
        contrasts = compute_perturbation_contrasts(rows)
        self.assertEqual(len(contrasts), 4)
        subjects = aggregate_subjects(contrasts, ("targeted_damage", "random_damage", "dose_contrast"))
        slopes = dose_slope(subjects)
        expected = (0.25 * math.log(0.7 / 0.6) + 0.50 * math.log(0.7 / 0.4)) / (0.25 ** 2 + 0.50 ** 2)
        self.assertAlmostEqual(slopes[0]["dose_slope"], expected)
        with self.assertRaises(ValueError):
            compute_perturbation_contrasts(rows[:-1])


if __name__ == "__main__":
    unittest.main()
