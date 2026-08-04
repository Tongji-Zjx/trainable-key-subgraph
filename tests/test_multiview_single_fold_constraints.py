from __future__ import absolute_import, division, print_function

import unittest

from scripts.summarize_multiview_stage_conditions import (
    _formally_admissible,
    _uot_promotion_passes,
)


class MultiviewSingleFoldConstraintTest(unittest.TestCase):
    def test_stable_static_is_control_only(self):
        row = {"static_mode": "stable", "v": "none"}
        self.assertFalse(_formally_admissible(row, "stage1"))
        self.assertFalse(_formally_admissible(row, "stage3"))

    def test_neural_and_residual_static_are_formal(self):
        self.assertTrue(
            _formally_admissible(
                {"static_mode": "neural", "v": "none"}, "stage1"
            )
        )
        self.assertTrue(
            _formally_admissible(
                {"static_mode": "residual", "v": "none"}, "stage1"
            )
        )

    def test_only_none_and_uot_v_are_formal(self):
        for mode in ("none", "uot"):
            self.assertTrue(
                _formally_admissible(
                    {"static_mode": "residual", "v": mode}, "stage2"
                )
            )
        for mode in ("legacy", "shuffled"):
            self.assertFalse(
                _formally_admissible(
                    {"static_mode": "residual", "v": mode}, "stage2"
                )
            )

    def test_uot_promotion_compares_against_no_v(self):
        self.assertTrue(_uot_promotion_passes(0.0036))
        self.assertFalse(_uot_promotion_passes(0.0))
        self.assertFalse(_uot_promotion_passes(-0.001))
        self.assertFalse(_uot_promotion_passes(None))


if __name__ == "__main__":
    unittest.main()
