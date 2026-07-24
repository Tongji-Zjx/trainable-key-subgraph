from __future__ import absolute_import, division, print_function

import unittest

from keysubgraph.models.hard_stse_types import (
    HardSelectionSchedule,
    HardSTSEConfig,
)


class HardSTSETypesTest(unittest.TestCase):
    def test_schedule_is_constant_then_linear_then_fixed(self):
        schedule = HardSelectionSchedule()
        self.assertEqual(schedule.ratios(1), (0.90, 0.80))
        self.assertEqual(schedule.ratios(10), (0.90, 0.80))
        middle = schedule.ratios(20)
        self.assertAlmostEqual(middle[0], 0.70)
        self.assertAlmostEqual(middle[1], 0.55)
        self.assertEqual(schedule.ratios(30), (0.50, 0.30))
        self.assertEqual(schedule.ratios(100), (0.50, 0.30))

    def test_variants_freeze_selection_and_theory_contract(self):
        valid = (
            HardSTSEConfig(variant="M0", selection_mode="full", use_sgw=False),
            HardSTSEConfig(variant="M1", selection_mode="random", use_sgw=False),
            HardSTSEConfig(variant="M2", selection_mode="learned", use_sgw=False),
            HardSTSEConfig(variant="M3", selection_mode="learned", use_sgw=True),
        )
        self.assertEqual(tuple(item.variant for item in valid), ("M0", "M1", "M2", "M3"))
        with self.assertRaisesRegex(ValueError, "disagree"):
            HardSTSEConfig(variant="M3", selection_mode="learned", use_sgw=False)

    def test_generic_model_rejects_identity_and_community_embeddings(self):
        with self.assertRaisesRegex(ValueError, "node identity"):
            HardSTSEConfig(use_node_identity_embedding=True)
        with self.assertRaisesRegex(ValueError, "community"):
            HardSTSEConfig(use_raw_community_embedding=True)


if __name__ == "__main__":
    unittest.main()
