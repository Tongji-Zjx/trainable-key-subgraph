from __future__ import absolute_import, division, print_function

import unittest

from keysubgraph.models.dual_stse_hard_sgw_types import (
    DUAL_EXPERIMENT_VARIANTS,
    DUAL_TRAINING_STAGES,
    DualSTSEHardSGWConfig,
)


class DualSTSEContractTest(unittest.TestCase):
    def test_default_contract_freezes_verified_dimensions(self):
        config = DualSTSEHardSGWConfig()
        self.assertEqual(config.stse_input_dim, 18)
        self.assertEqual(config.stse_output_dim, 64)
        self.assertEqual(config.selector_node_feature_dim, 15)
        self.assertEqual(config.selector_edge_base_dim, 6)
        self.assertEqual(config.sgw_core_dim, 18)
        self.assertEqual(config.sgw_output_dim, 34)
        self.assertEqual(config.fusion_input_dim, 128)
        self.assertTrue(config.exact_sgw_detached)
        self.assertFalse(config.use_learned_temporal_encoder)

    def test_invalid_contract_cannot_silently_change_architecture(self):
        with self.assertRaisesRegex(ValueError, "18 -> 64"):
            DualSTSEHardSGWConfig(stse_output_dim=32)
        with self.assertRaisesRegex(ValueError, "15-D and 6-D"):
            DualSTSEHardSGWConfig(selector_node_feature_dim=12)
        with self.assertRaisesRegex(ValueError, "detached"):
            DualSTSEHardSGWConfig(exact_sgw_detached=False)
        with self.assertRaisesRegex(ValueError, "temporal"):
            DualSTSEHardSGWConfig(use_learned_temporal_encoder=True)

    def test_experiment_and_stage_names_are_explicit(self):
        self.assertEqual(DUAL_EXPERIMENT_VARIANTS, ("D0", "D1", "D2", "D3", "D4"))
        self.assertEqual(
            DUAL_TRAINING_STAGES,
            ("selector_proxy", "sgw_classifier", "fusion"),
        )


if __name__ == "__main__":
    unittest.main()
