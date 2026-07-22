from __future__ import absolute_import, division, print_function

import unittest

from keysubgraph.models.tg_sgw_types import (
    TG_SGW_MODEL_NAME,
    TG_SGW_SOFT_TEACHER_STAGE,
    TGSGWContract,
    TGSGWTheoryConfig,
    validate_tg_sgw_checkpoint_header,
)


class TGSGWContractTest(unittest.TestCase):
    def test_default_contract_matches_verified_dimensions(self):
        contract = TGSGWContract()
        self.assertEqual(len(contract.theory.spectral_quantile_grid), 16)
        self.assertEqual(contract.theory.core_feature_dim, 18)
        self.assertEqual(contract.theory.classification_feature_dim, 34)
        self.assertEqual(contract.dimensions.final_representation_dim, 226)
        self.assertEqual(contract.theory.time_quantity, "speed")
        self.assertTrue(contract.theory.laplacian_add_eta_to_numerator)

    def test_noncanonical_theory_choices_are_rejected(self):
        with self.assertRaises(ValueError):
            TGSGWTheoryConfig(time_quantity="increment")
        with self.assertRaises(ValueError):
            TGSGWTheoryConfig(laplacian_add_eta_to_numerator=False)
        with self.assertRaises(ValueError):
            TGSGWTheoryConfig(gw_external_half_factor=True)

    def test_checkpoint_header_rejects_legacy_and_wrong_stage(self):
        good = {
            "model_name": TG_SGW_MODEL_NAME,
            "schema_version": 1,
            "stage": TG_SGW_SOFT_TEACHER_STAGE,
        }
        validate_tg_sgw_checkpoint_header(good, TG_SGW_SOFT_TEACHER_STAGE)
        with self.assertRaises(ValueError):
            validate_tg_sgw_checkpoint_header({}, TG_SGW_SOFT_TEACHER_STAGE)
        with self.assertRaises(ValueError):
            validate_tg_sgw_checkpoint_header(good, "hard_student")


if __name__ == "__main__":
    unittest.main()
