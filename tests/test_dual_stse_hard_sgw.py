from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.dual_sgw_scaler import DualSGWStandardizer
from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_stse_hard_sgw import (
    DualSTSEHardSGWClassifier,
)
from tests.test_exact_stse_model import _exact_sample


class DualSTSEHardSGWModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(331)
        self.batch = ExactSTSEBatch(
            (
                _exact_sample("dual-a", 0, 2),
                _exact_sample("dual-b", 1, 3),
            )
        )
        self.model = DualSTSEHardSGWClassifier().eval()
        self.scaler = DualSGWStandardizer(
            mean=torch.zeros(34),
            scale=torch.ones(34),
            sample_count=10,
            protocol_sha256="protocol",
            selector_checkpoint_sha256="selector",
        )

    def test_fusion_shapes_and_stse_auxiliary_reuse(self):
        self.model.set_sgw_standardizer(self.scaler)
        features = torch.randn(2, 34)
        output = self.model(self.batch, exact_sgw_features=features)
        with torch.no_grad():
            expected = self.model.stse_channel.model(self.batch).logits
        self.assertTrue(torch.equal(output.stse_logits, expected))
        self.assertEqual(tuple(output.sgw_logits.shape), (2, 2))
        self.assertEqual(tuple(output.fusion_logits.shape), (2, 2))
        self.assertEqual(
            tuple(output.fusion_representation.shape), (2, 128)
        )
        self.assertEqual(output.diagnostics["uses_coordinates"], False)
        self.assertEqual(
            output.diagnostics["uses_learned_temporal_encoder"], False
        )

    def test_exact_features_require_train_only_scaler(self):
        with self.assertRaisesRegex(RuntimeError, "train-only scaler"):
            self.model(
                self.batch, exact_sgw_features=torch.randn(2, 34)
            )

    def test_selector_proxy_and_stage_freezing_are_explicit(self):
        self.model.set_stage_trainability("selector_proxy")
        output = self.model(
            self.batch, compute_selector_proxy=True
        )
        self.assertEqual(
            tuple(output.selector_proxy_logits.shape), (2, 2)
        )
        self.assertIsNotNone(output.diagnostics["proxy"])
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in self.model.stse_channel.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in self.model.selector.parameters()
            )
        )

    def test_sgw_changes_do_not_change_stse_auxiliary_logits(self):
        self.model.set_sgw_standardizer(self.scaler)
        first = self.model(
            self.batch, exact_sgw_features=torch.zeros(2, 34)
        )
        second = self.model(
            self.batch, exact_sgw_features=torch.ones(2, 34)
        )
        self.assertTrue(
            torch.equal(first.stse_logits, second.stse_logits)
        )
        self.assertFalse(
            torch.allclose(first.fusion_logits, second.fusion_logits)
        )


if __name__ == "__main__":
    unittest.main()
