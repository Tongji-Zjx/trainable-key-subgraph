from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.dual_sgw_scaler import DualSGWStandardizer
from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_stse_hard_sgw import (
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (
    DualSTSEHardSGWCriterion,
)
from tests.test_exact_stse_model import _exact_sample


class DualSTSEHardSGWLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(337)
        self.batch = ExactSTSEBatch(
            (
                _exact_sample("loss-a", 0, 2),
                _exact_sample("loss-b", 1, 3),
            )
        )
        self.labels = torch.tensor([0, 1])
        self.model = DualSTSEHardSGWClassifier()
        self.model.set_sgw_standardizer(
            DualSGWStandardizer(
                torch.zeros(34),
                torch.ones(34),
                2,
                "protocol",
                "selector",
            )
        )
        self.criterion = DualSTSEHardSGWCriterion()

    def test_selector_loss_has_proxy_budget_and_fidelity_terms(self):
        output = self.model(
            self.batch, compute_selector_proxy=True
        )
        loss = self.criterion(
            output, self.labels, "selector_proxy"
        )
        self.assertGreater(float(loss.selector_proxy_ce), 0.0)
        self.assertGreaterEqual(float(loss.node_budget), 0.0)
        self.assertGreaterEqual(float(loss.edge_budget), 0.0)
        self.assertGreaterEqual(float(loss.laplacian), 0.0)
        self.assertGreaterEqual(float(loss.gw_proxy), 0.0)
        loss.total.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.selector.parameters()
            )
        )

    def test_sgw_and_fusion_stages_count_only_their_declared_heads(self):
        output = self.model(
            self.batch, exact_sgw_features=torch.randn(2, 34)
        )
        sgw = self.criterion(
            output, self.labels, "sgw_classifier"
        )
        self.assertTrue(torch.equal(sgw.total, sgw.sgw_ce))
        self.assertEqual(float(sgw.fusion_ce), 0.0)
        fusion = self.criterion(output, self.labels, "fusion")
        expected = (
            fusion.fusion_ce
            + 0.3 * fusion.stse_ce
            + 0.5 * fusion.sgw_ce
        )
        self.assertTrue(torch.allclose(fusion.total, expected))
        self.assertEqual(float(fusion.selector_proxy_ce), 0.0)

    def test_missing_stage_output_is_rejected(self):
        output = self.model(self.batch)
        with self.assertRaisesRegex(ValueError, "SGW logits"):
            self.criterion(output, self.labels, "fusion")
        with self.assertRaisesRegex(ValueError, "proxy logits"):
            self.criterion(output, self.labels, "selector_proxy")


if __name__ == "__main__":
    unittest.main()
