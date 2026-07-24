from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from keysubgraph.models.dual_sgw_proxy import DualSGWProxy
from tests.test_exact_stse_model import _exact_sample


class DualSGWProxyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(313)
        self.batch = ExactSTSEBatch(
            (
                _exact_sample("proxy-a", 0, 2),
                _exact_sample("proxy-b", 1, 3),
            )
        )
        self.selector = DualHardSGWSelector()
        self.proxy = DualSGWProxy()

    def test_proxy_is_34_dimensional_masked_and_finite(self):
        selected = self.selector(self.batch)
        output = self.proxy(self.batch, selected.hard_windows)
        self.assertEqual(tuple(output.core.shape), (2, 18))
        self.assertEqual(tuple(output.variation.shape), (2, 16))
        self.assertEqual(tuple(output.representation.shape), (2, 34))
        self.assertEqual(tuple(output.transition_mask.shape), (2, 2))
        self.assertEqual(output.diagnostics["is_exact_gw"], False)
        self.assertTrue(torch.isfinite(output.representation).all())
        self.assertTrue(torch.isfinite(output.laplacian_fidelity))
        self.assertTrue(torch.isfinite(output.gw_fidelity))

    def test_proxy_classification_gradient_reaches_both_scorers(self):
        selected = self.selector(self.batch)
        output = self.proxy(self.batch, selected.hard_windows)
        classifier = torch.nn.Linear(34, 2)
        labels = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(
            classifier(output.representation), labels
        )
        loss = (
            loss
            + 0.01 * output.laplacian_fidelity
            + 0.01 * output.gw_fidelity
        )
        loss.backward()
        node_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.selector.scorer.node_scorer.parameters()
            if parameter.grad is not None
        )
        edge_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.selector.scorer.edge_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(node_gradient, 0.0)
        self.assertGreater(edge_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
