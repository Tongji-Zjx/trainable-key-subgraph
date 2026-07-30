from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from keysubgraph.models.dual_sgw_proxy import DualSGWProxy
from keysubgraph.models.dual_stse_hard_sgw import (
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (
    DualSTSEHardSGWCriterion,
    DualSTSEHardSGWLossConfig,
)
from tests.test_exact_stse_model import _exact_sample


class DualSelectorTransferTest(unittest.TestCase):
    def setUp(self):
        self._rng_state = torch.get_rng_state()
        torch.manual_seed(20260730)
        self.samples = (
            _exact_sample("transfer-a", 0, 2),
            _exact_sample("transfer-b", 1, 3),
        )
        self.batch = ExactSTSEBatch(self.samples)

    def tearDown(self):
        torch.set_rng_state(self._rng_state)

    def test_soft_graph_is_same_node_signed_and_matches_formula(self):
        selector = DualHardSGWSelector()
        output = selector(self.batch, selection_mode="learned")
        for sample, soft_windows, hard_windows in zip(
            self.samples, output.soft_windows, output.hard_windows
        ):
            for time_index, (soft, hard) in enumerate(
                zip(soft_windows, hard_windows)
            ):
                adjacency = sample.graph.adjacency[time_index]
                selection = hard.selection
                expected = (
                    adjacency
                    * selection.node_probabilities[:, None]
                    * selection.node_probabilities[None, :]
                    * selection.edge_probabilities
                )
                expected = 0.5 * (expected + expected.transpose(0, 1))
                expected = expected.clone()
                expected.fill_diagonal_(0.0)
                self.assertEqual(
                    tuple(soft.adjacency_soft.shape),
                    tuple(adjacency.shape),
                )
                self.assertTrue(bool(soft.node_mask.all()))
                self.assertTrue(
                    torch.allclose(
                        soft.adjacency_soft,
                        expected,
                        atol=1.0e-7,
                        rtol=0.0,
                    )
                )
                negative = adjacency < 0.0
                positive = adjacency > 0.0
                self.assertTrue(
                    bool((soft.adjacency_soft[negative] < 0.0).all())
                )
                self.assertTrue(
                    bool((soft.adjacency_soft[positive] > 0.0).all())
                )

    def test_soft_to_hard_quantization_is_finite_and_differentiable(self):
        selector = DualHardSGWSelector()
        selected = selector(self.batch, selection_mode="learned")
        transfer = DualSGWProxy().compare_soft_and_hard(
            selected.soft_windows, selected.hard_windows
        )
        self.assertEqual(
            tuple(transfer.per_sample_spectral_quantization.shape),
            (2,),
        )
        self.assertEqual(
            tuple(transfer.per_sample_gw_quantization_proxy.shape),
            (2,),
        )
        self.assertTrue(torch.isfinite(transfer.spectral_quantization))
        self.assertTrue(torch.isfinite(transfer.gw_quantization_proxy))
        self.assertGreaterEqual(
            float(transfer.spectral_quantization), 0.0
        )
        self.assertGreaterEqual(
            float(transfer.gw_quantization_proxy), 0.0
        )
        (
            transfer.spectral_quantization
            + transfer.gw_quantization_proxy
        ).backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in selector.scorer.parameters()
            )
        )

    def test_full_soft_hard_objective_reaches_both_scorers(self):
        model = DualSTSEHardSGWClassifier()
        output = model(
            self.batch,
            compute_selector_proxy=True,
            selector_objective="full_soft_hard",
        )
        self.assertIsNotNone(output.selector_soft_proxy_logits)
        self.assertIsNotNone(output.selector_hard_proxy_logits)
        criterion = DualSTSEHardSGWCriterion(
            DualSTSEHardSGWLossConfig(
                selector_objective="full_soft_hard",
                soft_warmup_epochs=0,
            )
        )
        loss = criterion(
            output,
            torch.tensor([0, 1]),
            "selector_proxy",
            epoch=1,
        )
        self.assertGreater(float(loss.selector_soft_ce), 0.0)
        self.assertGreater(float(loss.selector_hard_ce), 0.0)
        self.assertGreaterEqual(float(loss.soft_hard_spectral), 0.0)
        self.assertGreaterEqual(float(loss.soft_hard_gw), 0.0)
        self.assertGreaterEqual(float(loss.soft_hard_kd), 0.0)
        loss.total.backward()
        node_gradients = [
            parameter.grad
            for parameter in model.selector.scorer.node_scorer.parameters()
        ]
        edge_gradients = [
            parameter.grad
            for parameter in model.selector.scorer.edge_scorer.parameters()
        ]
        self.assertTrue(
            any(
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                and float(gradient.abs().sum()) > 0.0
                for gradient in node_gradients
            )
        )
        self.assertTrue(
            any(
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                and float(gradient.abs().sum()) > 0.0
                for gradient in edge_gradients
            )
        )

    def test_current_objective_remains_the_default(self):
        model = DualSTSEHardSGWClassifier()
        output = model(self.batch, compute_selector_proxy=True)
        self.assertIsNone(output.selector_soft_proxy_logits)
        self.assertIsNotNone(output.selector_hard_proxy_logits)
        self.assertIs(output.diagnostics["proxy"], output.diagnostics["hard_proxy"])
        self.assertIsNone(output.diagnostics["soft_proxy"])
        self.assertIsNone(output.diagnostics["soft_hard_transfer"])


if __name__ == "__main__":
    unittest.main()
