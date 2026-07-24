from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.models.hard_stse_loss import (
    HardSTSECriterion,
    HardSTSELossConfig,
)
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from tests.test_full_graph_classifier import _sample


class HardSTSELossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(157)
        self.batch = GraphSequenceBatch(
            (_sample("loss-a", 0, 2), _sample("loss-b", 1, 3))
        )

    def test_m3_multihead_loss_and_detached_exact_features(self):
        config = HardSTSEConfig(
            variant="M3",
            selection_mode="learned",
            use_sgw=True,
            dropout=0.0,
        )
        model = HardSTSETemporalSGWClassifier(config)
        output = model(
            self.batch, epoch=40, compute_theory_proxies=True
        )
        self.assertEqual(tuple(output.final_representation.shape), (2, 258))
        self.assertEqual(tuple(output.theory_logits.shape), (2, 2))
        self.assertTrue(
            output.diagnostics["theory"].exact_features_detached
        )
        criterion = HardSTSECriterion(config)
        loss = criterion(
            output,
            self.batch.labels,
            epoch=40,
            class_weights=torch.tensor((0.7, 1.3)),
        )
        loss.total.backward()
        modules = (
            model.scorer.node_scorer,
            model.scorer.edge_scorer,
            model.window_encoder,
            model.temporal_encoder,
            model.sgw_branch.sequence_gru,
            model.fusion_head,
            model.theory_head,
        )
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_curriculum_enables_budget_then_theory(self):
        config = HardSTSEConfig(
            variant="M2",
            selection_mode="learned",
            use_sgw=False,
            dropout=0.0,
        )
        criterion = HardSTSECriterion(
            config,
            HardSTSELossConfig(theory_ramp_epochs=10),
        )
        first = criterion._curriculum_weights(1)
        middle = criterion._curriculum_weights(20)
        final = criterion._curriculum_weights(40)
        self.assertEqual(first["budget"], 0.0)
        self.assertEqual(first["laplacian"], 0.0)
        self.assertGreater(middle["budget"], 0.0)
        self.assertEqual(middle["laplacian"], 0.0)
        self.assertAlmostEqual(final["budget"], 0.10)
        self.assertAlmostEqual(final["laplacian"], 0.05)
        self.assertAlmostEqual(final["gw_proxy"], 0.02)


if __name__ == "__main__":
    unittest.main()
