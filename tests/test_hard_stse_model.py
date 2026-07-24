from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from tests.test_full_graph_classifier import _sample


class HardSTSEModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        self.batch = GraphSequenceBatch(
            (_sample("m0-a", 0, 2), _sample("m0-b", 1, 3))
        )
        self.model = HardSTSETemporalSGWClassifier(
            HardSTSEConfig(dropout=0.0)
        )

    def test_m0_complete_graph_forward_and_lengths(self):
        output = self.model(self.batch)
        self.assertEqual(tuple(output.fusion_logits.shape), (2, 2))
        self.assertEqual(tuple(output.neural_representation.shape), (2, 192))
        self.assertIsNone(output.theory_logits)
        self.assertAlmostEqual(
            output.diagnostics["actual_edge_candidate_ratio"], 1.0
        )
        self.assertAlmostEqual(
            output.diagnostics["actual_edge_original_ratio"], 1.0
        )
        self.assertEqual(len(output.hard_windows), 2)
        self.assertEqual(tuple(len(item) for item in output.hard_windows), (2, 3))
        for sample, windows in zip(self.batch, output.hard_windows):
            for time_index, window in enumerate(windows):
                self.assertTrue(torch.equal(
                    window.hard_edge_mask,
                    sample.edge_mask[time_index],
                ))

    def test_m0_classification_reaches_window_and_temporal_encoders(self):
        output = self.model(self.batch)
        torch.nn.functional.cross_entropy(
            output.fusion_logits, self.batch.labels
        ).backward()
        modules = (
            self.model.window_encoder.node_encoder,
            self.model.window_encoder.edge_encoder,
            self.model.window_encoder.window_mlp,
            self.model.temporal_encoder.bigru,
            self.model.neural_head,
        )
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_m1_random_hard_graph_is_frozen_by_selection_seed(self):
        model = HardSTSETemporalSGWClassifier(
            HardSTSEConfig(
                variant="M1",
                selection_mode="random",
                use_sgw=False,
                dropout=0.0,
            )
        )
        model.eval()
        with torch.no_grad():
            first = model(self.batch, random_selection_seed=29)
            second = model(self.batch, random_selection_seed=29)
        for left_sample, right_sample in zip(
            first.hard_windows, second.hard_windows
        ):
            for left, right in zip(left_sample, right_sample):
                self.assertTrue(torch.equal(
                    left.hard_edge_mask, right.hard_edge_mask
                ))

    def test_m2_classification_and_proxies_reach_both_scorers(self):
        model = HardSTSETemporalSGWClassifier(
            HardSTSEConfig(
                variant="M2",
                selection_mode="learned",
                use_sgw=False,
                dropout=0.0,
            )
        )
        output = model(
            self.batch,
            epoch=30,
            compute_theory_proxies=True,
        )
        loss = (
            torch.nn.functional.cross_entropy(
                output.fusion_logits, self.batch.labels
            )
            + 0.01 * output.diagnostics["laplacian_proxy"]
            + 0.01 * output.diagnostics["gw_proxy"]
        )
        loss.backward()
        for module in (model.scorer.node_scorer, model.scorer.edge_scorer):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)
        self.assertGreaterEqual(
            float(output.diagnostics["laplacian_proxy"]), 0.0
        )
        self.assertGreaterEqual(float(output.diagnostics["gw_proxy"]), 0.0)
        self.assertGreater(
            output.diagnostics["actual_edge_candidate_ratio"], 0.0
        )
        self.assertLessEqual(
            output.diagnostics["actual_edge_candidate_ratio"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
