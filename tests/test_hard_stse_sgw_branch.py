from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.models.hard_stse_sgw_branch import HardSTSESGWBranch
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from tests.test_full_graph_classifier import _sample


class HardSTSESGWBranchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(149)
        self.config = HardSTSEConfig(
            variant="M3",
            selection_mode="learned",
            use_sgw=True,
            dropout=0.0,
        )
        self.batch = GraphSequenceBatch(
            (_sample("sgw-a", 0, 2), _sample("sgw-b", 1, 3))
        )

    def _hard_windows(self):
        model = HardSTSETemporalSGWClassifier(self.config)
        _, windows, _ = model._encode_neural(self.batch, 30, 42)
        return windows

    def test_exact_feature_shapes_detachment_and_sequence_gradients(self):
        branch = HardSTSESGWBranch(self.config)
        windows = self._hard_windows()
        times = tuple(tuple(sample.window_starts) for sample in self.batch)
        output, diagnostics = branch(windows, times)
        self.assertEqual(tuple(output.core.shape), (2, 18))
        self.assertEqual(tuple(output.fixed.shape), (2, 34))
        self.assertEqual(tuple(output.sequence.shape), (2, 32))
        self.assertEqual(tuple(output.representation.shape), (2, 66))
        self.assertTrue(output.exact_features_detached)
        self.assertFalse(output.core.requires_grad)
        self.assertEqual(tuple(diagnostics["fixed_raw"].shape), (2, 34))
        output.sequence.square().sum().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in branch.sequence_gru.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_train_only_standardizer_contract(self):
        branch = HardSTSESGWBranch(self.config)
        with self.assertRaises(ValueError):
            branch.fit_fixed_standardizer(torch.zeros(2, 33))
        training = torch.stack(
            (torch.arange(34.0), torch.arange(34.0) + 2.0), dim=0
        )
        branch.fit_fixed_standardizer(training)
        self.assertTrue(bool(branch.fixed_scaler_fitted.item()))
        standardized = (
            training - branch.fixed_mean.unsqueeze(0)
        ) / branch.fixed_scale.unsqueeze(0)
        self.assertTrue(torch.allclose(
            standardized.mean(dim=0), torch.zeros(34), atol=1.0e-6
        ))


if __name__ == "__main__":
    unittest.main()
