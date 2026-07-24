from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_stse_classification_features import (
    HardSTSEClassificationFeatureBuilder,
)
from keysubgraph.features.hard_stse_hard_graph import build_hard_stse_window
from keysubgraph.models.hard_stse_selector import select_hard_stse_window
from tests.test_full_graph_classifier import _sample


def _full_window(sample, time_index):
    count = sample.adjacency[time_index].shape[0]
    probabilities = torch.full((count,), 0.5, requires_grad=True)
    edges = torch.full((count, count), 0.5, requires_grad=True)
    selection = select_hard_stse_window(
        probabilities,
        edges,
        sample.communities[time_index],
        sample.edge_mask[time_index],
        1.0,
        1.0,
        2,
        1,
        "full",
        sample.sample_key,
        time_index,
    )
    return build_hard_stse_window(sample, time_index, selection)


class HardSTSEClassificationFeatureTest(unittest.TestCase):
    def test_features_are_recomputed_from_hard_signed_adjacency(self):
        sample = _sample("classification", 0, 2)
        first_window = _full_window(sample, 0)
        second_window = _full_window(sample, 1)
        builder = HardSTSEClassificationFeatureBuilder()
        first = builder.build_timepoint(sample, 0, first_window, None)
        second = builder.build_timepoint(
            sample, 1, second_window, first_window
        )
        self.assertEqual(tuple(first.node_features.shape), (3, 14))
        self.assertEqual(tuple(first.edge_features.shape), (3, 3, 7))
        self.assertEqual(tuple(first.graph_statistics.shape), (14,))
        self.assertAlmostEqual(float(first.node_features[0, 0]), 1.0)
        self.assertAlmostEqual(float(first.node_features[0, 1]), 0.7)
        self.assertAlmostEqual(float(first.node_features[0, 2]), 0.3)
        self.assertAlmostEqual(float(first.edge_features[0, 2, 0]), -0.3)
        self.assertAlmostEqual(float(first.edge_features[0, 2, 1]), 0.3)
        self.assertFalse(bool(first.delta_degree_mask.any()))
        self.assertTrue(bool(second.delta_degree_mask.all()))
        self.assertTrue(bool(second.graph_statistic_mask[9]))

    def test_hard_feature_loss_reaches_selection_probabilities(self):
        sample = _sample("gradient", 1, 1)
        window = _full_window(sample, 0)
        features = HardSTSEClassificationFeatureBuilder().build_timepoint(
            sample, 0, window, None
        )
        loss = (
            features.node_features[:, :3].sum()
            + features.edge_features[:, :, :2].sum()
        )
        loss.backward()
        node_probability = window.selection.node_probabilities
        edge_probability = window.selection.edge_probabilities
        self.assertIsNotNone(node_probability.grad)
        self.assertIsNotNone(edge_probability.grad)
        self.assertGreater(float(node_probability.grad.abs().sum()), 0.0)
        self.assertGreater(float(edge_probability.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
