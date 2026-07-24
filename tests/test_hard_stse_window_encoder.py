from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_stse_classification_features import (
    HardSTSEClassificationFeatureBuilder,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from keysubgraph.models.hard_stse_window_encoder import HardSTSEWindowEncoder
from tests.test_hard_stse_classification_features import _full_window
from tests.test_full_graph_classifier import _sample


class HardSTSEWindowEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        self.sample = _sample("window", 0, 1)
        self.hard_window = _full_window(self.sample, 0)
        self.features = HardSTSEClassificationFeatureBuilder().build_timepoint(
            self.sample, 0, self.hard_window, None
        )
        self.config = HardSTSEConfig(dropout=0.0)

    def test_output_dimensions_attention_and_gradients(self):
        encoder = HardSTSEWindowEncoder(self.config)
        output = encoder(self.features)
        self.assertEqual(tuple(output.node_pooled.shape), (384,))
        self.assertEqual(tuple(output.edge_pooled.shape), (256,))
        self.assertEqual(tuple(output.raw_representation.shape), (654,))
        self.assertEqual(tuple(output.embedding.shape), (128,))
        self.assertAlmostEqual(
            float(output.node_attention[self.features.node_mask].sum()),
            1.0,
            places=6,
        )
        upper = torch.triu(self.features.edge_mask, diagonal=1)
        self.assertAlmostEqual(
            float(output.edge_attention[upper].sum()), 1.0, places=6
        )
        weights = torch.linspace(
            0.1, 1.0, output.embedding.numel(),
            dtype=output.embedding.dtype,
        )
        (output.embedding * weights).sum().backward()
        for module in (encoder.node_encoder, encoder.edge_encoder, encoder.window_mlp):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_consistent_node_permutation_does_not_change_embedding(self):
        encoder = HardSTSEWindowEncoder(self.config)
        encoder.eval()
        permutation = torch.tensor([2, 0, 1])
        features = self.features
        permuted = type(features)(
            time_index=features.time_index,
            node_features=features.node_features.index_select(0, permutation),
            edge_features=features.edge_features.index_select(
                0, permutation
            ).index_select(1, permutation),
            graph_statistics=features.graph_statistics,
            graph_statistic_mask=features.graph_statistic_mask,
            node_mask=features.node_mask.index_select(0, permutation),
            edge_mask=features.edge_mask.index_select(
                0, permutation
            ).index_select(1, permutation),
            delta_degree_mask=features.delta_degree_mask.index_select(
                0, permutation
            ),
            delta_edge_mask=features.delta_edge_mask.index_select(
                0, permutation
            ).index_select(1, permutation),
        )
        with torch.no_grad():
            direct = encoder(features).embedding
            reordered = encoder(permuted).embedding
        self.assertTrue(torch.allclose(direct, reordered, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
