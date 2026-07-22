from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.models import MaskedGraphPooling, SignedGraphEncoder


class TGSGWSignedEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.5, 1.0]],
            dtype=torch.float32,
        )
        self.adjacency = torch.tensor(
            [[0.0, 0.7, -0.4], [0.7, 0.0, 0.2], [-0.4, 0.2, 0.0]],
            dtype=torch.float32,
        )
        self.encoder = SignedGraphEncoder(3, 8, num_layers=2, dropout=0.0)
        self.pooling = MaskedGraphPooling(8, output_dim=6, dropout=0.0)
        self.encoder.eval()
        self.pooling.eval()

    def test_node_equivariance_and_graph_invariance(self):
        permutation = torch.tensor([2, 0, 1])
        direct = self.encoder(self.features, self.adjacency)
        permuted = self.encoder(
            self.features.index_select(0, permutation),
            self.adjacency.index_select(0, permutation).index_select(1, permutation),
        )
        self.assertTrue(torch.allclose(permuted, direct.index_select(0, permutation), atol=1.0e-6))
        self.assertTrue(
            torch.allclose(self.pooling(direct), self.pooling(permuted), atol=1.0e-6)
        )

    def test_positive_and_negative_channels_are_not_collapsed(self):
        positive = self.adjacency.abs()
        negative = -self.adjacency.abs()
        positive.fill_diagonal_(0.0)
        negative.fill_diagonal_(0.0)
        self.assertFalse(
            torch.allclose(
                self.encoder(self.features, positive),
                self.encoder(self.features, negative),
            )
        )

    def test_masked_padding_does_not_change_graph_embedding(self):
        direct = self.pooling(self.encoder(self.features, self.adjacency))
        padded_features = torch.cat((self.features, torch.randn(2, 3)), dim=0)
        padded_adjacency = torch.randn(5, 5)
        padded_adjacency = 0.5 * (padded_adjacency + padded_adjacency.t())
        padded_adjacency[:3, :3] = self.adjacency
        mask = torch.tensor([True, True, True, False, False])
        padded_nodes = self.encoder(padded_features, padded_adjacency, node_mask=mask)
        padded = self.pooling(padded_nodes, node_mask=mask)
        self.assertTrue(torch.allclose(direct, padded, atol=1.0e-6))
        self.assertEqual(float(padded_nodes[~mask].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
