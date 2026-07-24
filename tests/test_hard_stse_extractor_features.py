from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_stse_extractor_features import (
    HardSTSEExtractorFeatureBuilder,
)
from tests.test_full_graph_classifier import _sample


class HardSTSEExtractorFeatureTest(unittest.TestCase):
    def setUp(self):
        self.builder = HardSTSEExtractorFeatureBuilder()

    def test_schema_contains_explicit_invalid_delta_masks(self):
        sample = _sample("feature", 0, 2)
        first = self.builder.build_timepoint(sample, 0)
        second = self.builder.build_timepoint(sample, 1)
        self.assertEqual(tuple(first.node_features.shape), (3, 15))
        self.assertEqual(tuple(first.edge_base_features.shape), (3, 3, 6))
        self.assertFalse(bool(first.delta_degree_mask.any()))
        self.assertFalse(bool(first.delta_edge_mask.any()))
        self.assertTrue(torch.equal(first.node_features[:, 4], torch.zeros(3)))
        self.assertTrue(torch.equal(first.edge_base_features[:, :, 4], torch.zeros(3, 3)))
        self.assertTrue(bool(second.delta_degree_mask.all()))
        off_diagonal = ~torch.eye(3, dtype=torch.bool)
        self.assertTrue(bool(second.delta_edge_mask[off_diagonal].all()))
        self.assertTrue(torch.equal(
            second.edge_base_features[:, :, 4].bool(),
            second.delta_edge_mask,
        ))

    def test_signed_edges_and_community_identity_are_separate_coordinates(self):
        sample = _sample("signed", 1, 1)
        features = self.builder.build_timepoint(sample, 0)
        self.assertAlmostEqual(float(features.edge_base_features[0, 2, 0]), -0.3)
        self.assertAlmostEqual(float(features.edge_base_features[0, 2, 1]), 0.3)
        self.assertEqual(float(features.edge_base_features[0, 1, 5]), 1.0)
        self.assertEqual(float(features.edge_base_features[0, 2, 5]), 0.0)
        self.assertFalse(hasattr(self.builder, "embedding"))

    def test_local_clustering_is_finite_and_permutation_equivariant(self):
        sample = _sample("permutation", 0, 1)
        direct = self.builder.build_timepoint(sample, 0).node_features
        permutation = torch.tensor([2, 0, 1])
        permuted = _sample("permutation", 0, 1)
        graph = permuted.adjacency[0].index_select(0, permutation).index_select(
            1, permutation
        )
        object.__setattr__(permuted, "adjacency", (graph,))
        object.__setattr__(
            permuted,
            "edge_mask",
            (permuted.edge_mask[0].index_select(0, permutation).index_select(1, permutation),),
        )
        object.__setattr__(
            permuted,
            "communities",
            (permuted.communities[0].index_select(0, permutation),),
        )
        object.__setattr__(
            permuted,
            "node_names",
            (tuple(permuted.node_names[0][index] for index in permutation.tolist()),),
        )
        reordered = self.builder.build_timepoint(permuted, 0).node_features
        self.assertTrue(torch.allclose(reordered, direct.index_select(0, permutation)))


if __name__ == "__main__":
    unittest.main()
