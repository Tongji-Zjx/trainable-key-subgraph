from __future__ import absolute_import, division, print_function

import math
import unittest

import torch

# Import the data package first.  The legacy package-level re-exports have a
# known initialization-order dependency between data, models and features.
from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.sv_hard_graph_features import (
    SVHardSampleFeatureBuilder,
    SVHardNodeFeatureBuilder,
    SVStaticVariationExtractor,
)


def _window(adjacency, communities=(0, 0, 1), names=None, time=0.0):
    adjacency = torch.tensor(adjacency, dtype=torch.float32)
    if names is None:
        names = tuple("roi-{}".format(index) for index in range(adjacency.shape[0]))
    return HardGraphWindow(
        adjacency=adjacency,
        communities=torch.tensor(communities, dtype=torch.long),
        node_names=tuple(names),
        node_ids=tuple(names),
        time_start=float(time),
        edge_presence_threshold=0.0,
        window_valid=True,
    )


class SVHardGraphFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.base = _window(
            (
                (0.0, 0.5, -0.3),
                (0.5, 0.0, 0.2),
                (-0.3, 0.2, 0.0),
            )
        )

    def test_node_features_are_15d_and_recomputed_from_hard_graph(self):
        changed = _window(
            (
                (0.0, 0.7, -0.1),
                (0.7, 0.0, 0.0),
                (-0.1, 0.0, 0.0),
            ),
            time=1.0,
        )
        features = SVHardNodeFeatureBuilder().build_sequence(
            (self.base, changed)
        )
        self.assertEqual(tuple(features[0].node_features.shape), (3, 15))
        self.assertEqual(tuple(features[1].node_features.shape), (3, 15))
        self.assertFalse(bool(features[0].delta_degree_mask.any()))
        self.assertTrue(bool(features[1].delta_degree_mask.all()))
        expected_degree = changed.adjacency.abs().sum(dim=-1)
        self.assertTrue(
            torch.allclose(
                features[1].node_features[:, 0], expected_degree
            )
        )
        self.assertTrue(torch.isfinite(features[1].node_features).all())

    def test_signed_static_structure_formulas(self):
        node = SVHardNodeFeatureBuilder().build_sequence((self.base,))
        static, variation, window_mask, transition_mask = (
            SVStaticVariationExtractor().build(node)
        )
        structure = static[16:]
        entropy = -(
            (2.0 / 3.0) * math.log(2.0 / 3.0)
            + (1.0 / 3.0) * math.log(1.0 / 3.0)
        ) / math.log(2.0)
        expected = torch.tensor(
            (
                2.0 / 3.0,
                1.0 / 3.0,
                0.7 / 3.0,
                0.3 / 3.0,
                0.5,
                0.0,
                0.2 / 2.0,
                0.3 / 2.0,
                0.5,
                0.7,
                2.0 / 3.0,
                entropy,
            ),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(structure, expected, atol=1.0e-6))
        self.assertEqual(tuple(static.shape), (28,))
        self.assertTrue(torch.equal(variation, torch.zeros(16)))
        self.assertEqual(window_mask.tolist(), [True])
        self.assertEqual(tuple(transition_mask.shape), (0,))

    def test_node_permutation_and_community_relabel_do_not_change_summary(self):
        permutation = torch.tensor([2, 0, 1], dtype=torch.long)
        adjacency = self.base.adjacency.index_select(
            0, permutation
        ).index_select(1, permutation)
        communities = self.base.communities.index_select(0, permutation)
        communities = torch.where(
            communities == 0,
            torch.tensor(91),
            torch.tensor(-7),
        )
        names = tuple(
            self.base.node_names[int(index)] for index in permutation.tolist()
        )
        permuted = HardGraphWindow(
            adjacency=adjacency,
            communities=communities,
            node_names=names,
            node_ids=names,
            time_start=0.0,
            edge_presence_threshold=0.0,
            window_valid=True,
        )
        builder = SVHardSampleFeatureBuilder()
        left = builder.build((self.base,))
        right = builder.build((permuted,))
        self.assertTrue(
            torch.allclose(
                left.static_features,
                right.static_features,
                atol=1.0e-6,
            )
        )
        self.assertTrue(torch.equal(left.variation, right.variation))

    def test_constant_sequence_has_zero_variation_and_invalid_gap_is_ignored(self):
        second = _window(
            self.base.adjacency.tolist(), time=1.0
        )
        builder = SVHardSampleFeatureBuilder()
        repeated = builder.build((self.base, second))
        self.assertTrue(torch.equal(repeated.variation, torch.zeros(16)))
        self.assertEqual(repeated.transition_mask.tolist(), [True])

        gap = builder.build((self.base, None, second))
        self.assertTrue(torch.equal(gap.variation, torch.zeros(16)))
        self.assertEqual(gap.window_mask.tolist(), [True, False, True])
        self.assertEqual(gap.transition_mask.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
