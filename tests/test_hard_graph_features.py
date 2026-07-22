from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_graph_features import HardGraphFeatureBuilder, HardGraphWindow


def _window(adjacency, names, communities, time_start, node_ids=None, valid=True):
    return HardGraphWindow(
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        communities=torch.tensor(communities, dtype=torch.long),
        node_names=tuple(names),
        node_ids=tuple(node_ids) if node_ids is not None else None,
        time_start=float(time_start),
        edge_presence_threshold=0.0,
        window_valid=valid,
    )


class HardGraphFeatureBuilderTest(unittest.TestCase):
    def test_features_are_recomputed_from_selected_union_edges(self):
        full = _window(
            [[0.0, 0.5, 0.9], [0.5, 0.0, -0.4], [0.9, -0.4, 0.0]],
            ["a", "b", "c"], [0, 0, 1], 0.0,
        )
        hard = _window(
            [[0.0, 0.5, 0.0], [0.5, 0.0, -0.4], [0.0, -0.4, 0.0]],
            ["a", "b", "c"], [0, 0, 1], 0.0,
        )
        builder = HardGraphFeatureBuilder()
        full_features = builder.build_sequence([full])[0]
        hard_features = builder.build_sequence([hard])[0]
        self.assertEqual(tuple(hard_features.node_features.shape), (3, 13))
        self.assertEqual(tuple(hard_features.edge_features.shape), (3, 3, 4))
        self.assertFalse(torch.allclose(full_features.node_features, hard_features.node_features))
        self.assertEqual(hard_features.source, "hard_union_recomputed")
        self.assertLess(float(hard_features.edge_features[1, 2, 0]), 0.0)

    def test_temporal_alignment_uses_stable_identity_not_row_index(self):
        first = _window(
            [[0.0, 1.0], [1.0, 0.0]], ["roi-a", "roi-b"], [0, 1], 0.0,
            node_ids=["id-a", "id-b"],
        )
        second = _window(
            [[0.0, 2.0], [2.0, 0.0]], ["renamed-b", "renamed-a"], [1, 0], 1.0,
            node_ids=["id-b", "id-a"],
        )
        features = HardGraphFeatureBuilder().build_sequence([first, second])
        self.assertTrue(bool(features[1].delta_degree_mask.all()))
        self.assertTrue(torch.allclose(features[1].node_features[:, 5], torch.ones(2)))

    def test_invalid_window_breaks_adjacent_difference_chain(self):
        valid = _window([[0.0, 1.0], [1.0, 0.0]], ["a", "b"], [0, 1], 0.0)
        invalid = _window([[0.0]], ["x"], [0], 1.0, valid=False)
        later = _window([[0.0, 2.0], [2.0, 0.0]], ["a", "b"], [0, 1], 2.0)
        features = HardGraphFeatureBuilder().build_sequence([valid, invalid, later])
        self.assertIsNone(features[1])
        self.assertFalse(bool(features[2].delta_degree_mask.any()))
        self.assertEqual(float(features[2].node_features[:, 5].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
