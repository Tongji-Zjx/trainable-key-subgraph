from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: E402
from keysubgraph.features.graph_features import (  # noqa: E402
    GraphFeatureBuilder,
    align_current_to_previous,
)


def _sample(adjacency, names, communities):
    masks = []
    for graph in adjacency:
        mask = graph.abs() > 0.0
        mask.fill_diagonal_(False)
        masks.append(mask)
    return GraphSequenceSample(
        sample_key="SITE/sample",
        sample_id="sample",
        site="SITE",
        subject_id="subject",
        session_id="1",
        label=0,
        split="train",
        relative_path="SITE/0/sample.pt",
        adjacency=tuple(adjacency),
        edge_mask=tuple(masks),
        node_names=tuple(tuple(item) for item in names),
        communities=tuple(communities),
        window_starts=torch.arange(len(adjacency), dtype=torch.float32),
        source_global_threshold=0.1,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


class GraphFeaturesTest(unittest.TestCase):
    def setUp(self):
        graph = torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, -0.25],
                [0.0, -0.25, 0.0],
            ]
        )
        communities = torch.tensor([0, 0, 1])
        permutation = torch.tensor([2, 0, 1])
        self.permuted_sample = _sample(
            [graph, graph.index_select(0, permutation).index_select(1, permutation)],
            [("a", "b", "c"), ("c", "a", "b")],
            [communities, communities.index_select(0, permutation)],
        )
        self.builder = GraphFeatureBuilder(epsilon=1e-8)

    def test_alignment_uses_stable_names_instead_of_row_numbers(self):
        indices, mask = align_current_to_previous(
            ("c", "a", "new"), ("a", "b", "c")
        )
        self.assertEqual(indices.tolist(), [2, 0, -1])
        self.assertEqual(mask.tolist(), [True, True, False])

        features = self.builder.build_timepoint(self.permuted_sample, 1)
        self.assertTrue(bool(features.delta_degree_mask.all()))
        self.assertTrue(torch.allclose(features.delta_degree, torch.zeros(3)))
        self.assertTrue(
            torch.allclose(features.delta_edge_weight, torch.zeros(3, 3))
        )
        self.assertEqual(int(features.delta_edge_mask.sum()), 6)

    def test_node_features_follow_verified_formulas(self):
        features = self.builder.build_timepoint(self.permuted_sample, 0)

        self.assertEqual(tuple(features.degree.shape), (3,))
        self.assertTrue(torch.allclose(features.degree, torch.tensor([0.5, 0.75, 0.25])))
        self.assertFalse(bool(features.delta_degree_mask.any()))
        self.assertEqual(tuple(features.community_features.shape), (3, 7))
        self.assertTrue(bool(torch.isfinite(features.community_features).all()))
        self.assertAlmostEqual(float(features.community_features[0, 0]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(features.community_features[0, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(features.community_features[0, 5]), 1.0, places=6)
        self.assertAlmostEqual(float(features.community_features[2, 4]), 0.125, places=6)
        self.assertEqual(tuple(features.node_features.shape), (3, 9))

    def test_feature_schema_contains_no_spatial_fields(self):
        features = self.builder.build_timepoint(self.permuted_sample, 0)

        self.assertFalse(hasattr(self.permuted_sample, "coordinates"))
        self.assertFalse(hasattr(features, "neighbor_coordinates"))
        self.assertEqual(tuple(features.node_features.shape), (3, 9))
        self.assertEqual(tuple(features.edge_features.shape), (3, 3, 23))

    def test_edge_features_preserve_signed_and_absolute_values(self):
        features = self.builder.build_timepoint(self.permuted_sample, 0)

        self.assertEqual(tuple(features.edge_features.shape), (3, 3, 23))
        edge_tail = features.edge_features[1, 2, -5:]
        self.assertTrue(
            torch.allclose(edge_tail, torch.tensor([-0.25, 0.25, 0.0, 0.0, 0.0]))
        )
        self.assertTrue(bool(features.edge_mask[1, 2]))

    def test_missing_nodes_have_safe_zero_differences_and_false_masks(self):
        previous_graph = self.permuted_sample.adjacency[0]
        current_graph = torch.tensor(
            [[0.0, -0.25, 0.4], [-0.25, 0.0, 0.0], [0.4, 0.0, 0.0]]
        )
        sample = _sample(
            [previous_graph, current_graph],
            [("a", "b", "c"), ("b", "c", "d")],
            [torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])],
        )

        features = self.builder.build_timepoint(sample, 1)

        self.assertEqual(features.delta_degree_mask.tolist(), [True, True, False])
        self.assertEqual(float(features.delta_degree[2]), 0.0)
        self.assertFalse(bool(features.delta_edge_mask[2].any()))
        self.assertFalse(bool(features.delta_edge_mask[:, 2].any()))
        self.assertEqual(float(features.delta_edge_weight[2].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
