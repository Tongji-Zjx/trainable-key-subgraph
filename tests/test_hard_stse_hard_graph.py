from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_stse_hard_graph import build_hard_stse_window
from keysubgraph.models.hard_stse_selector import select_hard_stse_window
from tests.test_full_graph_classifier import _sample


class HardSTSEHardGraphTest(unittest.TestCase):
    def test_padded_ste_and_cropped_bool_views_are_numerically_identical(self):
        sample = _sample("dual", 0, 1)
        node_logits = torch.tensor([1.0, 0.5, -0.2], requires_grad=True)
        edge_logits = torch.tensor(
            [[0.0, 2.0, -1.0], [2.0, 0.0, 1.0], [-1.0, 1.0, 0.0]],
            requires_grad=True,
        )
        node = torch.sigmoid(node_logits)
        edge = torch.sigmoid(edge_logits)
        selection = select_hard_stse_window(
            node,
            edge,
            sample.communities[0],
            sample.edge_mask[0],
            node_ratio=2.0 / 3.0,
            edge_ratio=1.0,
            node_minimum=2,
            edge_minimum=1,
            selection_mode="learned",
            sample_key=sample.sample_key,
            time_index=0,
        )
        window = build_hard_stse_window(sample, 0, selection)
        self.assertTrue(window.window_valid)
        self.assertIsNotNone(window.cropped_graph)
        indices = torch.nonzero(window.hard_node_mask, as_tuple=False).flatten()
        padded_crop = window.adjacency_st.detach().index_select(
            0, indices
        ).index_select(1, indices)
        self.assertTrue(torch.allclose(
            padded_crop, window.cropped_graph.adjacency
        ))
        original_nonzero = sample.adjacency[0][window.hard_edge_mask]
        hard_nonzero = window.adjacency_st.detach()[window.hard_edge_mask]
        self.assertTrue(torch.equal(original_nonzero, hard_nonzero))
        window.adjacency_st.abs().sum().backward()
        self.assertGreater(float(node_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(edge_logits.grad.abs().sum()), 0.0)

    def test_empty_candidate_produces_explicit_invalid_window(self):
        sample = _sample("empty", 0, 1)
        no_edges = torch.zeros(3, 3, dtype=torch.bool)
        selection = select_hard_stse_window(
            torch.full((3,), 0.5),
            torch.zeros(3, 3),
            sample.communities[0],
            no_edges,
            0.5,
            0.5,
            2,
            1,
            "learned",
            sample.sample_key,
            0,
        )
        window = build_hard_stse_window(sample, 0, selection)
        self.assertFalse(window.window_valid)
        self.assertIsNone(window.cropped_graph)
        self.assertEqual(window.selection.actual_node_count, 0)


if __name__ == "__main__":
    unittest.main()
