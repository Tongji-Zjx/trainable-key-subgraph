from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from tests.test_exact_stse_model import _exact_sample


class DualHardSGWSelectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311)
        self.sample = _exact_sample("dual-selector", 1, 3)
        self.batch = ExactSTSEBatch((self.sample,))
        self.selector = DualHardSGWSelector()

    def test_learned_selection_preserves_signs_and_creates_no_edges(self):
        result = self.selector(self.batch, selection_mode="learned")
        self.assertEqual(len(result.hard_windows[0]), 3)
        for time_index, hard in enumerate(result.hard_windows[0]):
            source = self.sample.graph.adjacency[time_index]
            forward = hard.adjacency_st.detach()
            kept = forward != 0.0
            self.assertTrue(torch.equal(forward[kept], source[kept]))
            self.assertFalse(bool((kept & (source == 0.0)).any()))
            self.assertTrue(torch.equal(kept, kept.transpose(0, 1)))
        self.assertEqual(
            result.diagnostics["candidate_community_coverage"], 1.0
        )

    def test_random_selection_is_stable_and_budget_matched(self):
        first = self.selector(
            self.batch, selection_mode="random", random_seed=79
        )
        second = self.selector(
            self.batch, selection_mode="random", random_seed=79
        )
        for left, right in zip(
            first.hard_windows[0], second.hard_windows[0]
        ):
            self.assertTrue(
                torch.equal(left.hard_node_mask, right.hard_node_mask)
            )
            self.assertTrue(
                torch.equal(left.hard_edge_mask, right.hard_edge_mask)
            )
        self.assertAlmostEqual(
            first.diagnostics["candidate_node_ratio"], 2.0 / 3.0
        )

    def test_full_mode_is_the_unmodified_signed_graph(self):
        result = self.selector(self.batch, selection_mode="full")
        for time_index, hard in enumerate(result.hard_windows[0]):
            self.assertTrue(
                torch.equal(
                    hard.adjacency_st.detach(),
                    self.sample.graph.adjacency[time_index],
                )
            )
        self.assertEqual(result.diagnostics["actual_node_ratio"], 1.0)
        self.assertEqual(result.diagnostics["actual_edge_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
