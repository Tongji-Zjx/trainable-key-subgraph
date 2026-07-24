from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.features.hard_stse_extractor_features import (
    HardSTSEExtractorFeatureBuilder,
)
from keysubgraph.models.hard_stse_selector import (
    HardSTSEScorer,
    select_hard_stse_window,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from tests.test_full_graph_classifier import _sample


class HardSTSEScorerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)
        self.config = HardSTSEConfig(
            variant="M2",
            selection_mode="learned",
            use_sgw=False,
            dropout=0.0,
        )
        self.features = HardSTSEExtractorFeatureBuilder().build_timepoint(
            _sample("selector", 0, 2), 1
        )

    def test_scores_are_symmetric_bounded_and_masked(self):
        scorer = HardSTSEScorer(self.config)
        output = scorer(
            self.features.node_features,
            self.features.edge_base_features,
            self.features.edge_presence_mask,
        )
        self.assertEqual(tuple(output.node_probabilities.shape), (3,))
        self.assertEqual(tuple(output.edge_probabilities.shape), (3, 3))
        self.assertTrue(bool((output.node_probabilities > 0.0).all()))
        self.assertTrue(bool((output.node_probabilities < 1.0).all()))
        self.assertTrue(torch.allclose(
            output.edge_probabilities,
            output.edge_probabilities.transpose(0, 1),
        ))
        self.assertTrue(torch.equal(
            output.edge_probabilities > 0.0,
            self.features.edge_presence_mask,
        ))

    def test_classification_surrogate_reaches_both_scorers(self):
        scorer = HardSTSEScorer(self.config)
        output = scorer(
            self.features.node_features,
            self.features.edge_base_features,
            self.features.edge_presence_mask,
        )
        loss = output.node_probabilities.sum() + output.edge_probabilities.sum()
        loss.backward()
        node_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in scorer.node_scorer.parameters()
            if parameter.grad is not None
        )
        edge_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in scorer.edge_scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(node_gradient, 0.0)
        self.assertGreater(edge_gradient, 0.0)

    def test_community_selection_budget_is_deterministic_and_removes_isolates(self):
        node = torch.tensor([0.9, 0.8, 0.7, 0.6], requires_grad=True)
        edge = torch.full((4, 4), 0.5, requires_grad=True)
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask.fill_diagonal_(False)
        communities = torch.tensor([0, 0, 1, 1])
        kwargs = dict(
            node_probabilities=node,
            edge_probabilities=edge,
            communities=communities,
            edge_presence_mask=mask,
            node_ratio=0.5,
            edge_ratio=0.5,
            node_minimum=2,
            edge_minimum=1,
            selection_mode="random",
            sample_key="sample",
            time_index=3,
            random_seed=17,
        )
        first = select_hard_stse_window(**kwargs)
        second = select_hard_stse_window(**kwargs)
        self.assertTrue(torch.equal(first.hard_node_mask, second.hard_node_mask))
        self.assertTrue(torch.equal(first.hard_edge_mask, second.hard_edge_mask))
        self.assertEqual(first.requested_node_count, 2)
        self.assertEqual(first.requested_edge_count, 1)
        self.assertEqual(first.actual_node_count, 2)
        self.assertEqual(first.actual_edge_count, 1)
        selected_communities = communities[first.hard_node_mask]
        self.assertEqual(set(selected_communities.tolist()), {0, 1})

    def test_straight_through_masks_are_hard_forward_and_soft_backward(self):
        node_logits = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        edge_logits = torch.randn(3, 3, requires_grad=True)
        node = torch.sigmoid(node_logits)
        edge = torch.sigmoid(0.5 * (edge_logits + edge_logits.transpose(0, 1)))
        mask = torch.ones(3, 3, dtype=torch.bool)
        mask.fill_diagonal_(False)
        selection = select_hard_stse_window(
            node, edge, torch.tensor([0, 1, 1]), mask,
            2.0 / 3.0, 1.0, 2, 1, "learned", "gradient", 0,
        )
        self.assertTrue(torch.equal(
            selection.straight_through_node_mask.detach(),
            selection.hard_node_mask.to(torch.float32),
        ))
        self.assertTrue(torch.equal(
            selection.straight_through_edge_mask.detach().bool(),
            selection.hard_edge_mask,
        ))
        (
            selection.straight_through_node_mask.sum()
            + selection.straight_through_edge_mask.sum()
        ).backward()
        self.assertGreater(float(node_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(edge_logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
