from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.dual_hard_sgw_selector import DualHardSGWSelector
from keysubgraph.models.dual_stse_hard_sgw_types import DualSTSEHardSGWConfig
from keysubgraph.models.fixed_k_subgraph_selector import (
    select_object_conditioned_subgraphs,
)
from keysubgraph.models.theory_multi_object_selector import (
    TheoryGuidedMultiObjectScorer,
    signed_laplacian_eigenvectors,
)
from tests.test_exact_stse_model import _exact_sample


def _fixture(count=8):
    torch.manual_seed(20260807)
    raw = torch.randn(count, count) * 0.15
    adjacency = 0.5 * (raw + raw.transpose(0, 1))
    adjacency.fill_diagonal_(0.0)
    mask = adjacency.abs() > 0.01
    mask.fill_diagonal_(False)
    node = torch.randn(count, 15)
    edge = torch.zeros(count, count, 6)
    edge[:, :, 0] = adjacency
    edge[:, :, 1] = adjacency.abs()
    edge[:, :, 2] = 0.5 * adjacency
    edge[:, :, 3] = 0.5 * adjacency.abs()
    edge[:, :, 4] = mask
    return node, edge, adjacency, mask


class TheoryMultiObjectSelectorTest(unittest.TestCase):
    def test_signed_spectral_multi_object_shapes_and_gradients(self):
        node, edge, adjacency, mask = _fixture()
        scorer = TheoryGuidedMultiObjectScorer(
            hidden_dim=24,
            edge_hidden_dim=12,
            object_count=3,
            spectral_dim=4,
            graph_layers=2,
            dropout=0.0,
        )
        output = scorer(node, edge, mask, adjacency)
        self.assertEqual(tuple(output.node_hidden.shape), (8, 24))
        self.assertEqual(tuple(output.node_probabilities.shape), (8,))
        self.assertEqual(tuple(output.edge_probabilities.shape), (8, 8))
        self.assertEqual(tuple(output.object_node_probabilities.shape), (3, 8))
        self.assertEqual(tuple(output.object_edge_probabilities.shape), (3, 8, 8))
        self.assertEqual(tuple(output.next_object_states.shape), (3, 24))
        self.assertTrue(torch.equal(output.edge_probabilities == 0.0, ~mask))
        loss = (
            output.node_probabilities.mean()
            + output.object_node_probabilities.mean()
            + output.object_edge_probabilities.mean()
            + output.regularization.overlap
            + output.regularization.reconstruction
            + output.regularization.coverage
        )
        loss.backward()
        self.assertGreater(
            float(scorer.encoder.input[1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(scorer.object_queries.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(scorer.object_edge_head[0].weight.grad.abs().sum()), 0.0
        )

    def test_temporal_state_changes_object_fields_without_roi_or_coordinates(self):
        node, edge, adjacency, mask = _fixture()
        scorer = TheoryGuidedMultiObjectScorer(
            hidden_dim=16,
            edge_hidden_dim=8,
            object_count=3,
            spectral_dim=3,
            graph_layers=1,
            dropout=0.0,
        ).eval()
        first = scorer(node, edge, mask, adjacency)
        second = scorer(
            node, edge, mask, adjacency, first.next_object_states
        )
        self.assertGreater(
            float(
                (
                    first.object_node_probabilities
                    - second.object_node_probabilities
                ).abs().max()
            ),
            1.0e-6,
        )
        self.assertGreaterEqual(float(second.regularization.temporal), 0.0)

    def test_cached_spectrum_is_forward_equivalent(self):
        node, edge, adjacency, mask = _fixture()
        scorer = TheoryGuidedMultiObjectScorer(
            hidden_dim=16,
            edge_hidden_dim=8,
            object_count=3,
            spectral_dim=4,
            graph_layers=1,
            dropout=0.0,
        ).eval()
        spectrum = scorer.encoder.spectral_features(adjacency, mask)
        direct = scorer(node, edge, mask, adjacency)
        cached = scorer(
            node,
            edge,
            mask,
            adjacency,
            spectral_features=spectrum,
        )
        self.assertTrue(
            torch.equal(
                direct.object_node_probabilities,
                cached.object_node_probabilities,
            )
        )
        self.assertTrue(
            torch.equal(
                direct.object_edge_probabilities,
                cached.object_edge_probabilities,
            )
        )

    def test_invalid_edges_and_signed_spectrum_are_safe(self):
        node, edge, adjacency, mask = _fixture()
        vectors = signed_laplacian_eigenvectors(adjacency, 10)
        self.assertEqual(tuple(vectors.shape), (8, 10))
        self.assertTrue(bool(torch.isfinite(vectors).all()))
        scorer = TheoryGuidedMultiObjectScorer(
            hidden_dim=16,
            edge_hidden_dim=8,
            object_count=3,
            spectral_dim=4,
            graph_layers=1,
            dropout=0.0,
        ).eval()
        first = scorer(node, edge, mask, adjacency)
        changed_edge = edge.clone()
        changed_edge[~mask] = 999.0
        second = scorer(node, changed_edge, mask, adjacency)
        self.assertTrue(
            torch.allclose(
                first.object_edge_probabilities,
                second.object_edge_probabilities,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertGreater(int((adjacency > 0.0).sum()), 0)
        self.assertGreater(int((adjacency < 0.0).sum()), 0)

    def test_object_conditioned_hardening_preserves_ste_and_union(self):
        count = 12
        global_node = torch.full((count,), 0.6, requires_grad=True)
        global_edge = torch.full((count, count), 0.7, requires_grad=True)
        mask = torch.ones(count, count, dtype=torch.bool)
        mask.fill_diagonal_(False)
        objects = torch.full((3, count), 0.05)
        objects[0, 0:4] = 0.95
        objects[1, 4:8] = 0.95
        objects[2, 8:12] = 0.95
        objects.requires_grad_()
        object_edges = torch.full((3, count, count), 0.1)
        for index, start in enumerate((0, 4, 8)):
            object_edges[index, start : start + 4, start : start + 4] = 0.9
        object_edges.requires_grad_()
        selected = select_object_conditioned_subgraphs(
            global_node,
            global_edge,
            objects,
            object_edges,
            mask,
            per_object_node_ratio=0.25,
            edge_ratio=0.5,
        )
        self.assertEqual(len(selected.subgraphs), 3)
        node_union = torch.zeros(count, dtype=torch.bool)
        edge_union = torch.zeros(count, count, dtype=torch.bool)
        for item in selected.subgraphs:
            node_union |= item.hard_node_mask
            edge_union |= item.hard_edge_mask
        self.assertTrue(torch.equal(node_union, selected.union.hard_node_mask))
        self.assertTrue(torch.equal(edge_union, selected.union.hard_edge_mask))
        upper = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        self.assertLessEqual(
            float(selected.pairwise_node_overlap[upper].max()), 0.40
        )
        loss = selected.union.straight_through_node_mask.sum()
        loss = loss + selected.union.straight_through_edge_mask.sum()
        loss.backward()
        self.assertGreater(float(global_node.grad.abs().sum()), 0.0)
        self.assertGreater(float(objects.grad.abs().sum()), 0.0)
        self.assertGreater(float(object_edges.grad.abs().sum()), 0.0)

    def test_hardening_falls_back_from_isolated_preferred_seeds(self):
        count = 8
        global_node = torch.full((count,), 0.6)
        global_edge = torch.full((count, count), 0.7)
        mask = torch.zeros(count, count, dtype=torch.bool)
        for left, right in ((4, 5), (5, 6), (6, 7), (4, 7)):
            mask[left, right] = True
            mask[right, left] = True
        objects = torch.full((3, count), 0.1)
        objects[:, :4] = 0.99
        object_edges = torch.full((3, count, count), 0.5)
        selected = select_object_conditioned_subgraphs(
            global_node,
            global_edge,
            objects,
            object_edges,
            mask,
            per_object_node_ratio=0.25,
            edge_ratio=0.5,
            candidate_multiplier=4,
        )
        self.assertEqual(len(selected.subgraphs), 3)
        self.assertTrue(
            all(item.actual_edge_count >= 1 for item in selected.subgraphs)
        )

    def test_dual_selector_integration_emits_three_learned_objects(self):
        torch.manual_seed(17)
        sample = _exact_sample("multi-object", 1, 3)
        config = DualSTSEHardSGWConfig(
            selector_architecture="theory_multi_object",
            selector_object_temporal_state=True,
            critical_subgraph_count=3,
            critical_node_ratio_per_object=0.67,
            node_minimum=2,
            edge_minimum=1,
        )
        selector = DualHardSGWSelector(config)
        selector.eval()
        output = selector(
            ExactSTSEBatch((sample,)), selection_mode="learned"
        )
        cache_count = len(selector._spectral_cache)
        repeated = selector(
            ExactSTSEBatch((sample,)), selection_mode="learned"
        )
        self.assertEqual(cache_count, sample.graph.num_timepoints)
        self.assertEqual(len(selector._spectral_cache), cache_count)
        for first, second in zip(
            output.hard_windows[0], repeated.hard_windows[0]
        ):
            self.assertTrue(
                torch.equal(
                    first.hard_node_mask, second.hard_node_mask
                )
            )
            self.assertTrue(
                torch.equal(
                    first.hard_edge_mask, second.hard_edge_mask
                )
            )
        self.assertEqual(len(output.hard_subgraphs[0]), 3)
        self.assertTrue(
            all(len(objects) == 3 for objects in output.hard_subgraphs[0])
        )
        self.assertEqual(
            output.diagnostics["selector_architecture"],
            "theory_multi_object",
        )
        self.assertTrue(output.diagnostics["uses_signed_graph_encoder"])
        self.assertTrue(output.diagnostics["uses_object_temporal_state"])
        terms = output.diagnostics["multi_object_regularization"]
        self.assertIsNotNone(terms)
        total = sum(terms.values())
        total = total + sum(
            window.adjacency_st.abs().sum()
            for window in output.hard_windows[0]
        )
        total.backward()
        self.assertGreater(
            float(selector.scorer.object_queries.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(selector.scorer.object_edge_head[0].weight.grad.abs().sum()),
            0.0,
        )

    def test_fast_runtime_preserves_hard_decisions(self):
        torch.manual_seed(29)
        sample = _exact_sample("fast-runtime", 1, 3)
        common = dict(
            selector_architecture="theory_multi_object",
            selector_object_temporal_state=True,
            critical_subgraph_count=3,
            critical_node_ratio_per_object=0.67,
            node_minimum=2,
            edge_minimum=1,
        )
        reference = DualHardSGWSelector(
            DualSTSEHardSGWConfig(**common)
        ).eval()
        accelerated = DualHardSGWSelector(
            DualSTSEHardSGWConfig(
                selector_fast_runtime=True, **common
            )
        ).eval()
        accelerated.load_state_dict(reference.state_dict())
        batch = ExactSTSEBatch((sample,))
        expected = reference(batch, selection_mode="learned")
        actual = accelerated(batch, selection_mode="learned")
        self.assertTrue(
            actual.diagnostics["detailed_diagnostics_skipped"]
        )
        for expected_window, actual_window in zip(
            expected.hard_windows[0], actual.hard_windows[0]
        ):
            self.assertTrue(
                torch.equal(
                    expected_window.hard_node_mask,
                    actual_window.hard_node_mask,
                )
            )
            self.assertTrue(
                torch.equal(
                    expected_window.hard_edge_mask,
                    actual_window.hard_edge_mask,
                )
            )


if __name__ == "__main__":
    unittest.main()
