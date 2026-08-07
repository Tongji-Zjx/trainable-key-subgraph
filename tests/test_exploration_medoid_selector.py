from __future__ import absolute_import, division, print_function

import unittest
from types import SimpleNamespace

import torch

from keysubgraph.models.exploration_medoid_selector import (
    ExplorationObjectCandidate,
    exploration_candidate_similarity,
    select_exploration_medoids,
    stack_real_candidate_memories,
)
from keysubgraph.models.fixed_k_subgraph_selector import (
    select_object_conditioned_subgraphs,
)
from keysubgraph.models.theory_multi_object_selector import (
    MultiObjectTemporalMemory,
)
from scripts.audit_exploration_medoid_selector import _stratified_indices


def _candidate(window_index, cluster_index, sign=1.0):
    count = 6
    node_ids = tuple("roi_{}".format(index) for index in range(count))
    left = 2 * int(cluster_index)
    right = left + 1
    hard_nodes = torch.zeros(1, count, dtype=torch.bool)
    hard_nodes[0, left : right + 1] = True
    hard_edges = torch.zeros(1, count, count, dtype=torch.bool)
    hard_edges[0, left, right] = True
    hard_edges[0, right, left] = True
    nodes = torch.full((1, count), 0.03)
    selected_probability = 0.72 + 0.02 * float(window_index)
    nodes[0, left : right + 1] = selected_probability
    edges = torch.zeros(1, count, count)
    edges[0, left, right] = selected_probability
    edges[0, right, left] = selected_probability
    signed = torch.zeros_like(edges)
    signed[0, left, right] = float(sign) * 0.25
    signed[0, right, left] = float(sign) * 0.25
    representation = torch.zeros(1, 3)
    representation[0, int(cluster_index)] = 1.0
    spectrum = representation.clone()
    centroid = torch.tensor(
        [[10.0 * float(cluster_index), 0.0, 0.0]],
        dtype=torch.float32,
    )
    memory = MultiObjectTemporalMemory(
        object_states=representation.clone(),
        node_probabilities=nodes,
        edge_probabilities=edges,
        node_ids=node_ids,
        signed_edge_values=signed,
        object_representations=representation,
        coordinate_centroids=centroid,
        spectral_descriptors=spectrum,
        hard_node_masks=hard_nodes,
        hard_edge_masks=hard_edges,
        seed_indices=torch.tensor([left]),
    )
    return ExplorationObjectCandidate(
        window_index=int(window_index),
        object_index=int(cluster_index),
        quality=selected_probability,
        memory=memory,
    )


class ExplorationMedoidSelectorTest(unittest.TestCase):
    def test_formal_audit_panel_is_site_and_label_stratified(self):
        assignments = tuple(
            SimpleNamespace(site=site, label=label)
            for site in ("A", "B", "C")
            for label in (0, 1)
            for _ in range(3)
        )
        indices = _stratified_indices(assignments, limit=6, seed=43)
        strata = {
            (assignments[index].site, assignments[index].label)
            for index in indices
        }
        self.assertEqual(len(indices), 6)
        self.assertEqual(len(strata), 6)

    def test_selects_diverse_cross_window_real_medoids(self):
        candidates = [
            _candidate(window, cluster)
            for window in range(3)
            for cluster in range(3)
        ]
        selected = select_exploration_medoids(
            candidates,
            object_count=3,
            shortlist_multiplier=3,
        )
        anchor_node_sets = {
            tuple(
                torch.nonzero(
                    candidates[index].memory.hard_node_masks[0],
                    as_tuple=False,
                ).flatten().tolist()
            )
            for index in selected.anchor_indices
        }
        self.assertEqual(anchor_node_sets, {(0, 1), (2, 3), (4, 5)})
        self.assertEqual(selected.support_window_counts, (3, 3, 3))
        self.assertEqual(selected.unsupported_anchor_count, 0)
        self.assertLess(selected.mean_cross_window_cluster_similarity, 1.000001)
        self.assertGreater(selected.mean_cross_window_cluster_similarity, 0.50)
        self.assertTrue(
            all(candidates[index].window_index == 2 for index in selected.recent_indices)
        )
        self.assertEqual(len(selected.shortlist_indices), 9)

    def test_stacked_memory_is_reindexed_real_candidate_not_average(self):
        candidates = [_candidate(0, 0), _candidate(1, 1), _candidate(2, 2)]
        current_ids = tuple(reversed(candidates[0].memory.node_ids))
        adjacency = torch.ones(6, 6) - torch.eye(6)
        edge_mask = adjacency.to(torch.bool)
        stacked = stack_real_candidate_memories(
            candidates,
            (0, 1, 2),
            current_ids,
            adjacency,
            edge_mask,
        )
        for slot, candidate in enumerate(candidates):
            expected = torch.tensor(
                [
                    float(candidate.memory.node_probabilities[0, 5 - index])
                    for index in range(6)
                ]
            )
            self.assertTrue(
                torch.equal(stacked.node_probabilities[slot], expected)
            )
            self.assertEqual(
                int(stacked.hard_edge_masks[slot].sum()),
                int(candidate.memory.hard_edge_masks[0].sum()),
            )

    def test_signed_edge_disagreement_reduces_similarity(self):
        positive = _candidate(0, 0, sign=1.0)
        same = _candidate(1, 0, sign=1.0)
        opposite = _candidate(2, 0, sign=-1.0)
        similarity = exploration_candidate_similarity(
            (positive, same, opposite)
        )
        self.assertGreater(float(similarity[0, 1]), float(similarity[0, 2]))

    def test_long_term_anchor_can_guide_hard_growth(self):
        count = 8
        global_nodes = torch.full((count,), 0.60)
        global_edges = torch.full((count, count), 0.70)
        object_nodes = torch.full((2, count), 0.30)
        object_nodes[0, 0:2] = torch.tensor([0.90, 0.85])
        object_nodes[1, 2:4] = torch.tensor([0.90, 0.85])
        object_edges = torch.full((2, count, count), 0.60)
        edge_mask = torch.ones(count, count, dtype=torch.bool)
        edge_mask.fill_diagonal_(False)
        anchor_nodes = torch.zeros(2, count, dtype=torch.bool)
        anchor_nodes[0, 4:6] = True
        anchor_nodes[1, 6:8] = True
        anchor_edges = torch.zeros(2, count, count, dtype=torch.bool)
        anchor_edges[0, 4, 5] = True
        anchor_edges[0, 5, 4] = True
        anchor_edges[1, 6, 7] = True
        anchor_edges[1, 7, 6] = True
        selected = select_object_conditioned_subgraphs(
            global_nodes,
            global_edges,
            object_nodes,
            object_edges,
            edge_mask,
            per_object_node_ratio=0.25,
            edge_ratio=0.5,
            anchor_node_masks=anchor_nodes,
            anchor_edge_masks=anchor_edges,
            anchor_continuity_bonus=2.0,
            anchor_node_growth_bonus=2.0,
            anchor_edge_growth_bonus=1.0,
        )
        for index, subgraph in enumerate(selected.subgraphs):
            self.assertGreaterEqual(
                int((subgraph.hard_node_mask & anchor_nodes[index]).sum()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
