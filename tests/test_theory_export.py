from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.extraction.hard_extractor import HardSubgraphCandidate  # noqa: E402
from keysubgraph.theory import (  # noqa: E402
    HardExportFidelityEvaluator,
    SpectralGWEvolutionEncoder,
    SpectralGWGreedyExporter,
    TheoryBoundEvaluator,
    build_hard_union_graph,
    one_step_sgw_error_bound,
    spectral_winf_exact,
)


def _candidate(seed, nodes, edges, adjacency, score):
    return HardSubgraphCandidate(
        seed_node=seed,
        node_ids=tuple(nodes),
        node_names=tuple("n{}".format(node) for node in nodes),
        edge_index=tuple(edges),
        original_edge_weights=tuple(float(adjacency[left, right]) for left, right in edges),
        node_scores=tuple(0.8 for _ in nodes),
        edge_scores=tuple(0.8 for _ in edges),
        community_labels=tuple(0 for _ in nodes),
        delta_degree=tuple(0.0 for _ in nodes),
        delta_degree_mask=tuple(False for _ in nodes),
        delta_edge_weight=tuple(0.0 for _ in edges),
        delta_edge_mask=tuple(False for _ in edges),
        score_node=score,
        score_edge=score,
        score_connectivity=1.0,
        score_dynamic=0.0,
        score_local_confidence=0.0,
        candidate_score=score,
    )


class TheoryExportTest(unittest.TestCase):
    def setUp(self):
        self.adjacency = torch.tensor(
            [
                [0.0, 0.7, 0.0, 0.0],
                [0.7, 0.0, -0.5, 0.0],
                [0.0, -0.5, 0.0, 0.4],
                [0.0, 0.0, 0.4, 0.0],
            ],
            dtype=torch.float64,
        )
        self.mask = self.adjacency.abs() > 0.0
        self.names = ("a", "b", "c", "d")
        self.fidelity = HardExportFidelityEvaluator(
            laplacian_eta=1.0e-4,
            heat_kernel_t=0.5,
            spectral_quantile_grid=(0.25, 0.5, 0.75),
            train_entropic_reg=0.05,
            train_max_iter=10,
            train_sinkhorn_iter=10,
            tolerance=1.0e-6,
            eval_entropic_reg=0.02,
            eval_max_iter=20,
            eval_sinkhorn_iter=20,
        )

    def test_union_has_unique_original_signed_edges(self):
        first = _candidate(0, (0, 1, 2), ((0, 1), (1, 2)), self.adjacency, 0.8)
        second = _candidate(2, (1, 2, 3), ((1, 2), (2, 3)), self.adjacency, 0.7)
        union = build_hard_union_graph(
            (first, second), self.adjacency, self.names, self.mask
        )
        self.assertEqual(union.node_ids, (0, 1, 2, 3))
        self.assertEqual(len(union.edge_index), len(set(union.edge_index)))
        self.assertEqual(union.edge_index, ((0, 1), (1, 2), (2, 3)))
        self.assertEqual(tuple(torch.sign(union.adjacency[union.adjacency != 0]).tolist()), (1.0, 1.0, -1.0, -1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            build_hard_union_graph(
                (_candidate(0, (0, 2), ((0, 2),), self.adjacency, 1.0),),
                self.adjacency,
                self.names,
                self.mask,
            )

    def test_identity_fidelity_and_spectral_error_chain(self):
        complete = _candidate(
            0,
            (0, 1, 2, 3),
            ((0, 1), (1, 2), (2, 3)),
            self.adjacency,
            1.0,
        )
        union = build_hard_union_graph(
            (complete,), self.adjacency, self.names, self.mask
        )
        result = self.fidelity.evaluate(
            self.adjacency, self.adjacency, self.mask, union
        )
        self.assertLess(result.full_to_soft_laplacian_fro_error, 1.0e-10)
        self.assertLess(result.full_to_soft_gw_error, 1.0e-10)
        self.assertLess(result.soft_to_hard_spectral_winf, 1.0e-10)
        self.assertLess(result.soft_to_hard_gw_error, 1.0e-10)
        self.assertLessEqual(
            result.full_to_hard_spectral_winf,
            result.full_to_soft_spectral_winf
            + result.soft_to_hard_spectral_winf
            + 1.0e-8,
        )

    def test_greedy_is_bounded_monotone_and_avoids_redundancy(self):
        first = _candidate(0, (0, 1), ((0, 1),), self.adjacency, 0.8)
        second = _candidate(2, (2, 3), ((2, 3),), self.adjacency, 0.7)
        redundant = _candidate(1, (0, 1), ((0, 1),), self.adjacency, 0.6)
        greedy = SpectralGWGreedyExporter(
            self.fidelity,
            beta_lambda=0.0,
            beta_gw=0.0,
            beta_overlap=1.0,
            min_export_gain=0.0,
        )
        selected, union, trace = greedy.select(
            (first, second, redundant),
            self.adjacency,
            self.names,
            self.mask,
            self.adjacency * 0.8,
            max_k=3,
        )
        self.assertEqual(tuple(item.seed_node for item in selected), (0, 2))
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(len(trace), len(selected))
        self.assertTrue(all(item["marginal_gain"] >= 0.0 for item in trace))
        self.assertTrue(
            all(left["objective"] <= right["objective"] for left, right in zip(trace[:-1], trace[1:]))
        )
        self.assertEqual(union.num_edges, 2)
        empty, empty_union, empty_trace = greedy.select(
            (), self.adjacency, self.names, self.mask, self.adjacency, max_k=3
        )
        self.assertEqual(empty, ())
        self.assertIsNone(empty_union)
        self.assertEqual(empty_trace, ())

    def test_greedy_can_reject_high_old_score_when_set_fidelity_worsens(self):
        first = _candidate(0, (0, 1), ((0, 1),), self.adjacency, 2.0)
        harmful = _candidate(2, (2, 3), ((2, 3),), self.adjacency, 0.5)

        def controlled_error(soft_adjacency, edge_mask, union):
            error = 0.0 if union.num_edges == 1 else 10.0
            return error, 0.0, True, 1, 0.0

        original = self.fidelity.fast_soft_to_hard
        self.fidelity.fast_soft_to_hard = controlled_error
        try:
            greedy = SpectralGWGreedyExporter(
                self.fidelity,
                beta_lambda=1.0,
                beta_gw=0.0,
                beta_overlap=0.0,
                min_export_gain=0.0,
            )
            selected, _, trace = greedy.select(
                (first, harmful),
                self.adjacency,
                self.names,
                self.mask,
                self.adjacency,
                max_k=2,
            )
        finally:
            self.fidelity.fast_soft_to_hard = original
        self.assertEqual(tuple(item.seed_node for item in selected), (0,))
        self.assertEqual(len(trace), 1)

    def test_evolution_supports_different_node_counts_and_permutation(self):
        smaller = self.adjacency[:3, :3]
        smaller_mask = smaller.abs() > 0.0
        encoder = SpectralGWEvolutionEncoder(self.fidelity)
        result = encoder.encode(
            (self.adjacency, smaller),
            (self.mask, smaller_mask),
            (0.0, 2.0),
        )
        self.assertEqual(len(result.gamma), 1)
        self.assertEqual(len(result.aggregated), 2 * (3 + 2) + 3)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = self.adjacency.index_select(0, permutation).index_select(1, permutation)
        permuted_mask = permuted.abs() > 0.0
        invariant = encoder.encode(
            (self.adjacency, permuted),
            (self.mask, permuted_mask),
            (0.0, 1.0),
        )
        self.assertLess(max(abs(value) for value in invariant.gamma[0]), 1.0e-6)

    def test_one_step_bound_and_theory_status_wording(self):
        bound = one_step_sgw_error_bound(0.1, 0.2, 0.05, 0.07, 3, 2.0)
        self.assertGreater(bound, 0.0)
        evaluator = TheoryBoundEvaluator(bootstrap_repeats=50, seed=7)
        report = evaluator.evaluate(
            ((0.0, 0.0), (0.1, 0.0), (2.0, 0.0), (2.1, 0.0)),
            ((0.0, 0.0), (0.1, 0.0), (2.0, 0.0), (2.1, 0.0)),
            (0, 0, 1, 1),
        )
        self.assertTrue(report["lower_bound_positive"])
        self.assertEqual(report["sufficient_condition_status"], "verified")
        failed = evaluator.evaluate(
            ((0.0,), (0.1,), (0.2,), (0.3,)),
            ((1.0,), (1.1,), (-0.8,), (-0.7,)),
            (0, 0, 1, 1),
        )
        self.assertFalse(failed["lower_bound_positive"])
        self.assertEqual(failed["sufficient_condition_status"], "not_verified")

    def test_actual_one_step_sgw_error_obeys_derived_bound(self):
        full_next = self.adjacency.clone()
        full_next[0, 1] *= 0.8
        full_next[1, 0] *= 0.8
        hard_now = self.adjacency.clone()
        hard_now[1, 2] *= 0.9
        hard_now[2, 1] *= 0.9
        hard_next = full_next.clone()
        hard_next[1, 2] *= 0.9
        hard_next[2, 1] *= 0.9
        encoder = SpectralGWEvolutionEncoder(self.fidelity)
        full_evolution = encoder.encode(
            (self.adjacency, full_next), (self.mask, self.mask), (0.0, 2.0)
        )
        hard_evolution = encoder.encode(
            (hard_now, hard_next), (self.mask, self.mask), (0.0, 2.0)
        )
        actual = torch.linalg.vector_norm(
            torch.tensor(full_evolution.gamma[0])
            - torch.tensor(hard_evolution.gamma[0])
        )
        endpoint_errors = []
        for full, hard in ((self.adjacency, hard_now), (full_next, hard_next)):
            _, full_spectrum, full_metric = self.fidelity._geometry(full, self.mask)
            _, hard_spectrum, hard_metric = self.fidelity._geometry(hard, self.mask)
            endpoint_errors.append(
                (
                    float(spectral_winf_exact(full_spectrum.eigenvalues, hard_spectrum.eigenvalues)),
                    float(self.fidelity.eval_gw(full_metric, hard_metric).distance),
                )
            )
        bound = one_step_sgw_error_bound(
            endpoint_errors[0][0],
            endpoint_errors[1][0],
            endpoint_errors[0][1],
            endpoint_errors[1][1],
            3,
            2.0,
        )
        self.assertLessEqual(float(actual), bound + 1.0e-4)


if __name__ == "__main__":
    unittest.main()
