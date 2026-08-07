from __future__ import absolute_import, division, print_function

import unittest

from keysubgraph.diagnostics.retrospective_consensus import (
    accepted_assignment_metrics,
    best_object_assignment,
    transition_phase,
)


class RetrospectiveConsensusDiagnosticTest(unittest.TestCase):
    def test_best_assignment_recovers_permuted_objects(self):
        previous = (
            ({0, 1}, {(0, 1)}),
            ({4, 5}, {(4, 5)}),
            ({8, 9}, {(8, 9)}),
        )
        current = (previous[2], previous[0], previous[1])
        result = best_object_assignment(previous, current)
        self.assertEqual(result["permutation"], (1, 2, 0))
        self.assertAlmostEqual(result["mean_node_jaccard"], 1.0)
        self.assertAlmostEqual(result["mean_edge_jaccard"], 1.0)

    def test_rejected_matches_reduce_coverage_adjusted_score(self):
        previous = (({0, 1}, {(0, 1)}), ({2, 3}, {(2, 3)}))
        current = (({0, 1}, {(0, 1)}), ({2, 3}, {(2, 3)}))
        result = accepted_assignment_metrics(previous, current, (0, -1))
        self.assertAlmostEqual(result["accepted_mean_node_jaccard"], 1.0)
        self.assertAlmostEqual(result["acceptance_rate"], 0.5)
        self.assertAlmostEqual(result["coverage_adjusted_node_jaccard"], 0.5)

    def test_phase_boundaries_are_explicit(self):
        self.assertEqual(transition_phase(1, 3, 4), "exploration_internal")
        self.assertEqual(transition_phase(3, 3, 4), "exploration_boundary")
        self.assertEqual(transition_phase(4, 3, 4), "history_ramp")
        self.assertEqual(transition_phase(7, 3, 4), "steady_state")


if __name__ == "__main__":
    unittest.main()
