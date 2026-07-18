from __future__ import absolute_import, division, print_function

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.edge_perturbation import (  # noqa: E402
    build_edge_perturbation_payloads,
    edge_deletion_order,
    perturb_key_subgraph,
    perturbation_sources,
)
from keysubgraph.data.baseline_dataset import _local_subgraph  # noqa: E402
from keysubgraph.features.graph_features import GraphFeatureBuilder  # noqa: E402
from tests.test_analysis import _control_sample  # noqa: E402


def _subgraph():
    return {
        "seed_node": 0,
        "node_ids": [0, 1, 2, 3],
        "node_names": ["a", "b", "c", "d"],
        "community_labels": [0, 0, 1, 1],
        "edge_index": [[0, 1], [0, 2], [1, 2], [2, 3]],
        "original_edge_weights": [0.5, -0.3, -0.8, 0.2],
        "edge_scores": [0.9, 0.1, 0.8, 0.2],
        "delta_edge_weight": [0.05, -0.02, 0.1, -0.04],
        "delta_edge_mask": [True, True, True, False],
        "node_scores": [0.9, 0.8, 0.7, 0.6],
    }


def _payload():
    full = _subgraph()
    small = copy.deepcopy(full)
    for field in (
        "edge_index", "original_edge_weights", "edge_scores",
        "delta_edge_weight", "delta_edge_mask",
    ):
        small[field] = small[field][:1]
    return {
        "sample_key": "SITE/sample",
        "subgraph_source": "key",
        "timepoints": [
            {
                "time_index": 0,
                "candidate_pool": [],
                "num_valid_subgraphs": 2,
                "subgraphs": [full, small],
            },
            {
                "time_index": 1,
                "candidate_pool": [],
                "num_valid_subgraphs": 1,
                "subgraphs": [copy.deepcopy(full)],
            },
        ],
    }


class EdgePerturbationTest(unittest.TestCase):
    def test_dataset_rebuilds_perturbed_signed_features_and_checks_provenance(self):
        source = _subgraph()
        source["original_edge_weights"] = [0.5, -0.3, 0.4, 0.6]
        perturbed = perturb_key_subgraph(
            source, "targeted", 0.25, "SITE/sample", 0, 0, seed=11
        )
        sample = _control_sample()
        rebuilt = _local_subgraph(
            perturbed, sample, 0, GraphFeatureBuilder()
        )
        self.assertEqual(rebuilt.node_count, 4)
        self.assertEqual(rebuilt.edge_count, 3)
        self.assertEqual(float(rebuilt.adjacency[0, 1]), 0.0)
        self.assertLess(float(rebuilt.adjacency[0, 2]), 0.0)
        self.assertTrue(bool(rebuilt.edge_mask[0, 2]))

        tampered = copy.deepcopy(perturbed)
        tampered["edge_perturbation"]["deleted_weights"][0] = -9.0
        with self.assertRaisesRegex(ValueError, "raw graph"):
            _local_subgraph(tampered, sample, 0, GraphFeatureBuilder())

    def test_targeted_deletion_is_nested_and_uses_signed_edge_scores(self):
        source = _subgraph()
        quarter = perturb_key_subgraph(
            source, "targeted", 0.25, "SITE/sample", 0, 0, seed=11
        )
        half = perturb_key_subgraph(
            source, "targeted", 0.50, "SITE/sample", 0, 0, seed=11
        )
        self.assertEqual(
            quarter["edge_perturbation"]["deleted_source_positions"], [0]
        )
        self.assertEqual(
            half["edge_perturbation"]["deleted_source_positions"], [0, 2]
        )
        self.assertEqual(half["edge_perturbation"]["deleted_negative_count"], 1)
        self.assertEqual(half["edge_perturbation"]["deleted_positive_count"], 1)
        self.assertEqual(quarter["node_ids"], source["node_ids"])
        self.assertEqual(quarter["edge_scores"], [0.1, 0.8, 0.2])
        self.assertEqual(quarter["original_edge_weights"], [-0.3, -0.8, 0.2])
        self.assertEqual(quarter["delta_edge_mask"], [True, True, False])
        self.assertEqual(
            set(quarter["edge_perturbation"]["deleted_source_positions"]),
            set(half["edge_perturbation"]["deleted_source_positions"][:1]),
        )

    def test_random_deletion_is_frozen_nested_and_count_matched(self):
        source = _subgraph()
        first = edge_deletion_order(source, "random", "SITE/sample", 0, 0, 2026)
        second = edge_deletion_order(source, "random", "SITE/sample", 0, 0, 2026)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(4)))
        quarter = perturb_key_subgraph(
            source, "random", 0.25, "SITE/sample", 0, 0, seed=2026
        )
        half = perturb_key_subgraph(
            source, "random", 0.50, "SITE/sample", 0, 0, seed=2026
        )
        self.assertEqual(
            quarter["edge_perturbation"]["deleted_source_positions"], first[:1]
        )
        self.assertEqual(
            half["edge_perturbation"]["deleted_source_positions"], first[:2]
        )
        targeted = perturb_key_subgraph(
            source, "targeted", 0.50, "SITE/sample", 0, 0, seed=2026
        )
        self.assertEqual(
            targeted["edge_perturbation"]["deleted_edge_count"],
            half["edge_perturbation"]["deleted_edge_count"],
        )

    def test_payloads_share_exact_filtered_tuple_and_dose_inventory(self):
        payloads, audit = build_edge_perturbation_payloads(
            _payload(), seed=2026
        )
        self.assertTrue(audit["included"])
        self.assertEqual(audit["key_tuple_count"], 3)
        self.assertEqual(audit["matched_tuple_count"], 2)
        self.assertEqual(audit["dropped_lt_two_edge_tuple_count"], 1)
        self.assertEqual(tuple(payloads), perturbation_sources())
        for source, payload in payloads.items():
            self.assertEqual(payload["subgraph_source"], source)
            self.assertEqual(
                [len(window["subgraphs"]) for window in payload["timepoints"]],
                [1, 1],
            )
            self.assertTrue(all("candidate_pool" not in window for window in payload["timepoints"]))
        for ratio in ("010", "025", "050"):
            targeted = payloads["key_edge_targeted_{}".format(ratio)]
            random = payloads["key_edge_random_{}".format(ratio)]
            for targeted_window, random_window in zip(
                targeted["timepoints"], random["timepoints"]
            ):
                left = targeted_window["subgraphs"][0]["edge_perturbation"]
                right = random_window["subgraphs"][0]["edge_perturbation"]
                self.assertEqual(left["deleted_edge_count"], right["deleted_edge_count"])
                self.assertEqual(left["retained_edge_count"], right["retained_edge_count"])

    def test_invalid_ratio_inventory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "begin with zero"):
            build_edge_perturbation_payloads(_payload(), ratios=(0.1, 0.5))
        with self.assertRaisesRegex(ValueError, "increasing"):
            build_edge_perturbation_payloads(_payload(), ratios=(0.0, 0.5, 0.25))


if __name__ == "__main__":
    unittest.main()
