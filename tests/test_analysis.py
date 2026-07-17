from __future__ import absolute_import, division, print_function

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.controls import (  # noqa: E402
    generate_random_controls,
    generate_top_degree_controls,
    select_low_score_controls,
)
from keysubgraph.analysis.statistics import (  # noqa: E402
    apply_bh_fdr,
    run_structural_analysis,
)
from keysubgraph.analysis.original_graph import (  # noqa: E402
    build_original_graph_record,
    compute_original_graph_metrics,
    iter_original_graph_metrics,
    iter_original_graph_records,
)
from keysubgraph.analysis.structural_metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_sample_metrics,
    compute_subgraph_metrics,
)
from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: E402
from keysubgraph.data.baseline_controls import (  # noqa: E402
    build_matched_source_payloads,
)
from keysubgraph.data.baseline_dataset import _local_subgraph  # noqa: E402
from keysubgraph.data.key_rewiring import (  # noqa: E402
    build_key_rewired_payloads,
    rewire_key_subgraph,
)
from keysubgraph.features.graph_features import GraphFeatureBuilder  # noqa: E402


def _record(sample_id="sample", label=0):
    return {
        "sample_id": sample_id,
        "site": "SITE",
        "label": label,
        "split": "test",
        "fold": None,
        "time_index": 1,
        "subgraph_index": 0,
        "node_ids": [0, 1, 2],
        "node_names": ["a", "b", "c"],
        "edge_index": [[0, 1], [1, 2]],
        "original_edge_weights": [0.5, -0.25],
        "edge_presence_threshold": 0.0,
        "community_labels": [0, 0, 1],
        "delta_degree": [1.0, -2.0, 0.0],
        "delta_degree_mask": [True, True, True],
        "delta_edge_weight": [0.1, -0.2],
        "delta_edge_mask": [True, True],
        "time_mask": True,
        "node_mask": [True, True, True],
        "subgraph_mask": True,
        "num_valid_subgraphs": 1,
        "original_graph_ref": "SITE/0/sample.pt",
        "candidate_pool_ref": "sample#time=1",
        "source": "key",
        "repeat_index": None,
    }


def _control_sample():
    graph = torch.tensor(
        [
            [0.0, 0.5, -0.3, 0.2, 0.1],
            [0.5, 0.0, 0.4, -0.2, 0.3],
            [-0.3, 0.4, 0.0, 0.6, -0.1],
            [0.2, -0.2, 0.6, 0.0, 0.5],
            [0.1, 0.3, -0.1, 0.5, 0.0],
        ]
    )
    mask = graph.abs() > 0
    mask.fill_diagonal_(False)
    return GraphSequenceSample(
        sample_key="SITE/sample",
        sample_id="sample",
        site="SITE",
        subject_id="subject",
        session_id="1",
        label=0,
        split="test",
        relative_path="SITE/0/sample.pt",
        adjacency=(graph,),
        edge_mask=(mask,),
        node_names=(("a", "b", "c", "d", "e"),),
        communities=(torch.tensor([0, 0, 1, 1, 1]),),
        window_starts=torch.tensor([0.0]),
        source_global_threshold=0.1,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


class AnalysisTest(unittest.TestCase):
    def test_signed_key_rewiring_is_reproducible_and_preserves_invariants(self):
        key = _record()
        key.update({
            "time_index": 0,
            "node_ids": [0, 1, 2, 3, 4],
            "node_names": ["a", "b", "c", "d", "e"],
            "community_labels": [0, 0, 1, 1, 1],
            "edge_index": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "original_edge_weights": [0.5, 0.4, 0.6, 0.5],
            "delta_edge_weight": [0.1, 0.2, 0.3, 0.4],
            "delta_edge_mask": [True, True, True, True],
        })
        # Include a negative edge so sign-specific preservation is exercised.
        key["edge_index"][1] = [0, 2]
        key["original_edge_weights"][1] = -0.3

        first = rewire_key_subgraph(key, "SITE/sample", 0, 0, seed=17)
        second = rewire_key_subgraph(key, "SITE/sample", 0, 0, seed=17)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(first["source"], "key_rewired")
        self.assertEqual(first["node_ids"], key["node_ids"])
        self.assertEqual(len(first["edge_index"]), len(key["edge_index"]))
        self.assertNotEqual(
            {tuple(edge) for edge in first["edge_index"]},
            {tuple(edge) for edge in key["edge_index"]},
        )
        self.assertEqual(
            sorted(first["original_edge_weights"]),
            sorted(key["original_edge_weights"]),
        )
        self.assertEqual(
            sum(value > 0.0 for value in first["original_edge_weights"]), 3
        )
        self.assertEqual(
            sum(value < 0.0 for value in first["original_edge_weights"]), 1
        )
        self.assertEqual(
            len(first["rewiring"]["source_edge_index"]), len(key["edge_index"])
        )

    def test_key_rewired_dataset_validates_provenance_and_rebuilds_features(self):
        sample = _control_sample()
        key = _record()
        key.update({
            "time_index": 0,
            "node_ids": [0, 1, 2, 3, 4],
            "node_names": ["a", "b", "c", "d", "e"],
            "community_labels": [0, 0, 1, 1, 1],
            "edge_index": [[0, 1], [0, 2], [1, 2], [2, 3]],
            "original_edge_weights": [0.5, -0.3, 0.4, 0.6],
        })
        rewired = rewire_key_subgraph(key, sample.sample_key, 0, 0, seed=23)
        subgraph = _local_subgraph(rewired, sample, 0, GraphFeatureBuilder())

        self.assertEqual(subgraph.edge_count, 4)
        self.assertTrue(torch.equal(subgraph.edge_mask, subgraph.adjacency.abs() > 0.0))
        self.assertTrue(any(float(value) < 0.0 for value in subgraph.edge_weight))
        self.assertTrue(torch.allclose(
            subgraph.node_features[:, 0], subgraph.adjacency.abs().sum(dim=1)
        ))

        tampered = dict(rewired)
        tampered["original_edge_weights"] = list(rewired["original_edge_weights"])
        tampered["original_edge_weights"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "provenance"):
            _local_subgraph(tampered, sample, 0, GraphFeatureBuilder())

    def test_key_rewired_payload_drops_unrewirable_tuples_and_matches_inventory(self):
        key = _record()
        key.update({
            "time_index": 0,
            "node_ids": [0, 1, 2, 3],
            "node_names": ["a", "b", "c", "d"],
            "community_labels": [0, 0, 1, 1],
            "edge_index": [[0, 1], [1, 2]],
            "original_edge_weights": [0.5, -0.25],
        })
        complete = dict(key)
        complete.update({
            "node_ids": [0, 1, 2],
            "node_names": ["a", "b", "c"],
            "community_labels": [0, 0, 1],
            "edge_index": [[0, 1], [0, 2], [1, 2]],
            "original_edge_weights": [0.5, -0.3, 0.4],
        })
        payload = {
            "sample_key": "SITE/sample",
            "timepoints": [{
                "time_index": 0,
                "time_mask": True,
                "subgraphs": [key, complete],
                "candidate_pool": [key, complete],
                "num_valid_subgraphs": 2,
            }],
        }

        sources, audit = build_key_rewired_payloads(payload, seed=31)

        self.assertTrue(audit["included"])
        self.assertEqual(audit["key_tuple_count"], 2)
        self.assertEqual(audit["matched_tuple_count"], 1)
        self.assertEqual(set(sources), {"key", "key_rewired"})
        for source_payload in sources.values():
            timepoint = source_payload["timepoints"][0]
            self.assertNotIn("candidate_pool", timepoint)
            self.assertEqual(timepoint["num_valid_subgraphs"], 1)

    def test_signed_metrics_and_community_ratios(self):
        metrics = compute_subgraph_metrics(_record())

        self.assertEqual(metrics["node_count"], 3.0)
        self.assertEqual(metrics["edge_count"], 2.0)
        self.assertAlmostEqual(metrics["density"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["abs_edge_weight_mean"], 0.375)
        self.assertAlmostEqual(metrics["abs_connection_sum"], 0.75)
        self.assertAlmostEqual(metrics["positive_connection_sum"], 0.5)
        self.assertAlmostEqual(metrics["negative_connection_magnitude_sum"], 0.25)
        self.assertAlmostEqual(metrics["node_dynamic_mean_abs"], 1.0)
        self.assertAlmostEqual(metrics["edge_dynamic_mean_abs"], 0.15)
        self.assertEqual(metrics["positive_intra_ratio"], 1.0)
        self.assertEqual(metrics["positive_inter_ratio"], 0.0)
        self.assertEqual(metrics["negative_intra_ratio"], 0.0)
        self.assertEqual(metrics["negative_inter_ratio"], 1.0)

    def test_sample_aggregation_ignores_metric_nan(self):
        first = compute_subgraph_metrics(_record())
        second = dict(first)
        first["node_dynamic_mean_abs"] = float("nan")
        second["node_dynamic_mean_abs"] = 2.0
        aggregated = aggregate_sample_metrics([first, second])[0]
        self.assertEqual(aggregated["valid_subgraph_count"], 2)
        self.assertEqual(aggregated["node_dynamic_mean_abs"], 2.0)
        self.assertEqual(aggregated["node_dynamic_mean_abs__valid_count"], 1)

    def test_controls_are_matched_and_reproducible(self):
        sample = _control_sample()
        key = _record()
        key.update(
            {
                "time_index": 0,
                "node_ids": [0, 1, 2],
                "edge_index": [[0, 1], [1, 2]],
                "score_connectivity": 1.0,
            }
        )
        low = dict(key)
        low.update(
            {
                "node_ids": [2, 3, 4],
                "node_names": ["c", "d", "e"],
                "edge_index": [[2, 3], [3, 4]],
                "original_edge_weights": [0.6, 0.5],
                "community_labels": [1, 1, 1],
                "delta_degree": [0.0, 0.0, 0.0],
                "delta_degree_mask": [False, False, False],
                "delta_edge_weight": [0.0, 0.0],
                "delta_edge_mask": [False, False],
                "candidate_score": 0.1,
                "seed_node": 3,
            }
        )
        key["candidate_score"] = 0.9
        key["seed_node"] = 1
        payload = {
            "timepoints": [
                {"time_index": 0, "subgraphs": [key], "candidate_pool": [key, low]}
            ]
        }
        first = generate_random_controls(sample, payload, repeats=3, seed=9)
        second = generate_random_controls(sample, payload, repeats=3, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(len(row["node_ids"]) == 3 for row in first))
        self.assertTrue(all(len(row["edge_index"]) == 2 for row in first))
        top = generate_top_degree_controls(sample, payload)
        self.assertEqual(len(top), 1)
        self.assertEqual(len(top[0]["node_ids"]), 3)
        self.assertEqual(len(top[0]["edge_index"]), 2)
        low_controls = select_low_score_controls(payload)
        self.assertEqual(len(low_controls), 1)
        self.assertEqual(low_controls[0]["source"], "low_score")
        self.assertEqual(low_controls[0]["node_ids"], [2, 3, 4])

    def test_baseline_control_payloads_share_exact_tuple_inventory(self):
        sample = _control_sample()
        key = _record()
        key.update(
            {
                "time_index": 0,
                "node_ids": [0, 1, 2],
                "edge_index": [[0, 1], [1, 2]],
                "score_connectivity": 1.0,
                "candidate_score": 0.9,
                "seed_node": 1,
            }
        )
        low = dict(key)
        low.update(
            {
                "node_ids": [2, 3, 4],
                "node_names": ["c", "d", "e"],
                "edge_index": [[2, 3], [3, 4]],
                "original_edge_weights": [0.6, 0.5],
                "community_labels": [1, 1, 1],
                "candidate_score": 0.1,
                "seed_node": 3,
            }
        )
        payload = {
            "timepoints": [
                {
                    "time_index": 0,
                    "time_mask": True,
                    "num_valid_subgraphs": 1,
                    "subgraphs": [key],
                    "candidate_pool": [key, low],
                }
            ]
        }

        sources, audit = build_matched_source_payloads(
            sample, payload, random_seed=9, random_repeat_index=0
        )

        self.assertTrue(audit["included"])
        self.assertEqual(audit["matched_tuple_count"], 1)
        self.assertEqual(
            set(sources), {"key", "low_score", "top_degree", "random"}
        )
        for source, source_payload in sources.items():
            timepoint = source_payload["timepoints"][0]
            self.assertNotIn("candidate_pool", timepoint)
            self.assertEqual(timepoint["num_valid_subgraphs"], 1)
            self.assertEqual(timepoint["subgraphs"][0]["source"], source)
            self.assertEqual(len(timepoint["subgraphs"][0]["node_ids"]), 3)
            self.assertEqual(len(timepoint["subgraphs"][0]["edge_index"]), 2)

        payload["timepoints"][0]["candidate_pool"] = [key]
        sources, audit = build_matched_source_payloads(sample, payload, random_seed=9)
        self.assertEqual(sources, {})
        self.assertFalse(audit["included"])
        self.assertEqual(audit["empty_timepoints"], [0])

    def test_original_graph_record_uses_all_nodes_and_signed_edges(self):
        sample = _control_sample()
        record = build_original_graph_record(sample, 0)

        self.assertEqual(record["source"], "original")
        self.assertEqual(record["node_ids"], [0, 1, 2, 3, 4])
        self.assertEqual(len(record["edge_index"]), 10)
        self.assertTrue(any(weight > 0.0 for weight in record["original_edge_weights"]))
        self.assertTrue(any(weight < 0.0 for weight in record["original_edge_weights"]))
        self.assertTrue(all(left < right for left, right in record["edge_index"]))
        metrics = compute_subgraph_metrics(record)
        direct_metrics = compute_original_graph_metrics(sample, 0)
        self.assertEqual(metrics["node_count"], 5.0)
        self.assertEqual(metrics["edge_count"], 10.0)
        self.assertAlmostEqual(metrics["density"], 1.0)
        self.assertTrue(math.isnan(metrics["edge_dynamic_mean_abs"]))
        for metric in METRIC_NAMES:
            if math.isnan(metrics[metric]):
                self.assertTrue(math.isnan(direct_metrics[metric]))
            else:
                self.assertAlmostEqual(metrics[metric], direct_metrics[metric], places=6)
        self.assertEqual(list(iter_original_graph_records((sample,))), [record])
        iterated_metrics = list(iter_original_graph_metrics((sample,)))
        self.assertEqual(len(iterated_metrics), 1)
        self.assertEqual(iterated_metrics[0]["sample_id"], direct_metrics["sample_id"])
        for metric in METRIC_NAMES:
            if math.isnan(direct_metrics[metric]):
                self.assertTrue(math.isnan(iterated_metrics[0][metric]))
            else:
                self.assertAlmostEqual(
                    iterated_metrics[0][metric], direct_metrics[metric], places=6
                )

    def test_statistics_and_fdr_outputs(self):
        adjusted = apply_bh_fdr([0.01, 0.04, 0.03])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        records = []
        for label in (0, 1):
            for index in range(5):
                record = _record("{}_{}".format(label, index), label)
                shift = label * 0.2
                record["original_edge_weights"] = [0.5 + shift, -0.25 - shift]
                records.append(record)
        with tempfile.TemporaryDirectory() as temporary:
            paths = run_structural_analysis(records, Path(temporary))
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertEqual(len(METRIC_NAMES), 15)


if __name__ == "__main__":
    unittest.main()
