from __future__ import absolute_import, division, print_function

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: E402
from keysubgraph.extraction.hard_extractor import (  # noqa: E402
    HardExtractionConfig,
    HardSubgraphExtractor,
    candidate_overlap,
    export_hard_sample,
)
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)


def _sample():
    graph = torch.tensor(
        [[0.0, -0.4, 0.2], [-0.4, 0.0, 0.3], [0.2, 0.3, 0.0]]
    )
    mask = graph.abs() > 0
    mask.fill_diagonal_(False)
    return GraphSequenceSample(
        sample_key="SITE/sample",
        sample_id="sample",
        site="SITE",
        subject_id="subject",
        session_id="1",
        label=1,
        split="validation",
        relative_path="SITE/1/sample.pt",
        adjacency=(graph,),
        edge_mask=(mask,),
        node_names=(("a", "b", "c"),),
        communities=(torch.tensor([0, 0, 0]),),
        window_starts=torch.tensor([0.0]),
        source_global_threshold=0.1,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


class HardExtractorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        model = SoftGraphClassifier(
            SoftExtractorConfig(
                node_score_hidden_dim=4,
                edge_score_hidden_dim=4,
                graph_hidden_dim=6,
                graph_layers=1,
                classifier_hidden_dim=4,
                dropout=0.0,
            )
        )
        self.config = HardExtractionConfig(
            seeds_per_community=1,
            neighborhood_hops=1,
            max_nodes=3,
            max_edges=3,
            min_nodes=2,
            min_edges=1,
            top_k=3,
        )
        self.extractor = HardSubgraphExtractor(model, self.config)

    def test_hard_path_is_frozen_and_keeps_negative_edges(self):
        result = self.extractor.extract_sample(_sample())

        self.assertTrue(all(not parameter.requires_grad for parameter in self.extractor.model.parameters()))
        self.assertEqual(len(result.timepoints), 1)
        timepoint = result.timepoints[0]
        self.assertEqual(len(timepoint.candidate_pool), 1)
        self.assertEqual(timepoint.num_valid_subgraphs, 1)
        self.assertEqual(timepoint.subgraph_mask, (True, False, False))
        self.assertIsNotNone(timepoint.union_graph)
        self.assertIsNotNone(timepoint.fidelity)
        self.assertEqual(len(timepoint.spectral_gw_greedy_trace), 1)
        candidate = timepoint.selected_subgraphs[0]
        self.assertEqual(candidate.node_ids, (0, 1, 2))
        self.assertEqual(len(candidate.edge_index), 3)
        self.assertTrue(any(weight < 0.0 for weight in candidate.original_edge_weights))
        self.assertTrue(all(left < right for left, right in candidate.edge_index))
        self.assertEqual(candidate.delta_degree_mask, (False, False, False))
        self.assertEqual(candidate.delta_edge_mask, (False, False, False))
        self.assertAlmostEqual(candidate_overlap(candidate, candidate), 2.0, places=6)

    def test_export_contains_required_schema_and_signed_weights(self):
        result = self.extractor.extract_sample(_sample())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "sample.json"
            export_hard_sample(
                result,
                output,
                self.config,
                checkpoint,
                "protocol-hash",
            )
            with output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            timepoint = payload["timepoints"][0]
            subgraph = timepoint["subgraphs"][0]
            required = {
                "sample_id",
                "site",
                "label",
                "split",
                "fold",
                "time_index",
                "subgraph_index",
                "node_ids",
                "node_names",
                "edge_index",
                "original_edge_weights",
                "edge_presence_threshold",
                "node_scores",
                "edge_scores",
                "candidate_score",
                "community_labels",
                "delta_degree",
                "delta_edge_weight",
                "delta_degree_mask",
                "delta_edge_mask",
                "time_mask",
                "node_mask",
                "subgraph_mask",
                "num_valid_subgraphs",
                "original_graph_ref",
                "candidate_pool_ref",
            }
            self.assertTrue(required <= set(subgraph))
            self.assertTrue(any(weight < 0.0 for weight in subgraph["original_edge_weights"]))
            self.assertEqual(len(timepoint["candidate_pool"]), 1)
            self.assertTrue(timepoint["hard_union_available"])
            self.assertEqual(timepoint["union_num_nodes"], 3)
            self.assertEqual(timepoint["union_num_edges"], 3)
            self.assertTrue(any(weight < 0.0 for weight in timepoint["union_original_edge_weights"]))
            self.assertIn("soft_to_hard_spectral_winf", timepoint)
            self.assertIn("soft_to_hard_gw_error", timepoint)
            self.assertIn("spectral_gw_greedy_trace", timepoint)
            self.assertIn("H_SGW_full", payload)
            self.assertIn("Gamma_SGW_hard", payload)
            with self.assertRaises(FileExistsError):
                export_hard_sample(
                    result,
                    output,
                    self.config,
                    checkpoint,
                    "protocol-hash",
                )


if __name__ == "__main__":
    unittest.main()
