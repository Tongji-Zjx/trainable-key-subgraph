from __future__ import absolute_import, division, print_function

import csv
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

from keysubgraph.data.sample_index import (  # noqa: E402
    IndexBuildConfig,
    build_sample_index,
    inspect_sample,
    write_index_artifacts,
)


def _signed_adjacency(node_count):
    graph = torch.zeros(node_count, node_count, dtype=torch.float32)
    if node_count >= 2:
        graph[0, 1] = 0.6
        graph[1, 0] = 0.6
    if node_count >= 3:
        graph[1, 2] = -0.4
        graph[2, 1] = -0.4
    return graph


def _payload(graphs, communities, coordinates=None, node_names=None):
    if node_names is None:
        counts = [int(graph.shape[0]) for graph in graphs]
        if len(set(counts)) == 1:
            node_names = ["node_{}".format(index) for index in range(counts[0])]
        else:
            node_names = [
                ["node_{}".format(index) for index in range(count)]
                for count in counts
            ]
    adjacency = torch.stack(graphs) if len({tuple(item.shape) for item in graphs}) == 1 else graphs
    community_sequence = (
        torch.stack(communities)
        if len({tuple(item.shape) for item in communities}) == 1
        else communities
    )
    payload = {
        "adjacency": adjacency,
        "node_names": node_names,
        "community_sequence": community_sequence,
        "window_starts": torch.arange(len(graphs), dtype=torch.float32),
        "global_threshold": 0.5,
        "t_r": 2.0,
    }
    if coordinates is not None:
        payload["coords"] = coordinates
    return payload


class SampleIndexTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _save(self, site, label, filename, payload):
        path = self.root / site / str(label) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        return path

    def test_valid_signed_sample_is_included_and_identity_is_parsed(self):
        graphs = [_signed_adjacency(3), _signed_adjacency(3)]
        communities = [torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])]
        coordinates = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        path = self._save(
            "SITE",
            1,
            "SITE_subject42_2.pt",
            _payload(graphs, communities, coordinates),
        )

        record = inspect_sample(path, IndexBuildConfig(self.root))

        self.assertTrue(record.included)
        self.assertEqual(record.label, 1)
        self.assertEqual(record.site, "SITE")
        self.assertEqual(record.subject_id, "subject42")
        self.assertEqual(record.session_id, "2")
        self.assertEqual(record.num_timepoints, 2)
        self.assertEqual(record.min_num_nodes, 3)
        self.assertEqual(record.max_num_nodes, 3)
        self.assertTrue(record.has_positive_edges)
        self.assertTrue(record.has_negative_edges)

    def test_variable_node_counts_are_preserved(self):
        graphs = [_signed_adjacency(3), _signed_adjacency(2)]
        communities = [torch.tensor([0, 0, 1]), torch.tensor([0, 0])]
        coordinates = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        ]
        path = self._save(
            "VARIABLE",
            0,
            "VARIABLE_subject_1.pt",
            _payload(graphs, communities, coordinates),
        )

        record = inspect_sample(path, IndexBuildConfig(self.root))

        self.assertTrue(record.included)
        self.assertEqual(record.num_timepoints, 2)
        self.assertEqual(record.min_num_nodes, 2)
        self.assertEqual(record.max_num_nodes, 3)

    def test_invalid_community_and_empty_graph_are_excluded(self):
        graph = torch.zeros(3, 3)
        communities = [torch.tensor([0, -1, 1])]
        coordinates = torch.zeros(3, 3)
        path = self._save(
            "BAD",
            1,
            "BAD_subject_1.pt",
            _payload([graph], communities, coordinates),
        )

        record = inspect_sample(path, IndexBuildConfig(self.root))

        self.assertFalse(record.included)
        reasons = set(record.exclusion_reasons.split("|"))
        self.assertIn("community_has_negative_label", reasons)
        self.assertNotIn("coordinates_all_zero", reasons)
        self.assertIn("empty_timepoints:1", reasons)

    def test_zero_and_missing_coordinates_are_included(self):
        graph = _signed_adjacency(3)
        communities = [torch.tensor([0, 0, 1])]
        zero_path = self._save(
            "ZERO",
            0,
            "ZERO_subject_1.pt",
            _payload([graph], communities, torch.zeros(3, 3)),
        )
        missing_path = self._save(
            "MISSING",
            1,
            "MISSING_subject_1.pt",
            _payload([graph], communities),
        )

        zero_record = inspect_sample(zero_path, IndexBuildConfig(self.root))
        missing_record = inspect_sample(missing_path, IndexBuildConfig(self.root))

        self.assertTrue(zero_record.included)
        self.assertFalse(zero_record.coords_valid)
        self.assertTrue(missing_record.included)
        self.assertFalse(missing_record.coords_valid)

    def test_artifacts_are_deterministic_and_partition_inventory(self):
        valid_payload = _payload(
            [_signed_adjacency(3)],
            [torch.tensor([0, 0, 1])],
            torch.eye(3),
        )
        invalid_payload = _payload(
            [_signed_adjacency(3)],
            [torch.tensor([0, -1, 1])],
            torch.zeros(3, 3),
        )
        self._save("Z_SITE", 0, "Z_SITE_b_1.pt", valid_payload)
        self._save("A_SITE", 1, "A_SITE_a_1.pt", invalid_payload)

        records = build_sample_index(IndexBuildConfig(self.root))
        output_directory = self.root / "index"
        paths = write_index_artifacts(records, output_directory)

        self.assertEqual([record.site for record in records], ["A_SITE", "Z_SITE"])
        with paths["summary"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["included_samples"], 1)
        self.assertEqual(summary["excluded_samples"], 1)

        with paths["inventory"].open("r", encoding="utf-8", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        with paths["index"].open("r", encoding="utf-8", newline="") as handle:
            included = list(csv.DictReader(handle))
        with paths["exclusions"].open("r", encoding="utf-8", newline="") as handle:
            excluded = list(csv.DictReader(handle))
        self.assertEqual(len(inventory), len(included) + len(excluded))
        self.assertEqual(included[0]["site"], "Z_SITE")
        self.assertEqual(excluded[0]["site"], "A_SITE")


if __name__ == "__main__":
    unittest.main()
