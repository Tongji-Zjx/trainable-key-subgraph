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

from keysubgraph.data.data_protocol import (  # noqa: E402
    freeze_data_protocol,
    validate_data_protocol,
)
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceBatch,
    GraphSequenceDataset,
    create_data_loader,
)


def _graph(node_count, negative=True):
    adjacency = torch.zeros(node_count, node_count)
    if node_count > 1:
        adjacency[0, 1] = 0.5
        adjacency[1, 0] = 0.5
    if negative and node_count > 2:
        adjacency[1, 2] = -0.25
        adjacency[2, 1] = -0.25
    return adjacency


def _payload(node_counts):
    graphs = [_graph(count) for count in node_counts]
    communities = [
        torch.tensor([0] * max(1, count - 1) + [1], dtype=torch.long)
        for count in node_counts
    ]
    coordinates = [
        torch.arange(1, count * 3 + 1, dtype=torch.float32).reshape(count, 3)
        for count in node_counts
    ]
    names = [
        ["node_{}".format(index) for index in range(count)] for count in node_counts
    ]
    same_size = len(set(node_counts)) == 1
    return {
        "adjacency": torch.stack(graphs) if same_size else graphs,
        "coords": coordinates[0] if same_size else coordinates,
        "node_names": names[0] if same_size else names,
        "community_sequence": torch.stack(communities) if same_size else communities,
        "window_starts": torch.arange(len(node_counts), dtype=torch.float32),
        "global_threshold": 0.123,
        "t_r": 2.0,
    }


class GraphDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "data" / "dataset"
        self.index_path = self.root / "outputs" / "index" / "sample_index.csv"
        self.splits_csv = self.root / "outputs" / "splits" / "splits.csv"
        self.splits_json = self.root / "outputs" / "splits" / "splits.json"
        self.rows = []
        specifications = [
            ("train_a", 0, "train", [3, 2]),
            ("train_b", 1, "train", [4]),
            ("train_c", 0, "train", [3]),
            ("train_d", 1, "train", [2, 3, 4]),
            ("valid_a", 0, "validation", [3]),
            ("test_a", 1, "test", [4]),
        ]
        for name, label, split, node_counts in specifications:
            relative_path = "SITE/{}/SITE_{}_1.pt".format(label, name)
            path = self.dataset_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_payload(node_counts), str(path))
            self.rows.append(
                {
                    "sample_key": "SITE/SITE_{}_1".format(name),
                    "sample_id": "SITE_{}_1".format(name),
                    "site": "SITE",
                    "subject_id": name,
                    "session_id": "1",
                    "label": label,
                    "relative_path": relative_path,
                    "included": "True",
                    "group_id": "SITE::{}".format(name),
                    "split": split,
                    "seed": 42,
                }
            )
        self._write_artifacts()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_artifacts(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index_fields = [
            "sample_key",
            "sample_id",
            "site",
            "subject_id",
            "session_id",
            "label",
            "relative_path",
            "included",
        ]
        with self.index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=index_fields)
            writer.writeheader()
            writer.writerows(
                {name: row[name] for name in index_fields} for row in self.rows
            )

        self.splits_csv.parent.mkdir(parents=True, exist_ok=True)
        split_fields = [
            "sample_key",
            "sample_id",
            "site",
            "subject_id",
            "session_id",
            "group_id",
            "label",
            "relative_path",
            "split",
            "seed",
        ]
        assignments = [
            {name: row[name] for name in split_fields} for row in self.rows
        ]
        with self.splits_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=split_fields)
            writer.writeheader()
            writer.writerows(assignments)
        json_assignments = []
        for row in assignments:
            converted = dict(row)
            converted["label"] = int(converted["label"])
            converted["seed"] = int(converted["seed"])
            json_assignments.append(converted)
        with self.splits_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_index_sha256": file_sha256(self.index_path),
                    "seed": 42,
                    "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                    "group_key": "site::subject_id",
                    "assignments": json_assignments,
                },
                handle,
            )

    def _dataset(self, split="train"):
        return GraphSequenceDataset(
            self.dataset_root,
            self.index_path,
            self.splits_csv,
            split=split,
            edge_presence_threshold=0.0,
        )

    def test_dataset_preserves_variable_lengths_and_signed_edges(self):
        sample = self._dataset()[0]

        self.assertEqual(sample.node_counts, (3, 2))
        self.assertEqual(sample.num_timepoints, 2)
        self.assertEqual(tuple(sample.communities[0].shape), (3,))
        self.assertTrue(sample.edge_mask[0][1, 2])
        self.assertLess(float(sample.adjacency[0][1, 2]), 0.0)
        self.assertEqual(sample.label, 0)
        self.assertEqual(sample.source_global_threshold, 0.123)
        self.assertEqual(sample.edge_presence_threshold, 0.0)

    def test_edge_mask_uses_absolute_threshold_for_negative_edges(self):
        dataset = GraphSequenceDataset(
            self.dataset_root,
            self.index_path,
            self.splits_csv,
            split="train",
            edge_presence_threshold=0.2,
        )
        sample = dataset[0]

        self.assertTrue(bool(sample.edge_mask[0][0, 1]))
        self.assertTrue(bool(sample.edge_mask[0][1, 2]))
        self.assertFalse(bool(sample.edge_mask[0][0, 2]))
        self.assertLess(float(sample.adjacency[0][1, 2]), 0.0)

    def test_coordinates_are_optional_and_non_blocking(self):
        first_path = self.dataset_root / self.rows[0]["relative_path"]
        first_payload = torch.load(str(first_path), map_location="cpu", weights_only=False)
        first_payload["coords"] = [torch.zeros(3, 3), torch.zeros(2, 3)]
        torch.save(first_payload, str(first_path))

        second_path = self.dataset_root / self.rows[1]["relative_path"]
        second_payload = torch.load(str(second_path), map_location="cpu", weights_only=False)
        second_payload.pop("coords")
        torch.save(second_payload, str(second_path))

        dataset = self._dataset()
        zero_coordinate_sample = dataset[0]
        missing_coordinate_sample = dataset[1]

        self.assertEqual(zero_coordinate_sample.node_counts, (3, 2))
        self.assertEqual(missing_coordinate_sample.node_counts, (4,))
        self.assertFalse(hasattr(zero_coordinate_sample, "coordinates"))
        self.assertFalse(hasattr(missing_coordinate_sample, "coordinates"))

    def test_list_batch_does_not_pad_or_truncate(self):
        batch = next(iter(create_data_loader(self._dataset(), batch_size=4, seed=7)))

        self.assertIsInstance(batch, GraphSequenceBatch)
        self.assertEqual(len(batch), 4)
        self.assertEqual(sorted(sample.num_timepoints for sample in batch), [1, 1, 2, 3])
        self.assertIn((3, 2), [sample.node_counts for sample in batch])
        self.assertIn((2, 3, 4), [sample.node_counts for sample in batch])
        self.assertEqual(tuple(batch.labels.shape), (4,))

    def test_loader_order_is_reproducible_and_eval_cannot_shuffle(self):
        dataset = self._dataset()
        first = [
            key
            for batch in create_data_loader(dataset, batch_size=2, seed=99)
            for key in batch.sample_keys
        ]
        second = [
            key
            for batch in create_data_loader(dataset, batch_size=2, seed=99)
            for key in batch.sample_keys
        ]
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "must not shuffle"):
            create_data_loader(self._dataset("validation"), 1, shuffle=True)

    def test_all_partition_uses_every_sample_and_can_shuffle(self):
        for row in self.rows:
            row["split"] = "all"
        self._write_artifacts()

        dataset = self._dataset("all")
        loaded_keys = {
            key
            for batch in create_data_loader(dataset, batch_size=2, seed=17)
            for key in batch.sample_keys
        }

        self.assertEqual(len(dataset), len(self.rows))
        self.assertEqual(loaded_keys, {row["sample_key"] for row in self.rows})

    def test_protocol_freezes_and_validates_artifact_hashes(self):
        protocol_path = self.root / "configs" / "data_protocol.json"
        payload = freeze_data_protocol(
            project_root=self.root,
            dataset_root=self.dataset_root,
            sample_index_csv=self.index_path,
            splits_csv=self.splits_csv,
            splits_json=self.splits_json,
            output_path=protocol_path,
        )

        self.assertEqual(payload["sample_count"], 6)
        self.assertEqual(payload["edge_presence_threshold"], 0.0)
        self.assertEqual(validate_data_protocol(protocol_path, self.root), payload)
        with self.assertRaises(FileExistsError):
            freeze_data_protocol(
                self.root,
                self.dataset_root,
                self.index_path,
                self.splits_csv,
                self.splits_json,
                protocol_path,
            )


if __name__ == "__main__":
    unittest.main()
