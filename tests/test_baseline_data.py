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

from keysubgraph.data.baseline_collate import baseline_padded_collate  # noqa: E402
from keysubgraph.data.baseline_dataset import (  # noqa: E402
    BaselineHardSubgraphDataset,
    BaselineSequenceSample,
    BaselineSubgraph,
    BaselineWindow,
)
from keysubgraph.data.baseline_manifest import build_baseline_manifest  # noqa: E402
from keysubgraph.data.data_protocol import freeze_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402


class BaselineDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "data"
        self.index_csv = self.root / "index.csv"
        self.splits_csv = self.root / "splits.csv"
        self.splits_json = self.root / "splits.json"
        self.protocol_path = self.root / "protocol.json"
        self.export_dir = self.root / "exports" / "validation"
        self.manifest_dir = self.root / "manifest"
        self.sample_id = "SITE_subject_1"
        self.sample_key = "SITE/{}".format(self.sample_id)
        self.relative_path = "SITE/1/{}.pt".format(self.sample_id)
        self._write_raw_sample()
        self._write_index_and_split()
        freeze_data_protocol(
            self.root,
            self.dataset_root,
            self.index_csv,
            self.splits_csv,
            self.splits_json,
            self.protocol_path,
            edge_presence_threshold=0.0,
            overwrite=True,
        )
        self._write_export()
        build_baseline_manifest(
            self.root,
            self.protocol_path,
            self.export_dir.parent,
            "validation",
            self.manifest_dir,
            overwrite=True,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_raw_sample(self):
        first = torch.zeros(4, 4)
        first[0, 1] = first[1, 0] = 0.5
        first[1, 3] = first[3, 1] = -0.3
        first[0, 3] = first[3, 0] = 0.2
        second = torch.zeros(3, 3)
        second[0, 2] = second[2, 0] = -0.4
        second[1, 2] = second[2, 1] = 0.25
        payload = {
            "adjacency": [first, second],
            "node_names": [
                ["node_0", "node_1", "node_2", "node_3"],
                ["node_0", "node_1", "node_2"],
            ],
            "community_sequence": [
                torch.tensor([0, 0, 1, 1], dtype=torch.long),
                torch.tensor([0, 1, 0], dtype=torch.long),
            ],
            "window_starts": torch.tensor([0.0, 1.0]),
            "global_threshold": 0.0,
            "t_r": 2.0,
        }
        path = self.dataset_root / self.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        control_path = self.dataset_root / "SITE/0/SITE_control_1.pt"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(control_path))

    def _write_index_and_split(self):
        index_row = {
            "sample_key": self.sample_key,
            "sample_id": self.sample_id,
            "site": "SITE",
            "subject_id": "subject",
            "session_id": "1",
            "label": 1,
            "relative_path": self.relative_path,
            "included": True,
        }
        control_index_row = {
            "sample_key": "SITE/SITE_control_1",
            "sample_id": "SITE_control_1",
            "site": "SITE",
            "subject_id": "control",
            "session_id": "1",
            "label": 0,
            "relative_path": "SITE/0/SITE_control_1.pt",
            "included": True,
        }
        with self.index_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index_row))
            writer.writeheader()
            writer.writerow(index_row)
            writer.writerow(control_index_row)
        assignment = {
            "sample_key": self.sample_key,
            "sample_id": self.sample_id,
            "site": "SITE",
            "subject_id": "subject",
            "session_id": "1",
            "group_id": "SITE::subject",
            "label": 1,
            "relative_path": self.relative_path,
            "split": "validation",
            "seed": 42,
        }
        control_assignment = {
            "sample_key": "SITE/SITE_control_1",
            "sample_id": "SITE_control_1",
            "site": "SITE",
            "subject_id": "control",
            "session_id": "1",
            "group_id": "SITE::control",
            "label": 0,
            "relative_path": "SITE/0/SITE_control_1.pt",
            "split": "train",
            "seed": 42,
        }
        with self.splits_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(assignment))
            writer.writeheader()
            writer.writerow(assignment)
            writer.writerow(control_assignment)
        with self.splits_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_index_sha256": file_sha256(self.index_csv),
                    "seed": 42,
                    "ratios": {"validation": 1.0},
                    "group_key": "site::subject_id",
                    "assignments": [assignment, control_assignment],
                },
                handle,
            )

    @staticmethod
    def _subgraph(time_index, node_ids, names, communities, edges, weights):
        return {
            "sample_id": "SITE_subject_1",
            "site": "SITE",
            "label": 1,
            "split": "validation",
            "time_index": time_index,
            "node_ids": node_ids,
            "node_names": names,
            "community_labels": communities,
            "edge_index": edges,
            "original_edge_weights": weights,
            "node_mask": [True] * len(node_ids),
            "subgraph_mask": True,
            "time_mask": True,
        }

    def _write_export(self):
        self.export_dir.mkdir(parents=True, exist_ok=True)
        first = self._subgraph(
            0,
            [3, 1, 0],
            ["node_3", "node_1", "node_0"],
            [1, 0, 0],
            [[3, 1], [1, 0]],
            [-0.3, 0.5],
        )
        second = self._subgraph(
            1,
            [2, 0],
            ["node_2", "node_0"],
            [0, 0],
            [[2, 0]],
            [-0.4],
        )
        payload = {
            "schema_version": 1,
            "sample_key": self.sample_key,
            "sample_id": self.sample_id,
            "site": "SITE",
            "subject_id": "subject",
            "session_id": "1",
            "label": 1,
            "split": "validation",
            "relative_path": self.relative_path,
            "edge_presence_threshold": 0.0,
            "hard_extraction_config": {"top_k": 1},
            "checkpoint_sha256": "checkpoint-hash",
            "data_protocol_sha256": file_sha256(self.protocol_path),
            "timepoints": [
                {
                    "time_index": 0,
                    "time_mask": True,
                    "num_valid_subgraphs": 1,
                    "subgraphs": [first],
                },
                {
                    "time_index": 1,
                    "time_mask": True,
                    "num_valid_subgraphs": 1,
                    "subgraphs": [second],
                },
            ],
        }
        with (self.export_dir / (self.sample_id + ".json")).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle)

    def _dataset(self, verify_exports=True):
        return BaselineHardSubgraphDataset(
            self.root,
            self.manifest_dir / "baseline_manifest.json",
            verify_exports=verify_exports,
        )

    def test_manifest_and_dataset_reconstruct_exact_exported_graph(self):
        sample = self._dataset()[0]
        subgraph = sample.windows[0].subgraphs[0]

        self.assertEqual(sample.num_timepoints, 2)
        self.assertEqual(tuple(subgraph.node_features.shape), (3, 12))
        self.assertEqual(subgraph.edge_index.tolist(), [[0, 1], [1, 2]])
        self.assertAlmostEqual(float(subgraph.adjacency[0, 1]), -0.3)
        self.assertAlmostEqual(float(subgraph.adjacency[1, 2]), 0.5)
        self.assertEqual(float(subgraph.adjacency[0, 2]), 0.0)
        self.assertTrue(bool((subgraph.adjacency < 0).any()))
        self.assertTrue(bool((subgraph.adjacency > 0).any()))

    def test_static_features_preserve_positive_and_negative_strength(self):
        subgraph = self._dataset()[0].windows[0].subgraphs[0]

        center = subgraph.node_features[1]
        self.assertAlmostEqual(float(center[0]), 0.8, places=6)
        self.assertAlmostEqual(float(center[1]), 0.5, places=6)
        self.assertAlmostEqual(float(center[2]), 0.3, places=6)
        self.assertAlmostEqual(float(center[3]), 0.625, places=6)
        self.assertAlmostEqual(float(center[4]), 0.375, places=6)

    def test_matched_control_manifest_binds_source_and_common_cohort(self):
        source_root = self.root / "matched_sources"
        low_export_dir = source_root / "low_score" / "validation"
        low_export_dir.mkdir(parents=True)
        original_path = self.export_dir / (self.sample_id + ".json")
        with original_path.open("r", encoding="utf-8") as handle:
            control_payload = json.load(handle)
        control_payload["subgraph_source"] = "low_score"
        control_path = low_export_dir / original_path.name
        with control_path.open("w", encoding="utf-8") as handle:
            json.dump(control_payload, handle)
        matched_path = source_root / "matched_control_manifest.json"
        with matched_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "immutable": True,
                    "purpose": "baseline_matched_subgraph_sources",
                    "split": "validation",
                    "data_protocol_sha256": file_sha256(self.protocol_path),
                    "sources": ["key", "low_score", "top_degree", "random"],
                    "included_sample_keys": [self.sample_key],
                    "source_records": {
                        "low_score": [
                            {
                                "sample_key": self.sample_key,
                                "sha256": file_sha256(control_path),
                            }
                        ]
                    },
                },
                handle,
            )
        control_manifest_dir = self.root / "low_score_manifest"

        payload = build_baseline_manifest(
            self.root,
            self.protocol_path,
            source_root / "low_score",
            "validation",
            control_manifest_dir,
            matched_control_manifest_path=matched_path,
            subgraph_source="low_score",
        )
        dataset = BaselineHardSubgraphDataset(
            self.root, control_manifest_dir / "baseline_manifest.json"
        )

        self.assertEqual(payload["subgraph_source"], "low_score")
        self.assertEqual(dataset[0].sample_key, self.sample_key)
        self.assertTrue(payload["matched_control_manifest_sha256"])

    def test_label_is_target_only_and_metadata_changes_are_rejected(self):
        dataset = self._dataset(verify_exports=False)
        original_features = dataset[0].windows[0].subgraphs[0].node_features.clone()
        export_path = self.export_dir / (self.sample_id + ".json")
        with export_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["label"] = 0
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            dataset[0]
        self.assertTrue(torch.isfinite(original_features).all())

    def test_derived_partition_reads_source_split_but_returns_downstream_split(self):
        parent_path = self.manifest_dir / "baseline_manifest.json"
        with parent_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        split_csv = self.root / "derived_splits.csv"
        split_json = self.root / "derived_splits.json"
        split_csv.write_text("frozen downstream split\n", encoding="utf-8")
        with split_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "purpose": "baseline_classifier_downstream_split",
                    "assignments": [
                        {"sample_key": self.sample_key, "split": "train"}
                    ],
                },
                handle,
            )
        payload.update(
            {
                "manifest_kind": "derived_downstream_partition",
                "split": "train",
                "source_split": "validation",
                "parent_manifest": parent_path.as_posix(),
                "parent_manifest_sha256": file_sha256(parent_path),
                "downstream_splits_csv": split_csv.as_posix(),
                "downstream_splits_csv_sha256": file_sha256(split_csv),
                "downstream_splits_json": split_json.as_posix(),
                "downstream_splits_json_sha256": file_sha256(split_json),
            }
        )
        for record in payload["records"]:
            record["split"] = "train"
            record["source_split"] = "validation"
        derived_path = self.root / "derived" / "baseline_manifest.json"
        derived_path.parent.mkdir()
        with derived_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        dataset = BaselineHardSubgraphDataset(self.root, derived_path)
        sample = dataset[0]

        self.assertEqual(dataset.source_split, "validation")
        self.assertEqual(dataset.split, "train")
        self.assertEqual(sample.split, "train")
        self.assertEqual(sample.label, 1)


class BaselineCollateTest(unittest.TestCase):
    @staticmethod
    def _subgraph(node_count, offset):
        adjacency = torch.zeros(node_count, node_count)
        edges = []
        weights = []
        for index in range(node_count - 1):
            weight = (0.1 + offset) * (-1.0 if index % 2 else 1.0)
            adjacency[index, index + 1] = weight
            adjacency[index + 1, index] = weight
            edges.append((index, index + 1))
            weights.append(weight)
        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
        return BaselineSubgraph(
            node_ids=torch.arange(node_count),
            node_names=tuple("node_{}".format(index) for index in range(node_count)),
            community_labels=torch.zeros(node_count, dtype=torch.long),
            adjacency=adjacency,
            edge_mask=adjacency.abs() > 0.0,
            node_features=torch.full((node_count, 12), float(offset)),
            edge_index=edge_index,
            edge_weight=torch.tensor(weights, dtype=torch.float32),
            structural_features=torch.arange(11, dtype=torch.float32),
            structural_mask=torch.ones(11, dtype=torch.bool),
        )

    @classmethod
    def _sample(cls, name, label, window_sizes):
        windows = []
        offset = 1
        for time_index, node_counts in enumerate(window_sizes):
            subgraphs = []
            for node_count in node_counts:
                subgraphs.append(cls._subgraph(node_count, offset))
                offset += 1
            windows.append(BaselineWindow(time_index, tuple(subgraphs)))
        return BaselineSequenceSample(
            sample_key="SITE/{}".format(name),
            sample_id=name,
            site="SITE",
            subject_id=name,
            session_id="1",
            label=label,
            split="train",
            windows=tuple(windows),
        )

    def test_padded_collate_preserves_all_variable_lengths(self):
        first = self._sample("first", 0, ((2, 3), (4,)))
        second = self._sample("second", 1, ((2,),))

        batch = baseline_padded_collate((first, second))

        self.assertEqual(tuple(batch.node_features.shape), (4, 4, 12))
        self.assertEqual(tuple(batch.adjacency.shape), (4, 4, 4))
        self.assertEqual(batch.subgraph_to_window.tolist(), [0, 0, 1, 2])
        self.assertEqual(batch.window_to_sample.tolist(), [0, 0, 1])
        self.assertEqual(batch.window_subgraph_count.tolist(), [2, 1, 1])
        self.assertEqual(batch.window_index.tolist(), [[0, 1], [2, -1]])
        self.assertEqual(batch.time_mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(batch.node_mask.sum(dim=1).tolist(), [2, 3, 4, 2])
        self.assertEqual(batch.labels.tolist(), [0, 1])

    def test_padding_regions_are_zero_and_masked(self):
        sample = self._sample("sample", 0, ((2, 4),))

        batch = baseline_padded_collate((sample,))

        self.assertFalse(bool(batch.node_mask[0, 2:].any()))
        self.assertEqual(float(batch.node_features[0, 2:].abs().sum()), 0.0)
        self.assertEqual(float(batch.adjacency[0, 2:, :].abs().sum()), 0.0)
        self.assertEqual(float(batch.adjacency[0, :, 2:].abs().sum()), 0.0)
        self.assertFalse(bool(batch.edge_mask[0, 2:, :].any()))


if __name__ == "__main__":
    unittest.main()
