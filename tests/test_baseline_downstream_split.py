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

from keysubgraph.data.baseline_downstream_split import (  # noqa: E402
    create_baseline_downstream_splits,
)
from keysubgraph.data.baseline_manifest import read_baseline_manifest  # noqa: E402
from keysubgraph.data.data_split import SplitConfig, file_sha256  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.training.baseline_trainer import (  # noqa: E402
    BaselineTrainingConfig,
    read_baseline_checkpoint_payload,
    train_baseline,
)
from tests.test_baseline_model import _sequence_batch  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


class BaselineDownstreamSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "data").mkdir()
        index_path = self.root / "index.csv"
        splits_csv = self.root / "source_splits.csv"
        splits_json = self.root / "source_splits.json"
        index_path.write_text("index\n", encoding="utf-8")
        splits_csv.write_text("splits\n", encoding="utf-8")
        splits_json.write_text("{}\n", encoding="utf-8")
        self.protocol_path = self.root / "protocol.json"
        _write_json(
            self.protocol_path,
            {
                "schema_version": 1,
                "immutable": True,
                "experiment_mode": "all_samples_exploratory",
                "paths": {
                    "dataset_root": "data",
                    "sample_index_csv": "index.csv",
                    "splits_csv": "source_splits.csv",
                    "splits_json": "source_splits.json",
                },
                "sha256": {
                    "sample_index_csv": file_sha256(index_path),
                    "splits_csv": file_sha256(splits_csv),
                    "splits_json": file_sha256(splits_json),
                },
                "edge_presence_threshold": 0.0,
            },
        )
        records = []
        for label in (0, 1):
            for subject_index in range(6):
                subject_id = "label{}_subject{}".format(label, subject_index)
                for session_index in range(2):
                    sample_id = "{}_session{}".format(subject_id, session_index)
                    sample_key = "SITE/{}".format(sample_id)
                    export_path = self.root / "exports" / (sample_id + ".json")
                    _write_json(export_path, {"sample_key": sample_key})
                    records.append(
                        {
                            "sample_key": sample_key,
                            "sample_id": sample_id,
                            "site": "SITE",
                            "subject_id": subject_id,
                            "session_id": str(session_index),
                            "label": label,
                            "split": "all",
                            "source_split": "all",
                            "relative_path": "unused/{}.pt".format(sample_id),
                            "hard_subgraph_json": export_path.as_posix(),
                            "hard_subgraph_sha256": file_sha256(export_path),
                            "timepoint_count": 2,
                            "subgraph_count": 3,
                            "checkpoint_sha256": "extractor-hash",
                            "data_protocol_sha256": file_sha256(self.protocol_path),
                            "edge_presence_threshold": 0.0,
                        }
                    )
        self.parent_manifest = self.root / "parent" / "baseline_manifest.json"
        _write_json(
            self.parent_manifest,
            {
                "schema_version": 1,
                "immutable": True,
                "evidence_level": "exploratory_in_sample",
                "data_protocol": self.protocol_path.as_posix(),
                "data_protocol_sha256": file_sha256(self.protocol_path),
                "split": "all",
                "source_split": "all",
                "checkpoint_sha256": "extractor-hash",
                "hard_extraction_config": {"top_k": 5},
                "sample_count": len(records),
                "timepoint_count": 2 * len(records),
                "subgraph_count": 3 * len(records),
                "records": records,
            },
        )
        self.config = SplitConfig(seed=42, max_class_ratio_deviation=0.05)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_split_is_reproducible_stratified_and_group_aware(self):
        first = create_baseline_downstream_splits(
            self.root, self.parent_manifest, self.root / "first", self.config
        )
        second = create_baseline_downstream_splits(
            self.root, self.parent_manifest, self.root / "second", self.config
        )
        with Path(first["splits_json"]).open("r", encoding="utf-8") as handle:
            first_payload = json.load(handle)
        with Path(second["splits_json"]).open("r", encoding="utf-8") as handle:
            second_payload = json.load(handle)
        first_assignments = first_payload["assignments"]
        second_assignments = second_payload["assignments"]
        self.assertEqual(first_assignments, second_assignments)
        self.assertEqual(first_payload["summary"]["total_samples"], 24)
        self.assertFalse(first_payload["summary"]["checks"]["sample_overlap"])
        self.assertFalse(first_payload["summary"]["checks"]["group_overlap"])
        self.assertTrue(
            first_payload["summary"]["checks"]["class_ratios_reasonable"]
        )
        subject_splits = {}
        for assignment in first_assignments:
            previous = subject_splits.setdefault(
                assignment["subject_id"], assignment["split"]
            )
            self.assertEqual(previous, assignment["split"])

    def test_derived_manifests_bind_parent_and_split_artifacts(self):
        result = create_baseline_downstream_splits(
            self.root, self.parent_manifest, self.root / "derived", self.config
        )
        sets = {}
        for split, path in result["manifests"].items():
            payload, records = read_baseline_manifest(path, self.root)
            self.assertEqual(payload["split"], split)
            self.assertEqual(payload["source_split"], "all")
            self.assertEqual(
                payload["parent_manifest_sha256"], file_sha256(self.parent_manifest)
            )
            self.assertTrue(all(record.split == split for record in records))
            self.assertTrue(all(record.source_split == "all" for record in records))
            sets[split] = {record.sample_key for record in records}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertEqual(len(set.union(*sets.values())), 24)

        splits_csv = Path(result["splits_csv"])
        splits_csv.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split artifact hash mismatch"):
            read_baseline_manifest(result["manifests"]["train"], self.root)

    def test_non_all_parent_is_rejected(self):
        with self.parent_manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["split"] = "train"
        payload["source_split"] = "train"
        for record in payload["records"]:
            record["split"] = "train"
            record["source_split"] = "train"
        invalid = self.root / "invalid_parent.json"
        _write_json(invalid, payload)
        with self.assertRaisesRegex(ValueError, "split='all'"):
            create_baseline_downstream_splits(
                self.root, invalid, self.root / "invalid_output", self.config
            )

    def test_training_checkpoint_binds_parent_and_downstream_split(self):
        result = create_baseline_downstream_splits(
            self.root, self.parent_manifest, self.root / "training_split", self.config
        )
        train_path = Path(result["manifests"]["train"])
        validation_path = Path(result["manifests"]["validation"])
        _, train_records = read_baseline_manifest(train_path, self.root)
        model = SignedSequenceBaseline(
            BaselineModelConfig(
                node_hidden_dim=8,
                signed_gnn_layers=1,
                signed_gnn_dropout=0.0,
                fusion_dim=12,
                gru_hidden_dim=10,
                classifier_hidden_dim=6,
                classifier_dropout=0.0,
            )
        )
        training_result = train_baseline(
            model=model,
            train_loader=[_sequence_batch()],
            validation_loader=[_sequence_batch()],
            train_labels=[record.label for record in train_records],
            device=torch.device("cpu"),
            config=BaselineTrainingConfig(epochs=1, early_stopping_patience=1),
            output_dir=self.root / "training_output",
            train_manifest_path=train_path,
            validation_manifest_path=validation_path,
            project_root=self.root,
        )
        checkpoint = read_baseline_checkpoint_payload(
            training_result["best_checkpoint"]
        )
        self.assertEqual(
            checkpoint["parent_manifest_sha256"], file_sha256(self.parent_manifest)
        )
        self.assertEqual(
            checkpoint["downstream_splits_json_sha256"],
            file_sha256(Path(result["splits_json"])),
        )


if __name__ == "__main__":
    unittest.main()
