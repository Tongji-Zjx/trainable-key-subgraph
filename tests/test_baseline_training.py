from __future__ import absolute_import, division, print_function

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.training.baseline_trainer import (  # noqa: E402
    BaselineTrainingConfig,
    baseline_class_weights,
    baseline_metrics,
    load_baseline_checkpoint,
    select_balanced_accuracy_threshold,
    train_baseline,
)
from tests.test_baseline_model import _sequence_batch  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_manifest(root, protocol_path, split):
    manifest_dir = root / (split + "_manifest")
    manifest_path = manifest_dir / "baseline_manifest.json"
    record = {
        "sample_key": "SITE/{}_sample".format(split),
        "sample_id": "{}_sample".format(split),
        "site": "SITE",
        "subject_id": "{}_subject".format(split),
        "session_id": "1",
        "label": 0,
        "split": split,
        "relative_path": "unused.pt",
        "hard_subgraph_json": "unused.json",
        "hard_subgraph_sha256": "unused",
        "timepoint_count": 1,
        "subgraph_count": 1,
        "checkpoint_sha256": "extractor-checkpoint",
        "data_protocol_sha256": file_sha256(protocol_path),
        "edge_presence_threshold": 0.0,
    }
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "immutable": True,
            "evidence_level": "exploratory_in_sample",
            "data_protocol": "protocol.json",
            "data_protocol_sha256": file_sha256(protocol_path),
            "split": split,
            "checkpoint_sha256": "extractor-checkpoint",
            "sample_count": 1,
            "records": [record],
        },
    )
    return manifest_path


class BaselineTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "data").mkdir()
        artifacts = {}
        for name in ("index.csv", "splits.csv", "splits.json"):
            path = self.root / name
            path.write_text(name + "\n", encoding="utf-8")
            artifacts[name] = file_sha256(path)
        self.protocol_path = self.root / "protocol.json"
        _write_json(
            self.protocol_path,
            {
                "schema_version": 1,
                "immutable": True,
                "paths": {
                    "dataset_root": "data",
                    "sample_index_csv": "index.csv",
                    "splits_csv": "splits.csv",
                    "splits_json": "splits.json",
                },
                "sha256": {
                    "sample_index_csv": artifacts["index.csv"],
                    "splits_csv": artifacts["splits.csv"],
                    "splits_json": artifacts["splits.json"],
                },
                "edge_presence_threshold": 0.0,
            },
        )
        self.train_manifest = _write_manifest(
            self.root, self.protocol_path, "train"
        )
        self.validation_manifest = _write_manifest(
            self.root, self.protocol_path, "validation"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_threshold_metrics_and_class_weights(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]

        threshold = select_balanced_accuracy_threshold(labels, probabilities)
        metrics = baseline_metrics(labels, probabilities, threshold)
        weights = baseline_class_weights([0, 0, 0, 1])

        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertGreater(metrics["unweighted_log_loss"], 0.0)
        self.assertAlmostEqual(float(weights[0]), 2.0 / 3.0)
        self.assertAlmostEqual(float(weights[1]), 2.0)

    def test_train_checkpoint_history_and_reload(self):
        torch.manual_seed(12)
        config = BaselineModelConfig(
            node_hidden_dim=8,
            signed_gnn_layers=1,
            signed_gnn_dropout=0.0,
            fusion_dim=12,
            gru_hidden_dim=10,
            classifier_hidden_dim=6,
            classifier_dropout=0.0,
            history_mode="truncate_history",
            history_keep_ratio=0.5,
        )
        model = SignedSequenceBaseline(config)
        output_dir = self.root / "training"
        result = train_baseline(
            model,
            train_loader=[_sequence_batch()],
            validation_loader=[_sequence_batch()],
            train_labels=[0, 1],
            device=torch.device("cpu"),
            config=BaselineTrainingConfig(
                epochs=2,
                early_stopping_patience=2,
                seed=12,
            ),
            output_dir=output_dir,
            train_manifest_path=self.train_manifest,
            validation_manifest_path=self.validation_manifest,
            project_root=self.root,
        )

        self.assertTrue(result["best_checkpoint"].is_file())
        self.assertTrue(result["last_checkpoint"].is_file())
        with result["history"].open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        self.assertEqual(len(history), 2)
        self.assertIn("unweighted_log_loss", history[-1]["validation"])

        restored = SignedSequenceBaseline(config)
        checkpoint = load_baseline_checkpoint(
            result["best_checkpoint"], restored, device=torch.device("cpu")
        )
        self.assertEqual(checkpoint["model_config"], asdict(config))
        self.assertEqual(checkpoint["model_config"]["history_mode"], "truncate_history")
        self.assertEqual(checkpoint["model_config"]["history_keep_ratio"], 0.5)
        self.assertIn("classification_threshold", checkpoint)
        restored.eval()
        self.assertTrue(torch.isfinite(restored(_sequence_batch()).logits).all())

    def test_legacy_full_checkpoint_defaults_history_keep_ratio(self):
        config = BaselineModelConfig(
            node_hidden_dim=8,
            signed_gnn_layers=1,
            signed_gnn_dropout=0.0,
            fusion_dim=12,
            gru_hidden_dim=10,
            classifier_hidden_dim=6,
            classifier_dropout=0.0,
        )
        model = SignedSequenceBaseline(config)
        path = self.root / "legacy_full.pt"
        payload = {
            "schema_version": 1,
            "training_mode": "signed_sequence_baseline",
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        }
        del payload["model_config"]["history_keep_ratio"]
        torch.save(payload, str(path))

        restored = SignedSequenceBaseline(config)
        checkpoint = load_baseline_checkpoint(
            path, restored, device=torch.device("cpu")
        )

        self.assertNotIn("history_keep_ratio", checkpoint["model_config"])
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
