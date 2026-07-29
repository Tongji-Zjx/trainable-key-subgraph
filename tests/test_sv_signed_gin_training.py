from __future__ import absolute_import, division, print_function

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.sv_signed_gin_artifact import (
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
    save_sv_signed_gin_record,
)
from keysubgraph.data.sv_signed_gin_dataset import (
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_signed_gin_manifest import (
    write_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (
    fit_sv_signed_gin_standardizers,
    save_sv_signed_gin_standardizers,
)
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    SVSignedGINTrainingConfig,
    balanced_classification_loss,
    load_sv_signed_gin_checkpoint,
    train_sv_signed_gin_classifier,
)


def _record(key, label, split, offset):
    node = torch.full((3, 15), float(offset))
    adjacency = torch.tensor(
        ((0.0, 0.3, -0.2), (0.3, 0.0, 0.1), (-0.2, 0.1, 0.0))
    )
    return SVSignedGINRecord(
        sample_key=key,
        sample_id=key,
        subject_id=key,
        site="site-a",
        label=label,
        split=split,
        windows=(
            SVSignedGINWindowRecord(node, adjacency, 0.0),
            SVSignedGINWindowRecord(node + 0.1, adjacency, 1.0),
        ),
        static_features=torch.full((28,), float(offset)),
        variation=torch.full((16,), float(offset) / 2.0),
        window_mask=torch.tensor((True, True)),
        transition_mask=torch.tensor((True,)),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


def _write_manifest(root, name, records):
    pairs = []
    artifact_root = root / (name + "_artifacts")
    for record in records:
        path = artifact_root / (record.sample_key + ".pt")
        save_sv_signed_gin_record(record, path)
        pairs.append((record, path))
    manifest = root / name / "manifest.json"
    write_sv_signed_gin_manifest(pairs, manifest)
    return manifest


class SVSignedGINTrainingTest(unittest.TestCase):
    def test_class_weight_does_not_cancel_for_single_sample(self):
        logits = torch.zeros((1, 2), requires_grad=True)
        weights = torch.tensor((0.5, 2.0))
        negative = balanced_classification_loss(
            logits, torch.tensor((0,)), weights
        )
        positive = balanced_classification_loss(
            logits, torch.tensor((1,)), weights
        )
        self.assertAlmostEqual(
            float(positive / negative), 4.0, places=5
        )

    def test_microbatch_accumulation_matches_full_batch_gradient(self):
        full_logits = torch.tensor(
            ((0.2, -0.1), (-0.3, 0.4)), requires_grad=True
        )
        labels = torch.tensor((0, 1))
        weights = torch.tensor((0.75, 1.25))
        balanced_classification_loss(
            full_logits, labels, weights
        ).backward()
        full_gradient = full_logits.grad.clone()

        micro_logits = full_logits.detach().clone().requires_grad_(True)
        for index in range(2):
            loss = balanced_classification_loss(
                micro_logits[index : index + 1],
                labels[index : index + 1],
                weights,
            )
            (loss / 2.0).backward()
        self.assertTrue(
            torch.allclose(full_gradient, micro_logits.grad, atol=1.0e-7)
        )

    def test_training_saves_loadable_best_checkpoint_and_thresholds(self):
        train_records = (
            _record("train-0a", 0, "train", -1.0),
            _record("train-0b", 0, "train", -0.5),
            _record("train-1a", 1, "train", 0.5),
            _record("train-1b", 1, "train", 1.0),
        )
        validation_records = (
            _record("validation-0", 0, "validation", -0.75),
            _record("validation-1", 1, "validation", 0.75),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = _write_manifest(
                root, "train", train_records
            )
            validation_manifest = _write_manifest(
                root, "validation", validation_records
            )
            scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            scaler_path = root / "scaler.json"
            save_sv_signed_gin_standardizers(scaler, scaler_path)
            train = SVSignedGINDataset(train_manifest, scaler_path)
            validation = SVSignedGINDataset(
                validation_manifest, scaler_path
            )
            train_loader = create_sv_signed_gin_loader(
                train, 2, 42, True
            )
            validation_loader = create_sv_signed_gin_loader(
                validation, 2, 42, False
            )
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant="sv_static_variation", dropout=0.0
                )
            )
            provenance = {
                "protocol_sha256": "protocol",
                "selector_checkpoint_sha256": "selector",
                "selection_mode": "learned",
                "selection_seed": 42,
                "train_manifest_sha256": file_sha256(train_manifest),
                "validation_manifest_sha256": file_sha256(
                    validation_manifest
                ),
                "scaler_sha256": file_sha256(scaler_path),
            }
            output = root / "training"
            result = train_sv_signed_gin_classifier(
                model,
                train_loader,
                validation_loader,
                train.labels,
                torch.device("cpu"),
                SVSignedGINTrainingConfig(
                    epochs=2,
                    gradient_accumulation_steps=2,
                    early_stopping_patience=0,
                    selection_metric="composite_auc",
                    seed=42,
                ),
                output,
                provenance,
            )
            checkpoint = load_sv_signed_gin_checkpoint(
                output / "best_checkpoint.pt",
                model,
                torch.device("cpu"),
                provenance,
            )
            legacy_path = output / "legacy_checkpoint.pt"
            legacy = torch.load(
                str(output / "best_checkpoint.pt"),
                map_location=torch.device("cpu"),
            )
            legacy["model_config"].pop("gin_residual")
            legacy["model_config"].pop("gin_jumping_knowledge")
            legacy["model_config"].pop("gin_compact_readout")
            legacy["model_config"].pop("gin_batch_normalization")
            torch.save(legacy, str(legacy_path))
            legacy_checkpoint = load_sv_signed_gin_checkpoint(
                legacy_path,
                model,
                torch.device("cpu"),
                provenance,
            )
            evaluation_exists = (output / "best_evaluation.json").exists()
        self.assertEqual(result["epochs_completed"], 2)
        self.assertIn("balanced_accuracy", checkpoint["validation_thresholds"])
        self.assertEqual(
            legacy_checkpoint["model_config"]["variant"],
            "sv_static_variation",
        )
        self.assertTrue(evaluation_exists)

    def test_multibranch_training_records_auxiliary_loss(self):
        train_records = (
            _record("train-0a", 0, "train", -1.0),
            _record("train-0b", 0, "train", -0.5),
            _record("train-1a", 1, "train", 0.5),
            _record("train-1b", 1, "train", 1.0),
        )
        validation_records = (
            _record("validation-0", 0, "validation", -0.75),
            _record("validation-1", 1, "validation", 0.75),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = _write_manifest(
                root, "train", train_records
            )
            validation_manifest = _write_manifest(
                root, "validation", validation_records
            )
            scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            scaler_path = root / "scaler.json"
            save_sv_signed_gin_standardizers(scaler, scaler_path)
            train = SVSignedGINDataset(train_manifest, scaler_path)
            validation = SVSignedGINDataset(
                validation_manifest, scaler_path
            )
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant="signed_gin_multibranch_late_fusion",
                    gin_hidden_dim=8,
                    attention_hidden_dim=4,
                    channel_projection_dim=4,
                    fusion_hidden_dim=4,
                    dropout=0.0,
                    message_mode="signed_normalized",
                    pooling="mean_std",
                    gin_residual=True,
                    gin_jumping_knowledge=True,
                    gin_compact_readout=True,
                    gin_batch_normalization=True,
                )
            )
            output = root / "training"
            train_sv_signed_gin_classifier(
                model,
                create_sv_signed_gin_loader(
                    train, 2, 42, True
                ),
                create_sv_signed_gin_loader(
                    validation, 2, 42, False
                ),
                train.labels,
                torch.device("cpu"),
                SVSignedGINTrainingConfig(
                    epochs=1,
                    early_stopping_patience=0,
                    selection_metric="roc_auc",
                    auxiliary_loss_weight=0.25,
                    seed=42,
                ),
                output,
                {
                    "protocol_sha256": "protocol",
                    "selector_checkpoint_sha256": "selector",
                    "selection_mode": "learned",
                    "selection_seed": 42,
                    "train_manifest_sha256": file_sha256(
                        train_manifest
                    ),
                    "validation_manifest_sha256": file_sha256(
                        validation_manifest
                    ),
                    "scaler_sha256": file_sha256(scaler_path),
                },
            )
            history = json.loads(
                (output / "history.json").read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                (output / "best_evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            diagnostic_output = root / "diagnostic"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "diagnose_sv_late_fusion.py"
                    ),
                    "--checkpoint",
                    str(output / "best_checkpoint.pt"),
                    "--train-manifest",
                    str(train_manifest),
                    "--validation-manifest",
                    str(validation_manifest),
                    "--scaler",
                    str(scaler_path),
                    "--output-dir",
                    str(diagnostic_output),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            diagnostic = json.loads(
                (diagnostic_output / "diagnostic.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_train_manifest_sha256 = file_sha256(
                train_manifest
            )
        self.assertGreater(
            history[0]["train"]["auxiliary_loss"], 0.0
        )
        self.assertEqual(
            set(evaluation["branch_metrics"]),
            {"gin", "static_spectral", "variation"},
        )
        self.assertAlmostEqual(
            sum(evaluation["fusion_weights"].values()), 1.0, places=6
        )
        self.assertEqual(diagnostic["parameter_updates"], 0)
        self.assertEqual(
            diagnostic["provenance"]["train_manifest_sha256"],
            expected_train_manifest_sha256,
        )


if __name__ == "__main__":
    unittest.main()
