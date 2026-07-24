from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_sgw_scaler import DualSGWStandardizer
from keysubgraph.data.exact_stse_dataset import (
    ExactSTSEBatch,
    exact_stse_collate,
)
from keysubgraph.models.dual_stse_hard_sgw import (
    DualSTSEHardSGWClassifier,
)
from keysubgraph.models.dual_stse_hard_sgw_loss import (
    DualSTSEHardSGWLossConfig,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (
    DualTrainingConfig,
    load_dual_checkpoint,
    train_dual_stage,
)
from tests.test_exact_stse_model import _exact_sample


class DualTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(347)
        self.samples = (
            _exact_sample("dual-train-a", 0, 2),
            _exact_sample("dual-train-b", 1, 3),
        )
        self.batch = exact_stse_collate(self.samples)
        self.features = {
            sample.sample_key: torch.randn(34)
            for sample in self.samples
        }

    def test_sgw_stage_saves_auc_checkpoint_and_validation_threshold(self):
        model = DualSTSEHardSGWClassifier()
        model.set_sgw_standardizer(
            DualSGWStandardizer(
                torch.zeros(34),
                torch.ones(34),
                2,
                "protocol",
                "selector",
            )
        )
        provenance = {
            "stse_checkpoint_sha256": "stse",
            "selector_checkpoint_sha256": "selector",
            "sgw_scaler_sha256": "scaler",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = train_dual_stage(
                model=model,
                train_loader=[self.batch],
                validation_loader=[self.batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=DualTrainingConfig(
                    stage="sgw_classifier",
                    epochs=1,
                    early_stopping_patience=0,
                ),
                loss_config=DualSTSEHardSGWLossConfig(),
                output_dir=Path(directory) / "run",
                protocol_sha256="protocol",
                provenance=provenance,
                train_feature_lookup=self.features,
                validation_feature_lookup=self.features,
            )
            self.assertEqual(result["epochs_completed"], 1)
            self.assertTrue(
                0.0 <= result["validation_threshold"] <= 1.0
            )
            payload = load_dual_checkpoint(
                result["best_checkpoint"],
                model,
                torch.device("cpu"),
                expected_stage="sgw_classifier",
                expected_protocol_sha256="protocol",
                expected_provenance=provenance,
            )
            self.assertEqual(
                payload["selection_metric"], "validation_roc_auc"
            )
            self.assertIsNotNone(payload["validation_threshold"])
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_dual_checkpoint(
                    result["best_checkpoint"],
                    model,
                    torch.device("cpu"),
                    expected_provenance={"wrong": "value"},
                )


if __name__ == "__main__":
    unittest.main()
