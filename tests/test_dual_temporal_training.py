from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_temporal_dataset import DualTemporalBatch
from keysubgraph.models.dual_variation_temporal import (
    DualVariationTemporalClassifier,
    DualVariationTemporalConfig,
)
from keysubgraph.training.dual_variation_temporal_trainer import (
    DualTemporalTrainingConfig,
    load_dual_temporal_checkpoint,
    run_dual_temporal_epoch,
    train_dual_temporal_classifier,
)


def _batch():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    values = torch.zeros((4, 3, 16), dtype=torch.float32)
    values[:2] = -1.0
    values[2:] = 1.0
    return DualTemporalBatch(
        sample_keys=("a", "b", "c", "d"),
        labels=labels,
        transition_values=values,
        time_mask=torch.ones((4, 3), dtype=torch.bool),
        sequence_lengths=torch.tensor([3, 3, 3, 3], dtype=torch.long),
        base_logits=torch.zeros((4, 2), dtype=torch.float32),
    )


class DualTemporalTrainingTest(unittest.TestCase):
    def test_epoch_reports_final_and_temporal_metrics(self):
        model = DualVariationTemporalClassifier(
            DualVariationTemporalConfig(
                variant="T4_variation_bigru_residual", dropout=0.0
            )
        )
        metrics = run_dual_temporal_epoch(
            model,
            [_batch()],
            torch.device("cpu"),
            torch.ones(2),
        )
        self.assertEqual(metrics["sample_count"], 4)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["temporal_roc_auc"])
        self.assertAlmostEqual(metrics["alpha"], 0.1, places=5)

    def test_training_saves_best_checkpoint_and_frozen_thresholds(self):
        model = DualVariationTemporalClassifier(
            DualVariationTemporalConfig(
                variant="T1_variation_mean_mlp",
                projection_hidden_dim=8,
                temporal_output_dim=4,
                classifier_hidden_dim=4,
                dropout=0.0,
            )
        )
        provenance = {
            "train_manifest_sha256": "train",
            "validation_manifest_sha256": "validation",
            "temporal_scaler_sha256": "scaler",
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = train_dual_temporal_classifier(
                model=model,
                train_loader=[_batch()],
                validation_loader=[_batch()],
                train_labels=(0, 0, 1, 1),
                device=torch.device("cpu"),
                config=DualTemporalTrainingConfig(
                    epochs=2,
                    early_stopping_patience=0,
                    scheduler_patience=0,
                    seed=7,
                ),
                output_dir=Path(temporary),
                provenance=provenance,
            )
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            self.assertTrue(Path(result["best_evaluation"]).is_file())
            restored = DualVariationTemporalClassifier(model.config)
            payload = load_dual_temporal_checkpoint(
                result["best_checkpoint"],
                restored,
                torch.device("cpu"),
                expected_provenance=provenance,
            )
            self.assertEqual(
                set(payload["validation_thresholds"]),
                {"accuracy", "balanced_accuracy"},
            )
            self.assertEqual(payload["threshold_fit_split"], "validation")


if __name__ == "__main__":
    unittest.main()
