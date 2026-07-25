from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from keysubgraph.models.dual_sgw_feature_classifier import (
    DualSGWFeatureClassifier,
    DualSGWFeatureClassifierConfig,
)
from keysubgraph.training.dual_sgw_feature_trainer import (
    DualSGWFeatureTrainingConfig,
    load_dual_sgw_feature_checkpoint,
    train_dual_sgw_feature_classifier,
)


class DualSGWFeatureClassifierTest(unittest.TestCase):
    def test_linear_and_small_mlp_have_the_registered_capacity(self):
        linear = DualSGWFeatureClassifier(
            DualSGWFeatureClassifierConfig("linear")
        )
        small = DualSGWFeatureClassifier(
            DualSGWFeatureClassifierConfig("small_mlp")
        )
        values = torch.randn(5, 34)
        self.assertEqual(tuple(linear(values).shape), (5, 2))
        self.assertEqual(tuple(small(values).shape), (5, 2))
        self.assertEqual(
            sum(parameter.numel() for parameter in linear.parameters()),
            70,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in small.parameters()),
            594,
        )
        with self.assertRaises(ValueError):
            linear(torch.randn(5, 33))

    def test_training_saves_validation_thresholds_and_reloadable_checkpoint(
        self,
    ):
        samples = []
        for index in range(12):
            label = index % 2
            features = torch.zeros(34)
            features[0] = -1.0 if label == 0 else 1.0
            features[1] = float(index) / 12.0
            samples.append(
                {
                    "sample_key": "sample-{}".format(index),
                    "label": label,
                    "features": features,
                }
            )
        train_loader = DataLoader(samples[:8], batch_size=4, shuffle=False)
        validation_loader = DataLoader(
            samples[8:], batch_size=4, shuffle=False
        )
        provenance = {"protocol_sha256": "protocol"}
        model = DualSGWFeatureClassifier(
            DualSGWFeatureClassifierConfig("linear")
        )
        with tempfile.TemporaryDirectory() as directory:
            result = train_dual_sgw_feature_classifier(
                model,
                train_loader,
                validation_loader,
                [item["label"] for item in samples[:8]],
                torch.device("cpu"),
                DualSGWFeatureTrainingConfig(
                    epochs=2,
                    early_stopping_patience=0,
                    seed=7,
                ),
                Path(directory),
                provenance,
            )
            self.assertTrue(Path(result["best_checkpoint"]).is_file())
            self.assertEqual(
                set(result["validation_thresholds"]),
                {"balanced_accuracy", "accuracy"},
            )
            reloaded = DualSGWFeatureClassifier(
                DualSGWFeatureClassifierConfig("linear")
            )
            payload = load_dual_sgw_feature_checkpoint(
                result["best_checkpoint"],
                reloaded,
                torch.device("cpu"),
                expected_provenance=provenance,
            )
            self.assertEqual(payload["training_config"]["seed"], 7)
            self.assertIsNotNone(payload["validation_thresholds"])


if __name__ == "__main__":
    unittest.main()
