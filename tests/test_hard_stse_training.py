from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.models.hard_stse_loss import HardSTSELossConfig
from keysubgraph.models.hard_stse_temporal_sgw import (
    HardSTSETemporalSGWClassifier,
)
from keysubgraph.models.hard_stse_types import HardSTSEConfig
from keysubgraph.training.hard_stse_trainer import (
    HardSTSETrainingConfig,
    load_hard_stse_checkpoint,
    train_hard_stse,
)
from tests.test_full_graph_classifier import _sample


class HardSTSETrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(163)
        self.batch = GraphSequenceBatch(
            (_sample("train-a", 0, 2), _sample("train-b", 1, 3))
        )

    def test_train_checkpoint_history_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            output = root / "run"
            model = HardSTSETemporalSGWClassifier(
                HardSTSEConfig(dropout=0.0)
            )
            result = train_hard_stse(
                model=model,
                train_loader=[self.batch],
                validation_loader=[self.batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=HardSTSETrainingConfig(
                    epochs=2,
                    early_stopping_patience=0,
                    scheduler_patience=1,
                ),
                loss_config=HardSTSELossConfig(),
                output_dir=output,
                protocol_path=protocol,
                protocol_sha256="unit-test",
            )
            self.assertEqual(result["epochs_completed"], 2)
            self.assertTrue((output / "best_checkpoint.pt").is_file())
            self.assertTrue((output / "last_checkpoint.pt").is_file())
            history = json.loads(
                (output / "history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(history), 2)
            self.assertTrue(bool(model.window_encoder.graph_statistic_fitted))

            resumed = HardSTSETemporalSGWClassifier(
                HardSTSEConfig(dropout=0.0)
            )
            resumed_result = train_hard_stse(
                model=resumed,
                train_loader=[self.batch],
                validation_loader=[self.batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=HardSTSETrainingConfig(
                    epochs=3,
                    early_stopping_patience=0,
                    scheduler_patience=1,
                ),
                loss_config=HardSTSELossConfig(),
                output_dir=output,
                protocol_path=protocol,
                protocol_sha256="unit-test",
                resume_checkpoint=output / "last_checkpoint.pt",
            )
            self.assertEqual(resumed_result["epochs_completed"], 3)
            checkpoint = load_hard_stse_checkpoint(
                output / "last_checkpoint.pt",
                resumed,
                torch.device("cpu"),
                expected_protocol_sha256="unit-test",
            )
            self.assertEqual(checkpoint["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
