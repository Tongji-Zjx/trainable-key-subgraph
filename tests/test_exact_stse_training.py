from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.exact_stse_dataset import ExactSTSEBatch
from keysubgraph.models.exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
)
from keysubgraph.training.exact_stse_trainer import (
    ExactSTSETrainingConfig,
    load_exact_stse_checkpoint,
    train_exact_stse,
)
from tests.test_exact_stse_model import _exact_sample


class ExactSTSETrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(223)
        self.batch = ExactSTSEBatch(
            (
                _exact_sample("train-a", 0, 2),
                _exact_sample("train-b", 1, 3),
            )
        )

    def test_train_checkpoint_history_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            output = root / "run"
            config = ExactSTSEConfig(
                use_coordinates=False, dropout=0.0
            )
            model = ExactSTSEClassifier(config)
            model.reset_parameters_with_seed(42)
            result = train_exact_stse(
                model=model,
                train_loader=[self.batch],
                validation_loader=[self.batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=ExactSTSETrainingConfig(
                    epochs=2,
                    early_stopping_patience=0,
                    scheduler_patience=1,
                ),
                output_dir=output,
                protocol_path=protocol,
                protocol_sha256="unit-test",
            )
            self.assertEqual(result["epochs_completed"], 2)
            self.assertTrue((output / "best_checkpoint.pt").is_file())
            self.assertTrue((output / "last_checkpoint.pt").is_file())
            self.assertTrue((output / "best_evaluation.json").is_file())
            history = json.loads(
                (output / "history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(history), 2)

            resumed = ExactSTSEClassifier(config)
            resumed_result = train_exact_stse(
                model=resumed,
                train_loader=[self.batch],
                validation_loader=[self.batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=ExactSTSETrainingConfig(
                    epochs=3,
                    early_stopping_patience=0,
                    scheduler_patience=1,
                ),
                output_dir=output,
                protocol_path=protocol,
                protocol_sha256="unit-test",
                resume_checkpoint=output / "last_checkpoint.pt",
            )
            self.assertEqual(resumed_result["epochs_completed"], 3)
            checkpoint = load_exact_stse_checkpoint(
                output / "last_checkpoint.pt",
                resumed,
                torch.device("cpu"),
                expected_protocol_sha256="unit-test",
            )
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(
                checkpoint["model_config"]["use_coordinates"], False
            )

    def test_checkpoint_rejects_wrong_variant_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            model = ExactSTSEClassifier(
                ExactSTSEConfig(use_coordinates=False, dropout=0.0)
            )
            train_exact_stse(
                model,
                [self.batch],
                [self.batch],
                (0, 1),
                torch.device("cpu"),
                ExactSTSETrainingConfig(epochs=1),
                root / "run",
                protocol,
                "unit-test",
            )
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_exact_stse_checkpoint(
                    root / "run" / "best_checkpoint.pt",
                    ExactSTSEClassifier(
                        ExactSTSEConfig(
                            use_coordinates=True, dropout=0.0
                        )
                    ),
                    torch.device("cpu"),
                )
            with self.assertRaisesRegex(ValueError, "protocol hash"):
                load_exact_stse_checkpoint(
                    root / "run" / "best_checkpoint.pt",
                    ExactSTSEClassifier(
                        ExactSTSEConfig(
                            use_coordinates=False, dropout=0.0
                        )
                    ),
                    torch.device("cpu"),
                    expected_protocol_sha256="different",
                )


if __name__ == "__main__":
    unittest.main()
