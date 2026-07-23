from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.models import (
    FullGraphClassifierConfig,
    FullGraphClassifierOutput,
    FullGraphSequenceClassifier,
)
from keysubgraph.training import (
    FullGraphTrainingConfig,
    run_full_graph_classifier_epoch,
    train_full_graph_classifier,
)
from tests.test_full_graph_classifier import _sample


class _Loader(object):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch


class _FixedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0.2, -0.1]], dtype=torch.float32))

    def forward(self, batch):
        count = len(batch)
        logits = self.logits.expand(count, -1)
        return FullGraphClassifierOutput(
            logits=logits,
            representation=logits.new_zeros((count, 192)),
            sequence_lengths=torch.ones(count, dtype=torch.long),
            prototype_attention=None,
        )


class FullGraphTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.batch = GraphSequenceBatch(
            (_sample("train-a", 0, 1), _sample("train-b", 1, 2))
        )

    def test_batch_size_one_class_weight_is_not_cancelled(self):
        batch = GraphSequenceBatch((_sample("minority", 1, 1),))
        model = _FixedLogitModel()
        weights = torch.tensor([0.5, 2.0])
        metrics = run_full_graph_classifier_epoch(
            model,
            _Loader(batch),
            torch.device("cpu"),
            weights,
            optimizer=None,
        )
        expected = torch.nn.functional.cross_entropy(
            model.logits, torch.tensor([1])
        ).item()
        self.assertAlmostEqual(metrics["unweighted_log_loss"], expected, places=6)
        self.assertAlmostEqual(metrics["weighted_loss"], 2.0 * expected, places=6)

    def test_proto_usage_and_training_artifacts(self):
        model = FullGraphSequenceClassifier(
            FullGraphClassifierConfig(
                encoder_type="sgg_bigru_proto",
                signed_gnn_layers=1,
                classifier_hidden_dims=(8,),
                gated_gnn_dropout=0.0,
                classifier_dropout=0.0,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            result = train_full_graph_classifier(
                model,
                _Loader(self.batch),
                _Loader(self.batch),
                [0, 1],
                torch.device("cpu"),
                FullGraphTrainingConfig(
                    epochs=2,
                    weight_decay=0.0,
                    early_stopping_patience=0,
                    scheduler_patience=2,
                    seed=81,
                ),
                root / "run",
                protocol,
                "test-sha",
            )
            self.assertEqual(result["epochs_completed"], 2)
            self.assertTrue(result["best_checkpoint"].is_file())
            self.assertTrue(result["last_checkpoint"].is_file())
            evaluation = json.loads(
                result["best_evaluation"].read_text(encoding="utf-8")
            )
            usage = evaluation["train"]["prototype_usage"]
            self.assertEqual(len(usage), 16)
            self.assertAlmostEqual(sum(usage), 1.0, places=5)
            self.assertEqual(
                evaluation["selection"]["primary"], "validation_roc_auc"
            )


if __name__ == "__main__":
    unittest.main()
