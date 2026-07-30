from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.data.data_split import file_sha256
from keysubgraph.features.structured_short_term_features import (
    StructuredShortTermStandardizer,
)
from keysubgraph.models.structured_short_term import (
    StructuredShortTermClassifier,
    StructuredShortTermConfig,
)
from keysubgraph.training.structured_short_term_trainer import (
    StructuredShortTermTrainingConfig,
    evaluate_structured_short_term,
    load_structured_short_term_checkpoint,
    train_structured_short_term,
)
from tests.test_structured_short_term import (
    _graph_sample,
    _identity_standardizer,
)


class StructuredShortTermTrainingTest(unittest.TestCase):
    def test_checkpoint_threshold_and_evaluation_round_trip(self):
        torch.manual_seed(99)
        train_batch = GraphSequenceBatch(
            (
                _graph_sample("train-0", 0, 2),
                _graph_sample("train-1", 1, 3),
            )
        )
        validation_batch = GraphSequenceBatch(
            (
                _graph_sample("validation-0", 0, 2, split="validation"),
                _graph_sample("validation-1", 1, 4, split="validation"),
            )
        )
        config = StructuredShortTermConfig(
            hidden_dim=16,
            node_ffn_dim=24,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=32,
            memory_slots=4,
            statistics_embedding_dim=8,
            classifier_hidden_dims=(12, 6),
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text('{"immutable": true}\n', encoding="utf-8")
            protocol_hash = file_sha256(protocol)
            base = _identity_standardizer()
            standardizer = StructuredShortTermStandardizer(
                node_mean=base.node_mean,
                node_std=base.node_std,
                community_mean=base.community_mean,
                community_std=base.community_std,
                train_sample_count=base.train_sample_count,
                train_window_count=base.train_window_count,
                train_node_count=base.train_node_count,
                protocol_sha256=protocol_hash,
                edge_presence_threshold=0.0,
            )
            standardizer_path = root / "standardizer.json"
            standardizer.save(standardizer_path)
            model = StructuredShortTermClassifier(config, standardizer)
            result = train_structured_short_term(
                model=model,
                train_loader=[train_batch],
                validation_loader=[validation_batch],
                train_labels=(0, 1),
                device=torch.device("cpu"),
                training_config=StructuredShortTermTrainingConfig(
                    epochs=2,
                    learning_rate=1.0e-3,
                    early_stopping_patience=0,
                    scheduler_patience=1,
                    selection_metric="roc_auc",
                    seed=42,
                ),
                output_dir=root / "training",
                protocol_path=protocol,
                protocol_sha256=protocol_hash,
                standardizer_path=standardizer_path,
                standardizer_sha256=file_sha256(standardizer_path),
            )
            self.assertTrue(result["best_checkpoint"].is_file())
            checkpoint = load_structured_short_term_checkpoint(
                result["best_checkpoint"],
                model,
                torch.device("cpu"),
                protocol_hash,
                file_sha256(standardizer_path),
            )
            self.assertEqual(
                set(checkpoint["validation_thresholds"]),
                {"balanced_accuracy", "accuracy"},
            )
            metrics = evaluate_structured_short_term(
                model,
                [validation_batch],
                torch.device("cpu"),
                checkpoint["class_weights"],
                checkpoint["validation_thresholds"]["balanced_accuracy"],
            )
            self.assertEqual(metrics["sample_count"], 2)
            self.assertEqual(len(metrics["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
