from __future__ import absolute_import, division, print_function

import tempfile
import unittest
import json
from pathlib import Path

import torch

from keysubgraph.crossfit.author_short_term_runner import (
    AUTHOR_SHORT_TERM_BRANCH,
    build_author_short_term_crossfit_fold_commands,
)
from keysubgraph.crossfit.author_short_term_summary import (
    summarize_author_short_term_crossfit,
)
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.graph_dataset import GraphSequenceBatch
from keysubgraph.features.paper_short_term_pst import (
    PaperShortTermCommunityFrequency,
)
from keysubgraph.models.author_short_term import (
    AuthorNoCoordinateShortTermClassifier,
    AuthorShortTermConfig,
    author_short_term_config,
)
from keysubgraph.training.author_short_term_trainer import (
    AuthorBalancedBatchSampler,
    AuthorShortTermTrainingConfig,
    evaluate_author_short_term,
    fit_author_threshold,
    model_from_author_short_term_checkpoint,
    train_author_short_term,
)
from tests.test_structured_short_term import _graph_sample


def _frequency():
    return PaperShortTermCommunityFrequency(
        counts=((3, 12), (7, 12)),
        total_count=24,
        train_sample_count=2,
        train_window_count=5,
        protocol_sha256="a" * 64,
        train_manifest_sha256="b" * 64,
        train_sample_keys_sha256="c" * 64,
        outer_fold=0,
    )


def _small_config():
    return AuthorShortTermConfig(
        window_embedding_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        memory_slots=4,
        memory_dim=8,
        transformer_dropout=0.0,
        window_dropout=0.0,
        maximum_windows=8,
        maximum_nodes=4,
    )


class AuthorShortTermModelTest(unittest.TestCase):
    def test_profile_hyperparameters_match_author_wrappers(self):
        adhd = author_short_term_config("adhd")
        wmrc = author_short_term_config("wmrc")
        self.assertEqual(adhd.window_embedding_dim, 192)
        self.assertEqual(adhd.transformer_layers, 3)
        self.assertEqual(wmrc.window_embedding_dim, 96)
        self.assertEqual(wmrc.transformer_layers, 2)
        self.assertEqual(adhd.transformer_heads, 8)
        self.assertEqual(adhd.community_embedding_dim, 32)
        self.assertEqual(adhd.memory_slots, 32)
        self.assertEqual(adhd.memory_dim, 128)

    def test_variable_sequence_forward_uses_no_coordinates_and_exact_first_delta(self):
        torch.manual_seed(91)
        left = _graph_sample("left", 0, 2)
        right = _graph_sample("right", 1, 4, node_count=3)
        model = AuthorNoCoordinateShortTermClassifier(
            _small_config(), _frequency()
        ).eval()
        memory = model.memory.memory.detach().clone()
        with torch.no_grad():
            output = model(GraphSequenceBatch((left, right)))
        self.assertEqual(tuple(output.logits.shape), (2,))
        self.assertEqual(output.time_mask.sum(dim=1).tolist(), [2, 4])
        self.assertEqual(tuple(output.memory_attention.shape), (2, 4))
        self.assertTrue(
            torch.allclose(
                output.memory_attention.sum(dim=1), torch.ones(2)
            )
        )
        self.assertFalse(output.diagnostics["uses_coordinates"])
        self.assertEqual(output.diagnostics["node_feature_dim"], 34)
        first_delta = output.diagnostics[
            "first_window_delta_absolute_degree"
        ]
        expected = left.adjacency[0].abs().sum(dim=1)
        self.assertTrue(torch.allclose(first_delta[0, :4], expected))
        self.assertTrue(torch.equal(memory, model.memory.memory))

    def test_author_padding_semantics_are_batch_dependent_only_in_graph_stats(self):
        torch.manual_seed(92)
        short = _graph_sample("short", 0, 2)
        long = _graph_sample("long", 1, 4)
        model = AuthorNoCoordinateShortTermClassifier(
            _small_config(), _frequency()
        ).eval()
        with torch.no_grad():
            alone = model(GraphSequenceBatch((short,)))
            together = model(GraphSequenceBatch((short, long)))
        self.assertEqual(tuple(alone.graph_statistics.shape), (1, 3))
        self.assertFalse(
            torch.allclose(
                alone.graph_statistics[0], together.graph_statistics[0]
            )
        )
        self.assertTrue(together.diagnostics["author_padding_semantics"])


class AuthorShortTermTrainingTest(unittest.TestCase):
    def test_balanced_sampler_and_threshold_follow_author_rules(self):
        sampler = AuthorBalancedBatchSampler([0, 0, 0, 1], 4, 42)
        batch = next(iter(sampler))
        labels = [[0, 0, 0, 1][index] for index in batch]
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)
        threshold = fit_author_threshold(
            [0, 1], [0.2, 0.8], metric="balanced"
        )
        self.assertAlmostEqual(threshold, 0.21)

    def test_training_checkpoint_and_frozen_threshold_round_trip(self):
        train = GraphSequenceBatch(
            (
                _graph_sample("train-0", 0, 2),
                _graph_sample("train-1", 1, 3),
            )
        )
        validation = GraphSequenceBatch(
            (
                _graph_sample("val-0", 0, 2, split="validation"),
                _graph_sample("val-1", 1, 3, split="validation"),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            frequency_path = root / "frequency.json"
            protocol.write_text('{"immutable": true}\n', encoding="utf-8")
            frequency = _frequency()
            frequency.save(frequency_path)
            model = AuthorNoCoordinateShortTermClassifier(
                _small_config(), frequency
            )
            config = AuthorShortTermTrainingConfig(
                profile="adhd",
                epochs=1,
                learning_rate=1.0e-3,
                early_stopping_minimum_epochs=0,
                early_stopping_patience=0,
                scheduler_minimum_learning_rate=1.0e-6,
                seed=42,
            )
            result = train_author_short_term(
                model,
                [train],
                [validation],
                (0, 1),
                torch.device("cpu"),
                config,
                root / "training",
                protocol,
                file_sha256(protocol),
                frequency_path,
                file_sha256(frequency_path),
            )
            self.assertTrue(result["best_checkpoint"].is_file())
            self.assertTrue(
                (root / "training" / "best_accuracy_checkpoint.pt").is_file()
            )
            self.assertTrue(
                (root / "training" / "best_roc_auc_checkpoint.pt").is_file()
            )
            restored, checkpoint = model_from_author_short_term_checkpoint(
                result["best_checkpoint"], torch.device("cpu")
            )
            self.assertEqual(
                set(checkpoint["validation_thresholds"]),
                {"balanced_accuracy", "accuracy"},
            )
            metrics = evaluate_author_short_term(
                restored,
                [validation],
                torch.device("cpu"),
                checkpoint["positive_class_weight"],
                checkpoint["training_config"]["label_smoothing"],
                checkpoint["validation_thresholds"]["balanced_accuracy"],
            )
            self.assertEqual(metrics["sample_count"], 2)
            self.assertEqual(len(metrics["predictions"]), 2)


class AuthorShortTermCrossfitRunnerTest(unittest.TestCase):
    def test_plan_uses_author_profile_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_author_short_term_crossfit_fold_commands(
                Path.cwd(), root, 1, "adhd", epochs=12
            )
        self.assertEqual(
            [item[0] for item in plan],
            [
                "community_frequency",
                "train",
                "evaluate_validation",
                "evaluate_test",
            ],
        )
        for _, _, artifact in plan:
            self.assertIn(AUTHOR_SHORT_TERM_BRANCH, str(artifact))
        train = plan[1][1]
        self.assertEqual(train[train.index("--profile") + 1], "adhd")
        self.assertEqual(train[train.index("--epochs") + 1], "12")
        self.assertEqual(train[train.index("--batch-size") + 1], "32")
        self.assertEqual(
            train[train.index("--seed") + 1], "784341473"
        )
        self.assertNotIn("--standardizer", train)
        frequency = plan[0][1]
        self.assertEqual(frequency[frequency.index("--outer-fold") + 1], "1")

    def test_summary_requires_exact_oof_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outer = {
                0: (("a", 0, 0.1), ("b", 1, 0.9)),
                1: (("c", 0, 0.2), ("d", 1, 0.8)),
            }
            validation = {0: outer[1], 1: outer[0]}
            assignment_rows = []
            for fold in (0, 1):
                for role, values in (
                    ("outer_test", outer[fold]),
                    ("inner_validation", validation[fold]),
                ):
                    for key, label, _ in values:
                        assignment_rows.append(
                            {
                                "outer_fold": fold,
                                "role": role,
                                "sample_key": key,
                                "site": "S",
                                "label": label,
                            }
                        )
            assignments = root / "assignments" / "fold_assignments.json"
            assignments.parent.mkdir(parents=True)
            assignments.write_text(
                json.dumps(
                    {
                        "purpose": "confirmatory_cross_fitted_fold_roles",
                        "immutable": True,
                        "num_outer_folds": 2,
                        "assignments": assignment_rows,
                    }
                ),
                encoding="utf-8",
            )
            seed = 784341473
            for fold in (0, 1):
                evaluation = (
                    root
                    / "fold_{}".format(fold)
                    / AUTHOR_SHORT_TERM_BRANCH
                    / "evaluation_seed{}".format(seed)
                )
                evaluation.mkdir(parents=True)
                for split, values in (
                    ("validation", validation[fold]),
                    ("test", outer[fold]),
                ):
                    predictions = [
                        {
                            "sample_key": key,
                            "site": "S",
                            "label": label,
                            "positive_probability": probability,
                            "prediction": int(probability >= 0.5),
                        }
                        for key, label, probability in values
                    ]
                    payload = {
                        "model_name": "author_no_coordinate_short_term",
                        "profile": "adhd",
                        "split": split,
                        "threshold_source": "frozen_validation",
                        "threshold_fit_split": "validation",
                        "threshold_strategy": "balanced_accuracy",
                        "threshold": 0.5,
                        "metrics": {},
                        "predictions": predictions,
                    }
                    (evaluation / "{}_evaluation.json".format(split)).write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            result = summarize_author_short_term_crossfit(
                root, assignments, "adhd"
            )
            self.assertEqual(result["metrics"]["sample_count"], 4)
            self.assertAlmostEqual(
                result["metrics"]["pooled_oof_roc_auc"], 1.0
            )
            self.assertTrue(result["summary_markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
