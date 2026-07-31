from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.analysis.theory_neural_diagnostics import (
    build_theory_neural_diagnostics,
)

from keysubgraph.data.theory_neural_artifact import (
    TheoryNeuralRecord,
    TheoryNeuralWindowRecord,
    load_theory_neural_record,
    save_theory_neural_record,
)
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.theory_neural_features import TheoryNeuralFeatureBuilder
from keysubgraph.models.theory_guided_neural import (
    THEORY_NEURAL_VARIANTS,
    TheoryGuidedNeuralClassifier,
    TheoryNeuralBatch,
    TheoryNeuralConfig,
    EdgeAwareSignedLayer,
    TheoryNeuralSampleInput,
    TheoryNeuralWindowInput,
)
from keysubgraph.theory.sgw_core_features import SGWCoreConfig
from keysubgraph.training.sv_signed_gin_trainer import (
    balanced_classification_loss,
)
from keysubgraph.training.theory_guided_neural_trainer import (
    EMAClassCenters,
    TheoryNeuralTrainingConfig,
    auxiliary_scale,
    run_theory_neural_epoch,
    train_theory_neural_classifier,
)


def _hard(adjacency, ids, communities, time):
    return HardGraphWindow(
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        communities=torch.tensor(communities, dtype=torch.long),
        node_names=tuple(ids),
        node_ids=tuple(ids),
        time_start=float(time),
        edge_presence_threshold=0.0,
    )


def _features():
    windows = (
        _hard(
            [[0.0, 0.5, -0.3], [0.5, 0.0, 0.2], [-0.3, 0.2, 0.0]],
            ("a", "b", "c"),
            (0, 0, 1),
            0.0,
        ),
        _hard(
            [[0.0, -0.4, 0.6], [-0.4, 0.0, 0.1], [0.6, 0.1, 0.0]],
            ("c", "a", "d"),
            (1, 0, 1),
            1.0,
        ),
    )
    config = SGWCoreConfig(gw_max_iter=3, gw_sinkhorn_iter=5)
    return TheoryNeuralFeatureBuilder(config).build(windows)


def _sample(features, key="sample", label=1):
    windows = tuple(
        TheoryNeuralWindowInput(
            window.node_features,
            window.adjacency,
            window.edge_features,
            window.spectral_quantiles,
        )
        if window is not None
        else None
        for window in features.windows
    )
    return TheoryNeuralSampleInput(
        sample_key=key,
        label=label,
        windows=windows,
        window_mask=features.window_mask,
        transition_targets=features.transition_features,
        transition_mask=features.transition_mask,
    )


class TheoryGuidedNeuralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = _features()

    def setUp(self):
        self._torch_rng_state = torch.random.get_rng_state()

    def tearDown(self):
        torch.random.set_rng_state(self._torch_rng_state)

    def test_edge_schema_preserves_sign_delta_and_community(self):
        first, second = self.features.windows
        self.assertEqual(tuple(first.edge_features.shape), (3, 3, 6))
        self.assertAlmostEqual(float(first.edge_features[0, 2, 0]), -0.3, places=6)
        self.assertAlmostEqual(float(first.edge_features[0, 2, 1]), 0.3, places=6)
        self.assertEqual(float(first.edge_features[0, 1, 5]), 1.0)
        # second node 0 is stable id c and node 1 is stable id a; previous
        # c--a was -0.3 and current is -0.4.
        self.assertAlmostEqual(float(second.edge_features[0, 1, 2]), -0.1, places=6)
        self.assertEqual(float(second.edge_features[0, 2, 4]), 0.0)

    def test_artifact_round_trip(self):
        windows = tuple(
            TheoryNeuralWindowRecord(
                window.node_features,
                window.adjacency,
                window.edge_features,
                window.spectral_quantiles,
                window.communities,
                window.node_ids,
                window.time_start,
            )
            for window in self.features.windows
        )
        record = TheoryNeuralRecord(
            sample_key="site/id",
            sample_id="id",
            subject_id="id",
            site="site",
            label=1,
            split="train",
            windows=windows,
            window_mask=self.features.window_mask,
            transition_features=self.features.transition_features,
            transition_mask=self.features.transition_mask,
            gw_solver_converged=self.features.gw_solver_converged,
            protocol_sha256="p",
            selector_checkpoint_sha256="s",
            selection_mode="learned",
            selection_seed=42,
            feature_schema_sha256="f",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pt"
            save_theory_neural_record(record, path)
            loaded = load_theory_neural_record(path)
        self.assertEqual(loaded.sample_key, record.sample_key)
        self.assertTrue(torch.equal(loaded.transition_mask, record.transition_mask))

    def test_all_variants_forward_and_n3_auxiliary_gradients(self):
        batch = TheoryNeuralBatch((_sample(self.features),))
        for variant in THEORY_NEURAL_VARIANTS:
            model = TheoryGuidedNeuralClassifier(
                TheoryNeuralConfig(variant=variant, dropout=0.0)
            )
            output = model(batch)
            self.assertEqual(tuple(output.logits.shape), (1, 2))
            self.assertTrue(bool(torch.isfinite(output.logits).all()))
            if variant in ("N3_theory_reconstruction", "N4_ema_center"):
                sample = output.samples[0]
                self.assertEqual(tuple(sample.q_predictions.shape), (2, 16))
                self.assertEqual(tuple(sample.gamma_predictions.shape), (1, 18))
                loss = (
                    output.logits.square().mean()
                    + sample.q_predictions.square().mean()
                    + sample.gamma_predictions.square().mean()
                )
                loss.backward()
                self.assertIsNotNone(model.q_head.weight.grad)
                self.assertIsNotNone(model.gamma_head.weight.grad)

    def test_film_zero_initialization_is_exactly_n1(self):
        torch.manual_seed(9)
        n1 = TheoryGuidedNeuralClassifier(
            TheoryNeuralConfig(variant="N1_edge_aware", dropout=0.0)
        )
        n2 = TheoryGuidedNeuralClassifier(
            TheoryNeuralConfig(variant="N2_spectral_film", dropout=0.0)
        )
        n2.load_state_dict(n1.state_dict())
        batch = TheoryNeuralBatch((_sample(self.features),))
        n1.eval()
        n2.eval()
        self.assertTrue(torch.equal(n1(batch).logits, n2(batch).logits))

    def test_consistent_node_permutation_preserves_logits(self):
        sample = _sample(self.features)
        permutations = (torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]))
        windows = []
        for window, order in zip(sample.windows, permutations):
            windows.append(
                TheoryNeuralWindowInput(
                    window.node_features.index_select(0, order),
                    window.adjacency.index_select(0, order).index_select(1, order),
                    window.edge_features.index_select(0, order).index_select(1, order),
                    window.spectral_quantiles,
                )
            )
        permuted = TheoryNeuralSampleInput(
            sample.sample_key,
            sample.label,
            tuple(windows),
            sample.window_mask,
            sample.transition_targets,
            sample.transition_mask,
        )
        model = TheoryGuidedNeuralClassifier(
            TheoryNeuralConfig(variant="N2_spectral_film", dropout=0.0)
        ).eval()
        first = model(TheoryNeuralBatch((sample,))).logits
        second = model(TheoryNeuralBatch((permuted,))).logits
        self.assertTrue(torch.allclose(first, second, atol=1.0e-6, rtol=0.0))

    def test_invalid_window_is_ignored(self):
        sample = _sample(self.features)
        padded = TheoryNeuralSampleInput(
            sample.sample_key,
            sample.label,
            (sample.windows[0], None, sample.windows[1]),
            torch.tensor([True, False, True]),
            torch.zeros((2, 18)),
            torch.tensor([False, False]),
        )
        model = TheoryGuidedNeuralClassifier(
            TheoryNeuralConfig(variant="N1_edge_aware", dropout=0.0)
        ).eval()
        original = model(TheoryNeuralBatch((sample,))).logits
        actual = model(TheoryNeuralBatch((padded,))).logits
        self.assertTrue(torch.allclose(original, actual, atol=1.0e-6, rtol=0.0))

    def test_edge_sign_routes_messages_and_edge_features_matter(self):
        torch.manual_seed(12)
        layer = EdgeAwareSignedLayer(4, 6, 0.0).eval()
        states = torch.randn(3, 4)
        positive = torch.tensor(
            [[0.0, 0.5, 0.2], [0.5, 0.0, 0.4], [0.2, 0.4, 0.0]]
        )
        edge_features = torch.zeros(3, 3, 6)
        edge_features[..., 0] = positive
        edge_features[..., 1] = positive.abs()
        first, first_norms = layer(states, positive, edge_features)
        negative = -positive
        negative_features = edge_features.clone()
        negative_features[..., 0] = negative
        second, second_norms = layer(states, negative, negative_features)
        self.assertGreater(float(first_norms[0]), 0.0)
        self.assertEqual(float(first_norms[1]), 0.0)
        self.assertEqual(float(second_norms[0]), 0.0)
        self.assertGreater(float(second_norms[1]), 0.0)
        changed = edge_features.clone()
        changed[..., 2] = 0.75
        third, _ = layer(states, positive, changed)
        self.assertFalse(torch.allclose(first, third))

    def test_class_weighting_auxiliary_schedule_and_ema_centers(self):
        logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
        labels = torch.tensor([0, 1])
        weights = torch.tensor([0.5, 2.0])
        actual = balanced_classification_loss(logits, labels, weights)
        raw = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        self.assertTrue(torch.allclose(actual, (raw * weights[labels]).mean()))
        config = TheoryNeuralTrainingConfig(
            epochs=1, auxiliary_warmup_epochs=2, auxiliary_ramp_epochs=4
        )
        self.assertEqual(auxiliary_scale(2, config), 0.0)
        self.assertEqual(auxiliary_scale(4, config), 0.5)
        self.assertEqual(auxiliary_scale(7, config), 1.0)
        centers = EMAClassCenters.create(3, torch.device("cpu"), 0.9)
        representations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        centers.update(representations, labels)
        self.assertTrue(bool(centers.initialized.all()))
        self.assertAlmostEqual(float(centers.loss(representations, labels)), 0.0)

    def test_training_saves_frozen_validation_threshold(self):
        samples = tuple(
            _sample(self.features, "sample{}".format(index), index % 2)
            for index in range(4)
        )

        class Dataset(object):
            sample_keys = tuple(sample.sample_key for sample in samples)
            sites = ("site",) * 4
            labels = tuple(sample.label for sample in samples)

        class Loader(object):
            dataset = Dataset()

            def __len__(self):
                return 2

            def __iter__(self):
                yield TheoryNeuralBatch(samples[:2])
                yield TheoryNeuralBatch(samples[2:])

        model = TheoryGuidedNeuralClassifier(
            TheoryNeuralConfig(variant="N4_ema_center", dropout=0.0)
        )
        config = TheoryNeuralTrainingConfig(
            epochs=1,
            gradient_accumulation_steps=2,
            early_stopping_patience=0,
            selection_metric="roc_auc",
            auxiliary_warmup_epochs=0,
            auxiliary_ramp_epochs=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = train_theory_neural_classifier(
                model,
                Loader(),
                Loader(),
                Dataset.labels,
                torch.device("cpu"),
                config,
                Path(directory) / "run",
                {"fixture": "unit"},
            )
            checkpoint = torch.load(
                str(Path(directory) / "run" / "best_checkpoint.pt"),
                map_location="cpu",
            )
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(checkpoint["threshold_fit_split"], "validation")
        self.assertIn("balanced_accuracy", checkpoint["validation_thresholds"])
        self.assertIsNotNone(checkpoint["class_centers"])

    def test_frozen_diagnostics_report_rank_fisher_and_probes(self):
        train = {
            "labels": [0, 0, 1, 1],
            "sites": ["a", "b", "a", "b"],
            "representations": [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]],
            "node_summaries": [[0.0], [0.1], [1.0], [1.1]],
            "edge_summaries": [[0.0], [0.2], [0.8], [1.0]],
        }
        validation = {
            "labels": [0, 0, 1, 1],
            "sites": ["a", "b", "a", "b"],
            "representations": [[0.0, 0.1], [0.2, 0.0], [0.9, 1.0], [1.0, 0.9]],
            "node_summaries": [[0.0], [0.2], [0.9], [1.0]],
            "edge_summaries": [[0.1], [0.2], [0.9], [1.0]],
        }
        mechanism = (
            "q_errors", "gamma_errors", "film_gamma_norms", "film_beta_norms",
            "positive_message_norms", "negative_message_norms",
            "signed_difference_norms",
        )
        for payload in (train, validation):
            payload.update({name: [] for name in mechanism})
        actual = build_theory_neural_diagnostics(train, validation)
        self.assertFalse(actual["uses_test"])
        self.assertGreater(actual["representation"]["fisher_ratio"], 0.0)
        self.assertGreater(actual["label_probes"]["representations"], 0.5)


if __name__ == "__main__":
    unittest.main()
