from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.theory_neural_features import TheoryNeuralFeatureBuilder
from keysubgraph.models.neuralized_sv import (
    NEURALIZED_SV_VARIANTS,
    NeuralizedSVClassifier,
    NeuralizedSVConfig,
)
from keysubgraph.models.theory_guided_neural import (
    TheoryNeuralBatch,
    TheoryNeuralSampleInput,
    TheoryNeuralWindowInput,
)
from keysubgraph.theory.sgw_core_features import SGWCoreConfig
from keysubgraph.training.theory_guided_neural_trainer import (
    TheoryNeuralTrainingConfig,
    train_theory_neural_classifier,
)


def _window(adjacency, node_ids, communities, time_start):
    return HardGraphWindow(
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        communities=torch.tensor(communities, dtype=torch.long),
        node_names=tuple(node_ids),
        node_ids=tuple(node_ids),
        time_start=float(time_start),
        edge_presence_threshold=0.0,
    )


def _sample():
    windows = (
        _window(
            [[0.0, 0.7, -0.2], [0.7, 0.0, 0.3], [-0.2, 0.3, 0.0]],
            ("a", "b", "c"),
            (0, 0, 1),
            0.0,
        ),
        _window(
            [[0.0, -0.5, 0.4], [-0.5, 0.0, 0.2], [0.4, 0.2, 0.0]],
            ("c", "a", "d"),
            (1, 0, 1),
            1.0,
        ),
        _window(
            [[0.0, 0.6, -0.1], [0.6, 0.0, -0.4], [-0.1, -0.4, 0.0]],
            ("d", "c", "a"),
            (1, 1, 0),
            2.0,
        ),
    )
    features = TheoryNeuralFeatureBuilder(
        SGWCoreConfig(gw_max_iter=3, gw_sinkhorn_iter=5)
    ).build(windows)
    inputs = tuple(
        TheoryNeuralWindowInput(
            item.node_features,
            item.adjacency,
            item.edge_features,
            item.spectral_quantiles,
        )
        for item in features.windows
    )
    return TheoryNeuralSampleInput(
        sample_key="site/sample",
        label=1,
        windows=inputs,
        window_mask=features.window_mask,
        transition_targets=features.transition_features,
        transition_mask=features.transition_mask,
    )


class NeuralizedSVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = _sample()

    def setUp(self):
        self.random_state = torch.random.get_rng_state()

    def tearDown(self):
        torch.random.set_rng_state(self.random_state)

    def test_all_variants_forward_and_theory_decoders(self):
        batch = TheoryNeuralBatch((self.sample,))
        for variant in NEURALIZED_SV_VARIANTS:
            model = NeuralizedSVClassifier(
                NeuralizedSVConfig(variant=variant, dropout=0.0)
            )
            output = model(batch)
            self.assertEqual(tuple(output.logits.shape), (1, 2))
            self.assertEqual(tuple(output.representations.shape), (1, 64))
            self.assertTrue(bool(torch.isfinite(output.logits).all()))
            item = output.samples[0]
            if model.config.uses_static:
                self.assertEqual(tuple(item.q_predictions.shape), (3, 16))
            else:
                self.assertIsNone(item.q_predictions)
            if model.config.uses_evolution:
                self.assertEqual(tuple(item.gamma_predictions.shape), (2, 18))
            else:
                self.assertIsNone(item.gamma_predictions)

    def test_nsv_is_exactly_static_at_zero_residual_initialization(self):
        torch.manual_seed(77)
        static = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NS_static_spectral", dropout=0.0)
        )
        joint = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NSV_safe_residual", dropout=0.0)
        )
        joint.load_state_dict(static.state_dict())
        batch = TheoryNeuralBatch((self.sample,))
        static.eval()
        joint.eval()
        self.assertTrue(
            torch.equal(static(batch).representations, joint(batch).representations)
        )
        self.assertTrue(torch.equal(static(batch).logits, joint(batch).logits))

    def test_classification_and_auxiliary_losses_reach_graph_encoder(self):
        model = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NSV_safe_residual", dropout=0.0)
        )
        output = model(TheoryNeuralBatch((self.sample,)))
        item = output.samples[0]
        loss = (
            output.logits.square().mean()
            + item.q_predictions.square().mean()
            + item.gamma_predictions.square().mean()
        )
        loss.backward()
        self.assertIsNotNone(model.graph_encoder.node_projection[0].weight.grad)
        self.assertIsNotNone(
            model.graph_encoder.layers[0].positive[0].weight.grad
        )
        self.assertIsNotNone(
            model.graph_encoder.layers[0].negative[0].weight.grad
        )
        self.assertIsNotNone(model.q_decoder.weight.grad)
        self.assertIsNotNone(model.gamma_decoder.weight.grad)

    def test_signed_edges_change_output(self):
        model = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NS_static_spectral", dropout=0.0)
        ).eval()
        batch = TheoryNeuralBatch((self.sample,))
        first = model(batch).logits
        windows = []
        for window in self.sample.windows:
            adjacency = window.adjacency.abs()
            edge_features = window.edge_features.clone()
            edge_features[..., 0] = adjacency
            windows.append(
                TheoryNeuralWindowInput(
                    window.node_features,
                    adjacency,
                    edge_features,
                    window.spectral_quantiles,
                )
            )
        changed = TheoryNeuralSampleInput(
            self.sample.sample_key,
            self.sample.label,
            tuple(windows),
            self.sample.window_mask,
            self.sample.transition_targets,
            self.sample.transition_mask,
        )
        second = model(TheoryNeuralBatch((changed,))).logits
        self.assertFalse(torch.allclose(first, second))

    def test_consistent_node_permutation_is_invariant(self):
        permutations = (
            torch.tensor([2, 0, 1]),
            torch.tensor([1, 2, 0]),
            torch.tensor([2, 1, 0]),
        )
        windows = []
        for window, order in zip(self.sample.windows, permutations):
            windows.append(
                TheoryNeuralWindowInput(
                    window.node_features.index_select(0, order),
                    window.adjacency.index_select(0, order).index_select(1, order),
                    window.edge_features.index_select(0, order).index_select(1, order),
                    window.spectral_quantiles,
                )
            )
        permuted = TheoryNeuralSampleInput(
            self.sample.sample_key,
            self.sample.label,
            tuple(windows),
            self.sample.window_mask,
            self.sample.transition_targets,
            self.sample.transition_mask,
        )
        model = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NSV_safe_residual", dropout=0.0)
        ).eval()
        first = model(TheoryNeuralBatch((self.sample,))).logits
        second = model(TheoryNeuralBatch((permuted,))).logits
        self.assertTrue(torch.allclose(first, second, atol=1.0e-6, rtol=0.0))

    def test_training_checkpoint_uses_new_model_identity(self):
        samples = []
        for index in range(4):
            samples.append(
                TheoryNeuralSampleInput(
                    sample_key="sample{}".format(index),
                    label=index % 2,
                    windows=self.sample.windows,
                    window_mask=self.sample.window_mask,
                    transition_targets=self.sample.transition_targets,
                    transition_mask=self.sample.transition_mask,
                )
            )

        class Dataset(object):
            sample_keys = tuple(item.sample_key for item in samples)
            sites = ("site",) * 4
            labels = tuple(item.label for item in samples)

        class Loader(object):
            dataset = Dataset()

            def __len__(self):
                return 2

            def __iter__(self):
                yield TheoryNeuralBatch(tuple(samples[:2]))
                yield TheoryNeuralBatch(tuple(samples[2:]))

        model = NeuralizedSVClassifier(
            NeuralizedSVConfig(variant="NSV_safe_residual", dropout=0.0)
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
            train_theory_neural_classifier(
                model,
                Loader(),
                Loader(),
                Dataset.labels,
                torch.device("cpu"),
                config,
                Path(directory) / "run",
                {"fixture": "neuralized_sv"},
            )
            checkpoint = torch.load(
                str(Path(directory) / "run" / "best_checkpoint.pt"),
                map_location="cpu",
            )
        self.assertEqual(checkpoint["model_name"], model.model_name)
        self.assertEqual(checkpoint["model_config"]["variant"], "NSV_safe_residual")



if __name__ == "__main__":
    unittest.main()
