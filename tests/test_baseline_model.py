from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.baseline_collate import BaselineBatch  # noqa: E402
from keysubgraph.models.baseline_classifier import (  # noqa: E402
    BaselineModelConfig,
    SignedSequenceBaseline,
)
from keysubgraph.models.baseline_subgraph_encoder import (  # noqa: E402
    SignedSubgraphEncoder,
    WindowMeanPooling,
)
from keysubgraph.models.signed_message_passing import (  # noqa: E402
    SignedMessagePassingLayer,
)


def _signed_graph():
    adjacency = torch.zeros(1, 3, 3)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 0.6
    adjacency[0, 1, 2] = adjacency[0, 2, 1] = -0.4
    features = torch.arange(36, dtype=torch.float32).reshape(1, 3, 12) / 10.0
    mask = torch.ones(1, 3, dtype=torch.bool)
    return features, adjacency, mask


class SignedMessagePassingTest(unittest.TestCase):
    def test_missing_sign_neighbors_produce_zero_messages(self):
        torch.manual_seed(1)
        layer = SignedMessagePassingLayer(12, 8, dropout=0.0)
        features, adjacency, mask = _signed_graph()

        positive, negative = layer.signed_messages(features, adjacency, mask)

        self.assertEqual(float(positive[0, 2].abs().sum()), 0.0)
        self.assertEqual(float(negative[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(positive[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(negative[0, 2].abs().sum()), 0.0)

    def test_changing_negative_edge_to_positive_changes_output(self):
        torch.manual_seed(2)
        layer = SignedMessagePassingLayer(12, 8, dropout=0.0)
        layer.eval()
        features, adjacency, mask = _signed_graph()

        signed_output = layer(features, adjacency, mask)
        changed = adjacency.abs()
        positive_output = layer(features, changed, mask)

        self.assertFalse(torch.allclose(signed_output, positive_output))

    def test_gradients_reach_both_sign_branches(self):
        torch.manual_seed(3)
        layer = SignedMessagePassingLayer(12, 8, dropout=0.0)
        features, adjacency, mask = _signed_graph()

        layer(features, adjacency, mask).sum().backward()

        self.assertGreater(
            float(layer.positive_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(layer.negative_projection.weight.grad.abs().sum()), 0.0
        )


class SignedSubgraphEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.encoder = SignedSubgraphEncoder(
            node_feature_dim=12, hidden_dim=16, layers=2, dropout=0.0
        )
        self.encoder.eval()

    def test_padding_does_not_change_subgraph_embedding(self):
        features, adjacency, mask = _signed_graph()
        original = self.encoder(features, adjacency, mask)
        padded_features = torch.zeros(1, 5, 12)
        padded_adjacency = torch.zeros(1, 5, 5)
        padded_mask = torch.zeros(1, 5, dtype=torch.bool)
        padded_features[:, :3] = features
        padded_adjacency[:, :3, :3] = adjacency
        padded_mask[:, :3] = True

        padded = self.encoder(padded_features, padded_adjacency, padded_mask)

        self.assertTrue(torch.allclose(original, padded, atol=1e-6, rtol=0.0))

    def test_consistent_node_permutation_does_not_change_embedding(self):
        features, adjacency, mask = _signed_graph()
        original = self.encoder(features, adjacency, mask)
        permutation = torch.tensor([2, 0, 1])
        permuted_features = features.index_select(1, permutation)
        permuted_adjacency = adjacency.index_select(1, permutation).index_select(
            2, permutation
        )

        permuted = self.encoder(permuted_features, permuted_adjacency, mask)

        self.assertTrue(torch.allclose(original, permuted, atol=1e-6, rtol=0.0))

    def test_window_pooling_uses_only_mapped_subgraphs(self):
        pooling = WindowMeanPooling()
        embeddings = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]])
        mapping = torch.tensor([0, 0, 1], dtype=torch.long)

        windows = pooling(embeddings, mapping, 2, torch.tensor([2, 1]))

        self.assertTrue(torch.equal(windows, torch.tensor([[2.0, 4.0], [10.0, 20.0]])))


def _sequence_batch(extra_time_padding=False):
    first_features, first_adjacency, first_mask = _signed_graph()
    features = torch.cat((first_features, first_features + 0.2, first_features + 0.4), dim=0)
    adjacency = torch.cat((first_adjacency, first_adjacency, first_adjacency), dim=0)
    node_mask = torch.cat((first_mask, first_mask, first_mask), dim=0)
    time_width = 3 if extra_time_padding else 2
    window_index = torch.full((2, time_width), -1, dtype=torch.long)
    window_index[0, 0] = 0
    window_index[0, 1] = 1
    window_index[1, 0] = 2
    time_mask = window_index >= 0
    return BaselineBatch(
        node_features=features,
        adjacency=adjacency,
        edge_mask=adjacency.abs() > 0.0,
        node_mask=node_mask,
        subgraph_to_window=torch.tensor([0, 1, 2]),
        window_to_sample=torch.tensor([0, 0, 1]),
        window_time_index=torch.tensor([0, 1, 0]),
        window_subgraph_count=torch.tensor([1, 1, 1]),
        window_index=window_index,
        time_mask=time_mask,
        labels=torch.tensor([0, 1]),
        sample_keys=("SITE/a", "SITE/b"),
        sample_ids=("a", "b"),
        subject_ids=("a", "b"),
        sites=("SITE", "SITE"),
    )


class SignedSequenceBaselineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = SignedSequenceBaseline(
            BaselineModelConfig(
                node_hidden_dim=16,
                fusion_dim=24,
                gru_hidden_dim=20,
                classifier_hidden_dim=10,
                signed_gnn_dropout=0.0,
                classifier_dropout=0.0,
            )
        )

    def test_forward_shapes_and_invalid_time_state(self):
        self.model.eval()
        output = self.model(_sequence_batch())

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.subgraph_embeddings.shape), (3, 32))
        self.assertEqual(tuple(output.window_embeddings.shape), (3, 32))
        self.assertEqual(tuple(output.padded_window_embeddings.shape), (2, 2, 24))
        self.assertEqual(tuple(output.hidden_states.shape), (2, 2, 20))
        self.assertTrue(
            torch.allclose(
                output.hidden_states[1, 0], output.hidden_states[1, 1], atol=0.0, rtol=0.0
            )
        )

    def test_invalid_time_padding_does_not_change_logits(self):
        self.model.eval()
        original = self.model(_sequence_batch()).logits
        padded = self.model(_sequence_batch(extra_time_padding=True)).logits

        self.assertTrue(torch.allclose(original, padded, atol=1e-6, rtol=0.0))

    def test_classification_loss_backpropagates_through_all_modules(self):
        output = self.model(_sequence_batch())
        loss = torch.nn.functional.cross_entropy(
            output.logits, torch.tensor([0, 1])
        )

        loss.backward()

        modules = (
            self.model.subgraph_encoder.layers[0].positive_projection,
            self.model.subgraph_encoder.layers[0].negative_projection,
            self.model.input_projection,
            self.model.gru_cell,
            self.model.classifier[0],
            self.model.classifier[-1],
        )
        for module in modules:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(all(item is not None for item in gradients))
            self.assertTrue(all(bool(torch.isfinite(item).all()) for item in gradients))


if __name__ == "__main__":
    unittest.main()
