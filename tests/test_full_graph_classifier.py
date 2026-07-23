from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch, GraphSequenceSample
from keysubgraph.models import (
    FullGraphClassifierConfig,
    FullGraphSequenceClassifier,
    SymmetricSignedEdgeGatedLayer,
)


def _sample(key, label, windows):
    adjacency, masks, names, communities = [], [], [], []
    for index in range(windows):
        scale = 1.0 + 0.1 * index
        graph = torch.tensor(
            [
                [0.0, 0.7 * scale, -0.3],
                [0.7 * scale, 0.0, 0.2],
                [-0.3, 0.2, 0.0],
            ],
            dtype=torch.float32,
        )
        mask = graph.abs() > 0.0
        mask.fill_diagonal_(False)
        adjacency.append(graph)
        masks.append(mask)
        names.append(("roi-a", "roi-b", "roi-c"))
        communities.append(torch.tensor([0, 0, 1], dtype=torch.long))
    return GraphSequenceSample(
        sample_key=key,
        sample_id=key,
        site="site",
        subject_id=key,
        session_id="1",
        label=label,
        split="train",
        relative_path=key + ".pt",
        adjacency=tuple(adjacency),
        edge_mask=tuple(masks),
        node_names=tuple(names),
        communities=tuple(communities),
        window_starts=torch.arange(windows, dtype=torch.float32),
        source_global_threshold=0.0,
        repetition_time=1.0,
        edge_presence_threshold=0.0,
    )


def _config(encoder_type):
    return FullGraphClassifierConfig(
        encoder_type=encoder_type,
        signed_gnn_layers=2,
        classifier_hidden_dims=(16,),
        baseline_dropout=0.0,
        gated_gnn_dropout=0.0,
        classifier_dropout=0.0,
    )


class FullGraphClassifierTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.left = _sample("left", 0, 2)
        self.right = _sample("right", 1, 3)
        self.batch = GraphSequenceBatch((self.left, self.right))

    def test_controlled_baseline_bypasses_extractor_and_preserves_lengths(self):
        model = FullGraphSequenceClassifier(_config("signed_gnn_tcn"))
        output = model(self.batch)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.representation.shape), (2, 192))
        self.assertEqual(output.sequence_lengths.tolist(), [2, 3])
        self.assertIsNone(output.prototype_attention)
        names = tuple(name for name, _ in model.named_parameters())
        self.assertFalse(any("score" in name for name in names))

    def test_proto_encoder_outputs_normalized_usage_and_all_modules_receive_gradients(self):
        model = FullGraphSequenceClassifier(_config("sgg_bigru_proto"))
        normalized_windows = []
        handle = model.encoder.graph_pooling_normalization.register_forward_hook(
            lambda module, inputs, output: normalized_windows.append(
                output.detach()
            )
        )
        output = model(self.batch)
        handle.remove()
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.prototype_attention.shape), (2, 16))
        self.assertTrue(
            torch.allclose(
                output.prototype_attention.sum(dim=-1),
                torch.ones(2),
                atol=1.0e-6,
            )
        )
        self.assertTrue(normalized_windows)
        for window in normalized_windows:
            self.assertAlmostEqual(float(window.mean()), 0.0, places=5)
            self.assertAlmostEqual(
                float(window.var(unbiased=False)), 1.0, delta=5.0e-4
            )
        torch.nn.functional.cross_entropy(output.logits, self.batch.labels).backward()
        modules = (
            model.encoder.graph_encoder,
            model.encoder.temporal_encoder,
            model.encoder.prototype_codebook,
            model.classifier,
        )
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_symmetric_gate_cannot_create_directed_weights(self):
        layer = SymmetricSignedEdgeGatedLayer(4, 3, dropout=0.0, epsilon=1.0e-8)
        hidden = torch.randn(3, 4)
        adjacency = torch.tensor(
            [[0.0, 0.7, -0.2], [0.7, 0.0, 0.4], [-0.2, 0.4, 0.0]]
        )
        mask = adjacency.abs() > 0.0
        edge_embeddings = torch.randn(3, 3, 3)
        edge_embeddings = 0.5 * (
            edge_embeddings + edge_embeddings.transpose(0, 1)
        )
        _, gates = layer(hidden, adjacency, mask, edge_embeddings)
        self.assertTrue(torch.allclose(gates, gates.transpose(0, 1)))
        self.assertTrue(torch.all(gates[mask] > 0.0))

    def test_packed_bigru_ignores_other_samples_padding(self):
        model = FullGraphSequenceClassifier(_config("sgg_bigru_proto"))
        model.eval()
        with torch.no_grad():
            alone = model(GraphSequenceBatch((self.left,))).logits[0]
            batched = model(self.batch).logits[0]
        self.assertTrue(torch.allclose(alone, batched, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
