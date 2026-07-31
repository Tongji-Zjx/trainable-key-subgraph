from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from keysubgraph.data.graph_dataset import GraphSequenceBatch, GraphSequenceSample
from keysubgraph.features.structured_short_term_features import (
    COMMUNITY_SUMMARY_NAMES,
    NODE_FEATURE_NAMES,
    StructuredShortTermFeatureBuilder,
    StructuredShortTermStandardizer,
    fit_structured_short_term_standardizer,
)
from keysubgraph.models.structured_short_term import (
    PAPER_ALIGNED_VARIANT,
    StructuredShortTermClassifier,
    StructuredShortTermConfig,
)


def _graph_sample(key, label, window_count, node_count=4, split="train"):
    base = torch.tensor(
        [
            [0.0, 0.5, -0.3, 0.2],
            [0.5, 0.0, 0.1, -0.4],
            [-0.3, 0.1, 0.0, 0.6],
            [0.2, -0.4, 0.6, 0.0],
        ],
        dtype=torch.float32,
    )[:node_count, :node_count]
    adjacency = []
    masks = []
    names = []
    communities = []
    for index in range(window_count):
        graph = base * (1.0 + 0.1 * index)
        adjacency.append(graph)
        mask = graph.abs() > 0.0
        mask.fill_diagonal_(False)
        masks.append(mask)
        names.append(tuple("roi-{}".format(value) for value in range(node_count)))
        labels = torch.tensor([7, 7, 3, 3], dtype=torch.long)[:node_count]
        communities.append(labels)
    return GraphSequenceSample(
        sample_key=key,
        sample_id=key,
        site="site",
        subject_id=key,
        session_id="1",
        label=label,
        split=split,
        relative_path=key + ".pt",
        adjacency=tuple(adjacency),
        edge_mask=tuple(masks),
        node_names=tuple(names),
        communities=tuple(communities),
        window_starts=torch.arange(window_count, dtype=torch.float32) * 5.0,
        source_global_threshold=0.0,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


def _identity_standardizer():
    return StructuredShortTermStandardizer(
        node_mean=(0.0,) * len(NODE_FEATURE_NAMES),
        node_std=(1.0,) * len(NODE_FEATURE_NAMES),
        community_mean=(0.0,) * len(COMMUNITY_SUMMARY_NAMES),
        community_std=(1.0,) * len(COMMUNITY_SUMMARY_NAMES),
        train_sample_count=2,
        train_window_count=5,
        train_node_count=20,
        protocol_sha256="a" * 64,
        edge_presence_threshold=0.0,
    )


def _permuted(sample, permutation, relabel=False):
    mapping = {7: 91, 3: 12}
    communities = []
    for labels in sample.communities:
        labels = labels.index_select(0, permutation)
        if relabel:
            labels = torch.tensor(
                [mapping[int(value)] for value in labels.tolist()],
                dtype=torch.long,
            )
        communities.append(labels)
    return GraphSequenceSample(
        sample_key=sample.sample_key,
        sample_id=sample.sample_id,
        site=sample.site,
        subject_id=sample.subject_id,
        session_id=sample.session_id,
        label=sample.label,
        split=sample.split,
        relative_path=sample.relative_path,
        adjacency=tuple(
            item.index_select(0, permutation).index_select(1, permutation)
            for item in sample.adjacency
        ),
        edge_mask=tuple(
            item.index_select(0, permutation).index_select(1, permutation)
            for item in sample.edge_mask
        ),
        node_names=tuple(
            tuple(names[int(index)] for index in permutation.tolist())
            for names in sample.node_names
        ),
        communities=tuple(communities),
        window_starts=sample.window_starts.clone(),
        source_global_threshold=sample.source_global_threshold,
        repetition_time=sample.repetition_time,
        edge_presence_threshold=sample.edge_presence_threshold,
    )


class StructuredShortTermFeatureTest(unittest.TestCase):
    def test_signed_strengths_and_community_structure_are_exact(self):
        sample = _graph_sample("signed", 0, 1)
        feature = StructuredShortTermFeatureBuilder().build_sample(sample)[0]
        node_zero = feature.node_features[0]
        self.assertTrue(
            torch.allclose(
                node_zero[:5],
                torch.tensor([1.0, 0.7, 0.3, 0.7, 0.3]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(node_zero[5:8], torch.zeros(3), atol=1.0e-7)
        )
        self.assertTrue(
            torch.allclose(
                node_zero[8:],
                torch.tensor([0.5, 0.5, 0.0, 0.1, 0.15, 1.0, 1.0]),
                atol=1.0e-6,
            )
        )

    def test_community_relabel_and_node_permutation_are_invariant(self):
        sample = _graph_sample("permutation", 1, 2)
        permutation = torch.tensor([2, 0, 3, 1])
        changed = _permuted(sample, permutation, relabel=True)
        builder = StructuredShortTermFeatureBuilder()
        original = builder.build_sample(sample)
        reordered = builder.build_sample(changed)
        inverse = torch.argsort(permutation)
        for left, right in zip(original, reordered):
            self.assertTrue(
                torch.allclose(
                    left.node_features,
                    right.node_features.index_select(0, inverse),
                    atol=1.0e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    left.community_summary,
                    right.community_summary,
                    atol=1.0e-6,
                )
            )

    def test_standardizer_is_train_only_and_round_trips(self):
        train = (_graph_sample("a", 0, 2), _graph_sample("b", 1, 3))
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            protocol.write_text('{"frozen": true}\n', encoding="utf-8")
            scaler = fit_structured_short_term_standardizer(
                train,
                protocol,
                edge_presence_threshold=0.0,
            )
            path = Path(directory) / "standardizer.json"
            scaler.save(path)
            loaded = StructuredShortTermStandardizer.load(path)
            self.assertEqual(loaded.to_dict(), scaler.to_dict())
            self.assertEqual(loaded.train_sample_count, 2)
            self.assertEqual(loaded.train_window_count, 5)
            with self.assertRaisesRegex(ValueError, "only consume train"):
                fit_structured_short_term_standardizer(
                    (_graph_sample("validation", 0, 1, split="validation"),),
                    protocol,
                    edge_presence_threshold=0.0,
                )


class StructuredShortTermModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        self.left = _graph_sample("left", 0, 2)
        self.right = _graph_sample("right", 1, 4, node_count=3)
        self.config = StructuredShortTermConfig(
            hidden_dim=16,
            node_ffn_dim=24,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=32,
            memory_slots=5,
            statistics_embedding_dim=8,
            classifier_hidden_dims=(12, 6),
            dropout=0.0,
        )

    def test_complete_variable_length_forward_and_no_forbidden_embedding(self):
        model = StructuredShortTermClassifier(
            self.config,
            _identity_standardizer(),
        ).eval()
        batch = GraphSequenceBatch((self.left, self.right))
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.time_mask.shape), (2, 4))
        self.assertEqual(output.time_mask.sum(dim=1).tolist(), [2, 4])
        self.assertEqual(tuple(output.memory_attention.shape), (2, 5))
        self.assertTrue(
            torch.allclose(
                output.memory_attention.sum(dim=1),
                torch.ones(2),
                atol=1.0e-6,
            )
        )
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in model.modules()))
        self.assertFalse(output.diagnostics["uses_coordinates"])
        self.assertFalse(output.diagnostics["uses_community_embedding"])

    def test_list_batch_does_not_change_a_sample_and_memory_does_not_mutate(self):
        model = StructuredShortTermClassifier(
            self.config,
            _identity_standardizer(),
        ).eval()
        memory_before = model.memory_readout.memory.detach().clone()
        with torch.no_grad():
            alone = model(GraphSequenceBatch((self.left,))).logits[0]
            together = model(
                GraphSequenceBatch((self.left, self.right))
            ).logits[0]
        self.assertTrue(torch.allclose(alone, together, atol=1.0e-6))
        self.assertTrue(
            torch.equal(memory_before, model.memory_readout.memory.detach())
        )

    def test_gradients_reach_window_transformer_memory_and_classifier(self):
        model = StructuredShortTermClassifier(
            self.config,
            _identity_standardizer(),
        )
        batch = GraphSequenceBatch((self.left, self.right))
        output = model(batch)
        torch.nn.functional.cross_entropy(output.logits, batch.labels).backward()
        selected = (
            model.window_encoder.input_projection.weight,
            model.temporal_encoder.layers[0].self_attn.in_proj_weight,
            model.memory_readout.memory,
            model.statistics_projection.weight,
            model.classifier[0].weight,
        )
        for parameter in selected:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_consistent_node_permutation_preserves_prediction(self):
        model = StructuredShortTermClassifier(
            self.config,
            _identity_standardizer(),
        ).eval()
        changed = _permuted(
            self.left,
            torch.tensor([2, 0, 3, 1]),
            relabel=True,
        )
        with torch.no_grad():
            original = model(GraphSequenceBatch((self.left,))).logits
            permuted = model(GraphSequenceBatch((changed,))).logits
        self.assertTrue(torch.allclose(original, permuted, atol=1.0e-6))

    def test_paper_aligned_variant_uses_raw_community_and_no_statistics(self):
        config = StructuredShortTermConfig(
            hidden_dim=16,
            node_ffn_dim=24,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=32,
            memory_slots=5,
            statistics_embedding_dim=8,
            classifier_hidden_dims=(12, 6),
            dropout=0.0,
            variant=PAPER_ALIGNED_VARIANT,
            community_vocab_size=128,
            community_embedding_dim=6,
        )
        model = StructuredShortTermClassifier(
            config,
            _identity_standardizer(),
        )
        batch = GraphSequenceBatch((self.left, self.right))
        memory_before = model.memory_readout.memory.detach().clone()
        output = model(batch)
        first = output.window_encodings[0][0]
        self.assertEqual(tuple(first.node_features.shape), (4, 8))
        self.assertTrue(
            torch.allclose(
                first.absolute_degree,
                torch.tensor([1.0, 1.0, 1.0, 1.2]),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                first.delta_absolute_degree,
                torch.zeros_like(first.delta_absolute_degree),
            )
        )
        self.assertEqual(first.community_indices.tolist(), [8, 8, 4, 4])
        self.assertEqual(tuple(output.sequence_statistics.shape), (2, 0))
        self.assertEqual(
            tuple(output.statistics_representation.shape),
            (2, 0),
        )
        self.assertEqual(tuple(output.final_representation.shape), (2, 32))
        self.assertIsNotNone(output.memory_update)
        self.assertTrue(output.diagnostics["uses_community_embedding"])
        self.assertFalse(output.diagnostics["uses_sequence_statistics"])
        loss = torch.nn.functional.cross_entropy(
            output.logits,
            batch.labels,
        )
        loss.backward()
        self.assertGreater(
            float(
                model.window_encoder.community_embedding.weight.grad
                .abs().sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                model.memory_readout.write_projection.weight.grad
                .abs().sum()
            ),
            0.0,
        )
        model.commit_memory_write(output.memory_update)
        self.assertFalse(
            torch.equal(memory_before, model.memory_readout.memory)
        )

    def test_paper_memory_is_read_only_during_evaluation(self):
        config = StructuredShortTermConfig(
            hidden_dim=16,
            node_ffn_dim=24,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=32,
            memory_slots=5,
            statistics_embedding_dim=8,
            classifier_hidden_dims=(12, 6),
            dropout=0.0,
            variant=PAPER_ALIGNED_VARIANT,
            community_vocab_size=128,
            community_embedding_dim=6,
        )
        model = StructuredShortTermClassifier(
            config,
            _identity_standardizer(),
        ).eval()
        before = model.memory_readout.memory.detach().clone()
        with torch.no_grad():
            output = model(GraphSequenceBatch((self.left, self.right)))
        self.assertIsNone(output.memory_update)
        self.assertTrue(torch.equal(before, model.memory_readout.memory))


if __name__ == "__main__":
    unittest.main()
