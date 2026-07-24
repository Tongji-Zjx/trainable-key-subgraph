from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.exact_stse_dataset import (
    ExactSTSEBatch,
    ExactSTSESample,
    _coordinate_sequence,
)
from keysubgraph.data.graph_dataset import GraphSequenceSample
from keysubgraph.models.exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
    ExactSTSEFeatureBuilder,
)
from tests.test_full_graph_classifier import _sample


def _exact_sample(key, label, windows, coordinates=None):
    graph = _sample(key, label, windows)
    if coordinates is None:
        base = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [0.0, 2.0, 1.0],
                [-2.0, 1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        coordinates = tuple(base.clone() for _ in range(windows))
    return ExactSTSESample(graph=graph, coordinates=tuple(coordinates))


def _permuted(sample, permutation):
    graph = sample.graph
    adjacency = tuple(
        item.index_select(0, permutation).index_select(1, permutation)
        for item in graph.adjacency
    )
    masks = tuple(
        item.index_select(0, permutation).index_select(1, permutation)
        for item in graph.edge_mask
    )
    names = tuple(
        tuple(names[int(index)] for index in permutation.tolist())
        for names in graph.node_names
    )
    communities = tuple(
        item.index_select(0, permutation) for item in graph.communities
    )
    permuted_graph = GraphSequenceSample(
        sample_key=graph.sample_key,
        sample_id=graph.sample_id,
        site=graph.site,
        subject_id=graph.subject_id,
        session_id=graph.session_id,
        label=graph.label,
        split=graph.split,
        relative_path=graph.relative_path,
        adjacency=adjacency,
        edge_mask=masks,
        node_names=names,
        communities=communities,
        window_starts=graph.window_starts.clone(),
        source_global_threshold=graph.source_global_threshold,
        repetition_time=graph.repetition_time,
        edge_presence_threshold=graph.edge_presence_threshold,
    )
    return ExactSTSESample(
        graph=permuted_graph,
        coordinates=tuple(
            item.index_select(0, permutation) for item in sample.coordinates
        ),
    )


class ExactSTSEModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        self.left = _exact_sample("left", 0, 2)
        self.right = _exact_sample("right", 1, 3)
        self.batch = ExactSTSEBatch((self.left, self.right))

    def test_input_dimensions_and_coordinate_validation(self):
        self.assertEqual(ExactSTSEConfig(use_coordinates=True).input_dim, 24)
        self.assertEqual(ExactSTSEConfig(use_coordinates=False).input_dim, 18)
        shared = torch.ones(3, 3)
        sequence = _coordinate_sequence(shared, (3, 3))
        self.assertEqual(len(sequence), 2)
        with self.assertRaisesRegex(ValueError, "all zero"):
            _coordinate_sequence(torch.zeros(3, 3), (3,))

    def test_degree_delta_and_signed_neighbor_coordinate_formulas(self):
        current = torch.tensor(
            [
                [0.0, 0.5, -0.25],
                [0.5, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
            ]
        )
        previous = torch.tensor(
            [
                [0.0, 0.25, -0.25],
                [0.25, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
            ]
        )
        coordinates = self.left.coordinates[0]
        features = ExactSTSEFeatureBuilder(
            ExactSTSEConfig(use_coordinates=True)
        ).build(
            current,
            previous,
            ("a", "b", "c"),
            ("a", "b", "c"),
            coordinates,
            torch.tensor([0, 0, 1]),
        )
        self.assertTrue(
            torch.allclose(features.degree, torch.tensor([0.75, 0.5, 0.25]))
        )
        self.assertTrue(
            torch.allclose(
                features.delta_degree, torch.tensor([0.25, 0.25, 0.0])
            )
        )
        expected = (current / features.degree[:, None]).matmul(coordinates)
        self.assertTrue(
            torch.allclose(features.neighbor_coordinates, expected)
        )
        unsigned = (current.abs() / features.degree[:, None]).matmul(
            coordinates
        )
        self.assertFalse(
            torch.allclose(features.neighbor_coordinates, unsigned)
        )

    def test_first_window_delta_is_zero_and_lengths_are_preserved(self):
        model = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=True, dropout=0.0)
        )
        output = model(self.batch)
        self.assertEqual(output.diagnostics["sequence_lengths"], (2, 3))
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        for sample_encoding in output.window_encodings:
            self.assertTrue(
                torch.allclose(
                    sample_encoding[0].features.delta_degree,
                    torch.zeros_like(
                        sample_encoding[0].features.delta_degree
                    ),
                )
            )

    def test_list_batch_and_node_mask_ignore_padding(self):
        model = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=True, dropout=0.0)
        ).eval()
        with torch.no_grad():
            alone = model(ExactSTSEBatch((self.left,))).logits[0]
            batched = model(self.batch).logits[0]
        self.assertTrue(torch.allclose(alone, batched, atol=1.0e-6))

        graph = self.left.graph.adjacency[0]
        padded_graph = torch.zeros(4, 4)
        padded_graph[:3, :3] = graph
        coordinates = torch.cat(
            (self.left.coordinates[0], torch.tensor([[999.0, -999.0, 2.0]])),
            dim=0,
        )
        encoder = model.window_encoder
        with torch.no_grad():
            reference = encoder(
                graph,
                graph,
                self.left.graph.node_names[0],
                self.left.graph.node_names[0],
                self.left.coordinates[0],
                self.left.graph.communities[0],
            ).window_embedding
            padded = encoder(
                padded_graph,
                padded_graph,
                ("roi-a", "roi-b", "roi-c", "padding"),
                ("roi-a", "roi-b", "roi-c", "padding"),
                coordinates,
                torch.tensor([0, 0, 1, 7]),
                node_mask=torch.tensor([True, True, True, False]),
            ).window_embedding
        self.assertTrue(torch.allclose(reference, padded, atol=1.0e-6))

    def test_no_coordinate_ablation_cannot_read_coordinates(self):
        changed = ExactSTSESample(
            graph=self.left.graph,
            coordinates=tuple(
                item * -17.0 + 5.0 for item in self.left.coordinates
            ),
        )
        no_coordinate = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=False, dropout=0.0)
        ).eval()
        coordinate = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=True, dropout=0.0)
        ).eval()
        with torch.no_grad():
            left_no = no_coordinate(ExactSTSEBatch((self.left,))).logits
            changed_no = no_coordinate(ExactSTSEBatch((changed,))).logits
            left_coordinate = coordinate(
                ExactSTSEBatch((self.left,))
            ).logits
            changed_coordinate = coordinate(
                ExactSTSEBatch((changed,))
            ).logits
        self.assertTrue(torch.equal(left_no, changed_no))
        self.assertFalse(torch.allclose(left_coordinate, changed_coordinate))

    def test_consistent_node_permutation_does_not_change_prediction(self):
        permutation = torch.tensor([2, 0, 1])
        permuted = _permuted(self.left, permutation)
        for use_coordinates in (True, False):
            model = ExactSTSEClassifier(
                ExactSTSEConfig(
                    use_coordinates=use_coordinates, dropout=0.0
                )
            ).eval()
            with torch.no_grad():
                original = model(
                    ExactSTSEBatch((self.left,))
                ).logits
                reordered = model(
                    ExactSTSEBatch((permuted,))
                ).logits
            self.assertTrue(
                torch.allclose(original, reordered, atol=1.0e-6)
            )

    def test_gradients_and_paired_initialization(self):
        coordinate = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=True, dropout=0.0)
        )
        no_coordinate = ExactSTSEClassifier(
            ExactSTSEConfig(use_coordinates=False, dropout=0.0)
        )
        coordinate.reset_parameters_with_seed(42)
        no_coordinate.reset_parameters_with_seed(42)
        self.assertTrue(
            torch.equal(
                coordinate.window_encoder.ffn_linear1.weight,
                no_coordinate.window_encoder.ffn_linear1.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                coordinate.classifier[0].weight,
                no_coordinate.classifier[0].weight,
            )
        )
        output = coordinate(self.batch)
        torch.nn.functional.cross_entropy(
            output.logits, self.batch.labels
        ).backward()
        modules = (
            coordinate.window_encoder.community_embedding,
            coordinate.window_encoder.input_projection,
            coordinate.window_encoder.ffn_linear1,
            coordinate.classifier,
        )
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
