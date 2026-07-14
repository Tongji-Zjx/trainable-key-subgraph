from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceBatch,
    GraphSequenceSample,
)
from keysubgraph.models.losses import compute_soft_graph_loss  # noqa: E402
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)


def _adjacency(node_count, scale=1.0):
    graph = torch.zeros(node_count, node_count)
    for index in range(node_count - 1):
        value = scale * (0.4 if index % 2 == 0 else -0.3)
        graph[index, index + 1] = value
        graph[index + 1, index] = value
    return graph


def _sample(key, label, node_counts):
    adjacency = tuple(_adjacency(count, 1.0 + time * 0.1) for time, count in enumerate(node_counts))
    masks = []
    names = []
    communities = []
    for graph, count in zip(adjacency, node_counts):
        mask = graph.abs() > 0
        mask.fill_diagonal_(False)
        masks.append(mask)
        names.append(tuple("node_{}".format(index) for index in range(count)))
        communities.append(torch.tensor([index % 2 for index in range(count)]))
    return GraphSequenceSample(
        sample_key=key,
        sample_id=key.replace("/", "_"),
        site="SITE",
        subject_id=key,
        session_id="1",
        label=label,
        split="train",
        relative_path="unused.pt",
        adjacency=adjacency,
        edge_mask=tuple(masks),
        node_names=tuple(names),
        communities=tuple(communities),
        window_starts=torch.arange(len(node_counts), dtype=torch.float32),
        source_global_threshold=0.1,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


class SoftExtractorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.batch = GraphSequenceBatch(
            (
                _sample("SITE/a", 0, (3, 2)),
                _sample("SITE/b", 1, (4,)),
            )
        )
        self.config = SoftExtractorConfig(
            node_score_hidden_dim=8,
            edge_score_hidden_dim=8,
            graph_hidden_dim=10,
            graph_layers=2,
            classifier_hidden_dim=6,
            dropout=0.0,
        )
        self.model = SoftGraphClassifier(self.config)

    def test_forward_preserves_variable_lengths_scores_and_signs(self):
        output = self.model(self.batch, return_details=True)

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.sample_embeddings.shape), (2, 10))
        self.assertEqual(tuple(output.node_retention_ratios.shape), (3,))
        self.assertEqual(tuple(output.edge_retention_ratios.shape), (3,))
        self.assertEqual([len(item) for item in output.selections], [2, 1])
        for sample, selections in zip(self.batch, output.selections):
            for time_index, selection in enumerate(selections):
                node_scores = selection.node_scores
                edge_scores = selection.edge_scores
                edge_mask = sample.edge_mask[time_index]
                adjacency = sample.adjacency[time_index]
                self.assertGreaterEqual(float(node_scores.min()), 0.0)
                self.assertLessEqual(float(node_scores.max()), 1.0)
                self.assertGreaterEqual(float(edge_scores.min()), 0.0)
                self.assertLessEqual(float(edge_scores.max()), 1.0)
                self.assertTrue(torch.allclose(edge_scores, edge_scores.transpose(0, 1)))
                self.assertEqual(float(edge_scores[~edge_mask].abs().sum()), 0.0)
                self.assertEqual(tuple(selection.soft_adjacency.shape), tuple(adjacency.shape))
                self.assertTrue(
                    torch.equal(
                        torch.sign(selection.soft_adjacency[edge_mask]),
                        torch.sign(adjacency[edge_mask]),
                    )
                )
        self.assertFalse(self.model.uses_raw_community_embedding)
        self.assertEqual(self.model.training_mode, "soft_graph")

    def test_negative_edge_remains_negative_after_soft_selection(self):
        graph = torch.tensor(
            [[0.0, -0.8, 0.0], [-0.8, 0.0, 0.4], [0.0, 0.4, 0.0]]
        )
        mask = graph.abs() > 0.0
        mask.fill_diagonal_(False)
        sample = GraphSequenceSample(
            sample_key="SITE/negative",
            sample_id="negative",
            site="SITE",
            subject_id="negative",
            session_id="1",
            label=0,
            split="train",
            relative_path="unused.pt",
            adjacency=(graph,),
            edge_mask=(mask,),
            node_names=(("a", "b", "c"),),
            communities=(torch.tensor([0, 0, 1]),),
            window_starts=torch.tensor([0.0]),
            source_global_threshold=0.1,
            repetition_time=2.0,
            edge_presence_threshold=0.0,
        )

        _, selection = self.model.score_timepoint(sample, 0)

        self.assertLess(float(selection.soft_adjacency[0, 1]), 0.0)
        self.assertGreater(float(selection.soft_adjacency[1, 2]), 0.0)

    def test_list_batch_size_variation_does_not_change_sample_output(self):
        target = _sample("SITE/target", 0, (3, 2))
        other = _sample("SITE/other", 1, (6,))
        self.model.eval()
        with torch.no_grad():
            alone = self.model(GraphSequenceBatch((target,))).logits[0]
            together = self.model(GraphSequenceBatch((target, other))).logits[0]

        self.assertTrue(torch.allclose(alone, together, atol=1e-6))

    def test_classification_and_budget_loss_backpropagate_to_both_scorers(self):
        output = self.model(self.batch)
        loss = compute_soft_graph_loss(
            output,
            self.batch.labels,
            target_node_ratio=0.25,
            target_edge_ratio=0.2,
            budget_weight=0.5,
        )
        loss.total.backward()

        self.assertTrue(bool(torch.isfinite(loss.total)))
        node_gradient = self.model.node_scorer.network[0].weight.grad
        edge_gradient = self.model.edge_scorer.network[0].weight.grad
        self.assertIsNotNone(node_gradient)
        self.assertIsNotNone(edge_gradient)
        self.assertGreater(float(node_gradient.abs().sum()), 0.0)
        self.assertGreater(float(edge_gradient.abs().sum()), 0.0)

    def test_prediction_is_invariant_to_consistent_node_permutation(self):
        sample = _sample("SITE/original", 0, (4,))
        permutation = torch.tensor([2, 0, 3, 1])
        graph = sample.adjacency[0].index_select(0, permutation).index_select(1, permutation)
        mask = graph.abs() > 0
        mask.fill_diagonal_(False)
        permuted = GraphSequenceSample(
            sample_key="SITE/permuted",
            sample_id="permuted",
            site=sample.site,
            subject_id="permuted",
            session_id=sample.session_id,
            label=sample.label,
            split=sample.split,
            relative_path=sample.relative_path,
            adjacency=(graph,),
            edge_mask=(mask,),
            node_names=(tuple(sample.node_names[0][index] for index in permutation.tolist()),),
            communities=(sample.communities[0].index_select(0, permutation),),
            window_starts=sample.window_starts,
            source_global_threshold=sample.source_global_threshold,
            repetition_time=sample.repetition_time,
            edge_presence_threshold=sample.edge_presence_threshold,
        )
        self.model.eval()
        with torch.no_grad():
            original_logits = self.model(GraphSequenceBatch((sample,))).logits
            permuted_logits = self.model(GraphSequenceBatch((permuted,))).logits
        self.assertTrue(torch.allclose(original_logits, permuted_logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
