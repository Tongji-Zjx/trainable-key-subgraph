from __future__ import absolute_import, division, print_function

import unittest
import tempfile
from pathlib import Path

import torch

from keysubgraph.data.graph_dataset import GraphSequenceBatch, GraphSequenceSample
from keysubgraph.models import (
    TGSoftTeacher,
    TGSoftTeacherConfig,
    TGSoftTeacherLossConfig,
    compute_tg_soft_teacher_loss,
    tg_soft_teacher_ablation_weights,
)
from keysubgraph.training.tg_soft_teacher_trainer import (
    TGSoftTeacherTrainingConfig,
    run_tg_soft_teacher_epoch,
    train_tg_soft_teacher,
)


def _sample(key, label, windows):
    adjacency = []
    masks = []
    names = []
    communities = []
    for index in range(windows):
        scale = 1.0 + 0.1 * index
        graph = torch.tensor(
            [[0.0, 0.7 * scale, -0.3], [0.7 * scale, 0.0, 0.2], [-0.3, 0.2, 0.0]],
            dtype=torch.float32,
        )
        adjacency.append(graph)
        mask = graph.abs() > 0.0
        mask.fill_diagonal_(False)
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


class _OneBatchLoader(object):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch


class TGSoftTeacherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.batch = GraphSequenceBatch((_sample("a", 0, 2), _sample("b", 1, 3)))
        self.model = TGSoftTeacher(
            TGSoftTeacherConfig(
                node_score_hidden_dim=16,
                edge_score_hidden_dim=8,
                signed_gnn_hidden_dim=16,
                signed_gnn_layers=2,
                classifier_hidden_dims=(16,),
                dropout=0.0,
                diffusion_time=0.2,
            )
        )

    def test_forward_contract_and_variable_length_mask(self):
        output = self.model(self.batch, return_details=True)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.representation.shape), (2, 192))
        self.assertEqual(output.time_mask.sum(dim=1).tolist(), [2, 3])
        self.assertEqual(output.node_retention_ratios.numel(), 5)
        self.assertEqual(output.gw_identity_upper_bounds_squared.numel(), 5)
        self.assertEqual(len(output.selections), 2)
        self.assertLess(float(output.selections[0][0].soft_adjacency[0, 2]), 0.0)

    def test_joint_loss_backpropagates_to_scorers_encoder_and_tcn(self):
        output = self.model(self.batch)
        config = TGSoftTeacherLossConfig(
            laplacian_max_weight=0.5,
            gw_identity_max_weight=0.1,
            theory_warmup_epochs=10,
        )
        loss = compute_tg_soft_teacher_loss(output, self.batch.labels, epoch=5, config=config)
        self.assertAlmostEqual(loss.effective_laplacian_weight, 0.25)
        self.assertAlmostEqual(loss.effective_gw_weight, 0.05)
        loss.total.backward()
        modules = (self.model.node_scorer, self.model.edge_scorer, self.model.graph_encoder, self.model.temporal_encoder)
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_batch_size_one_preserves_dataset_class_weight(self):
        batch = GraphSequenceBatch((_sample("minority", 1, 2),))
        output = self.model(batch)
        config = TGSoftTeacherLossConfig(
            budget_weight=0.0,
            laplacian_max_weight=0.0,
            gw_identity_max_weight=0.0,
            supervised_contrastive_weight=0.0,
        )
        unweighted = compute_tg_soft_teacher_loss(
            output, batch.labels, epoch=1, config=config
        )
        weighted = compute_tg_soft_teacher_loss(
            output,
            batch.labels,
            epoch=1,
            config=config,
            class_weights=torch.tensor([1.0, 3.0]),
        )
        self.assertTrue(
            torch.allclose(weighted.classification, 3.0 * unweighted.classification)
        )

    def test_score_distribution_matches_unpadded_nodes_and_real_edges(self):
        output = self.model(self.batch, return_details=True)
        node_values = torch.cat(
            [item.node_scores for sample in output.selections for item in sample]
        )
        edge_values = []
        for sample, selections in zip(self.batch.samples, output.selections):
            for time_index, item in enumerate(selections):
                upper = torch.triu(sample.edge_mask[time_index], diagonal=1)
                edge_values.append(item.edge_scores[upper])
        edge_values = torch.cat(edge_values)
        for values, statistics in (
            (node_values, output.node_score_statistics),
            (edge_values, output.edge_score_statistics),
        ):
            self.assertEqual(statistics.count, values.numel())
            self.assertTrue(torch.allclose(statistics.total, values.sum()))
            self.assertTrue(torch.allclose(statistics.squared_total, values.square().sum()))
            self.assertTrue(torch.allclose(statistics.minimum, values.min()))
            self.assertTrue(torch.allclose(statistics.maximum, values.max()))

    def test_minimal_ablation_presets_are_nested(self):
        self.assertEqual(
            tg_soft_teacher_ablation_weights("classification_only"),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            tg_soft_teacher_ablation_weights("classification_budget"),
            (0.10, 0.0, 0.0),
        )
        self.assertEqual(
            tg_soft_teacher_ablation_weights("classification_budget_laplacian"),
            (0.10, 0.50, 0.0),
        )
        self.assertEqual(
            tg_soft_teacher_ablation_weights("full"),
            (0.10, 0.50, 0.10),
        )

    def test_epoch_runner_updates_parameters_and_reports_named_fidelity(self):
        before = self.model.node_scorer.network[0].weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        metrics = run_tg_soft_teacher_epoch(
            self.model,
            _OneBatchLoader(self.batch),
            torch.device("cpu"),
            epoch=1,
            loss_config=TGSoftTeacherLossConfig(theory_warmup_epochs=2),
            optimizer=optimizer,
        )
        self.assertEqual(metrics["sample_count"], 2)
        self.assertIn("gw_identity_upper_bound", metrics)
        self.assertIn("node_score_std", metrics)
        self.assertIn("edge_score_entropy", metrics)
        self.assertGreater(metrics["node_score_count"], 0)
        self.assertGreater(metrics["edge_score_count"], 0)
        self.assertIsNotNone(metrics["mean_gradient_norm"])
        self.assertFalse(torch.allclose(before, self.model.node_scorer.network[0].weight))

    def test_training_can_resume_without_restarting_history(self):
        loader = _OneBatchLoader(self.batch)
        loss_config = TGSoftTeacherLossConfig(theory_warmup_epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            first = train_tg_soft_teacher(
                self.model,
                loader,
                loader,
                [0, 1],
                torch.device("cpu"),
                loss_config,
                TGSoftTeacherTrainingConfig(epochs=1, early_stopping_patience=3),
                output_dir,
                Path(directory) / "protocol.json",
                "protocol-hash",
            )
            resumed_model = TGSoftTeacher(self.model.config)
            second = train_tg_soft_teacher(
                resumed_model,
                loader,
                loader,
                [0, 1],
                torch.device("cpu"),
                loss_config,
                TGSoftTeacherTrainingConfig(epochs=2, early_stopping_patience=3),
                output_dir,
                Path(directory) / "protocol.json",
                "protocol-hash",
                resume_checkpoint=first["last_checkpoint"],
            )
            self.assertEqual(second["epochs_completed"], 2)
            self.assertTrue(second["best_checkpoint"].is_file())
            self.assertTrue(second["last_checkpoint"].is_file())


if __name__ == "__main__":
    unittest.main()
