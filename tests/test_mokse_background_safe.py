import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keysubgraph.background.data import (
    BackgroundFeatureScaler,
    GlobalStaticGraphRecord,
    build_signed_connectivity_profile,
    signed_normalized_channels,
)
from keysubgraph.background.model import (
    GlobalBackgroundGCN,
    StaticBackgroundConfig,
    masked_community_mean_std,
)
from keysubgraph.background.safe_fusion import (
    SafeFusionConfig,
    apply_safe_fusion,
    fuse_logits,
    select_safe_fusion,
)
from keysubgraph.background.training import (
    BackgroundFusionDataset,
    BackgroundTrainingConfig,
    deterministic_signed_balanced_dropedge,
    train_background_model,
)


class MoKSEBackgroundSafeTest(unittest.TestCase):
    def adjacency(self):
        return torch.tensor(
            [
                [0.0, 0.8, 0.4, -0.2, -0.6],
                [0.8, 0.0, 0.3, -0.5, 0.0],
                [0.4, 0.3, 0.0, 0.7, -0.1],
                [-0.2, -0.5, 0.7, 0.0, 0.9],
                [-0.6, 0.0, -0.1, 0.9, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_signed_profile_is_ten_dimensional_and_permutation_equivariant(self):
        adjacency = self.adjacency()
        profile = build_signed_connectivity_profile(adjacency)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = adjacency.index_select(0, permutation).index_select(1, permutation)
        permuted_profile = build_signed_connectivity_profile(permuted)
        self.assertEqual(tuple(profile.shape), (5, 10))
        self.assertTrue(
            torch.allclose(permuted_profile, profile.index_select(0, permutation))
        )
        self.assertTrue(torch.all(profile[:, :4] >= 0.0))
        self.assertTrue(torch.all(profile[:, 4:8] >= 0.0))

    def test_signed_profile_empty_channel_uses_validity_flag(self):
        adjacency = torch.tensor(
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.2], [0.0, 0.2, 0.0]]
        )
        profile = build_signed_connectivity_profile(adjacency)
        self.assertTrue(torch.equal(profile[:, 4:8], torch.zeros(3, 4)))
        self.assertTrue(torch.equal(profile[:, -1], torch.zeros(3)))
        self.assertTrue(torch.equal(profile[:, -2], torch.ones(3)))

    def test_community_pooling_ignores_padding_and_community_ids(self):
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0], [99.0, 99.0]]]
        )
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        labels = torch.tensor([[7, 7, 11, -1]])
        renamed = torch.tensor([[101, 101, 3, -1]])
        first = masked_community_mean_std(values, labels, mask)
        second = masked_community_mean_std(values, renamed, mask)
        self.assertEqual(tuple(first.shape), (1, 4))
        self.assertTrue(torch.allclose(first, second))

    def test_community_residual_model_accepts_variable_community_counts(self):
        torch.manual_seed(3)
        features = torch.randn(2, 5, 6)
        positive = torch.zeros(2, 5, 5)
        negative = torch.zeros_like(positive)
        positive[:, 0, 1] = positive[:, 1, 0] = 0.5
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        communities = torch.tensor([[0, 0, 1, -1, -1], [4, 4, 7, 7, 9]])
        config = StaticBackgroundConfig(
            input_dim=6,
            hidden_dim=8,
            representation_dim=5,
            layers=2,
            dropout=0.0,
            enable_community_residual=True,
        )
        output = GlobalBackgroundGCN(config)(
            features, positive, negative, mask, communities
        )
        self.assertEqual(tuple(output["background_representation"].shape), (2, 5))
        self.assertAlmostEqual(float(output["community_gate"]), 0.05, places=6)

    def test_signed_dropedge_is_symmetric_subset_and_reproducible(self):
        adjacency = self.adjacency()
        raw_positive = adjacency.clamp_min(0.0)[None]
        raw_negative = (-adjacency.clamp_max(0.0))[None]
        mask = torch.ones(1, adjacency.shape[0], dtype=torch.bool)
        first = deterministic_signed_balanced_dropedge(
            raw_positive, raw_negative, mask, ("sample",), 0.5, 43, 2
        )
        second = deterministic_signed_balanced_dropedge(
            raw_positive, raw_negative, mask, ("sample",), 0.5, 43, 2
        )
        for index in range(4):
            self.assertTrue(torch.equal(first[index], second[index]))
        dropped_positive, dropped_negative = first[2], first[3]
        self.assertTrue(torch.equal(dropped_positive, dropped_positive.transpose(1, 2)))
        self.assertTrue(torch.equal(dropped_negative, dropped_negative.transpose(1, 2)))
        self.assertTrue(torch.all((dropped_positive == 0.0) | (dropped_positive == raw_positive)))
        self.assertTrue(torch.all((dropped_negative == 0.0) | (dropped_negative == raw_negative)))
        self.assertTrue(torch.isfinite(first[0]).all())
        self.assertTrue(torch.isfinite(first[1]).all())

    def validation_folds(self, background_good):
        folds = []
        for rotation in range(4):
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            subgraph = np.asarray([-0.2, -0.1, 0.2, 0.1], dtype=np.float64)
            background = (
                np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64)
                if background_good
                else -np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64)
            )
            folds.append(
                {
                    "sample_keys": np.asarray(
                        ["r{}-{}".format(rotation, index) for index in range(4)]
                    ),
                    "sites": np.asarray(["site"] * 4),
                    "labels": labels,
                    "subgraph_logits": subgraph,
                    "background_logits": background,
                }
            )
        return folds

    def test_a_one_and_adhd_fallback_exactly_reproduce_subgraph_logits(self):
        values = np.asarray([-2.0, 0.0, 3.0])
        background = np.asarray([100.0, -100.0, 20.0])
        self.assertTrue(np.array_equal(fuse_logits(values, background, 1.0, 0.01), values))
        selection = select_safe_fusion(
            self.validation_folds(background_good=False),
            SafeFusionConfig(dataset="adhd"),
        )
        self.assertEqual(selection["selected_source"], "subgraph_exact_fallback")
        self.assertTrue(
            np.array_equal(apply_safe_fusion(selection, values, background), values)
        )

    def test_wmrc_accepts_complementary_background_when_checks_pass(self):
        selection = select_safe_fusion(
            self.validation_folds(background_good=True),
            SafeFusionConfig(dataset="wmrc"),
        )
        self.assertTrue(selection["fusion_accepted"])
        self.assertEqual(selection["selected_source"], "safe_convex_fusion")
        self.assertLess(selection["selected_subgraph_weight"], 1.0)

    def test_two_view_training_and_top_checkpoint_ensemble_smoke(self):
        adjacency = self.adjacency()
        positive, negative = signed_normalized_channels(adjacency)
        records = []
        rows = []
        labels = []
        for index in range(8):
            label = index % 2
            features = torch.randn(5, 6) + 0.1 * label
            records.append(
                GlobalStaticGraphRecord(
                    sample_key="sample-{}".format(index),
                    sample_id="sample-{}".format(index),
                    site="site",
                    label=label,
                    source_path="unused",
                    source_sha256="0" * 64,
                    node_features=features,
                    community_labels=torch.tensor([0, 0, 1, 1, 2]),
                    raw_positive_adjacency=adjacency.clamp_min(0.0),
                    raw_negative_adjacency=-adjacency.clamp_max(0.0),
                    positive_adjacency=positive,
                    negative_adjacency=negative,
                    eigenvalues=torch.zeros(2),
                )
            )
            rows.append(
                {
                    "sample_key": "sample-{}".format(index),
                    "site": "site",
                }
            )
            labels.append(label)
        evolution = {
            "rows": rows,
            "base_logits": np.zeros(8, dtype=np.float32),
            "sample_embeddings": np.zeros((8, 24), dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
        }
        scaler = BackgroundFeatureScaler(mean=torch.zeros(6), scale=torch.ones(6))
        dataset = BackgroundFusionDataset(records, evolution, scaler)
        model_config = StaticBackgroundConfig(
            input_dim=6,
            hidden_dim=8,
            representation_dim=5,
            layers=2,
            dropout=0.0,
            enable_community_residual=True,
        )
        training_config = BackgroundTrainingConfig(
            epochs=2,
            batch_size=4,
            patience=2,
            seed=9,
            signed_dropedge_probability=0.25,
            lambda_consistency=0.05,
            checkpoint_ensemble_top_k=2,
            strict_deterministic=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            report = train_background_model(
                dataset,
                dataset,
                None,
                scaler,
                Path(directory),
                torch.device("cpu"),
                "background_only",
                model_config=model_config,
                training_config=training_config,
            )
            self.assertEqual(report["checkpoint_ensemble_size"], 2)
            self.assertNotIn("test", report["metrics"])
            self.assertTrue((Path(directory) / "checkpoint_top_2.pt").is_file())


if __name__ == "__main__":
    unittest.main()
