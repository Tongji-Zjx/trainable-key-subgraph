import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keysubgraph.background.data import (
    BackgroundFeatureScaler,
    GlobalStaticGraphRecord,
    build_relative_signed_connectivity_profile,
    fit_background_feature_scaler,
    fit_train_community_kappa,
    signed_normalized_channels,
)
from keysubgraph.background.model import (
    GlobalBackgroundGCN,
    StaticBackgroundConfig,
    masked_support_shrunk_community_mean_std,
)
from keysubgraph.background.s4_fusion import (
    S4AnchoredFusionConfig,
    S4StaticPromotionConfig,
    apply_s4_anchored_fusion,
    apply_s4_seed_ensemble,
    fit_s4_seed_ensemble,
    select_s4_static_promotion,
    select_s4_anchored_fusion,
)
from keysubgraph.background.training import (
    BackgroundFusionDataset,
    BackgroundTrainingConfig,
    ranking_margin_consistency_loss,
    train_background_model,
)


class MoKSEBackgroundS4Test(unittest.TestCase):
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

    def record(self, key, features, communities):
        adjacency = self.adjacency()
        positive, negative = signed_normalized_channels(adjacency)
        return GlobalStaticGraphRecord(
            sample_key=key,
            sample_id=key,
            site="site",
            label=0,
            source_path="unused",
            source_sha256="0" * 64,
            node_features=features,
            community_labels=communities,
            raw_positive_adjacency=adjacency.clamp_min(0.0),
            raw_negative_adjacency=-adjacency.clamp_max(0.0),
            positive_adjacency=positive,
            negative_adjacency=negative,
            eigenvalues=torch.zeros(2),
        )

    def test_relative_profile_is_scale_robust_and_empty_safe(self):
        adjacency = self.adjacency()
        first = build_relative_signed_connectivity_profile(adjacency)
        scaled = adjacency.clamp_min(0.0) * 7.0 + adjacency.clamp_max(0.0) * 3.0
        second = build_relative_signed_connectivity_profile(scaled)
        self.assertEqual(tuple(first.shape), (5, 10))
        self.assertTrue(torch.allclose(first, second, atol=2.0e-6))
        positive_only = adjacency.clamp_min(0.0)
        empty = build_relative_signed_connectivity_profile(positive_only)
        self.assertTrue(torch.equal(empty[:, 4:8], torch.zeros(5, 4)))
        self.assertTrue(torch.equal(empty[:, -1], torch.zeros(5)))
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(float(first[:, :8].abs().max()), 4.0)

    def test_binary_profile_flags_can_bypass_train_scaling(self):
        features = torch.randn(5, 30)
        features[:, -2:] = torch.tensor([1.0, 0.0])
        record = self.record("a", features, torch.tensor([0, 0, 1, 1, 2]))
        scaler = fit_background_feature_scaler(
            (record,), passthrough_indices=(28, 29)
        )
        transformed = scaler.transform(features)
        self.assertTrue(torch.equal(transformed[:, -2:], features[:, -2:]))
        restored = BackgroundFeatureScaler.from_dict(scaler.as_dict())
        self.assertEqual(restored.passthrough_indices, (28, 29))

    def test_train_community_kappa_and_support_shrinkage(self):
        features = torch.randn(5, 30)
        records = (
            self.record("a", features, torch.tensor([0, 0, 1, 1, 2])),
            self.record("b", features, torch.tensor([3, 3, 3, 4, 4])),
        )
        self.assertEqual(fit_train_community_kappa(records), 2.0)
        values = torch.tensor([[[4.0], [0.0], [10.0], [0.0], [0.0]]])
        labels = torch.tensor([[0, 0, 1, 2, 2]])
        mask = torch.ones(1, 5, dtype=torch.bool)
        pooled = masked_support_shrunk_community_mean_std(
            values, labels, mask, kappa=2.0
        )
        self.assertEqual(tuple(pooled.shape), (1, 2))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_s4_model_has_bounded_normalized_residual_gates(self):
        torch.manual_seed(4)
        config = StaticBackgroundConfig(
            input_dim=30,
            base_input_dim=20,
            profile_input_dim=10,
            hidden_dim=8,
            representation_dim=5,
            layers=2,
            dropout=0.0,
            encoder_variant="s4_robust",
            enable_community_residual=True,
            support_shrunk_community=True,
            community_kappa=2.0,
            community_gate_max=0.10,
            community_gate_initial=0.02,
        )
        model = GlobalBackgroundGCN(config).eval()
        features = torch.randn(1, 5, 30)
        adjacency = self.adjacency()
        positive, negative = signed_normalized_channels(adjacency)
        output = model(
            features,
            positive[None],
            negative[None],
            torch.ones(1, 5, dtype=torch.bool),
            torch.tensor([[0, 0, 1, 1, 2]]),
        )
        self.assertAlmostEqual(float(output["profile_gate"]), 0.04, places=6)
        self.assertAlmostEqual(float(output["community_gate"]), 0.02, places=6)
        self.assertFalse(model.profile_output_norm.elementwise_affine)
        self.assertFalse(model.community_residual[-1].elementwise_affine)
        self.assertTrue(torch.isfinite(output["background_logit"]).all())

    def test_margin_consistency_and_single_class_are_safe(self):
        first = torch.tensor([-1.0, 0.5, -0.2, 1.2])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            float(ranking_margin_consistency_loss(first, first, labels)), 0.0
        )
        same_class = torch.ones(4)
        self.assertEqual(
            float(ranking_margin_consistency_loss(first, first + 1.0, same_class)),
            0.0,
        )

    def test_s4_training_smoke_exports_gate_diagnostics(self):
        records = []
        rows = []
        labels = []
        communities = torch.tensor([0, 0, 1, 1, 2])
        for index in range(8):
            label = index % 2
            features = torch.randn(5, 30) + 0.05 * label
            record = self.record("sample-{}".format(index), features, communities)
            record = GlobalStaticGraphRecord(
                **dict(record.__dict__, label=label)
            )
            records.append(record)
            rows.append({"sample_key": record.sample_key, "site": "site"})
            labels.append(label)
        evolution = {
            "rows": rows,
            "base_logits": np.zeros(8, dtype=np.float32),
            "sample_embeddings": np.zeros((8, 4), dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
        }
        scaler = fit_background_feature_scaler(
            records, passthrough_indices=(28, 29)
        )
        dataset = BackgroundFusionDataset(records, evolution, scaler)
        model_config = StaticBackgroundConfig(
            input_dim=30,
            base_input_dim=20,
            profile_input_dim=10,
            hidden_dim=8,
            representation_dim=5,
            layers=1,
            dropout=0.0,
            encoder_variant="s4_robust",
            enable_community_residual=True,
            support_shrunk_community=True,
            community_kappa=2.0,
            community_gate_max=0.10,
            community_gate_initial=0.02,
        )
        training_config = BackgroundTrainingConfig(
            epochs=1,
            batch_size=4,
            patience=1,
            seed=7,
            lambda_rank=0.05,
            lambda_gate=1.0e-3,
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
            metrics = report["metrics"]["validation"]
            self.assertIn("profile_gate", metrics)
            self.assertIn("community_gate", metrics)
            self.assertAlmostEqual(metrics["profile_gate"], 0.04, places=3)

    def seed_payloads(self, role="development_oof"):
        keys = np.asarray(["a", "b", "c", "d"])
        base = np.asarray([-1.0, 0.7, -0.4, 1.2])
        return [
            {
                "seed": seed,
                "sample_keys": keys,
                "logits": base * scale + shift,
                "prediction_role": role,
            }
            for seed, scale, shift in (
                (43, 1.0, 0.0), (44, 2.0, 0.3), (45, 0.5, -0.2)
            )
        ]

    def test_seed_ensemble_uses_oof_and_never_averages_representations(self):
        with self.assertRaises(ValueError):
            fit_s4_seed_ensemble(self.seed_payloads(role="train_in_sample"))
        fit = fit_s4_seed_ensemble(self.seed_payloads())
        output = apply_s4_seed_ensemble(fit, self.seed_payloads(role="fixed_test"))
        self.assertFalse(fit["representation_averaging"])
        self.assertEqual(tuple(output["standardized_seed_logits"].shape), (4, 3))
        expected_raw = np.median(
            np.stack([row["logits"] for row in self.seed_payloads()], axis=1),
            axis=1,
        )
        self.assertTrue(np.allclose(output["raw_median_logit"], expected_raw))

    def fusion_folds(self):
        folds = []
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        subgraph = np.asarray([
            1.62434536, -0.61175641, -0.52817175, -1.07296862,
            0.86540763, -2.30153870, 1.74481176, -0.76120690,
        ])
        static = np.asarray([
            -0.84048045, 0.87531481, -0.26894603, -0.03007035,
            -1.16120860, 0.80797282, -0.43311528, 0.45005437,
        ])
        for fold in range(4):
            folds.append({
                "sample_keys": np.asarray([
                    "fold{}-{}".format(fold, index) for index in range(8)
                ]),
                "sites": np.asarray(["site"] * 8),
                "labels": labels,
                "subgraph_logits": subgraph,
                "static_scores": static,
                "static_uncertainty": np.full(8, 0.1),
                "prediction_role": "development_oof",
            })
        return folds

    def test_anchored_fusion_selects_from_oof_and_beta_zero_is_exact(self):
        selection = select_s4_anchored_fusion(
            self.fusion_folds(),
            S4AnchoredFusionConfig(dataset="wmrc"),
        )
        self.assertGreater(selection["selected_beta"], 0.0)
        fold = self.fusion_folds()[0]
        fallback = dict(selection)
        fallback["selected_beta"] = 0.0
        fallback["selected_source"] = "subgraph_exact_fallback"
        applied = apply_s4_anchored_fusion(
            fallback,
            fold["subgraph_logits"],
            fold["static_scores"],
            fold["static_uncertainty"],
        )
        self.assertTrue(np.array_equal(
            applied["fused_logits"], fold["subgraph_logits"]
        ))
        invalid = [dict(row) for row in self.fusion_folds()]
        invalid[0]["prediction_role"] = "fixed_test"
        with self.assertRaises(ValueError):
            select_s4_anchored_fusion(
                invalid, S4AnchoredFusionConfig(dataset="wmrc")
            )

    def test_static_promotion_is_explicitly_test_guided(self):
        s3 = [
            {"roc_auc": 0.55, "accuracy": 0.60, "auprc": 0.50,
             "site_stratified_roc_auc": 0.52},
            {"roc_auc": 0.57, "accuracy": 0.61, "auprc": 0.51,
             "site_stratified_roc_auc": 0.53},
            {"roc_auc": 0.56, "accuracy": 0.60, "auprc": 0.50,
             "site_stratified_roc_auc": 0.52},
        ]
        s4 = [
            {"roc_auc": 0.57, "accuracy": 0.61, "auprc": 0.52,
             "site_stratified_roc_auc": 0.54},
            {"roc_auc": 0.58, "accuracy": 0.61, "auprc": 0.52,
             "site_stratified_roc_auc": 0.54},
            {"roc_auc": 0.57, "accuracy": 0.60, "auprc": 0.51,
             "site_stratified_roc_auc": 0.53},
        ]
        result = select_s4_static_promotion(
            s3, s4, S4StaticPromotionConfig()
        )
        self.assertTrue(result["s4_promoted"])
        self.assertTrue(result["test_guided_architecture_selection"])
        self.assertFalse(result["unbiased_generalization_estimate"])


if __name__ == "__main__":
    unittest.main()
