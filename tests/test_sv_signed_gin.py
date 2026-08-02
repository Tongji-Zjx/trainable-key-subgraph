from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
from keysubgraph.models.sv_signed_gin import (
    SV_DEFAULT_VARIANT,
    SV_SIGNED_GIN_VARIANTS,
    SignedGINKeySubgraphEncoder,
    SignedGINLayer,
    SVSignedGINBatch,
    SVSignedGINClassifier,
    SVSignedGINConfig,
    SVSignedGINSampleInput,
    SVSignedGINWindowInput,
)


def _sample(key="sample", label=0, permutation=None):
    features = torch.arange(45, dtype=torch.float32).reshape(3, 15) / 20.0
    adjacency = torch.tensor(
        (
            (0.0, 0.5, -0.2),
            (0.5, 0.0, 0.3),
            (-0.2, 0.3, 0.0),
        ),
        dtype=torch.float32,
    )
    communities = torch.tensor((11, 11, 29), dtype=torch.long)
    if permutation is not None:
        features = features.index_select(0, permutation)
        adjacency = adjacency.index_select(0, permutation).index_select(
            1, permutation
        )
        communities = communities.index_select(0, permutation)
    windows = (
        SVSignedGINWindowInput(
            features,
            adjacency,
            time_position=0,
            hks=torch.linspace(0.1, 0.9, 18).reshape(3, 6),
            diffusion_eigenvalues=torch.tensor((0.1, 0.8, 1.4)),
            diffusion_eigenvectors=torch.eye(3),
            spectral_delta_to_next=torch.linspace(-0.5, 0.5, 16),
            communities=communities,
        ),
        SVSignedGINWindowInput(
            features + 0.1,
            adjacency * 0.9,
            time_position=1,
            hks=torch.linspace(0.2, 1.0, 18).reshape(3, 6),
            diffusion_eigenvalues=torch.tensor((0.1, 0.7, 1.3)),
            diffusion_eigenvectors=torch.eye(3),
            communities=communities,
        ),
    )
    return SVSignedGINSampleInput(
        sample_key=key,
        label=label,
        windows=windows,
        static_features=torch.linspace(0.0, 1.0, 28),
        variation=torch.linspace(0.1, 0.8, 16),
        spectral_direction=torch.linspace(-0.5, 0.5, 16),
        diffusion_geometry=torch.linspace(0.0, 1.0, 28),
    )


class SVSignedGINTest(unittest.TestCase):
    def test_default_profile_is_formal_svg(self):
        config = SVSignedGINConfig()
        self.assertEqual(
            config.variant,
            "signed_gin_multibranch_late_fusion",
        )
        self.assertEqual(config.variant, SV_DEFAULT_VARIANT)
        self.assertTrue(config.uses_static)
        self.assertTrue(config.uses_variation)
        self.assertTrue(config.uses_gin)
        self.assertTrue(config.uses_late_fusion)
        self.assertEqual(config.message_mode, "signed_normalized")
        self.assertEqual(config.pooling, "mean_std")
        self.assertTrue(config.gin_residual)
        self.assertTrue(config.gin_jumping_knowledge)
        self.assertTrue(config.gin_compact_readout)
        self.assertTrue(config.gin_batch_normalization)
        output = SVSignedGINClassifier(config)(
            SVSignedGINBatch((_sample("a", 0), _sample("b", 1)))
        )
        self.assertEqual(
            tuple(output.branch_logits),
            ("gin", "static_spectral", "variation"),
        )

    def test_legacy_svg_config_gains_only_default_sidecar_dimensions(self):
        current = SVSignedGINConfig()
        legacy = {
            key: value
            for key, value in current.__dict__.items()
            if key
            not in (
                "spectral_direction_dim",
                "diffusion_geometry_dim",
            )
        }
        restored = SVSignedGINConfig(**legacy)
        self.assertEqual(restored, current)
        self.assertFalse(restored.uses_theory_geometry)

    def test_signed_aggregation_uses_sign_and_magnitude(self):
        layer = SignedGINLayer(2, dropout=0.0)
        states = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        positive = torch.tensor(((0.0, 0.5), (0.5, 0.0)))
        negative = -positive
        positive_output = layer.signed_aggregate(states, positive)
        negative_output = layer.signed_aggregate(states, negative)
        stronger = layer.signed_aggregate(states, 2.0 * positive)
        self.assertTrue(
            torch.allclose(positive_output[0], torch.tensor((2.5, 4.0)))
        )
        self.assertTrue(
            torch.allclose(negative_output[0], torch.tensor((-0.5, 0.0)))
        )
        self.assertTrue(
            torch.allclose(stronger[0], torch.tensor((4.0, 6.0)))
        )

    def test_normalized_signed_aggregation_preserves_sign(self):
        layer = SignedGINLayer(
            1,
            dropout=0.0,
            learnable_epsilon=False,
            message_mode="signed_normalized",
        )
        states = torch.tensor(((1.0,), (3.0,)))
        positive = torch.tensor(((0.0, 2.0), (2.0, 0.0)))
        negative = -positive
        positive_output = layer.signed_aggregate(states, positive)
        negative_output = layer.signed_aggregate(states, negative)
        self.assertTrue(
            torch.allclose(positive_output[0], torch.tensor((4.0,)))
        )
        self.assertTrue(
            torch.allclose(negative_output[0], torch.tensor((-2.0,)))
        )

    def test_attention_and_node_permutation_invariance(self):
        torch.manual_seed(719)
        config = SVSignedGINConfig(dropout=0.0)
        encoder = SignedGINKeySubgraphEncoder(config).eval()
        original = encoder(_sample())
        permuted = encoder(
            _sample(permutation=torch.tensor([2, 0, 1]))
        )
        self.assertTrue(
            torch.allclose(
                original.representation,
                permuted.representation,
                atol=1.0e-6,
            )
        )
        for weights in original.node_attention:
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
            self.assertTrue(bool((weights >= 0.0).all()))

    def test_d1_pooling_is_node_and_community_relabel_invariant(self):
        torch.manual_seed(720)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="svg_v2_d1_community_pooling",
                dropout=0.0,
            )
        ).eval()
        original = _sample("a", 0)
        permutation = torch.tensor((2, 0, 1))
        permuted = _sample("a", 0, permutation=permutation)
        relabeled_windows = tuple(
            SVSignedGINWindowInput(
                node_features=window.node_features,
                adjacency=window.adjacency,
                time_position=window.time_position,
                hks=window.hks,
                diffusion_eigenvalues=window.diffusion_eigenvalues,
                diffusion_eigenvectors=window.diffusion_eigenvectors,
                spectral_delta_to_next=window.spectral_delta_to_next,
                communities=torch.where(
                    window.communities == 11,
                    torch.tensor(101),
                    torch.tensor(7),
                ),
            )
            for window in original.windows
        )
        relabeled = SVSignedGINSampleInput(
            sample_key=original.sample_key,
            label=original.label,
            windows=relabeled_windows,
            static_features=original.static_features,
            variation=original.variation,
            spectral_direction=original.spectral_direction,
            diffusion_geometry=original.diffusion_geometry,
        )
        first = model(SVSignedGINBatch((original,))).logits
        second = model(SVSignedGINBatch((permuted,))).logits
        third = model(SVSignedGINBatch((relabeled,))).logits
        self.assertTrue(torch.allclose(first, second, atol=1.0e-6))
        self.assertTrue(torch.allclose(first, third, atol=1.0e-6))

    def test_all_variants_have_expected_dimensions(self):
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        for variant in SV_SIGNED_GIN_VARIANTS:
            if variant == "svg_v2_e1_multi_budget":
                continue
            extra = {}
            if (
                variant
                == "signed_gin_static_anchor_residual_attention"
            ):
                extra = {
                    "pooling": "mean_std",
                    "gin_compact_readout": True,
                    "gin_residual_attention": True,
                }
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant=variant, dropout=0.0, **extra
                )
            )
            output = model(batch)
            expected = model.config.fusion_input_dim
            self.assertEqual(tuple(output.logits.shape), (2, 2))
            self.assertEqual(
                tuple(output.final_representation.shape), (2, expected)
            )
            self.assertEqual(
                output.diagnostics["preserves_signed_edges"], True
            )

    def test_e1_is_fixed_mean_of_three_shared_budget_views(self):
        torch.manual_seed(721)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="svg_v2_e1_multi_budget", dropout=0.0
            )
        ).eval()
        first = _sample("a-low", 0)
        middle = _sample("a-middle", 0)
        high = _sample("a-high", 0)
        sample = SVSignedGINSampleInput(
            sample_key="a",
            label=0,
            windows=middle.windows,
            static_features=middle.static_features,
            variation=middle.variation,
            budget_views=(first, middle, high),
        )
        output = model(SVSignedGINBatch((sample,)))
        individual = [
            model._forward_standard(SVSignedGINBatch((view,)))
            for view in (first, middle, high)
        ]
        projected = {
            "gin": torch.stack(
                [value.gin_projection for value in individual], dim=0
            ).mean(dim=0),
            "static_spectral": individual[1].static_projection,
            "variation": individual[1].variation_projection,
        }
        expected_branch_logits = {
            name: model.branch_classifiers[name](projected[name])
            for name in model.config.active_branch_names
        }
        weights = torch.softmax(model.fusion_log_weights, dim=0)
        expected_logits = (
            torch.stack(
                [
                    expected_branch_logits[name]
                    for name in model.config.active_branch_names
                ],
                dim=0,
            )
            * weights[:, None, None]
        ).sum(dim=0)
        self.assertTrue(
            torch.allclose(
                output.logits,
                expected_logits,
                atol=1.0e-7,
            )
        )
        self.assertEqual(
            output.diagnostics["multi_budget_fusion"],
            "fixed_equal_mean",
        )
        torch.nn.functional.cross_entropy(
            output.logits, torch.tensor((0,))
        ).backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_s_and_sv_are_strict_deletion_ablations(self):
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        cases = (
            (
                "static_spectral_only",
                ("static_spectral",),
                False,
                False,
            ),
            (
                "static_spectral_variation_late_fusion",
                ("static_spectral", "variation"),
                False,
                True,
            ),
        )
        for variant, branches, uses_gin, uses_variation in cases:
            model = SVSignedGINClassifier(
                SVSignedGINConfig(variant=variant, dropout=0.0)
            )
            output = model(batch)
            self.assertEqual(
                tuple(output.branch_logits), branches
            )
            self.assertEqual(model.encoder is not None, uses_gin)
            self.assertEqual(
                model.variation_projection is not None,
                uses_variation,
            )
            self.assertEqual(output.encoder_outputs, ())
            self.assertIsNone(output.gin_representation)
            self.assertAlmostEqual(
                float(output.fusion_weights.sum()), 1.0, places=6
            )

    def test_multibranch_fusion_is_nonnegative_and_supervises_all_branches(self):
        torch.manual_seed(731)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_multibranch_late_fusion",
                dropout=0.0,
                message_mode="signed_normalized",
                pooling="mean_std",
                gin_residual=True,
                gin_jumping_knowledge=True,
                gin_compact_readout=True,
                gin_batch_normalization=True,
            )
        )
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        output = model(batch)
        self.assertEqual(
            set(output.branch_logits),
            {"gin", "static_spectral", "variation"},
        )
        self.assertTrue(bool((output.fusion_weights >= 0.0).all()))
        self.assertAlmostEqual(
            float(output.fusion_weights.sum()), 1.0, places=6
        )
        loss = torch.nn.functional.cross_entropy(
            output.logits, batch.labels
        )
        loss = loss + 0.25 * torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    logits, batch.labels
                )
                for logits in output.branch_logits.values()
            ]
        ).mean()
        loss.backward()
        for name, branch in model.branch_classifiers.items():
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in branch.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, name)
        self.assertGreater(
            float(model.fusion_log_weights.grad.abs().sum()), 0.0
        )

    def test_theory_geometry_branches_are_explicit_and_trainable(self):
        torch.manual_seed(739)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_multibranch_theory_geometry",
                gin_hidden_dim=8,
                attention_hidden_dim=4,
                channel_projection_dim=4,
                fusion_hidden_dim=4,
                dropout=0.0,
            )
        )
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        output = model(batch)
        self.assertEqual(
            tuple(output.branch_logits),
            (
                "gin",
                "static_spectral",
                "variation",
                "spectral_direction",
                "diffusion_geometry",
            ),
        )
        self.assertEqual(
            tuple(output.spectral_direction_projection.shape), (2, 4)
        )
        self.assertEqual(
            tuple(output.diffusion_geometry_projection.shape), (2, 4)
        )
        loss = torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    logits, batch.labels
                )
                for logits in output.branch_logits.values()
            ]
        ).mean()
        loss.backward()
        for name, module in (
            (
                "spectral_direction",
                model.spectral_direction_projection,
            ),
            (
                "diffusion_geometry",
                model.diffusion_geometry_projection,
            ),
        ):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, name)

    def test_theory_geometry_variant_rejects_missing_sidecars(self):
        sample = _sample()
        missing = SVSignedGINSampleInput(
            sample_key=sample.sample_key,
            label=sample.label,
            windows=sample.windows,
            static_features=sample.static_features,
            variation=sample.variation,
        )
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_multibranch_theory_geometry",
                dropout=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "sidecars"):
            model(SVSignedGINBatch((missing,)))

    def test_compact_batch_normalized_gin_supports_singleton_batch(self):
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_multibranch_late_fusion",
                dropout=0.0,
                message_mode="signed_normalized",
                pooling="mean_std",
                gin_residual=True,
                gin_jumping_knowledge=True,
                gin_compact_readout=True,
                gin_batch_normalization=True,
            )
        )
        model.train()
        output = model(SVSignedGINBatch((_sample("only", 1),)))
        self.assertEqual(tuple(output.gin_representation.shape), (1, 64))
        self.assertEqual(tuple(output.gin_projection.shape), (1, 16))
        output.logits.sum().backward()

    def test_static_anchor_residual_starts_exactly_at_anchor(self):
        torch.manual_seed(739)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_static_anchor_residual",
                dropout=0.0,
                message_mode="signed_normalized",
                pooling="mean_std",
                gin_residual=True,
                gin_jumping_knowledge=True,
                gin_compact_readout=True,
                gin_batch_normalization=True,
            )
        ).eval()
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        output = model(batch)
        self.assertTrue(
            torch.equal(
                output.logits,
                output.branch_logits["static_spectral"],
            )
        )
        self.assertTrue(
            torch.equal(
                output.branch_logits["gin"],
                torch.zeros_like(output.branch_logits["gin"]),
            )
        )
        self.assertTrue(
            torch.equal(
                output.branch_logits["variation"],
                torch.zeros_like(output.branch_logits["variation"]),
            )
        )
        self.assertGreaterEqual(float(output.residual_gates["gin"]), 0.0)
        self.assertLess(float(output.residual_gates["gin"]), 0.01)

    def test_residual_stage_freezes_anchor_and_trains_experts(self):
        torch.manual_seed(743)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_static_anchor_residual",
                gin_hidden_dim=8,
                attention_hidden_dim=4,
                channel_projection_dim=4,
                fusion_hidden_dim=4,
                dropout=0.0,
                message_mode="signed_normalized",
                pooling="mean_std",
                gin_residual=True,
                gin_jumping_knowledge=True,
                gin_compact_readout=True,
                gin_batch_normalization=True,
            )
        )
        model.set_training_stage("residual_experts")
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        output = model(batch)
        loss = torch.nn.functional.cross_entropy(
            output.logits, batch.labels
        )
        loss = loss + 0.25 * torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    output.branch_logits[name], batch.labels
                )
                for name in ("gin", "variation")
            ]
        ).mean()
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.static_projection.parameters()
            )
        )
        for name in ("gin", "variation"):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in model.branch_classifiers[name].parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0, name)
            self.assertIsNotNone(
                model.residual_gate_logits[name].grad
            )

    def test_residual_attention_is_zero_output_and_trainable(self):
        torch.manual_seed(751)
        common = dict(
            gin_hidden_dim=8,
            attention_hidden_dim=4,
            channel_projection_dim=4,
            fusion_hidden_dim=4,
            dropout=0.0,
            message_mode="signed_normalized",
            pooling="mean_std",
            gin_residual=True,
            gin_jumping_knowledge=True,
            gin_compact_readout=True,
            gin_batch_normalization=True,
        )
        v1a = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_static_anchor_residual",
                **common
            )
        )
        v1b = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant=(
                    "signed_gin_static_anchor_residual_attention"
                ),
                gin_residual_attention=True,
                **common
            )
        )
        v1a.reset_residual_fusion_parameters(42)
        v1b.reset_residual_fusion_parameters(42)
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        v1a.eval()
        v1b.eval()
        output_a = v1a(batch)
        output_b = v1b(batch)
        self.assertTrue(
            torch.equal(
                output_a.branch_logits["static_spectral"],
                output_b.branch_logits["static_spectral"],
            )
        )
        self.assertTrue(
            torch.equal(
                output_a.gin_representation,
                output_b.gin_representation,
            )
        )
        self.assertTrue(torch.equal(output_a.logits, output_b.logits))
        self.assertLess(
            float(output_b.residual_gates["attention"]), 0.01
        )

        v1b.set_training_stage("residual_experts")
        with torch.no_grad():
            v1b.encoder.attention_residual_projection[-1].weight.fill_(
                0.1
            )
            v1b.branch_classifiers["gin"][-1].weight.fill_(0.1)
        loss = torch.nn.functional.cross_entropy(
            v1b(batch).logits, batch.labels
        )
        loss.backward()
        attention_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in v1b.encoder.attention.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(attention_gradient, 0.0)

    def test_classification_gradient_reaches_signed_gin_and_attention(self):
        torch.manual_seed(727)
        model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_variation", dropout=0.0
            )
        )
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        loss = torch.nn.functional.cross_entropy(
            model(batch).logits, batch.labels
        )
        loss.backward()
        gin_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        attention_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.attention.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gin_gradient, 0.0)
        self.assertGreater(attention_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
