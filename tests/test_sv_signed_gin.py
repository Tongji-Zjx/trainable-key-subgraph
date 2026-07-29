from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
from keysubgraph.models.sv_signed_gin import (
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
    if permutation is not None:
        features = features.index_select(0, permutation)
        adjacency = adjacency.index_select(0, permutation).index_select(
            1, permutation
        )
    windows = (
        SVSignedGINWindowInput(features, adjacency),
        SVSignedGINWindowInput(features + 0.1, adjacency * 0.9),
    )
    return SVSignedGINSampleInput(
        sample_key=key,
        label=label,
        windows=windows,
        static_features=torch.linspace(0.0, 1.0, 28),
        variation=torch.linspace(0.1, 0.8, 16),
    )


class SVSignedGINTest(unittest.TestCase):
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

    def test_all_variants_have_expected_dimensions(self):
        batch = SVSignedGINBatch(
            (_sample("a", 0), _sample("b", 1))
        )
        for variant in SV_SIGNED_GIN_VARIANTS:
            model = SVSignedGINClassifier(
                SVSignedGINConfig(variant=variant, dropout=0.0)
            )
            output = model(batch)
            expected = (
                48
                if variant
                in (
                    "signed_gin_static_variation",
                    "signed_gin_multibranch_late_fusion",
                    "signed_gin_static_anchor_residual",
                )
                else 32
            )
            self.assertEqual(tuple(output.logits.shape), (2, 2))
            self.assertEqual(
                tuple(output.final_representation.shape), (2, expected)
            )
            self.assertEqual(
                output.diagnostics["preserves_signed_edges"], True
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
