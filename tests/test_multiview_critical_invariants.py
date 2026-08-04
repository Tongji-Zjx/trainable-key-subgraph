from __future__ import absolute_import, division, print_function

import unittest
from dataclasses import replace

import torch

from keysubgraph.features.multiview_critical import (
    MultiViewCriticalBatch,
    MultiViewCriticalFeatureBuilder,
    unbalanced_sinkhorn,
)
from keysubgraph.models.multiview_critical import (
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
    MultiViewCriticalShortTermFusion,
    SignedSpectralGCNIIEncoder,
    multiview_critical_loss,
)
from keysubgraph.training.multiview_critical_trainer import (
    load_multiview_checkpoint,
)
from tests.test_multiview_critical import _FakeAuthor, _FakeAuthorBatch, _cache


def _tiny_config(**updates):
    values = dict(
        hidden_dim=8,
        static_layers=1,
        object_layers=1,
        full_layers=1,
        dropout=0.0,
        classifier_hidden_dim=4,
    )
    values.update(updates)
    return MultiViewCriticalConfig(**values)


def _built(label=1):
    cache, full = _cache(label)
    return MultiViewCriticalFeatureBuilder(uot_iterations=10).build(
        cache, full_graph_windows=full
    )


class MultiViewCriticalInvariantTest(unittest.TestCase):
    def test_negative_edges_reach_signed_channel_and_isolated_nodes_are_finite(self):
        encoder = SignedSpectralGCNIIEncoder(
            input_dim=24,
            edge_feature_dim=6,
            hidden_dim=8,
            layers=1,
            dropout=0.0,
            alpha=0.1,
            theta=0.5,
            use_attention=True,
        ).eval()
        adjacency = torch.tensor(
            [[0.0, -0.7, 0.0], [-0.7, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        output = encoder(
            torch.randn(3, 15),
            torch.randn(3, 9),
            adjacency,
            torch.zeros(3, 3, 6),
        )
        diagnostics = encoder.layers[0].last_message_diagnostics
        self.assertGreater(diagnostics["negative_edge_message_norm"], 0.0)
        self.assertTrue(torch.isfinite(output.graph_embedding).all())
        self.assertTrue(torch.isfinite(output.node_states).all())

    def test_zero_residual_gates_recover_frozen_anchors_exactly(self):
        sample = _built()
        model = MultiViewCriticalClassifier(_tiny_config()).eval()
        with torch.no_grad():
            model.static_gate.zero_()
            model.g_gate.zero_()
            output = model(MultiViewCriticalBatch((sample,)))
            item = output.samples[0]
            stable = model.stable_projection(sample.stable_static)
            expected = stable + torch.tanh(model.v_gate) * model.v_projection(
                item.evolution_representation
            )
        self.assertTrue(torch.equal(item.static_representation, stable))
        self.assertTrue(torch.equal(item.representation, expected))

    def test_object_renumbering_and_batch_permutation_do_not_change_outputs(self):
        first, second = _built(0), _built(1)
        object_permutations = []
        windows = []
        for window in first.hard_windows:
            count = len(window.objects)
            permutation = torch.arange(count - 1, -1, -1)
            object_permutations.append(permutation)
            windows.append(
                replace(
                    window,
                    objects=tuple(window.objects[index] for index in permutation.tolist()),
                    object_coupling=window.object_coupling.index_select(
                        0, permutation
                    ).index_select(1, permutation),
                )
            )
        transitions = []
        for transition in first.transitions:
            if transition is None:
                transitions.append(None)
                continue
            left = object_permutations[transition.source_index]
            right = object_permutations[transition.target_index]
            transitions.append(
                replace(
                    transition,
                    object_cost=transition.object_cost.index_select(
                        0, left
                    ).index_select(1, right),
                    transport_plan=transition.transport_plan.index_select(
                        0, left
                    ).index_select(1, right),
                )
            )
        renumbered = replace(
            first,
            hard_windows=tuple(windows),
            transitions=tuple(transitions),
        )
        model = MultiViewCriticalClassifier(_tiny_config()).eval()
        with torch.no_grad():
            original = model(MultiViewCriticalBatch((first, second))).logits
            reordered = model(MultiViewCriticalBatch((second, renumbered))).logits
        self.assertTrue(torch.allclose(original[0], reordered[1], atol=1.0e-6))
        self.assertTrue(torch.allclose(original[1], reordered[0], atol=1.0e-6))

    def test_unequal_object_counts_and_missing_transition_are_mask_safe(self):
        sample = _built()
        right = sample.hard_windows[1]
        reduced_right = replace(
            right,
            objects=(right.objects[0],),
            object_coupling=right.object_coupling[:1, :1],
        )
        transition = sample.transitions[0]
        unequal = replace(
            sample,
            hard_windows=(sample.hard_windows[0], reduced_right),
            transitions=(
                replace(
                    transition,
                    object_cost=transition.object_cost[:, :1],
                    transport_plan=transition.transport_plan[:, :1],
                ),
            ),
        )
        missing = replace(
            sample,
            hard_windows=(sample.hard_windows[0], None),
            full_windows=(sample.full_windows[0], None),
            transitions=(None,),
            window_mask=torch.tensor((True, False)),
            transition_mask=torch.tensor((False,)),
        )
        model = MultiViewCriticalClassifier(_tiny_config(enable_g=False)).eval()
        with torch.no_grad():
            unequal_output = model(MultiViewCriticalBatch((unequal,)))
            missing_output = model(MultiViewCriticalBatch((missing,)))
        self.assertTrue(torch.isfinite(unequal_output.logits).all())
        self.assertTrue(torch.isfinite(missing_output.logits).all())
        self.assertEqual(missing_output.samples[0].q_predictions.shape[0], 1)
        losses = multiview_critical_loss(missing_output, missing_output.logits.new_tensor((1,), dtype=torch.long))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["delta_q_loss"]), 0.0)

    def test_transport_normalization_and_shuffled_control_are_deterministic(self):
        plan = unbalanced_sinkhorn(
            torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
            iterations=50,
        )
        self.assertGreater(float(plan.diag().sum()), float((plan.sum() - plan.diag().sum())))
        right = torch.randn(2, 5)
        aligned = plan.matmul(right) / plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        scaled = (7.0 * plan).matmul(right) / (7.0 * plan).sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        self.assertTrue(torch.allclose(aligned, scaled, atol=1.0e-7))

        sample = _built()
        model = MultiViewCriticalClassifier(
            _tiny_config(correspondence_mode="shuffled")
        ).eval()
        with torch.no_grad():
            first = model(MultiViewCriticalBatch((sample,))).logits
            second = model(MultiViewCriticalBatch((sample,))).logits
        self.assertTrue(torch.equal(first, second))

    def test_fusion_rejects_alignment_errors_and_is_batch_equivariant(self):
        first, second = _built(0), _built(1)
        model = MultiViewCriticalClassifier(_tiny_config()).eval()
        fusion = MultiViewCriticalShortTermFusion(model, _FakeAuthor()).eval()
        batch = MultiViewCriticalBatch((first, second))
        author = _FakeAuthorBatch(batch.sample_keys, torch.randn(2, 3))
        with torch.no_grad():
            output = fusion(batch, author).logits
            permuted = fusion(
                MultiViewCriticalBatch((second, first)),
                _FakeAuthorBatch(
                    (second.sample_key, first.sample_key), author.values.flip(0)
                ),
            ).logits
        self.assertTrue(torch.allclose(output, permuted.flip(0), atol=1.0e-6))
        with self.assertRaises(ValueError):
            fusion(batch, _FakeAuthorBatch(("wrong/a", "wrong/b"), author.values))

    def test_checkpoint_rejects_input_schema_mismatch(self):
        model = MultiViewCriticalClassifier(_tiny_config())
        payload = {
            "model_name": model.model_name,
            "model_config": model.config_dict(),
            "model_state_dict": model.state_dict(),
        }
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, str(path))
            mismatch = MultiViewCriticalClassifier(
                _tiny_config(hidden_dim=10, classifier_hidden_dim=5)
            )
            with self.assertRaises(ValueError):
                load_multiview_checkpoint(path, mismatch, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
