from __future__ import absolute_import, division, print_function

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from keysubgraph.features.hard_graph_cache import (
    CachedHardWindow,
    HardGraphSampleCache,
)
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.features.multiview_critical import (
    MultiViewCriticalBatch,
    MultiViewCriticalFeatureBuilder,
    SignedSpectralInvariantBuilder,
    unbalanced_sinkhorn,
)
from keysubgraph.models.multiview_critical import (
    MultiViewCriticalClassifier,
    MultiViewCriticalConfig,
    MultiViewCriticalShortTermFusion,
    multiview_critical_loss,
)
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.multiview_critical import (
    MultiViewCriticalDataset,
    MultiViewCriticalRecord,
    fit_multiview_scaler,
    load_multiview_record,
    save_multiview_record,
    save_multiview_scaler,
    write_multiview_manifest,
)
from keysubgraph.training.multiview_critical_trainer import (
    MultiViewTrainingConfig,
    load_multiview_checkpoint,
    train_multiview_critical,
)
from keysubgraph.theory.spectral_gw import DifferentiableGWLoss


def _window(adjacency, time_start):
    count = int(adjacency.shape[0])
    return HardGraphWindow(
        adjacency=adjacency,
        communities=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        node_names=tuple("roi{}".format(index) for index in range(count)),
        node_ids=tuple("roi{}".format(index) for index in range(count)),
        time_start=float(time_start),
        edge_presence_threshold=0.0,
        window_valid=True,
    )


def _cache(label=1):
    first = torch.tensor(
        [
            [0.0, 0.7, -0.2, 0.0],
            [0.7, 0.0, 0.1, -0.3],
            [-0.2, 0.1, 0.0, -0.8],
            [0.0, -0.3, -0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    second = torch.tensor(
        [
            [0.0, 0.6, -0.1, 0.0],
            [0.6, 0.0, 0.2, -0.4],
            [-0.1, 0.2, 0.0, -0.7],
            [0.0, -0.4, -0.7, 0.0],
        ],
        dtype=torch.float32,
    )
    windows = (_window(first, 0.0), _window(second, 1.0))
    cached = tuple(CachedHardWindow(item, None, ()) for item in windows)
    return HardGraphSampleCache(
        sample_key="site/sample{}".format(label),
        sample_id="sample{}".format(label),
        label=int(label),
        split="train",
        windows=cached,
        time_values=(0.0, 1.0),
        time_mask=(True, True),
        eligible_for_stage_c=True,
        exclusion_reason=None,
        data_protocol_sha256="protocol",
        teacher_checkpoint_sha256="selector",
    ), windows


class _FakeAuthorBatch(object):
    def __init__(self, sample_keys, values):
        self.sample_keys = sample_keys
        self.values = values


class _FakeAuthor(nn.Module):
    def __init__(self, representation_dim=12):
        super().__init__()
        self.projection = nn.Linear(3, representation_dim)
        self.classifier_norm = nn.LayerNorm(representation_dim)
        self.classifier = nn.Linear(representation_dim, 1)

    def forward(self, batch):
        representation = self.projection(batch.values)
        logits = self.classifier(self.classifier_norm(representation)).squeeze(-1)
        return SimpleNamespace(final_representation=representation, logits=logits)


class _TinyLoader(list):
    def __init__(self, batches, labels):
        super().__init__(batches)
        self.dataset = SimpleNamespace(labels=tuple(labels))


class MultiViewCriticalTest(unittest.TestCase):
    def test_spectral_features_are_eigenvector_sign_invariant_by_construction(self):
        cache, windows = _cache()
        builder = SignedSpectralInvariantBuilder()
        first = builder(windows[0].adjacency)
        second = builder(windows[0].adjacency.clone())
        self.assertEqual(tuple(first.shape), (4, 9))
        self.assertTrue(torch.allclose(first, second, atol=1.0e-7))
        self.assertTrue(torch.isfinite(first).all())

    def test_uot_supports_unequal_object_counts(self):
        cost = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.2, 0.5]])
        plan = unbalanced_sinkhorn(
            cost,
            torch.tensor([0.6, 0.4]),
            torch.tensor([0.2, 0.5, 0.3]),
            iterations=20,
        )
        self.assertEqual(tuple(plan.shape), (2, 3))
        self.assertTrue(torch.isfinite(plan).all())
        self.assertGreater(float(plan.sum()), 0.0)

    def test_fused_gw_uses_node_attributes_and_signed_profiles(self):
        distance = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        solver = DifferentiableGWLoss(max_iter=20, sinkhorn_iter=20)
        structural = solver(distance, distance)
        feature_cost = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
        fused = solver(
            distance, distance, feature_cost=feature_cost, feature_weight=0.25
        )
        self.assertAlmostEqual(float(structural.distance), 0.0, places=7)
        self.assertGreater(float(fused.distance), 0.0)

        cache, full = _cache()
        builder = MultiViewCriticalFeatureBuilder(uot_iterations=5)
        first = builder.build(cache, full_graph_windows=full)
        self.assertTrue(torch.isfinite(first.transitions[0].object_cost).all())
        self.assertGreater(float(first.transitions[0].object_cost.sum()), 0.0)

    def test_delta_q_decoder_target_has_uniform_rate_units(self):
        cache, windows = _cache()
        delayed = (
            windows[0],
            _window(windows[1].adjacency, 2.0),
        )
        delayed_cache = HardGraphSampleCache(
            sample_key=cache.sample_key,
            sample_id=cache.sample_id,
            label=cache.label,
            split=cache.split,
            windows=tuple(CachedHardWindow(item, None, ()) for item in delayed),
            time_values=(0.0, 2.0),
            time_mask=(True, True),
            eligible_for_stage_c=True,
            exclusion_reason=None,
            data_protocol_sha256=cache.data_protocol_sha256,
            teacher_checkpoint_sha256=cache.teacher_checkpoint_sha256,
        )
        builder = MultiViewCriticalFeatureBuilder(uot_iterations=5)
        raw = builder.theory.build(delayed, delayed_cache.time_values).transition_features[0]
        built = builder.build(delayed_cache, full_graph_windows=delayed)
        target = built.transitions[0].delta_q_target
        self.assertTrue(torch.allclose(target[:16], raw[:16] / 2.0, atol=1.0e-7))
        self.assertTrue(torch.allclose(target[16:], raw[16:], atol=1.0e-7))

    def test_signed_features_and_graph_output_survive_node_permutation(self):
        cache, windows = _cache(1)
        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        permuted_windows = []
        for window in windows:
            adjacency = window.adjacency.index_select(0, permutation).index_select(1, permutation)
            permuted_windows.append(
                HardGraphWindow(
                    adjacency=adjacency,
                    communities=window.communities.index_select(0, permutation),
                    node_names=tuple(window.node_names[index] for index in permutation.tolist()),
                    node_ids=tuple(window.node_ids[index] for index in permutation.tolist()),
                    time_start=window.time_start,
                    edge_presence_threshold=window.edge_presence_threshold,
                    window_valid=True,
                )
            )
        permuted_cache = HardGraphSampleCache(
            sample_key=cache.sample_key,
            sample_id=cache.sample_id,
            label=cache.label,
            split=cache.split,
            windows=tuple(CachedHardWindow(item, None, ()) for item in permuted_windows),
            time_values=cache.time_values,
            time_mask=cache.time_mask,
            eligible_for_stage_c=True,
            exclusion_reason=None,
            data_protocol_sha256=cache.data_protocol_sha256,
            teacher_checkpoint_sha256=cache.teacher_checkpoint_sha256,
        )
        builder = MultiViewCriticalFeatureBuilder(uot_iterations=10)
        original = builder.build(cache, full_graph_windows=windows)
        permuted = builder.build(permuted_cache, full_graph_windows=tuple(permuted_windows))
        self.assertLess(float(original.hard_windows[0].adjacency.min()), 0.0)
        self.assertLess(float(permuted.hard_windows[0].adjacency.min()), 0.0)
        model = MultiViewCriticalClassifier(
            MultiViewCriticalConfig(
                hidden_dim=8, static_layers=2, object_layers=2,
                full_layers=2, dropout=0.0, classifier_hidden_dim=4,
            )
        ).eval()
        with torch.no_grad():
            first = model(MultiViewCriticalBatch((original,))).logits
            second = model(MultiViewCriticalBatch((permuted,))).logits
        self.assertTrue(torch.allclose(first, second, atol=1.0e-5, rtol=1.0e-5))

    def test_complete_model_forward_backward_and_fusion(self):
        builder = MultiViewCriticalFeatureBuilder()
        cache_a, full_a = _cache(0)
        cache_b, full_b = _cache(1)
        sample_a = builder.build(cache_a, full_graph_windows=full_a)
        sample_b = builder.build(cache_b, full_graph_windows=full_b)
        batch = MultiViewCriticalBatch((sample_a, sample_b))
        config = MultiViewCriticalConfig(
            hidden_dim=16,
            static_layers=2,
            object_layers=2,
            full_layers=2,
            classifier_hidden_dim=8,
        )
        model = MultiViewCriticalClassifier(config)
        self.assertIsNot(model.static_encoder, model.object_encoder)
        output = model(batch)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertFalse(output.diagnostics["static_and_object_encoders_share_parameters"])
        self.assertFalse(output.diagnostics["uses_g_decoder"])
        losses = multiview_critical_loss(output, batch.labels)
        losses["loss"].backward()
        self.assertIsNotNone(model.static_encoder.positive_initial[0].weight.grad)
        self.assertIsNotNone(model.static_encoder.layers[0].edge_gate.weight.grad)
        self.assertIsNotNone(model.object_encoder.positive_initial[0].weight.grad)
        self.assertIsNotNone(model.object_context_projection[0].weight.grad)
        self.assertIsNotNone(model.q_decoder.weight.grad)
        self.assertIsNotNone(model.delta_q_decoder.weight.grad)

        author = _FakeAuthor()
        fusion = MultiViewCriticalShortTermFusion(model, author)
        author_batch = _FakeAuthorBatch(
            batch.sample_keys, torch.randn(2, 3)
        )
        fused = fusion(batch, author_batch)
        author_only = author(author_batch).logits
        fused_binary = fused.logits[:, 1] - fused.logits[:, 0]
        self.assertTrue(torch.allclose(fused_binary, author_only, atol=1.0e-6))
        self.assertEqual(tuple(fused.logits.shape), (2, 2))

    def test_artifact_scaler_and_train_provenance_round_trip(self):
        builder = MultiViewCriticalFeatureBuilder(uot_iterations=5)
        records = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for label in (0, 1):
                cache, full = _cache(label)
                record = MultiViewCriticalRecord(
                    sample_id=cache.sample_id,
                    subject_id=cache.sample_id,
                    site="site",
                    split="train",
                    features=builder.build(cache, full_graph_windows=full),
                    protocol_sha256="protocol",
                    selector_checkpoint_sha256="selector",
                    feature_schema_sha256="schema",
                )
                path = root / "{}.pt".format(label)
                save_multiview_record(record, path)
                with self.assertRaises(FileExistsError):
                    save_multiview_record(record, path)
                paths.append(path)
                records.append(load_multiview_record(path))
            manifest = write_multiview_manifest(
                paths, root / "manifest.json", root
            )
            scaler = fit_multiview_scaler(records, file_sha256(manifest))
            scaler_path = save_multiview_scaler(scaler, root / "scaler.pt")
            dataset = MultiViewCriticalDataset(root, manifest, scaler_path)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.labels, (0, 1))
            self.assertTrue(torch.isfinite(dataset[0].stable_static).all())

    def test_checkpoint_training_and_strict_reload(self):
        builder = MultiViewCriticalFeatureBuilder(uot_iterations=5)
        samples = []
        for label in (0, 1):
            cache, full = _cache(label)
            samples.append(builder.build(cache, full_graph_windows=full))
        batch = MultiViewCriticalBatch(tuple(samples))
        loader = _TinyLoader([batch], (0, 1))
        config = MultiViewCriticalConfig(
            hidden_dim=8, static_layers=1, object_layers=1,
            full_layers=1, dropout=0.0, classifier_hidden_dim=4,
        )
        model = MultiViewCriticalClassifier(config)
        with tempfile.TemporaryDirectory() as directory:
            result = train_multiview_critical(
                model, loader, loader, torch.device("cpu"), directory,
                MultiViewTrainingConfig(
                    epochs=1, early_stopping_patience=0,
                    max_train_batches=1, max_validation_batches=1,
                ),
            )
            checkpoint = Path(result["best_checkpoint"])
            self.assertTrue(checkpoint.is_file())
            restored = MultiViewCriticalClassifier(config)
            payload = load_multiview_checkpoint(checkpoint, restored, torch.device("cpu"))
            self.assertEqual(payload["epoch"], 1)

            single_class_batch = MultiViewCriticalBatch((samples[0], samples[0]))
            # The dataset inventory remains a valid two-class training cohort,
            # while a one-batch smoke limit may expose only one class.
            single_class_loader = _TinyLoader([single_class_batch], (0, 1))
            fallback = train_multiview_critical(
                MultiViewCriticalClassifier(config),
                single_class_loader,
                single_class_loader,
                torch.device("cpu"),
                Path(directory) / "single_class",
                MultiViewTrainingConfig(
                    epochs=1, early_stopping_patience=0,
                    max_train_batches=1, max_validation_batches=1,
                ),
            )
            self.assertIsNone(fallback["best_auc"])
            self.assertTrue(Path(fallback["best_checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
