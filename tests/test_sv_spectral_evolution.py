from __future__ import absolute_import, division, print_function

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, Dataset

from keysubgraph.data.sv_signed_gin_artifact import (
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
)
from keysubgraph.data.sv_spectral_evolution import (
    extract_spectral_transition_segments,
    fit_sv_spectral_transition_standardizer,
    spectral_quantile_grid,
)
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.models.sv_spectral_evolution import (
    SVSpectralEvolutionBatch,
    SVSpectralEvolutionClassifier,
    SVSpectralEvolutionSampleInput,
)
from keysubgraph.training.sv_spectral_evolution_trainer import (
    SVSpectralEvolutionTrainingConfig,
    load_sv_spectral_evolution_checkpoint,
    train_sv_spectral_evolution_classifier,
)
from keysubgraph.theory.spectral_gw import (
    SignedLaplacianBuilder,
    SpectralStateExtractor,
)


def _window(adjacency, time_start):
    adjacency = torch.tensor(adjacency, dtype=torch.float32)
    return SVSignedGINWindowRecord(
        node_features=torch.zeros(
            adjacency.shape[0], 15, dtype=torch.float32
        ),
        adjacency=adjacency,
        time_start=float(time_start),
    )


def _state(window):
    edge_mask = window.adjacency.abs() > 0.0
    edge_mask.fill_diagonal_(False)
    laplacian = SignedLaplacianBuilder(1.0e-3)(
        window.adjacency, edge_mask=edge_mask
    )
    return SpectralStateExtractor(spectral_quantile_grid())(
        laplacian
    ).quantiles


def _record(split="train"):
    windows = (
        _window(
            [[0.0, 0.5, -0.2], [0.5, 0.0, 0.1], [-0.2, 0.1, 0.0]],
            0,
        ),
        _window(
            [[0.0, 0.4, -0.3], [0.4, 0.0, 0.2], [-0.3, 0.2, 0.0]],
            1,
        ),
        None,
        _window(
            [[0.0, 0.2, -0.4], [0.2, 0.0, 0.3], [-0.4, 0.3, 0.0]],
            3,
        ),
        _window(
            [[0.0, 0.1, -0.5], [0.1, 0.0, 0.4], [-0.5, 0.4, 0.0]],
            4,
        ),
    )
    deltas = (
        (_state(windows[1]) - _state(windows[0])).abs(),
        (_state(windows[4]) - _state(windows[3])).abs(),
    )
    return SVSignedGINRecord(
        sample_key="site/sample",
        sample_id="sample",
        subject_id="subject",
        site="site",
        label=1,
        split=split,
        windows=windows,
        static_features=torch.zeros(28),
        variation=torch.stack(deltas).mean(dim=0),
        window_mask=torch.tensor(
            [True, True, False, True, True], dtype=torch.bool
        ),
        transition_mask=torch.tensor(
            [True, False, False, True], dtype=torch.bool
        ),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


class SVSpectralEvolutionTest(unittest.TestCase):
    def test_signed_transitions_reconstruct_variation_and_split_gaps(self):
        record = _record()
        segments = extract_spectral_transition_segments(record)
        self.assertEqual(len(segments), 2)
        self.assertEqual([item.shape[0] for item in segments], [1, 1])
        self.assertEqual(tuple(segments[0].shape), (1, 32))
        reconstructed = torch.cat(segments, dim=0)[:, 16:].mean(dim=0)
        self.assertTrue(
            torch.allclose(reconstructed, record.variation, atol=5.0e-5)
        )
        self.assertTrue(
            torch.allclose(
                segments[0][:, :16].abs(), segments[0][:, 16:]
            )
        )

    def test_transition_scaler_is_train_only_and_finite(self):
        record = _record()
        scaler = fit_sv_spectral_transition_standardizer(
            [record], "manifest"
        )
        self.assertEqual(scaler.train_sample_count, 1)
        self.assertEqual(scaler.train_transition_count, 2)
        values = torch.cat(
            extract_spectral_transition_segments(record), dim=0
        )
        standardized = scaler.standardize(values)
        self.assertTrue(bool(torch.isfinite(standardized).all()))
        with self.assertRaises(ValueError):
            fit_sv_spectral_transition_standardizer(
                [_record(split="validation")], "manifest"
            )

    def test_zero_initialized_residual_exactly_matches_frozen_anchor(self):
        torch.manual_seed(7)
        anchor = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="static_spectral_only", dropout=0.0
            )
        )
        model = SVSpectralEvolutionClassifier(anchor)
        samples = tuple(
            SVSpectralEvolutionSampleInput(
                sample_key="sample_{}".format(index),
                label=index,
                static_features=torch.randn(28),
                transition_segments=(torch.randn(3 + index, 32),),
            )
            for index in (0, 1)
        )
        batch = SVSpectralEvolutionBatch(samples)
        model.train()
        output = model(batch)
        with torch.no_grad():
            static = torch.stack(
                [sample.static_features for sample in samples], dim=0
            )
            anchor_logits = anchor.branch_classifiers[
                "static_spectral"
            ](anchor.static_projection(static[:, :16]))
        self.assertTrue(torch.equal(output.logits, anchor_logits))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.static_anchor.parameters()
            )
        )
        labels = batch.labels
        loss = torch.nn.functional.cross_entropy(
            output.logits, labels
        ) + 0.25 * torch.nn.functional.cross_entropy(
            output.dynamic_logits, labels
        )
        loss.backward()
        self.assertIsNotNone(
            model.dynamic_classifier[-1].weight.grad
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.static_anchor.parameters()
            )
        )

    def test_training_saves_thresholds_and_reloadable_checkpoint(self):
        class TinyDataset(Dataset):
            def __init__(self):
                self.samples = tuple(
                    SVSpectralEvolutionSampleInput(
                        sample_key="sample_{}".format(index),
                        label=index % 2,
                        static_features=torch.full(
                            (28,), float(index % 2)
                        ),
                        transition_segments=(
                            torch.full(
                                (3, 32), float(index % 2)
                            ),
                        ),
                    )
                    for index in range(4)
                )
                self.sites = ("site",) * 4

            @property
            def sample_keys(self):
                return tuple(item.sample_key for item in self.samples)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index]

        dataset = TinyDataset()
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda values: SVSpectralEvolutionBatch(
                tuple(values)
            ),
        )
        anchor = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="static_spectral_only", dropout=0.0
            )
        )
        model = SVSpectralEvolutionClassifier(anchor)
        provenance = {"purpose": "unit_test"}
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = train_sv_spectral_evolution_classifier(
                model,
                loader,
                loader,
                [0, 1, 0, 1],
                torch.device("cpu"),
                SVSpectralEvolutionTrainingConfig(
                    epochs=1,
                    selection_metric="roc_auc",
                    early_stopping_patience=0,
                    seed=9,
                ),
                output,
                provenance,
            )
            self.assertEqual(result["epochs_completed"], 1)
            payload = load_sv_spectral_evolution_checkpoint(
                output / "best_checkpoint.pt",
                model,
                torch.device("cpu"),
                expected_provenance=provenance,
            )
            self.assertIn(
                "balanced_accuracy",
                payload["validation_thresholds"],
            )
            self.assertTrue(
                (output / "best_evaluation.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
