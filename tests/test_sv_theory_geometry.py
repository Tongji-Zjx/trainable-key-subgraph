from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.sv_signed_gin_artifact import (
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
    save_sv_signed_gin_record,
)
from keysubgraph.data.sv_signed_gin_manifest import (
    write_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (
    fit_sv_signed_gin_standardizers,
    save_sv_signed_gin_standardizers,
)
from keysubgraph.data.sv_signed_gin_dataset import (
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_theory_geometry import (
    SVTheoryAugmentedDataset,
    build_sv_theory_feature_payload,
    fit_sv_theory_feature_standardizer,
    load_sv_theory_feature_payload,
    save_sv_theory_feature_payload,
    save_sv_theory_feature_standardizer,
)
from keysubgraph.features.sv_theory_geometry import (
    SV_DIFFUSION_GEOMETRY_DIM,
    SV_SPECTRAL_DIRECTION_DIM,
    SVTheoryGeometryExtractor,
)
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    run_sv_signed_gin_epoch,
)


def _adjacency(offset):
    matrix = torch.tensor(
        (
            (0.0, 0.6, -0.2, 0.0),
            (0.6, 0.0, 0.3, -0.1),
            (-0.2, 0.3, 0.0, 0.4),
            (0.0, -0.1, 0.4, 0.0),
        ),
        dtype=torch.float32,
    )
    if offset:
        matrix = matrix.clone()
        matrix[0, 3] = float(offset)
        matrix[3, 0] = float(offset)
        matrix[1, 2] = matrix[1, 2] - 0.12
        matrix[2, 1] = matrix[1, 2]
    return matrix


def _window(adjacency, time_start):
    node_count = int(adjacency.shape[0])
    return SVSignedGINWindowRecord(
        node_features=torch.arange(
            node_count * 15, dtype=torch.float32
        ).reshape(node_count, 15),
        adjacency=adjacency,
        time_start=float(time_start),
    )


def _record(key, label, split, offset):
    first = _adjacency(0.0)
    second = _adjacency(offset)
    return SVSignedGINRecord(
        sample_key=key,
        sample_id=key,
        subject_id="subject-" + key,
        site="site-a",
        label=int(label),
        split=split,
        windows=(_window(first, 0.0), _window(second, 1.0)),
        static_features=(
            torch.linspace(0.0, 1.0, 28) + float(offset)
        ),
        variation=(
            torch.linspace(0.0, 0.5, 16) + float(offset)
        ),
        window_mask=torch.tensor((True, True)),
        transition_mask=torch.tensor((True,)),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


def _write_manifest(root, records, name):
    pairs = []
    artifact_dir = root / (name + "_artifacts")
    for record in records:
        path = artifact_dir / (record.sample_key + ".pt")
        save_sv_signed_gin_record(record, path)
        pairs.append((record, path))
    manifest = root / name / "manifest.json"
    write_sv_signed_gin_manifest(pairs, manifest)
    return manifest


class SVTheoryGeometryTest(unittest.TestCase):
    def test_direction_is_signed_and_diffusion_is_permutation_invariant(self):
        extractor = SVTheoryGeometryExtractor()
        first = _window(_adjacency(0.0), 0.0)
        second = _window(_adjacency(-0.35), 1.0)
        forward = extractor.build((first, second))
        backward = extractor.build((second, first))
        self.assertEqual(
            tuple(forward.spectral_direction.shape),
            (SV_SPECTRAL_DIRECTION_DIM,),
        )
        self.assertEqual(
            tuple(forward.diffusion_geometry.shape),
            (SV_DIFFUSION_GEOMETRY_DIM,),
        )
        self.assertTrue(
            torch.allclose(
                forward.spectral_direction,
                -backward.spectral_direction,
                atol=1.0e-6,
            )
        )
        self.assertGreater(
            float(forward.spectral_direction.abs().sum()), 0.0
        )

        permutation = torch.tensor((2, 0, 3, 1))
        permuted = tuple(
            _window(
                window.adjacency.index_select(
                    0, permutation
                ).index_select(1, permutation),
                index,
            )
            for index, window in enumerate((first, second))
        )
        permuted_features = extractor.build(permuted)
        self.assertTrue(
            torch.allclose(
                forward.spectral_direction,
                permuted_features.spectral_direction,
                atol=2.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                forward.diffusion_geometry,
                permuted_features.diffusion_geometry,
                atol=2.0e-6,
            )
        )

    def test_missing_window_breaks_transition_and_is_ignored(self):
        extractor = SVTheoryGeometryExtractor()
        first = _window(_adjacency(0.0), 0.0)
        second = _window(_adjacency(-0.35), 2.0)
        features = extractor.build((first, None, second))
        self.assertTrue(
            torch.equal(
                features.window_mask,
                torch.tensor((True, False, True)),
            )
        )
        self.assertTrue(
            torch.equal(
                features.transition_mask,
                torch.tensor((False, False)),
            )
        )
        self.assertTrue(
            torch.equal(
                features.spectral_direction,
                torch.zeros(SV_SPECTRAL_DIRECTION_DIM),
            )
        )
        expected = torch.stack(
            (
                extractor.build((first,)).diffusion_geometry,
                extractor.build((second,)).diffusion_geometry,
            )
        ).mean(dim=0)
        self.assertTrue(
            torch.allclose(
                features.diffusion_geometry, expected, atol=1.0e-6
            )
        )

    def test_negative_edge_is_present_and_not_replaced_by_magnitude(self):
        extractor = SVTheoryGeometryExtractor()
        negative = _adjacency(-0.35)
        positive = negative.clone()
        positive[0, 3] = 0.35
        positive[3, 0] = 0.35
        absent = negative.clone()
        absent[0, 3] = 0.0
        absent[3, 0] = 0.0
        negative_state = extractor.build(
            (_window(negative, 0.0),)
        ).diffusion_geometry
        positive_state = extractor.build(
            (_window(positive, 0.0),)
        ).diffusion_geometry
        absent_state = extractor.build(
            (_window(absent, 0.0),)
        ).diffusion_geometry
        self.assertFalse(
            torch.allclose(negative_state, positive_state)
        )
        self.assertFalse(
            torch.allclose(negative_state, absent_state)
        )

    def test_sidecar_roundtrip_train_only_scaling_and_dataset_join(self):
        train_records = (
            _record("train-a", 0, "train", -0.25),
            _record("train-b", 1, "train", 0.35),
        )
        validation_records = (
            _record("validation-a", 1, "validation", 0.15),
        )
        extractor = SVTheoryGeometryExtractor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = _write_manifest(
                root, train_records, "train"
            )
            validation_manifest = _write_manifest(
                root, validation_records, "validation"
            )
            base_scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            base_scaler_path = root / "base_scaler.json"
            save_sv_signed_gin_standardizers(
                base_scaler, base_scaler_path
            )

            train_payload = build_sv_theory_feature_payload(
                train_records,
                file_sha256(train_manifest),
                extractor,
            )
            validation_payload = build_sv_theory_feature_payload(
                validation_records,
                file_sha256(validation_manifest),
                extractor,
            )
            train_cache = root / "train_theory.pt"
            validation_cache = root / "validation_theory.pt"
            save_sv_theory_feature_payload(
                train_payload, train_cache
            )
            save_sv_theory_feature_payload(
                validation_payload, validation_cache
            )
            scaler = fit_sv_theory_feature_standardizer(
                train_payload, file_sha256(train_cache)
            )
            theory_scaler = root / "theory_scaler.json"
            save_sv_theory_feature_standardizer(
                scaler, theory_scaler
            )
            train = SVTheoryAugmentedDataset(
                train_manifest,
                base_scaler_path,
                train_cache,
                theory_scaler,
            )
            validation = SVTheoryAugmentedDataset(
                validation_manifest,
                base_scaler_path,
                validation_cache,
                theory_scaler,
            )
            loader = create_sv_signed_gin_loader(
                train,
                batch_size=2,
                seed=42,
                shuffle=False,
            )
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant=(
                        "signed_gin_multibranch_theory_geometry"
                    ),
                    gin_hidden_dim=8,
                    attention_hidden_dim=4,
                    channel_projection_dim=4,
                    fusion_hidden_dim=4,
                    dropout=0.0,
                )
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            epoch = run_sv_signed_gin_epoch(
                model,
                loader,
                torch.device("cpu"),
                torch.ones(2),
                optimizer=optimizer,
                auxiliary_loss_weight=0.25,
            )
            reloaded = load_sv_theory_feature_payload(train_cache)

        self.assertEqual(reloaded["sample_count"], 2)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        direction = torch.stack(
            [sample.spectral_direction for sample in train.samples]
        )
        diffusion = torch.stack(
            [sample.diffusion_geometry for sample in train.samples]
        )
        self.assertTrue(
            torch.allclose(
                direction.mean(dim=0),
                torch.zeros(SV_SPECTRAL_DIRECTION_DIM),
                atol=1.0e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                diffusion.mean(dim=0),
                torch.zeros(SV_DIFFUSION_GEOMETRY_DIM),
                atol=1.0e-5,
            )
        )
        self.assertTrue(
            bool(
                torch.isfinite(
                    validation.samples[0].spectral_direction
                ).all()
            )
        )
        self.assertEqual(epoch["sample_count"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(epoch["loss"])))


if __name__ == "__main__":
    unittest.main()
