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
from keysubgraph.data.sv_spectral_diffusion import (
    SVSpectralDiffusionAugmentedDataset,
    build_sv_spectral_diffusion_record,
    fit_sv_spectral_diffusion_standardizer,
    load_sv_spectral_diffusion_record,
    save_sv_spectral_diffusion_record,
    save_sv_spectral_diffusion_standardizer,
    write_sv_spectral_diffusion_manifest,
)
from keysubgraph.features.sv_spectral_diffusion import (
    SV_HKS_DIM,
    SVSpectralDiffusionExtractor,
    exact_heat_diffusion_message,
)
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINBatch,
    SVSignedGINClassifier,
    SVSignedGINConfig,
)
from keysubgraph.training.sv_signed_gin_trainer import (
    run_sv_signed_gin_epoch,
)


def _adjacency(offset=0.0):
    value = torch.tensor(
        (
            (0.0, 0.6, -0.2, 0.0),
            (0.6, 0.0, 0.3, -0.1),
            (-0.2, 0.3, 0.0, 0.4),
            (0.0, -0.1, 0.4, 0.0),
        ),
        dtype=torch.float32,
    )
    value = value.clone()
    value[0, 3] = float(offset)
    value[3, 0] = float(offset)
    return value


def _base_record(key, label, split, offset):
    windows = []
    for position, adjacency in enumerate(
        (_adjacency(0.0), _adjacency(offset))
    ):
        windows.append(
            SVSignedGINWindowRecord(
                node_features=(
                    torch.arange(60, dtype=torch.float32).reshape(4, 15)
                    / 50.0
                    + float(offset)
                ),
                adjacency=adjacency,
                time_start=float(position),
            )
        )
    return SVSignedGINRecord(
        sample_key=key,
        sample_id=key,
        subject_id="subject-" + key,
        site="site-a",
        label=int(label),
        split=split,
        windows=tuple(windows),
        static_features=torch.linspace(0.0, 1.0, 28) + float(offset),
        variation=torch.linspace(0.0, 0.5, 16) + abs(float(offset)),
        window_mask=torch.tensor((True, True)),
        transition_mask=torch.tensor((True,)),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


def _base_manifest(root, name, records):
    pairs = []
    for record in records:
        path = root / name / "base" / (record.sample_key + ".pt")
        save_sv_signed_gin_record(record, path)
        pairs.append((record, path))
    path = root / name / "manifest.json"
    write_sv_signed_gin_manifest(pairs, path)
    return path, pairs


def _spectral_manifest(root, name, base_manifest, base_pairs):
    extractor = SVSpectralDiffusionExtractor()
    pairs = []
    for source, source_path in base_pairs:
        record = build_sv_spectral_diffusion_record(
            source,
            file_sha256(source_path),
            file_sha256(base_manifest),
            extractor,
        )
        path = root / name / "spectral" / (source.sample_key + ".pt")
        save_sv_spectral_diffusion_record(record, path)
        pairs.append((record, path))
    path = root / name / "spectral_manifest.json"
    write_sv_spectral_diffusion_manifest(pairs, path)
    return path, pairs


class SVSpectralDiffusionTest(unittest.TestCase):
    def test_hks_is_permutation_equivariant_and_heat_message_is_exact(self):
        extractor = SVSpectralDiffusionExtractor()
        adjacency = _adjacency(-0.25)
        state = extractor.build_window(adjacency)
        permutation = torch.tensor((2, 0, 3, 1))
        permuted = extractor.build_window(
            adjacency.index_select(0, permutation).index_select(1, permutation)
        )
        self.assertEqual(tuple(state.hks.shape), (4, SV_HKS_DIM))
        self.assertTrue(
            torch.allclose(
                state.hks.index_select(0, permutation),
                permuted.hks,
                atol=2.0e-5,
            )
        )
        nodes = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7.0
        actual = exact_heat_diffusion_message(
            nodes, state.eigenvalues, state.eigenvectors, 0.5
        )
        laplacian = (
            state.eigenvectors
            .matmul(torch.diag(state.eigenvalues))
            .matmul(state.eigenvectors.transpose(0, 1))
        )
        expected = torch.matrix_exp(-0.5 * laplacian).matmul(nodes)
        self.assertTrue(torch.allclose(actual, expected, atol=2.0e-5))

    def test_sidecar_roundtrip_train_only_scaler_and_g2_gradient(self):
        train_records = (
            _base_record("train-a", 0, "train", -0.25),
            _base_record("train-b", 1, "train", 0.35),
        )
        validation_records = (
            _base_record("validation-a", 1, "validation", 0.15),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest, train_pairs = _base_manifest(
                root, "train", train_records
            )
            validation_manifest, validation_pairs = _base_manifest(
                root, "validation", validation_records
            )
            base_scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            base_scaler_path = root / "base_scaler.json"
            save_sv_signed_gin_standardizers(base_scaler, base_scaler_path)
            train_spectral, train_spectral_pairs = _spectral_manifest(
                root, "train", train_manifest, train_pairs
            )
            validation_spectral, _ = _spectral_manifest(
                root,
                "validation",
                validation_manifest,
                validation_pairs,
            )
            manifest_payload = {
                "split": "train",
                "source_manifest_sha256": file_sha256(train_manifest),
                "protocol_sha256": "protocol",
                "selector_checkpoint_sha256": "selector",
                "selection_mode": "learned",
                "selection_seed": 42,
            }
            scaler = fit_sv_spectral_diffusion_standardizer(
                manifest_payload,
                [record for record, _ in train_spectral_pairs],
                file_sha256(train_spectral),
            )
            spectral_scaler_path = root / "spectral_scaler.json"
            save_sv_spectral_diffusion_standardizer(
                scaler, spectral_scaler_path
            )
            train = SVSpectralDiffusionAugmentedDataset(
                train_manifest,
                base_scaler_path,
                train_spectral,
                spectral_scaler_path,
            )
            validation = SVSpectralDiffusionAugmentedDataset(
                validation_manifest,
                base_scaler_path,
                validation_spectral,
                spectral_scaler_path,
            )
            reloaded = load_sv_spectral_diffusion_record(
                train_spectral_pairs[0][1]
            )
            self.assertEqual(reloaded.sample_key, "train-a")
            all_hks = torch.cat(
                [window.hks for sample in train for window in sample.windows]
            )
            self.assertTrue(
                torch.allclose(
                    all_hks.mean(dim=0),
                    torch.zeros(SV_HKS_DIM),
                    atol=2.0e-4,
                ),
                str(all_hks.mean(dim=0)),
            )
            self.assertTrue(
                bool(torch.isfinite(validation[0].windows[0].hks).all())
            )
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant="svg_v2_c3_g2",
                    gin_hidden_dim=8,
                    attention_hidden_dim=4,
                    channel_projection_dim=4,
                    fusion_hidden_dim=4,
                    dropout=0.0,
                )
            )
            loader = torch.utils.data.DataLoader(
                train,
                batch_size=2,
                shuffle=False,
                collate_fn=lambda values: SVSignedGINBatch(tuple(values)),
            )
            batch = next(iter(loader))
            output = model(batch)
            auxiliary = torch.nn.functional.smooth_l1_loss(
                output.signed_delta_q_predictions,
                output.signed_delta_q_targets,
            )
            auxiliary.backward()
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in model.signed_delta_q_head.parameters()
                if parameter.grad is not None
            )
            model.zero_grad(set_to_none=True)
            metrics = run_sv_signed_gin_epoch(
                model,
                loader,
                torch.device("cpu"),
                torch.ones(2),
                signed_delta_q_weight=0.05,
            )
        self.assertGreater(metrics["signed_delta_q_loss"], 0.0)
        self.assertGreater(gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
