from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.sv_signed_gin_artifact import (
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
    load_sv_signed_gin_record,
    save_sv_signed_gin_record,
)
from keysubgraph.data.sv_signed_gin_dataset import (
    SVSignedGINDataset,
    create_sv_signed_gin_loader,
)
from keysubgraph.data.sv_signed_gin_manifest import (
    write_sv_signed_gin_manifest,
)
from keysubgraph.data.sv_signed_gin_scaler import (
    fit_sv_signed_gin_standardizers,
    save_sv_signed_gin_standardizers,
)


def _record(key, label, split, offset=0.0, node_count=3):
    node = (
        torch.arange(node_count * 15, dtype=torch.float32)
        .reshape(node_count, 15)
        / 20.0
        + float(offset)
    )
    adjacency = torch.zeros((node_count, node_count))
    for index in range(node_count - 1):
        value = 0.2 + 0.1 * index
        if index % 2:
            value = -value
        adjacency[index, index + 1] = value
        adjacency[index + 1, index] = value
    windows = (
        SVSignedGINWindowRecord(node, adjacency, 0.0),
        SVSignedGINWindowRecord(node + 0.1, adjacency * 0.8, 1.0),
    )
    return SVSignedGINRecord(
        sample_key=key,
        sample_id=key,
        subject_id="subject-" + key,
        site="site-a",
        label=label,
        split=split,
        windows=windows,
        static_features=torch.arange(28, dtype=torch.float32) + offset,
        variation=torch.arange(16, dtype=torch.float32) / 10.0 + offset,
        window_mask=torch.tensor((True, True)),
        transition_mask=torch.tensor((True,)),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


class SVSignedGINDataTest(unittest.TestCase):
    def _manifest(self, directory, records, name):
        pairs = []
        for record in records:
            path = directory / (record.sample_key + ".pt")
            save_sv_signed_gin_record(record, path)
            pairs.append((record, path))
        manifest = directory / name / "manifest.json"
        write_sv_signed_gin_manifest(pairs, manifest)
        return manifest

    def test_artifact_roundtrip_preserves_variable_signed_graphs(self):
        record = _record("one", 0, "train", node_count=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pt"
            save_sv_signed_gin_record(record, path)
            loaded = load_sv_signed_gin_record(path)
        self.assertEqual(loaded.valid_window_count, 2)
        self.assertEqual(tuple(loaded.windows[0].node_features.shape), (2, 15))
        self.assertTrue(bool((loaded.windows[0].adjacency > 0.0).any()))

    def test_train_only_scalers_and_list_batching(self):
        train_records = (
            _record("train-a", 0, "train", 0.0, 2),
            _record("train-b", 1, "train", 2.0, 3),
        )
        validation_records = (
            _record("validation-a", 1, "validation", 1.0, 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = self._manifest(
                root / "train_artifacts",
                train_records,
                "index",
            )
            validation_manifest = self._manifest(
                root / "validation_artifacts",
                validation_records,
                "index",
            )
            scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            scaler_path = root / "scaler.json"
            save_sv_signed_gin_standardizers(scaler, scaler_path)
            train = SVSignedGINDataset(train_manifest, scaler_path)
            validation = SVSignedGINDataset(
                validation_manifest, scaler_path
            )
            loader = create_sv_signed_gin_loader(
                train,
                batch_size=2,
                seed=42,
                shuffle=False,
            )
            batch = next(iter(loader))
        self.assertEqual(len(batch), 2)
        self.assertEqual(
            tuple(batch.samples[0].windows[0].node_features.shape), (2, 15)
        )
        self.assertEqual(
            tuple(batch.samples[1].windows[0].node_features.shape), (3, 15)
        )
        self.assertEqual(len(validation), 1)
        nodes = torch.cat(
            [
                window.node_features
                for sample in train.samples
                for window in sample.windows
            ],
            dim=0,
        )
        self.assertTrue(
            torch.allclose(nodes.mean(dim=0), torch.zeros(15), atol=1.0e-5)
        )

    def test_scaler_rejects_validation_records(self):
        with self.assertRaisesRegex(ValueError, "train only"):
            fit_sv_signed_gin_standardizers(
                (_record("v", 0, "validation"),), "manifest"
            )


if __name__ == "__main__":
    unittest.main()
