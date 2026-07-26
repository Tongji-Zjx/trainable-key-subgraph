from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.data_split import file_sha256
from keysubgraph.data.dual_temporal_artifact import (
    DualTemporalVariationRecord,
    save_dual_temporal_record,
)
from keysubgraph.data.dual_temporal_dataset import (
    DualTemporalDataset,
    collate_dual_temporal,
    shuffle_dual_temporal_batch,
)
from keysubgraph.data.dual_temporal_manifest import (
    write_dual_temporal_manifest,
)
from keysubgraph.data.dual_temporal_scaler import (
    fit_dual_temporal_standardizer,
    save_dual_temporal_standardizer,
)


def _record(
    key, split, values, mask, exact_manifest="exact_manifest"
):
    values = torch.tensor(values, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    return DualTemporalVariationRecord(
        sample_key=key,
        label=0 if key.endswith("0") else 1,
        split=split,
        window_count=int(values.shape[0]) + 1,
        transition_values=values,
        transition_mask=mask,
        base_logits=torch.tensor([0.2, -0.1]),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        exact_head_checkpoint_sha256="head",
        sgw_scaler_sha256="sgw",
        exact_manifest_sha256=exact_manifest,
        selection_mode="learned",
        selection_seed=42,
    )


class DualTemporalDataTest(unittest.TestCase):
    def test_train_only_scaler_uses_only_valid_transitions(self):
        records = (
            _record(
                "sample0",
                "train",
                [[1.0] * 16, [99.0] * 16],
                [True, False],
            ),
            _record(
                "sample1",
                "train",
                [[3.0] * 16],
                [True],
            ),
        )
        scaler = fit_dual_temporal_standardizer(records, "manifest")
        torch.testing.assert_close(scaler.mean, torch.full((16,), 2.0))
        self.assertEqual(scaler.valid_transition_count, 2)
        centered = scaler(
            torch.stack((records[0].transition_values[0], records[1].transition_values[0]))
        )
        torch.testing.assert_close(
            centered.mean(dim=0), torch.zeros(16), atol=1.0e-6, rtol=0.0
        )
        invalid = _record(
            "sample1", "validation", [[1.0] * 16], [True]
        )
        with self.assertRaisesRegex(ValueError, "train only"):
            fit_dual_temporal_standardizer((invalid,), "manifest")

    def test_dataset_compacts_nonprefix_mask_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = (
                _record(
                    "sample0",
                    "train",
                    [[1.0] * 16, [0.0] * 16, [3.0] * 16],
                    [True, False, True],
                ),
                _record(
                    "sample1",
                    "train",
                    [[2.0] * 16],
                    [True],
                ),
            )
            rows = []
            for record in records:
                path = root / (record.sample_key + ".pt")
                save_dual_temporal_record(record, path)
                rows.append((record, path))
            manifest = root / "manifest.json"
            write_dual_temporal_manifest(rows, manifest)
            scaler = fit_dual_temporal_standardizer(
                records, file_sha256(manifest)
            )
            scaler_path = root / "scaler.json"
            save_dual_temporal_standardizer(scaler, scaler_path)
            dataset = DualTemporalDataset(manifest, scaler_path)
            self.assertEqual(
                dataset[0]["transition_values"].shape[0], 2
            )
            batch = collate_dual_temporal([dataset[0], dataset[1]])
            self.assertEqual(batch.sequence_lengths.tolist(), [2, 1])
            self.assertEqual(batch.time_mask.tolist(), [[True, True], [True, False]])
            self.assertEqual(tuple(batch.transition_values.shape), (2, 2, 16))

    def test_collate_preserves_zero_length_samples(self):
        empty = {
            "sample_key": "empty",
            "label": 0,
            "transition_values": torch.zeros((0, 16)),
            "base_logits": torch.tensor([0.3, -0.2]),
        }
        batch = collate_dual_temporal([empty])
        self.assertEqual(batch.sequence_lengths.tolist(), [0])
        self.assertEqual(tuple(batch.transition_values.shape), (1, 1, 16))
        self.assertFalse(bool(batch.time_mask.any()))

    def test_time_shuffle_is_deterministic_and_preserves_inventory(self):
        samples = [
            {
                "sample_key": "a",
                "label": 0,
                "transition_values": torch.arange(
                    48, dtype=torch.float32
                ).reshape(3, 16),
                "base_logits": torch.tensor([0.1, 0.2]),
            },
            {
                "sample_key": "b",
                "label": 1,
                "transition_values": torch.ones((1, 16)),
                "base_logits": torch.tensor([0.2, 0.1]),
            },
        ]
        batch = collate_dual_temporal(samples)
        first = shuffle_dual_temporal_batch(batch, 2026)
        second = shuffle_dual_temporal_batch(batch, 2026)
        torch.testing.assert_close(
            first.transition_values, second.transition_values
        )
        torch.testing.assert_close(
            first.transition_values[1], batch.transition_values[1]
        )
        torch.testing.assert_close(
            first.transition_values[0].sort(dim=0).values,
            batch.transition_values[0].sort(dim=0).values,
        )
        self.assertTrue(torch.equal(first.time_mask, batch.time_mask))

    def test_validation_may_bind_a_different_exact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_records = (
                _record(
                    "train0",
                    "train",
                    [[1.0] * 16],
                    [True],
                    exact_manifest="exact_train",
                ),
                _record(
                    "train1",
                    "train",
                    [[2.0] * 16],
                    [True],
                    exact_manifest="exact_train",
                ),
            )
            train_rows = []
            for record in train_records:
                path = root / (record.sample_key + ".pt")
                save_dual_temporal_record(record, path)
                train_rows.append((record, path))
            train_manifest = root / "train_manifest.json"
            write_dual_temporal_manifest(train_rows, train_manifest)
            scaler = fit_dual_temporal_standardizer(
                train_records, file_sha256(train_manifest)
            )
            scaler_path = root / "scaler.json"
            save_dual_temporal_standardizer(scaler, scaler_path)
            validation_record = _record(
                "validation0",
                "validation",
                [[1.5] * 16],
                [True],
                exact_manifest="exact_validation",
            )
            validation_path = root / "validation0.pt"
            save_dual_temporal_record(
                validation_record, validation_path
            )
            validation_manifest = root / "validation_manifest.json"
            write_dual_temporal_manifest(
                [(validation_record, validation_path)],
                validation_manifest,
            )
            dataset = DualTemporalDataset(
                validation_manifest, scaler_path
            )
            self.assertEqual(dataset.split, "validation")
            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
