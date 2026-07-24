from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_sgw_scaler import (
    fit_dual_sgw_standardizer,
    load_dual_sgw_standardizer,
    save_dual_sgw_standardizer,
)
from keysubgraph.models.dual_exact_sgw import DualSGWFeatureRecord


def _record(key, split, offset):
    representation = torch.arange(34, dtype=torch.float32) + offset
    return DualSGWFeatureRecord(
        sample_key=key,
        label=int(offset) % 2,
        split=split,
        selection_mode="learned",
        selection_seed=42,
        core=representation[:18],
        variation=representation[18:],
        representation=representation,
        transition_mask=torch.tensor([True]),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
    )


class DualSGWScalerTest(unittest.TestCase):
    def test_fit_is_train_only_and_centers_training_features(self):
        records = (
            _record("a", "train", 0.0),
            _record("b", "train", 2.0),
        )
        scaler = fit_dual_sgw_standardizer(records)
        values = torch.stack(
            [record.representation for record in records]
        )
        transformed = scaler(values)
        self.assertTrue(
            torch.allclose(
                transformed.mean(dim=0),
                torch.zeros(34),
                atol=1.0e-6,
            )
        )
        with self.assertRaisesRegex(ValueError, "train only"):
            fit_dual_sgw_standardizer(
                records + (_record("c", "validation", 4.0),)
            )

    def test_json_round_trip_preserves_provenance_and_values(self):
        scaler = fit_dual_sgw_standardizer(
            (_record("a", "train", 0.0),)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scaler.json"
            save_dual_sgw_standardizer(scaler, path)
            loaded = load_dual_sgw_standardizer(path)
            self.assertEqual(loaded.sample_count, 1)
            self.assertEqual(loaded.protocol_sha256, "protocol")
            self.assertEqual(loaded.selection_mode, "learned")
            self.assertEqual(loaded.selection_seed, 42)
            self.assertTrue(torch.equal(loaded.mean, scaler.mean))
            self.assertTrue(torch.equal(loaded.scale, scaler.scale))
            with self.assertRaises(FileExistsError):
                save_dual_sgw_standardizer(scaler, path)


if __name__ == "__main__":
    unittest.main()
