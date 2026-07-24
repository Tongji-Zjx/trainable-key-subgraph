from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_sgw_manifest import (
    dual_feature_filename,
    read_dual_sgw_manifest,
    write_dual_sgw_manifest,
)
from keysubgraph.models.dual_exact_sgw import (
    DualSGWFeatureRecord,
    save_dual_sgw_feature_record,
)


class DualSGWManifestTest(unittest.TestCase):
    def _record(self, key):
        representation = torch.arange(34, dtype=torch.float32)
        return DualSGWFeatureRecord(
            sample_key=key,
            label=1,
            split="train",
            selection_mode="learned",
            selection_seed=42,
            core=representation[:18],
            variation=representation[18:],
            representation=representation,
            transition_mask=torch.tensor([True]),
            protocol_sha256="protocol",
            selector_checkpoint_sha256="selector",
        )

    def test_manifest_round_trip_checks_coverage_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for key in ("site/sample-a", "site/sample-b"):
                record = self._record(key)
                path = root / dual_feature_filename(key)
                save_dual_sgw_feature_record(record, path)
                records.append((record, path))
            manifest = write_dual_sgw_manifest(
                records,
                root / "manifest.json",
                "protocol",
                "selector",
                "learned",
                42,
            )
            payload, loaded, lookup = read_dual_sgw_manifest(manifest)
            self.assertEqual(payload["sample_count"], 2)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(lookup), {"site/sample-a", "site/sample-b"})
            self.assertNotIn(b"\r\n", manifest.read_bytes())
            with self.assertRaises(FileExistsError):
                write_dual_sgw_manifest(
                    records,
                    manifest,
                    "protocol",
                    "selector",
                    "learned",
                    42,
                )


if __name__ == "__main__":
    unittest.main()
