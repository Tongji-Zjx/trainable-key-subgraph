from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.background.conditional_oof import build_conditional_oof_cache
from keysubgraph.tge.dataset import TGEPrecomputedDataset, file_sha256
from keysubgraph.tge.types import TGESample


class MoKSEConditionalOOFTest(unittest.TestCase):
    def _write_manifest(self, root, split, rows):
        directory = root / split
        samples = directory / "samples"
        samples.mkdir(parents=True)
        records = []
        for index, (site, label, subject) in enumerate(rows):
            sample = TGESample(
                sample_key="{}/{}".format(site, subject),
                sample_id=subject,
                subject_id=subject,
                site=site,
                label=label,
                split=split,
                trajectories=(),
                provenance={},
            )
            path = samples / "{}.pt".format(index)
            torch.save(sample, str(path))
            records.append(
                {
                    "sample_key": sample.sample_key,
                    "sample_id": subject,
                    "subject_id": subject,
                    "site": site,
                    "label": label,
                    "split": split,
                    "artifact_path": "samples/{}.pt".format(index),
                    "artifact_sha256": file_sha256(path),
                }
            )
        manifest = {
            "artifact_type": "tge_preprocessed_manifest",
            "feature_schema_version": 2,
            "protocol_sha256": "protocol",
            "selector_checkpoint_sha256": "selector",
            "split": split,
            "sample_count": len(records),
            "records": records,
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_derived_splits_are_disjoint_and_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_rows = []
            for site in ("A", "B"):
                for label in (0, 1):
                    for index in range(5):
                        train_rows.append((site, label, "{}_{}_{}".format(site, label, index)))
            target_rows = [
                ("A", 0, "target_a0"),
                ("A", 1, "target_a1"),
                ("B", 0, "target_b0"),
                ("B", 1, "target_b1"),
            ]
            train = self._write_manifest(root, "train", train_rows)
            target = self._write_manifest(root, "validation", target_rows)
            report = build_conditional_oof_cache(
                train, target, root / "derived", validation_fraction=0.20, seed=7
            )
            paths = report["manifests"]
            datasets = {
                "train": TGEPrecomputedDataset(paths["inner_train"], "train"),
                "validation": TGEPrecomputedDataset(
                    paths["inner_validation"], "validation"
                ),
                "test": TGEPrecomputedDataset(paths["oof_target"], "test"),
            }
            keys = {
                name: {row["sample_key"] for row in dataset.records}
                for name, dataset in datasets.items()
            }
            self.assertFalse(keys["train"] & keys["validation"])
            self.assertFalse(keys["train"] & keys["test"])
            self.assertFalse(keys["validation"] & keys["test"])
            self.assertEqual({datasets["validation"][0].split}, {"validation"})
            self.assertEqual({datasets["test"][0].split}, {"test"})
            self.assertTrue(
                report["conditional_on_frozen_selector_and_trajectory_cache"]
            )
            self.assertFalse(report["end_to_end_selector_oof"])


if __name__ == "__main__":
    unittest.main()
