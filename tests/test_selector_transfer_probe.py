from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.analysis.selector_transfer_probe import (
    compare_selector_transfer_conditions,
    selector_transfer_probe_markdown,
)
from keysubgraph.data.sv_signed_gin_artifact import (
    SVSignedGINRecord,
    SVSignedGINWindowRecord,
    save_sv_signed_gin_record,
)
from keysubgraph.data.sv_signed_gin_manifest import (
    write_sv_signed_gin_manifest,
)


def _write_manifest(root, condition, split, offsets):
    output = root / condition / split
    records = []
    for index, (label, offset) in enumerate(offsets):
        key = "{}-{:02d}".format(split, index)
        record = SVSignedGINRecord(
            sample_key=key,
            sample_id=key,
            subject_id="subject-" + key,
            site="site-{}".format(index % 2),
            label=label,
            split=split,
            windows=(
                SVSignedGINWindowRecord(
                    node_features=torch.tensor(
                        ((1.0, 0.0), (0.0, 1.0)),
                        dtype=torch.float32,
                    ).repeat(1, 8)[:, :15],
                    adjacency=torch.tensor(
                        ((0.0, 0.5), (0.5, 0.0)),
                        dtype=torch.float32,
                    ),
                    time_start=0.0,
                ),
            ),
            static_features=torch.linspace(
                -1.0, 1.0, 28
            ) + float(offset),
            variation=torch.linspace(
                -0.5, 0.5, 16
            ) + float(offset),
            window_mask=torch.tensor((True,)),
            transition_mask=torch.zeros(0, dtype=torch.bool),
            protocol_sha256="protocol-sha",
            selector_checkpoint_sha256=condition + "-sha",
            selection_mode=(
                "random" if condition == "random" else "learned"
            ),
            selection_seed=42,
        )
        path = output / (key + ".pt")
        save_sv_signed_gin_record(record, path)
        records.append((record, path))
    manifest = output / "manifest.json"
    write_sv_signed_gin_manifest(records, manifest)
    return manifest


class SelectorTransferProbeTest(unittest.TestCase):
    def test_conditions_use_aligned_train_only_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [(0, -1.2), (0, -0.8), (1, 0.8), (1, 1.2)]
            validation = [
                (0, -1.1),
                (0, -0.7),
                (1, 0.7),
                (1, 1.1),
            ]
            current_train = _write_manifest(
                root, "current", "train", train
            )
            current_validation = _write_manifest(
                root, "current", "validation", validation
            )
            random_train = _write_manifest(
                root, "random", "train", train
            )
            random_validation = _write_manifest(
                root, "random", "validation", validation
            )
            payload = compare_selector_transfer_conditions(
                (
                    ("current", current_train, current_validation),
                    ("random", random_train, random_validation),
                ),
                seed=7,
            )
            self.assertFalse(payload["test_used"])
            self.assertEqual(len(payload["rows"]), 2)
            self.assertEqual(
                payload["rows"][0]["roc_auc"], 1.0
            )
            self.assertEqual(
                payload["rows"][1]["delta_auc_vs_reference"], 0.0
            )
            report = selector_transfer_probe_markdown(payload)
            self.assertIn("train-only", report)
            self.assertIn("| random |", report)

    def test_misaligned_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [(0, -1.0), (0, -0.5), (1, 0.5), (1, 1.0)]
            validation = [
                (0, -1.0),
                (0, -0.5),
                (1, 0.5),
                (1, 1.0),
            ]
            current_train = _write_manifest(
                root, "current", "train", train
            )
            current_validation = _write_manifest(
                root, "current", "validation", validation
            )
            random_train = _write_manifest(
                root, "random", "train", train
            )
            random_validation = _write_manifest(
                root, "random", "validation", validation[:-1]
            )
            with self.assertRaisesRegex(ValueError, "align by sample"):
                compare_selector_transfer_conditions(
                    (
                        (
                            "current",
                            current_train,
                            current_validation,
                        ),
                        (
                            "random",
                            random_train,
                            random_validation,
                        ),
                    )
                )


if __name__ == "__main__":
    unittest.main()
