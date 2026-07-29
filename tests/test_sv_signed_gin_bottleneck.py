from __future__ import absolute_import, division, print_function

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from keysubgraph.data.graph_dataset import GraphSequenceSample  # noqa: F401
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
from keysubgraph.analysis.sv_signed_gin_bottleneck import (
    analyze_sv_signed_gin_bottleneck,
    collect_sv_diagnostics,
    diagnose_sv_sample,
    frozen_channel_masking,
    representation_statistics,
    write_sv_signed_gin_bottleneck_artifacts,
)
from keysubgraph.models.sv_signed_gin import (
    SVSignedGINBatch,
    SVSignedGINClassifier,
    SVSignedGINConfig,
    SVSignedGINSampleInput,
    SVSignedGINWindowInput,
)


def _sample(key, label, offset):
    features = (
        torch.arange(45, dtype=torch.float32).reshape(3, 15)
        / 20.0
        + float(offset)
    )
    adjacency = torch.tensor(
        (
            (0.0, 0.5, -0.2),
            (0.5, 0.0, -0.3),
            (-0.2, -0.3, 0.0),
        ),
        dtype=torch.float32,
    )
    return SVSignedGINSampleInput(
        sample_key=key,
        label=label,
        windows=(
            SVSignedGINWindowInput(features, adjacency),
            SVSignedGINWindowInput(features + 0.1, adjacency * 0.8),
        ),
        static_features=(
            torch.linspace(-1.0, 1.0, 28) + float(offset)
        ),
        variation=(
            torch.linspace(-0.5, 0.5, 16) + float(offset)
        ),
    )


def _record(key, label, split, offset, site):
    sample = _sample(key, label, offset)
    return SVSignedGINRecord(
        sample_key=sample.sample_key,
        sample_id=sample.sample_key,
        subject_id="subject-" + sample.sample_key,
        site=site,
        label=label,
        split=split,
        windows=tuple(
            SVSignedGINWindowRecord(
                window.node_features, window.adjacency, float(index)
            )
            for index, window in enumerate(sample.windows)
        ),
        static_features=sample.static_features,
        variation=sample.variation,
        window_mask=torch.tensor((True, True)),
        transition_mask=torch.tensor((True,)),
        protocol_sha256="protocol",
        selector_checkpoint_sha256="selector",
        selection_mode="learned",
        selection_seed=42,
    )


def _manifest(root, name, records):
    pairs = []
    artifact_root = root / (name + "_artifacts")
    for record in records:
        path = artifact_root / (record.sample_key + ".pt")
        save_sv_signed_gin_record(record, path)
        pairs.append((record, path))
    path = root / name / "manifest.json"
    write_sv_signed_gin_manifest(pairs, path)
    return path


class _Dataset(object):
    def __init__(self, split, samples, sites):
        self.split = split
        self.samples = list(samples)
        self.sites = list(sites)
        self.subject_ids = [
            "subject-" + sample.sample_key for sample in samples
        ]

    @property
    def sample_keys(self):
        return tuple(sample.sample_key for sample in self.samples)

    @property
    def labels(self):
        return tuple(sample.label for sample in self.samples)

    def __len__(self):
        return len(self.samples)


class SVSignedGINBottleneckTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)
        self.model = SVSignedGINClassifier(
            SVSignedGINConfig(
                variant="signed_gin_static_variation",
                dropout=0.0,
            )
        ).eval()
        self.train = _Dataset(
            "train",
            (
                _sample("train-0a", 0, -1.0),
                _sample("train-0b", 0, -0.5),
                _sample("train-1a", 1, 0.5),
                _sample("train-1b", 1, 1.0),
            ),
            ("a", "b", "a", "b"),
        )
        self.validation = _Dataset(
            "validation",
            (
                _sample("validation-0a", 0, -0.8),
                _sample("validation-0b", 0, -0.3),
                _sample("validation-1a", 1, 0.3),
                _sample("validation-1b", 1, 0.8),
            ),
            ("a", "b", "a", "b"),
        )

    def test_manual_diagnostic_matches_frozen_forward(self):
        sample = self.train.samples[0]
        diagnosed = diagnose_sv_sample(self.model, sample)
        expected = self.model(SVSignedGINBatch((sample,))).logits[0]
        self.assertTrue(
            torch.allclose(diagnosed["logits"], expected, atol=1.0e-6)
        )
        self.assertEqual(
            diagnosed["representations"]["final_representation"].shape,
            (48,),
        )
        self.assertEqual(len(diagnosed["cancellation"]), 4)
        self.assertEqual(len(diagnosed["attention"]), 2)

    def test_representation_statistics_detects_collapse(self):
        collapsed = np.ones((6, 4), dtype=np.float64)
        result = representation_statistics(
            collapsed, (0, 0, 0, 1, 1, 1)
        )
        self.assertAlmostEqual(result["effective_rank"], 0.0)
        self.assertAlmostEqual(
            result["mean_pairwise_cosine"], 1.0
        )
        self.assertAlmostEqual(result["active_feature_fraction"], 0.0)

    def test_complete_analysis_is_read_only_and_writes_artifacts(self):
        before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        train = collect_sv_diagnostics(
            self.model, self.train, torch.device("cpu")
        )
        validation = collect_sv_diagnostics(
            self.model, self.validation, torch.device("cpu")
        )
        masks = frozen_channel_masking(
            self.model,
            train,
            validation,
            threshold=0.5,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(masks), 11)
        self.assertEqual(masks[0]["condition"], "all")
        self.assertIn("site_stratified_roc_auc", masks[0])
        self.assertEqual(masks[0]["eligible_site_count"], 2)
        self.assertEqual(len(masks[0]["per_site"]), 2)
        result = analyze_sv_signed_gin_bottleneck(
            train,
            validation,
            self.model,
            threshold=0.5,
            device=torch.device("cpu"),
            seed=811,
        )
        self.assertFalse(result["test_used"])
        self.assertEqual(result["parameter_update_count"], 0)
        self.assertTrue(result["forward_consistency"]["passed"])
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            paths = write_sv_signed_gin_bottleneck_artifacts(
                result, output
            )
            payload = json.loads(
                (output / "diagnostic.json").read_text(
                    encoding="utf-8"
                )
            )
            summary_exists = Path(paths["summary"]).exists()
            site_csv_exists = Path(
                paths["channel_masking_by_site"]
            ).exists()
        self.assertFalse(payload["test_used"])
        self.assertTrue(summary_exists)
        self.assertTrue(site_csv_exists)

    def test_command_line_entry_point_validates_provenance(self):
        train_records = (
            _record("train-0a", 0, "train", -1.0, "a"),
            _record("train-0b", 0, "train", -0.5, "b"),
            _record("train-1a", 1, "train", 0.5, "a"),
            _record("train-1b", 1, "train", 1.0, "b"),
        )
        validation_records = (
            _record("validation-0a", 0, "validation", -0.8, "a"),
            _record("validation-0b", 0, "validation", -0.3, "b"),
            _record("validation-1a", 1, "validation", 0.3, "a"),
            _record("validation-1b", 1, "validation", 0.8, "b"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = _manifest(
                root, "train", train_records
            )
            validation_manifest = _manifest(
                root, "validation", validation_records
            )
            scaler = fit_sv_signed_gin_standardizers(
                train_records, file_sha256(train_manifest)
            )
            scaler_path = root / "scaler.json"
            save_sv_signed_gin_standardizers(scaler, scaler_path)
            model = SVSignedGINClassifier(
                SVSignedGINConfig(
                    variant="signed_gin_static_variation",
                    dropout=0.0,
                )
            )
            provenance = {
                "protocol_sha256": "protocol",
                "selector_checkpoint_sha256": "selector",
                "selection_mode": "learned",
                "selection_seed": 42,
                "train_manifest_sha256": file_sha256(
                    train_manifest
                ),
                "validation_manifest_sha256": file_sha256(
                    validation_manifest
                ),
                "scaler_sha256": file_sha256(scaler_path),
            }
            checkpoint = root / "best_checkpoint.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "model_name": "sv_hard_sgw_signed_gin",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config_dict(),
                    "provenance": provenance,
                    "validation_thresholds": {
                        "balanced_accuracy": 0.5
                    },
                },
                str(checkpoint),
            )
            output = root / "output"
            environment = dict(os.environ)
            source = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            environment["PYTHONPATH"] = (
                source
                + os.pathsep
                + environment.get("PYTHONPATH", "")
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "diagnose_sv_signed_gin_bottleneck.py"
                    ),
                    "--train-manifest",
                    str(train_manifest),
                    "--validation-manifest",
                    str(validation_manifest),
                    "--scaler",
                    str(scaler_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                ],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
            )
            diagnostic_exists = (output / "diagnostic.json").exists()
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + "\n" + completed.stderr,
        )
        self.assertTrue(diagnostic_exists)


if __name__ == "__main__":
    unittest.main()
