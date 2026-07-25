from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_sgw_feature_dataset import (
    DualSGWFeatureDataset,
    create_dual_sgw_feature_loader,
)
from keysubgraph.data.dual_sgw_manifest import (
    dual_feature_filename,
    write_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (
    fit_dual_sgw_standardizer,
    save_dual_sgw_standardizer,
)
from keysubgraph.models.dual_exact_sgw import (
    DualSGWFeatureRecord,
    save_dual_sgw_feature_record,
)
from keysubgraph.training.dual_sgw_feature_ensemble import (
    build_dual_sgw_probability_ensemble,
)


def _record(key, label, split, offset):
    representation = torch.arange(34, dtype=torch.float32) + offset
    return DualSGWFeatureRecord(
        sample_key=key,
        label=label,
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


def _evaluation(split, seed, predictions):
    return {
        "classifier_type": "linear",
        "protocol_sha256": "protocol",
        "manifest_sha256": "{}-manifest".format(split),
        "scaler_sha256": "scaler",
        "manifest_provenance": {
            "protocol_sha256": "protocol",
            "selector_checkpoint_sha256": "selector",
            "selection_mode": "learned",
            "selection_seed": 42,
        },
        "seed": seed,
        "split": split,
        "predictions": predictions,
    }


class DualSGWFeatureDataAndEnsembleTest(unittest.TestCase):
    def test_dataset_loads_only_cached_standardized_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_records = [
                _record("train-a", 0, "train", 0.0),
                _record("train-b", 1, "train", 2.0),
            ]
            manifest_rows = []
            for record in train_records:
                path = root / dual_feature_filename(record.sample_key)
                save_dual_sgw_feature_record(record, path)
                manifest_rows.append((record, path))
            manifest = write_dual_sgw_manifest(
                manifest_rows,
                root / "manifest.json",
                "protocol",
                "selector",
                "learned",
                42,
            )
            scaler = fit_dual_sgw_standardizer(train_records)
            scaler_path = save_dual_sgw_standardizer(
                scaler, root / "scaler.json"
            )
            dataset = DualSGWFeatureDataset(manifest, scaler_path)
            self.assertEqual(len(dataset), 2)
            stacked = torch.stack(
                [dataset[index]["features"] for index in range(2)]
            )
            self.assertTrue(
                torch.allclose(stacked.mean(dim=0), torch.zeros(34))
            )
            loader = create_dual_sgw_feature_loader(
                dataset,
                batch_size=2,
                seed=42,
                shuffle=True,
            )
            self.assertEqual(
                tuple(next(iter(loader))["features"].shape), (2, 34)
            )

    def test_ensemble_aligns_by_key_and_freezes_validation_thresholds(self):
        validation_one = [
            {
                "sample_key": "a",
                "label": 0,
                "positive_probability": 0.2,
            },
            {
                "sample_key": "b",
                "label": 1,
                "positive_probability": 0.7,
            },
        ]
        validation_two = list(reversed([
            {
                "sample_key": "a",
                "label": 0,
                "positive_probability": 0.4,
            },
            {
                "sample_key": "b",
                "label": 1,
                "positive_probability": 0.9,
            },
        ]))
        test_one = [
            {
                "sample_key": "c",
                "label": 0,
                "positive_probability": 0.3,
            },
            {
                "sample_key": "d",
                "label": 1,
                "positive_probability": 0.8,
            },
        ]
        test_two = list(reversed([
            {
                "sample_key": "c",
                "label": 0,
                "positive_probability": 0.5,
            },
            {
                "sample_key": "d",
                "label": 1,
                "positive_probability": 0.6,
            },
        ]))
        payload = build_dual_sgw_probability_ensemble(
            [
                _evaluation("validation", 42, validation_one),
                _evaluation("validation", 43, validation_two),
            ],
            [
                _evaluation("test", 42, test_one),
                _evaluation("test", 43, test_two),
            ],
        )
        self.assertEqual(payload["component_seeds"], [42, 43])
        probabilities = {
            item["sample_key"]: item["positive_probability"]
            for item in payload["validation"]["predictions"]
        }
        self.assertAlmostEqual(probabilities["a"], 0.3)
        self.assertAlmostEqual(probabilities["b"], 0.8)
        self.assertEqual(
            payload["validation_thresholds"],
            {
                "balanced_accuracy": 0.5,
                "accuracy": 0.5,
            },
        )
        mismatch = _evaluation("test", 43, test_two)
        mismatch["predictions"][0]["label"] = 0
        with self.assertRaisesRegex(ValueError, "labels"):
            build_dual_sgw_probability_ensemble(
                [
                    _evaluation("validation", 42, validation_one),
                    _evaluation("validation", 43, validation_two),
                ],
                [
                    _evaluation("test", 42, test_one),
                    mismatch,
                ],
            )


if __name__ == "__main__":
    unittest.main()
