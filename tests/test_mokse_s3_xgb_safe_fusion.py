import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "search_mokse_s3_xgb_safe_fusion.py"


class MoKSES3XGBSafeFusionTest(unittest.TestCase):
    @staticmethod
    def _write_static(path, prefix, labels, seed):
        rng = np.random.RandomState(seed)
        labels = np.asarray(labels, dtype=np.int64)
        count = labels.size
        representation = rng.normal(size=(count, 24)).astype(np.float32)
        representation[:, 0] += 0.4 * labels
        logits = (0.25 * representation[:, 0] - 0.1).astype(np.float32)
        np.savez_compressed(
            str(path),
            sample_keys=np.asarray(
                ["{}_{}".format(prefix, index) for index in range(count)], dtype=str
            ),
            sites=np.asarray(["site_a"] * count, dtype=str),
            labels=labels,
            background_representations=representation,
            background_logits=logits,
        )

    @staticmethod
    def _write_predictions(path, prefix, labels, logits):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("sample_key", "site", "label", "final_logit"))
            for index, (label, logit) in enumerate(zip(labels, logits)):
                writer.writerow((
                    "{}_{}".format(prefix, index), "site_a", int(label), float(logit)
                ))

    def test_shared_fourfold_search_writes_auditable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static_dirs = []
            prediction_dirs = []
            train_labels = [0, 1] * 6
            validation_labels = [0, 1, 0, 1]
            test_labels = [0, 1, 0, 1, 0, 1]
            for fold in range(4):
                static = root / "static" / ("fold_{}".format(fold))
                prediction = root / "subgraph" / ("fold_{}".format(fold))
                static.mkdir(parents=True)
                prediction.mkdir(parents=True)
                self._write_static(
                    static / "train_features.npz", "train{}_".format(fold),
                    train_labels, 10 + fold,
                )
                self._write_static(
                    static / "validation_features.npz", "val{}_".format(fold),
                    validation_labels, 20 + fold,
                )
                self._write_static(
                    static / "test_features.npz", "test", test_labels, 30 + fold,
                )
                self._write_predictions(
                    prediction / "validation_predictions.csv",
                    "val{}_".format(fold), validation_labels,
                    np.asarray([-0.2, 0.2, -0.1, 0.1]),
                )
                self._write_predictions(
                    prediction / "test_predictions.csv", "test", test_labels,
                    np.asarray([-0.3, 0.3, -0.2, 0.2, -0.1, 0.1]),
                )
                static_dirs.append(static)
                prediction_dirs.append(prediction)
            output = root / "result"
            command = [
                sys.executable, str(SCRIPT), "--dataset", "adhd",
                "--output-dir", str(output), "--trials", "2", "--nthread", "1",
            ]
            for path in static_dirs:
                command.extend(("--static-fold-dir", str(path)))
            for path in prediction_dirs:
                command.extend(("--subgraph-prediction-dir", str(path)))
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, capture_output=True)
            report = json.loads((output / "search_results.json").read_text("utf-8"))
            self.assertEqual(
                report["artifact_type"],
                "mokse_s3_xgb_safe_fusion_test_guided_fourfold_v1",
            )
            self.assertTrue(report["fourfold_shared_xgb_candidate"])
            self.assertTrue(report["test_used_for_xgb_parameter_selection"])
            self.assertFalse(report["test_used_for_fusion_weight_selection"])
            self.assertEqual(report["trial_count"], 2)
            self.assertEqual(len(report["best"]["rotations"]), 4)
            self.assertTrue((output / "boosters" / "fold_0_booster.json").is_file())
            self.assertTrue(
                (output / "predictions" / "fold_3" / "test_predictions.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
