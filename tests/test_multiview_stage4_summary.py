from __future__ import absolute_import, division, print_function

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _evaluation(path, threshold, auc):
    payload = {
        "threshold": threshold,
        "metrics": {
            "sample_count": 10,
            "roc_auc": auc,
            "balanced_accuracy": 0.6,
            "accuracy": 0.6,
            "f1": 0.5,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class MultiViewStage4SummaryTest(unittest.TestCase):
    def _fixture(self, root, mismatched_test_threshold=False):
        arguments = []
        aucs = {"author": 0.55, "critical": 0.60, "fusion": 0.65}
        for model in ("author", "critical", "fusion"):
            validation = root / "{}_validation.json".format(model)
            test = root / "{}_test.json".format(model)
            _evaluation(validation, 0.4, aucs[model])
            _evaluation(
                test,
                0.5 if mismatched_test_threshold and model == "fusion" else 0.4,
                aucs[model] - 0.01,
            )
            arguments.extend(
                (
                    "--{}-validation".format(model),
                    str(validation),
                    "--{}-test".format(model),
                    str(test),
                )
            )
        for stage in (2, 3):
            path = root / "stage{}_selection.json".format(stage)
            path.write_text(
                json.dumps({"test_used": False, "condition": "frozen"}),
                encoding="utf-8",
            )
            arguments.extend(("--stage{}-selection".format(stage), str(path)))
        return arguments

    def test_summary_computes_frozen_increment_without_test_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_multiview_stage4.py",
                ]
                + self._fixture(root)
                + ["--output-dir", str(output)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["test_used_for_selection"])
            self.assertFalse(summary["test_threshold_refit"])
            self.assertAlmostEqual(
                summary["increments"]["test"]["fusion_minus_author_auc"],
                0.10,
            )
            self.assertAlmostEqual(
                summary["increments"]["test"]["fusion_minus_critical_auc"],
                0.05,
            )

    def test_summary_rejects_test_threshold_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_multiview_stage4.py",
                ]
                + self._fixture(root, mismatched_test_threshold=True)
                + ["--output-dir", str(root / "output")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"validation/test thresholds differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
