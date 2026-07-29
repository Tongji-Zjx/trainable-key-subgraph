from __future__ import absolute_import, division, print_function

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class SVLateFusionGateTest(unittest.TestCase):
    def test_gate_uses_validation_metrics_and_all_frozen_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            improved = root / "improved.json"
            diagnostic = root / "diagnostic.json"
            output = root / "gate"
            _write(
                baseline,
                {
                    "metrics": {
                        "balanced_accuracy": {
                            "roc_auc": 0.60,
                            "composite_auc": 0.55,
                        }
                    }
                },
            )
            _write(
                improved,
                {
                    "metrics": {
                        "balanced_accuracy": {
                            "roc_auc": 0.64,
                            "composite_auc": 0.58,
                        }
                    },
                    "branch_metrics": {
                        "gin": {"roc_auc": 0.63},
                        "static_spectral": {"roc_auc": 0.62},
                        "variation": {"roc_auc": 0.61},
                    },
                },
            )
            _write(
                diagnostic,
                {
                    "splits": {
                        "validation": {
                            "representations": {
                                "gin_representation": {
                                    "normalized_effective_rank": 0.15
                                },
                                "gin_projection": {
                                    "mean_pairwise_cosine": 0.98
                                },
                            }
                        }
                    }
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "check_sv_late_fusion_gate.py"
                    ),
                    "--baseline-evaluation",
                    str(baseline),
                    "--improved-evaluation",
                    str(improved),
                    "--improved-diagnostic",
                    str(diagnostic),
                    "--output-dir",
                    str(output),
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result = json.loads(
                (output / "gate.json").read_text(encoding="utf-8")
            )
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["test_used"])


if __name__ == "__main__":
    unittest.main()
