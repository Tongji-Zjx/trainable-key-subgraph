from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.selector_transfer_oof import (  # noqa: E402
    evaluate_selector_transfer_outer_fold,
)


def _split(name, keys, labels, offset=0.0):
    features = np.zeros((len(keys), 44), dtype=np.float64)
    features[:, 0] = np.asarray(labels, dtype=np.float64) + offset
    return {
        "manifest": {
            "split": name,
            "protocol_sha256": "protocol",
            "selector_checkpoint_sha256": "selector",
        },
        "sample_keys": list(keys),
        "labels": np.asarray(labels, dtype=int),
        "sites": ["S"] * len(keys),
        "features": features,
    }


class SelectorTransferOOFTest(unittest.TestCase):
    def test_probe_fits_train_thresholds_validation_and_tests_once(self):
        data = {
            "train": _split("train", ("a", "b", "c", "d"), (0, 1, 0, 1)),
            "validation": _split(
                "validation", ("e", "f", "g", "h"), (0, 1, 0, 1)
            ),
            "test": _split("test", ("i", "j", "k", "l"), (0, 1, 0, 1)),
        }
        with patch(
            "keysubgraph.analysis.selector_transfer_oof._load_probe_split",
            side_effect=lambda path: data[str(path)],
        ):
            result = evaluate_selector_transfer_outer_fold(
                Path("train"),
                Path("validation"),
                Path("test"),
                seed=42,
            )
        self.assertFalse(result["test_used_for_fitting"])
        self.assertEqual(
            result["probe"]["threshold_fit_split"], "validation"
        )
        self.assertEqual(
            len(result["evaluations"]["test"]["predictions"]), 4
        )
        self.assertAlmostEqual(
            result["evaluations"]["test"]["metrics"]["roc_auc"], 1.0
        )

    def test_overlap_fails_closed(self):
        data = {
            "train": _split("train", ("a", "b"), (0, 1)),
            "validation": _split("validation", ("c", "d"), (0, 1)),
            "test": _split("test", ("a", "e"), (0, 1)),
        }
        with patch(
            "keysubgraph.analysis.selector_transfer_oof._load_probe_split",
            side_effect=lambda path: data[str(path)],
        ):
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_selector_transfer_outer_fold(
                    Path("train"),
                    Path("validation"),
                    Path("test"),
                )


if __name__ == "__main__":
    unittest.main()
