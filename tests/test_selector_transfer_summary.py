from __future__ import absolute_import, division, print_function

import json
import tempfile
import unittest
from pathlib import Path

from keysubgraph.analysis.selector_transfer_summary import (
    selector_transfer_formal_markdown,
    summarize_selector_transfer_formal,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


class SelectorTransferSummaryTest(unittest.TestCase):
    def test_summary_freezes_two_dataset_success_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("wmrc", "adhd"):
                for index, objective in enumerate(
                    ("current", "full_soft", "full_soft_hard")
                ):
                    _write_json(
                        root
                        / "{}_{}".format(dataset, objective)
                        / "best_evaluation.json",
                        {
                            "best_epoch": index + 1,
                            "validation": {
                                "roc_auc": 0.50 + 0.05 * index,
                                "hard_roc_auc": 0.50 + 0.05 * index,
                                "soft_roc_auc": (
                                    None
                                    if objective == "current"
                                    else 0.52 + 0.05 * index
                                ),
                                "balanced_accuracy": 0.55,
                                "accuracy": 0.56,
                            },
                        },
                    )
                rows = [
                    {
                        "name": name,
                        "roc_auc": auc,
                        "delta_auc_vs_reference": auc - 0.55,
                        "site_stratified_roc_auc": auc - 0.01,
                        "balanced_accuracy": 0.55,
                        "accuracy": 0.56,
                        "f1": 0.50,
                    }
                    for name, auc in (
                        ("current", 0.55),
                        ("full_soft", 0.58),
                        ("full_soft_hard", 0.62),
                        ("random", 0.51),
                        ("full", 0.57),
                    )
                ]
                _write_json(
                    root
                    / "fair_probe_{}".format(dataset)
                    / "probe"
                    / "comparison.json",
                    {"test_used": False, "rows": rows},
                )
            payload = summarize_selector_transfer_formal(root)
            self.assertTrue(
                payload[
                    "e3_consistently_beats_current_and_random"
                ]
            )
            self.assertEqual(
                payload["datasets"]["wmrc"]["fair_probe_winner"],
                "full_soft_hard",
            )
            report = selector_transfer_formal_markdown(payload)
            self.assertIn("E3 − E0 AUROC", report)
            self.assertIn("**是**", report)


if __name__ == "__main__":
    unittest.main()
