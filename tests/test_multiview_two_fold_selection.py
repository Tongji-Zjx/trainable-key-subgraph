from __future__ import absolute_import, division, print_function

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MultiViewTwoFoldSelectionTest(unittest.TestCase):
    def _summary(self, root, name, stage, rows):
        path = root / (name + ".json")
        path.write_text(json.dumps({
            "artifact_type": "multiview_stage_validation_summary",
            "stage": stage,
            "test_used": False,
            "conditions": rows,
        }), encoding="utf-8")
        return path

    def _run(self, root, stage, fold_a, fold_b, masking=()):
        output = root / (stage + "_out")
        command = [
            sys.executable, "scripts/summarize_multiview_two_fold_selection.py",
            "--stage", stage,
            "--fold", "a={}".format(fold_a),
            "--fold", "b={}".format(fold_b),
            "--output-dir", str(output),
        ]
        for name, path in masking:
            command.extend(("--masking", "{}={}".format(name, path)))
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return json.loads((output / "selection.json").read_text(encoding="utf-8"))

    def test_stable_cannot_win_formal_s_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_a = [
                {"condition": "S_stable", "static_mode": "stable", "v": "none", "g": False, "roc_auc": 0.95},
                {"condition": "S_neural", "static_mode": "neural", "v": "none", "g": False, "roc_auc": 0.60},
                {"condition": "S_residual", "static_mode": "residual", "v": "none", "g": False, "roc_auc": 0.70},
            ]
            rows_b = [dict(row, roc_auc=row["roc_auc"] - 0.01) for row in rows_a]
            result = self._run(
                root, "s",
                self._summary(root, "a", "stage1", rows_a),
                self._summary(root, "b", "stage1", rows_b),
            )
            self.assertEqual(result["decision"]["selected"], "residual")
            self.assertIn("stable", result["excluded_controls"])

    def test_legacy_and_shuffled_cannot_win_formal_v_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def rows(none, legacy, uot, shuffled):
                return [
                    {"condition": "none", "static_mode": "residual", "v": "none", "g": False, "roc_auc": none},
                    {"condition": "legacy", "static_mode": "residual", "v": "legacy", "g": False, "roc_auc": legacy},
                    {"condition": "uot", "static_mode": "residual", "v": "uot", "g": False, "roc_auc": uot},
                    {"condition": "shuffled", "static_mode": "residual", "v": "shuffled", "g": False, "roc_auc": shuffled},
                ]
            result = self._run(
                root, "v",
                self._summary(root, "a", "stage2", rows(0.60, 0.99, 0.64, 0.61)),
                self._summary(root, "b", "stage2", rows(0.61, 0.98, 0.63, 0.60)),
            )
            self.assertEqual(result["decision"]["selected"], "uot")
            self.assertEqual(result["official_candidates"], ["none", "uot"])
            self.assertIn("legacy", result["excluded_controls"])
            self.assertIn("shuffled", result["excluded_controls"])

    def test_uot_falls_back_to_none_when_correspondence_control_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def rows(uot, shuffled):
                return [
                    {"condition": "none", "static_mode": "neural", "v": "none", "g": False, "roc_auc": 0.60},
                    {"condition": "legacy", "static_mode": "neural", "v": "legacy", "g": False, "roc_auc": 0.80},
                    {"condition": "uot", "static_mode": "neural", "v": "uot", "g": False, "roc_auc": uot},
                    {"condition": "shuffled", "static_mode": "neural", "v": "shuffled", "g": False, "roc_auc": shuffled},
                ]
            result = self._run(
                root, "v",
                self._summary(root, "a", "stage2", rows(0.63, 0.65)),
                self._summary(root, "b", "stage2", rows(0.62, 0.64)),
            )
            self.assertEqual(result["decision"]["selected"], "none")
            self.assertFalse(result["decision"]["uot_gate_passes"])

    def test_g_requires_two_fold_gain_and_masking_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def rows(off, on):
                return [
                    {"condition": "off", "static_mode": "residual", "v": "none", "g": False, "roc_auc": off},
                    {"condition": "on", "static_mode": "residual", "v": "none", "g": True, "roc_auc": on},
                ]
            fold_a = self._summary(root, "a", "stage3", rows(0.60, 0.63))
            fold_b = self._summary(root, "b", "stage3", rows(0.61, 0.62))
            masking = []
            for name, delta in (("a", -0.02), ("b", -0.01)):
                path = root / ("mask_" + name + ".json")
                path.write_text(json.dumps({"conditions": {"mask_g": {"delta_auc_vs_all": delta}}}), encoding="utf-8")
                masking.append((name, path))
            result = self._run(root, "g", fold_a, fold_b, masking)
            self.assertEqual(result["decision"]["selected"], "with_g")
            self.assertTrue(result["decision"]["g_gate_passes"])


if __name__ == "__main__":
    unittest.main()
