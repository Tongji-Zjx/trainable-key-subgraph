from __future__ import absolute_import, division, print_function

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiViewStageCompletionAuditTest(unittest.TestCase):
    def _fixture(self, base):
        root = base / "experiment"
        provenance = {
            "protocol_sha256": "protocol",
            "selector_checkpoint_sha256": "selector",
            "feature_schema_sha256": "schema",
        }
        manifests = {}
        for offset, split in enumerate(("train", "validation", "test")):
            manifest = root / "full" / split / "manifest.json"
            _write(
                manifest,
                dict(
                    provenance,
                    artifact_type="theory_guided_multiview_critical_manifest",
                    split=split,
                    records=[
                        {"sample_key": "{}/zero".format(split), "label": 0},
                        {"sample_key": "{}/one".format(split), "label": 1},
                    ],
                ),
            )
            manifests[split] = manifest
            _write(
                root / "full" / split / "audit.json",
                {
                    "artifact_type": "multiview_critical_stage0_audit",
                    "split": split,
                    "sample_count": 2,
                    "fgw_convergence_fraction": 0.8,
                    "singleton_object_fraction": 0.2,
                },
            )
        scaler = SimpleNamespace(
            train_manifest_sha256=_sha(manifests["train"]),
            protocol_sha256="protocol",
            selector_checkpoint_sha256="selector",
            feature_schema_sha256="schema",
        )
        torch.save(
            {
                "artifact_type": "multiview_train_scaler",
                "scaler": scaler,
            },
            str(root / "scaler.pt"),
        )

        _write(
            root / "stage1" / "summary" / "summary.json",
            {
                "test_used": False,
                "conditions": [
                    {"condition": name, "v": "none", "g": False}
                    for name in ("S_stable", "S_neural", "S_residual")
                ],
                "decision": {"best_validation_condition": "S_residual"},
            },
        )
        _write(
            root / "stage2" / "summary" / "summary.json",
            {
                "test_used": False,
                "conditions": [
                    {"condition": mode, "v": mode, "g": False}
                    for mode in ("none", "legacy", "uot", "shuffled")
                ],
                "decision": {
                    "validation_screen_passes": False,
                    "paired_oof_gate_evaluated": False,
                },
            },
        )
        _write(
            root / "stage2" / "frozen_selection.json",
            {"v_mode": "legacy", "test_used": False},
        )
        _write(
            root / "stage3" / "summary" / "summary.json",
            {
                "test_used": False,
                "conditions": [
                    {"condition": "without_g", "v": "legacy", "g": False},
                    {"condition": "with_g", "v": "legacy", "g": True},
                ],
                "decision": {"g_auc_delta": -0.01},
            },
        )
        _write(
            root / "stage3" / "with_g" / "validation_channel_masking.json",
            {"artifact_type": "multiview_frozen_channel_masking_diagnostic"},
        )
        _write(
            root / "stage3" / "frozen_selection.json",
            {
                "test_used": False,
                "source": "validation_only_stage3_gate_and_masking",
            },
        )
        _write(
            root / "stage4" / "summary.json",
            {
                "test_used_for_selection": False,
                "test_threshold_refit": False,
                "models": {"author": {}, "critical": {}, "fusion": {}},
            },
        )
        (root / "stage4" / "COMPLETE").touch()
        smoke = base / "smoke"
        smoke.mkdir()
        (smoke / "best_checkpoint.pt").touch()
        overfit = base / "overfit"
        _write(
            overfit / "history.json",
            [
                {
                    "train": {
                        "classification_loss": 1.0,
                        "q_loss": 1.0,
                        "delta_q_loss": 1.0,
                    }
                },
                {
                    "train": {
                        "classification_loss": 0.1,
                        "q_loss": 0.2,
                        "delta_q_loss": 0.3,
                    }
                },
            ],
        )
        return root, smoke, overfit

    def test_complete_fixture_passes_all_frozen_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, smoke, overfit = self._fixture(base)
            output = base / "audit"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_multiview_stage_completion.py",
                    "--root",
                    str(root),
                    "--smoke-dir",
                    str(smoke),
                    "--overfit-dir",
                    str(overfit),
                    "--output-dir",
                    str(output),
                    "--require-complete",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            result = json.loads(
                (output / "completion_audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
