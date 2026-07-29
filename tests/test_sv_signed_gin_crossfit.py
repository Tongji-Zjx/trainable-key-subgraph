from __future__ import absolute_import, division, print_function

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.sv_signed_gin_runner import (  # noqa: E402
    build_sv_crossfit_fold_commands,
)
from keysubgraph.crossfit.sv_signed_gin_summary import (  # noqa: E402
    summarize_sv_signed_gin_crossfit,
)


VARIANT = "signed_gin_static_variation"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SVSignedGINCrossfitRunnerTest(unittest.TestCase):
    def test_plan_is_ordered_resumable_and_uses_fold_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                root,
                1,
                variants=(VARIANT,),
                device="cuda",
                seed=42,
            )
            self.assertEqual(
                [item[0] for item in plan],
                [
                    "selector",
                    "cache_train",
                    "cache_validation",
                    "cache_test",
                    "scaler",
                    "train_{}".format(VARIANT),
                    "evaluate_{}".format(VARIANT),
                ],
            )
            for _, command, artifact in plan:
                joined = " ".join(command)
                self.assertIn("fold_1", joined)
                self.assertTrue(str(artifact).startswith(str(root.resolve())))
            evaluation = plan[-1]
            self.assertIn("--threshold-strategy", evaluation[1])
            self.assertIn("balanced_accuracy", evaluation[1])

    def test_late_fusion_plan_enables_all_frozen_improvements(self):
        variant = "signed_gin_multibranch_late_fusion"
        with tempfile.TemporaryDirectory() as directory:
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                Path(directory),
                0,
                variants=(variant,),
                device="cpu",
                seed=42,
                selector_epochs=1,
                model_epochs=1,
                num_workers=0,
            )
        train_command = next(
            command
            for name, command, _ in plan
            if name == "train_{}".format(variant)
        )
        expected = (
            "--message-mode",
            "signed_normalized",
            "--pooling",
            "mean_std",
            "--gin-residual",
            "--gin-jumping-knowledge",
            "--auxiliary-loss-weight",
            "0.25",
        )
        for value in expected:
            self.assertIn(value, train_command)


class SVSignedGINCrossfitSummaryTest(unittest.TestCase):
    def _fixture(self, root):
        assignment_rows = []
        predictions = {
            0: (
                {"sample_key": "a", "site": "S", "label": 0,
                 "positive_probability": 0.1},
                {"sample_key": "b", "site": "S", "label": 1,
                 "positive_probability": 0.9},
            ),
            1: (
                {"sample_key": "c", "site": "S", "label": 0,
                 "positive_probability": 0.4},
                {"sample_key": "d", "site": "S", "label": 1,
                 "positive_probability": 0.6},
            ),
        }
        for fold, rows in predictions.items():
            for row in rows:
                assignment_rows.append(
                    {
                        "outer_fold": fold,
                        "role": "outer_test",
                        "sample_key": row["sample_key"],
                        "sample_id": row["sample_key"],
                        "site": row["site"],
                        "subject_id": row["sample_key"],
                        "session_id": "",
                        "label": row["label"],
                        "group_id": "S::{}".format(row["sample_key"]),
                    }
                )
            evaluation = {
                "split": "test",
                "variant": VARIANT,
                "threshold_fit_split": "validation",
                "threshold_strategy": "balanced_accuracy",
                "threshold": 0.5,
                "metrics": {
                    "roc_auc": 1.0,
                    "site_stratified_roc_auc": 1.0,
                    "balanced_accuracy": 1.0,
                    "accuracy": 1.0,
                    "f1": 1.0,
                },
                "predictions": list(rows),
            }
            _write_json(
                root
                / "fold_{}".format(fold)
                / "models"
                / "{}_seed42".format(VARIANT)
                / "outer_test_evaluation.json",
                evaluation,
            )
        assignments = root / "assignments" / "fold_assignments.json"
        _write_json(
            assignments,
            {
                "immutable": True,
                "purpose": "confirmatory_cross_fitted_fold_roles",
                "num_outer_folds": 2,
                "assignments": assignment_rows,
            },
        )
        return assignments

    def test_summary_requires_exact_oof_coverage_and_uses_fold_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = self._fixture(root)
            result = summarize_sv_signed_gin_crossfit(
                root, assignments
            )
            metrics = result["metrics"]
            self.assertEqual(metrics["sample_count"], 4)
            self.assertAlmostEqual(metrics["pooled_oof_roc_auc"], 1.0)
            self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
            self.assertTrue(result["summary_markdown"].is_file())
            self.assertTrue(result["predictions_csv"].is_file())
            with self.assertRaises(FileExistsError):
                summarize_sv_signed_gin_crossfit(root, assignments)

    def test_summary_rejects_a_fold_prediction_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = self._fixture(root)
            evaluation_path = (
                root
                / "fold_0"
                / "models"
                / "{}_seed42".format(VARIANT)
                / "outer_test_evaluation.json"
            )
            payload = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            payload["predictions"] = payload["predictions"][:1]
            _write_json(evaluation_path, payload)
            with self.assertRaisesRegex(
                ValueError, "frozen fold assignment"
            ):
                summarize_sv_signed_gin_crossfit(root, assignments)


if __name__ == "__main__":
    unittest.main()
