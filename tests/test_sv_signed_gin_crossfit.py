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
    SV_CROSSFIT_DEFAULT_VARIANT,
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
    def test_default_plan_trains_formal_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                Path(directory),
                0,
                device="cpu",
                selector_epochs=1,
                model_epochs=1,
                num_workers=0,
            )
        stage = "train_{}".format(SV_CROSSFIT_DEFAULT_VARIANT)
        train_command = next(
            command for name, command, _ in plan if name == stage
        )
        self.assertEqual(
            SV_CROSSFIT_DEFAULT_VARIANT,
            "signed_gin_multibranch_late_fusion",
        )
        self.assertIn("--message-mode", train_command)
        self.assertIn("signed_normalized", train_command)
        self.assertIn("--pooling", train_command)
        self.assertIn("mean_std", train_command)
        self.assertIn("--gin-residual", train_command)
        self.assertIn("--gin-jumping-knowledge", train_command)
        self.assertIn("--gin-compact-readout", train_command)
        self.assertIn("--gin-batch-normalization", train_command)
        self.assertIn("--auxiliary-loss-weight", train_command)
        self.assertIn("0.25", train_command)

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
            "--gin-compact-readout",
            "--gin-batch-normalization",
            "--auxiliary-loss-weight",
            "0.25",
        )
        for value in expected:
            self.assertIn(value, train_command)

    def test_s_and_sv_plans_reuse_fold_artifacts_without_gin_baseline(self):
        variants = (
            "static_spectral_only",
            "static_spectral_variation_late_fusion",
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                Path(directory),
                2,
                variants=variants,
                device="cuda",
                seed=42,
            )
        stages = [name for name, _, _ in plan]
        for variant in variants:
            self.assertIn("train_{}".format(variant), stages)
            self.assertIn("evaluate_{}".format(variant), stages)
        self.assertNotIn(
            "train_signed_gin_multibranch_late_fusion", stages
        )

    def test_static_anchor_residual_plan_enables_two_stage_training(self):
        variant = "signed_gin_static_anchor_residual"
        with tempfile.TemporaryDirectory() as directory:
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                Path(directory),
                0,
                variants=(variant,),
                device="cpu",
                seed=42,
                selector_epochs=1,
                model_epochs=3,
                num_workers=0,
            )
        train_command = next(
            command
            for name, command, _ in plan
            if name == "train_{}".format(variant)
        )
        self.assertIn("--static-anchor-epochs", train_command)
        static_index = train_command.index("--static-anchor-epochs")
        self.assertEqual(train_command[static_index + 1], "3")
        self.assertIn(
            "--residual-gate-penalty-weight", train_command
        )

    def test_residual_attention_plan_changes_only_attention_flag(self):
        variants = (
            "signed_gin_static_anchor_residual",
            "signed_gin_static_anchor_residual_attention",
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = build_sv_crossfit_fold_commands(
                PROJECT_ROOT,
                Path(directory),
                0,
                variants=variants,
                device="cpu",
                seed=42,
                selector_epochs=1,
                model_epochs=1,
                num_workers=0,
            )
        commands = {
            name: command
            for name, command, _ in plan
            if name.startswith("train_")
        }
        v1a = commands["train_{}".format(variants[0])]
        v1b = commands["train_{}".format(variants[1])]
        self.assertNotIn("--gin-residual-attention", v1a)
        self.assertIn("--gin-residual-attention", v1b)
        filtered = [
            value
            for value in v1b
            if value != "--gin-residual-attention"
        ]
        # Run directories and variant values are expected to differ.
        differences = [
            (left, right)
            for left, right in zip(v1a, filtered)
            if left != right
        ]
        self.assertEqual(len(differences), 2)


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
            summary = json.loads(
                result["summary_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["primary_metric"], "outer_fold_roc_auc_mean"
            )
            self.assertTrue(summary["pooled_oof_roc_auc_is_auxiliary"])
            self.assertAlmostEqual(summary["primary_metric_value"], 1.0)
            markdown = result["summary_markdown"].read_text(
                encoding="utf-8"
            )
            self.assertLess(
                markdown.index("| **Mean fold AUROC"),
                markdown.index("| Pooled OOF AUROC"),
            )
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
