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

from keysubgraph.crossfit.structured_short_term_runner import (  # noqa: E402
    build_structured_short_term_crossfit_fold_commands,
)
from keysubgraph.crossfit.structured_short_term_summary import (  # noqa: E402
    MODEL_NAME,
    summarize_structured_short_term_crossfit,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class StructuredShortTermCrossfitRunnerTest(unittest.TestCase):
    def test_plan_is_fold_local_ordered_and_evaluates_validation_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_structured_short_term_crossfit_fold_commands(
                PROJECT_ROOT,
                root,
                2,
                device="cuda",
                seed=43,
                epochs=12,
                batch_size=3,
                evaluation_batch_size=5,
                num_workers=1,
            )
        self.assertEqual(
            [item[0] for item in plan],
            [
                "standardizer",
                "train",
                "evaluate_validation",
                "evaluate_test",
            ],
        )
        for _, command, artifact in plan:
            self.assertIn("fold_2", " ".join(command))
            self.assertTrue(str(artifact).startswith(str(root.resolve())))
        train = plan[1][1]
        self.assertEqual(train[train.index("--epochs") + 1], "12")
        self.assertEqual(train[train.index("--batch-size") + 1], "3")
        for _, command, _ in plan[2:]:
            self.assertIn("--threshold-strategy", command)
            self.assertIn("balanced_accuracy", command)

    def test_plan_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            build_structured_short_term_crossfit_fold_commands(
                PROJECT_ROOT,
                Path("output"),
                -1,
            )
        with self.assertRaises(ValueError):
            build_structured_short_term_crossfit_fold_commands(
                PROJECT_ROOT,
                Path("output"),
                0,
                batch_size=0,
            )


class StructuredShortTermCrossfitSummaryTest(unittest.TestCase):
    def _prediction(self, key, label, probability):
        return {
            "sample_key": key,
            "site": "S",
            "label": label,
            "positive_probability": probability,
            "prediction": int(probability >= 0.5),
        }

    def _evaluation(self, split, predictions, threshold=0.5):
        return {
            "model_name": MODEL_NAME,
            "split": split,
            "threshold_source": "frozen_validation",
            "threshold_fit_split": "validation",
            "threshold_strategy": "balanced_accuracy",
            "threshold": threshold,
            "metrics": {},
            "predictions": predictions,
        }

    def _fixture(self, root):
        outer = {
            0: [
                self._prediction("a", 0, 0.1),
                self._prediction("b", 1, 0.9),
            ],
            1: [
                self._prediction("c", 0, 0.2),
                self._prediction("d", 1, 0.8),
            ],
        }
        validation = {0: outer[1], 1: outer[0]}
        assignments = []
        for fold in (0, 1):
            for role, rows in (
                ("outer_test", outer[fold]),
                ("inner_validation", validation[fold]),
            ):
                for row in rows:
                    assignments.append(
                        {
                            "outer_fold": fold,
                            "role": role,
                            "sample_key": row["sample_key"],
                            "site": row["site"],
                            "label": row["label"],
                        }
                    )
            evaluation = (
                root
                / "fold_{}".format(fold)
                / "structured_short_term"
                / "evaluation_seed42"
            )
            _write_json(
                evaluation / "validation_evaluation.json",
                self._evaluation("validation", validation[fold]),
            )
            _write_json(
                evaluation / "test_evaluation.json",
                self._evaluation("test", outer[fold]),
            )
        assignment_path = (
            root / "assignments" / "fold_assignments.json"
        )
        _write_json(
            assignment_path,
            {
                "immutable": True,
                "purpose": "confirmatory_cross_fitted_fold_roles",
                "num_outer_folds": 2,
                "assignments": assignments,
            },
        )
        return assignment_path

    def test_summary_audits_exact_coverage_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = self._fixture(root)
            result = summarize_structured_short_term_crossfit(
                root,
                assignments,
            )
            self.assertEqual(result["metrics"]["sample_count"], 4)
            self.assertAlmostEqual(
                result["metrics"]["pooled_oof_roc_auc"],
                1.0,
            )
            self.assertAlmostEqual(
                result["metrics"]["balanced_accuracy"],
                1.0,
            )
            self.assertTrue(result["summary_json"].is_file())
            self.assertTrue(result["predictions_csv"].is_file())
            self.assertTrue(result["summary_markdown"].is_file())
            payload = json.loads(
                result["summary_json"].read_text(encoding="utf-8")
            )
            self.assertTrue(
                payload["checks"]["every_sample_predicted_once"]
            )
            self.assertFalse(
                payload["checks"]["test_threshold_fitting"]
            )
            with self.assertRaises(FileExistsError):
                summarize_structured_short_term_crossfit(
                    root,
                    assignments,
                )

    def test_summary_rejects_outer_prediction_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = self._fixture(root)
            path = (
                root
                / "fold_0"
                / "structured_short_term"
                / "evaluation_seed42"
                / "test_evaluation.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["predictions"] = payload["predictions"][:1]
            _write_json(path, payload)
            with self.assertRaisesRegex(
                ValueError,
                "frozen fold assignment",
            ):
                summarize_structured_short_term_crossfit(
                    root,
                    assignments,
                )

    def test_summary_rejects_validation_test_threshold_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = self._fixture(root)
            path = (
                root
                / "fold_1"
                / "structured_short_term"
                / "evaluation_seed42"
                / "test_evaluation.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["threshold"] = 0.6
            for row in payload["predictions"]:
                row["prediction"] = int(
                    row["positive_probability"] >= 0.6
                )
            _write_json(path, payload)
            with self.assertRaisesRegex(
                ValueError,
                "different thresholds",
            ):
                summarize_structured_short_term_crossfit(
                    root,
                    assignments,
                )


if __name__ == "__main__":
    unittest.main()
