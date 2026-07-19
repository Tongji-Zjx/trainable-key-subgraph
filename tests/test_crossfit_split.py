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

from keysubgraph.data.crossfit_split import (  # noqa: E402
    CrossfitFoldAssignment,
    OuterFoldAssignment,
    create_crossfit_fold_assignments,
    create_outer_folds,
    read_crossfit_fold_artifacts,
    read_outer_fold_artifacts,
    validate_crossfit_fold_assignments,
    validate_outer_folds,
    write_crossfit_fold_artifacts,
    write_outer_fold_artifacts,
)
from keysubgraph.data.data_split import IndexSample  # noqa: E402


def _samples():
    rows = []
    for label in (0, 1):
        for subject_index in range(10):
            subject_id = "subject_{}_{}".format(label, subject_index)
            for session_index in range(2):
                sample_id = "{}_session{}".format(subject_id, session_index)
                rows.append(
                    IndexSample(
                        sample_key="SITE/{}".format(sample_id),
                        sample_id=sample_id,
                        site="SITE",
                        subject_id=subject_id,
                        session_id=str(session_index),
                        label=label,
                        group_id="unused",
                        relative_path="unused/{}.pt".format(sample_id),
                    )
                )
    return rows


class CrossfitOuterSplitTest(unittest.TestCase):
    def test_outer_folds_are_reproducible_grouped_and_stratified(self):
        first = create_outer_folds(_samples(), num_folds=5, seed=202607)
        second = create_outer_folds(_samples(), num_folds=5, seed=202607)
        self.assertEqual(first, second)
        summary = validate_outer_folds(first, 5)
        self.assertEqual(summary["total_samples"], 40)
        self.assertEqual(summary["total_subjects"], 20)
        for fold in summary["folds"].values():
            self.assertEqual(fold["sample_count"], 8)
            self.assertEqual(fold["class_counts"], {"0": 4, "1": 4})
        subject_folds = {}
        for item in first:
            previous = subject_folds.setdefault(item.subject_id, item.outer_test_fold)
            self.assertEqual(previous, item.outer_test_fold)

    def test_artifacts_are_immutable_and_bound_to_source_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample_index.csv"
            source.write_text("frozen index\n", encoding="utf-8")
            result = write_outer_fold_artifacts(
                create_outer_folds(_samples()), root / "folds", source
            )
            payload, assignments = read_outer_fold_artifacts(result["json"], source)
            self.assertEqual(payload["num_outer_folds"], 5)
            self.assertEqual(len(assignments), 40)
            with self.assertRaises(FileExistsError):
                write_outer_fold_artifacts(assignments, root / "folds", source)
            source.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_outer_fold_artifacts(result["json"], source)

    def test_validation_rejects_subject_crossing_folds(self):
        assignments = create_outer_folds(_samples())
        target = assignments[0]
        assignments.append(
            OuterFoldAssignment(
                sample_key="duplicate-subject-new-sample",
                sample_id="new",
                site=target.site,
                subject_id=target.subject_id,
                session_id="new",
                label=target.label,
                outer_test_fold=(target.outer_test_fold + 1) % 5,
            )
        )
        with self.assertRaisesRegex(ValueError, "subject crosses"):
            validate_outer_folds(assignments, 5)

    def test_empty_subject_id_is_rejected(self):
        sample = _samples()[0]
        invalid = IndexSample(
            sample_key=sample.sample_key,
            sample_id=sample.sample_id,
            site=sample.site,
            subject_id="",
            session_id=sample.session_id,
            label=sample.label,
            group_id=sample.group_id,
            relative_path=sample.relative_path,
        )
        with self.assertRaisesRegex(ValueError, "non-empty subject_id"):
            create_outer_folds([invalid] + _samples()[1:])


class CrossfitInnerSplitTest(unittest.TestCase):
    def test_inner_roles_are_reproducible_and_outer_test_is_isolated(self):
        samples = _samples()
        outer = create_outer_folds(samples)
        first = create_crossfit_fold_assignments(samples, outer)
        second = create_crossfit_fold_assignments(samples, outer)
        self.assertEqual(first, second)
        summary = validate_crossfit_fold_assignments(first, 5, len(samples))
        outer_by_key = {item.sample_key: item.outer_test_fold for item in outer}
        for item in first:
            self.assertEqual(
                item.role == "outer_test",
                outer_by_key[item.sample_key] == item.outer_fold,
            )
        for fold in summary["folds"].values():
            self.assertFalse(fold["checks"]["sample_role_overlap"])
            self.assertFalse(fold["checks"]["subject_role_overlap"])
            self.assertTrue(fold["checks"]["all_roles_have_both_classes"])

    def test_inner_artifacts_bind_outer_split_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample_index.csv"
            source.write_text("frozen index\n", encoding="utf-8")
            outer = create_outer_folds(_samples())
            outer_result = write_outer_fold_artifacts(
                outer, root / "folds", source
            )
            assignments = create_crossfit_fold_assignments(_samples(), outer)
            result = write_crossfit_fold_artifacts(
                assignments,
                root / "folds",
                outer_result["json"],
                source,
            )
            payload, loaded = read_crossfit_fold_artifacts(
                result["json"], outer_result["json"], source
            )
            self.assertEqual(payload["inner_validation_ratio"], 0.1875)
            self.assertEqual(loaded, assignments)
            Path(outer_result["json"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outer split hash mismatch"):
                read_crossfit_fold_artifacts(
                    result["json"], outer_result["json"], source
                )

    def test_validation_rejects_subject_crossing_inner_roles(self):
        assignments = create_crossfit_fold_assignments(
            _samples(), create_outer_folds(_samples())
        )
        target = next(item for item in assignments if item.role == "inner_train")
        assignments.append(
            CrossfitFoldAssignment(
                outer_fold=target.outer_fold,
                sample_key="new-sample",
                sample_id="new",
                site=target.site,
                subject_id=target.subject_id,
                session_id="new",
                label=target.label,
                role="inner_validation",
            )
        )
        with self.assertRaisesRegex(ValueError, "subject crosses"):
            validate_crossfit_fold_assignments(assignments, 5)


if __name__ == "__main__":
    unittest.main()
