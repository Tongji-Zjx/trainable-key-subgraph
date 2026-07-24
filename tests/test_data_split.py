from __future__ import absolute_import, division, print_function

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.data_split import (  # noqa: E402
    SplitAssignment,
    SplitConfig,
    create_data_splits,
    read_sample_index,
    read_split_assignments,
    summarize_assignments,
    validate_assignments,
    write_split_artifacts,
)


class DataSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_index(self):
        path = self.root / "sample_index.csv"
        fieldnames = [
            "sample_key",
            "sample_id",
            "site",
            "subject_id",
            "session_id",
            "label",
            "relative_path",
            "included",
        ]
        rows = []
        for index in range(40):
            label = 0 if index < 24 else 1
            subject_id = "subject_{}".format(index)
            sample_id = "SITE_{}_1".format(subject_id)
            rows.append(
                {
                    "sample_key": "SITE/{}".format(sample_id),
                    "sample_id": sample_id,
                    "site": "SITE",
                    "subject_id": subject_id,
                    "session_id": "1",
                    "label": str(label),
                    "relative_path": "SITE/{}/{}.pt".format(label, sample_id),
                    "included": "True",
                }
            )
        # A second scan of the same subject verifies group isolation.
        rows.append(
            {
                "sample_key": "SITE/SITE_subject_0_2",
                "sample_id": "SITE_subject_0_2",
                "site": "SITE",
                "subject_id": "subject_0",
                "session_id": "2",
                "label": "0",
                "relative_path": "SITE/0/SITE_subject_0_2.pt",
                "included": "True",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_split_is_reproducible_stratified_and_group_aware(self):
        index_path = self._write_index()
        samples = read_sample_index(index_path)
        config = SplitConfig(seed=123, max_class_ratio_deviation=0.10)

        first = create_data_splits(samples, config)
        second = create_data_splits(samples, config)

        self.assertEqual(first, second)
        by_subject = {}
        for assignment in first:
            previous = by_subject.setdefault(assignment.group_id, assignment.split)
            self.assertEqual(previous, assignment.split)
        summary = summarize_assignments(first, config)
        self.assertFalse(summary["checks"]["sample_overlap"])
        self.assertFalse(summary["checks"]["group_overlap"])
        self.assertTrue(summary["checks"]["class_ratios_reasonable"])
        self.assertEqual(
            sum(item["sample_count"] for item in summary["splits"].values()), 41
        )

    def test_artifacts_are_immutable_by_default_and_include_assignments(self):
        index_path = self._write_index()
        config = SplitConfig(max_class_ratio_deviation=0.10)
        assignments = create_data_splits(read_sample_index(index_path), config)
        output_dir = self.root / "splits"

        paths = write_split_artifacts(assignments, output_dir, index_path, config)

        with paths["json"].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertNotIn(b"\r\n", paths["json"].read_bytes())
        self.assertEqual(len(payload["assignments"]), len(assignments))
        self.assertEqual(payload["seed"], 42)
        self.assertTrue(payload["group_aware"])
        self.assertEqual(len(payload["source_index_sha256"]), 64)
        loaded = read_split_assignments(paths["csv"])
        self.assertEqual(loaded, assignments)
        with self.assertRaises(FileExistsError):
            write_split_artifacts(assignments, output_dir, index_path, config)

    def test_validation_rejects_a_group_crossing_splits(self):
        base = {
            "sample_id": "sample",
            "site": "SITE",
            "subject_id": "subject",
            "session_id": "1",
            "group_id": "SITE::subject",
            "label": 0,
            "relative_path": "SITE/0/sample.pt",
            "seed": 42,
        }
        assignments = []
        for index, split in enumerate(("train", "validation", "test")):
            row = dict(base)
            row["sample_key"] = "SITE/sample_{}".format(index)
            row["split"] = split
            assignments.append(SplitAssignment(**row))
        with self.assertRaisesRegex(ValueError, "subject groups overlap"):
            validate_assignments(
                assignments, SplitConfig(max_class_ratio_deviation=1.0)
            )


if __name__ == "__main__":
    unittest.main()
