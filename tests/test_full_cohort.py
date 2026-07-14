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

from keysubgraph.data.data_protocol import (  # noqa: E402
    _portable_path,
    freeze_data_protocol,
    validate_data_protocol,
)
from keysubgraph.data.data_split import read_sample_index, read_split_assignments  # noqa: E402
from keysubgraph.data.full_cohort import (  # noqa: E402
    FULL_COHORT_MODE,
    create_full_cohort_assignments,
    write_full_cohort_artifacts,
)


class FullCohortTest(unittest.TestCase):
    def test_portable_path_does_not_expand_a_dataset_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_root = root / "project"
            external_data = root / "external_data"
            project_root.mkdir()
            external_data.mkdir()
            link = project_root / "data"
            try:
                link.symlink_to(external_data, target_is_directory=True)
            except OSError as error:
                self.skipTest("directory symlinks are unavailable: {}".format(error))

            self.assertEqual(_portable_path(link, project_root), "data")

    def test_every_indexed_sample_is_frozen_into_all_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "data"
            index_path = root / "index" / "sample_index.csv"
            index_path.parent.mkdir(parents=True)
            fields = [
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
            for index, label in enumerate((0, 1, 0)):
                relative_path = "SITE/{}/SITE_s{}_1.pt".format(label, index)
                data_path = dataset_root / relative_path
                data_path.parent.mkdir(parents=True, exist_ok=True)
                data_path.touch()
                rows.append(
                    {
                        "sample_key": "SITE/SITE_s{}_1".format(index),
                        "sample_id": "SITE_s{}_1".format(index),
                        "site": "SITE",
                        "subject_id": "s{}".format(index),
                        "session_id": "1",
                        "label": label,
                        "relative_path": relative_path,
                        "included": "True",
                    }
                )
            with index_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            samples = read_sample_index(index_path)
            assignments = create_full_cohort_assignments(samples, seed=23)
            artifacts = write_full_cohort_artifacts(
                assignments, index_path, root / "assignments"
            )
            protocol_path = root / "protocol.json"
            protocol = freeze_data_protocol(
                root,
                dataset_root,
                index_path,
                artifacts["csv"],
                artifacts["json"],
                protocol_path,
            )

            loaded = read_split_assignments(artifacts["csv"])
            with artifacts["json"].open("r", encoding="utf-8") as handle:
                assignment_payload = json.load(handle)
            self.assertEqual(len(loaded), 3)
            self.assertEqual({item.split for item in loaded}, {"all"})
            self.assertEqual(assignment_payload["class_counts"], {"0": 2, "1": 1})
            self.assertEqual(protocol["experiment_mode"], FULL_COHORT_MODE)
            self.assertEqual(protocol["split_ratios"], {"all": 1.0})
            self.assertEqual(validate_data_protocol(protocol_path, root), protocol)


if __name__ == "__main__":
    unittest.main()
