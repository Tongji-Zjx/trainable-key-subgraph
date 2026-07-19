from __future__ import absolute_import, print_function

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.crossfit.fold_protocol import prepare_fold_protocol  # noqa: E402
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import read_split_assignments  # noqa: E402


class CrossfitFoldProtocolTest(unittest.TestCase):
    def test_fold_zero_is_portable_complete_and_leak_free(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / "outputs")) as temporary:
            result = prepare_fold_protocol(
                PROJECT_ROOT, PROJECT_ROOT / "configs/crossfit/fold_assignments.json",
                PROJECT_ROOT / "configs/data_protocol_all_samples.json", 0,
                Path(temporary),
            )
            protocol = validate_data_protocol(result["protocol"], PROJECT_ROOT)
            rows = read_split_assignments(PROJECT_ROOT / protocol["paths"]["splits_csv"])
            self.assertEqual(len(rows), 938)
            self.assertEqual({row.split for row in rows}, {"train", "validation", "test"})
            groups = {name: {row.group_id for row in rows if row.split == name} for name in ("train", "validation", "test")}
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            payload = json.loads(result["protocol"].read_text(encoding="utf-8"))
            self.assertEqual(payload["crossfit"]["outer_fold"], 0)
            self.assertFalse(Path(payload["paths"]["splits_csv"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
