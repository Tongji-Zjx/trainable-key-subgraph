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

from keysubgraph.data.crossfit_controls import (  # noqa: E402
    build_key_random_payloads,
    freeze_key_random_inventory,
)
from tests.test_analysis import _control_sample, _record  # noqa: E402


def _payload():
    key = _record()
    key.update({
        "time_index": 0,
        "node_ids": [0, 1, 2],
        "edge_index": [[0, 1], [1, 2]],
        "score_connectivity": 1.0,
    })
    return {"label": 1, "timepoints": [{
        "time_index": 0, "num_valid_subgraphs": 1,
        "subgraphs": [key], "candidate_pool": [key],
    }]}


class CrossfitControlTest(unittest.TestCase):
    def test_random_is_reproducible_and_exactly_budget_matched(self):
        sample = _control_sample()
        first, first_audit = build_key_random_payloads(sample, _payload(), 19, 0)
        second, second_audit = build_key_random_payloads(sample, _payload(), 19, 0)
        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)
        self.assertTrue(first_audit["included"])
        key = first["key"]["timepoints"][0]["subgraphs"][0]
        random = first["random"]["timepoints"][0]["subgraphs"][0]
        self.assertEqual(len(key["node_ids"]), len(random["node_ids"]))
        self.assertEqual(len(key["edge_index"]), len(random["edge_index"]))
        self.assertNotEqual(
            (key["node_ids"], key["edge_index"]),
            (random["node_ids"], random["edge_index"]),
        )

    def test_frozen_inventory_is_label_free_shared_and_immutable(self):
        _, audit = build_key_random_payloads(_control_sample(), _payload(), 23, 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold_0.json"
            first = freeze_key_random_inventory(
                [{"sample_key": "SITE/sample", "audit": audit}], path, 0, 23
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first, loaded)
            self.assertNotIn("label", json.dumps(loaded))
            self.assertEqual(loaded["sources"], ["key", "random"])
            self.assertNotIn("model_seed", loaded)
            with self.assertRaises(FileExistsError):
                freeze_key_random_inventory(
                    [{"sample_key": "SITE/sample", "audit": audit}], path, 0, 23
                )

    def test_freezer_rejects_fields_that_could_change_selection(self):
        _, audit = build_key_random_payloads(_control_sample(), _payload(), 29, 0)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                freeze_key_random_inventory(
                    [{"sample_key": "x", "audit": audit, "label": 1}],
                    Path(temporary) / "bad.json", 0, 29,
                )


if __name__ == "__main__":
    unittest.main()
