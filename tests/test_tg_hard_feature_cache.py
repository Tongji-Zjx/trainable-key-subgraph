from __future__ import absolute_import, division, print_function

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.features import (  # noqa: E402
    HardExportFeatureAdapter,
    load_hard_graph_cache,
    save_hard_graph_cache,
)


def _subgraph(nodes, edge, weight, score=0.8):
    return {
        "node_ids": list(nodes),
        "edge_index": [list(edge)],
        "original_edge_weights": [weight],
        "candidate_score": score,
        "seed_node": nodes[0],
    }


def _valid_timepoint(index, start, nodes, names, communities, edge, weight):
    return {
        "time_index": index,
        "time_start": start,
        "window_valid": True,
        "hard_union_available": True,
        "union_node_ids": list(nodes),
        "union_node_names": list(names),
        "union_community_labels": list(communities),
        "union_edge_index": [list(edge)],
        "union_original_edge_weights": [weight],
        "subgraphs": [_subgraph(nodes, edge, weight)],
    }


class TGHardFeatureCacheTest(unittest.TestCase):
    def _payload(self):
        return {
            "schema_version": 1,
            "sample_key": "SITE/sample",
            "sample_id": "sample",
            "label": 1,
            "split": "train",
            "edge_presence_threshold": 0.0,
            "data_protocol_sha256": "protocol",
            "checkpoint_sha256": "teacher",
            "timepoints": [
                _valid_timepoint(0, 0.0, (10, 11), ("a", "b"), (0, 0), (10, 11), -0.5),
                {
                    "time_index": 1,
                    "time_start": 1.0,
                    "window_valid": False,
                    "hard_union_available": False,
                    "subgraphs": [],
                },
                _valid_timepoint(2, 2.0, (11, 12), ("b", "c"), (0, 1), (11, 12), 0.7),
            ],
        }

    def test_adapter_recomputes_signed_features_and_masks_empty_window(self):
        cache = HardExportFeatureAdapter().build(self._payload())
        self.assertEqual(cache.time_mask, (True, False, True))
        self.assertTrue(cache.eligible_for_stage_c)
        first = cache.windows[0]
        third = cache.windows[2]
        self.assertEqual(tuple(first.features.node_features.shape), (2, 13))
        self.assertEqual(tuple(first.features.edge_features.shape), (2, 2, 4))
        self.assertLess(float(first.features.edge_features[0, 1, 0]), 0.0)
        self.assertGreater(float(first.features.edge_features[0, 1, 1]), 0.0)
        self.assertFalse(bool(first.features.delta_degree_mask.any()))
        self.assertFalse(bool(third.features.delta_degree_mask.any()))
        self.assertEqual(tuple(first.subgraphs[0].union_node_indices.tolist()), (0, 1))

    def test_adapter_does_not_read_full_graph_feature_fields(self):
        payload = self._payload()
        polluted = copy.deepcopy(payload)
        polluted["timepoints"][0]["full_graph_node_features"] = [[999.0] * 13] * 2
        clean = HardExportFeatureAdapter().build(payload)
        other = HardExportFeatureAdapter().build(polluted)
        self.assertTrue(torch.equal(
            clean.windows[0].features.node_features,
            other.windows[0].features.node_features,
        ))

    def test_cache_round_trip_and_minimum_window_exclusion(self):
        payload = self._payload()
        payload["timepoints"][2]["window_valid"] = False
        payload["timepoints"][2]["hard_union_available"] = False
        cache = HardExportFeatureAdapter().build(payload)
        self.assertFalse(cache.eligible_for_stage_c)
        self.assertEqual(cache.exclusion_reason, "fewer_than_two_valid_hard_windows")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.pt"
            save_hard_graph_cache(cache, path)
            loaded = load_hard_graph_cache(path)
        self.assertEqual(loaded.sample_key, cache.sample_key)
        self.assertEqual(loaded.time_mask, cache.time_mask)
        self.assertTrue(torch.equal(
            loaded.windows[0].features.edge_features,
            cache.windows[0].features.edge_features,
        ))


if __name__ == "__main__":
    unittest.main()
