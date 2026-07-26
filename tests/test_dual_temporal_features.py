from __future__ import absolute_import, division, print_function

import tempfile
import unittest
from pathlib import Path

import torch

from keysubgraph.data.dual_temporal_artifact import (
    DualTemporalVariationRecord,
    load_dual_temporal_record,
    save_dual_temporal_record,
)
from keysubgraph.data.dual_temporal_manifest import (
    read_dual_temporal_manifest,
    write_dual_temporal_manifest,
)
from keysubgraph.features.dual_temporal_variation import (
    DualTemporalVariationExtractor,
)
from keysubgraph.features.hard_graph_features import HardGraphWindow
from keysubgraph.theory.tg_features import (
    SGWFeatureExtractor,
    SGWTheoryFeatureConfig,
)


def _window(adjacency, time_start):
    adjacency = torch.tensor(adjacency, dtype=torch.float32)
    count = int(adjacency.shape[0])
    return HardGraphWindow(
        adjacency=adjacency,
        communities=torch.arange(count, dtype=torch.long),
        node_names=tuple("n{}".format(index) for index in range(count)),
        time_start=float(time_start),
        edge_presence_threshold=0.0,
    )


class DualTemporalFeaturesTest(unittest.TestCase):
    def test_transition_values_follow_adjacent_spectral_states(self):
        windows = (
            _window(
                [[0.0, 0.4, -0.2], [0.4, 0.0, 0.1], [-0.2, 0.1, 0.0]],
                0,
            ),
            _window(
                [[0.0, 0.2, -0.5], [0.2, 0.0, 0.3], [-0.5, 0.3, 0.0]],
                1,
            ),
            _window(
                [[0.0, -0.1, -0.2], [-0.1, 0.0, 0.7], [-0.2, 0.7, 0.0]],
                2,
            ),
        )
        output = DualTemporalVariationExtractor().compute(windows)
        expected = (
            output.spectral_quantiles[1:]
            - output.spectral_quantiles[:-1]
        ).abs()
        torch.testing.assert_close(output.values, expected)
        self.assertEqual(tuple(output.values.shape), (2, 16))
        self.assertTrue(bool(output.mask.all()))
        exact = SGWFeatureExtractor(
            SGWTheoryFeatureConfig()
        ).compute_hard_graph_sequence(windows, (0.0, 1.0, 2.0))
        torch.testing.assert_close(
            output.values[output.mask].mean(dim=0),
            exact.h_variation,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        self.assertEqual(output.mask.tolist(), exact.transition_mask.tolist())

    def test_invalid_middle_window_creates_nonprefix_mask(self):
        first = _window(
            [[0.0, -0.3], [-0.3, 0.0]], 0
        )
        third = _window(
            [[0.0, 0.6], [0.6, 0.0]], 2
        )
        output = DualTemporalVariationExtractor().compute(
            (first, None, third)
        )
        self.assertEqual(output.mask.tolist(), [False, False])
        self.assertTrue(bool((output.values == 0.0).all()))

    def _record(self, key, split, path_seed=42):
        return DualTemporalVariationRecord(
            sample_key=key,
            label=0 if key.endswith("0") else 1,
            split=split,
            window_count=3,
            transition_values=torch.tensor(
                [[0.1] * 16, [0.0] * 16], dtype=torch.float32
            ),
            transition_mask=torch.tensor([True, False]),
            base_logits=torch.tensor([0.2, -0.1]),
            protocol_sha256="protocol",
            selector_checkpoint_sha256="selector",
            exact_head_checkpoint_sha256="head",
            sgw_scaler_sha256="scaler",
            exact_manifest_sha256="exact_manifest",
            selection_mode="learned",
            selection_seed=path_seed,
        )

    def test_artifact_and_manifest_round_trip_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for key in ("sample0", "sample1"):
                record = self._record(key, "train")
                path = root / (key + ".pt")
                save_dual_temporal_record(record, path)
                loaded = load_dual_temporal_record(path)
                self.assertEqual(loaded.sample_key, key)
                records.append((record, path))
            manifest = root / "manifest.json"
            write_dual_temporal_manifest(records, manifest)
            payload, loaded_records = read_dual_temporal_manifest(manifest)
            self.assertEqual(payload["sample_count"], 2)
            self.assertEqual(len(loaded_records), 2)
            with self.assertRaises(FileExistsError):
                write_dual_temporal_manifest(records, manifest)


if __name__ == "__main__":
    unittest.main()
