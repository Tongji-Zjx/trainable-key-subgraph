from __future__ import absolute_import, division, print_function

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from keysubgraph.crossfit.sv_spectral_evolution_comparison import (
    compare_sv_spectral_evolution,
)


def _summary(root, variant, probabilities):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = (0, 0, 1, 1)
    for index, (label, probability) in enumerate(
        zip(labels, probabilities)
    ):
        rows.append(
            {
                "fold": index % 2,
                "sample_key": "sample_{}".format(index),
                "site": "site_{}".format(index % 2),
                "label": label,
                "positive_probability": probability,
                "threshold": 0.5,
                "predicted_label": int(probability >= 0.5),
            }
        )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "artifact_type": "sv_signed_gin_crossfit_oof_summary",
                "variant": variant,
                "checks": {"every_sample_predicted_once": True},
                "metrics": {"sample_count": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    with (root / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SVSpectralEvolutionComparisonTest(unittest.TestCase):
    def test_paired_comparison_keeps_s_se_and_shuffle_aligned(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            s_dir = root / "s"
            se_dir = root / "se"
            shuffled_dir = root / "shuffled"
            _summary(
                s_dir,
                "static_spectral_only",
                (0.4, 0.6, 0.5, 0.7),
            )
            _summary(
                se_dir,
                "static_spectral_neural_evolution",
                (0.2, 0.3, 0.7, 0.8),
            )
            _summary(
                shuffled_dir,
                "static_spectral_neural_evolution_time_shuffled",
                (0.3, 0.6, 0.4, 0.7),
            )
            result = compare_sv_spectral_evolution(
                [("dummy", s_dir, se_dir, shuffled_dir)],
                root / "out",
                bootstrap_repeats=20,
                permutation_repeats=20,
                seed=3,
            )
            values = result["datasets"]["dummy"]
            self.assertEqual(values["sample_count"], 4)
            self.assertGreater(
                values["contrasts"]["SE_minus_S"]["auc_difference"],
                0.0,
            )
            self.assertTrue(result["comparison_markdown"].is_file())


if __name__ == "__main__":
    unittest.main()
