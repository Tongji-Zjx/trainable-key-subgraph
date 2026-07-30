from __future__ import absolute_import, division, print_function

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_sv_full_hard_channel_experiment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_sv_full_hard_channel_experiment", str(SCRIPT)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SVFullHardRunnerTest(unittest.TestCase):
    def test_plan_is_independent_and_fuses_last(self):
        args = SimpleNamespace(
            protocol=Path("protocol.json"),
            selector_checkpoint=Path("selector.pt"),
            output_dir=Path("experiment"),
            variant="sv_static_variation",
            device="cuda",
            epochs=5,
            batch_size=4,
            gradient_accumulation_steps=2,
            num_workers=2,
            seed=42,
            learning_rate=0.001,
            weight_decay=0.0001,
            gradient_clip=1.0,
            early_stopping_patience=3,
            selection_metric="roc_auc",
            alpha_grid=(0.0, 0.5, 1.0),
        )
        steps = MODULE.build_experiment_steps(args)
        names = [step["name"] for step in steps]
        self.assertEqual(len(steps), 15)
        self.assertEqual(names[-1], "fuse_full_hard")
        hard_cache = steps[0]["command"]
        full_cache = steps[7]["command"]
        self.assertIn("--selector-checkpoint", hard_cache)
        self.assertNotIn("--selector-checkpoint", full_cache)
        self.assertIn("learned", hard_cache)
        self.assertIn("full", full_cache)
        hard_train = next(
            step for step in steps if step["name"] == "train_hard"
        )
        full_train = next(
            step for step in steps if step["name"] == "train_full"
        )
        self.assertEqual(
            hard_train["command"][
                hard_train["command"].index("--variant") + 1
            ],
            full_train["command"][
                full_train["command"].index("--variant") + 1
            ],
        )


if __name__ == "__main__":
    unittest.main()
