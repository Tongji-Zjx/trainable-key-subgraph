from __future__ import absolute_import, division, print_function

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.graph_dataset import (  # noqa: E402
    GraphSequenceSample,
    create_data_loader,
)
from keysubgraph.models.soft_extractor import (  # noqa: E402
    SoftExtractorConfig,
    SoftGraphClassifier,
)
from keysubgraph.training.trainer import (  # noqa: E402
    TrainingConfig,
    load_checkpoint,
    train_model,
)


def _sample(index, label, split):
    value = 0.2 + index * 0.02
    graph = torch.tensor(
        [[0.0, value, 0.0], [value, 0.0, -0.3], [0.0, -0.3, 0.0]]
    )
    mask = graph.abs() > 0
    mask.fill_diagonal_(False)
    return GraphSequenceSample(
        sample_key="SITE/{}_{}".format(split, index),
        sample_id="{}_{}".format(split, index),
        site="SITE",
        subject_id=str(index),
        session_id="1",
        label=label,
        split=split,
        relative_path="unused.pt",
        adjacency=(graph,),
        edge_mask=(mask,),
        node_names=(("a", "b", "c"),),
        communities=(torch.tensor([0, 0, 1]),),
        window_starts=torch.tensor([0.0]),
        source_global_threshold=0.1,
        repetition_time=2.0,
        edge_presence_threshold=0.0,
    )


class _Samples(Dataset):
    def __init__(self, split, count):
        self.split = split
        self.samples = [_sample(index, index % 2, split) for index in range(count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TrainingTest(unittest.TestCase):
    def test_train_checkpoint_history_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "data_protocol.json"
            with protocol_path.open("w", encoding="utf-8") as handle:
                json.dump({"test": True}, handle)
            protocol = {
                "sha256": {
                    "sample_index_csv": "a",
                    "splits_csv": "b",
                    "splits_json": "c",
                },
                "edge_presence_threshold": 0.0,
            }
            train_data = _Samples("train", 6)
            validation_data = _Samples("validation", 4)
            train_loader = create_data_loader(train_data, 2, seed=3)
            validation_loader = create_data_loader(validation_data, 2, seed=3)
            model_config = SoftExtractorConfig(
                node_score_hidden_dim=4,
                edge_score_hidden_dim=4,
                graph_hidden_dim=6,
                graph_layers=1,
                classifier_hidden_dim=4,
                dropout=0.0,
            )
            model = SoftGraphClassifier(model_config)
            config = TrainingConfig(epochs=2, seed=3, selection_metric="roc_auc")
            output_dir = root / "run"

            result = train_model(
                model,
                train_loader,
                validation_loader,
                [sample.label for sample in train_data.samples],
                torch.device("cpu"),
                config,
                output_dir,
                protocol_path,
                protocol,
            )

            self.assertTrue(result["best_checkpoint"].is_file())
            self.assertTrue(result["last_checkpoint"].is_file())
            self.assertTrue(result["history"].is_file())
            with result["history"].open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["validation"]["sample_count"], 4)
            self.assertIn("roc_auc", history[-1]["validation"])

            restored = SoftGraphClassifier(model_config)
            checkpoint = load_checkpoint(
                result["last_checkpoint"], restored, device=torch.device("cpu")
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["edge_presence_threshold"], 0.0)

            resumed_config = TrainingConfig(
                epochs=3, seed=3, selection_metric="roc_auc"
            )
            resumed = train_model(
                restored,
                train_loader,
                validation_loader,
                [sample.label for sample in train_data.samples],
                torch.device("cpu"),
                resumed_config,
                output_dir,
                protocol_path,
                protocol,
                resume_checkpoint=result["last_checkpoint"],
            )
            self.assertEqual(resumed["epochs_completed"], 3)
            with resumed["history"].open("r", encoding="utf-8") as handle:
                resumed_history = json.load(handle)
            self.assertEqual(resumed_history[-1]["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
