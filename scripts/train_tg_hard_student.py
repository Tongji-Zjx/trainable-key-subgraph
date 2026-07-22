"""Train the Stage-C 226-D hard student with frozen-teacher distillation."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.data.tg_student_dataset import (  # noqa: E402
    TGHardStudentDataset,
    create_tg_hard_student_loader,
)
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.features import TGTheoryFeatureStandardizer  # noqa: E402
from keysubgraph.models import (  # noqa: E402
    TGHardSGWClassifier,
    TGHardStudentLossConfig,
    TGSoftTeacher,
    TGSoftTeacherConfig,
)
from keysubgraph.training import (  # noqa: E402
    TGHardStudentTrainingConfig,
    load_tg_soft_teacher_checkpoint,
    train_tg_hard_student,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-scaler", type=Path, required=True)
    parser.add_argument("--theory-scaler", type=Path, required=True)
    for split in ("train", "validation"):
        parser.add_argument("--{}-hard-cache-dir".format(split), type=Path, required=True)
        parser.add_argument("--{}-theory-feature-dir".format(split), type=Path, required=True)
        parser.add_argument("--{}-teacher-target-dir".format(split), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--frozen-graph-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--graph-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_accuracy", "roc_auc", "unweighted_log_loss"),
        default="balanced_accuracy",
    )
    parser.add_argument("--lambda-kd", type=float, default=0.50)
    parser.add_argument("--lambda-representation", type=float, default=0.10)
    parser.add_argument("--lambda-supcon", type=float, default=0.05)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_payload(path):
    try:
        return torch.load(str(path.resolve()), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path.resolve()), map_location="cpu")


def main():
    args = parse_args()
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    if protocol.get("protocol_name") != "strict_theory":
        raise ValueError("Stage-C classification requires strict_theory data")
    protocol_hash = file_sha256(args.protocol)
    teacher_hash = file_sha256(args.teacher_checkpoint)
    scaler = TGTheoryFeatureStandardizer.load(args.theory_scaler)
    if scaler.data_protocol_sha256 != protocol_hash:
        raise ValueError("theory scaler and protocol hash differ")
    if scaler.teacher_checkpoint_sha256 != teacher_hash:
        raise ValueError("theory scaler and teacher checkpoint hash differ")
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = TGHardStudentDataset(
            getattr(args, "{}_hard_cache_dir".format(split)),
            getattr(args, "{}_theory_feature_dir".format(split)),
            getattr(args, "{}_teacher_target_dir".format(split)),
            scaler, expected_split=split,
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    train_loader = create_tg_hard_student_loader(
        datasets["train"], args.batch_size, args.seed, args.num_workers, shuffle=True
    )
    validation_loader = create_tg_hard_student_loader(
        datasets["validation"], args.batch_size, args.seed, args.num_workers, shuffle=False
    )
    teacher_payload = _load_payload(args.teacher_checkpoint)
    teacher = TGSoftTeacher(TGSoftTeacherConfig(**teacher_payload["model_config"]))
    load_tg_soft_teacher_checkpoint(
        args.teacher_checkpoint, teacher, torch.device("cpu"),
        expected_protocol_sha256=protocol_hash,
    )
    teacher.eval()
    model = TGHardSGWClassifier()
    loss_config = TGHardStudentLossConfig(
        knowledge_distillation_weight=args.lambda_kd,
        representation_distillation_weight=args.lambda_representation,
        supervised_contrastive_weight=args.lambda_supcon,
        knowledge_distillation_temperature=args.kd_temperature,
    )
    training_config = TGHardStudentTrainingConfig(
        epochs=2 if args.smoke else args.epochs,
        frozen_graph_epochs=1 if args.smoke else args.frozen_graph_epochs,
        head_learning_rate=args.head_learning_rate,
        graph_encoder_learning_rate=args.graph_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        selection_metric=args.selection_metric,
        seed=args.seed,
        max_train_batches=1 if args.smoke else None,
        max_validation_batches=1 if args.smoke else None,
    )
    started = time.perf_counter()
    result = train_tg_hard_student(
        model, train_loader, validation_loader, datasets["train"].labels,
        device, loss_config, training_config, args.output_dir, scaler,
        protocol_hash, teacher_hash, file_sha256(args.candidate_scaler),
        file_sha256(args.theory_scaler), teacher_model=teacher,
        resume_checkpoint=args.resume,
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    printable.update({
        "device": str(device), "debug_smoke": bool(args.smoke),
        "elapsed_seconds": time.perf_counter() - started,
        "train_sample_count": len(datasets["train"]),
        "validation_sample_count": len(datasets["validation"]),
    })
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
