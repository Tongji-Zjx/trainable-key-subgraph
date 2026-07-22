"""Reproducible training, evaluation, and checkpoint utilities."""

from .trainer import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    set_reproducible_seed,
    train_model,
)

__all__ = [
    "TrainingConfig",
    "evaluate_model",
    "load_checkpoint",
    "set_reproducible_seed",
    "train_model",
]
from .tg_soft_teacher_checkpoint import (
    load_tg_soft_teacher_checkpoint,
    save_tg_soft_teacher_checkpoint,
)
from .tg_hard_student_checkpoint import (
    load_tg_hard_student_checkpoint,
    save_tg_hard_student_checkpoint,
)
from .tg_soft_teacher_trainer import (
    TGSoftTeacherTrainingConfig,
    run_tg_soft_teacher_epoch,
    train_tg_soft_teacher,
)
from .tg_hard_student_trainer import (
    TGHardStudentTrainingConfig,
    build_tg_hard_student_optimizer,
    initialize_student_graph_encoder,
    run_tg_hard_student_epoch,
    set_student_graph_encoder_trainable,
    train_tg_hard_student,
)

__all__.extend(
    [
        "TGSoftTeacherTrainingConfig",
        "load_tg_soft_teacher_checkpoint",
        "run_tg_soft_teacher_epoch",
        "save_tg_soft_teacher_checkpoint",
        "train_tg_soft_teacher",
        "load_tg_hard_student_checkpoint",
        "save_tg_hard_student_checkpoint",
        "TGHardStudentTrainingConfig",
        "build_tg_hard_student_optimizer",
        "initialize_student_graph_encoder",
        "run_tg_hard_student_epoch",
        "set_student_graph_encoder_trainable",
        "train_tg_hard_student",
    ]
)
