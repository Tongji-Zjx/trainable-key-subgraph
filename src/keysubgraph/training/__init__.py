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
from .full_graph_classifier_trainer import (
    FullGraphTrainingConfig,
    load_full_graph_classifier_checkpoint,
    run_full_graph_classifier_epoch,
    train_full_graph_classifier,
)
from .hard_stse_trainer import (
    HardSTSETrainingConfig,
    fit_hard_stse_standardizers,
    hard_stse_config_from_dict,
    load_hard_stse_checkpoint,
    run_hard_stse_epoch,
    train_hard_stse,
)
from .exact_stse_trainer import (
    ExactSTSETrainingConfig,
    exact_stse_config_from_dict,
    load_exact_stse_checkpoint,
    run_exact_stse_epoch,
    train_exact_stse,
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
        "FullGraphTrainingConfig",
        "load_full_graph_classifier_checkpoint",
        "run_full_graph_classifier_epoch",
        "train_full_graph_classifier",
        "HardSTSETrainingConfig",
        "fit_hard_stse_standardizers",
        "hard_stse_config_from_dict",
        "load_hard_stse_checkpoint",
        "run_hard_stse_epoch",
        "train_hard_stse",
        "ExactSTSETrainingConfig",
        "exact_stse_config_from_dict",
        "load_exact_stse_checkpoint",
        "run_exact_stse_epoch",
        "train_exact_stse",
    ]
)
