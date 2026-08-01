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
from .structured_short_term_trainer import (
    StructuredShortTermTrainingConfig,
    evaluate_structured_short_term,
    load_structured_short_term_checkpoint,
    model_from_structured_short_term_checkpoint,
    run_structured_short_term_epoch,
    train_structured_short_term,
)
from .author_short_term_trainer import (
    AuthorBalancedBatchSampler,
    AuthorShortTermTrainingConfig,
    author_short_term_training_config,
    create_author_short_term_evaluation_loader,
    create_author_short_term_train_loader,
    evaluate_author_short_term,
    fit_author_threshold,
    model_from_author_short_term_checkpoint,
    run_author_short_term_epoch,
    train_author_short_term,
)
from .dual_stse_hard_sgw_trainer import (
    DualTrainingConfig,
    load_dual_checkpoint,
    run_dual_epoch,
    train_dual_stage,
)
from .dual_sgw_feature_trainer import (
    DualSGWFeatureTrainingConfig,
    binary_metrics,
    fit_binary_threshold,
    load_dual_sgw_feature_checkpoint,
    run_dual_sgw_feature_epoch,
    train_dual_sgw_feature_classifier,
)
from .dual_sgw_feature_ensemble import (
    average_evaluation_probabilities,
    build_dual_sgw_probability_ensemble,
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
        "StructuredShortTermTrainingConfig",
        "evaluate_structured_short_term",
        "load_structured_short_term_checkpoint",
        "model_from_structured_short_term_checkpoint",
        "run_structured_short_term_epoch",
        "train_structured_short_term",
        "AuthorBalancedBatchSampler",
        "AuthorShortTermTrainingConfig",
        "author_short_term_training_config",
        "create_author_short_term_evaluation_loader",
        "create_author_short_term_train_loader",
        "evaluate_author_short_term",
        "fit_author_threshold",
        "model_from_author_short_term_checkpoint",
        "run_author_short_term_epoch",
        "train_author_short_term",
        "DualTrainingConfig",
        "load_dual_checkpoint",
        "run_dual_epoch",
        "train_dual_stage",
        "DualSGWFeatureTrainingConfig",
        "binary_metrics",
        "fit_binary_threshold",
        "load_dual_sgw_feature_checkpoint",
        "run_dual_sgw_feature_epoch",
        "train_dual_sgw_feature_classifier",
        "average_evaluation_probabilities",
        "build_dual_sgw_probability_ensemble",
    ]
)
