"""Trainable soft key-subgraph extractor models."""

from .soft_extractor import (
    BatchModelOutput,
    SoftExtractorConfig,
    SoftGraphClassifier,
    TimepointSelection,
)
from .losses import SoftGraphLoss, compute_soft_graph_loss
from .node_only_subgraph_encoder import NodeOnlyLayer, NodeOnlySubgraphEncoder
from .masked_pooling import MaskedGraphPooling
from .masked_tcn import MaskedTCNEncoder, MaskedTemporalConvBlock, pad_temporal_sequences
from .signed_graph_encoder import SignedGraphEncoder, SignedMessageLayer
from .tg_soft_teacher import (
    TGSoftTeacher,
    TGSoftTeacherConfig,
    TGSoftTeacherOutput,
    TGSoftTimepointOutput,
)
from .tg_soft_teacher_loss import (
    TGSoftTeacherLoss,
    TGSoftTeacherLossConfig,
    compute_tg_soft_teacher_loss,
)
from .tg_hard_classifier import (
    TGHardClassifierConfig,
    TGHardClassifierOutput,
    TGHardSGWClassifier,
)
from .tg_hard_student_loss import (
    TGHardStudentLoss,
    TGHardStudentLossConfig,
    compute_tg_hard_student_loss,
)

__all__ = [
    "BatchModelOutput",
    "SoftExtractorConfig",
    "SoftGraphClassifier",
    "SoftGraphLoss",
    "TimepointSelection",
    "compute_soft_graph_loss",
    "NodeOnlyLayer",
    "NodeOnlySubgraphEncoder",
    "MaskedGraphPooling",
    "MaskedTCNEncoder",
    "MaskedTemporalConvBlock",
    "pad_temporal_sequences",
    "SignedGraphEncoder",
    "SignedMessageLayer",
    "TGSoftTeacher",
    "TGSoftTeacherConfig",
    "TGSoftTeacherOutput",
    "TGSoftTimepointOutput",
    "TGSoftTeacherLoss",
    "TGSoftTeacherLossConfig",
    "TGHardClassifierConfig",
    "TGHardClassifierOutput",
    "TGHardSGWClassifier",
    "TGHardStudentLoss",
    "TGHardStudentLossConfig",
    "compute_tg_hard_student_loss",
    "compute_tg_soft_teacher_loss",
]
