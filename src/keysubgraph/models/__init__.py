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
    TGScoreStatistics,
    TGSoftTimepointOutput,
)
from .tg_soft_teacher_loss import (
    TGSoftTeacherLoss,
    TGSoftTeacherLossConfig,
    TG_SOFT_TEACHER_ABLATIONS,
    compute_tg_soft_teacher_loss,
    tg_soft_teacher_ablation_weights,
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
from .full_graph_classifier import (
    FULL_GRAPH_ENCODERS,
    FullGraphClassifierConfig,
    FullGraphClassifierOutput,
    FullGraphSequenceClassifier,
    PackedBiGRUEncoder,
    PrototypeCodebook,
    SignedEdgeGatedGraphEncoder,
    SignedGatedBiGRUPrototypeEncoder,
    SignedGNNTCNFullGraphEncoder,
    SymmetricSignedEdgeGatedLayer,
)
from .exact_stse import (
    ExactSTSEClassifier,
    ExactSTSEConfig,
    ExactSTSEFeatureBuilder,
    ExactSTSEOutput,
    ExactSTSEWindowEncoder,
)
from .dual_stse_hard_sgw_types import (
    DUAL_EXPERIMENT_VARIANTS,
    DUAL_TRAINING_STAGES,
    DualSTSEHardSGWConfig,
    DualSTSEHardSGWOutput,
)
from .dual_stse_channel import (
    DualSTSEChannelOutput,
    ExistingNoCoordSTSEChannel,
)
from .dual_hard_sgw_selector import (
    DualHardSelectionOutput,
    DualHardSGWSelector,
)
from .dual_sgw_proxy import DualSGWProxy, DualSGWProxyOutput
from .dual_exact_sgw import (
    DualExactSGWBranch,
    DualExactSGWOutput,
    DualSGWFeatureRecord,
    load_dual_sgw_feature_record,
    save_dual_sgw_feature_record,
)
from .dual_stse_hard_sgw import DualSTSEHardSGWClassifier
from .dual_stse_hard_sgw_loss import (
    DualSTSEHardSGWCriterion,
    DualSTSEHardSGWLoss,
    DualSTSEHardSGWLossConfig,
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
    "TGScoreStatistics",
    "TGSoftTimepointOutput",
    "TGSoftTeacherLoss",
    "TGSoftTeacherLossConfig",
    "TG_SOFT_TEACHER_ABLATIONS",
    "TGHardClassifierConfig",
    "TGHardClassifierOutput",
    "TGHardSGWClassifier",
    "TGHardStudentLoss",
    "TGHardStudentLossConfig",
    "compute_tg_hard_student_loss",
    "compute_tg_soft_teacher_loss",
    "tg_soft_teacher_ablation_weights",
    "FULL_GRAPH_ENCODERS",
    "FullGraphClassifierConfig",
    "FullGraphClassifierOutput",
    "FullGraphSequenceClassifier",
    "PackedBiGRUEncoder",
    "PrototypeCodebook",
    "SignedEdgeGatedGraphEncoder",
    "SignedGatedBiGRUPrototypeEncoder",
    "SignedGNNTCNFullGraphEncoder",
    "SymmetricSignedEdgeGatedLayer",
    "ExactSTSEClassifier",
    "ExactSTSEConfig",
    "ExactSTSEFeatureBuilder",
    "ExactSTSEOutput",
    "ExactSTSEWindowEncoder",
    "DUAL_EXPERIMENT_VARIANTS",
    "DUAL_TRAINING_STAGES",
    "DualSTSEHardSGWConfig",
    "DualSTSEHardSGWOutput",
    "DualSTSEChannelOutput",
    "ExistingNoCoordSTSEChannel",
    "DualHardSelectionOutput",
    "DualHardSGWSelector",
    "DualSGWProxy",
    "DualSGWProxyOutput",
    "DualExactSGWBranch",
    "DualExactSGWOutput",
    "DualSGWFeatureRecord",
    "load_dual_sgw_feature_record",
    "save_dual_sgw_feature_record",
    "DualSTSEHardSGWClassifier",
    "DualSTSEHardSGWCriterion",
    "DualSTSEHardSGWLoss",
    "DualSTSEHardSGWLossConfig",
]
