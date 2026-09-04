"""Global static-background branch for MoKSE-Net-BG."""

from .data import (
    BackgroundFeatureScaler,
    GlobalStaticGraphRecord,
    RELATIVE_SIGNED_CONNECTIVITY_PROFILE_NAMES,
    SIGNED_CONNECTIVITY_PROFILE_NAMES,
    STATIC_FEATURE_NAMES,
    build_global_static_record,
    build_relative_signed_connectivity_profile,
    build_signed_connectivity_profile,
    fit_background_feature_scaler,
    fit_train_community_kappa,
    load_tge_manifest_records,
)
from .model import (
    GlobalBackgroundGCN,
    MoKSEBackgroundFusion,
    MoKSEBackgroundModel,
    StaticBackgroundConfig,
    masked_community_mean_std,
    masked_support_shrunk_community_mean_std,
)
from .safe_fusion import (
    SafeFusionConfig,
    apply_safe_fusion,
    select_safe_fusion,
)
from .s4_fusion import (
    S4AnchoredFusionConfig,
    S4StaticPromotionConfig,
    apply_s4_anchored_fusion,
    apply_s4_seed_ensemble,
    fit_s4_seed_ensemble,
    select_s4_static_promotion,
    select_s4_anchored_fusion,
)

__all__ = [
    "BackgroundFeatureScaler",
    "GlobalStaticGraphRecord",
    "RELATIVE_SIGNED_CONNECTIVITY_PROFILE_NAMES",
    "SIGNED_CONNECTIVITY_PROFILE_NAMES",
    "STATIC_FEATURE_NAMES",
    "GlobalBackgroundGCN",
    "MoKSEBackgroundFusion",
    "MoKSEBackgroundModel",
    "StaticBackgroundConfig",
    "SafeFusionConfig",
    "S4AnchoredFusionConfig",
    "S4StaticPromotionConfig",
    "build_global_static_record",
    "build_relative_signed_connectivity_profile",
    "build_signed_connectivity_profile",
    "fit_background_feature_scaler",
    "fit_train_community_kappa",
    "load_tge_manifest_records",
    "masked_community_mean_std",
    "masked_support_shrunk_community_mean_std",
    "apply_safe_fusion",
    "select_safe_fusion",
    "fit_s4_seed_ensemble",
    "apply_s4_seed_ensemble",
    "select_s4_anchored_fusion",
    "select_s4_static_promotion",
    "apply_s4_anchored_fusion",
]
