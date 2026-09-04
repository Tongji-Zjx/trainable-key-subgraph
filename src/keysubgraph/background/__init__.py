"""Global static-background branch for MoKSE-Net-BG."""

from .data import (
    BackgroundFeatureScaler,
    GlobalStaticGraphRecord,
    SIGNED_CONNECTIVITY_PROFILE_NAMES,
    STATIC_FEATURE_NAMES,
    build_global_static_record,
    build_signed_connectivity_profile,
    fit_background_feature_scaler,
    load_tge_manifest_records,
)
from .model import (
    GlobalBackgroundGCN,
    MoKSEBackgroundFusion,
    MoKSEBackgroundModel,
    StaticBackgroundConfig,
    masked_community_mean_std,
)
from .safe_fusion import (
    SafeFusionConfig,
    apply_safe_fusion,
    select_safe_fusion,
)

__all__ = [
    "BackgroundFeatureScaler",
    "GlobalStaticGraphRecord",
    "SIGNED_CONNECTIVITY_PROFILE_NAMES",
    "STATIC_FEATURE_NAMES",
    "GlobalBackgroundGCN",
    "MoKSEBackgroundFusion",
    "MoKSEBackgroundModel",
    "StaticBackgroundConfig",
    "SafeFusionConfig",
    "build_global_static_record",
    "build_signed_connectivity_profile",
    "fit_background_feature_scaler",
    "load_tge_manifest_records",
    "masked_community_mean_std",
    "apply_safe_fusion",
    "select_safe_fusion",
]
