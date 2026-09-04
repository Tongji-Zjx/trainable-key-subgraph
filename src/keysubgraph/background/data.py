"""Immutable static-graph features and manifest matching for MoKSE-Net-BG."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


STATIC_FEATURE_NAMES = (
    "absolute_strength",
    "positive_strength",
    "negative_strength",
    "positive_strength_ratio",
    "negative_strength_ratio",
    "community_relative_size",
    "intra_positive_strength",
    "intra_negative_strength",
    "inter_positive_strength",
    "inter_negative_strength",
    "intra_positive_density",
    "intra_negative_density",
)

SIGNED_CONNECTIVITY_PROFILE_NAMES = (
    "positive_q25",
    "positive_q50",
    "positive_q75",
    "positive_q90",
    "negative_q25",
    "negative_q50",
    "negative_q75",
    "negative_q90",
    "has_positive_edge",
    "has_negative_edge",
)

RELATIVE_SIGNED_CONNECTIVITY_PROFILE_NAMES = tuple(
    "relative_{}".format(name)
    if not name.startswith("has_") else name
    for name in SIGNED_CONNECTIVITY_PROFILE_NAMES
)

DEFAULT_PROFILE_QUANTILES = (0.25, 0.50, 0.75, 0.90)


def _safe_torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tge_manifest_records(path: Path, expected_split: str) -> Tuple[dict, ...]:
    """Read the frozen TGE manifest without rediscovering subjects."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(dict(row) for row in payload.get("records", ()))
    if not records:
        raise ValueError("TGE manifest does not contain records")
    seen = set()
    for row in records:
        key = str(row.get("sample_key", ""))
        if not key or key in seen:
            raise ValueError("TGE manifest sample keys must be non-empty and unique")
        seen.add(key)
        if str(row.get("split")) != str(expected_split):
            raise ValueError("TGE manifest split mismatch")
        if int(row.get("label")) not in (0, 1):
            raise ValueError("TGE manifest labels must be binary")
    return records


def resolve_global_graph_path(global_root: Path, row: Mapping[str, object]) -> Path:
    """Resolve by site/label/sample_id; bare subject IDs are never used."""

    site = str(row.get("site", ""))
    sample_id = str(row.get("sample_id", ""))
    label = int(row.get("label"))
    if not site or not sample_id:
        raise ValueError("global graph matching requires site and sample_id")
    filename = sample_id if sample_id.endswith(".pt") else sample_id + ".pt"
    path = Path(global_root) / site / str(label) / filename
    if not path.is_file():
        raise FileNotFoundError("missing global graph for {}: {}".format(
            row.get("sample_key"), path
        ))
    return path


def build_static_node_features(
    adjacency: torch.Tensor,
    communities: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Reproduce the project's 12-D label-free signed structural features."""

    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    communities = torch.as_tensor(communities, dtype=torch.long)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("global adjacency must be square")
    count = int(adjacency.shape[0])
    if communities.shape != (count,):
        raise ValueError("global community labels do not align with nodes")
    positive = adjacency.clamp_min(0.0)
    negative = (-adjacency.clamp_max(0.0))
    absolute = adjacency.abs()
    positive_strength = positive.sum(dim=-1)
    negative_strength = negative.sum(dim=-1)
    strength = absolute.sum(dim=-1)
    denominator = (positive_strength + negative_strength).clamp_min(epsilon)

    same = communities[:, None] == communities[None, :]
    not_self = ~torch.eye(count, dtype=torch.bool)
    intra = same & not_self
    inter = (~same) & not_self
    community_size = same.sum(dim=-1).to(adjacency.dtype)
    intra_denominator = (community_size - 1.0).clamp_min(1.0)
    inter_denominator = (float(count) - community_size).clamp_min(1.0)

    intra_positive = (positive * intra).sum(dim=-1) / intra_denominator
    intra_negative = (negative * intra).sum(dim=-1) / intra_denominator
    inter_positive = (positive * inter).sum(dim=-1) / inter_denominator
    inter_negative = (negative * inter).sum(dim=-1) / inter_denominator

    positive_density = torch.zeros(count, dtype=adjacency.dtype)
    negative_density = torch.zeros(count, dtype=adjacency.dtype)
    for value in torch.unique(communities):
        members = torch.nonzero(communities == value, as_tuple=False).flatten()
        size = int(members.numel())
        if size < 2:
            continue
        subgraph = adjacency.index_select(0, members).index_select(1, members)
        upper = torch.triu(torch.ones_like(subgraph, dtype=torch.bool), diagonal=1)
        possible = float(size * (size - 1)) + epsilon
        positive_density[members] = (
            2.0 * ((subgraph > 0.0) & upper).sum().to(adjacency.dtype) / possible
        )
        negative_density[members] = (
            2.0 * ((subgraph < 0.0) & upper).sum().to(adjacency.dtype) / possible
        )

    result = torch.stack(
        (
            strength,
            positive_strength,
            negative_strength,
            positive_strength / denominator,
            negative_strength / denominator,
            community_size / float(count),
            intra_positive,
            intra_negative,
            inter_positive,
            inter_negative,
            positive_density,
            negative_density,
        ),
        dim=-1,
    )
    if result.shape != (count, len(STATIC_FEATURE_NAMES)):
        raise RuntimeError("unexpected static feature shape")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("global static features contain non-finite values")
    return result


def build_signed_connectivity_profile(
    adjacency: torch.Tensor,
    quantiles: Sequence[float] = DEFAULT_PROFILE_QUANTILES,
) -> torch.Tensor:
    """Return permutation-equivariant signed edge-weight summaries per node.

    Negative edges are summarized by magnitude.  Empty sign channels receive
    zero placeholders together with an explicit validity flag.  The default
    upper-tail statistic is q90 rather than a maximum so that the profile is
    less sensitive to a single extreme edge and to varying node counts.
    """

    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("signed connectivity profile requires a square adjacency")
    requested = tuple(float(value) for value in quantiles)
    if len(requested) != 4 or any(value < 0.0 or value > 1.0 for value in requested):
        raise ValueError("signed connectivity profile requires four quantiles in [0,1]")
    count = int(adjacency.shape[0])
    output = torch.zeros(count, 2 * len(requested) + 2, dtype=adjacency.dtype)
    quantile_tensor = torch.tensor(requested, dtype=adjacency.dtype)
    for node in range(count):
        row = adjacency[node]
        positive = row[row > 0.0]
        negative = (-row[row < 0.0])
        if positive.numel():
            output[node, : len(requested)] = torch.quantile(
                positive, quantile_tensor
            )
            output[node, -2] = 1.0
        if negative.numel():
            output[node, len(requested): 2 * len(requested)] = torch.quantile(
                negative, quantile_tensor
            )
            output[node, -1] = 1.0
    if output.shape != (count, len(SIGNED_CONNECTIVITY_PROFILE_NAMES)):
        raise RuntimeError("unexpected signed connectivity profile shape")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("signed connectivity profile contains non-finite values")
    return output


def build_relative_signed_connectivity_profile(
    adjacency: torch.Tensor,
    quantiles: Sequence[float] = DEFAULT_PROFILE_QUANTILES,
    relative_epsilon: float = 1.0e-3,
    absolute_epsilon: float = 1.0e-12,
    log_clip: float = 4.0,
) -> torch.Tensor:
    """Return scale-robust node profiles relative to the whole signed graph.

    Positive and negative-magnitude channels use separate graph references.
    The numerical floor scales with the median non-zero graph weight, making
    the ratios invariant to ordinary uniform rescaling of either sign channel.
    Binary validity flags remain untransformed in the last two columns.
    """

    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("relative signed profile requires a square adjacency")
    requested = tuple(float(value) for value in quantiles)
    if len(requested) != 4 or any(value < 0.0 or value > 1.0 for value in requested):
        raise ValueError("relative signed profile requires four quantiles in [0,1]")
    if relative_epsilon <= 0.0 or absolute_epsilon <= 0.0 or log_clip <= 0.0:
        raise ValueError("relative profile numerical safeguards must be positive")

    count = int(adjacency.shape[0])
    output = torch.zeros(count, 2 * len(requested) + 2, dtype=adjacency.dtype)
    quantile_tensor = torch.tensor(requested, dtype=adjacency.dtype)
    upper_mask = torch.triu(
        torch.ones_like(adjacency, dtype=torch.bool), diagonal=1
    )

    channels = (
        (adjacency, adjacency[upper_mask & (adjacency > 0.0)], 0, -2),
        (-adjacency, (-adjacency)[upper_mask & (adjacency < 0.0)], len(requested), -1),
    )
    for row_values, graph_values, offset, validity_column in channels:
        if graph_values.numel() == 0:
            continue
        graph_quantiles = torch.quantile(graph_values, quantile_tensor)
        graph_scale = float(torch.quantile(graph_values, 0.5))
        epsilon = max(float(absolute_epsilon), float(relative_epsilon) * graph_scale)
        for node in range(count):
            values = row_values[node]
            values = values[values > 0.0]
            if values.numel() == 0:
                continue
            node_quantiles = torch.quantile(values, quantile_tensor)
            relative = torch.log(
                (node_quantiles + epsilon) / (graph_quantiles + epsilon)
            ).clamp(min=-float(log_clip), max=float(log_clip))
            output[node, offset: offset + len(requested)] = relative
            output[node, validity_column] = 1.0
    if output.shape != (count, len(RELATIVE_SIGNED_CONNECTIVITY_PROFILE_NAMES)):
        raise RuntimeError("unexpected relative signed profile shape")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("relative signed connectivity profile is non-finite")
    return output


def signed_laplacian_encoding(
    adjacency: torch.Tensor,
    dimensions: int = 8,
    epsilon: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lowest signed-normalized-Laplacian eigenvectors with fixed signs.

    The first vector is retained.  Unlike an unsigned normalized Laplacian,
    an unbalanced signed graph has no guaranteed zero-valued trivial mode.
    """

    adjacency64 = torch.as_tensor(adjacency, dtype=torch.float64, device="cpu")
    count = int(adjacency64.shape[0])
    if adjacency64.shape != (count, count):
        raise ValueError("spectral encoding requires a square adjacency")
    if dimensions < 1 or dimensions > count:
        raise ValueError("invalid spectral encoding dimension")
    degree = adjacency64.abs().sum(dim=-1)
    inverse = degree.clamp_min(epsilon).rsqrt()
    laplacian = torch.eye(count, dtype=torch.float64) - (
        inverse[:, None] * adjacency64 * inverse[None, :]
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    vectors = eigenvectors[:, :dimensions].clone()
    for column in range(dimensions):
        vector = vectors[:, column]
        pivot = int(vector.abs().argmax().item())
        if float(vector[pivot]) < 0.0:
            vectors[:, column].mul_(-1.0)
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("spectral encoding contains non-finite values")
    return vectors.to(torch.float32), eigenvalues[:dimensions].to(torch.float32)


def signed_normalized_channels(
    adjacency: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    positive = adjacency.clamp_min(0.0)
    negative = (-adjacency.clamp_max(0.0))

    def normalize(channel):
        inverse = channel.sum(dim=-1).clamp_min(epsilon).rsqrt()
        return inverse[:, None] * channel * inverse[None, :]

    return normalize(positive), normalize(negative)


@dataclass(frozen=True)
class GlobalStaticGraphRecord:
    sample_key: str
    sample_id: str
    site: str
    label: int
    source_path: str
    source_sha256: str
    node_features: torch.Tensor
    community_labels: torch.Tensor
    raw_positive_adjacency: torch.Tensor
    raw_negative_adjacency: torch.Tensor
    positive_adjacency: torch.Tensor
    negative_adjacency: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.node_features.shape[0])


def build_global_static_record(
    global_root: Path,
    row: Mapping[str, object],
    spectral_dimensions: int = 8,
    cache_dir: Optional[Path] = None,
    include_signed_profile: bool = False,
    profile_quantiles: Sequence[float] = DEFAULT_PROFILE_QUANTILES,
    signed_profile_mode: str = "absolute",
    relative_profile_epsilon: float = 1.0e-3,
    relative_profile_log_clip: float = 4.0,
) -> GlobalStaticGraphRecord:
    sample_key = str(row["sample_key"])
    path = resolve_global_graph_path(Path(global_root), row)
    cache_path = None
    quantile_token = "_".join(
        "q{:g}".format(100.0 * float(value)) for value in profile_quantiles
    )
    if signed_profile_mode not in ("absolute", "relative"):
        raise ValueError("signed profile mode must be absolute or relative")
    if include_signed_profile:
        profile_token = "profile10_{}_{}".format(signed_profile_mode, quantile_token)
        if signed_profile_mode == "relative":
            profile_token += "_eps{:g}_clip{:g}".format(
                relative_profile_epsilon, relative_profile_log_clip
            )
        feature_version = "static12_spec{}_{}".format(
            spectral_dimensions, profile_token
        )
    else:
        feature_version = "static12_spec{}".format(spectral_dimensions)
    if cache_dir is not None:
        token = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:20]
        cache_path = Path(cache_dir) / (token + ".pt")
        if cache_path.is_file():
            cached = _safe_torch_load(cache_path)
            if (
                cached.get("sample_key") == sample_key
                and int(cached.get("spectral_dimensions", -1)) == spectral_dimensions
                and int(cached.get("cache_schema_version", -1)) == 2
                and cached.get("feature_version") == feature_version
                and cached.get("source_size") == path.stat().st_size
                and cached.get("source_mtime_ns") == path.stat().st_mtime_ns
            ):
                return GlobalStaticGraphRecord(**cached["record"])

    payload = _safe_torch_load(path)
    adjacency = torch.as_tensor(payload["adjacency"], dtype=torch.float32)
    communities = torch.as_tensor(payload["community_labels"], dtype=torch.long)
    count = int(adjacency.shape[0])
    if adjacency.shape != (count, count):
        raise ValueError("global adjacency is not square")
    if not torch.allclose(adjacency, adjacency.transpose(0, 1), atol=1.0e-7):
        raise ValueError("global adjacency is not symmetric")
    if not torch.allclose(torch.diagonal(adjacency), torch.zeros(count), atol=1.0e-7):
        raise ValueError("global adjacency diagonal is not zero")
    if not bool(torch.isfinite(adjacency).all()):
        raise ValueError("global adjacency contains non-finite values")
    names = payload.get("node_names")
    coordinates = payload.get("coords")
    if names is None or len(names) != count:
        raise ValueError("global node names do not align")
    if coordinates is None or tuple(torch.as_tensor(coordinates).shape) != (count, 3):
        raise ValueError("global coordinates do not align")

    static = build_static_node_features(adjacency, communities)
    feature_blocks = [static]
    spectral, eigenvalues = signed_laplacian_encoding(
        adjacency, dimensions=spectral_dimensions
    )
    feature_blocks.append(spectral)
    if include_signed_profile:
        if signed_profile_mode == "relative":
            profile = build_relative_signed_connectivity_profile(
                adjacency,
                profile_quantiles,
                relative_epsilon=relative_profile_epsilon,
                log_clip=relative_profile_log_clip,
            )
        else:
            profile = build_signed_connectivity_profile(adjacency, profile_quantiles)
        feature_blocks.append(profile)
    raw_positive = adjacency.clamp_min(0.0)
    raw_negative = -adjacency.clamp_max(0.0)
    positive, negative = signed_normalized_channels(adjacency)
    record = GlobalStaticGraphRecord(
        sample_key=sample_key,
        sample_id=str(row["sample_id"]),
        site=str(row["site"]),
        label=int(row["label"]),
        source_path=str(path.resolve()),
        source_sha256=_sha256(path),
        node_features=torch.cat(tuple(feature_blocks), dim=-1),
        community_labels=communities,
        raw_positive_adjacency=raw_positive,
        raw_negative_adjacency=raw_negative,
        positive_adjacency=positive,
        negative_adjacency=negative,
        eigenvalues=eigenvalues,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "artifact_type": "mokse_global_static_cache_v2",
                "cache_schema_version": 2,
                "sample_key": sample_key,
                "spectral_dimensions": spectral_dimensions,
                "feature_version": feature_version,
                "source_size": path.stat().st_size,
                "source_mtime_ns": path.stat().st_mtime_ns,
                "record": record.__dict__,
            },
            str(temporary),
        )
        temporary.replace(cache_path)
    return record


@dataclass(frozen=True)
class BackgroundFeatureScaler:
    mean: torch.Tensor
    scale: torch.Tensor
    passthrough_indices: Tuple[int, ...] = ()

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) / self.scale.to(values)

    def as_dict(self) -> dict:
        return {
            "artifact_type": "mokse_background_train_only_scaler_v1",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "passthrough_indices": list(self.passthrough_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]):
        if payload.get("artifact_type") != "mokse_background_train_only_scaler_v1":
            raise ValueError("background scaler artifact type mismatch")
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            scale=torch.tensor(payload["scale"], dtype=torch.float32),
            passthrough_indices=tuple(
                int(value) for value in payload.get("passthrough_indices", ())
            ),
        )


def fit_background_feature_scaler(
    records: Iterable[GlobalStaticGraphRecord],
    epsilon: float = 1.0e-6,
    passthrough_indices: Sequence[int] = (),
) -> BackgroundFeatureScaler:
    values = torch.cat(tuple(record.node_features for record in records), dim=0)
    if values.numel() == 0:
        raise ValueError("cannot fit background scaler on empty records")
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(epsilon)
    indices = tuple(sorted(set(int(index) for index in passthrough_indices)))
    if any(index < 0 or index >= values.shape[1] for index in indices):
        raise ValueError("background scaler passthrough index is out of range")
    if indices:
        mean = mean.clone()
        scale = scale.clone()
        mean[list(indices)] = 0.0
        scale[list(indices)] = 1.0
    return BackgroundFeatureScaler(
        mean=mean,
        scale=scale,
        passthrough_indices=indices,
    )


def fit_train_community_kappa(
    records: Iterable[GlobalStaticGraphRecord],
) -> float:
    """Fit the support-shrinkage reference from training communities only."""

    sizes = []
    for record in records:
        labels = torch.as_tensor(record.community_labels, dtype=torch.long)
        for label in torch.unique(labels[labels >= 0]):
            sizes.append(float((labels == label).sum()))
    if not sizes:
        raise ValueError("cannot fit community kappa without train communities")
    return float(torch.quantile(torch.tensor(sizes, dtype=torch.float64), 0.5))
