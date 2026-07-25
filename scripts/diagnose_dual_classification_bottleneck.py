"""Run all frozen D3 classification-bottleneck diagnostics at once."""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.analysis.dual_classification_bottleneck import (  # noqa: E402
    FEATURE_BLOCKS,
    analyze_dual_classification_bottleneck,
    write_dual_classification_bottleneck_artifacts,
)
from keysubgraph.data.data_protocol import validate_data_protocol  # noqa: E402
from keysubgraph.data.data_split import file_sha256  # noqa: E402
from keysubgraph.data.dual_sgw_manifest import (  # noqa: E402
    read_dual_sgw_manifest,
)
from keysubgraph.data.dual_sgw_scaler import (  # noqa: E402
    load_dual_sgw_standardizer,
)
from keysubgraph.data.exact_stse_dataset import (  # noqa: E402
    ExactSTSEBatch,
    ExactSTSEDataset,
    create_exact_stse_loader,
)
from keysubgraph.models.dual_stse_hard_sgw import (  # noqa: E402
    DualSTSEHardSGWClassifier,
)
from keysubgraph.training.dual_stse_hard_sgw_trainer import (  # noqa: E402
    load_dual_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--sgw-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, required=True
    )
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument(
        "--stability-samples-per-split", type=int, default=24
    )
    parser.add_argument(
        "--adjacency-perturbation-fraction", type=float, default=0.01
    )
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_provenance(
    protocol_sha,
    selector_sha,
    scaler_sha,
    train_manifest,
    validation_manifest,
    scaler,
    sgw_payload,
):
    if (
        train_manifest["split"] != "train"
        or validation_manifest["split"] != "validation"
    ):
        raise ValueError("bottleneck manifests use wrong splits")
    keys = (
        "protocol_sha256",
        "selector_checkpoint_sha256",
        "selection_mode",
        "selection_seed",
    )
    for key in keys:
        if train_manifest[key] != validation_manifest[key]:
            raise ValueError(
                "bottleneck manifests disagree on {}".format(key)
            )
    if train_manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("bottleneck manifest protocol mismatch")
    if train_manifest["selector_checkpoint_sha256"] != selector_sha:
        raise ValueError("bottleneck selector provenance mismatch")
    if train_manifest["selection_mode"] != "learned":
        raise ValueError("bottleneck diagnosis requires learned D3")
    if (
        scaler.protocol_sha256 != protocol_sha
        or scaler.selector_checkpoint_sha256 != selector_sha
        or scaler.selection_mode != train_manifest["selection_mode"]
        or int(scaler.selection_seed)
        != int(train_manifest["selection_seed"])
    ):
        raise ValueError("bottleneck scaler provenance mismatch")
    checkpoint_provenance = sgw_payload.get("provenance", {})
    if (
        checkpoint_provenance.get("selector_checkpoint_sha256")
        != selector_sha
        or checkpoint_provenance.get("sgw_scaler_sha256")
        != scaler_sha
    ):
        raise ValueError("bottleneck classifier provenance mismatch")


def _dataset(protocol, split):
    paths = protocol["paths"]
    return ExactSTSEDataset(
        PROJECT_ROOT / paths["dataset_root"],
        PROJECT_ROOT / paths["sample_index_csv"],
        PROJECT_ROOT / paths["splits_csv"],
        split,
        protocol["edge_presence_threshold"],
        require_coordinates=False,
    )


def _validate_coverage(dataset, records, name):
    expected = {
        item.sample_key: int(item.label) for item in dataset.assignments
    }
    actual = {record.sample_key: int(record.label) for record in records}
    if expected != actual:
        raise ValueError(
            "{} cache does not exactly cover the split".format(name)
        )


def _proxy_representation(model, batch, hard_windows):
    cores = []
    variations = []
    for sample, windows in zip(batch, hard_windows):
        core, variation, _ = model.proxy._sequence(
            windows,
            tuple(
                float(value) for value in sample.graph.window_starts
            ),
        )
        cores.append(core)
        variations.append(variation)
    return torch.cat(
        (torch.stack(cores, dim=0), torch.stack(variations, dim=0)),
        dim=-1,
    )


def _head_layers(head, scaler, raw):
    scaled = scaler(raw)
    hidden_linear = head[0](scaled)
    hidden_activation = head[1](hidden_linear)
    hidden_regularized = head[2](hidden_activation)
    logits = head[3](hidden_regularized)
    probabilities = torch.softmax(logits, dim=-1)[:, 1]
    return {
        "scaled_input": scaled,
        "hidden_linear": hidden_linear,
        "hidden_activation": hidden_activation,
        "hidden_regularized": hidden_regularized,
        "logits": logits,
        "positive_probability": probabilities[:, None],
    }


def _cutoff_margin(values, count):
    values = values.detach().reshape(-1)
    count = int(count)
    if count < 1 or count >= int(values.numel()):
        return None
    sorted_values = torch.sort(values, descending=True).values
    return float((sorted_values[count - 1] - sorted_values[count]).cpu())


def _edge_margin(selection):
    candidate = (
        selection.candidate_node_mask[:, None]
        & selection.candidate_node_mask[None, :]
        & (selection.edge_probabilities > 0.0)
    )
    upper = torch.triu(candidate, diagonal=1)
    indices = torch.nonzero(upper, as_tuple=False)
    if indices.numel() < 1:
        return None
    left, right = indices[:, 0], indices[:, 1]
    scores = selection.edge_probabilities[left, right] * torch.sqrt(
        (
            selection.node_probabilities[left]
            * selection.node_probabilities[right]
        ).clamp_min(0.0)
    )
    return _cutoff_margin(scores, selection.requested_edge_count)


def _jaccard(left, right):
    left = set(left)
    right = set(right)
    union = left | right
    return float(len(left & right)) / float(len(union)) if union else 1.0


def _node_set(sample, time_index, mask):
    names = sample.graph.node_names[time_index]
    return {
        names[index]
        for index, value in enumerate(mask.detach().cpu().tolist())
        if value
    }


def _edge_set(sample, time_index, mask):
    names = sample.graph.node_names[time_index]
    indices = torch.nonzero(
        torch.triu(mask.detach().cpu(), diagonal=1), as_tuple=False
    )
    return {
        tuple(sorted((names[int(left)], names[int(right)])))
        for left, right in indices.tolist()
    }


def _stable_seed(sample_key, time_index, seed):
    payload = "{}|{}|{}".format(sample_key, time_index, seed)
    return int(
        hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16
    )


def _perturb_batch(batch, fraction, seed):
    perturbed_samples = []
    for sample in batch:
        adjacency_values = []
        for time_index, adjacency in enumerate(sample.graph.adjacency):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(sample.sample_key, time_index, seed)
            )
            noise = torch.randn(
                tuple(adjacency.shape),
                generator=generator,
                dtype=torch.float32,
            ).to(adjacency)
            noise = 0.5 * (noise + noise.transpose(0, 1))
            factor = 1.0 + float(fraction) * torch.tanh(noise)
            perturbed = adjacency * factor
            perturbed = perturbed * sample.graph.edge_mask[
                time_index
            ].to(perturbed)
            perturbed.fill_diagonal_(0.0)
            adjacency_values.append(perturbed)
        graph = replace(
            sample.graph, adjacency=tuple(adjacency_values)
        )
        perturbed_samples.append(replace(sample, graph=graph))
    return ExactSTSEBatch(tuple(perturbed_samples))


def _stability_rows(
    batch,
    baseline,
    perturbed,
    perturbation_fraction,
):
    rows = []
    for sample_index, sample in enumerate(batch):
        baseline_windows = baseline.hard_windows[sample_index]
        perturbed_windows = perturbed.hard_windows[sample_index]
        previous_nodes = None
        previous_edges = None
        for time_index, (base_window, changed_window) in enumerate(
            zip(baseline_windows, perturbed_windows)
        ):
            base = base_window.selection
            changed = changed_window.selection
            base_nodes = _node_set(
                sample, time_index, base.hard_node_mask
            )
            changed_nodes = _node_set(
                sample, time_index, changed.hard_node_mask
            )
            base_edges = _edge_set(
                sample, time_index, base.hard_edge_mask
            )
            changed_edges = _edge_set(
                sample, time_index, changed.hard_edge_mask
            )
            rows.append(
                {
                    "sample_key": sample.sample_key,
                    "split": sample.split,
                    "label": sample.label,
                    "time_index": time_index,
                    "node_score_margin": _cutoff_margin(
                        base.node_probabilities,
                        base.requested_node_count,
                    ),
                    "edge_score_margin": _edge_margin(base),
                    "node_probability_mean": float(
                        base.node_probabilities.mean().detach().cpu()
                    ),
                    "node_probability_std": float(
                        base.node_probabilities.std(
                            unbiased=False
                        ).detach().cpu()
                    ),
                    "node_perturbation_jaccard": _jaccard(
                        base_nodes, changed_nodes
                    ),
                    "edge_perturbation_jaccard": _jaccard(
                        base_edges, changed_edges
                    ),
                    "temporal_node_jaccard": (
                        _jaccard(previous_nodes, base_nodes)
                        if previous_nodes is not None
                        else None
                    ),
                    "temporal_edge_jaccard": (
                        _jaccard(previous_edges, base_edges)
                        if previous_edges is not None
                        else None
                    ),
                    "adjacency_perturbation_fraction": float(
                        perturbation_fraction
                    ),
                }
            )
            previous_nodes = base_nodes
            previous_edges = base_edges
    return rows


def _scan_split(
    split,
    dataset,
    exact_lookup,
    selector_model,
    exact_model,
    scaler,
    device,
    selection_seed,
    num_workers,
    stability_keys,
    perturbation_fraction,
    perturbation_seed,
):
    loader = create_exact_stse_loader(
        dataset,
        batch_size=1,
        seed=selection_seed,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    labels = []
    proxy_features = []
    exact_features = []
    path_probabilities = {
        "proxy_all": [],
        "exact_all": [],
        **{
            "replace_{}".format(name): []
            for name, _, _ in FEATURE_BLOCKS
        },
    }
    layer_values = {
        "{}_{}".format(source, layer): []
        for source in ("proxy", "exact")
        for layer in (
            "scaled_input",
            "hidden_linear",
            "hidden_activation",
            "hidden_regularized",
            "logits",
            "positive_probability",
        )
    }
    stability_rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            batch = cpu_batch.to(device)
            selected = selector_model.selector(
                batch,
                selection_mode="learned",
                random_seed=selection_seed,
            )
            proxy_raw = _proxy_representation(
                selector_model, batch, selected.hard_windows
            ).to(torch.float32)
            exact_raw = torch.stack(
                [
                    exact_lookup[key].to(torch.float32)
                    for key in batch.sample_keys
                ],
                dim=0,
            ).to(device)
            proxy_layers = _head_layers(
                exact_model.sgw_auxiliary_head, scaler, proxy_raw
            )
            exact_layers = _head_layers(
                exact_model.sgw_auxiliary_head, scaler, exact_raw
            )
            labels.extend(int(sample.label) for sample in batch)
            proxy_features.extend(proxy_raw.cpu().tolist())
            exact_features.extend(exact_raw.cpu().tolist())
            path_probabilities["proxy_all"].extend(
                proxy_layers["positive_probability"]
                .reshape(-1)
                .cpu()
                .tolist()
            )
            path_probabilities["exact_all"].extend(
                exact_layers["positive_probability"]
                .reshape(-1)
                .cpu()
                .tolist()
            )
            for name, start, stop in FEATURE_BLOCKS:
                hybrid = proxy_raw.clone()
                hybrid[:, start:stop] = exact_raw[:, start:stop]
                probability = _head_layers(
                    exact_model.sgw_auxiliary_head, scaler, hybrid
                )["positive_probability"]
                path_probabilities[
                    "replace_{}".format(name)
                ].extend(probability.reshape(-1).cpu().tolist())
            for source, values in (
                ("proxy", proxy_layers),
                ("exact", exact_layers),
            ):
                for layer, tensor in values.items():
                    layer_values[
                        "{}_{}".format(source, layer)
                    ].extend(tensor.cpu().tolist())
            if batch.sample_keys[0] in stability_keys:
                changed_batch = _perturb_batch(
                    batch,
                    perturbation_fraction,
                    perturbation_seed,
                )
                changed = selector_model.selector(
                    changed_batch,
                    selection_mode="learned",
                    random_seed=selection_seed,
                )
                stability_rows.extend(
                    _stability_rows(
                        batch,
                        selected,
                        changed,
                        perturbation_fraction,
                    )
                )
            print(
                "{} processed {}/{} elapsed={:.1f}s".format(
                    split,
                    index + 1,
                    len(dataset),
                    time.perf_counter() - started,
                ),
                flush=True,
            )
    return {
        "labels": labels,
        "proxy": np.asarray(proxy_features, dtype=np.float64),
        "exact": np.asarray(exact_features, dtype=np.float64),
        "path_probabilities": path_probabilities,
        "layers": {
            name: np.asarray(values, dtype=np.float64)
            for name, values in layer_values.items()
        },
        "stability_rows": stability_rows,
    }


def _permutation_aucs(
    validation,
    exact_model,
    scaler,
    device,
    repeats,
    seed,
):
    features = torch.tensor(
        validation["proxy"], dtype=torch.float32, device=device
    )
    labels = np.asarray(validation["labels"], dtype=np.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    results = {name: [] for name, _, _ in FEATURE_BLOCKS}
    with torch.no_grad():
        for name, start, stop in FEATURE_BLOCKS:
            for _ in range(repeats):
                order = torch.randperm(
                    features.shape[0], generator=generator
                ).to(device)
                permuted = features.clone()
                permuted[:, start:stop] = features[
                    order, start:stop
                ]
                probability = _head_layers(
                    exact_model.sgw_auxiliary_head,
                    scaler,
                    permuted,
                )["positive_probability"].reshape(-1)
                results[name].append(
                    float(
                        roc_auc_score(
                            labels, probability.cpu().numpy()
                        )
                    )
                )
    return results


def main():
    args = parse_args()
    if (
        args.num_workers < 0
        or args.permutation_repeats < 1
        or args.stability_samples_per_split < 1
        or args.adjacency_perturbation_fraction <= 0.0
        or args.adjacency_perturbation_fraction >= 0.5
    ):
        raise ValueError("invalid bottleneck diagnostic configuration")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("bottleneck diagnostic output exists")
    protocol = validate_data_protocol(args.protocol, PROJECT_ROOT)
    protocol_sha = file_sha256(args.protocol)
    selector_sha = file_sha256(args.selector_checkpoint)
    scaler_sha = file_sha256(args.scaler)
    train_manifest, train_records, train_lookup = (
        read_dual_sgw_manifest(args.train_manifest)
    )
    validation_manifest, validation_records, validation_lookup = (
        read_dual_sgw_manifest(args.validation_manifest)
    )
    train_dataset = _dataset(protocol, "train")
    validation_dataset = _dataset(protocol, "validation")
    _validate_coverage(train_dataset, train_records, "train")
    _validate_coverage(
        validation_dataset, validation_records, "validation"
    )
    device = _device(args.device)
    selector_model = DualSTSEHardSGWClassifier().to(device)
    selector_payload = load_dual_checkpoint(
        args.selector_checkpoint,
        selector_model,
        device,
        expected_stage="selector_proxy",
        expected_protocol_sha256=protocol_sha,
    )
    exact_model = DualSTSEHardSGWClassifier().to(device)
    sgw_payload = load_dual_checkpoint(
        args.sgw_checkpoint,
        exact_model,
        device,
        expected_stage="sgw_classifier",
        expected_protocol_sha256=protocol_sha,
    )
    scaler = load_dual_sgw_standardizer(args.scaler).to(device).eval()
    _validate_provenance(
        protocol_sha,
        selector_sha,
        scaler_sha,
        train_manifest,
        validation_manifest,
        scaler,
        sgw_payload,
    )
    selector_model.eval()
    exact_model.eval()
    selection_seed = int(train_manifest["selection_seed"])
    random_source = random.Random(args.seed)

    def stability_keys(dataset):
        keys = sorted(item.sample_key for item in dataset.assignments)
        count = min(len(keys), args.stability_samples_per_split)
        return set(random_source.sample(keys, count))

    started = time.perf_counter()
    train = _scan_split(
        "train",
        train_dataset,
        train_lookup,
        selector_model,
        exact_model,
        scaler,
        device,
        selection_seed,
        args.num_workers,
        stability_keys(train_dataset),
        args.adjacency_perturbation_fraction,
        args.seed,
    )
    validation = _scan_split(
        "validation",
        validation_dataset,
        validation_lookup,
        selector_model,
        exact_model,
        scaler,
        device,
        selection_seed,
        args.num_workers,
        stability_keys(validation_dataset),
        args.adjacency_perturbation_fraction,
        args.seed + 1,
    )
    permutation_aucs = _permutation_aucs(
        validation,
        exact_model,
        scaler,
        device,
        args.permutation_repeats,
        args.seed,
    )
    analysis = analyze_dual_classification_bottleneck(
        train_labels=train["labels"],
        validation_labels=validation["labels"],
        train_proxy=train["proxy"],
        train_exact=train["exact"],
        validation_proxy=validation["proxy"],
        validation_exact=validation["exact"],
        path_probabilities={
            "train": train["path_probabilities"],
            "validation": validation["path_probabilities"],
        },
        permutation_aucs=permutation_aucs,
        layer_representations={
            "train": train["layers"],
            "validation": validation["layers"],
        },
        selector_stability_rows=(
            train["stability_rows"] + validation["stability_rows"]
        ),
    )
    provenance = {
        "read_only_frozen_models": True,
        "test_split_used": False,
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": protocol_sha,
        "selector_checkpoint": str(
            Path(args.selector_checkpoint).resolve()
        ),
        "selector_checkpoint_sha256": selector_sha,
        "selector_best_epoch": int(selector_payload["best_epoch"]),
        "sgw_checkpoint": str(Path(args.sgw_checkpoint).resolve()),
        "sgw_checkpoint_sha256": file_sha256(args.sgw_checkpoint),
        "sgw_best_epoch": int(sgw_payload["best_epoch"]),
        "scaler": str(Path(args.scaler).resolve()),
        "scaler_sha256": scaler_sha,
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest": str(
            Path(args.validation_manifest).resolve()
        ),
        "validation_manifest_sha256": file_sha256(
            args.validation_manifest
        ),
        "selection_seed": selection_seed,
        "diagnostic_seed": int(args.seed),
        "permutation_repeats": int(args.permutation_repeats),
        "stability_samples_per_split": int(
            args.stability_samples_per_split
        ),
        "adjacency_perturbation_fraction": float(
            args.adjacency_perturbation_fraction
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    paths = write_dual_classification_bottleneck_artifacts(
        output_dir, analysis, provenance
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": {
                    name: str(path) for name, path in paths.items()
                },
                "summary": analysis["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
