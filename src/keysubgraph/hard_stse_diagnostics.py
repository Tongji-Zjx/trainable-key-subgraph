"""Auditable invariants and collapse summaries for Hard-STSE."""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List

import torch


def representation_summary(values: torch.Tensor) -> Dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("representation diagnostics require shape [B, D]")
    detached = values.detach().to(dtype=torch.float64, device="cpu")
    variance = detached.var(dim=0, unbiased=False)
    norms = detached.norm(dim=-1)
    if detached.shape[0] > 1:
        normalized = detached / norms[:, None].clamp_min(1.0e-12)
        cosine = normalized.matmul(normalized.transpose(0, 1))
        upper = torch.triu(
            torch.ones_like(cosine, dtype=torch.bool), diagonal=1
        )
        mean_cosine = float(cosine[upper].mean())
    else:
        mean_cosine = None
    return {
        "sample_count": int(detached.shape[0]),
        "dimension": int(detached.shape[1]),
        "mean_feature_variance": float(variance.mean()),
        "maximum_feature_variance": float(variance.max()),
        "active_feature_fraction": float((variance > 1.0e-6).double().mean()),
        "mean_pairwise_cosine": mean_cosine,
        "representation_norm": {
            "mean": float(norms.mean()),
            "standard_deviation": float(
                norms.var(unbiased=False).clamp_min(0.0).sqrt()
            ),
        },
    }


def audit_hard_stse_output(batch, output) -> Dict[str, Any]:
    if len(batch) != len(output.hard_windows):
        raise ValueError("hard output does not align with the input batch")
    failures: List[str] = []
    inventory = []
    valid_windows = 0
    total_windows = 0
    for sample, windows in zip(batch, output.hard_windows):
        if len(windows) != sample.num_timepoints:
            failures.append("{}:window_count".format(sample.sample_key))
        starts = tuple(float(value) for value in sample.window_starts)
        if any(right <= left for left, right in zip(starts[:-1], starts[1:])):
            failures.append("{}:time_order".format(sample.sample_key))
        inventory.append(
            {
                "sample_key": sample.sample_key,
                "label": int(sample.label),
                "timepoint_count": int(sample.num_timepoints),
                "window_starts": list(starts),
            }
        )
        for time_index, hard in enumerate(windows):
            total_windows += 1
            source_edges = sample.edge_mask[time_index].to(
                device=hard.hard_edge_mask.device, dtype=torch.bool
            )
            if bool((hard.hard_edge_mask & ~source_edges).any()):
                failures.append(
                    "{}:{}:new_edge".format(sample.sample_key, time_index)
                )
            source = sample.adjacency[time_index].to(hard.adjacency_st)
            expected = source * hard.hard_edge_mask.to(source.dtype)
            if not torch.allclose(
                hard.adjacency_st.detach(), expected, atol=1.0e-7, rtol=0.0
            ):
                failures.append(
                    "{}:{}:signed_weight".format(
                        sample.sample_key, time_index
                    )
                )
            if not torch.equal(
                hard.hard_edge_mask, hard.hard_edge_mask.transpose(0, 1)
            ):
                failures.append(
                    "{}:{}:asymmetric".format(sample.sample_key, time_index)
                )
            if hard.window_valid:
                valid_windows += 1
                cropped = hard.cropped_graph
                mask = (
                    cropped.adjacency.abs()
                    > float(cropped.edge_presence_threshold)
                )
                mask.fill_diagonal_(False)
                if not bool(mask.any(dim=-1).all()):
                    failures.append(
                        "{}:{}:isolated_node".format(
                            sample.sample_key, time_index
                        )
                    )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "sample_inventory": inventory,
        "total_window_count": total_windows,
        "valid_window_count": valid_windows,
        "invalid_window_count": total_windows - valid_windows,
    }
