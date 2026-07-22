"""Versioned and atomic TG-SGW soft-teacher checkpoint utilities."""

from __future__ import absolute_import, division, print_function

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from keysubgraph.models.tg_sgw_types import (
    TG_SGW_CHECKPOINT_SCHEMA_VERSION,
    TG_SGW_MODEL_NAME,
    TG_SGW_SOFT_TEACHER_STAGE,
    TGSGWContract,
    validate_tg_sgw_checkpoint_header,
)


def _rng_state(data_loader=None) -> Dict[str, Any]:
    generator = getattr(data_loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": generator.get_state() if generator is not None else None,
    }


def _restore_rng_state(payload: Dict[str, Any], data_loader=None) -> None:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        raise ValueError("TG-SGW checkpoint is missing reproducibility state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator = getattr(data_loader, "generator", None)
    if generator is not None and state.get("data_loader_generator") is not None:
        generator.set_state(state["data_loader_generator"])


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def build_tg_soft_teacher_checkpoint(
    model,
    optimizer,
    epoch: int,
    history,
    loss_config,
    training_config,
    protocol_path: Path,
    protocol_sha256: str,
    best_epoch: int,
    best_selection_value: float,
    data_loader=None,
) -> Dict[str, Any]:
    if epoch < 1:
        raise ValueError("checkpoint epoch must be positive")
    contract = TGSGWContract()
    return {
        "model_name": TG_SGW_MODEL_NAME,
        "schema_version": TG_SGW_CHECKPOINT_SCHEMA_VERSION,
        "stage": TG_SGW_SOFT_TEACHER_STAGE,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "loss_config": asdict(loss_config),
        "training_config": asdict(training_config),
        "contract": contract.to_dict(),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_selection_value": float(best_selection_value),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": str(protocol_sha256),
        "rng_state": _rng_state(data_loader),
        "torch_version": str(torch.__version__),
    }


def save_tg_soft_teacher_checkpoint(path: Path, **kwargs) -> Dict[str, Any]:
    payload = build_tg_soft_teacher_checkpoint(**kwargs)
    _atomic_torch_save(path, payload)
    return payload


def _trusted_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(str(Path(path).resolve()), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(Path(path).resolve()), map_location=device)


def load_tg_soft_teacher_checkpoint(
    path: Path,
    model,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    expected_protocol_sha256: Optional[str] = None,
    data_loader=None,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    payload = _trusted_torch_load(path, device)
    validate_tg_sgw_checkpoint_header(payload, TG_SGW_SOFT_TEACHER_STAGE)
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("TG-SGW soft-teacher model configuration mismatch")
    if expected_protocol_sha256 is not None and payload.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("TG-SGW checkpoint data protocol hash mismatch")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        _restore_rng_state(payload, data_loader=data_loader)
    return payload

