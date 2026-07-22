"""Versioned checkpoint contract for the Stage-C 226-D hard student."""

from __future__ import absolute_import, print_function

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from keysubgraph.models.tg_sgw_types import (
    TG_SGW_CHECKPOINT_SCHEMA_VERSION,
    TG_SGW_HARD_STUDENT_STAGE,
    TG_SGW_MODEL_NAME,
    TGSGWContract,
    validate_tg_sgw_checkpoint_header,
)


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def save_tg_hard_student_checkpoint(
    path: Path,
    model,
    theory_standardizer,
    epoch: int,
    protocol_sha256: str,
    teacher_checkpoint_sha256: str,
    candidate_scaler_sha256: str,
    theory_scaler_sha256: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    history=None,
) -> Dict[str, Any]:
    if epoch < 0:
        raise ValueError("hard student checkpoint epoch cannot be negative")
    payload = {
        "model_name": TG_SGW_MODEL_NAME,
        "schema_version": TG_SGW_CHECKPOINT_SCHEMA_VERSION,
        "stage": TG_SGW_HARD_STUDENT_STAGE,
        "epoch": int(epoch),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "theory_standardizer": theory_standardizer.to_dict(),
        "protocol_sha256": str(protocol_sha256),
        "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
        "candidate_scaler_sha256": str(candidate_scaler_sha256),
        "theory_scaler_sha256": str(theory_scaler_sha256),
        "contract": TGSGWContract().to_dict(),
        "history": list(history or ()),
    }
    _atomic_save(path, payload)
    return payload


def load_tg_hard_student_checkpoint(
    path: Path,
    model,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    expected_protocol_sha256: Optional[str] = None,
    expected_teacher_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        payload = torch.load(str(Path(path).resolve()), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(Path(path).resolve()), map_location=device)
    validate_tg_sgw_checkpoint_header(payload, TG_SGW_HARD_STUDENT_STAGE)
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("TG hard student model configuration mismatch")
    if expected_protocol_sha256 is not None and payload.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("TG hard student protocol hash mismatch")
    if expected_teacher_checkpoint_sha256 is not None and payload.get("teacher_checkpoint_sha256") != expected_teacher_checkpoint_sha256:
        raise ValueError("TG hard student teacher checkpoint mismatch")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise ValueError("TG hard student checkpoint has no optimizer state")
        optimizer.load_state_dict(state)
    return payload
