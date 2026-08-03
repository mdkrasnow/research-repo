"""Checkpoint invariants and reproducible state fingerprints."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping

import torch


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def nested_state_sha256(value) -> str:
    """Stable digest for nested optimizer/config state without temp files."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda x: str(x)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for entry in item:
                visit(entry)
        elif isinstance(item, float):
            digest.update(b"float" + struct.pack("!d", item))
        elif isinstance(item, (int, bool)):
            digest.update(f"{type(item).__name__}:{item}".encode("ascii"))
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(f"{type(item).__name__}:{item}".encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def validate_base_checkpoint(checkpoint: dict, expected_epoch: int) -> dict:
    required = {"model", "ema", "opt", "epoch", "step", "args"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"base checkpoint missing required fields: {missing}")
    if int(checkpoint["epoch"]) != int(expected_epoch):
        raise ValueError(
            f"base epoch mismatch: expected {expected_epoch}, got {checkpoint['epoch']}"
        )
    optimizer = checkpoint["opt"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("base checkpoint optimizer state is empty")
    return {
        "epoch": int(checkpoint["epoch"]),
        "step": int(checkpoint["step"]),
        "model_sha256": tensor_mapping_sha256(checkpoint["model"]),
        "ema_sha256": tensor_mapping_sha256(checkpoint["ema"]),
        "optimizer_sha256": nested_state_sha256(checkpoint["opt"]),
        "optimizer_state_entries": len(optimizer["state"]),
        "optimizer_param_groups": len(optimizer["param_groups"]),
        "scheduler_present": "scheduler" in checkpoint,
    }


def models_numerically_identical(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[name], right[name]) for name in left)
