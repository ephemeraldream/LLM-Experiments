from __future__ import annotations

import os
from typing import IO, Any, BinaryIO

import torch

CheckpointTarget = str | os.PathLike | BinaryIO | IO[bytes]
CheckpointPayload = dict[str, Any]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
    extra_state: dict | None = None,
) -> None:
    payload: CheckpointPayload = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, out)


def load_checkpoint_payload(src: CheckpointTarget) -> CheckpointPayload:
    try:
        return torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(src, map_location="cpu")


def restore_checkpoint_payload(
    payload: CheckpointPayload,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_state_to_parameter_devices(optimizer)
    return int(payload["iteration"])


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = load_checkpoint_payload(src)
    return restore_checkpoint_payload(payload, model=model, optimizer=optimizer)


def move_optimizer_state_to_parameter_devices(optimizer: torch.optim.Optimizer) -> None:
    for param_group in optimizer.param_groups:
        for parameter in param_group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(parameter.device)
