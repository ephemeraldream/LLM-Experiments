from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    cosine_cycle_iters: int | None = None
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float | None = 1.0

    def resolved_cosine_cycle_iters(self, max_iters: int) -> int | None:
        if self.cosine_cycle_iters is None:
            return max_iters
        return self.cosine_cycle_iters


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 16
    max_iters: int = 500
    eval_every: int = 100
    eval_steps: int = 20
    log_every: int = 10
    save_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    experiment_name: str | None = None
    experiment_dir: str | os.PathLike | None = None
    tensorboard_logdir: str | os.PathLike | None = None


@dataclass(slots=True)
class TrainResult:
    iteration: int
    train_loss: float
    valid_loss: float | None
    best_valid_loss: float | None
    checkpoint_path: str | None
    best_checkpoint_path: str | None
    experiment_dir: str | None
