from __future__ import annotations

import math

import torch


DEFAULT_INIT_STD = 0.02


def init_weight_(weight: torch.Tensor, std: float = DEFAULT_INIT_STD) -> None:
    torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


def init_residual_weight_(weight: torch.Tensor, num_layers: int, std: float = DEFAULT_INIT_STD) -> None:
    scaled_std = std / math.sqrt(2 * max(num_layers, 1))
    init_weight_(weight, std=scaled_std)
