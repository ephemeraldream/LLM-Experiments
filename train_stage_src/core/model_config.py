from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 8
    d_ff: int | None = None
    theta: float = 10_000.0
    tie_embeddings: bool = True
    init_std: float = 0.02


def model_config_from_dict(
    raw: Mapping[str, Any] | ModelConfig,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> ModelConfig:
    if isinstance(raw, ModelConfig):
        return raw

    data = dict(raw)
    if "tie_embeddings" not in data:
        data["tie_embeddings"] = _infer_tied_embeddings(state_dict)
    if "init_std" not in data:
        data["init_std"] = 0.02
    return ModelConfig(**data)


def _infer_tied_embeddings(state_dict: Mapping[str, torch.Tensor] | None) -> bool:
    if state_dict is None:
        return True

    emb_weight = state_dict.get("emb.emb_mat")
    lm_head_weight = state_dict.get("lm_head.W")
    if emb_weight is None or lm_head_weight is None:
        return True
    return torch.equal(emb_weight, lm_head_weight)
