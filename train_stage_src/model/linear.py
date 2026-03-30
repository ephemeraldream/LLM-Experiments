import torch.nn as nn
import torch
from einops import einsum

from train_stage_src.core.init import DEFAULT_INIT_STD, init_weight_

class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device=None,
        dtype=None,
        init_std: float = DEFAULT_INIT_STD,
        weight: nn.Parameter | None = None,
    ):
        super().__init__()
        if weight is None:
            factory_kwargs = {"device": device, "dtype": dtype}
            weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
            init_weight_(weight, std=init_std)
        self.W = weight

    def forward(self, x):
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")