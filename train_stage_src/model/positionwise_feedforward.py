from einops import einsum
from torch import nn
import torch

from train_stage_src.core.init import DEFAULT_INIT_STD, init_residual_weight_, init_weight_



class FFN(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff: int | None = None,
        device=None,
        dtype=None,
        init_std: float = DEFAULT_INIT_STD,
        num_layers: int = 1,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        if d_ff is None:
            raw_d_ff = int(8 / 3 * d_model)
            d_ff = ((raw_d_ff + 63) // 64) * 64

        self.W1 = nn.Parameter(torch.empty(size=(d_ff, self.d_model), **factory_kwargs))
        self.W3 = nn.Parameter(torch.empty(size=(d_ff, self.d_model), **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty(size=(self.d_model, d_ff), **factory_kwargs))
        init_weight_(self.W1, std=init_std)
        init_weight_(self.W3, std=init_std)
        init_residual_weight_(self.W2, num_layers=num_layers, std=init_std)

    def forward(self, x: torch.Tensor):
        w1_x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3_x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1_x * torch.sigmoid(w1_x)
        inner_x = silu * w3_x
        return einsum(self.W2, inner_x, "d_model d_ff, ... d_ff -> ... d_model")