from einops import reduce
import torch
from torch import nn 


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            reduce(x ** 2, '... d_model -> ... 1', 'mean') + self.eps
        )
        x_norm = x / rms   
        out = x_norm * self.g  

        return out.to(in_type)