from torch import nn
import torch
from train_stage_src.core.init import DEFAULT_INIT_STD
from train_stage_src.model.positionwise_feedforward import FFN
from train_stage_src.model.RMSNorm import RMSNorm
from train_stage_src.model.scaled_dot_product_attn import MultiHeadAttention



class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int | None = None,
        theta: float = 10000,
        device=None,
        dtype=None,
        init_std: float = DEFAULT_INIT_STD,
        num_layers: int = 1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=True,
            theta=theta,
            device=device,
            dtype=dtype,
            init_std=init_std,
            num_layers=num_layers,
        )
        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
            init_std=init_std,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mha1 = self.attn_norm(x)
        x_mha1 = self.mha(x_mha1)
        x = x + x_mha1
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn
        return x