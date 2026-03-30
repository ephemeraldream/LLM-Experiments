import torch
from torch import nn
import einops
import math
from train_stage_src.core.init import DEFAULT_INIT_STD, init_residual_weight_, init_weight_
from train_stage_src.model.RoPe import RotaryPositionalEmbedding



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        mask=None,
        use_rope: bool = False,
        device=None,
        dtype=None,
        init_std: float = DEFAULT_INIT_STD,
        num_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.mask = mask
        self.use_rope = use_rope
        self.device = device
        
        factory_kwargs = {"device": device, "dtype": dtype}

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

        self.Wq = nn.Parameter(torch.empty(size=(d_model, d_model), **factory_kwargs))
        self.Wk = nn.Parameter(torch.empty(size=(d_model, d_model), **factory_kwargs))
        self.Wv = nn.Parameter(torch.empty(size=(d_model, d_model), **factory_kwargs))
        self.Wo = nn.Parameter(torch.empty(size=(d_model, d_model), **factory_kwargs))

        init_weight_(self.Wq, std=init_std)
        init_weight_(self.Wk, std=init_std)
        init_weight_(self.Wv, std=init_std)
        init_residual_weight_(self.Wo, num_layers=num_layers, std=init_std)

    def forward(self, x: torch.Tensor, token_positions=None):
        _, seq_len, _ = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        Q = einops.einsum(x, self.Wq, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        K = einops.einsum(x, self.Wk, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        V = einops.einsum(x, self.Wv, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        
        Q_heads = einops.rearrange(Q, 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        K_heads = einops.rearrange(K, 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        V_heads = einops.rearrange(V, 'batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v', num_heads=self.num_heads)
        
        if self.use_rope:
            Q_heads_rope = torch.zeros_like(Q_heads)
            K_heads_rope = torch.zeros_like(K_heads)
            for h in range(self.num_heads):
                Q_heads_rope[:, h, :, :] = self.rope.forward(Q_heads[:, h, :, :], token_positions)
                K_heads_rope[:, h, :, :] = self.rope.forward(K_heads[:, h, :, :], token_positions)
            Q_heads, K_heads = Q_heads_rope, K_heads_rope

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        if self.mask is not None:
            final_mask = causal_mask & self.mask
        else:
            final_mask = causal_mask
        attention_output = scaled_dot_product_attention(Q_heads, K_heads, V_heads, final_mask)
        attention_output = einops.rearrange(attention_output, 'batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)')
        output = einops.einsum(attention_output, self.Wo, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        
        return output 
        

def scaled_dot_product_attention(Q,K,V, mask=None):
    QtK = einops.einsum(Q,K, 'batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k') / math.sqrt(K.shape[-1])
    if mask is not None:
        QtK = QtK.masked_fill(~mask, float('-inf'))
    scores = torch.softmax(QtK, dim=-1)
    return einops.einsum(scores, V, 'batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v')



