import torch
from torch import nn

from train_stage_src.core.init import DEFAULT_INIT_STD
from train_stage_src.model.embedding import Embedding
from train_stage_src.model.linear import Linear
from train_stage_src.model.RMSNorm import RMSNorm
from train_stage_src.model.transformer_block import TransformerBlock


class GPTZero(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int | None = None,
        theta: float = 10000.0,
        tie_embeddings: bool = True,
        init_std: float = DEFAULT_INIT_STD,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.init_std = init_std

        self.emb = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
            init_std=init_std,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    d_ff=d_ff,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                    init_std=init_std,
                    num_layers=num_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
            init_std=init_std,
            weight=self.emb.emb_mat if tie_embeddings else None,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[-1] > self.context_length:
            raise ValueError(
                f"sequence length {input_ids.shape[-1]} exceeds context_length={self.context_length}"
            )

        x = self.emb(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        upgraded_state_dict = dict(state_dict)
        for layer_idx in range(self.num_layers):
            old_key = f"transformer_blocks.{layer_idx}.rmsnorm.g"
            attn_key = f"transformer_blocks.{layer_idx}.attn_norm.g"
            ffn_key = f"transformer_blocks.{layer_idx}.ffn_norm.g"
            if old_key in upgraded_state_dict:
                old_value = upgraded_state_dict.pop(old_key)
                upgraded_state_dict.setdefault(attn_key, old_value.clone())
                upgraded_state_dict.setdefault(ffn_key, old_value.clone())

        if self.tie_embeddings:
            emb_key = "emb.emb_mat"
            lm_head_key = "lm_head.W"
            if emb_key in upgraded_state_dict and lm_head_key not in upgraded_state_dict:
                upgraded_state_dict[lm_head_key] = upgraded_state_dict[emb_key]
            if lm_head_key in upgraded_state_dict and emb_key not in upgraded_state_dict:
                upgraded_state_dict[emb_key] = upgraded_state_dict[lm_head_key]

        return super().load_state_dict(upgraded_state_dict, strict=strict, assign=assign)
