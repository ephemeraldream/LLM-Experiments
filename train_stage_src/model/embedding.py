from torch import nn
import torch

from train_stage_src.core.init import DEFAULT_INIT_STD, init_weight_


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None,
        init_std: float = DEFAULT_INIT_STD,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.emb_mat = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), **factory_kwargs))
        init_weight_(self.emb_mat, std=init_std)

    def forward(self, token_ids: torch.Tensor):
        return self.emb_mat[token_ids]
