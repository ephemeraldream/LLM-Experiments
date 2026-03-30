from einops import rearrange, reduce, einsum
from torch import nn
import torch
import numpy as np
import math


class RoPeBruteForce(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.R = self._prepare()
        self.register_buffer("R", self.R, persistent=False)
        
           
    def _prepare(self) -> torch.Tensor:
        """R_i"""
        thetas = [[i/(self.theta**((2 * k)/self.d_k)) for k in range(self.d_k // 2)] for i in range(self.max_seq_len)]
        R = torch.zeros(size=(self.d_k, self.d_k, self.max_seq_len))
        for i in range(self.max_seq_len):
            R_ik_mats = []
            for k in range(0,self.d_k // 2):
                R_ik = torch.tensor([[math.cos(thetas[i][k]), -math.sin(thetas[i][k])],
                                    [math.sin(thetas[i][k]),math.cos(thetas[i][k])]])
                R_ik_mats.append(R_ik)
            R[:,:,i] = torch.block_diag(*R_ik_mats)
        print(R.data)
        return R


    
            
            
        
    
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        """x: (..., seq_len, d_model)"""
        return einsum(self.R, x, ' d_model d_model seq_len, ... seq_len d_model -> ... seq_len d_model')
                


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        R = self._prepare()
        self.register_buffer("R", R, persistent=False)

    def _prepare(self) -> torch.Tensor:
        thetas = [
            [i / (self.theta ** ((2 * k) / self.d_k)) for k in range(self.d_k // 2)]
            for i in range(self.max_seq_len)
        ]
        angles = torch.tensor(thetas, dtype=torch.float32, device=self.device)  
        cosines,sines = torch.cos(angles), torch.sin(angles) 
        return torch.stack((cosines, sines), dim=0) 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.to(dtype=torch.long, device=self.R.device)

        cos = self.R[0][token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.R[1][token_positions].to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]  # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k//2)

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        out = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)
        return out
