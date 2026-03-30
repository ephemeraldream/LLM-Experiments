from torch import nn 
import torch


def softmax(x:torch.Tensor, dim_i: int):
    x_max = x.max(dim=dim_i, keepdim=True).values
    x_shifted = x - x_max
    exp_x = x_shifted.exp()
    return exp_x / exp_x.sum(dim=dim_i, keepdim=True)    
