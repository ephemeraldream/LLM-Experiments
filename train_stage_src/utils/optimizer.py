from collections.abc import Callable
import math
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float | None = None,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        lam: float | None = None,
    ):
        if weight_decay is None:
            weight_decay = 0.0 if lam is None else lam
        elif lam is not None and not math.isclose(weight_decay, lam):
            raise ValueError("weight_decay and lam must match when both are provided")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {lam}")

        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t = state["step"]
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> AdamW:
    return AdamW(
        build_parameter_groups(model, weight_decay=weight_decay),
        lr=lr,
        betas=betas,
        eps=eps,
    )


def build_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))

        if use_weight_decay(name, parameter):
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    return [
        {"params": decay_params, "weight_decay": weight_decay, "apply_weight_decay": True},
        {"params": no_decay_params, "weight_decay": 0.0, "apply_weight_decay": False},
    ]


def use_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2:
        return False
    if name.endswith(".g"):
        return False
    if ".emb." in name or name.endswith("emb_mat"):
        return False
    return True


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def apply_optimizer_config(
    optimizer: torch.optim.Optimizer,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["betas"] = betas
        param_group["eps"] = eps
        if param_group.get("apply_weight_decay", False):
            param_group["weight_decay"] = weight_decay
        else:
            param_group["weight_decay"] = 0.0