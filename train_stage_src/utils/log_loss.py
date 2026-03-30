import torch

def log_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
    flat_targets = targets.reshape(-1)

    max_els = flat_logits.amax(dim=-1, keepdim=True)
    losumexp = max_els.squeeze(-1) + torch.log(torch.exp(flat_logits - max_els).sum(dim=-1))
    target_logits = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    losses = losumexp - target_logits
    return losses.mean()
