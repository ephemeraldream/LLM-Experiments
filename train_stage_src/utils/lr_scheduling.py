import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    t = it
    tw = warmup_iters
    tc = cosine_cycle_iters
    alpha_max = max_learning_rate
    alpha_min = min_learning_rate

    if t < tw:
        return (t / tw) * alpha_max
    if t > tc:
        return alpha_min

    progress = (t - tw) / (tc - tw)
    return alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
