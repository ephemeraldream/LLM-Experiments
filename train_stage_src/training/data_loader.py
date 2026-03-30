import numpy as np
import torch

def get_batch(
    dataset: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {tuple(dataset.shape)}")

    dataset_len = len(dataset)
    if dataset_len <= context_length:
        raise ValueError(
            f"dataset is too short for context_length={context_length}: len(dataset)={dataset_len}"
        )

    max_start = dataset_len - context_length
    starts = (
        np.random.randint(0, max_start, size=batch_size)
        if rng is None
        else rng.integers(0, max_start, size=batch_size)
    )
    offsets = np.arange(context_length, dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]

    if isinstance(dataset, torch.Tensor):
        index_tensor = torch.from_numpy(indices).to(device=dataset.device)
        x = dataset[index_tensor].to(device=device, dtype=torch.long)
        y = dataset[index_tensor + 1].to(device=device, dtype=torch.long)
        return x, y

    x_np = np.asarray(dataset[indices], dtype=np.int64)
    y_np = np.asarray(dataset[indices + 1], dtype=np.int64)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y


def data_loader(
    x: np.ndarray | torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(
        dataset=x,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        rng=rng,
    )