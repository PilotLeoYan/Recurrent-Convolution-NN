import torch
from torch.nn import L1Loss, MSELoss


def metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """ """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    mae_loss = L1Loss()
    mse_loss = MSELoss()

    return {
        'mae': mae_loss(pred, target).item(),
        'mse': mse_loss(pred, target).item(),
    }
