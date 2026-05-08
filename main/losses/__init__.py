from torch import nn


def get_loss_fn() -> nn.MSELoss:
    """
    """
    return nn.MSELoss()


__all__ = [
    'get_loss_fn'
]