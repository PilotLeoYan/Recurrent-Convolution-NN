from torch import optim


def get_optimizer(
    model_parameters,
    lr: float,
    weight_decay: float,
):
    """
    """
    return optim.AdamW(
        params=model_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


__all__ = [
    'get_optimizer'
]