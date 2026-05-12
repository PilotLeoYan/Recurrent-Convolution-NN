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


def get_lr_scheduler(
    optimizer,
    step_size: int,
    gamma: float
):
    """
    """
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )


__all__ = [
    'get_optimizer',
    'get_lr_scheduler',
]
