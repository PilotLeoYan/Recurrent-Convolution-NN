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
    scheduler_type: str = "cosine",
    # StepLR params
    step_size: int = 5,
    gamma: float = 0.5,
    # CosineAnnealingLR params
    T_max: int = 10,
    eta_min: float = 1e-6,
):

    """
    """
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"scheduler_type '{scheduler_type}' no reconocido. Usa 'cosine' o 'step'.")



__all__ = [
    'get_optimizer',
    'get_lr_scheduler',
]
