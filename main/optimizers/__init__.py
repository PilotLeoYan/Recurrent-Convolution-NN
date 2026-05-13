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
    scheduler_type: str = "cosine_warmup",
    step_size: int = 5,
    gamma: float = 0.5,
    T_max: int = 10,
    eta_min: float = 1e-6,
    patience: int = 3,
    warmup_epochs: int = 2,
):
    """
    """
    if scheduler_type == "cosine_warmup":
        # Warmup lineal → coseno. T_max ya sólo controla la fase coseno.
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(T_max - warmup_epochs, 1), eta_min=eta_min
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_type == "plateau":
        # El mejor para cuando no sabes cuántos epochs necesitas
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma,
            patience=patience, min_lr=eta_min,
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    raise ValueError(f"scheduler_type '{scheduler_type}' not recognized.")



__all__ = [
    'get_optimizer',
    'get_lr_scheduler',
]
