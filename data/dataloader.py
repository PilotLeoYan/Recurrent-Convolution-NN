from torch.utils.data import Dataset, DataLoader


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    train: bool = False,
    *,
    n_workers: int | None = None
) -> DataLoader:
    """
    """
    if n_workers is None:
        import os
        cpu_count = os.cpu_count()

        assert cpu_count is not None, f'Specify n_workers, cpu_count: {cpu_count}'

        n_workers = min(cpu_count // 2, 8)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=train,
    )
            