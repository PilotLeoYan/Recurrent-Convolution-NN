import torch
from torch.utils.data import DataLoader

from .dataset import MovingMNISTDataset
try:
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..utils.logger import get_logger

logger = get_logger(__name__)


def make_dataloader(
    dataset: MovingMNISTDataset,
    batch_size: int,
    train: bool = False,
    n_workers: int = 0,
) -> DataLoader:
    """
    """
    logger.info('Init dataloader from "%s"', dataset.filepath)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=n_workers,
        # pin_memory=True only benefits CUDA. On the CPU, it causes unnecessary overhead
        pin_memory=torch.cuda.is_available(), 
        persistent_workers=True if n_workers > 0 else False,
        prefetch_factor=None if n_workers == 0 else n_workers,
        drop_last=train,
    )
            