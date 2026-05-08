from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..utils.logger import get_logger

logger = get_logger(__name__)


class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        filepath: str | Path,
        transform = None,
    ):
        """
        """
        logger.info('Init MovingMNISTDataset from "%s"', filepath)

        self.filepath = filepath
        self.data = np.load(self.filepath, mmap_mode='r') # (# samples, seq_len, H, W)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = (
            torch.from_numpy(np.array(self.data[index]))
            .unsqueeze(1)
            .float()
            .div_(2550.0)
        ) # (seq_len, 1, H, W)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files