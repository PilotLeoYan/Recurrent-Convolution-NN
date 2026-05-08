from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        filepath: str | Path,
        transform = None,
    ):
        """
        """
        self.data = np.load(filepath, mmap_mode='r') # (seq_len, # samples, H, W)
        self.transform = transform

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        sample = torch.from_numpy(self.data[:, index, None]).float() / 255.0 # (seq_len, 1, H, W)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files