import torch
from .rcnn import RCNN2d, predict_rcnn2d
from .cgru import Conv2dGRU, predict_cgru
from .cnn import CNN, predict_cnn


def transpose_data(
    batch: torch.Tensor,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns inputs, labels
    (# samples, seq_len, C_in, H, W) -> (seq_len, # samples, C_in, H, W)
    """
    # .contiguous() to ensure a contiguous memory layout
    # without introducing spurious nodes into the graph
    batch = batch.transpose(0, 1).contiguous()
    return batch[:split], batch[split:]


__all__ = [
    'RCNN2d',
    'predict_rcnn2d',
    'Conv2dGRU',
    'predict_cgru',
    'CNN',
    'predict_cnn',
    'transpose_data',
]
