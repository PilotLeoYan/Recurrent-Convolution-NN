import torch
from .rcnn import RCNN2d, predict_rcnn2d
from .cgru import Conv2dGRU, predict_cgru
from .rnn import RNN, predict_rnn


def transpose_data(
    batch: torch.Tensor,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns inputs, labels
    """
    # from (# samples, seq_len, C_in, H, W) -> (seq_len, # samples, C_in, H, W)
    batch = batch.clone().transpose(0, 1)
    return batch[:split], batch[split:]

    
__all__ = [
    'RCNN2d',
    'predict_rcnn2d',
    'Conv2dGRU',
    'predict_cgru',
    'RNN',
'predict_rnn',
]
