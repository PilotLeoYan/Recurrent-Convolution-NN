import torch
from torch import nn
from torch.nn import functional as F


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11
) -> torch.Tensor:
    """SSIM simplified explanation of frame batches (B, C, H, W)"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sig_p  = (F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_p2).clamp(min=1e-8)
    sig_t  = (F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_t2).clamp(min=1e-8)
    sig_pt =  F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_pt
    num   = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    denom = (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2)
    ssim  = num / denom.clamp(min=1e-8)
    return (1 - ssim.clamp(-1, 1)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = pred.shape
        p = pred.view(T * B, C, H, W)
        t = target.reshape(T * B, C, H, W)
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * _ssim_loss(p, t, self.window_size)


def get_loss_fn(
    alpha: float = 0.85
) -> CombinedLoss:
    """
    """
    return CombinedLoss(alpha)


__all__ = [
    'get_loss_fn'
]
