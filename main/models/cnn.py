import torch
from torch import nn


class CNN(nn.Module):
    """
    Pure CNN baseline — no recurrence.

    Each frame is processed independently by a stack of convolutional
    layers.  There is no hidden state carried between timesteps; the
    model has zero temporal memory.

    This isolates the contribution of convolution alone, so any gap
    vs RCNN2d / Conv2dGRU is attributable to the recurrent component.

    Parameters
    ----------
    input_channels : C (channels per frame)
    hidden_channels: feature maps in intermediate conv layers
    kernel_size    : spatial kernel (same padding preserves H, W)
    units          : number of stacked conv layers
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        units: int,
    ):
        super().__init__()

        self.name = 'cnn'
        padding = kernel_size // 2

        layers: list[nn.Module] = []
        for u in range(units):
            in_ch = input_channels if u == 0 else hidden_channels
            layers += [
                nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding),
                nn.ReLU(),
            ]
            if u < units - 1:
                layers.append(nn.Dropout(p=0.2))

        self.encoder = nn.Sequential(*layers)

        # project hidden feature map back to pixel space
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, h0=None):
        """
        x  : (seq_len, batch, C, H, W)
        h0 : ignored — CNN has no recurrent state

        Returns
        -------
        projected : (seq_len, batch, C, H, W)
        h_out     : None  (no hidden state)
        """
        seq_len, batch, C, H, W = x.shape

        # process every frame independently
        outputs = []
        for t in range(seq_len):
            feat = self.encoder(x[t])            # (batch, hidden_channels, H, W)
            outputs.append(self.output_proj(feat))  # (batch, C, H, W)

        projected = torch.stack(outputs)         # (seq_len, batch, C, H, W)
        return projected, None                   # None: no hidden state

    def decode(
        self,
        pred_len: int,
        last_frame: torch.Tensor,
        h,                                       # ignored
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        last_frame : (batch, C, H, W)
        """
        predictions = []
        current = last_frame.unsqueeze(0)        # (1, batch, C, H, W)

        for t in range(pred_len):
            out, _ = self.forward(current)
            pred = out[0]                        # (batch, C, H, W)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current = targets[t].unsqueeze(0)
            else:
                current = pred.detach().unsqueeze(0)

        return torch.stack(predictions)          # (pred_len, batch, C, H, W)


def predict_cnn(
    model: CNN,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Encode input sequence, then decode autoregressively."""
    return model.decode(
        pred_len=labels.shape[0],
        last_frame=inputs[-1],
        h=None,
        targets=labels,
        teacher_forcing_ratio=kwargs['teacher_forcing_ratio'],
    )
