import torch
from torch import nn


class RCNN2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        units: int,
        activation: str = 'relu',
    ):
        super().__init__()

        self.hchns = hidden_channels
        self.units = units

        # init ModuleList with all Conv2d for input
        self.conv2d_ih = nn.ModuleList([
            nn.Conv2d(
                # the first unit receive ci, the rest hi
                input_channels if u == 0 else self.hchns,
                self.hchns,
                kernel_size,
                padding=kernel_size // 2 # same padding to hold Height, Width
            )
            for u in range(self.units)
        ])

        # init ModuleList with all Conv2d for hiddens
        self.conv2d_hh = nn.ModuleList([
            nn.Conv2d(
                self.hchns,
                self.hchns,
                kernel_size,
                padding=kernel_size // 2 # same padding to hold Height, Width
            )
            for u in range(self.units)
        ])

        # init activation function
        self.acti = nn.ModuleList([
            nn.ReLU() if activation == 'relu' else nn.Tanh()
            for u in range(self.units)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        h0: torch.Tensor | None = None
    ):
        seq_len, batch, c_in, H, W = x.shape

        if h0 is None:
            h = torch.zeros(self.units, batch, self.hchns, H, W,
                device=x.device)
        else:
            h = h0.clone()

        outputs = []

        for t in range(seq_len):
            for u in range(self.units):
                # for the first units, receive x[t]
                # the rest of units receive h from the previous unit
                input_x = x[t] if u == 0 else h[u - 1]

                ih = self.conv2d_ih[u](input_x)
                hh = self.conv2d_hh[u](h[u])
                h[u] = self.acti[u](ih + hh)
                #h[u] = torch.tanh(ih + hh)

            outputs.append(h[-1].clone())

        return torch.stack(outputs), h