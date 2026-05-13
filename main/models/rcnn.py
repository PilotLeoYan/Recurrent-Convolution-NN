import torch
from torch import nn


class RCNN2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        units: int,
        activation: str = "relu",
    ):
        super().__init__()

        self.name = 'rcnn2d'
        self.ichns = input_channels
        self.hchns = hidden_channels
        self.units = units

        # init ModuleList with all Conv2d for input
        self.conv2d_ih = nn.ModuleList(
            [
                nn.Conv2d(
                    # the first unit receive ci, the rest hi
                    input_channels if u == 0 else self.hchns,
                    self.hchns,
                    kernel_size,
                    padding=kernel_size // 2,  # same padding to hold Height, Width
                )
                for u in range(self.units)
            ]
        )

        # init ModuleList with all Conv2d for hiddens
        self.conv2d_hh = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hchns,
                    self.hchns,
                    kernel_size,
                    padding=kernel_size // 2,  # same padding to hold Height, Width
                )
                for u in range(self.units)
            ]
        )

        # init all dropout layers
        self.drops = nn.ModuleList([nn.Dropout(p=0.2) for u in range(self.units - 1)])

        # init activation function
        self.acti = nn.ModuleList(
            [
                nn.ReLU() if activation == "relu" else nn.Tanh()
                for u in range(self.units)
            ]
        )

        # maps the hidden state back to the pixel space
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hchns, self.ichns, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for u in range(self.units):
            nn.init.kaiming_normal_(self.conv2d_ih[u].weight, mode='fan_out', nonlinearity='relu') # type: ignore
            nn.init.orthogonal_(self.conv2d_hh[u].weight) # type: ignore
            nn.init.zeros_(self.conv2d_ih[u].bias) # type: ignore
            nn.init.zeros_(self.conv2d_hh[u].bias) # type: ignore
        nn.init.xavier_uniform_(self.output_proj[0].weight) # type: ignore
        nn.init.zeros_(self.output_proj[0].bias) # type: ignore

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        seq_len, batch, c_in, H, W = x.shape

        if h0 is None:
            h = [
                torch.zeros(batch, self.hchns, H, W, device=x.device) # type: ignore
                for i in range(self.units)
            ]
        else:
            h = [s.clone() for s in h0.unbind(0)]

        outputs = []

        for t in range(seq_len):
            for u in range(self.units):
                # for the first units, receive x[t]
                # the rest of units receive h from the previous unit
                input_x = x[t] if u == 0 else h[u - 1]

                ih = self.conv2d_ih[u](input_x)
                hh = self.conv2d_hh[u](h[u])

                if u < self.units - 1:
                    h[u] = self.acti[u](self.drops[u](ih + hh))
                else:
                    h[u] = self.acti[u](ih + hh)

            outputs.append(h[-1].clone())

        h_out = torch.stack(h) # (units, batch, hchns, H, W)
        projected = torch.stack([self.output_proj(o) for o in outputs])
        return projected, h_out

    def decode(
        self,
        pred_len: int,
        last_frame: torch.Tensor,
        h: torch.Tensor,
        targets: torch.Tensor | None = None,  # (pred_len, batch, C_in, H, W)
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        """
        predictions = []
        current = last_frame.unsqueeze(0)  # (1, batch, C_in, H, W)

        for t in range(pred_len):
            out, h = self.forward(current, h)  # (1, batch, C_in, H, W)

            pred = out[0]  # (batch, C_in, H, W)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current = targets[t].unsqueeze(0)  # ground truth
            else:
                current = pred.unsqueeze(0)

        return torch.stack(predictions)


def predict_rcnn2d(
    model: RCNN2d,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    """
    _, h = model(inputs)
    return model.decode(
        pred_len=labels.shape[0],
        last_frame=inputs[-1],
        h=h,
        targets=labels,
        teacher_forcing_ratio=kwargs['teacher_forcing_ratio'],
    )
