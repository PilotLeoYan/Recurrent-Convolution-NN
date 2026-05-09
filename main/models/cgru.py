import torch
from torch import nn


class Conv2dGRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        units: int,
    ):
        super().__init__()

        self.name = 'cgru'
        self.hchns = hidden_channels
        self.units = units

        def _make_convs(in_ch):
            """ """
            p = kernel_size // 2
            return nn.ModuleDict(
                {
                    "xz": nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=p),
                    "hz": nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size, padding=p
                    ),
                    "xr": nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=p),
                    "hr": nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size, padding=p
                    ),
                    "xn": nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=p),
                    "hn": nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size, padding=p
                    ),
                }
            )

        self.cells = nn.ModuleList(
            [
                _make_convs(input_channels if u == 0 else hidden_channels)
                for u in range(units)
            ]
        )

        self.drops = nn.ModuleList([nn.Dropout(p=0.2) for u in range(units - 1)])

        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _gru_cell(self, x: torch.Tensor, h: torch.Tensor, u: int) -> torch.Tensor:
        c = self.cells[u]
        # (z) update gate, (r) reset gate, (n) candidate
        z = torch.sigmoid(c["xz"](x) + c["hz"](h)) # type: ignore
        r = torch.sigmoid(c["xr"](x) + c["hr"](h)) # type: ignore
        n = torch.tanh(c["xn"](x) + c["hn"](r * h)) # type: ignore
        return (1 - z) * h + z * n

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        seq_len, batch, c_in, H, W = x.shape

        if h0 is None:
            h = [
                torch.zeros(batch, self.hchns, H, W, device=x.device)  # type: ignore
                for i in range(self.units)
            ]
        else:
            h = [s.clone() for s in h0.unbind(0)]

        outputs = []
        for t in range(seq_len):
            for u in range(self.units):
                inp = x[t] if u == 0 else h[u - 1]
                h[u] = self._gru_cell(inp, h[u], u)
                
                if u < self.units - 1:
                    h[u] = self.drops[u](h[u])
                outputs.append(h[-1].clone())

        h_out = torch.stack(h) # (units, batch, hchns, H, W)
        projected = torch.stack([self.output_proj(o) for o in outputs])
        return projected, h_out

    def decode(
        self,
        pred_len: int,
        last_frame: torch.Tensor,
        h: torch.Tensor,
        targets: torch.Tensor | None = None, # (pred_len, batch, C_in, H, W)
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """ """
        predictions = []
        current = last_frame.unsqueeze(0) # (1, batch, C_in, H, W)

        for t in range(pred_len):
            out, h = self.forward(current, h) # (1, batch, C_in, H, W)
            pred = out[0] # (batch, C_in, H, W)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current = targets[t].unsqueeze(0) # ground truth
            else:
                current = pred.detach().unsqueeze(0)

        return torch.stack(predictions)


def predict_cgru(
    model: Conv2dGRU, inputs: torch.Tensor, labels: torch.Tensor, **kwargs
) -> torch.Tensor:
    """ """
    _, h = model(inputs)
    return model.decode(
        pred_len=labels.shape[0],
        last_frame=inputs[-1],
        h=h,
        targets=labels,
        teacher_forcing_ratio=kwargs["teacher_forcing_ratio"],
    )
