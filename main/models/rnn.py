import torch
from torch import nn


class RNN(nn.Module):
    """
    Vanilla RNN baseline for spatiotemporal prediction.

    Uses 1×1 convolutions as per-pixel linear transforms so the model
    requires no knowledge of H or W at construction time and keeps the
    same interface as RCNN2d and Conv2dGRU.

    Recurrence (per unit):
        h_t = tanh( W_ih · x_t  +  W_hh · h_{t-1} )

    No gating, no spatial mixing (kernel=1) — intentionally minimal so
    any performance gap against RCNN2d / Conv2dGRU is attributable to
    those architectural choices.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        units: int,
    ):
        super().__init__()

        self.name = 'rnn'
        self.hchns = hidden_channels
        self.units = units

        # input → hidden  (1×1 conv = per-pixel linear, no spatial mixing)
        self.ih = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels if u == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                )
                for u in range(units)
            ]
        )

        # hidden → hidden
        self.hh = nn.ModuleList(
            [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
                for u in range(units)
            ]
        )

        self.drops = nn.ModuleList(
            [nn.Dropout(p=0.2) for _ in range(units - 1)]
        )

        # project hidden state back to pixel space
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        x  : (seq_len, batch, C_in, H, W)
        h0 : (units, batch, hidden_channels, H, W) or None

        Returns
        -------
        projected : (seq_len, batch, C_in, H, W)
        h_out     : (units, batch, hidden_channels, H, W)
        """
        seq_len, batch, c_in, H, W = x.shape

        if h0 is None:
            h = [
                torch.zeros(batch, self.hchns, H, W, device=x.device)
                for _ in range(self.units)
            ]
        else:
            h = [s.clone() for s in h0.unbind(0)]

        outputs = []

        for t in range(seq_len):
            for u in range(self.units):
                inp = x[t] if u == 0 else h[u - 1]
                h[u] = torch.tanh(self.ih[u](inp) + self.hh[u](h[u]))
                if u < self.units - 1:
                    h[u] = self.drops[u](h[u])

            outputs.append(h[-1].clone())

        h_out = torch.stack(h)                                          # (units, batch, hchns, H, W)
        projected = torch.stack([self.output_proj(o) for o in outputs]) # (seq_len, batch, C_in, H, W)
        return projected, h_out

    def decode(
        self,
        pred_len: int,
        last_frame: torch.Tensor,
        h: torch.Tensor,
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressive decoding.

        last_frame : (batch, C_in, H, W)
        h          : (units, batch, hidden_channels, H, W)
        targets    : (pred_len, batch, C_in, H, W) — used for teacher forcing
        """
        predictions = []
        current = last_frame.unsqueeze(0)  # (1, batch, C_in, H, W)

        for t in range(pred_len):
            out, h = self.forward(current, h)
            pred = out[0]                  # (batch, C_in, H, W)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current = targets[t].unsqueeze(0)
            else:
                current = pred.detach().unsqueeze(0)

        return torch.stack(predictions)   # (pred_len, batch, C_in, H, W)


def predict_rnn(
    model: RNN,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Encode the input sequence, then decode autoregressively."""
    _, h = model(inputs)
    return model.decode(
        pred_len=labels.shape[0],
        last_frame=inputs[-1],
        h=h,
        targets=labels,
        teacher_forcing_ratio=kwargs['teacher_forcing_ratio'],
    )
