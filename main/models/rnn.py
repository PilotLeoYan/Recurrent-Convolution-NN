import torch
from torch import nn


class RNN(nn.Module):
    """
    Pure vanilla RNN baseline — no convolution of any kind.

    Each frame is flattened to a 1-D vector, processed by fully-connected
    (Linear) layers, then reshaped back to the original spatial dims.

    Recurrence (per unit):
        h_t = tanh( W_ih · flatten(x_t)  +  W_hh · h_{t-1} )

    Because there is zero spatial mixing (no Conv2d, not even 1×1), any
    performance gap vs RCNN2d / Conv2dGRU is directly attributable to the
    convolutional inductive bias in those architectures.

    nn.LazyLinear is used for the input projection so H and W do not need
    to be known at construction time — the layer initialises itself on the
    first forward pass.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        units: int,
    ):
        super().__init__()

        self.name = 'rnn'
        self.hidden_size = hidden_size
        self.units = units
        self.input_channels = input_channels

        # LazyLinear: infers input size (C*H*W) on first forward call
        self.ih = nn.ModuleList(
            [
                nn.LazyLinear(hidden_size) if u == 0 else nn.Linear(hidden_size, hidden_size)
                for u in range(units)
            ]
        )

        # hidden → hidden (standard Linear, size known at init)
        self.hh = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(units)]
        )

        self.drops = nn.ModuleList(
            [nn.Dropout(p=0.2) for _ in range(units - 1)]
        )

        # Project hidden state back to flat pixel space, then reshape
        # LazyLinear infers hidden_size on first call
        self.output_proj = nn.Sequential(
            nn.LazyLinear(1),   # placeholder; replaced after first forward
        )
        # We rebuild output_proj properly after we know C*H*W (see _init_output_proj)
        self._output_proj_ready = False

    # ------------------------------------------------------------------
    def _init_output_proj(self, frame_size: int) -> None:
        """Called once on the first forward pass when H and W are known."""
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, frame_size),
            nn.Sigmoid(),
        ).to(next(self.hh[0].parameters()).device)
        self._output_proj_ready = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        x  : (seq_len, batch, C_in, H, W)
        h0 : (units, batch, hidden_size) or None

        Returns
        -------
        projected : (seq_len, batch, C_in, H, W)
        h_out     : (units, batch, hidden_size)
        """
        seq_len, batch, C, H, W = x.shape
        frame_size = C * H * W

        if not self._output_proj_ready:
            self._init_output_proj(frame_size)

        if h0 is None:
            h = [
                torch.zeros(batch, self.hidden_size, device=x.device)
                for _ in range(self.units)
            ]
        else:
            h = [s.clone() for s in h0.unbind(0)]

        outputs = []

        for t in range(seq_len):
            x_flat = x[t].reshape(batch, -1)   # (batch, C*H*W)

            for u in range(self.units):
                inp = x_flat if u == 0 else h[u - 1]
                h[u] = torch.tanh(self.ih[u](inp) + self.hh[u](h[u]))
                if u < self.units - 1:
                    h[u] = self.drops[u](h[u])

            outputs.append(h[-1].clone())       # (batch, hidden_size)

        h_out = torch.stack(h)                  # (units, batch, hidden_size)

        # project each hidden state back to (batch, C, H, W)
        projected = torch.stack(
            [self.output_proj(o).reshape(batch, C, H, W) for o in outputs]
        )                                       # (seq_len, batch, C, H, W)

        return projected, h_out

    # ------------------------------------------------------------------
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
        h          : (units, batch, hidden_size)
        targets    : (pred_len, batch, C_in, H, W)
        """
        predictions = []
        current = last_frame.unsqueeze(0)       # (1, batch, C_in, H, W)

        for t in range(pred_len):
            out, h = self.forward(current, h)
            pred = out[0]                       # (batch, C_in, H, W)
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current = targets[t].unsqueeze(0)
            else:
                current = pred.detach().unsqueeze(0)

        return torch.stack(predictions)         # (pred_len, batch, C_in, H, W)


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
