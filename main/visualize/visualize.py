"""
Generates PNG images comparing input frames, ground-truth future frames,
model predictions and absolute-error maps for a pretrained model.

Output layout (one figure per sample)
--------------------------------------
Row 0 – Input        : context frames fed to the encoder
Row 1 – Ground truth : target future frames
Row 2 – Prediction   : frames produced by the model
Row 3 – |Error|      : absolute pixel-wise difference (pred vs gt)

A summary grid is also saved with one column per sample.

Usage (via __main__.py)
-----------------------
    python -m main --visualize

Config keys (``config.json → "visualize"``)
-------------------------------------------
    model_type      : "rcnn2d" | "cgru" | "cnn"
    model_path      : path to the .pth checkpoint
    batch_size      : samples per DataLoader batch        (default 32)
    n_workers       : DataLoader worker processes         (default 0)
    split_seq_len   : frames used as encoder input        (default 10)
    device          : "cpu" | "cuda"                      (default "cpu")
    hidden_channels : model hyper-param                   (default 64)
    kernel_size     : model hyper-param                   (default 3)
    units           : model hyper-param                   (default 3)
    activation      : "relu" | "tanh"  (RCNN2d only)      (default "relu")
    n_samples       : sequences to visualise              (default 5)
    output_dir      : directory for saved PNGs            (default "saves/visualizations")
    colormap        : matplotlib cmap for frames          (default "gray")
    dpi             : figure resolution                   (default 150)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

try:
    from data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from models import RCNN2d, Conv2dGRU, CNN, transpose_data
    from train import get_prediction
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from ..models import RCNN2d, Conv2dGRU, CNN, transpose_data
    from ..train import get_prediction
    from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── model registry (mirrors evaluate.py) ─────────────────────────────────────
_MODEL_REGISTRY: dict[str, tuple[type, str]] = {
    "rcnn2d": (RCNN2d, "rcnn2d"),
    "cgru":   (Conv2dGRU, "cgru"),
    "cnn":    (CNN,  "cnn"),
}

# ── row labels shown on each figure ──────────────────────────────────────────
_ROW_LABELS = ["Input", "Ground truth", "Prediction", "|Error|"]



# Public entry-point


def visualize_inference(config: dict) -> None:
    """
    Load a pretrained model, run inference on *n_samples* test sequences
    and save comparison figures to *output_dir*.

    Parameters
    ----------
    config : dict
        The ``"visualize"`` section of ``config.json``.
    """
    if not config:
        logger.warning('No "visualize" section found in config.json.')
        return

    # ── resolve config values ────────────────────────────────────────────────
    model_type  = config.get("model_type", "cnn").lower()
    model_path  = config["model_path"]
    device      = config.get("device", "cpu")
    split       = config.get("split_seq_len", 10)
    n_samples   = config.get("n_samples", 5)
    output_dir  = Path(config.get("output_dir", "saves/visualizations"))
    colormap    = config.get("colormap", "gray")
    dpi         = config.get("dpi", 150)

    if model_type not in _MODEL_REGISTRY:
        logger.error(
            'Unknown model_type "%s". Valid options: %s.',
            model_type, list(_MODEL_REGISTRY.keys()),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Saving visualizations to "%s"', output_dir)

    # ── data ─────────────────────────────────────────────────────────────────
    test_loader = _load_test(config)
    first_batch = next(iter(test_loader))

    # ── model ─────────────────────────────────────────────────────────────────
    model_cls, model_name = _MODEL_REGISTRY[model_type]
    model, _ = _load_model(first_batch, config, model_cls, model_path)
    model.to(device)
    model.eval()
    logger.info('Loaded model "%s" from "%s"', model_name, model_path)

    # ── collect samples ───────────────────────────────────────────────────────
    samples_collected = 0
    all_inputs:  list[torch.Tensor] = []
    all_labels:  list[torch.Tensor] = []
    all_preds:   list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            if samples_collected >= n_samples:
                break

            inputs, labels = transpose_data(batch, split)
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = get_prediction(model, inputs, labels, teacher_forcing_ratio=0.0)

            # take as many samples as we still need from this batch
            remaining = n_samples - samples_collected
            take = min(remaining, inputs.shape[1])   # shape[1] == batch size

            # move to cpu for plotting (seq_len, take, C, H, W)
            all_inputs.append(inputs[:, :take].cpu())
            all_labels.append(labels[:, :take].cpu())
            all_preds.append(preds[:, :take].cpu())
            samples_collected += take

    # concatenate along the batch dimension
    inputs_all = torch.cat(all_inputs, dim=1)   # (in_len,  N, C, H, W)
    labels_all = torch.cat(all_labels, dim=1)   # (out_len, N, C, H, W)
    preds_all  = torch.cat(all_preds,  dim=1)   # (out_len, N, C, H, W)

    logger.info(
        "Generating figures for %d sample(s)  |  "
        "input_len=%d  pred_len=%d",
        samples_collected, inputs_all.shape[0], labels_all.shape[0],
    )

    # ── per-sample figures ────────────────────────────────────────────────────
    saved_paths: list[Path] = []
    for idx in range(samples_collected):
        inp  = inputs_all[:, idx, 0]   # (in_len,  H, W)
        lbl  = labels_all[:, idx, 0]   # (out_len, H, W)
        pred = preds_all[:,  idx, 0]   # (out_len, H, W)

        path = output_dir / f"sample_{idx:03d}.png"
        _save_sample_figure(inp, lbl, pred, path, colormap=colormap, dpi=dpi)
        saved_paths.append(path)
        logger.info("  Saved %s", path)

    # ── summary grid ─────────────────────────────────────────────────────────
    summary_path = output_dir / "summary.png"
    _save_summary_figure(
        inputs_all, labels_all, preds_all,
        summary_path, colormap=colormap, dpi=dpi,
    )
    logger.info("Summary figure saved to %s", summary_path)
    saved_paths.append(summary_path)

    logger.info("Done. %d file(s) written to %s", len(saved_paths), output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_sample_figure(
    inp:  torch.Tensor,   # (in_len,  H, W)  values in [0, 1]
    lbl:  torch.Tensor,   # (out_len, H, W)
    pred: torch.Tensor,   # (out_len, H, W)
    path: Path,
    *,
    colormap: str = "gray",
    dpi: int = 150,
) -> None:
    """
    Save a 4-row figure for a single sequence:
        Row 0 – Input frames
        Row 1 – Ground-truth frames
        Row 2 – Predicted frames
        Row 3 – Absolute error  |pred - gt|

    Input and prediction lengths can differ, so each row is sized
    independently.  The error row spans only the prediction window.
    """
    in_len  = inp.shape[0]
    out_len = lbl.shape[0]
    error   = (pred - lbl).abs()   # (out_len, H, W)

    # number of columns = max(in_len, out_len)
    n_cols = max(in_len, out_len)

    fig_w = max(n_cols * 1.3, 6.0)
    fig_h = 4 * 1.5 + 0.6          # 4 rows + title space
    fig, axes = plt.subplots(4, n_cols, figsize=(fig_w, fig_h))

    # ensure axes is always 2-D
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    vmax_err = float(error.max().item()) or 1.0

    for row, (frames, length, label, vmin, vmax) in enumerate([
        (inp,   in_len,  "Input",        0.0, 1.0),
        (lbl,   out_len, "Ground truth", 0.0, 1.0),
        (pred,  out_len, "Prediction",   0.0, 1.0),
        (error, out_len, "|Error|",       0.0, vmax_err),
    ]):
        cmap = colormap if row < 3 else "hot"

        for col in range(n_cols):
            ax = axes[row, col]
            if col < length:
                frame = frames[col].numpy()
                im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest")
                ax.set_title(f"t={col}", fontsize=6, pad=2)
            else:
                ax.axis("off")          # blank cell if this row is shorter
            ax.set_xticks([])
            ax.set_yticks([])

        # row label on the leftmost axis
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90,
                                labelpad=4, va="center")

    fig.suptitle(
        f"Inference  |  input={in_len} frames  →  pred={out_len} frames",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_summary_figure(
    inputs_all: torch.Tensor,   # (in_len,  N, C, H, W)
    labels_all: torch.Tensor,   # (out_len, N, C, H, W)
    preds_all:  torch.Tensor,   # (out_len, N, C, H, W)
    path: Path,
    *,
    colormap: str = "gray",
    dpi: int = 120,
) -> None:
    """
    Save a compact grid: one column per sample, showing the first input
    frame, the last ground-truth frame and the last predicted frame.
    """
    n_samples = inputs_all.shape[1]
    row_labels = ["First input", "Last GT", "Last pred", "|Error|"]
    n_rows = len(row_labels)

    fig_w = max(n_samples * 1.4, 5.0)
    fig_h = n_rows * 1.4 + 0.5
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(fig_w, fig_h))

    if n_samples == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_samples):
        first_inp   = inputs_all[0,  col, 0].numpy()   # first input frame
        last_lbl    = labels_all[-1, col, 0].numpy()   # last GT frame
        last_pred   = preds_all[-1,  col, 0].numpy()   # last predicted frame
        last_error  = np.abs(last_pred - last_lbl)

        frames_  = [first_inp, last_lbl, last_pred, last_error]
        cmaps_   = [colormap, colormap, colormap, "hot"]
        vmaxes_  = [1.0, 1.0, 1.0, float(last_error.max()) or 1.0]

        for row, (frame, cmap_, vmax_) in enumerate(zip(frames_, cmaps_, vmaxes_)):
            ax = axes[row, col]
            ax.imshow(frame, cmap=cmap_, vmin=0.0, vmax=vmax_,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=7, rotation=90,
                              labelpad=3, va="center")
            if row == 0:
                ax.set_title(f"#{col}", fontsize=7, pad=2)

    fig.suptitle("Inference summary", fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (mirrors evaluate.py structure)
# ─────────────────────────────────────────────────────────────────────────────

def _load_test(config: dict) -> torch.utils.data.DataLoader:
    paths = download_moving_mnist()
    test_ds = MovingMNISTDataset(paths[2])
    return make_dataloader(
        test_ds,
        batch_size=config.get("batch_size", 32),
        n_workers=config.get("n_workers", 0),
    )


def _load_model(
    batch: torch.Tensor,
    config: dict,
    model_cls: type,
    model_path: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate *model_cls* and load checkpoint weights."""
    kwargs: dict = dict(
        input_channels=batch.shape[2],
        hidden_channels=config.get("hidden_channels", 64),
        kernel_size=config.get("kernel_size", 3),
        units=config.get("units", 3),
    )
    if model_cls is RCNN2d:
        kwargs["activation"] = config.get("activation", "relu")

    model = model_cls(**kwargs)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = checkpoint["loss"]
    return model, loss_fn
