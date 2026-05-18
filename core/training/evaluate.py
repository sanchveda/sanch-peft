"""Public evaluation entrypoint: averaged loss over many batches.

Two callers:
    - core.training.train  uses this during training (it passes its own
      autocast ctx and device that were set up for the training loop)
    - experiments/lora/scripts/validate.py  uses this standalone after a
      training run (no training-loop state; ctx/device are inferred from
      the model)

Returns a flat dict like {"val_loss": float} or {"train_loss": ..., "val_loss": ...}
depending on which splits are requested.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

from core.data.base import CausalLMDataset
from core.training.config import TrainConfig


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: CausalLMDataset,
    cfg: TrainConfig,
    *,
    splits: Iterable[str] = ("val",),
    ctx=None,
    device: str | None = None,
) -> dict[str, float]:
    """Average per-token loss over cfg.eval_iters random batches per split."""
    if device is None:
        device = next(model.parameters()).device.type
    if ctx is None:
        ctx = nullcontext()

    was_training = model.training
    model.eval()

    out: dict[str, float] = {}
    for split in splits:
        losses = torch.zeros(cfg.eval_iters)
        for iteration in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, cfg.block_size)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, targets=y)
            losses[iteration] = loss.item()
        out[f"{split}_loss"] = losses.mean().item()

    if was_training:
        model.train()
    return out
