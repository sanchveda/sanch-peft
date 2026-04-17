"""Metric helper scaffolds for experiment reporting."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Placeholder helper for total parameter counting."""
    _ = model
    return 0


def count_trainable_parameters(model: nn.Module) -> int:
    """Placeholder helper for trainable parameter counting."""
    _ = model
    return 0
