"""
LoRA: Low-Rank Adaptation of frozen linear projections.

For a pretrained linear projection W, LoRA freezes W and learns a low-rank
update BA so that the adapted layer computes h = Wx + (alpha/r) * BAx.
"""

from __future__ import annotations

import math

import torch


class LoRALayer(torch.nn.Module):
    """Low-rank adapter path: returns (alpha/r) * BAx, no base contribution."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std_dev = 1.0 / math.sqrt(rank)
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LoRALinear(torch.nn.Module):
    """Linear layer with frozen base weights and an additive LoRA update."""

    def __init__(
        self,
        base_linear: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.lora_dropout = (
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        )
        self.lora = LoRALayer(
            in_dim=self.base_linear.in_features,
            out_dim=self.base_linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):
        return self.base_linear(x) + self.lora(self.lora_dropout(x))


def apply_lora_to_model(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """
    Placeholder hook for attaching LoRA modules to a model.

    A later implementation can:
    - identify target modules from `config["target_modules"]`
    - replace supported `nn.Linear` layers with `LoRALinear`
    - freeze the correct base parameters
    - return the adapted model ready for training
    """
    _ = config
    return model
