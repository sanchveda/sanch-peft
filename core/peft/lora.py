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


def apply_lora_to_model(
    model: torch.nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> torch.nn.Module:
    """Replace matching nn.Linear layers with LoRALinear; freeze the rest. In-place.

    target_modules is a list of dotted-name suffixes. A module matches if its
    full name in named_modules() ends with any string in the list. e.g.
    target_modules=['c_attn'] matches 'transformer.h.0.attn.c_attn'.
    target_modules=['attn.c_proj'] matches attention output projections but
    excludes 'mlp.c_proj'.

    Raises ValueError if no targets match — a silent no-op would produce a
    confusing "no trainable params" crash later in train().
    """
    for param in model.parameters():
        param.requires_grad = False

    swaps = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(name.endswith(target) for target in target_modules)
    ]
    if not swaps:
        raise ValueError(
            f"No torch.nn.Linear modules matched target_modules={target_modules}. "
            "Verify the suffixes match names in model.named_modules()."
        )
    for name, base_linear in swaps:
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr_name,
            LoRALinear(base_linear, rank=rank, alpha=alpha, dropout=dropout),
        )
    return model


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count). For LoRA, trainable << total."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return trainable, total
