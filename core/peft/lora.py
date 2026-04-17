"""
LoRA scaffolding for source-code-first parameter-efficient fine-tuning.

LoRA, or Low-Rank Adaptation, freezes a pretrained weight matrix and learns a
small low-rank update instead of updating the full parameter tensor. For a
linear projection W, LoRA approximates the trainable delta as B @ A where A and
B have a much smaller rank than the original layer dimensions.

This module is intentionally a structural starter rather than a full
implementation. It captures the interfaces and the main conceptual hooks
required for a from-scratch LoRA implementation without committing the
repository to a production-ready adapter injection path yet.
"""

from __future__ import annotations

from torch import nn


class LoRALinear(nn.Module):
    """Structural placeholder for a LoRA-wrapped linear layer."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Base weights would be frozen here so the pretrained layer stays fixed.
        # Example:
        # self.base_linear.weight.requires_grad = False
        # if self.base_linear.bias is not None:
        #     self.base_linear.bias.requires_grad = False

        # Low-rank matrices would be initialized here.
        # Example conceptual shapes:
        # A: [rank, in_features]
        # B: [out_features, rank]
        self.lora_A = None
        self.lora_B = None

        # Scaling would typically be defined as alpha / rank.
        self.scaling = None

    def forward(self, x):
        """
        Placeholder forward pass.

        A future implementation should:
        - compute the frozen base projection
        - compute the LoRA low-rank update
        - combine base output and scaled LoRA update
        """
        raise NotImplementedError("LoRALinear is a structural scaffold for now.")


def apply_lora_to_model(model: nn.Module, config: dict) -> nn.Module:
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
