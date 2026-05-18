"""Abstract base for causal-LM datasets.

A causal-LM dataset is fully described by per-split 1-D token streams.
Subclasses implement `tokens(split)`; the base class provides `get_batch`,
which samples random windows for next-token prediction:

    inputs  = tokens[i     : i + T]
    targets = tokens[i + 1 : i + T + 1]

Same windowing scheme nanoGPT uses. To add a new dataset, subclass this
and implement `tokens` — see core/data/shakespeare.py for an example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CausalLMDataset(ABC):
    @abstractmethod
    def tokens(self, split: str) -> torch.Tensor:
        """Return a 1-D torch.long tensor of token ids for the given split.

        Split is conventionally 'train' or 'val'. Subclasses may raise
        KeyError for unknown splits.
        """

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random-window sample one batch of (input, target) pairs."""
        data = self.tokens(split)
        if len(data) <= block_size:
            raise ValueError(
                f"Split '{split}' has {len(data)} tokens but block_size="
                f"{block_size}; need len(data) > block_size."
            )
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y
