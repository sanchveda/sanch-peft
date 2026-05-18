"""Configuration helpers for experiment-driven training runs.

Two pieces live here:
    - load_config(path)   reads a YAML file into a plain dict
    - TrainConfig         typed schema for the training-loop hyperparameters

Entry scripts read the YAML, then construct a TrainConfig from the relevant
fields. TrainConfig is intentionally separate from the full YAML dict because
the YAML may carry experiment-level fields (model_name, dataset_name,
target_modules, lora_rank, ...) that the training loop itself does not need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load an experiment configuration from a YAML file."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    _validate_config(config, config_path)
    return config


def _validate_config(config: dict, config_path: Path) -> None:
    """Placeholder validation hook for future schema checks."""
    _ = config
    _ = config_path

    # Future work:
    # - require a minimal set of experiment keys
    # - validate PEFT-specific fields per method
    # - add clearer error messages for malformed configs


@dataclass
class TrainConfig:
    """Typed hyperparameters consumed by core.training.train.train()."""

    # iters / cadence
    num_steps: int = 200
    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    # batching
    batch_size: int = 8
    block_size: int = 128
    gradient_accumulation_steps: int = 1
    # optimizer / schedule
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 20
    grad_clip: float = 1.0
    # system
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    dtype: str = "auto"  # "auto" | "float32" | "bfloat16" | "float16"
