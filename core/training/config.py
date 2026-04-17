"""Configuration helpers for experiment-driven training runs."""

from __future__ import annotations

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
