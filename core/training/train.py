"""Training entrypoint scaffold for PEFT experiments."""

from __future__ import annotations

import argparse

from core.training.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Run a PEFT training experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration for the experiment.",
    )
    return parser.parse_args()


def run_experiment(config: dict) -> None:
    """Structural dispatcher for future baseline, LoRA, and QLoRA runs."""
    experiment_name = config["experiment_name"]
    peft_method = config["peft_method"]

    print(f"Running experiment: {experiment_name}")
    print(f"PEFT method: {peft_method}")
    print(f"Model: {config.get('model_name', 'unset')}")
    print(f"Dataset: {config.get('dataset_name', 'unset')}")

    # Future work:
    # - wire baseline full fine-tuning here
    # - wire LoRA attachment and training here
    # - wire QLoRA preparation and training here


def main() -> None:
    """Load config and route to the structural experiment entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
