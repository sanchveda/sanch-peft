"""Minimal evaluation utilities for PEFT experiments."""

from __future__ import annotations


def evaluate_model(model, dataloader=None, metrics=None) -> dict:
    """
    Placeholder evaluation entrypoint.

    Args:
        model: The model to evaluate.
        dataloader: Optional evaluation dataloader.
        metrics: Optional metric collection or callable mapping.

    Returns:
        dict: A small placeholder metrics dictionary.
    """
    # This is a structural placeholder only.
    # Later iterations can add dataset-specific metrics,
    # generation-based evaluation, and result serialization.
    _ = model
    _ = dataloader
    _ = metrics

    return {
        "status": "evaluation_placeholder",
        "message": "Evaluation logic has not been implemented yet.",
    }
