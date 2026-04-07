"""
Future temporal prediction entrypoint for hydrograph/CNN models.

This will eventually load a trained temporal model and evaluate rolling
hydrograph windows for early failure warning.
"""

from __future__ import annotations


def predict_failure_timeline(*_args, **_kwargs):
    """Placeholder for future CNN-based temporal prediction."""
    raise NotImplementedError(
        "Temporal CNN prediction is not implemented yet. "
        "We first need a trained CNN model and a validated rolling-window dataset."
    )
