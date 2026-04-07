"""
Future CNN training entrypoint.

This file is intentionally lightweight and does not import PyTorch or
TensorFlow yet. We will add a real implementation after defining and validating
the temporal dataset.

Usage later:
    python -m swmm_resilience.ml.temporal.train_cnn
"""

from __future__ import annotations

from .schemas import TemporalDatasetSpec, TemporalWindowSpec


def describe_planned_training(
    dataset_spec: TemporalDatasetSpec | None = None,
    window_spec: TemporalWindowSpec | None = None,
) -> str:
    """Return a short description of the future CNN training setup."""
    dataset_spec = dataset_spec or TemporalDatasetSpec()
    window_spec = window_spec or TemporalWindowSpec()
    return (
        "Temporal CNN training scaffold\n"
        f"- source dataset: {dataset_spec.source_csv}\n"
        f"- temporal output: {dataset_spec.output_csv}\n"
        f"- window: {window_spec.window_min} min\n"
        f"- horizon: {window_spec.horizon_min} min\n"
        f"- step: {window_spec.step_min} min\n"
        f"- target: {window_spec.target}\n"
        "- status: pending time-series persistence and CNN implementation"
    )


def main():
    print(describe_planned_training())


if __name__ == "__main__":
    main()
