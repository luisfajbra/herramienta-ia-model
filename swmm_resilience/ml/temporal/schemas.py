"""
Shared schemas for the planned temporal/CNN workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RESULTS_DIR,
    ML_TEMPORAL_HORIZON_MIN,
    ML_TEMPORAL_STEP_MIN,
    ML_TEMPORAL_TARGET,
    ML_TEMPORAL_WINDOW_MIN,
)


@dataclass(frozen=True)
class TemporalWindowSpec:
    """Configuration for future rolling temporal windows."""

    window_min: int = ML_TEMPORAL_WINDOW_MIN
    horizon_min: int = ML_TEMPORAL_HORIZON_MIN
    step_min: int = ML_TEMPORAL_STEP_MIN
    target: str = ML_TEMPORAL_TARGET


@dataclass(frozen=True)
class TemporalDatasetSpec:
    """Paths and metadata for future temporal datasets."""

    source_csv: Path = DEFAULT_OUTPUT_CSV
    output_dir: Path = DEFAULT_RESULTS_DIR / "temporal"
    dataset_name: str = "temporal_windows"

    @property
    def output_csv(self) -> Path:
        return self.output_dir / f"{self.dataset_name}.csv"
