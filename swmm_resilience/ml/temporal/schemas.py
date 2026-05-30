"""
Shared schemas for the planned temporal/CNN workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RESULTS_DIR,
    ML_TEMPORAL_HORIZON_MIN,
    ML_TEMPORAL_RESAMPLE_MIN,
    ML_TEMPORAL_STEP_MIN,
    ML_TEMPORAL_TARGET,
    ML_TEMPORAL_WINDOW_MIN,
)


@dataclass(frozen=True)
class TemporalWindowSpec:
    """Configuration for rolling temporal windows."""

    window_min: int = ML_TEMPORAL_WINDOW_MIN
    horizon_min: int = ML_TEMPORAL_HORIZON_MIN
    step_min: int = ML_TEMPORAL_STEP_MIN
    resample_min: int = ML_TEMPORAL_RESAMPLE_MIN
    target: str = ML_TEMPORAL_TARGET


@dataclass
class TemporalWindowDataset:
    """Output of build_temporal_windows().

    Arrays are raw (unscaled). Normalisation is the caller's responsibility.
    """

    X_seq: np.ndarray      # [N, timesteps, temporal_features]  float32
    X_static: np.ndarray   # [N, static_features]               float32
    y_class: np.ndarray    # [N]  int8  — failure_within_horizon (0 or 1)
    y_reg: np.ndarray      # [N]  float32 — peak_flooding_lps in horizon
    groups: np.ndarray     # [N]  object  — run_id string, for GroupKFold
    meta: pd.DataFrame     # columns: run_id, node_id, window_start_min


@dataclass(frozen=True)
class TemporalDatasetSpec:
    """Paths and metadata for temporal datasets."""

    source_csv: Path = DEFAULT_OUTPUT_CSV
    output_dir: Path = DEFAULT_RESULTS_DIR / "temporal"
    dataset_name: str = "temporal_windows"

    @property
    def output_csv(self) -> Path:
        return self.output_dir / f"{self.dataset_name}.csv"
