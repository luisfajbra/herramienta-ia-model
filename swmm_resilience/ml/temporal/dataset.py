"""
Future temporal dataset construction for hydrograph/CNN experiments.

This module is intentionally a scaffold. The current database does not yet
persist node-level time series, so a real rolling-window dataset cannot be
built safely until that collection step is added.
"""

from __future__ import annotations

from .schemas import TemporalDatasetSpec, TemporalWindowSpec


REQUIRED_TIMESERIES_COLUMNS = [
    "run_id",
    "node_id",
    "step_index",
    "time_sec",
    "total_inflow_lps",
    "lateral_inflow_lps",
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
    "head_m",
    "failed_now",
]


def expected_timeseries_columns() -> list[str]:
    """Return the columns the future temporal dataset builder will expect."""
    return REQUIRED_TIMESERIES_COLUMNS.copy()


def build_temporal_windows(
    dataset_spec: TemporalDatasetSpec | None = None,
    window_spec: TemporalWindowSpec | None = None,
):
    """Placeholder for the future rolling-window dataset builder."""
    dataset_spec = dataset_spec or TemporalDatasetSpec()
    window_spec = window_spec or TemporalWindowSpec()
    raise NotImplementedError(
        "Temporal window generation is not implemented yet. "
        "Next step: persist node-level time series during hydrograph simulations, "
        f"then build {window_spec.window_min}-minute windows into {dataset_spec.output_csv}."
    )
