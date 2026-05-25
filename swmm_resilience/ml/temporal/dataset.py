"""
Temporal dataset helpers for hydrograph/CNN experiments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import TemporalDatasetSpec, TemporalWindowSpec


REQUIRED_TIMESERIES_COLUMNS = [
    "run_id",
    "network_hash",
    "node_id",
    "step_index",
    "time_sec",
    "time_min",
    "total_inflow_lps",
    "lateral_inflow_lps",
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
    "failed_now",
]


def expected_timeseries_columns() -> list[str]:
    """Return the columns the future temporal dataset builder will expect."""
    return REQUIRED_TIMESERIES_COLUMNS.copy()


def save_node_timeseries_parquet(records: list[dict], output_path: str | Path) -> Path:
    """Persist node-level timestep records to Parquet with a stable column order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame.from_records(records, columns=REQUIRED_TIMESERIES_COLUMNS)
    missing_columns = [
        column for column in REQUIRED_TIMESERIES_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias en node_timeseries: "
            + ", ".join(missing_columns)
        )

    try:
        dataframe.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "No se pudo guardar node_timeseries en Parquet. Instala 'pyarrow' "
            "o 'fastparquet' para habilitar este formato."
        ) from exc

    return output_path


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
