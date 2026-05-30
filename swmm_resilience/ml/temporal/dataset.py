"""
Temporal dataset helpers for hydrograph/CNN experiments.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import DEFAULT_DB_FILE, NETWORKS_DIR
from .schemas import TemporalDatasetSpec, TemporalWindowDataset, TemporalWindowSpec


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

TEMPORAL_COLS = [
    "total_inflow_lps",
    "lateral_inflow_lps",
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
]

STATIC_COLS = [
    "full_depth_m",
    "in_degree",
    "out_degree",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "upstream_capacity_lps",
    "downstream_capacity_lps",
]


def expected_timeseries_columns() -> list[str]:
    """Return the columns the temporal dataset builder expects."""
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
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    window_spec: TemporalWindowSpec | None = None,
    dataset_spec: TemporalDatasetSpec | None = None,  # reserved for future filtering
) -> TemporalWindowDataset:
    """Build sliding temporal windows from all Parquets in temporal_artifacts.

    Reads each registered Parquet, resamples to resample_min-minute intervals
    via forward-fill, and produces a sliding window dataset. Static features
    are joined from network_nodes using (network_hash, node_id == node_uid).
    No normalisation is applied — that is the caller's responsibility.
    """
    window_spec = window_spec or TemporalWindowSpec()

    resample_min = window_spec.resample_min
    window_steps = window_spec.window_min // resample_min
    horizon_steps = window_spec.horizon_min // resample_min
    step_steps = window_spec.step_min // resample_min

    if step_steps == 0:
        raise ValueError(
            f"step_min ({window_spec.step_min}) must be >= resample_min ({resample_min})"
        )

    all_X_seq: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    conn = sqlite3.connect(db_path)
    try:
        artifacts = conn.execute(
            "SELECT run_id, network_hash, parquet_path "
            "FROM temporal_artifacts ORDER BY created_at"
        ).fetchall()

        for run_id, network_hash, parquet_path in artifacts:
            df = pd.read_parquet(parquet_path)

            # Load static lookup: node_uid → float32 vector [7]
            static_rows = conn.execute(
                f"""SELECT node_uid, {', '.join(STATIC_COLS)}
                    FROM network_nodes
                    WHERE network_hash = ?""",
                (network_hash,),
            ).fetchall()
            static_lookup: dict[str, np.ndarray] = {
                row[0]: np.array(row[1:], dtype=np.float32)
                for row in static_rows
            }

            for node_id in df["node_id"].unique():
                if node_id not in static_lookup:
                    warnings.warn(
                        f"node_id '{node_id}' (run_id={run_id}) has no matching row in network_nodes "
                        f"for network_hash={network_hash!r}. Skipping.",
                        stacklevel=2,
                    )
                    continue
                x_static = static_lookup[node_id]

                node_df = (
                    df[df["node_id"] == node_id]
                    .sort_values("time_min")
                    .reset_index(drop=True)
                )
                if node_df.empty:
                    continue

                # Resample to regular resample_min-minute grid via forward-fill
                t_start = node_df["time_min"].iloc[0]
                t_end = node_df["time_min"].iloc[-1]
                n_grid = int(round((t_end - t_start) / resample_min)) + 1
                grid = t_start + np.arange(n_grid, dtype=float) * resample_min
                node_df = (
                    node_df.set_index("time_min")
                    .reindex(grid)
                    .ffill()
                    .dropna(subset=TEMPORAL_COLS)
                    .reset_index()
                )

                n = len(node_df)
                i = 0
                while i + window_steps + horizon_steps <= n:
                    window = node_df.iloc[i : i + window_steps]
                    horizon = node_df.iloc[i + window_steps : i + window_steps + horizon_steps]

                    all_X_seq.append(window[TEMPORAL_COLS].values.astype(np.float32))
                    all_X_static.append(x_static)
                    all_y_class.append(int((horizon["flooding_lps"] > 0).any()))
                    all_y_reg.append(float(horizon["flooding_lps"].max()))
                    all_groups.append(run_id)
                    meta_rows.append({
                        "run_id": run_id,
                        "node_id": node_id,
                        "window_start_min": float(node_df["time_min"].iloc[i]),
                    })
                    i += step_steps
    finally:
        conn.close()

    if not all_X_seq:
        return TemporalWindowDataset(
            X_seq=np.empty((0, window_steps, len(TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(STATIC_COLS)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
        )

    return TemporalWindowDataset(
        X_seq=np.stack(all_X_seq),
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=pd.DataFrame(meta_rows),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnosticar dataset temporal de ventanas.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_FILE)
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen del dataset.")
    args = parser.parse_args()

    if args.summary:
        _conn = sqlite3.connect(args.db)
        n_parquets = _conn.execute("SELECT COUNT(*) FROM temporal_artifacts").fetchone()[0]
        _conn.close()
        print(f"Parquets registrados: {n_parquets}")

        print("Construyendo ventanas...")
        _ds = build_temporal_windows(db_path=args.db)
        _n = _ds.X_seq.shape[0]
        print(f"Total de muestras: {_n}")
        if _n > 0:
            _pos = int(_ds.y_class.sum())
            _neg = _n - _pos
            print(f"  failure_within_horizon=1: {_pos} ({100 * _pos / _n:.1f}%)")
            print(f"  failure_within_horizon=0: {_neg} ({100 * _neg / _n:.1f}%)")
        else:
            print("No se generaron muestras.")
    else:
        parser.print_help()
