"""
Temporal dataset helpers for hydrograph/CNN experiments.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import DEFAULT_DB_FILE, DEFAULT_OUTPUT_CSV, NETWORKS_DIR
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

PRE_SWMM_TEMPORAL_COLS = [
    "total_inflow_lps",
    "lateral_inflow_lps",
]

SWMM_OUTPUT_TEMPORAL_COLS = [
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
]

TEMPORAL_COLS = PRE_SWMM_TEMPORAL_COLS + SWMM_OUTPUT_TEMPORAL_COLS

SURROGATE_TEMPORAL_COLS = PRE_SWMM_TEMPORAL_COLS

STATIC_COLS = [
    "full_depth_m",
    "in_degree",
    "out_degree",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "upstream_capacity_lps",
    "downstream_capacity_lps",
]


def _with_dataset_attrs(
    meta: pd.DataFrame,
    *,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
) -> pd.DataFrame:
    meta.attrs["temporal_feature_names"] = list(temporal_feature_names)
    meta.attrs["static_feature_names"] = list(static_feature_names)
    return meta


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
                row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
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
                    .drop_duplicates(subset=["time_min"], keep="last")
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
            meta=_with_dataset_attrs(
                pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
                temporal_feature_names=TEMPORAL_COLS,
                static_feature_names=STATIC_COLS,
            ),
        )

    meta = _with_dataset_attrs(
        pd.DataFrame(meta_rows),
        temporal_feature_names=TEMPORAL_COLS,
        static_feature_names=STATIC_COLS,
    )

    return TemporalWindowDataset(
        X_seq=np.stack(all_X_seq),
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=meta,
    )


def build_surrogate_dataset(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    resample_min: int = 5,
    use_temporal: bool = True,
) -> TemporalWindowDataset:
    """Build one sample per (run_id, node_id) for the surrogate CNN.

    Unlike build_temporal_windows, no sliding windows are produced.
    Each sample's X_seq is the full resampled inflow/hydraulic timeseries for
    that node in that run. Sequences are zero-padded to the longest T found.

    When use_temporal=False the inflow_multiplier for the run is appended as
    an extra column to X_static (giving 8 columns instead of 7), so the
    no-temporal ablation model receives the multiplier as a static feature.
    """
    all_X_seq_list: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    conn = sqlite3.connect(db_path)
    try:
        artifacts = conn.execute(
            "SELECT ta.run_id, ta.network_hash, ta.parquet_path, r.inflow_multiplier "
            "FROM temporal_artifacts ta "
            "JOIN runs r ON ta.run_id = r.run_id "
            "ORDER BY r.inflow_multiplier"
        ).fetchall()

        for run_id, network_hash, parquet_path, inflow_multiplier in artifacts:
            df = pd.read_parquet(parquet_path)

            static_rows = conn.execute(
                f"SELECT node_uid, {', '.join(STATIC_COLS)} "
                "FROM network_nodes WHERE network_hash = ?",
                (network_hash,),
            ).fetchall()
            static_lookup: dict[str, np.ndarray] = {
                row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
                for row in static_rows
            }

            for node_id in df["node_id"].unique():
                if node_id not in static_lookup:
                    warnings.warn(
                        f"node_id '{node_id}' (run_id={run_id}) not in network_nodes — skipping.",
                        stacklevel=2,
                    )
                    continue

                node_df = (
                    df[df["node_id"] == node_id]
                    .sort_values("time_min")
                    .drop_duplicates(subset=["time_min"], keep="last")
                    .reset_index(drop=True)
                )
                if node_df.empty:
                    continue

                # Resample to regular grid
                t_start = node_df["time_min"].iloc[0]
                t_end = node_df["time_min"].iloc[-1]
                n_grid = int(round((t_end - t_start) / resample_min)) + 1
                grid = t_start + np.arange(n_grid, dtype=float) * resample_min
                node_df = (
                    node_df.set_index("time_min")
                    .reindex(grid)
                    .ffill()
                    .dropna(subset=SURROGATE_TEMPORAL_COLS)
                    .reset_index()
                )
                if node_df.empty:
                    continue

                seq = node_df[SURROGATE_TEMPORAL_COLS].values.astype(np.float32)  # [T, 2]
                x_static = static_lookup[node_id]

                if not use_temporal and inflow_multiplier is not None:
                    x_static = np.append(x_static, float(inflow_multiplier)).astype(np.float32)

                all_X_seq_list.append(seq)
                all_X_static.append(x_static)
                all_y_class.append(int((node_df["flooding_lps"] > 0).any()))
                all_y_reg.append(float(node_df["flooding_lps"].max()))
                all_groups.append(run_id)
                meta_rows.append({"run_id": run_id, "node_id": node_id, "window_start_min": 0.0})
    finally:
        conn.close()

    if not all_X_seq_list:
        static_feature_names = STATIC_COLS + ([] if use_temporal else ["inflow_multiplier"])
        return TemporalWindowDataset(
            X_seq=np.empty((0, 1, len(SURROGATE_TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(static_feature_names)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=_with_dataset_attrs(
                pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
                temporal_feature_names=SURROGATE_TEMPORAL_COLS,
                static_feature_names=static_feature_names,
            ),
        )

    # Zero-pad sequences to the longest T found
    T_max = max(s.shape[0] for s in all_X_seq_list)
    padded = np.zeros((len(all_X_seq_list), T_max, len(SURROGATE_TEMPORAL_COLS)), dtype=np.float32)
    for i, seq in enumerate(all_X_seq_list):
        padded[i, : seq.shape[0], :] = seq

    static_feature_names = STATIC_COLS + ([] if use_temporal else ["inflow_multiplier"])
    meta = _with_dataset_attrs(
        pd.DataFrame(meta_rows),
        temporal_feature_names=SURROGATE_TEMPORAL_COLS,
        static_feature_names=static_feature_names,
    )

    return TemporalWindowDataset(
        X_seq=padded,
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=meta,
    )


SWMM_OUTPUT_COLS: list[str] = [
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "max_total_outflow_lps",
    "time_to_peak_outflow_min",
    "downstream_link_peak_flows_lps_json",
]

_UNIFIED_META_COLS: list[str] = [
    "network_hash",
    "network_file",
    "scenario_type",
    "spatial_pattern",
    "flooding_duration_min",
]

_UNIFIED_TARGET_COLS: list[str] = ["flooded", "peak_flooding_lps"]


def build_unified_dataset(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    db_path: Path = DEFAULT_DB_FILE,
    resample_min: int = 5,
) -> TemporalWindowDataset:
    """Build one sample per (run_id, node_id) for the unified RF vs CNN comparison.

    Reads static features from the CSV (dropping SWMM-output columns unavailable
    before running SWMM). Joins inflow timeseries from Parquet files registered in
    temporal_artifacts — one Parquet read per run for efficiency.

    Returned dataset is shared by both models:
    - RF/XGBoost: uses X_static directly (21 inference-available features)
    - CNN: uses X_seq [T, 2] + X_static
    """
    csv_path = Path(csv_path)
    df_csv = pd.read_csv(csv_path)

    # Build static feature matrix: drop SWMM outputs, metadata, targets
    drop_cols = set(SWMM_OUTPUT_COLS) | set(_UNIFIED_META_COLS) | set(_UNIFIED_TARGET_COLS)
    feat_df = df_csv.drop(columns=[c for c in drop_cols if c in df_csv.columns])

    # One-hot encode node_type (junction/outfall → 1 binary column, total stays 21)
    if "node_type" in feat_df.columns:
        feat_df = pd.get_dummies(feat_df, columns=["node_type"], drop_first=True)

    # Fill NaNs (source nodes have no upstream; outfalls have no downstream pipes)
    feat_df = feat_df.fillna(0.0)

    feature_cols: list[str] = [
        c for c in feat_df.columns if c not in ("run_id", "node_id")
    ]

    # Parquet lookup: run_id → path (one read per run below)
    conn = sqlite3.connect(db_path)
    try:
        parquet_rows = conn.execute(
            "SELECT run_id, parquet_path FROM temporal_artifacts"
        ).fetchall()
    finally:
        conn.close()
    parquet_lookup: dict[str, str] = {rid: ppath for rid, ppath in parquet_rows}

    all_X_seq: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    for run_id, run_group in df_csv.groupby("run_id", sort=False):
        parquet_path = parquet_lookup.get(str(run_id))
        if parquet_path is None:
            warnings.warn(f"run_id '{run_id}' has no temporal_artifact — skipping.", stacklevel=2)
            continue

        parquet_df = pd.read_parquet(parquet_path)

        for csv_idx in run_group.index:
            csv_row = df_csv.loc[csv_idx]
            node_id = str(csv_row["node_id"])

            node_df = (
                parquet_df[parquet_df["node_id"] == node_id]
                .sort_values("time_min")
                .drop_duplicates(subset=["time_min"], keep="last")
                .reset_index(drop=True)
            )
            if node_df.empty:
                continue

            t_start = node_df["time_min"].iloc[0]
            t_end = node_df["time_min"].iloc[-1]
            n_grid = int(round((t_end - t_start) / resample_min)) + 1
            grid = t_start + np.arange(n_grid, dtype=float) * resample_min
            node_df = (
                node_df.set_index("time_min")
                .reindex(grid)
                .ffill()
                .dropna(subset=SURROGATE_TEMPORAL_COLS)
                .reset_index()
            )
            if node_df.empty:
                continue

            seq = node_df[SURROGATE_TEMPORAL_COLS].values.astype(np.float32)
            x_static = feat_df.loc[csv_idx, feature_cols].values.astype(np.float32)

            all_X_seq.append(seq)
            all_X_static.append(x_static)
            all_y_class.append(int(csv_row["flooded"]))
            all_y_reg.append(float(csv_row["peak_flooding_lps"]))
            all_groups.append(str(run_id))
            meta_rows.append({"run_id": str(run_id), "node_id": node_id, "window_start_min": 0.0})

    if not all_X_seq:
        return TemporalWindowDataset(
            X_seq=np.empty((0, 1, len(SURROGATE_TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(feature_cols)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=_with_dataset_attrs(
                pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
                temporal_feature_names=SURROGATE_TEMPORAL_COLS,
                static_feature_names=feature_cols,
            ),
        )

    T_max = max(s.shape[0] for s in all_X_seq)
    F = len(SURROGATE_TEMPORAL_COLS)
    padded = np.zeros((len(all_X_seq), T_max, F), dtype=np.float32)
    for i, seq in enumerate(all_X_seq):
        padded[i, : seq.shape[0], :] = seq

    meta_df = _with_dataset_attrs(
        pd.DataFrame(meta_rows),
        temporal_feature_names=SURROGATE_TEMPORAL_COLS,
        static_feature_names=feature_cols,
    )

    return TemporalWindowDataset(
        X_seq=padded,
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=meta_df,
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
