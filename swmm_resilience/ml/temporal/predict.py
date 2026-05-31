"""
Temporal CNN inference + prediction map.

CLI
---
    python -m swmm_resilience.ml.temporal.predict \\
        --parquet data/networks/.../results/temporal/node_timeseries/run_XYZ.parquet \\
        --inp "data/networks/.../SWMM - ....inp" \\
        --output predictions.png
"""

from __future__ import annotations

import argparse
import sqlite3
import warnings
from pathlib import Path

import joblib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ...config import DEFAULT_DB_FILE, DEFAULT_SURROGATE_MAPS_DIR, DEFAULT_TEMPORAL_ARTIFACTS_DIR
from ...visualization._inp_parser import parse_conduits, parse_coordinates
from ...visualization.flood_map import plot_flood_map
from .dataset import STATIC_COLS, SURROGATE_TEMPORAL_COLS, TEMPORAL_COLS
from .models.cnn import SWMMTemporalCNN
from .models.surrogate_cnn import SWMMSurrogateCNN
from .models.surrogate_lstm import SWMMSurrogateLSTM
from .schemas import TemporalWindowSpec


_SURROGATE_MODELS: dict[str, type] = {
    "cnn": SWMMSurrogateCNN,
    "lstm": SWMMSurrogateLSTM,
}
_SURROGATE_PREFIXES: dict[str, str] = {
    "cnn": "surrogate_cnn",
    "lstm": "surrogate_lstm",
}


_PROB_CMAP = "plasma"
_NODE_SIZE_MIN = 30
_NODE_SIZE_MAX = 400
_FIG_SIZE = (13, 10)
_DPI = 150
_TOP_N = 8
_ANNOTATION_BBOX = dict(boxstyle="round,pad=0.3", fc="#FFFF99", ec="#AAAAAA", alpha=0.9)


def predict_from_parquet(
    parquet_path: Path,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    db_path: Path = DEFAULT_DB_FILE,
    task: str = "classification",
    device: str = "cpu",
) -> pd.DataFrame:
    """Run CNN inference on a single Parquet run.

    Returns a DataFrame with one row per node:
        node_id, max_flood_prob, mean_flood_prob,
        windows_total, windows_flood_predicted, actual_flooded
    """
    spec = TemporalWindowSpec()
    resample_min = spec.resample_min
    window_steps = spec.window_min // resample_min
    horizon_steps = spec.horizon_min // resample_min
    step_steps = spec.step_min // resample_min

    prefix = "cnn_classifier" if task == "classification" else "cnn_regressor"
    state_dict = torch.load(
        artifacts_dir / f"{prefix}_weights.pt",
        map_location=device,
        weights_only=True,
    )
    scaler_seq = joblib.load(artifacts_dir / f"{prefix}_scaler_seq.joblib")
    scaler_static = joblib.load(artifacts_dir / f"{prefix}_scaler_static.joblib")

    model = SWMMTemporalCNN(
        n_temporal_features=len(TEMPORAL_COLS),
        n_static_features=len(STATIC_COLS),
        task=task,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    df = pd.read_parquet(parquet_path)
    network_hash = str(df["network_hash"].iloc[0])

    conn = sqlite3.connect(db_path)
    try:
        static_rows = conn.execute(
            f"SELECT node_uid, {', '.join(STATIC_COLS)} "
            "FROM network_nodes WHERE network_hash = ?",
            (network_hash,),
        ).fetchall()
    finally:
        conn.close()

    static_lookup: dict[str, np.ndarray] = {
        row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
        for row in static_rows
    }

    records: list[dict] = []
    for node_id in df["node_id"].unique():
        if node_id not in static_lookup:
            warnings.warn(f"node_id '{node_id}' has no entry in network_nodes — skipping.")
            continue
        x_static_raw = static_lookup[node_id]

        node_df = (
            df[df["node_id"] == node_id]
            .sort_values("time_min")
            .drop_duplicates(subset=["time_min"], keep="last")
            .reset_index(drop=True)
        )
        if node_df.empty:
            continue

        actual_flooded = int((node_df["flooding_lps"] > 0).any())

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

        windows: list[np.ndarray] = []
        i = 0
        while i + window_steps + horizon_steps <= len(node_df):
            win = node_df.iloc[i: i + window_steps][TEMPORAL_COLS].values.astype(np.float32)
            windows.append(win)
            i += step_steps

        if not windows:
            records.append({
                "node_id": node_id,
                "max_flood_prob": float("nan"),
                "mean_flood_prob": float("nan"),
                "windows_total": 0,
                "windows_flood_predicted": 0,
                "actual_flooded": actual_flooded,
            })
            continue

        X_seq = np.stack(windows)          # [N, window_steps, n_features]
        N, T, F = X_seq.shape
        X_seq_sc = scaler_seq.transform(X_seq.reshape(-1, F)).reshape(N, T, F)
        X_static_sc = scaler_static.transform(np.tile(x_static_raw, (N, 1)))

        with torch.no_grad():
            probs = (
                model(
                    torch.tensor(X_seq_sc, dtype=torch.float32).to(device),
                    torch.tensor(X_static_sc, dtype=torch.float32).to(device),
                )
                .cpu()
                .numpy()
                .flatten()
            )

        records.append({
            "node_id": node_id,
            "max_flood_prob": float(probs.max()),
            "mean_flood_prob": float(probs.mean()),
            "windows_total": int(len(probs)),
            "windows_flood_predicted": int((probs >= 0.5).sum()),
            "actual_flooded": actual_flooded,
        })

    return pd.DataFrame(records)


def plot_prediction_map(
    predictions: pd.DataFrame,
    inp_path: Path,
    output_path: Path,
    title: str = "CNN Flood Prediction Map",
    vmax: float | None = None,
    high_risk_quantile: float = 0.75,
) -> Path:
    """Save a prediction map coloring nodes by max flood probability.

    Parameters
    ----------
    predictions       : DataFrame from predict_from_parquet().
                        Must have: node_id, max_flood_prob, actual_flooded.
    inp_path          : Path to the network .inp file.
    output_path       : Destination PNG path.
    title             : Figure title (two lines separated by \\n → bold + italic).
    vmax              : Colormap upper bound. None = use data max (better contrast
                        when probabilities are compressed below 0.5).
    high_risk_quantile: Quantile threshold above which a node is "predicted at risk"
                        for TP/FP/FN/TN stats (default 0.75 = top 25%).
    """
    inp_path = Path(inp_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)

    df = predictions.copy()
    df["node_id"] = df["node_id"].astype(str)
    df = df[df["node_id"].isin(coords)].reset_index(drop=True)
    df["x"] = df["node_id"].map(lambda n: coords[n][0])
    df["y"] = df["node_id"].map(lambda n: coords[n][1])

    # Replace NaN probs (nodes with zero windows) with 0
    df["max_flood_prob"] = df["max_flood_prob"].fillna(0.0)

    # Auto-scale colormap to data range so gradient is visible even when all
    # probabilities are well below 0.5 (common with few training epochs).
    p_max = float(df["max_flood_prob"].max())
    _vmax = vmax if vmax is not None else max(p_max, 1e-3)
    norm = mcolors.Normalize(vmin=0.0, vmax=_vmax)
    cmap = plt.colormaps[_PROB_CMAP]

    # Node sizes proportional to predicted probability
    sizes = _NODE_SIZE_MIN + df["max_flood_prob"].to_numpy() * (_NODE_SIZE_MAX - _NODE_SIZE_MIN)

    # Confusion categories — use quantile threshold so stats are meaningful even
    # when probabilities are compressed below 0.5 (undertrained models).
    risk_threshold = float(df["max_flood_prob"].quantile(high_risk_quantile))
    risk_threshold = max(risk_threshold, 1e-6)  # avoid all-zero edge case
    df["predicted_at_risk"] = df["max_flood_prob"] >= risk_threshold
    tp = df[df["predicted_at_risk"] & (df["actual_flooded"] == 1)]
    fp = df[df["predicted_at_risk"] & (df["actual_flooded"] == 0)]
    fn = df[~df["predicted_at_risk"] & (df["actual_flooded"] == 1)]
    tn = df[~df["predicted_at_risk"] & (df["actual_flooded"] == 0)]
    pct_label = f"{int(high_risk_quantile * 100)}th pct"

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # ── pipes ──────────────────────────────────────────────────────────────────
    for _lid, frm, to in conduits:
        if frm not in coords or to not in coords:
            continue
        x0, y0 = coords[frm]
        x1, y1 = coords[to]
        ax.plot([x0, x1], [y0, y1], color="#B0BEC5", lw=0.8, zorder=1)

    # ── all nodes: colored by predicted probability ────────────────────────────
    sc = ax.scatter(
        df["x"], df["y"],
        s=sizes,
        c=df["max_flood_prob"].to_numpy(),
        cmap=_PROB_CMAP,
        norm=norm,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )

    # ── actual flooded nodes: bold ring overlay ────────────────────────────────
    actual_wet = df[df["actual_flooded"] == 1]
    if not actual_wet.empty:
        ax.scatter(
            actual_wet["x"], actual_wet["y"],
            s=sizes[actual_wet.index] + 60,
            facecolors="none",
            edgecolors="#E53935",
            linewidths=1.8,
            zorder=3,
            label=f"Inundado (real): {len(actual_wet)} nodos",
        )

    # ── colorbar ───────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(
        f"Prob. máx. de inundación (CNN, escala 0–{_vmax:.3f})", fontsize=10
    )
    # Mark the quantile threshold on the colorbar
    thresh_pos = risk_threshold / _vmax
    if 0 < thresh_pos < 1:
        cbar.ax.axhline(thresh_pos, color="white", lw=1.5, linestyle="--")

    # ── top-N annotations ──────────────────────────────────────────────────────
    top = df.nlargest(_TOP_N, "max_flood_prob")
    top = top[top["max_flood_prob"] > 0.1]
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['node_id']}\n{row['max_flood_prob']:.2f}",
            xy=(row["x"], row["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            bbox=_ANNOTATION_BBOX,
            zorder=5,
        )

    # ── summary stats in legend ────────────────────────────────────────────────
    legend_entries = [
        f"Umbral: top {pct_label} de riesgo predicho",
        f"  TP (detectados):          {len(tp)}",
        f"  FP (falsas alarmas):       {len(fp)}",
        f"  FN (no detectados):        {len(fn)}",
        f"  TN (correctamente seguros): {len(tn)}",
    ]
    ax.annotate(
        "\n".join(legend_entries),
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8.5,
        fontfamily="monospace",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.85),
        zorder=6,
    )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_xlabel("Coordenada X (m)", fontsize=10)
    ax.set_ylabel("Coordenada Y (m)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)

    lines = title.split("\n", 1)
    fig.suptitle(lines[0], fontsize=12, fontweight="bold", y=1.01)
    if len(lines) > 1:
        ax.set_title(lines[1], fontsize=10, style="italic", pad=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def predict_surrogate_from_multiplier(
    multiplier: float,
    db_path: Path = DEFAULT_DB_FILE,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    device: str = "cpu",
    model_type: str = "cnn",
) -> pd.DataFrame:
    """Predict flood risk for every node given an inflow multiplier.

    No SWMM run required. Loads the Qx1.00 base hydrograph from the DB,
    scales total_inflow_lps and lateral_inflow_lps by `multiplier`, sets
    SWMM-output features (depth_m, depth_ratio, flooding_lps, total_outflow_lps)
    to zero, then runs SWMMSurrogateCNN.

    Returns DataFrame with columns:
        node_id, flood_prob, predicted_flooded, peak_flooding_lps_pred
    """
    artifacts_dir = Path(artifacts_dir)
    if model_type not in _SURROGATE_MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(_SURROGATE_MODELS.keys())}")

    prefix = _SURROGATE_PREFIXES[model_type]
    model_cls = _SURROGATE_MODELS[model_type]

    state_dict = torch.load(
        artifacts_dir / f"{prefix}_weights.pt",
        map_location=device,
        weights_only=True,
    )
    scaler_seq = joblib.load(artifacts_dir / f"{prefix}_scaler_seq.joblib")
    scaler_static = joblib.load(artifacts_dir / f"{prefix}_scaler_static.joblib")

    model = model_cls(
        n_temporal_features=len(SURROGATE_TEMPORAL_COLS),
        n_static_features=len(STATIC_COLS),
        use_temporal=True,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Qx1.00 base run (lowest multiplier in temporal_artifacts)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT ta.parquet_path, ta.network_hash "
            "FROM temporal_artifacts ta "
            "JOIN runs r ON ta.run_id = r.run_id "
            "ORDER BY r.inflow_multiplier ASC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("No temporal artifacts found in the database.")
        base_parquet_path, network_hash = row

        static_rows = conn.execute(
            f"SELECT node_uid, {', '.join(STATIC_COLS)} "
            "FROM network_nodes WHERE network_hash = ?",
            (network_hash,),
        ).fetchall()
    finally:
        conn.close()

    static_lookup: dict[str, np.ndarray] = {
        row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
        for row in static_rows
    }

    df = pd.read_parquet(base_parquet_path)

    records: list[dict] = []
    for node_id in df["node_id"].unique():
        if node_id not in static_lookup:
            warnings.warn(f"node_id '{node_id}' not in network_nodes — skipping.")
            continue

        node_df = (
            df[df["node_id"] == node_id]
            .sort_values("time_min")
            .drop_duplicates(subset=["time_min"], keep="last")
            .reset_index(drop=True)
        )
        if node_df.empty:
            continue

        # Resample to 5-min grid
        resample_min = 5
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
        seq *= multiplier  # both columns are inflow features

        T, F = seq.shape
        seq_sc = scaler_seq.transform(seq.reshape(-1, F)).reshape(1, T, F)
        x_static_raw = static_lookup[node_id]
        x_static_sc = scaler_static.transform(x_static_raw.reshape(1, -1))

        with torch.no_grad():
            cls_logit, reg_out = model(
                torch.tensor(seq_sc, dtype=torch.float32).to(device),
                torch.tensor(x_static_sc, dtype=torch.float32).to(device),
            )
            flood_prob = float(torch.sigmoid(cls_logit).cpu().item())
            peak_lps = float(reg_out.cpu().item())

        records.append({
            "node_id": node_id,
            "flood_prob": flood_prob,
            "predicted_flooded": int(flood_prob >= 0.5),
            "peak_flooding_lps_pred": max(peak_lps, 0.0),
        })

    return pd.DataFrame(records)


def plot_surrogate_map(
    predictions: pd.DataFrame,
    inp_path: Path,
    output_path: Path | None = None,
    multiplier: float | None = None,
    vmax: float | None = None,
    model_type: str = "cnn",
) -> Path:
    """Save a prediction map from surrogate CNN/LSTM output.

    Renames surrogate columns for plot_flood_map and saves the map
    with flooding volume (LPS) as the color/size gradient, not probability.
    """
    model_label = "CNN" if model_type == "cnn" else "LSTM"

    if output_path is None:
        out_dir = DEFAULT_SURROGATE_MAPS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if multiplier is not None:
            output_path = out_dir / f"surrogate_map_{model_type}_qx{multiplier:.2f}.png"
        else:
            output_path = out_dir / f"surrogate_map_{model_type}.png"

    df = predictions.rename(columns={
        "peak_flooding_lps_pred": "peak_flooding_lps",
        "predicted_flooded": "flooded",
    })
    df["source"] = f"Surrogate {model_label}"
    df["inflow_multiplier"] = multiplier if multiplier is not None else 1.0

    title = (
        f"Surrogate {model_label} — Volumen de Inundación Predicho"
        + (f"\nQx{multiplier:.2f}" if multiplier is not None else "")
    )
    return plot_flood_map(df, inp_path, output_path, title=title, vmax_global=vmax)


def predict_failure_timeline(*_args, **_kwargs):
    """Placeholder for future multi-step ahead timeline prediction (SP5)."""
    raise NotImplementedError(
        "Timeline prediction is reserved for SP5 (Temporal Predictor + Desktop)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CNN flood prediction on a single Parquet run and save a map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m swmm_resilience.ml.temporal.predict \\\n"
            '    --parquet data/networks/chico_hydro-qx1/results/temporal/node_timeseries/run_XYZ.parquet \\\n'
            '    --inp "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp" \\\n'
            "    --output predictions.png"
        ),
    )
    parser.add_argument(
        "--parquet", type=Path, required=True,
        help="Path to the node timeseries Parquet file for one run.",
    )
    parser.add_argument(
        "--inp", type=Path, required=True,
        help="Path to the SWMM .inp file (network topology + coordinates).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cnn_prediction_map.png"),
        help="Output PNG path (default: cnn_prediction_map.png).",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_TEMPORAL_ARTIFACTS_DIR,
        help="Directory containing model weights and scalers.",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_FILE,
        help="SQLite database path.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda/mps).")
    args = parser.parse_args()

    print(f"Loading model from: {args.artifacts_dir}")
    print(f"Running inference on: {args.parquet}")

    preds = predict_from_parquet(
        parquet_path=args.parquet,
        artifacts_dir=args.artifacts_dir,
        db_path=args.db,
        device=args.device,
    )

    n_nodes = len(preds)
    n_actual = int(preds["actual_flooded"].sum())
    n_predicted = int((preds["max_flood_prob"] >= 0.5).sum())
    print(f"\nResultados ({n_nodes} nodos):")
    print(f"  Inundados reales:     {n_actual}")
    print(f"  Predichos inundados:  {n_predicted}")

    try:
        run_id = str(pd.read_parquet(args.parquet)["run_id"].iloc[0])
    except Exception:
        run_id = args.parquet.stem

    inflow_multiplier: float | None = None
    try:
        conn = sqlite3.connect(args.db)
        row = conn.execute(
            "SELECT inflow_multiplier FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()
        if row:
            inflow_multiplier = float(row[0])
    except Exception:
        pass

    multiplier_label = f"Qx{inflow_multiplier:.2f}" if inflow_multiplier is not None else run_id[:8]

    title = (
        f"CNN — Predicción de Inundación\n"
        f"{multiplier_label} | Nodos: {n_nodes} | "
        f"Reales: {n_actual} | Predichos: {n_predicted}"
    )

    out = plot_prediction_map(
        predictions=preds,
        inp_path=args.inp,
        output_path=args.output,
        title=title,
    )
    print(f"\nMapa guardado en: {out}")


if __name__ == "__main__":
    main()
