from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.model_comparison import compute_volume_metrics

_SYMLOG_THRESH = 1.0  # m³


def _safe_r2(y_true, y_pred) -> str:
    from sklearn.metrics import r2_score
    arr = np.asarray(y_true, dtype=float)
    if len(arr) < 2 or np.var(arr) == 0:
        return "N/A"
    return f"{r2_score(arr, np.asarray(y_pred, dtype=float)):.3f}"


def _metrics_text(vol_swmm, vol_pred) -> str:
    m = compute_volume_metrics(vol_swmm, vol_pred)
    r2 = _safe_r2(vol_swmm, vol_pred)
    return (
        f"R²={r2}  MAE={m['mae_m3']:.2f} m³  "
        f"RMSE={m['rmse_m3']:.2f} m³  n={len(list(vol_swmm))}"
    )


def _draw_parity(ax, x, y, title: str):
    x_arr = list(x)
    y_arr = list(y)
    ax.scatter(x_arr, y_arr, alpha=0.6, s=20, color="steelblue")
    lim = max(max(x_arr, default=0), max(y_arr, default=0), 1e-3) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("SWMM (m³)")
    ax.set_ylabel("XGBoost (m³)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.text(
        0.02, 0.97, _metrics_text(x_arr, y_arr),
        transform=ax.transAxes, va="top", fontsize=7,
    )


def _draw_parity_symlog(ax, x, y, title: str):
    x_arr = list(x)
    y_arr = list(y)
    ax.scatter(x_arr, y_arr, alpha=0.6, s=20, color="steelblue")
    lim = max(max(x_arr, default=0), max(y_arr, default=0), _SYMLOG_THRESH) * 1.05
    ref = np.linspace(0, lim, 200)
    ax.plot(ref, ref, "r--", linewidth=0.8, label="y = x")
    ax.set_xscale("symlog", linthresh=_SYMLOG_THRESH)
    ax.set_yscale("symlog", linthresh=_SYMLOG_THRESH)
    ax.set_xlabel("SWMM (m³)")
    ax.set_ylabel("XGBoost (m³)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.text(
        0.02, 0.97, _metrics_text(x_arr, y_arr),
        transform=ax.transAxes, va="top", fontsize=7,
    )


def plot_parity_nodes(df: pd.DataFrame, out_dir: Path, scenario_id: str) -> list[Path]:
    """Parity plots (linear + symlog) of vol_swmm_m3 vs vol_pred_m3 per node."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = df["vol_swmm_m3"].tolist()
    y = df["vol_pred_m3"].tolist()
    title = f"Paridad por nodo — {scenario_id}"
    paths = []

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_parity(ax, x, y, title)
    p = out_dir / f"parity_nodes_linear_{scenario_id}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_parity_symlog(ax, x, y, title)
    p = out_dir / f"parity_nodes_symlog_{scenario_id}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def plot_parity_aggregated(df: pd.DataFrame, out_dir: Path, scenario_id: str) -> list[Path]:
    """Parity plots of total vol_swmm_m3 vs total vol_pred_m3 (one point)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = [df["vol_swmm_m3"].sum()]
    y = [df["vol_pred_m3"].sum()]
    title = f"Paridad red total — {scenario_id}"
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_parity(ax, x, y, title)
    p = out_dir / f"parity_agg_linear_{scenario_id}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_parity_symlog(ax, x, y, title)
    p = out_dir / f"parity_agg_symlog_{scenario_id}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def plot_node_profiles(df: pd.DataFrame, out_dir: Path, scenario_id: str) -> list[Path]:
    """Bar/line profiles of vol_swmm_m3 and vol_pred_m3 per node.

    Only nodes where at least one of the two volumes > 0 are included.
    Nodes are ordered by vol_swmm_m3 descending (node_id ascending as tiebreak).
    Returns empty list if all nodes have zero volumes.
    """
    visible = df[(df["vol_swmm_m3"] > 0) | (df["vol_pred_m3"] > 0)].copy()
    if visible.empty:
        return []

    visible = visible.sort_values(
        ["vol_swmm_m3", "node_id"], ascending=[False, True]
    )
    nodes = visible["node_id"].tolist()
    x = np.arange(len(nodes))
    w = max(8, min(24, len(nodes) * 0.35))
    paths = []

    for scale, suffix in [("linear", "linear"), ("symlog", "symlog")]:
        fig, ax = plt.subplots(figsize=(w, 5))
        ax.plot(x, visible["vol_swmm_m3"].tolist(), "b-o", label="SWMM", markersize=4)
        ax.plot(x, visible["vol_pred_m3"].tolist(), "o-", color="orange", label="XGBoost", markersize=4)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=_SYMLOG_THRESH)
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Volumen de inundación (m³)")
        ax.set_title(f"Perfil por nodo — {scenario_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"node_profile_{suffix}_{scenario_id}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths
