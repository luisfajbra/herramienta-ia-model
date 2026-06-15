from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.model_comparison import compute_volume_metrics
from .flood_map import plot_flood_map
from .labels import format_node_label

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
    title = f"Node-Level Parity - {scenario_id}"
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
    title = f"Total Network Parity - {scenario_id}"
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
    nodes = [format_node_label(node_id) for node_id in visible["node_id"]]
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
        ax.set_ylabel("Flood Volume (m³)")
        ax.set_title(f"Node Profile - {scenario_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"node_profile_{suffix}_{scenario_id}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


def plot_scenario_flood_maps(
    comp_df: pd.DataFrame,
    inp_path: Path,
    output_dir: Path,
    scenario_id: str,
) -> tuple[Path, Path]:
    """Create comparable SWMM and ML flood maps for one scenario."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swmm_volume = pd.to_numeric(comp_df["vol_swmm_m3"], errors="coerce").fillna(0.0)
    ml_volume = pd.to_numeric(comp_df["vol_pred_m3"], errors="coerce").fillna(0.0)
    vmax_global = max(float(swmm_volume.max()), float(ml_volume.max()), 1.0)

    swmm_data = pd.DataFrame(
        {
            "node_id": comp_df["node_id"],
            "flooded": comp_df["inunda_swmm"].astype(bool),
            "total_flood_volume_m3": swmm_volume,
        }
    )
    ml_data = pd.DataFrame(
        {
            "node_id": comp_df["node_id"],
            "flooded": comp_df["inunda_pred"].astype(bool),
            "total_flood_volume_m3": ml_volume,
        }
    )
    swmm_data.attrs["preferred_flood_metric"] = "total_flood_volume_m3"
    ml_data.attrs["preferred_flood_metric"] = "total_flood_volume_m3"

    swmm_path = output_dir / f"flood_map_swmm_{scenario_id}.png"
    ml_path = output_dir / f"flood_map_ml_{scenario_id}.png"
    plot_flood_map(
        node_data=swmm_data,
        inp_path=Path(inp_path),
        output_path=swmm_path,
        title=f"Flood Map - {scenario_id}\nSWMM Simulation",
        vmax_global=vmax_global,
    )
    plot_flood_map(
        node_data=ml_data,
        inp_path=Path(inp_path),
        output_path=ml_path,
        title=f"Flood Map - {scenario_id}\nML Prediction",
        vmax_global=vmax_global,
    )
    return swmm_path, ml_path


def plot_factor_comparison(
    comp_df: pd.DataFrame,
    output_dir: Path,
    factor: float,
) -> tuple[Path, Path]:
    """Plot per-node volumes and SWMM/XGBoost parity for one factor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = comp_df.sort_values("node_id").reset_index(drop=True)
    labels = [format_node_label(node_id) for node_id in ordered["node_id"]]
    x = np.arange(len(ordered))
    width = max(10, min(28, len(ordered) * 0.18))

    fig, ax = plt.subplots(figsize=(width, 6))
    ax.plot(
        x,
        ordered["vol_swmm_m3"],
        color="#2176ae",
        linewidth=1.8,
        label="SWMM",
    )
    ax.plot(
        x,
        ordered["vol_pred_m3"],
        color="#f28e2b",
        linewidth=1.8,
        label="XGBoost",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Flood Volume (m³)")
    ax.set_title(f"Flood Volume by Node - Factor {factor:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    profile_path = output_dir / f"volume_by_node_factor_{factor:.2f}.png"
    fig.savefig(profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    swmm = ordered["vol_swmm_m3"].astype(float)
    predicted = ordered["vol_pred_m3"].astype(float)
    limit = max(float(swmm.max()), float(predicted.max()), 1.0) * 1.05
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(swmm, predicted, color="#4c78a8", alpha=0.7, s=28)
    ax.plot([0.0, limit], [0.0, limit], "r--", linewidth=1.2, label="y = x")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("SWMM Flood Volume (m³)")
    ax.set_ylabel("XGBoost Flood Volume (m³)")
    ax.set_title(f"SWMM vs XGBoost Flood Volume - Factor {factor:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    parity_path = output_dir / f"parity_factor_{factor:.2f}.png"
    fig.savefig(parity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return profile_path, parity_path


def plot_flooded_swmm_node_profile(
    comp_df: pd.DataFrame,
    output_dir: Path,
    factor: float,
) -> Path | None:
    """Plot SWMM and XGBoost volumes only where the SWMM volume is positive."""
    visible = comp_df[
        pd.to_numeric(comp_df["vol_swmm_m3"], errors="coerce").fillna(0.0) > 0.0
    ].copy()
    if visible.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visible = visible.sort_values("node_id").reset_index(drop=True)
    labels = [format_node_label(node_id) for node_id in visible["node_id"]]
    x = np.arange(len(visible))
    width = max(10, min(28, len(visible) * 0.28))

    fig, ax = plt.subplots(figsize=(width, 6))
    ax.plot(
        x,
        visible["vol_swmm_m3"],
        color="#2176ae",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="SWMM",
    )
    ax.plot(
        x,
        visible["vol_pred_m3"],
        color="#f28e2b",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="XGBoost",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Flood Volume (m³)")
    ax.set_title(
        f"Flood Volume by SWMM-Flooded Node - Factor {factor:.2f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = (
        output_dir
        / f"volume_by_node_flooded_swmm_factor_{factor:.2f}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_totals_comparison(totals_df: pd.DataFrame, out_path: Path) -> Path | None:
    """Paired bar chart of total network flood volume per scenario, SWMM vs ML.

    totals_df must have columns: scenario_id, vol_total_swmm_m3,
    vol_total_pred_m3. Returns the written path, or None for empty input.
    """
    if totals_df.empty:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = totals_df["scenario_id"].astype(str).tolist()
    x = np.arange(len(scenarios))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(scenarios)), 5))
    ax.bar(x - width / 2, totals_df["vol_total_swmm_m3"], width,
           label="SWMM", color="steelblue")
    ax.bar(x + width / 2, totals_df["vol_total_pred_m3"], width,
           label="ML", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("Total network flood volume (m³)")
    ax.set_title("Total Flood Volume per Scenario — SWMM vs ML")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
