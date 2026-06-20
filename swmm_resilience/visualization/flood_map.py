"""
Figure 2 — Flood map.

Receives a standard DataFrame with node-level flood results and renders
a spatial map with a color+size gradient for flood volume when available,
falling back to peak_flooding_lps for legacy results.
Works identically regardless of whether data came from SWMM or ML.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..simulation.swmm_api_io import load_inp
from ._inp_parser import parse_conduits, parse_coordinates
from .labels import format_node_label

# ── visual constants ─────────────────────────────────────────────────────────
PIPE_COLOR = "#B0BEC5"
PIPE_LW = 0.8
NODE_DRY_COLOR = "#CFD8DC"
NODE_DRY_SIZE = 15
NODE_DRY_EDGE = "#90A4AE"
NODE_FLOOD_SIZE_MIN = 40
NODE_FLOOD_SIZE_MAX = 500
COLORMAP = "plasma"
ANNOTATION_BBOX = dict(boxstyle="round,pad=0.3", fc="#FFFF99", ec="#AAAAAA", alpha=0.9)
TOP_N_LABELS = 5
FIG_SIZE = (13, 10)
DPI = 150


def _scale_sizes(volumes: pd.Series, vmax: float) -> np.ndarray:
    """Map flood volumes linearly to marker sizes."""
    if vmax == 0:
        return np.full(len(volumes), NODE_FLOOD_SIZE_MIN)
    ratio = np.clip(volumes.to_numpy() / vmax, 0, 1)
    return NODE_FLOOD_SIZE_MIN + ratio * (NODE_FLOOD_SIZE_MAX - NODE_FLOOD_SIZE_MIN)


def plot_flood_map(
    node_data: pd.DataFrame,
    inp_path: Path | str,
    output_path: Path | str,
    title: str,
    vmax_global: float | None = None,
    runtime_text: str | None = None,
) -> Path:
    """Generate and save a flood map (experiment branch interface).

    Parameters
    ----------
    node_data    : DataFrame with columns [node_id, flooded, source,
                   inflow_multiplier] and either total_flood_volume_m3
                   or peak_flooding_lps.
    inp_path     : Path to the SWMM .inp file (topology + coordinates).
    output_path  : Destination PNG path.
    title        : Figure title (two lines separated by \\n are rendered as subtitle).
    vmax_global  : Color scale maximum. If None, uses the max in node_data.
    """
    inp_path = Path(inp_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)

    df = node_data.copy()
    df["node_id"] = df["node_id"].astype(str)
    df = df[df["node_id"].isin(coords)].reset_index(drop=True)

    df["x"] = df["node_id"].map(lambda n: coords[n][0])
    df["y"] = df["node_id"].map(lambda n: coords[n][1])

    has_volume = "total_flood_volume_m3" in df.columns
    has_peak = "peak_flooding_lps" in df.columns
    if not has_volume and not has_peak:
        raise ValueError("node_data must include total_flood_volume_m3 or peak_flooding_lps.")

    if has_volume:
        df["total_flood_volume_m3"] = df["total_flood_volume_m3"].clip(lower=0.0)
    if has_peak:
        df["peak_flooding_lps"] = df["peak_flooding_lps"].clip(lower=0.0)

    preferred_metric = node_data.attrs.get("preferred_flood_metric")
    if preferred_metric in {"total_flood_volume_m3", "peak_flooding_lps"} and preferred_metric in df.columns:
        metric_col = preferred_metric
    elif has_volume and (not has_peak or float(df["total_flood_volume_m3"].max()) > 0):
        metric_col = "total_flood_volume_m3"
    else:
        metric_col = "peak_flooding_lps"
    metric_label = (
        "Total Flood Volume (m3)"
        if metric_col == "total_flood_volume_m3"
        else "Peak Flooding Flow (L/s)"
    )
    metric_unit = "m3" if metric_col == "total_flood_volume_m3" else "L/s"

    vmax = vmax_global if vmax_global is not None else float(df[metric_col].max())
    if vmax == 0:
        vmax = 1.0

    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for _lid, frm, to in conduits:
        if frm not in coords or to not in coords:
            continue
        x0, y0 = coords[frm]
        x1, y1 = coords[to]
        ax.plot([x0, x1], [y0, y1], color=PIPE_COLOR, lw=PIPE_LW, zorder=1)

    dry = df[df[metric_col] <= 0]
    ax.scatter(
        dry["x"], dry["y"],
        s=NODE_DRY_SIZE,
        color=NODE_DRY_COLOR,
        edgecolors=NODE_DRY_EDGE,
        linewidths=0.5,
        zorder=2,
        label="Non-Flooded Nodes",
    )

    wet = df[df[metric_col] > 0].copy()
    if not wet.empty:
        sizes = _scale_sizes(wet[metric_col], vmax)
        sc = ax.scatter(
            wet["x"], wet["y"],
            s=sizes,
            c=wet[metric_col].to_numpy(),
            cmap=COLORMAP,
            norm=norm,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label="Flooded Nodes",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(metric_label, fontsize=10)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=COLORMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(metric_label, fontsize=10)

    top = df.nlargest(TOP_N_LABELS, metric_col)
    top = top[top[metric_col] > 0]
    for _, row in top.iterrows():
        ax.annotate(
            f"{format_node_label(row['node_id'])}\n{row[metric_col]:.1f} {metric_unit}",
            xy=(row["x"], row["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            bbox=ANNOTATION_BBOX,
            zorder=5,
        )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=0.8)
    ax.set_xlabel("X Coordinate (m)", fontsize=10)
    ax.set_ylabel("Y Coordinate (m)", fontsize=10)

    lines = title.split("\n", 1)
    fig.suptitle(lines[0], fontsize=12, fontweight="bold", y=1.02)
    if len(lines) > 1:
        ax.set_title(lines[1], fontsize=10, style="italic", pad=6)

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)

    if runtime_text:
        ax.text(
            0.98,
            0.02,
            runtime_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=ANNOTATION_BBOX,
            zorder=6,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_flood_map(
    inp_path: Path,
    vol_data: pd.DataFrame,
    factor: float,
    output_path: Path,
    network_name: str = "Network",
    colormap: str = "RdYlBu_r",
    show_labels_top_n: int = 5,
):
    """Render a network map with flood-volume gradient (origin/main interface).

    vol_data must contain node_id and either vol_inundacion_m3 or vol_pred_m3.
    Nodes with duplicate coordinates receive a ±2m X offset for visual separation.
    """
    inp = load_inp(inp_path)

    coords: dict = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = [float(c.x), float(c.y)]

    conduits_list = []
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            fn, tn = str(c.from_node), str(c.to_node)
            if fn in coords and tn in coords:
                conduits_list.append((fn, tn))

    coord_groups: dict = defaultdict(list)
    for nid, (x, y) in coords.items():
        coord_groups[(x, y)].append(nid)
    for (x, y), nodes in coord_groups.items():
        if len(nodes) == 2:
            coords[nodes[0]] = [x - 2.0, y]
            coords[nodes[1]] = [x + 2.0, y]

    vol_col = "vol_inundacion_m3" if "vol_inundacion_m3" in vol_data.columns else "vol_pred_m3"
    node_vol = dict(zip(vol_data["node_id"].astype(str), vol_data[vol_col].fillna(0.0)))

    fig, ax = plt.subplots(figsize=(14, 10))

    for fn, tn in conduits_list:
        ax.plot(
            [coords[fn][0], coords[tn][0]],
            [coords[fn][1], coords[tn][1]],
            color="#888888", linewidth=0.7, zorder=1,
        )

    vols = np.array([max(0.0, node_vol.get(nid, 0.0)) for nid in coords])
    max_vol = vols.max() if vols.max() > 0 else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=max_vol)
    cmap_obj = plt.get_cmap(colormap)

    for nid, (x, y) in coords.items():
        vol = max(0.0, node_vol.get(nid, 0.0))
        if vol > 0:
            ax.scatter(x, y, c=[cmap_obj(norm(vol))], s=40 + (vol / max_vol) * 180,
                       zorder=3, edgecolors="none")
        else:
            ax.scatter(x, y, color="#aec6e8", s=15, zorder=2)

    sorted_nids = sorted(coords, key=lambda n: node_vol.get(n, 0.0), reverse=True)
    for nid in sorted_nids[:show_labels_top_n]:
        vol = node_vol.get(nid, 0.0)
        if vol > 0:
            x, y = coords[nid]
            ax.annotate(
                f"{format_node_label(nid)}\n{vol:.1f} m³", (x, y),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Flood Volume (m³)", shrink=0.7)
    ax.set_title(f"{network_name} - Scale Factor: {factor:.2f}", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_path}")
    return output_path
