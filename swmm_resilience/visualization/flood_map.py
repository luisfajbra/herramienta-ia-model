"""
Figure 2 — Flood map.

Receives a standard DataFrame with node-level flood results and renders
a spatial map with a color+size gradient for peak_flooding_lps.
Works identically regardless of whether data came from SWMM or ML.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from ._inp_parser import parse_conduits, parse_coordinates

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
) -> Path:
    """
    Generate and save a flood map.

    Parameters
    ----------
    node_data    : DataFrame with columns [node_id, peak_flooding_lps, flooded,
                   source, inflow_multiplier].
    inp_path     : Path to the SWMM .inp file (topology + coordinates).
    output_path  : Destination PNG path.
    title        : Figure title (two lines separated by \\n are rendered as subtitle).
    vmax_global  : Color scale maximum. If None, uses the max in node_data.
                   Pass the global max across all runs for cross-run comparability.

    Returns
    -------
    Path to the saved PNG.
    """
    inp_path = Path(inp_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)

    # align node_data to coords (drop nodes without coordinates)
    df = node_data.copy()
    df["node_id"] = df["node_id"].astype(str)
    df = df[df["node_id"].isin(coords)].reset_index(drop=True)

    df["x"] = df["node_id"].map(lambda n: coords[n][0])
    df["y"] = df["node_id"].map(lambda n: coords[n][1])

    vmax = vmax_global if vmax_global is not None else float(df["peak_flooding_lps"].max())
    if vmax == 0:
        vmax = 1.0  # avoid degenerate colormap

    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap(COLORMAP)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # ── pipes ────────────────────────────────────────────────────────────────
    for _lid, frm, to in conduits:
        if frm not in coords or to not in coords:
            continue
        x0, y0 = coords[frm]
        x1, y1 = coords[to]
        ax.plot([x0, x1], [y0, y1], color=PIPE_COLOR, lw=PIPE_LW, zorder=1)

    # ── dry nodes ────────────────────────────────────────────────────────────
    dry = df[df["peak_flooding_lps"] <= 0]
    ax.scatter(
        dry["x"], dry["y"],
        s=NODE_DRY_SIZE,
        color=NODE_DRY_COLOR,
        edgecolors=NODE_DRY_EDGE,
        linewidths=0.5,
        zorder=2,
        label="Nodos sin inundación",
    )

    # ── flooded nodes ────────────────────────────────────────────────────────
    wet = df[df["peak_flooding_lps"] > 0].copy()
    if not wet.empty:
        sizes = _scale_sizes(wet["peak_flooding_lps"], vmax)
        colors = cmap(norm(wet["peak_flooding_lps"].to_numpy()))
        sc = ax.scatter(
            wet["x"], wet["y"],
            s=sizes,
            c=wet["peak_flooding_lps"].to_numpy(),
            cmap=COLORMAP,
            norm=norm,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label="Nodos inundados",
        )
        # colorbar
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Caudal pico de inundación (lps)", fontsize=10)
    else:
        # still render a colorbar even if nothing flooded
        sm = cm.ScalarMappable(norm=norm, cmap=COLORMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Caudal pico de inundación (lps)", fontsize=10)

    # ── top-N annotations ────────────────────────────────────────────────────
    top = df.nlargest(TOP_N_LABELS, "peak_flooding_lps")
    top = top[top["peak_flooding_lps"] > 0]
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['node_id']}\n{row['peak_flooding_lps']:.1f} lps",
            xy=(row["x"], row["y"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7.5,
            bbox=ANNOTATION_BBOX,
            zorder=5,
        )

    # ── legend + labels ──────────────────────────────────────────────────────
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=0.8)
    ax.set_xlabel("Coordenada X (m)", fontsize=10)
    ax.set_ylabel("Coordenada Y (m)", fontsize=10)

    # Two-line title: bold suptitle + italic ax title
    lines = title.split("\n", 1)
    fig.suptitle(lines[0], fontsize=12, fontweight="bold", y=1.02)
    if len(lines) > 1:
        ax.set_title(lines[1], fontsize=10, style="italic", pad=6)

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path
