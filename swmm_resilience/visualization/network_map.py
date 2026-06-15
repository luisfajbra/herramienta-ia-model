"""
Figure 1 — Network topology map.

Shows the pipe network from a SWMM .inp file with:
- Initial pipes (from leaf nodes) in one color
- Continuous pipes (from interior nodes) in another color
- Flow-direction arrows at midpoint of each pipe
- Nodes as uniform small dots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..simulation.swmm_api_io import load_inp
from ._inp_parser import classify_conduits, parse_conduits, parse_coordinates

# ── visual constants ─────────────────────────────────────────────────────────
COLOR_INITIAL = "#E07B39"      # warm orange — leaf / initial pipes
COLOR_CONTINUOUS = "#2C5F8A"   # steel blue  — interior / continuous pipes
COLOR_NODE = "#AABDCC"         # light blue-gray for all nodes
NODE_SIZE = 18
LINEWIDTH_INITIAL = 1.2
LINEWIDTH_CONTINUOUS = 1.0
ARROW_SCALE = 12               # annotation arrow size (points)
FIG_SIZE = (12, 10)
DPI = 150

_INITIAL_COLOR = "#2176ae"
_CONTINUOUS_COLOR = "#444444"


def _draw_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    """Draw a small directional arrow at the midpoint of a pipe segment."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length * length * 0.08
    uy = dy / length * length * 0.08
    ax.annotate(
        "",
        xy=(mx + ux, my + uy),
        xytext=(mx - ux, my - uy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=0.6,
            mutation_scale=ARROW_SCALE,
        ),
        zorder=3,
    )


def plot_network(
    inp_path: Path | str,
    output_path: Path | str,
    title: str | None = None,
) -> Path:
    """Generate and save the network topology map (experiment branch interface)."""
    inp_path = Path(inp_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)
    initial, continuous = classify_conduits(conduits)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    def _draw_pipes(pipes: list[tuple[str, str, str]], color: str, lw: float) -> None:
        for _lid, frm, to in pipes:
            if frm not in coords or to not in coords:
                continue
            x0, y0 = coords[frm]
            x1, y1 = coords[to]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2, solid_capstyle="round")
            _draw_arrow(ax, x0, y0, x1, y1, color)

    _draw_pipes(continuous, COLOR_CONTINUOUS, LINEWIDTH_CONTINUOUS)
    _draw_pipes(initial, COLOR_INITIAL, LINEWIDTH_INITIAL)

    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    ax.scatter(xs, ys, s=NODE_SIZE, color=COLOR_NODE, zorder=4, linewidths=0.4, edgecolors="#5A7A92")

    legend_handles = [
        mpatches.Patch(color=COLOR_INITIAL, label="Initial Pipe (Leaf Node)"),
        mpatches.Patch(color=COLOR_CONTINUOUS, label="Continuous Pipe"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_xlabel("X Coordinate (m)", fontsize=10)
    ax.set_ylabel("Y Coordinate (m)", fontsize=10)
    ax.set_title(title or f"Network Topology - {inp_path.stem}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_network_map(
    inp_path: Path,
    output_path: Path,
    network_name: str = "Network",
) -> Path:
    """Render network topology with pipe-type coloring and arrows (origin/main interface).

    Pipe classification:
      INITIAL   — from_node does NOT appear as any conduit's to_node (headwater)
      CONTINUOUS — from_node DOES appear as some conduit's to_node
    Outfall nodes (OUTFALLS section) are drawn as downward triangles.
    """
    inp = load_inp(inp_path)

    coords: dict[str, tuple[float, float]] = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = (float(c.x), float(c.y))

    conduits_dict: dict[str, tuple[str, str]] = {}
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduits_dict[str(lid)] = (str(c.from_node), str(c.to_node))

    outfalls: set[str] = set()
    if "OUTFALLS" in inp:
        for nid in inp["OUTFALLS"]:
            outfalls.add(str(nid))

    fig, ax = plt.subplots(figsize=(14, 12))

    for fn, tn in conduits_dict.values():
        if fn not in coords or tn not in coords:
            continue
        x0, y0 = coords[fn]
        x1, y1 = coords[tn]
        ax.plot([x0, x1], [y0, y1], color=_CONTINUOUS_COLOR, linewidth=1.0, zorder=1)

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) + abs(dy) > 1e-9:
            step = 0.01
            ax.annotate(
                "",
                xy=(mx + dx * step, my + dy * step),
                xytext=(mx - dx * step, my - dy * step),
                arrowprops=dict(arrowstyle="->", color=_CONTINUOUS_COLOR, lw=1.5, mutation_scale=16),
                zorder=2,
            )

    for nid, (x, y) in coords.items():
        if nid in outfalls:
            ax.scatter(x, y, marker="v", color="black", s=60, zorder=4)
        else:
            ax.scatter(x, y, color="black", s=8, zorder=3)

    legend_elements = [
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                   markersize=8, label="Outfall Node"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=13,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_title(network_name, fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Network map saved: {output_path}")
    return output_path
