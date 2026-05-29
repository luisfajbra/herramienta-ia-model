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


def _draw_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    """Draw a small directional arrow at the midpoint of a pipe segment."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return
    # Unit vector scaled to a small fraction of the segment
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
    """
    Generate and save the network topology map.

    Parameters
    ----------
    inp_path    : Path to the SWMM .inp file.
    output_path : Destination PNG path.
    title       : Optional figure title.

    Returns
    -------
    Path to the saved PNG.
    """
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

    # nodes
    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    ax.scatter(xs, ys, s=NODE_SIZE, color=COLOR_NODE, zorder=4, linewidths=0.4, edgecolors="#5A7A92")

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR_INITIAL, label="Tubería inicial (nodo hoja)"),
        mpatches.Patch(color=COLOR_CONTINUOUS, label="Tubería continua"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_xlabel("Coordenada X (m)", fontsize=10)
    ax.set_ylabel("Coordenada Y (m)", fontsize=10)
    ax.set_title(title or f"Topología de red — {inp_path.stem}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path
