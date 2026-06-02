import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path

from ..simulation.swmm_api_io import load_inp

_INITIAL_COLOR = "#2176ae"    # blue — pipe whose from_node has no upstream input
_CONTINUOUS_COLOR = "#444444"  # dark gray — pipe whose from_node receives upstream flow


def generate_network_map(
    inp_path: Path,
    output_path: Path,
    network_name: str = "Red",
) -> Path:
    """Render network topology with pipe-type coloring and flow-direction arrows.

    Pipe classification:
      INITIAL   — from_node does NOT appear as any conduit's to_node (headwater)
      CONTINUOUS — from_node DOES appear as some conduit's to_node
    Outfall nodes (OUTFALLS section) are drawn as downward triangles.
    All junction nodes are drawn as small circles with their ID as label.
    Arrow at the midpoint of each conduit indicates flow direction.
    """
    inp = load_inp(inp_path)

    coords: dict[str, tuple[float, float]] = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = (float(c.x), float(c.y))

    conduits: dict[str, tuple[str, str]] = {}
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduits[str(lid)] = (str(c.from_node), str(c.to_node))

    outfalls: set[str] = set()
    if "OUTFALLS" in inp:
        for nid in inp["OUTFALLS"]:
            outfalls.add(str(nid))

    to_nodes = {tn for _, (_, tn) in conduits.items()}

    fig, ax = plt.subplots(figsize=(14, 12))

    for fn, tn in conduits.values():
        if fn not in coords or tn not in coords:
            continue
        x0, y0 = coords[fn]
        x1, y1 = coords[tn]
        color = _CONTINUOUS_COLOR
        lw = 1.0

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=1)

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) + abs(dy) > 1e-9:
            step = 0.01
            ax.annotate(
                "",
                xy=(mx + dx * step, my + dy * step),
                xytext=(mx - dx * step, my - dy * step),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, mutation_scale=16),
                zorder=2,
            )

    for nid, (x, y) in coords.items():
        if nid in outfalls:
            ax.scatter(x, y, marker="v", color="black", s=60, zorder=4)
        else:
            ax.scatter(x, y, color="black", s=8, zorder=3)

    legend_elements = [
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                   markersize=8, label="Nodo de salida"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=13,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_title(network_name, fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapa de red guardado: {output_path}")
    return output_path
