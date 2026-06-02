from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ..simulation.swmm_api_io import load_inp


def generate_flood_map(
    inp_path: Path,
    vol_data: pd.DataFrame,
    factor: float,
    output_path: Path,
    network_name: str = "Red",
    colormap: str = "RdYlBu_r",
    show_labels_top_n: int = 5,
):
    """Render a network map with flood-volume gradient.

    vol_data must contain node_id and either vol_inundacion_m3 or vol_pred_m3.
    Nodes with duplicate coordinates receive a ±2m X offset for visual separation.
    """
    inp = load_inp(inp_path)

    coords: dict = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = [float(c.x), float(c.y)]

    conduits = []
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            fn, tn = str(c.from_node), str(c.to_node)
            if fn in coords and tn in coords:
                conduits.append((fn, tn))

    # Offset duplicate-coordinate node pairs
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

    for fn, tn in conduits:
        ax.plot(
            [coords[fn][0], coords[tn][0]],
            [coords[fn][1], coords[tn][1]],
            color="#888888", linewidth=0.7, zorder=1,
        )

    vols = np.array([max(0.0, node_vol.get(nid, 0.0)) for nid in coords])
    max_vol = vols.max() if vols.max() > 0 else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=max_vol)
    cmap = plt.get_cmap(colormap)

    for nid, (x, y) in coords.items():
        vol = max(0.0, node_vol.get(nid, 0.0))
        if vol > 0:
            ax.scatter(x, y, c=[cmap(norm(vol))], s=40 + (vol / max_vol) * 180,
                       zorder=3, edgecolors="none")
        else:
            ax.scatter(x, y, color="#aec6e8", s=15, zorder=2)

    sorted_nids = sorted(coords, key=lambda n: node_vol.get(n, 0.0), reverse=True)
    for nid in sorted_nids[:show_labels_top_n]:
        vol = node_vol.get(nid, 0.0)
        if vol > 0:
            x, y = coords[nid]
            ax.annotate(
                f"{nid}\n{vol:.1f} m³", (x, y),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Volumen de inundación (m³)", shrink=0.7)
    ax.set_title(f"{network_name} — Factor de escala: {factor:.2f}", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa guardado: {output_path}")
    return output_path
