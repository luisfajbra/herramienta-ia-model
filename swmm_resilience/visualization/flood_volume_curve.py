import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def _plot_dual(fig, ax_lin, ax_log, x, y, color, marker):
    for ax, scale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.plot(x, y, color=color, marker=marker, linewidth=2)
        ax.set_xlabel("Factor multiplicador de caudal")
        ax.set_ylabel("Volumen total inundado (m³)")
        ax.set_yscale(scale)
        ax.set_title("Escala lineal" if scale == "linear" else "Escala logarítmica")
        ax.grid(True, alpha=0.3)


def plot_flood_volume_curve(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Generate two flood-volume PNGs — one for SWMM data, one for ML predictions.

    Each PNG contains two subplots: linear scale (left) and log scale (right).
    df must have columns: factor, vol_total_swmm, vol_total_ml.
    Returns (path_swmm, path_ml).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    path_swmm = output_dir / "flood_volume_swmm.png"
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_dual(fig, ax_lin, ax_log, df["factor"], df["vol_total_swmm"], "#2176ae", "o")
    fig.suptitle("Volumen total de inundación — SWMM (real)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_swmm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de volumen SWMM guardada: {path_swmm}")

    path_ml = output_dir / "flood_volume_ml.png"
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_dual(fig, ax_lin, ax_log, df["factor"], df["vol_total_ml"], "#e07b39", "s")
    fig.suptitle("Volumen total de inundación — Predicción ML", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_ml, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de volumen ML guardada: {path_ml}")

    return path_swmm, path_ml
