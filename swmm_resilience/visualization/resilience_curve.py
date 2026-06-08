import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def _plot_single(ax, x, y, color, marker, title):
    ax.plot(x, y, color=color, marker=marker, linewidth=2)
    ax.set_xlabel("Factor multiplicador de caudal")
    ax.set_ylabel("Resiliencia (fracción de nodos no inundados)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_resilience_curve(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Generate two resilience PNGs — one for SWMM data, one for ML predictions.

    df must have columns: factor, resilience_swmm, resilience_ml.
    Returns (path_swmm, path_ml).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    path_swmm = output_dir / "resilience_swmm.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_single(ax, df["factor"], df["resilience_swmm"],
                 "#2176ae", "o", "Curva de resiliencia — SWMM (real)")
    fig.savefig(path_swmm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de resiliencia SWMM guardada: {path_swmm}")

    path_ml = output_dir / "resilience_ml.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_single(ax, df["factor"], df["resilience_ml"],
                 "#e07b39", "s", "Curva de resiliencia — Predicción ML")
    fig.savefig(path_ml, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de resiliencia ML guardada: {path_ml}")

    return path_swmm, path_ml
