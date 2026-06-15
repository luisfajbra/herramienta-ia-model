import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def _plot_dual(fig, ax_lin, ax_log, x, y, color, marker):
    for ax, scale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.plot(x, y, color=color, marker=marker, linewidth=2)
        ax.set_xlabel("Flow Multiplier")
        ax.set_ylabel("Total Flood Volume (m³)")
        ax.set_yscale(scale)
        ax.set_title("Linear Scale" if scale == "linear" else "Logarithmic Scale")
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
    fig.suptitle("Total Flood Volume - SWMM (Actual)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_swmm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SWMM flood volume curve saved: {path_swmm}")

    path_ml = output_dir / "flood_volume_ml.png"
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_dual(fig, ax_lin, ax_log, df["factor"], df["vol_total_ml"], "#e07b39", "s")
    fig.suptitle("Total Flood Volume - ML Prediction", fontsize=13)
    fig.tight_layout()
    fig.savefig(path_ml, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ML flood volume curve saved: {path_ml}")

    return path_swmm, path_ml


def plot_flood_volume_combined(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate a single PNG with SWMM and ML curves overlaid on the same axes.

    Two subplots: linear scale (left) and log scale (right).
    df must have columns: factor, vol_total_swmm, vol_total_ml.
    Returns path to the combined PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flood_volume_combined.png"

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.plot(df["factor"], df["vol_total_swmm"],
                color="#2176ae", marker="o", linewidth=2, label="SWMM (Actual)")
        ax.plot(df["factor"], df["vol_total_ml"],
                color="#e07b39", marker="s", linewidth=2, label="ML Prediction")
        ax.set_xlabel("Flow Multiplier")
        ax.set_ylabel("Total Flood Volume (m³)")
        ax.set_yscale(scale)
        ax.set_title("Linear Scale" if scale == "linear" else "Logarithmic Scale")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Total Flood Volume - SWMM vs ML", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined flood volume curve saved: {output_path}")
    return output_path
