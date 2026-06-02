import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def plot_resilience_curve(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot resilience vs factor for SWMM data and ML predictions.

    df must have columns: factor, resilience_swmm, resilience_ml.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["factor"], df["resilience_swmm"],
        color="#2176ae", marker="o", linewidth=2, label="SWMM (real)",
    )
    ax.plot(
        df["factor"], df["resilience_ml"],
        color="#e07b39", marker="s", linewidth=2, label="Predicción ML",
    )

    ax.set_xlabel("Factor multiplicador de caudal")
    ax.set_ylabel("Resiliencia (fracción de nodos no inundados)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Curva de resiliencia de la red")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curva de resiliencia guardada: {output_path}")
    return output_path
