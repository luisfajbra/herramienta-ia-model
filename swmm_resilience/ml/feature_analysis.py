from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from .trainer import FEATURE_COLS, make_classifier, make_regressor
from ..visualization.labels import feature_display_name


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    """Save Pearson heatmap, Spearman heatmap, and feature-target bar chart."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df = df[FEATURE_COLS]
    display_names = [feature_display_name(f) for f in FEATURE_COLS]
    n = len(FEATURE_COLS)

    def _save_heatmap(mat: np.ndarray, title: str, filename: str) -> None:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        display_mat = np.where(mask, np.nan, mat)
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(display_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not mask[i, j] and abs(mat[i, j]) >= 0.3:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    pearson = feat_df.corr(method="pearson").to_numpy()
    spearman = feat_df.corr(method="spearman").to_numpy()

    _save_heatmap(pearson, "Pearson Correlation — Features", "correlation_pearson.png")
    _save_heatmap(spearman, "Spearman Correlation — Features", "correlation_spearman.png")

    rho_inunda = [df[f].corr(df["inunda"], method="spearman") for f in FEATURE_COLS]
    flooded = df[df["inunda"] == 1]
    rho_vol = [
        flooded[f].corr(flooded["vol_inundacion_m3"], method="spearman")
        for f in FEATURE_COLS
    ]

    order = np.argsort(np.abs(rho_inunda))
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(
        y - bar_h / 2,
        [rho_inunda[i] for i in order],
        bar_h,
        label="vs. Flooded (all rows)",
        color="steelblue",
    )
    ax.barh(
        y + bar_h / 2,
        [rho_vol[i] for i in order],
        bar_h,
        label="vs. Flood Volume (flooded only)",
        color="darkorange",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([display_names[i] for i in order], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_title("Feature–Target Spearman Correlation", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
