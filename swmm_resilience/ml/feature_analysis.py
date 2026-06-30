from __future__ import annotations

import json
from pathlib import Path

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


def run_ablation(df: pd.DataFrame, config, out_dir: Path) -> dict:
    """LOSO ablation: full 17 features vs. 15 features (no duracion/tiempo_al_pico).

    Returns the results dict and writes ablation_results.json + ablation_comparison.png.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced_cols = [f for f in FEATURE_COLS if f not in ("duracion_horas", "tiempo_al_pico_h")]

    def _loso_metrics(feature_cols: list) -> dict:
        X = df[feature_cols].values
        y_clf = df["inunda"].values
        y_reg = df["vol_inundacion_m3"].values
        groups = df["factor_mult"].values

        loso = LeaveOneGroupOut()
        f1_scores: list[float] = []
        auc_scores: list[float] = []
        reg_true_parts: list[np.ndarray] = []
        reg_pred_parts: list[np.ndarray] = []

        for train_idx, test_idx in loso.split(X, y_clf, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
            yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]

            n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            clf = make_classifier(config, spw)
            clf.fit(X_tr, yc_tr)

            reg = make_regressor(config)
            flooded_tr = yc_tr == 1
            if flooded_tr.sum() > 0:
                reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))

            yc_pred = clf.predict(X_te)
            yc_prob = clf.predict_proba(X_te)[:, 1]

            f1_scores.append(float(f1_score(yc_te, yc_pred, zero_division=0)))
            has_both = yc_te.sum() > 0 and (1 - yc_te).sum() > 0
            auc_scores.append(
                float(roc_auc_score(yc_te, yc_prob)) if has_both else float("nan")
            )

            flooded_te = yc_te == 1
            if flooded_te.sum() > 0:
                yr_pred = np.expm1(reg.predict(X_te[flooded_te]))
                yr_pred = np.clip(yr_pred, 0.0, None)
                reg_true_parts.append(yr_te[flooded_te])
                reg_pred_parts.append(yr_pred)

        f1_mean = float(np.nanmean(f1_scores))
        auc_mean = float(np.nanmean(auc_scores))

        if reg_true_parts:
            y_true_all = np.concatenate(reg_true_parts)
            y_pred_all = np.concatenate(reg_pred_parts)
            ss_res = float(np.sum((y_true_all - y_pred_all) ** 2))
            ss_tot = float(np.sum((y_true_all - np.mean(y_true_all)) ** 2))
            nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            r2 = float(r2_score(y_true_all, y_pred_all))
        else:
            nse = r2 = float("nan")

        return {
            "classifier": {"f1": f1_mean, "auc_roc": auc_mean},
            "regressor_oracle": {"nse": nse, "r2": r2},
        }

    results = {
        "full": _loso_metrics(FEATURE_COLS),
        "reduced": _loso_metrics(reduced_cols),
    }

    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    metric_labels = ["F1", "AUC-ROC", "NSE", "R²"]
    full_vals = [
        results["full"]["classifier"]["f1"],
        results["full"]["classifier"]["auc_roc"],
        results["full"]["regressor_oracle"]["nse"],
        results["full"]["regressor_oracle"]["r2"],
    ]
    reduced_vals = [
        results["reduced"]["classifier"]["f1"],
        results["reduced"]["classifier"]["auc_roc"],
        results["reduced"]["regressor_oracle"]["nse"],
        results["reduced"]["regressor_oracle"]["r2"],
    ]

    x = np.arange(len(metric_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_full = ax.bar(x - width / 2, full_vals, width, label="Full (17 features)", color="steelblue")
    bars_red = ax.bar(x + width / 2, reduced_vals, width, label="Reduced (15 features)", color="darkorange")
    for bar in list(bars_full) + list(bars_red):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Full vs. Reduced Feature Set (LOSO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def plot_shap(clf_pipeline, reg_pipeline, df: pd.DataFrame, out_dir: Path) -> None:
    """Save SHAP beeswarm summaries and dependence plots for both models.

    Regressor SHAP values are in log1p(vol m³) space — axis labels note this.
    clf_pipeline and reg_pipeline are trusted local artifacts (written by train_models).
    """
    import shap  # lazy import — only needed when this function is called

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    display_names = [feature_display_name(f) for f in FEATURE_COLS]

    # ── Classifier ───────────────────────────────────────────────────────────
    X_clf = clf_pipeline.named_steps["imputer"].transform(df[FEATURE_COLS])
    clf_explainer = shap.TreeExplainer(clf_pipeline.named_steps["model"])
    clf_shap_values = clf_explainer.shap_values(X_clf)
    # XGBoost binary classifier: shap_values is either a 2D array or a list
    # [neg_class, pos_class] depending on shap version — normalise to pos_class
    if isinstance(clf_shap_values, list):
        clf_shap_values = clf_shap_values[1]

    shap.summary_plot(clf_shap_values, X_clf, feature_names=display_names, show=False)
    plt.title("SHAP Summary — Classifier", fontsize=12)
    plt.savefig(out_dir / "shap_classifier_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feat in ("duracion_horas", "tiempo_al_pico_h"):
        feat_idx = FEATURE_COLS.index(feat)
        shap.dependence_plot(
            feat_idx, clf_shap_values, X_clf,
            feature_names=display_names, show=False,
        )
        plt.savefig(
            out_dir / f"shap_dependence_classifier_{feat}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # ── Regressor (flooded rows only — model trained on log1p(vol)) ──────────
    df_flooded = df[df["inunda"] == 1].reset_index(drop=True)
    if df_flooded.empty:
        raise RuntimeError(
            "plot_shap: no flooded rows in dataset (inunda==1) — "
            "regressor SHAP requires at least one flooded sample"
        )
    X_reg = reg_pipeline.named_steps["imputer"].transform(df_flooded[FEATURE_COLS])
    reg_explainer = shap.TreeExplainer(reg_pipeline.named_steps["model"])
    reg_shap_values = reg_explainer.shap_values(X_reg)
    if isinstance(reg_shap_values, list):
        reg_shap_values = reg_shap_values[1]

    shap.summary_plot(reg_shap_values, X_reg, feature_names=display_names, show=False)
    plt.title(
        "SHAP Summary — Regressor (values in log1p(vol m³) space)", fontsize=12
    )
    plt.savefig(out_dir / "shap_regressor_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feat in ("duracion_horas", "tiempo_al_pico_h"):
        feat_idx = FEATURE_COLS.index(feat)
        shap.dependence_plot(
            feat_idx, reg_shap_values, X_reg,
            feature_names=display_names, show=False,
        )
        plt.savefig(
            out_dir / f"shap_dependence_regressor_{feat}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
