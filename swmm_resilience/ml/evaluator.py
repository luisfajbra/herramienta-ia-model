import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from ..config import Config
from .trainer import FEATURE_COLS, make_classifier, make_regressor


def _nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _avg(lst: list, key: str) -> float:
    vals = [d[key] for d in lst if not np.isnan(d.get(key, float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_metrics(lst: list) -> dict:
    if not lst:
        return {}
    return {k: _avg(lst, k) for k in lst[0]}


def _regressor_oracle_metrics(y_true, y_pred) -> dict:
    return {
        "nse": _nse(y_true, y_pred),
        "log_nse": _nse(np.log1p(y_true), np.log1p(y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _run_cv(df: pd.DataFrame, config: Config, cv) -> dict:
    X = df[FEATURE_COLS].values
    y_clf = df["inunda"].values
    y_reg = df["vol_inundacion_m3"].values
    groups = df["factor_mult"].values

    clf_m, reg_m, e2e_m = [], [], []
    by_factor: dict = {}

    for train_idx, test_idx in cv.split(X, y_clf, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
        yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]
        g_te = groups[test_idx]

        n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = make_classifier(config, spw)
        clf.fit(X_tr, yc_tr)

        reg = make_regressor(config)
        flooded_tr = yc_tr == 1
        if flooded_tr.sum() > 0:
            reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))

        # Level 1 — classifier
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]
        has_both_classes = yc_te.sum() > 0 and (1 - yc_te).sum() > 0
        clf_m.append({
            "precision": float(precision_score(yc_te, yc_pred, zero_division=0)),
            "recall": float(recall_score(yc_te, yc_pred, zero_division=0)),
            "f1": float(f1_score(yc_te, yc_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(yc_te, yc_prob)) if has_both_classes else float("nan"),
        })

        # Level 2 — regressor oracle (true labels used to filter)
        flooded_te = yc_te == 1
        if flooded_te.sum() > 0:
            yr_pred_oracle = np.expm1(reg.predict(X_te[flooded_te]))
            yr_pred_oracle = np.clip(yr_pred_oracle, a_min=0.0, a_max=None)
            yr_true_oracle = yr_te[flooded_te]
            reg_m.append(_regressor_oracle_metrics(yr_true_oracle, yr_pred_oracle))

        # Level 3 — end-to-end (predicted labels used to route to regressor)
        yr_pred_e2e = np.zeros(len(X_te))
        clf_flood_mask = yc_pred == 1
        if clf_flood_mask.sum() > 0:
            yr_pred_e2e[clf_flood_mask] = np.expm1(reg.predict(X_te[clf_flood_mask]))
            yr_pred_e2e = np.clip(yr_pred_e2e, a_min=0.0, a_max=None)
        e2e_m.append({
            "pct_nodos_correctos": float((yc_pred == yc_te).mean()),
            "rmse_vol_todos_nodos": float(np.sqrt(mean_squared_error(yr_te, yr_pred_e2e))),
            "vol_total_pred_m3": float(yr_pred_e2e.sum()),
            "vol_total_real_m3": float(yr_te.sum()),
        })

        # Stratify by factor
        if config.evaluation.stratify_by_factor:
            for fv in np.unique(g_te):
                fmask = g_te == fv
                fkey = f"{fv:.2f}"
                if fkey not in by_factor:
                    by_factor[fkey] = []
                by_factor[fkey].append({
                    "f1": float(f1_score(yc_te[fmask], yc_pred[fmask], zero_division=0)),
                    "rmse_vol": float(np.sqrt(mean_squared_error(yr_te[fmask], yr_pred_e2e[fmask]))),
                })

    result = {
        "classifier": _mean_metrics(clf_m),
        "regressor_oracle": _mean_metrics(reg_m),
        "end_to_end": _mean_metrics(e2e_m),
    }
    if config.evaluation.stratify_by_factor:
        result["by_factor"] = {k: _mean_metrics(v) for k, v in by_factor.items()}
    return result


def evaluate_models(df: pd.DataFrame, config: Config, output_dir: Path) -> dict:
    """Run LOSO and/or GroupKFold5 evaluation at 3 levels. Saves 4 JSON files.

    Oracle note: Level 2 regressor uses TRUE labels to filter flooded test rows.
    This is an optimistic upper bound — Level 3 end-to-end uses PREDICTED labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for method in config.evaluation.methods:
        print(f"  Evaluando {method}...")
        cv = LeaveOneGroupOut() if method == "LOSO" else GroupKFold(n_splits=5)
        all_results[method] = _run_cv(df, config, cv)

    def _save(data: dict, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    primary = all_results.get("LOSO", next(iter(all_results.values())))
    _save(primary.get("classifier", {}), output_dir / "metrics_classifier.json")
    _save(primary.get("regressor_oracle", {}), output_dir / "metrics_regressor.json")
    _save(primary.get("end_to_end", {}), output_dir / "metrics_endtoend.json")
    _save(
        {m: v.get("by_factor", {}) for m, v in all_results.items()},
        output_dir / "metrics_by_factor.json",
    )

    return all_results
