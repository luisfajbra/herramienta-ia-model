# swmm_resilience/ml/temporal/compare_surrogate.py
"""Unified comparison: XGBoost vs CNN (full) vs CNN (ablation) — same data, same folds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must set before loading OpenMP-linked libs (xgboost) alongside torch's libiomp5.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

# Import xgboost before torch to avoid duplicate OpenMP runtime segfaults on macOS.
try:
    from xgboost import XGBClassifier
except (ImportError, Exception):
    XGBClassifier = None

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ...config import DEFAULT_DB_FILE, DEFAULT_OUTPUT_CSV, DEFAULT_TEMPORAL_ARTIFACTS_DIR
from .dataset import build_unified_dataset
from .models.surrogate_cnn import SWMMSurrogateCNN
from .models.surrogate_lstm import SWMMSurrogateLSTM
from .schemas import TemporalWindowDataset


def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "auc_roc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _train_eval_model(
    model_cls,
    X_seq_tr: np.ndarray, X_static_tr: np.ndarray,
    y_cls_tr: np.ndarray, y_reg_tr: np.ndarray,
    X_seq_val: np.ndarray, X_static_val: np.ndarray,
    y_cls_val: np.ndarray, y_reg_val: np.ndarray,
    use_temporal: bool,
    n_epochs: int, batch_size: int, lr: float,
    alpha: float, beta: float, device: str,
) -> dict:
    N_tr, T, F = X_seq_tr.shape
    dev = torch.device(device)

    scaler_seq = StandardScaler()
    X_seq_tr_sc = scaler_seq.fit_transform(X_seq_tr.reshape(-1, F)).reshape(N_tr, T, F)
    X_seq_val_sc = scaler_seq.transform(X_seq_val.reshape(-1, F)).reshape(X_seq_val.shape[0], T, F)

    scaler_static = StandardScaler()
    X_static_tr_sc = scaler_static.fit_transform(X_static_tr)
    X_static_val_sc = scaler_static.transform(X_static_val)

    n_pos = max(float(y_cls_tr.sum()), 1.0)
    n_neg = float(len(y_cls_tr)) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(dev)

    model = model_cls(
        n_temporal_features=F,
        n_static_features=X_static_tr.shape[1],
        use_temporal=use_temporal,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_reg = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_seq_tr_sc, dtype=torch.float32),
            torch.tensor(X_static_tr_sc, dtype=torch.float32),
            torch.tensor(y_cls_tr.astype(np.float32)).unsqueeze(1),
            torch.tensor(y_reg_tr).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    for _ in range(n_epochs):
        model.train()
        epoch_loss, n_samples = 0.0, 0
        for x_seq_b, x_static_b, y_cls_b, y_reg_b in loader:
            x_seq_b, x_static_b = x_seq_b.to(dev), x_static_b.to(dev)
            y_cls_b, y_reg_b = y_cls_b.to(dev), y_reg_b.to(dev)
            optimizer.zero_grad()
            cls_logit, reg_out = model(x_seq_b if use_temporal else None, x_static_b)
            loss = alpha * criterion_cls(cls_logit, y_cls_b) + beta * criterion_reg(reg_out, y_reg_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_cls_b)
            n_samples += len(y_cls_b)
        scheduler.step(epoch_loss / max(n_samples, 1))

    model.eval()
    with torch.no_grad():
        cls_logit_v, reg_out_v = model(
            torch.tensor(X_seq_val_sc, dtype=torch.float32).to(dev) if use_temporal else None,
            torch.tensor(X_static_val_sc, dtype=torch.float32).to(dev),
        )
        cls_prob = torch.sigmoid(cls_logit_v).cpu().numpy().flatten()
        reg_pred = reg_out_v.cpu().numpy().flatten()

    cls_pred = (cls_prob >= 0.5).astype(int)
    metrics = _cls_metrics(y_cls_val, cls_pred, cls_prob)
    metrics["rmse"] = float(mean_squared_error(y_reg_val, reg_pred) ** 0.5)
    metrics["mae"] = float(mean_absolute_error(y_reg_val, reg_pred))
    return metrics


def compare_surrogate(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    db_path: Path = DEFAULT_DB_FILE,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_cv_folds: int = 5,
    alpha: float = 1.0,
    beta: float = 0.01,
    device: str = "cpu",
    _dataset: TemporalWindowDataset | None = None,
) -> pd.DataFrame:
    """Train XGBoost, CNN (full), and CNN (ablation) on identical GroupKFold splits.

    All models use only inference-available features — no SWMM outputs.
    Returns DataFrame with per-fold metrics for all three models.
    """
    if XGBClassifier is None:
        raise ImportError("xgboost is required: pip install xgboost")

    dataset = _dataset if _dataset is not None else build_unified_dataset(csv_path, db_path)

    groups = dataset.groups
    indices = np.arange(len(groups))
    actual_folds = min(n_cv_folds, len(np.unique(groups)))
    if actual_folds < 2:
        raise ValueError(f"Need at least 2 run groups; found {len(np.unique(groups))}.")

    gkf = GroupKFold(n_splits=actual_folds)
    fold_rows: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        X_static_tr = dataset.X_static[train_idx]
        X_static_val = dataset.X_static[val_idx]
        X_seq_tr = dataset.X_seq[train_idx]
        X_seq_val = dataset.X_seq[val_idx]
        y_cls_tr = dataset.y_class[train_idx]
        y_cls_val = dataset.y_class[val_idx]
        y_reg_tr = dataset.y_reg[train_idx]
        y_reg_val = dataset.y_reg[val_idx]

        row: dict = {
            "fold": fold_i,
            "train_groups": sorted(set(groups[train_idx].tolist())),
            "val_groups": sorted(set(groups[val_idx].tolist())),
        }

        # ── XGBoost ───────────────────────────────────────────────────────────
        n_pos = max(float(y_cls_tr.sum()), 1.0)
        n_neg = float(len(y_cls_tr)) - n_pos
        scaler_xgb = StandardScaler()
        Xtr_sc = scaler_xgb.fit_transform(X_static_tr)
        Xval_sc = scaler_xgb.transform(X_static_val)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss", random_state=42, verbosity=0,
            nthread=1,  # avoid multi-OpenMP conflict with torch on macOS
        )
        xgb.fit(Xtr_sc, y_cls_tr.astype(int))
        xgb_prob = xgb.predict_proba(Xval_sc)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        for k, v in _cls_metrics(y_cls_val, xgb_pred, xgb_prob).items():
            row[f"xgb_{k}"] = v

        # ── CNN full ─────────────────────────────────────────────────────────
        cnn_m = _train_eval_model(
            SWMMSurrogateCNN,
            X_seq_tr, X_static_tr, y_cls_tr, y_reg_tr,
            X_seq_val, X_static_val, y_cls_val, y_reg_val,
            use_temporal=True,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, device=device,
        )
        for k, v in cnn_m.items():
            row[f"cnn_{k}"] = v

        # ── CNN ablation ──────────────────────────────────────────────────────
        abl_m = _train_eval_model(
            SWMMSurrogateCNN,
            X_seq_tr, X_static_tr, y_cls_tr, y_reg_tr,
            X_seq_val, X_static_val, y_cls_val, y_reg_val,
            use_temporal=False,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, device=device,
        )
        for k, v in abl_m.items():
            row[f"cnn_abl_{k}"] = v

        # ── LSTM ─────────────────────────────────────────────────────────────
        lstm_m = _train_eval_model(
            SWMMSurrogateLSTM,
            X_seq_tr, X_static_tr, y_cls_tr, y_reg_tr,
            X_seq_val, X_static_val, y_cls_val, y_reg_val,
            use_temporal=True,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, device=device,
        )
        for k, v in lstm_m.items():
            row[f"lstm_{k}"] = v

        fold_rows.append(row)
        print(
            f"Fold {fold_i}: XGB F1={row['xgb_f1']:.3f}  "
            f"CNN F1={row['cnn_f1']:.3f}  LSTM F1={row['lstm_f1']:.3f}  "
            f"Ablation F1={row['cnn_abl_f1']:.3f}"
        )

    results_df = pd.DataFrame(fold_rows)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(artifacts_dir / "comparison_results.csv", index=False)
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare XGBoost vs CNN on unified surrogate dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("\n=== Unified Surrogate Comparison ===")
    print(f"Epochs: {args.epochs}  Folds: {args.folds}  Device: {args.device}")

    results = compare_surrogate(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_cv_folds=args.folds,
        device=args.device,
    )

    metrics = ["auc_roc", "f1", "precision", "recall", "accuracy"]
    print(f"\n{'Metric':<14} {'XGBoost':>10} {'CNN Full':>10} {'LSTM':>10} {'CNN Ablation':>14}")
    print("-" * 62)
    for m in metrics:
        xgb = results[f"xgb_{m}"].mean()
        cnn = results[f"cnn_{m}"].mean()
        lstm = results[f"lstm_{m}"].mean()
        abl = results[f"cnn_abl_{m}"].mean()
        print(f"  {m:<12} {xgb:>10.4f} {cnn:>10.4f} {lstm:>10.4f} {abl:>14.4f}")

    print(f"\nResults saved to: comparison_results.csv")


if __name__ == "__main__":
    main()
