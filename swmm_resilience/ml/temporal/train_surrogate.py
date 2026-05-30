# swmm_resilience/ml/temporal/train_surrogate.py
"""Surrogate CNN training pipeline for SWMM flood prediction."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ...config import DEFAULT_DB_FILE, DEFAULT_TEMPORAL_ARTIFACTS_DIR, NETWORKS_DIR
from .dataset import build_surrogate_dataset
from .models.surrogate_cnn import SWMMSurrogateCNN
from .schemas import TemporalWindowDataset


def train_surrogate(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_cv_folds: int = 5,
    alpha: float = 1.0,
    beta: float = 0.01,
    use_temporal: bool = True,
    device: str = "cpu",
    _dataset: TemporalWindowDataset | None = None,
) -> dict:
    """Train SWMMSurrogateCNN with GroupKFold. Returns per-fold metrics.

    Args:
        alpha: Weight for BCEWithLogitsLoss (classification).
        beta: Weight for MSELoss (regression).
        use_temporal: If False, trains the ablation model (no Conv branch,
            multiplier in static features). _dataset must have X_static with
            8 columns when use_temporal=False.
        _dataset: Inject pre-built dataset (testing only).
    """
    prefix = "surrogate_cnn" if use_temporal else "surrogate_cnn_notemporal"

    dataset = (
        _dataset
        if _dataset is not None
        else build_surrogate_dataset(db_path, networks_dir, use_temporal=use_temporal)
    )

    if dataset.X_seq.shape[0] == 0:
        raise ValueError("Dataset is empty — no surrogate samples found.")

    groups = dataset.groups
    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)
    actual_folds = min(n_cv_folds, n_unique)
    if actual_folds < n_cv_folds:
        warnings.warn(f"Only {n_unique} unique run groups; using {actual_folds} folds.", stacklevel=2)
    if actual_folds < 2:
        raise ValueError(f"GroupKFold needs at least 2 run groups; found {n_unique}.")

    gkf = GroupKFold(n_splits=actual_folds)
    dev = torch.device(device)
    indices = np.arange(dataset.X_seq.shape[0])

    fold_results: list[dict] = []
    best_fold_idx = 0
    best_val_loss = float("inf")
    best_state_dict: dict | None = None
    best_scaler_seq: StandardScaler | None = None
    best_scaler_static: StandardScaler | None = None

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        X_seq_tr = dataset.X_seq[train_idx]       # [N_tr, T, 6]
        X_seq_val = dataset.X_seq[val_idx]
        X_static_tr = dataset.X_static[train_idx]
        X_static_val = dataset.X_static[val_idx]
        y_cls_tr = dataset.y_class[train_idx].astype(np.float32)
        y_cls_val = dataset.y_class[val_idx].astype(np.float32)
        y_reg_tr = dataset.y_reg[train_idx]
        y_reg_val = dataset.y_reg[val_idx]

        # Scalers fitted on training fold only
        N_tr, T, F = X_seq_tr.shape
        scaler_seq = StandardScaler()
        X_seq_tr_sc = scaler_seq.fit_transform(X_seq_tr.reshape(-1, F)).reshape(N_tr, T, F)
        X_seq_val_sc = scaler_seq.transform(X_seq_val.reshape(-1, F)).reshape(
            X_seq_val.shape[0], T, F
        )
        scaler_static = StandardScaler()
        X_static_tr_sc = scaler_static.fit_transform(X_static_tr)
        X_static_val_sc = scaler_static.transform(X_static_val)

        # pos_weight for class imbalance
        n_pos = max(float(y_cls_tr.sum()), 1.0)
        n_neg = float(len(y_cls_tr)) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(dev)

        n_static_features = X_static_tr.shape[1]
        model = SWMMSurrogateCNN(
            n_temporal_features=F,
            n_static_features=n_static_features,
            use_temporal=use_temporal,
        ).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_reg = nn.MSELoss()

        train_ds = TensorDataset(
            torch.tensor(X_seq_tr_sc, dtype=torch.float32),
            torch.tensor(X_static_tr_sc, dtype=torch.float32),
            torch.tensor(y_cls_tr).unsqueeze(1),
            torch.tensor(y_reg_tr).unsqueeze(1),
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for _ in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_samples = 0
            for x_seq_b, x_static_b, y_cls_b, y_reg_b in loader:
                x_seq_b = x_seq_b.to(dev)
                x_static_b = x_static_b.to(dev)
                y_cls_b = y_cls_b.to(dev)
                y_reg_b = y_reg_b.to(dev)
                optimizer.zero_grad()
                x_seq_in = x_seq_b if use_temporal else None
                cls_logit, reg_out = model(x_seq_in, x_static_b)
                loss = alpha * criterion_cls(cls_logit, y_cls_b) + beta * criterion_reg(reg_out, y_reg_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_cls_b)
                n_samples += len(y_cls_b)
            scheduler.step(epoch_loss / max(n_samples, 1))

        # Evaluate
        model.eval()
        x_seq_v = torch.tensor(X_seq_val_sc, dtype=torch.float32).to(dev)
        x_static_v = torch.tensor(X_static_val_sc, dtype=torch.float32).to(dev)
        with torch.no_grad():
            x_seq_in = x_seq_v if use_temporal else None
            cls_logit_v, reg_out_v = model(x_seq_in, x_static_v)
            cls_prob = torch.sigmoid(cls_logit_v).cpu().numpy().flatten()
            reg_pred = reg_out_v.cpu().numpy().flatten()

        cls_pred = (cls_prob >= 0.5).astype(int)
        val_bce = float(
            criterion_cls(
                cls_logit_v.cpu(),
                torch.tensor(y_cls_val).unsqueeze(1),
            ).item()
        )
        try:
            val_auc = float(roc_auc_score(y_cls_val, cls_prob))
        except ValueError:
            val_auc = float("nan")

        val_rmse = float(mean_squared_error(y_reg_val, reg_pred) ** 0.5)
        try:
            val_r2 = float(r2_score(y_reg_val, reg_pred))
        except ValueError:
            val_r2 = float("nan")

        fold_result = {
            "fold": fold_i,
            "train_groups": sorted(set(groups[train_idx].tolist())),
            "val_groups": sorted(set(groups[val_idx].tolist())),
            "val_loss": val_bce,
            "val_auc_roc": val_auc,
            "val_accuracy": float(accuracy_score(y_cls_val, cls_pred)),
            "val_f1": float(f1_score(y_cls_val, cls_pred, zero_division=0)),
            "val_precision": float(precision_score(y_cls_val, cls_pred, zero_division=0)),
            "val_recall": float(recall_score(y_cls_val, cls_pred, zero_division=0)),
            "val_mae": float(mean_absolute_error(y_reg_val, reg_pred)),
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }
        fold_results.append(fold_result)

        if val_bce < best_val_loss:
            best_val_loss = val_bce
            best_fold_idx = fold_i
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_scaler_seq = scaler_seq
            best_scaler_static = scaler_static

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, artifacts_dir / f"{prefix}_weights.pt")
    joblib.dump(best_scaler_seq, artifacts_dir / f"{prefix}_scaler_seq.joblib")
    joblib.dump(best_scaler_static, artifacts_dir / f"{prefix}_scaler_static.joblib")
    pd.DataFrame(fold_results).to_csv(artifacts_dir / f"{prefix}_metrics.csv", index=False)

    return {"n_folds": actual_folds, "folds": fold_results, "best_fold": best_fold_idx}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SWMM surrogate CNN.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Ablation: disable temporal branch, use multiplier as static feature.",
    )
    args = parser.parse_args()

    use_temporal = not args.no_temporal
    label = "full" if use_temporal else "ablation (no temporal)"
    print(f"\n=== Training surrogate CNN [{label}] ===")
    result = train_surrogate(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_cv_folds=args.folds,
        device=args.device,
        use_temporal=use_temporal,
    )
    print(f"Folds: {result['n_folds']}, best fold: {result['best_fold']}")
    for fold in result["folds"]:
        print(
            f"  Fold {fold['fold']}: BCE={fold['val_loss']:.4f}  "
            f"AUC={fold['val_auc_roc']:.4f}  F1={fold['val_f1']:.4f}  "
            f"RMSE={fold['val_rmse']:.2f}"
        )


if __name__ == "__main__":
    main()
