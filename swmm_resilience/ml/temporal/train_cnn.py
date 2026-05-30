"""CNN 1D training pipeline for SWMM temporal datasets."""
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
from .dataset import build_temporal_windows
from .models.cnn import SWMMTemporalCNN
from .schemas import TemporalWindowDataset

_TASK_TO_PREFIX = {
    "classification": "cnn_classifier",
    "regression": "cnn_regressor",
}


def train_cnn(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    task: str = "classification",
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_cv_folds: int = 5,
    device: str = "cpu",
    _dataset: TemporalWindowDataset | None = None,
) -> dict:
    """Train CNN 1D with GroupKFold cross-validation. Returns per-fold metrics.

    Args:
        db_path: SQLite database with temporal_artifacts table.
        networks_dir: Root directory for network files.
        artifacts_dir: Directory where model weights and scalers are saved.
        task: 'classification' (BCELoss) or 'regression' (MSELoss).
        n_epochs: Training epochs per fold.
        batch_size: Mini-batch size.
        lr: AdamW initial learning rate.
        n_cv_folds: Number of GroupKFold splits (reduced if fewer unique groups).
        device: PyTorch device string ('cpu', 'cuda', 'mps').
        _dataset: Inject a pre-built dataset (testing only; skips DB query).

    Returns:
        dict with keys: task, n_folds, folds (list of per-fold dicts), best_fold.
    """
    if task not in _TASK_TO_PREFIX:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    dataset = _dataset if _dataset is not None else build_temporal_windows(db_path, networks_dir)

    if dataset.X_seq.shape[0] == 0:
        raise ValueError("Dataset is empty — no temporal windows found.")

    groups = dataset.groups
    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)
    actual_folds = min(n_cv_folds, n_unique)
    if actual_folds < n_cv_folds:
        warnings.warn(
            f"Only {n_unique} unique run_id groups; reducing n_cv_folds to {actual_folds}.",
            stacklevel=2,
        )

    gkf = GroupKFold(n_splits=actual_folds)
    prefix = _TASK_TO_PREFIX[task]
    dev = torch.device(device)

    fold_results: list[dict] = []
    best_fold_idx = 0
    best_val_loss = float("inf")
    best_state_dict: dict | None = None
    best_scaler_seq: StandardScaler | None = None
    best_scaler_static: StandardScaler | None = None

    indices = np.arange(dataset.X_seq.shape[0])

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        X_seq_tr = dataset.X_seq[train_idx]
        X_seq_val = dataset.X_seq[val_idx]
        X_static_tr = dataset.X_static[train_idx]
        X_static_val = dataset.X_static[val_idx]

        if task == "classification":
            y_tr = dataset.y_class[train_idx].astype(np.float32)
            y_val_arr = dataset.y_class[val_idx].astype(np.float32)
        else:
            y_tr = dataset.y_reg[train_idx]
            y_val_arr = dataset.y_reg[val_idx]

        # Fit scalers on training data only
        N_tr, T, F = X_seq_tr.shape
        scaler_seq = StandardScaler()
        X_seq_tr_sc = scaler_seq.fit_transform(X_seq_tr.reshape(-1, F)).reshape(N_tr, T, F)
        X_seq_val_sc = scaler_seq.transform(X_seq_val.reshape(-1, F)).reshape(
            X_seq_val.shape[0], T, F
        )

        scaler_static = StandardScaler()
        X_static_tr_sc = scaler_static.fit_transform(X_static_tr)
        X_static_val_sc = scaler_static.transform(X_static_val)

        train_ds = TensorDataset(
            torch.tensor(X_seq_tr_sc, dtype=torch.float32),
            torch.tensor(X_static_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = SWMMTemporalCNN(
            n_temporal_features=F,
            n_static_features=X_static_tr.shape[1],
            task=task,
        ).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion: nn.Module = nn.BCELoss() if task == "classification" else nn.MSELoss()

        for _ in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_samples = 0
            for x_seq_b, x_static_b, y_b in loader:
                x_seq_b = x_seq_b.to(dev)
                x_static_b = x_static_b.to(dev)
                y_b = y_b.to(dev)
                optimizer.zero_grad()
                out = model(x_seq_b, x_static_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_b)
                n_samples += len(y_b)
            scheduler.step(epoch_loss / max(n_samples, 1))

        # Evaluate
        model.eval()
        with torch.no_grad():
            x_seq_v = torch.tensor(X_seq_val_sc, dtype=torch.float32).to(dev)
            x_static_v = torch.tensor(X_static_val_sc, dtype=torch.float32).to(dev)
            raw_out = model(x_seq_v, x_static_v).cpu().numpy().flatten()

        train_groups_list = sorted(set(groups[train_idx].tolist()))
        val_groups_list = sorted(set(groups[val_idx].tolist()))

        if task == "classification":
            preds = (raw_out >= 0.5).astype(int)
            val_loss = float(
                criterion(
                    torch.tensor(raw_out, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1),
                ).item()
            )
            try:
                val_auc = float(roc_auc_score(y_val_arr, raw_out))
            except ValueError:
                val_auc = float("nan")
            fold_result = {
                "fold": fold_i,
                "train_groups": train_groups_list,
                "val_groups": val_groups_list,
                "val_loss": val_loss,
                "val_accuracy": float(accuracy_score(y_val_arr, preds)),
                "val_f1": float(f1_score(y_val_arr, preds, zero_division=0)),
                "val_precision": float(precision_score(y_val_arr, preds, zero_division=0)),
                "val_recall": float(recall_score(y_val_arr, preds, zero_division=0)),
                "val_auc_roc": val_auc,
            }
        else:
            val_loss = float(mean_squared_error(y_val_arr, raw_out))
            try:
                val_r2 = float(r2_score(y_val_arr, raw_out))
            except ValueError:
                val_r2 = float("nan")
            fold_result = {
                "fold": fold_i,
                "train_groups": train_groups_list,
                "val_groups": val_groups_list,
                "val_loss": val_loss,
                "val_mae": float(mean_absolute_error(y_val_arr, raw_out)),
                "val_rmse": float(val_loss ** 0.5),
                "val_r2": val_r2,
            }

        fold_results.append(fold_result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fold_idx = fold_i
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_scaler_seq = scaler_seq
            best_scaler_static = scaler_static

    # Persist best fold artifacts
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state_dict, artifacts_dir / f"{prefix}_weights.pt")
    joblib.dump(best_scaler_seq, artifacts_dir / f"{prefix}_scaler_seq.joblib")
    joblib.dump(best_scaler_static, artifacts_dir / f"{prefix}_scaler_static.joblib")
    pd.DataFrame(fold_results).to_csv(artifacts_dir / f"{prefix}_metrics.csv", index=False)

    return {
        "task": task,
        "n_folds": actual_folds,
        "folds": fold_results,
        "best_fold": best_fold_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena CNN 1D temporal para clasificación o regresión.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "all"],
        default="classification",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tasks = ["classification", "regression"] if args.task == "all" else [args.task]
    for t in tasks:
        print(f"\n=== Entrenando {t} ===")
        result = train_cnn(
            task=t,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_cv_folds=args.folds,
            device=args.device,
        )
        print(f"Folds completados: {result['n_folds']}, mejor fold: {result['best_fold']}")
        for fold in result["folds"]:
            print(f"  Fold {fold['fold']}: val_loss={fold['val_loss']:.4f}")


if __name__ == "__main__":
    main()
