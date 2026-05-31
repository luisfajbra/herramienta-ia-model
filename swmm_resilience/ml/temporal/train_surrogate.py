# swmm_resilience/ml/temporal/train_surrogate.py
"""Surrogate model training pipeline (CNN/LSTM) for SWMM flood prediction."""
from __future__ import annotations

import argparse
import json
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from .models.surrogate_lstm import SWMMSurrogateLSTM
from .schemas import TemporalWindowDataset


_MODEL_PREFIXES: dict[type, str] = {
    SWMMSurrogateCNN: "surrogate_cnn",
    SWMMSurrogateLSTM: "surrogate_lstm",
}

DEFAULT_SURROGATE_SEED = 42


def _set_torch_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _feature_names(
    dataset: TemporalWindowDataset,
    fallback_temporal: list[str],
    fallback_static_count: int,
) -> tuple[list[str], list[str]]:
    temporal = list(dataset.meta.attrs.get("temporal_feature_names", fallback_temporal))
    static = list(
        dataset.meta.attrs.get(
            "static_feature_names",
            [f"static_{i}" for i in range(fallback_static_count)],
        )
    )
    return temporal, static


def _model_type(model_cls: type) -> str:
    if model_cls is SWMMSurrogateCNN:
        return "cnn"
    if model_cls is SWMMSurrogateLSTM:
        return "lstm"
    return model_cls.__name__


def _multiplier_range(dataset: TemporalWindowDataset, static_feature_names: list[str]) -> dict[str, float]:
    if "inflow_multiplier" in dataset.meta.columns:
        values = pd.to_numeric(dataset.meta["inflow_multiplier"], errors="coerce").dropna()
    elif "inflow_multiplier" in static_feature_names:
        col_idx = static_feature_names.index("inflow_multiplier")
        values = pd.Series(dataset.X_static[:, col_idx])
    else:
        return {}

    if values.empty:
        return {}
    return {
        "min_multiplier": float(values.min()),
        "max_multiplier": float(values.max()),
    }


def _write_surrogate_manifest(
    artifacts_dir: Path,
    *,
    prefix: str,
    model_type: str,
    seed: int,
    dataset: TemporalWindowDataset,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
    use_temporal: bool,
) -> Path:
    run_ids = sorted({str(group) for group in dataset.groups.tolist()})
    manifest: dict[str, Any] = {
        "model_type": model_type,
        "prefix": prefix,
        "seed": int(seed),
        "use_temporal": bool(use_temporal),
        "trained_run_ids": run_ids,
        "trained_rows": int(dataset.X_seq.shape[0]),
        "temporal_feature_names": list(temporal_feature_names),
        "static_feature_names": list(static_feature_names),
        "regression_target": "peak_flooding_lps",
        "regression_target_transform": "log1p",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(_multiplier_range(dataset, static_feature_names))
    path = artifacts_dir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def _fit_surrogate_final_model(
    *,
    dataset: TemporalWindowDataset,
    model_cls: type,
    use_temporal: bool,
    n_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    beta: float,
    device: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], StandardScaler, StandardScaler]:
    dev = torch.device(device)
    X_seq = dataset.X_seq
    X_static = dataset.X_static
    y_cls = dataset.y_class.astype(np.float32)
    y_reg_log = np.log1p(dataset.y_reg.astype(np.float32))

    n_rows, timesteps, n_temporal_features = X_seq.shape
    scaler_seq = StandardScaler()
    X_seq_sc = scaler_seq.fit_transform(X_seq.reshape(-1, n_temporal_features)).reshape(
        n_rows, timesteps, n_temporal_features
    )
    scaler_static = StandardScaler()
    X_static_sc = scaler_static.fit_transform(X_static)

    n_pos = max(float(y_cls.sum()), 1.0)
    n_neg = float(len(y_cls)) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(dev)

    model = model_cls(
        n_temporal_features=n_temporal_features,
        n_static_features=X_static.shape[1],
        use_temporal=use_temporal,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_reg = nn.HuberLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_seq_sc, dtype=torch.float32),
            torch.tensor(X_static_sc, dtype=torch.float32),
            torch.tensor(y_cls).unsqueeze(1),
            torch.tensor(y_reg_log).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    for _ in range(n_epochs):
        model.train()
        for x_seq_b, x_static_b, y_cls_b, y_reg_b in loader:
            x_seq_b = x_seq_b.to(dev)
            x_static_b = x_static_b.to(dev)
            y_cls_b = y_cls_b.to(dev)
            y_reg_b = y_reg_b.to(dev)
            optimizer.zero_grad()
            cls_logit, reg_out = model(x_seq_b if use_temporal else None, x_static_b)
            loss = alpha * criterion_cls(cls_logit, y_cls_b) + beta * criterion_reg(reg_out, y_reg_b)
            loss.backward()
            optimizer.step()

    return {key: value.cpu().clone() for key, value in model.state_dict().items()}, scaler_seq, scaler_static


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
    model_cls: type = SWMMSurrogateCNN,
    device: str = "cpu",
    _dataset: TemporalWindowDataset | None = None,
) -> dict:
    """Train surrogate model (CNN/LSTM) with GroupKFold. Returns per-fold metrics.

    Args:
        alpha: Weight for BCEWithLogitsLoss (classification).
        beta: Weight for MSELoss (regression).
        use_temporal: If False, trains the ablation model (no Conv branch,
            multiplier in static features). _dataset must have X_static with
            8 columns when use_temporal=False.
        _dataset: Inject pre-built dataset (testing only).
    """
    seed = DEFAULT_SURROGATE_SEED
    _set_torch_determinism(seed)

    prefix = _MODEL_PREFIXES[model_cls]
    if not use_temporal:
        prefix += "_notemporal"

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

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        X_seq_tr = dataset.X_seq[train_idx]       # [N_tr, T, 6]
        X_seq_val = dataset.X_seq[val_idx]
        X_static_tr = dataset.X_static[train_idx]
        X_static_val = dataset.X_static[val_idx]
        y_cls_tr = dataset.y_class[train_idx].astype(np.float32)
        y_cls_val = dataset.y_class[val_idx].astype(np.float32)
        y_reg_tr = np.log1p(dataset.y_reg[train_idx].astype(np.float32))
        y_reg_val_raw = dataset.y_reg[val_idx]
        y_reg_val = np.log1p(y_reg_val_raw.astype(np.float32))

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
        model = model_cls(
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
        generator = torch.Generator()
        generator.manual_seed(seed + fold_i)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

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
            reg_pred_log = reg_out_v.cpu().numpy().flatten()
            reg_pred = np.expm1(reg_pred_log).clip(min=0.0)

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

        val_rmse = float(mean_squared_error(y_reg_val_raw, reg_pred) ** 0.5)
        try:
            val_r2 = float(r2_score(y_reg_val_raw, reg_pred))
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
            "val_mae": float(mean_absolute_error(y_reg_val_raw, reg_pred)),
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }
        fold_results.append(fold_result)

        if val_bce < best_val_loss:
            best_val_loss = val_bce
            best_fold_idx = fold_i

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    temporal_feature_names, static_feature_names = _feature_names(
        dataset,
        fallback_temporal=[f"temporal_{i}" for i in range(dataset.X_seq.shape[2])],
        fallback_static_count=dataset.X_static.shape[1],
    )
    final_state_dict, final_scaler_seq, final_scaler_static = _fit_surrogate_final_model(
        dataset=dataset,
        model_cls=model_cls,
        use_temporal=use_temporal,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        alpha=alpha,
        beta=beta,
        device=device,
        seed=seed,
    )
    torch.save(final_state_dict, artifacts_dir / f"{prefix}_weights.pt")
    joblib.dump(final_scaler_seq, artifacts_dir / f"{prefix}_scaler_seq.joblib")
    joblib.dump(final_scaler_static, artifacts_dir / f"{prefix}_scaler_static.joblib")
    _write_surrogate_manifest(
        artifacts_dir,
        prefix=prefix,
        model_type=_model_type(model_cls),
        seed=seed,
        dataset=dataset,
        temporal_feature_names=temporal_feature_names,
        static_feature_names=static_feature_names,
        use_temporal=use_temporal,
    )
    pd.DataFrame(fold_results).to_csv(artifacts_dir / f"{prefix}_metrics.csv", index=False)

    return {
        "n_folds": actual_folds,
        "folds": fold_results,
        "best_fold": best_fold_idx,
        "final_model_trained_on_all_groups": True,
    }


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
    parser.add_argument(
        "--model", default="cnn", choices=["cnn", "lstm"],
        help="Model architecture (default: cnn).",
    )
    args = parser.parse_args()

    use_temporal = not args.no_temporal
    model_cls = {"cnn": SWMMSurrogateCNN, "lstm": SWMMSurrogateLSTM}[args.model]
    label = "full" if use_temporal else "ablation (no temporal)"
    print(f"\n=== Training surrogate {args.model} [{label}] ===")
    result = train_surrogate(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_cv_folds=args.folds,
        device=args.device,
        use_temporal=use_temporal,
        model_cls=model_cls,
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
