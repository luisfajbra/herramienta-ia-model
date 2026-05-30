# SP3 — CNN 1D con PyTorch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SWMMTemporalCNN` (dual-branch CNN 1D) and `train_cnn()` with GroupKFold cross-validation, StandardScaler normalization, and artifact persistence.

**Architecture:** A two-branch PyTorch `nn.Module` fuses a Conv1D temporal branch and a dense static branch, then outputs a single value for classification (Sigmoid) or regression (linear). Training uses `GroupKFold` on `run_id` so no run leaks across folds; the best fold's weights and scalers are saved to disk.

**Tech Stack:** PyTorch, scikit-learn (`GroupKFold`, `StandardScaler`, metrics), joblib, pandas, numpy.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `swmm_resilience/ml/temporal/models/__init__.py` | Package marker + export |
| Create | `swmm_resilience/ml/temporal/models/cnn.py` | `SWMMTemporalCNN` nn.Module |
| Replace | `swmm_resilience/ml/temporal/train_cnn.py` | `train_cnn()` + CLI |
| Modify | `swmm_resilience/config.py` | Add `DEFAULT_TEMPORAL_ARTIFACTS_DIR` |
| Modify | `swmm_resilience/reset.py` | Add `.pt` + scan temporal artifacts dir |
| Modify | `swmm_resilience/ml/temporal/__init__.py` | Export `SWMMTemporalCNN` |
| Create | `tests/ml/temporal/test_cnn_model.py` | 5 TDD tests |

---

### Task 1: Write 5 Failing Tests (TDD)

**Files:**
- Create: `tests/ml/temporal/test_cnn_model.py`

Context: `tests/ml/__init__.py` and `tests/ml/temporal/__init__.py` already exist from SP2. The test file imports `SWMMTemporalCNN` from `swmm_resilience.ml.temporal.models.cnn` and `train_cnn` from `swmm_resilience.ml.temporal.train_cnn` — both will raise `ImportError` until Tasks 2–3 are done.

- [ ] **Step 1: Create the test file**

```python
# tests/ml/temporal/test_cnn_model.py
"""TDD tests for SWMMTemporalCNN and train_cnn (SP3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from swmm_resilience.ml.temporal.models.cnn import SWMMTemporalCNN
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.train_cnn import train_cnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_dataset(n_per_group: int = 20) -> TemporalWindowDataset:
    """Minimal dataset: 2 run_id groups, n_per_group samples each."""
    N = n_per_group * 2
    rng = np.random.RandomState(42)
    groups = np.array(["run_a"] * n_per_group + ["run_b"] * n_per_group, dtype=object)
    return TemporalWindowDataset(
        X_seq=rng.randn(N, 4, 6).astype(np.float32),
        X_static=rng.randn(N, 7).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=rng.rand(N).astype(np.float32),
        groups=groups,
        meta=pd.DataFrame(
            {
                "run_id": groups,
                "node_id": ["J-000"] * N,
                "window_start_min": np.arange(N, dtype=float),
            }
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardPassClassification:
    def test_output_shape_and_range(self):
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="classification")
        x_seq = torch.randn(8, 4, 6)
        x_static = torch.randn(8, 7)
        out = model(x_seq, x_static)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "Classification output must be in [0, 1]"


class TestForwardPassRegression:
    def test_output_shape(self):
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="regression")
        x_seq = torch.randn(8, 4, 6)
        x_static = torch.randn(8, 7)
        out = model(x_seq, x_static)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"


class TestTrainingLossDecreases:
    def test_loss_decreases_over_5_epochs(self):
        torch.manual_seed(0)
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="classification")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.BCELoss()
        x_seq = torch.randn(32, 4, 6)
        x_static = torch.randn(32, 7)
        y = torch.zeros(32, 1)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x_seq, x_static)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestArtifactsSavedAfterTraining:
    def test_all_artifacts_exist(self, tmp_path):
        dataset = _synthetic_dataset()
        artifacts_dir = tmp_path / "artifacts"

        train_cnn(
            artifacts_dir=artifacts_dir,
            task="classification",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )

        assert (artifacts_dir / "cnn_classifier_weights.pt").exists()
        assert (artifacts_dir / "cnn_classifier_scaler_seq.joblib").exists()
        assert (artifacts_dir / "cnn_classifier_scaler_static.joblib").exists()
        assert (artifacts_dir / "cnn_classifier_metrics.csv").exists()


class TestNoDataLeakageBetweenFolds:
    def test_train_val_groups_disjoint(self, tmp_path):
        dataset = _synthetic_dataset()

        result = train_cnn(
            artifacts_dir=tmp_path / "artifacts",
            task="classification",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )

        for fold in result["folds"]:
            train_groups = set(fold["train_groups"])
            val_groups = set(fold["val_groups"])
            overlap = train_groups & val_groups
            assert not overlap, f"Data leakage in fold {fold['fold']}: {overlap}"
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /Users/luis/herramienta-ia-model
pytest tests/ml/temporal/test_cnn_model.py -v 2>&1 | head -40
```

Expected: All 5 tests fail with `ModuleNotFoundError` or `ImportError` because `models/cnn.py` and `train_cnn` full impl don't exist yet.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/ml/temporal/test_cnn_model.py
git commit -m "test(sp3): add 5 failing TDD tests for SWMMTemporalCNN and train_cnn"
```

---

### Task 2: Implement SWMMTemporalCNN

**Files:**
- Create: `swmm_resilience/ml/temporal/models/__init__.py`
- Create: `swmm_resilience/ml/temporal/models/cnn.py`

Context: `SWMMTemporalCNN` has two branches. The temporal branch uses `Conv1d` — note that `Conv1d` expects `[batch, channels, length]`, so `x_seq` arrives as `[batch, timesteps, features]` and must be permuted to `[batch, features, timesteps]` before the first conv. Static branch is two linear layers. Outputs merge at a 96-dim fused vector → 64-dim fusion → task-specific head.

- [ ] **Step 1: Create models package marker**

Create `swmm_resilience/ml/temporal/models/__init__.py`:

```python
from .cnn import SWMMTemporalCNN

__all__ = ["SWMMTemporalCNN"]
```

- [ ] **Step 2: Implement the CNN module**

Create `swmm_resilience/ml/temporal/models/cnn.py`:

```python
"""Dual-branch CNN 1D for SWMM temporal node data."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMTemporalCNN(nn.Module):
    """CNN 1D + dense static branch for per-node classification or regression.

    Args:
        n_temporal_features: Number of time-series channels (default 6).
        n_static_features: Number of static node features (default 7).
        task: 'classification' (Sigmoid output in [0,1]) or 'regression' (linear output).
    """

    def __init__(
        self,
        n_temporal_features: int = 6,
        n_static_features: int = 7,
        task: str = "classification",
    ) -> None:
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        self.task = task

        # Temporal branch: Conv1d operates on [batch, channels, length]
        # Input arrives as [batch, timesteps=4, features=6] and is permuted before this branch.
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_temporal_features, 32, kernel_size=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → [batch, 64, 1]
        )

        # Static branch
        self.static_branch = nn.Sequential(
            nn.Linear(n_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Fusion: 64 (temporal) + 32 (static) = 96
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Task head
        if task == "classification":
            self.head: nn.Module = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        else:
            self.head = nn.Linear(64, 1)

    def forward(self, x_seq: Tensor, x_static: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_seq: [batch, timesteps, temporal_features]
            x_static: [batch, static_features]

        Returns:
            [batch, 1] — probability for classification, raw value for regression.
        """
        # Permute for Conv1d: [batch, features, timesteps]
        t = self.temporal_branch(x_seq.permute(0, 2, 1)).squeeze(-1)  # [batch, 64]
        s = self.static_branch(x_static)                               # [batch, 32]
        fused = self.fusion(torch.cat([t, s], dim=1))                  # [batch, 64]
        return self.head(fused)                                         # [batch, 1]
```

- [ ] **Step 3: Run the two forward-pass tests**

```bash
pytest tests/ml/temporal/test_cnn_model.py::TestForwardPassClassification \
       tests/ml/temporal/test_cnn_model.py::TestForwardPassRegression -v
```

Expected:
```
PASSED tests/ml/temporal/test_cnn_model.py::TestForwardPassClassification::test_output_shape_and_range
PASSED tests/ml/temporal/test_cnn_model.py::TestForwardPassRegression::test_output_shape
```

- [ ] **Step 4: Commit**

```bash
git add swmm_resilience/ml/temporal/models/__init__.py \
        swmm_resilience/ml/temporal/models/cnn.py
git commit -m "feat(sp3): implement SWMMTemporalCNN dual-branch CNN 1D module"
```

---

### Task 3: Add Config Constant + Implement train_cnn()

**Files:**
- Modify: `swmm_resilience/config.py`
- Replace: `swmm_resilience/ml/temporal/train_cnn.py`

Context: `DEFAULT_TEMPORAL_ARTIFACTS_DIR` goes under `results/temporal/model_artifacts/`. The `train_cnn()` function accepts `_dataset` (leading underscore = testing-only injection; production path calls `build_temporal_windows()`). `GroupKFold` requires `n_splits ≤ n_unique_groups` — the function auto-reduces with a warning. Best fold is chosen by minimum `val_loss`.

- [ ] **Step 1: Add DEFAULT_TEMPORAL_ARTIFACTS_DIR to config.py**

Open `swmm_resilience/config.py`. After line 38 (`DEFAULT_MODEL_ARTIFACTS_DIR = ...`), add:

```python
DEFAULT_TEMPORAL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "temporal" / "model_artifacts"
```

- [ ] **Step 2: Replace train_cnn.py with full implementation**

Overwrite `swmm_resilience/ml/temporal/train_cnn.py` completely:

```python
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
```

- [ ] **Step 3: Run all 5 tests**

```bash
pytest tests/ml/temporal/test_cnn_model.py -v
```

Expected:
```
PASSED tests/ml/temporal/test_cnn_model.py::TestForwardPassClassification::test_output_shape_and_range
PASSED tests/ml/temporal/test_cnn_model.py::TestForwardPassRegression::test_output_shape
PASSED tests/ml/temporal/test_cnn_model.py::TestTrainingLossDecreases::test_loss_decreases_over_5_epochs
PASSED tests/ml/temporal/test_cnn_model.py::TestArtifactsSavedAfterTraining::test_all_artifacts_exist
PASSED tests/ml/temporal/test_cnn_model.py::TestNoDataLeakageBetweenFolds::test_train_val_groups_disjoint

5 passed
```

- [ ] **Step 4: Commit**

```bash
git add swmm_resilience/config.py swmm_resilience/ml/temporal/train_cnn.py
git commit -m "feat(sp3): implement train_cnn() with GroupKFold, scalers, and artifact saving"
```

---

### Task 4: Update reset.py + Exports + Smoke Test

**Files:**
- Modify: `swmm_resilience/reset.py`
- Modify: `swmm_resilience/ml/temporal/__init__.py`

Context: `reset_artifacts()` currently scans only `results/model_artifacts/` with suffixes `{".joblib", ".json", ".csv", ".xlsx"}`. SP3 adds `.pt` files in `results/temporal/model_artifacts/` — both dirs must be scanned.

- [ ] **Step 1: Update reset_artifacts() in swmm_resilience/reset.py**

Find the `reset_artifacts` function (around line 87). Replace the entire function body:

Old:
```python
def reset_artifacts(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all ML artifact files (.joblib, manifest.json, metric CSVs/XLSX). Returns count deleted."""
    count = 0
    suffixes = {".joblib", ".json", ".csv", ".xlsx"}
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        artifacts_dir = net_dir / "results" / "model_artifacts"
        if not artifacts_dir.exists():
            continue
        for f in sorted(artifacts_dir.iterdir()):
            if f.is_file() and f.suffix in suffixes:
                f.unlink()
                _log(f"  Eliminado: {f.relative_to(networks_dir)}", callback)
                count += 1
    return count
```

New:
```python
def reset_artifacts(
    networks_dir: Path = NETWORKS_DIR,
    callback: Callable[[str], None] | None = None,
) -> int:
    """Delete all ML artifact files (.joblib, .json, .csv, .xlsx, .pt). Returns count deleted."""
    count = 0
    suffixes = {".joblib", ".json", ".csv", ".xlsx", ".pt"}
    for net_dir in sorted(networks_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        artifact_dirs = [
            net_dir / "results" / "model_artifacts",
            net_dir / "results" / "temporal" / "model_artifacts",
        ]
        for artifacts_dir in artifact_dirs:
            if not artifacts_dir.exists():
                continue
            for f in sorted(artifacts_dir.iterdir()):
                if f.is_file() and f.suffix in suffixes:
                    f.unlink()
                    _log(f"  Eliminado: {f.relative_to(networks_dir)}", callback)
                    count += 1
    return count
```

- [ ] **Step 2: Update temporal __init__.py to export SWMMTemporalCNN**

Open `swmm_resilience/ml/temporal/__init__.py`. Replace content:

```python
"""
Scaffolding for future temporal ML models based on hydrographs.

This package intentionally does not implement a CNN yet. It defines the
structure we will use later for time-window datasets, CNN training and
temporal prediction.
"""

from .models.cnn import SWMMTemporalCNN
from .schemas import TemporalDatasetSpec, TemporalWindowSpec

__all__ = ["TemporalDatasetSpec", "TemporalWindowSpec", "SWMMTemporalCNN"]
```

- [ ] **Step 3: CLI smoke test (import check)**

This test verifies the module loads cleanly without needing DB or data:

```bash
python -c "
from swmm_resilience.ml.temporal.train_cnn import train_cnn
from swmm_resilience.ml.temporal.models.cnn import SWMMTemporalCNN
from swmm_resilience.config import DEFAULT_TEMPORAL_ARTIFACTS_DIR
print('SWMMTemporalCNN:', SWMMTemporalCNN)
print('DEFAULT_TEMPORAL_ARTIFACTS_DIR:', DEFAULT_TEMPORAL_ARTIFACTS_DIR)
print('OK')
"
```

Expected output:
```
SWMMTemporalCNN: <class 'swmm_resilience.ml.temporal.models.cnn.SWMMTemporalCNN'>
DEFAULT_TEMPORAL_ARTIFACTS_DIR: /Users/luis/herramienta-ia-model/data/networks/chico_hydro-qx1/results/temporal/model_artifacts
OK
```

- [ ] **Step 4: Run full test suite — no regressions**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: All previously passing tests still pass (14 from SP1+SP2) + 5 new SP3 tests = **19 passed, 0 failed**.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/reset.py swmm_resilience/ml/temporal/__init__.py
git commit -m "feat(sp3): update reset.py to delete .pt artifacts; export SWMMTemporalCNN from temporal package"
```
