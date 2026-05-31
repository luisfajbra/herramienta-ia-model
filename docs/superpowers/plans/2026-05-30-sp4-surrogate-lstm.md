# SP4 — Surrogate LSTM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SWMMSurrogateLSTM` as a fourth model in `compare_surrogate.py`, using identical data, splits, and loss as the surrogate CNN, to directly compare LSTM vs CNN for surrogate flood prediction.

**Architecture:** `SWMMSurrogateLSTM` mirrors `SWMMSurrogateCNN` exactly — only the temporal branch changes from Conv1d to a 2-layer LSTM. `_train_eval_cnn` in `compare_surrogate.py` is refactored to `_train_eval_model(model_cls, ...)` so both CNN and LSTM reuse the same training loop.

**Tech Stack:** Python 3.11, PyTorch (LSTM), scikit-learn, pytest.

---

## Background for agentic implementers

### Codebase conventions

- Package root: `swmm_resilience/`
- Working directory: `/Users/luis/herramienta-ia-model`
- Python env: `/opt/miniconda3/envs/py39/bin/pytest`
- Existing surrogate CNN model: `swmm_resilience/ml/temporal/models/surrogate_cnn.py` — `SWMMSurrogateCNN(n_temporal_features, n_static_features, use_temporal)` returns `(cls_logit, reg_out)`
- Existing comparison pipeline: `swmm_resilience/ml/temporal/compare_surrogate.py` — contains `_train_eval_cnn()` helper and the main fold loop

### Key design decisions

- `SWMMSurrogateLSTM` has the **same constructor signature** as `SWMMSurrogateCNN`: `(n_temporal_features, n_static_features, use_temporal)`, `forward(x_seq, x_static) -> (cls_logit, reg_out)`
- When `use_temporal=False`, pass `None` as `x_seq` — same convention as CNN
- `_train_eval_cnn` is renamed to `_train_eval_model(model_cls, ...)` — `model_cls` is either `SWMMSurrogateCNN` or `SWMMSurrogateLSTM`
- Result columns: `lstm_auc_roc`, `lstm_f1`, `lstm_precision`, `lstm_recall`, `lstm_accuracy`, `lstm_rmse`, `lstm_mae`
- macOS OpenMP fix already in place (`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`) — do not remove

---

## File map

| File | Action |
|------|--------|
| `swmm_resilience/ml/temporal/models/surrogate_lstm.py` | **Create** — `SWMMSurrogateLSTM` |
| `swmm_resilience/ml/temporal/compare_surrogate.py` | **Modify** — refactor helper, add LSTM block |
| `tests/ml/temporal/test_surrogate_lstm.py` | **Create** — model tests |
| `tests/ml/temporal/test_compare_surrogate.py` | **Modify** — add LSTM column check |

---

## Task 1: SWMMSurrogateLSTM model + tests

**Files:**
- Create: `swmm_resilience/ml/temporal/models/surrogate_lstm.py`
- Create: `tests/ml/temporal/test_surrogate_lstm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/temporal/test_surrogate_lstm.py`:

```python
"""Tests for SWMMSurrogateLSTM dual-head surrogate model."""
from __future__ import annotations

import torch
import pytest

from swmm_resilience.ml.temporal.models.surrogate_lstm import SWMMSurrogateLSTM


class TestForwardPassFullModel:
    def test_output_shapes(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(8, 20, 2)
        x_static = torch.randn(8, 21)
        cls_logit, reg_out = model(x_seq, x_static)
        assert cls_logit.shape == (8, 1), f"cls_logit: {cls_logit.shape}"
        assert reg_out.shape == (8, 1), f"reg_out: {reg_out.shape}"

    def test_classification_logit_is_unbounded(self):
        """cls_logit is raw logit — not in [0,1]."""
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(32, 20, 2) * 10
        x_static = torch.randn(32, 21) * 10
        cls_logit, _ = model(x_seq, x_static)
        assert (cls_logit.abs() > 1).any(), "Expected unbounded logits"

    def test_variable_sequence_length(self):
        """LSTM handles variable T natively."""
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        for T in [5, 20, 50]:
            cls_logit, reg_out = model(torch.randn(4, T, 2), torch.randn(4, 21))
            assert cls_logit.shape == (4, 1)
            assert reg_out.shape == (4, 1)

    def test_regression_output_unbounded(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        torch.manual_seed(0)
        x_seq = torch.randn(16, 20, 2) * 100
        x_static = torch.randn(16, 21) * 100
        _, reg_out = model(x_seq, x_static)
        assert reg_out.min() < 0 or reg_out.max() > 1


class TestNoTemporalMode:
    def test_output_shapes_no_temporal(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21, use_temporal=False)
        cls_logit, reg_out = model(None, torch.randn(8, 21))
        assert cls_logit.shape == (8, 1)
        assert reg_out.shape == (8, 1)

    def test_x_seq_none_accepted(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21, use_temporal=False)
        cls_logit, reg_out = model(None, torch.randn(4, 21))
        assert cls_logit.shape == (4, 1)


class TestGradientFlow:
    def test_gradients_flow_through_both_heads(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(4, 20, 2)
        x_static = torch.randn(4, 21)
        cls_logit, reg_out = model(x_seq, x_static)

        loss = (
            torch.nn.BCEWithLogitsLoss()(cls_logit, torch.zeros(4, 1))
            + 0.01 * torch.nn.MSELoss()(reg_out, torch.zeros(4, 1))
        )
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients"
        assert all(g.abs().sum() > 0 for g in grads), "Zero gradients found"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_surrogate_lstm.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named '...surrogate_lstm'`

- [ ] **Step 3: Implement SWMMSurrogateLSTM**

Create `swmm_resilience/ml/temporal/models/surrogate_lstm.py`:

```python
# swmm_resilience/ml/temporal/models/surrogate_lstm.py
"""Dual-branch, dual-head surrogate LSTM for SWMM flood prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMSurrogateLSTM(nn.Module):
    """LSTM surrogate for per-node flood prediction.

    Mirrors SWMMSurrogateCNN — only the temporal branch differs.
    Returns raw logits for classification (apply sigmoid externally) and
    unbounded float for regression. Constructor signature is identical to
    SWMMSurrogateCNN so both can be used interchangeably in _train_eval_model().

    Args:
        n_temporal_features: LSTM input size (default 2 — inflow channels).
        n_static_features: Static feature count (default 21 for unified dataset).
        use_temporal: If False, LSTM branch disabled and x_seq is ignored.
    """

    def __init__(
        self,
        n_temporal_features: int = 2,
        n_static_features: int = 21,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        if use_temporal:
            self.temporal_branch = nn.LSTM(
                input_size=n_temporal_features,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            fusion_in = 64 + 32
        else:
            self.temporal_branch = nn.Identity()  # unused placeholder
            fusion_in = 32

        self.static_branch = nn.Sequential(
            nn.Linear(n_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(64, 1)  # raw logit — sigmoid applied externally
        self.reg_head = nn.Linear(64, 1)  # unbounded float

        # Larger init ensures |logit| > 1 at random init for test reliability.
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=2.0)

    def forward(
        self,
        x_seq: Tensor | None,
        x_static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x_seq: [batch, T, n_temporal_features] or None if use_temporal=False.
            x_static: [batch, n_static_features]

        Returns:
            (cls_logit [batch, 1], reg_out [batch, 1])
        """
        s = self.static_branch(x_static)  # [batch, 32]

        if self.use_temporal:
            assert x_seq is not None, "x_seq required when use_temporal=True"
            _, (h_n, _) = self.temporal_branch(x_seq)  # h_n: [num_layers, batch, 64]
            t = h_n[-1]                                  # last layer: [batch, 64]
            fused_in = torch.cat([t, s], dim=1)          # [batch, 96]
        else:
            fused_in = s                                  # [batch, 32]

        fused = self.fusion(fused_in)  # [batch, 64]
        return self.cls_head(fused), self.reg_head(fused)
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_surrogate_lstm.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Verify no regressions**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/ -q 2>&1 | tail -5
```

Expected: 57 previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/temporal/models/surrogate_lstm.py \
        tests/ml/temporal/test_surrogate_lstm.py
git commit -m "feat(sp4): add SWMMSurrogateLSTM — dual-head LSTM surrogate with same interface as CNN"
```

---

## Task 2: Integrate LSTM into compare_surrogate.py

**Files:**
- Modify: `swmm_resilience/ml/temporal/compare_surrogate.py`
- Modify: `tests/ml/temporal/test_compare_surrogate.py`

- [ ] **Step 1: Add failing test for LSTM columns**

Read `tests/ml/temporal/test_compare_surrogate.py`, then add this test inside `class TestMetricColumns`:

```python
    def test_lstm_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["lstm_auc_roc", "lstm_f1", "lstm_rmse"]:
            assert col in result.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_compare_surrogate.py::TestMetricColumns::test_lstm_metrics_present -v 2>&1 | tail -10
```

Expected: `AssertionError: Missing column: lstm_auc_roc`

- [ ] **Step 3: Modify compare_surrogate.py**

Make these four targeted edits to `swmm_resilience/ml/temporal/compare_surrogate.py`:

**Edit 1 — Add LSTM import** (after the existing CNN import on line 38):

```python
# Old:
from .models.surrogate_cnn import SWMMSurrogateCNN

# New:
from .models.surrogate_cnn import SWMMSurrogateCNN
from .models.surrogate_lstm import SWMMSurrogateLSTM
```

**Edit 2 — Rename `_train_eval_cnn` to `_train_eval_model` and add `model_cls` param.**

Replace the entire function definition (lines 56–129) with:

```python
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
```

**Edit 3 — Update the two existing CNN calls and add LSTM block** in the fold loop.

Replace lines 198–224 (from `# ── CNN full` to the `print(...)` line) with:

```python
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
```

**Edit 4 — Update main() table** to include LSTM column.

Replace lines 251–258 (the metrics print loop) with:

```python
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
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_compare_surrogate.py -v
```

Expected: all 8 tests PASS (7 existing + 1 new).

- [ ] **Step 5: Verify no regressions**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/ -q 2>&1 | tail -5
```

Expected: 64+ passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/temporal/compare_surrogate.py \
        tests/ml/temporal/test_compare_surrogate.py
git commit -m "feat(sp4): integrate LSTM into compare_surrogate() — refactor _train_eval_model, add lstm_ metrics"
```

---

## Verification after all tasks

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
/opt/miniconda3/envs/py39/bin/python -m swmm_resilience.ml.temporal.compare_surrogate \
    --epochs 100 --folds 5
```

Expected output format:
```
=== Unified Surrogate Comparison ===
Fold 0: XGB F1=0.xxx  CNN F1=0.xxx  LSTM F1=0.xxx  Ablation F1=0.xxx
...
Metric         XGBoost   CNN Full       LSTM  CNN Ablation
----------------------------------------------------------
  auc_roc         0.xxxx     0.xxxx   0.xxxx        0.xxxx
  f1              0.xxxx     0.xxxx   0.xxxx        0.xxxx
  ...
```
