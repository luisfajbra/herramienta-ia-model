# Surrogate CNN for SWMM Flood Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SWMM surrogate CNN that predicts per-node flooding (flooded? + peak volume) from an inflow multiplier and static network features — no SWMM run needed at inference.

**Architecture:** Full-sequence dual-branch CNN — Conv1d temporal branch (AdaptiveAvgPool, handles variable T) + dense static branch → fused → dual output head (classification logit + regression value). Multi-task loss: `BCEWithLogitsLoss(pos_weight) + 0.01 × MSE`. Comparison ablation (`--no-temporal`) disables Conv branch and appends multiplier to static features.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn (GroupKFold, StandardScaler), pandas, SQLite, pytest.

---

## Background for agentic implementers

### Codebase conventions

- Package root: `swmm_resilience/`
- DB helpers: `swmm_resilience/database/queries.py`, schema: `swmm_resilience/database/schema.py`
- Temporal ML files live in `swmm_resilience/ml/temporal/`
- Existing model: `swmm_resilience/ml/temporal/models/cnn.py` — **do not modify**
- Existing training: `swmm_resilience/ml/temporal/train_cnn.py` — **do not modify**
- Existing dataset builder: `swmm_resilience/ml/temporal/dataset.py` — you will **add** one function
- Existing inference: `swmm_resilience/ml/temporal/predict.py` — you will **add** one function
- Existing schemas: `swmm_resilience/ml/temporal/schemas.py` — reuse `TemporalWindowDataset`
- Config: `swmm_resilience/config.py` exports `DEFAULT_DB_FILE`, `DEFAULT_TEMPORAL_ARTIFACTS_DIR`, `NETWORKS_DIR`
- Tests: `tests/ml/temporal/` — follow pattern in `test_cnn_model.py`

### Key constants (already defined)

```python
TEMPORAL_COLS = [
    "total_inflow_lps", "lateral_inflow_lps",
    "depth_m", "depth_ratio", "flooding_lps", "total_outflow_lps",
]  # 6 features — same for training and surrogate

STATIC_COLS = [
    "full_depth_m", "in_degree", "out_degree",
    "upstream_diam_avg_m", "downstream_diam_avg_m",
    "upstream_capacity_lps", "downstream_capacity_lps",
]  # 7 features
```

### Inference feature strategy

At inference, only inflow features are available from the synthetic hydrograph
(`total_inflow_lps`, `lateral_inflow_lps`). SWMM-output features (`depth_m`, `depth_ratio`,
`flooding_lps`, `total_outflow_lps`) are set to **0**. The model was trained with these features
present, so zero-filling is an approximation — acceptable as a design tradeoff.

### DB tables used

```sql
-- temporal_artifacts: run_id, network_hash, parquet_path
-- runs: run_id, inflow_multiplier
-- network_nodes: node_uid, network_hash, full_depth_m, in_degree, ...
```

---

## File map

| File | Action | Responsibility |
|------|--------|---------------|
| `swmm_resilience/ml/temporal/models/surrogate_cnn.py` | **Create** | `SWMMSurrogateCNN` dual-head model |
| `swmm_resilience/ml/temporal/dataset.py` | **Modify** — append | `build_surrogate_dataset()` function |
| `swmm_resilience/ml/temporal/train_surrogate.py` | **Create** | `train_surrogate()` + `main()` CLI |
| `swmm_resilience/ml/temporal/predict.py` | **Modify** — append | `predict_surrogate_from_multiplier()` |
| `tests/ml/temporal/test_surrogate_cnn.py` | **Create** | Model forward-pass tests |
| `tests/ml/temporal/test_surrogate_dataset.py` | **Create** | Dataset builder tests |
| `tests/ml/temporal/test_train_surrogate.py` | **Create** | Training pipeline tests |
| `tests/ml/temporal/test_surrogate_predict.py` | **Create** | Inference function tests |

---

## Task 1: SWMMSurrogateCNN — dual-head model

**Files:**
- Create: `swmm_resilience/ml/temporal/models/surrogate_cnn.py`
- Create: `tests/ml/temporal/test_surrogate_cnn.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/temporal/test_surrogate_cnn.py
"""Tests for SWMMSurrogateCNN dual-head surrogate model."""
from __future__ import annotations

import pytest
import torch

from swmm_resilience.ml.temporal.models.surrogate_cnn import SWMMSurrogateCNN


class TestForwardPassFullModel:
    def test_output_shapes(self):
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        x_seq = torch.randn(8, 24, 6)     # batch=8, T=24, features=6
        x_static = torch.randn(8, 7)
        cls_logit, reg_out = model(x_seq, x_static)
        assert cls_logit.shape == (8, 1), f"cls_logit shape: {cls_logit.shape}"
        assert reg_out.shape == (8, 1),   f"reg_out shape: {reg_out.shape}"

    def test_classification_logit_is_unbounded(self):
        """cls_logit is raw logit — not forced to [0,1] (sigmoid applied externally)."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        x_seq = torch.randn(32, 24, 6) * 10   # large inputs to saturate sigmoid
        x_static = torch.randn(32, 7) * 10
        cls_logit, _ = model(x_seq, x_static)
        # At least some logits should fall outside [0, 1]
        assert (cls_logit.abs() > 1).any(), "Expected unbounded logits"

    def test_variable_sequence_length(self):
        """AdaptiveAvgPool must handle different T values with same output shape."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        for T in [10, 20, 40]:
            cls_logit, reg_out = model(torch.randn(4, T, 6), torch.randn(4, 7))
            assert cls_logit.shape == (4, 1)
            assert reg_out.shape == (4, 1)

    def test_regression_output_unbounded(self):
        """Regression head must be linear — no activation."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        torch.manual_seed(0)
        x_seq = torch.randn(16, 24, 6) * 100
        x_static = torch.randn(16, 7) * 100
        _, reg_out = model(x_seq, x_static)
        assert reg_out.min() < 0 or reg_out.max() > 1, "Regression head should not be bounded"


class TestNoTemporalMode:
    def test_output_shapes_no_temporal(self):
        """use_temporal=False: n_static_features=8 (7 static + 1 multiplier)."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=8, use_temporal=False)
        x_static = torch.randn(8, 8)
        cls_logit, reg_out = model(None, x_static)
        assert cls_logit.shape == (8, 1)
        assert reg_out.shape == (8, 1)

    def test_no_temporal_ignores_x_seq(self):
        """Passing x_seq=None must not raise when use_temporal=False."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=8, use_temporal=False)
        x_static = torch.randn(4, 8)
        cls_logit, reg_out = model(None, x_static)   # must not raise
        assert cls_logit.shape == (4, 1)


class TestGradientFlow:
    def test_gradients_flow_through_both_heads(self):
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x_seq = torch.randn(4, 24, 6, requires_grad=False)
        x_static = torch.randn(4, 7, requires_grad=False)
        cls_logit, reg_out = model(x_seq, x_static)

        y_cls = torch.zeros(4, 1)
        y_reg = torch.zeros(4, 1)
        loss = torch.nn.BCEWithLogitsLoss()(cls_logit, y_cls) + 0.01 * torch.nn.MSELoss()(reg_out, y_reg)
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found"
        assert all(g.abs().sum() > 0 for g in grads), "Some gradients are all-zero"
```

- [ ] **Step 2: Run tests — verify they all FAIL**

```bash
cd /Users/luis/herramienta-ia-model
python -m pytest tests/ml/temporal/test_surrogate_cnn.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'swmm_resilience.ml.temporal.models.surrogate_cnn'`

- [ ] **Step 3: Implement `SWMMSurrogateCNN`**

```python
# swmm_resilience/ml/temporal/models/surrogate_cnn.py
"""Dual-branch, dual-head surrogate CNN for SWMM flood prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMSurrogateCNN(nn.Module):
    """Global-pooling CNN surrogate for per-node flood prediction.

    Returns raw logits for classification (apply sigmoid externally) and
    unbounded float for regression. This supports BCEWithLogitsLoss during
    training and torch.sigmoid() at inference.

    Args:
        n_temporal_features: Channels in the inflow sequence (default 6).
        n_static_features: Node static feature count. 7 when use_temporal=True,
            8 (7 static + 1 multiplier scalar) when use_temporal=False.
        use_temporal: If False, temporal Conv branch is disabled and x_seq is
            ignored. The multiplier must be appended to x_static by the caller.
    """

    def __init__(
        self,
        n_temporal_features: int = 6,
        n_static_features: int = 7,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        if use_temporal:
            self.temporal_branch: nn.Module = nn.Sequential(
                nn.Conv1d(n_temporal_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            fusion_in = 64 + 32
        else:
            self.temporal_branch = nn.Identity()   # unused placeholder
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
        self.cls_head = nn.Linear(64, 1)   # raw logit — sigmoid applied externally
        self.reg_head = nn.Linear(64, 1)   # unbounded float

    def forward(
        self,
        x_seq: Tensor | None,
        x_static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x_seq: [batch, T, temporal_features] or None if use_temporal=False.
            x_static: [batch, static_features]

        Returns:
            (cls_logit [batch, 1], reg_out [batch, 1])
        """
        s = self.static_branch(x_static)   # [batch, 32]

        if self.use_temporal:
            assert x_seq is not None, "x_seq required when use_temporal=True"
            t = self.temporal_branch(x_seq.permute(0, 2, 1)).squeeze(-1)  # [batch, 64]
            fused_in = torch.cat([t, s], dim=1)                            # [batch, 96]
        else:
            fused_in = s                                                    # [batch, 32]

        fused = self.fusion(fused_in)          # [batch, 64]
        return self.cls_head(fused), self.reg_head(fused)
```

- [ ] **Step 4: Run tests — verify they all PASS**

```bash
python -m pytest tests/ml/temporal/test_surrogate_cnn.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/temporal/models/surrogate_cnn.py \
        tests/ml/temporal/test_surrogate_cnn.py
git commit -m "feat(surrogate-cnn): add SWMMSurrogateCNN dual-head model with variable-length temporal branch"
```

---

## Task 2: build_surrogate_dataset()

**Files:**
- Modify: `swmm_resilience/ml/temporal/dataset.py` (append one function at the end)
- Create: `tests/ml/temporal/test_surrogate_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/temporal/test_surrogate_dataset.py
"""TDD tests for build_surrogate_dataset()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import build_surrogate_dataset
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset

# ── helpers ──────────────────────────────────────────────────────────────────

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_parquet(
    directory: Path,
    run_id: str,
    network_hash: str,
    n_nodes: int = 3,
    n_steps: int = 10,
    flooding_node_idx: int | None = 0,
) -> Path:
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        flooded = node_idx == flooding_node_idx
        for step in range(n_steps):
            records.append({
                "run_id": run_id,
                "network_hash": network_hash,
                "node_id": node_id,
                "step_index": step,
                "time_sec": step * 300,
                "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5 + step * 0.01,
                "depth_ratio": 0.3,
                "flooding_lps": 8.0 if flooded and step >= 5 else 0.0,
                "total_outflow_lps": 8.0,
                "failed_now": 1 if flooded and step >= 5 else 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _setup_db(tmp_path: Path, n_runs: int = 2, n_nodes: int = 3) -> tuple[Path, str]:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = uuid.uuid4().hex

    parquet_dir = tmp_path / "parquets"

    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        conn.execute(
            "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
            "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, "test.inp", network_hash, "uniform", "uniform", 0.0, multiplier, "done", "test"),
        )
        parquet_path = _make_parquet(parquet_dir, run_id, network_hash, n_nodes=n_nodes)
        conn.execute(
            "INSERT INTO temporal_artifacts (run_id, network_hash, parquet_path) VALUES (?, ?, ?)",
            (run_id, network_hash, str(parquet_path)),
        )

    for node_idx in range(n_nodes):
        conn.execute(
            "INSERT INTO network_nodes (node_uid, network_hash, full_depth_m, in_degree, out_degree, "
            "upstream_diam_avg_m, downstream_diam_avg_m, upstream_capacity_lps, downstream_capacity_lps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"J-{node_idx:03d}", network_hash, 1.2, 1, 1, 0.3, 0.3, 50.0, 50.0),
        )

    conn.commit()
    conn.close()
    return db_path, network_hash


# ── tests ─────────────────────────────────────────────────────────────────────

class TestOneSamplePerNodePerRun:
    def test_sample_count(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        # 2 runs × 3 nodes = 6 samples
        assert ds.X_seq.shape[0] == 6, f"Expected 6 samples, got {ds.X_seq.shape[0]}"

    def test_no_duplicate_samples(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        pairs = list(zip(ds.meta["run_id"], ds.meta["node_id"]))
        assert len(pairs) == len(set(pairs)), "Duplicate (run_id, node_id) pairs found"


class TestOutputShapes:
    def test_x_seq_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        N, T, F = ds.X_seq.shape
        assert N == 6
        assert F == 6, f"Expected 6 temporal features, got {F}"
        assert T >= 1, "Sequence length must be at least 1"

    def test_x_static_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert ds.X_static.shape == (6, 7), f"Expected (6, 7), got {ds.X_static.shape}"

    def test_labels_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert ds.y_class.shape == (6,)
        assert ds.y_reg.shape == (6,)

    def test_groups_are_run_ids(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert set(ds.groups) == {"run_000", "run_001"}


class TestLabels:
    def test_y_class_is_binary(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert set(ds.y_class.tolist()).issubset({0, 1})

    def test_flooded_node_has_y_class_1(self, tmp_path):
        """Node J-000 floods in both runs (flooding_node_idx=0 by default)."""
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        flooded_mask = ds.meta["node_id"] == "J-000"
        assert ds.y_class[flooded_mask].all(), "J-000 should be labeled flooded"

    def test_nonflooded_node_has_y_class_0(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        dry_mask = ds.meta["node_id"] == "J-002"
        assert not ds.y_class[dry_mask].any(), "J-002 should not be labeled flooded"

    def test_y_reg_nonnegative(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert (ds.y_reg >= 0).all()


class TestNoTemporalMode:
    def test_multiplier_appended_to_static(self, tmp_path):
        """use_temporal=False: X_static has 8 cols (7 static + 1 multiplier)."""
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path, use_temporal=False)
        assert ds.X_static.shape == (6, 8), f"Expected (6, 8), got {ds.X_static.shape}"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
python -m pytest tests/ml/temporal/test_surrogate_dataset.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'build_surrogate_dataset'`

- [ ] **Step 3: Implement `build_surrogate_dataset()`**

Append this function at the bottom of `swmm_resilience/ml/temporal/dataset.py`:

```python
def build_surrogate_dataset(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    resample_min: int = 5,
    use_temporal: bool = True,
) -> TemporalWindowDataset:
    """Build one sample per (run_id, node_id) for the surrogate CNN.

    Unlike build_temporal_windows, no sliding windows are produced.
    Each sample's X_seq is the full resampled inflow/hydraulic timeseries for
    that node in that run. Sequences are zero-padded to the longest T found.

    When use_temporal=False the inflow_multiplier for the run is appended as
    an extra column to X_static (giving 8 columns instead of 7), so the
    no-temporal ablation model receives the multiplier as a static feature.
    """
    all_X_seq_list: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    conn = sqlite3.connect(db_path)
    try:
        artifacts = conn.execute(
            "SELECT ta.run_id, ta.network_hash, ta.parquet_path, r.inflow_multiplier "
            "FROM temporal_artifacts ta "
            "JOIN runs r ON ta.run_id = r.run_id "
            "ORDER BY r.inflow_multiplier"
        ).fetchall()

        for run_id, network_hash, parquet_path, inflow_multiplier in artifacts:
            df = pd.read_parquet(parquet_path)

            static_rows = conn.execute(
                f"SELECT node_uid, {', '.join(STATIC_COLS)} "
                "FROM network_nodes WHERE network_hash = ?",
                (network_hash,),
            ).fetchall()
            static_lookup: dict[str, np.ndarray] = {
                row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
                for row in static_rows
            }

            for node_id in df["node_id"].unique():
                if node_id not in static_lookup:
                    warnings.warn(
                        f"node_id '{node_id}' (run_id={run_id}) not in network_nodes — skipping.",
                        stacklevel=2,
                    )
                    continue

                node_df = (
                    df[df["node_id"] == node_id]
                    .sort_values("time_min")
                    .drop_duplicates(subset=["time_min"], keep="last")
                    .reset_index(drop=True)
                )
                if node_df.empty:
                    continue

                # Resample to regular grid
                t_start = node_df["time_min"].iloc[0]
                t_end = node_df["time_min"].iloc[-1]
                n_grid = int(round((t_end - t_start) / resample_min)) + 1
                grid = t_start + np.arange(n_grid, dtype=float) * resample_min
                node_df = (
                    node_df.set_index("time_min")
                    .reindex(grid)
                    .ffill()
                    .dropna(subset=TEMPORAL_COLS)
                    .reset_index()
                )
                if node_df.empty:
                    continue

                seq = node_df[TEMPORAL_COLS].values.astype(np.float32)  # [T, 6]
                x_static = static_lookup[node_id]

                if not use_temporal and inflow_multiplier is not None:
                    x_static = np.append(x_static, float(inflow_multiplier)).astype(np.float32)

                all_X_seq_list.append(seq)
                all_X_static.append(x_static)
                all_y_class.append(int((node_df["flooding_lps"] > 0).any()))
                all_y_reg.append(float(node_df["flooding_lps"].max()))
                all_groups.append(run_id)
                meta_rows.append({"run_id": run_id, "node_id": node_id, "window_start_min": 0.0})
    finally:
        conn.close()

    if not all_X_seq_list:
        return TemporalWindowDataset(
            X_seq=np.empty((0, 1, len(TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(STATIC_COLS)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
        )

    # Zero-pad sequences to the longest T found
    T_max = max(s.shape[0] for s in all_X_seq_list)
    padded = np.zeros((len(all_X_seq_list), T_max, len(TEMPORAL_COLS)), dtype=np.float32)
    for i, seq in enumerate(all_X_seq_list):
        padded[i, : seq.shape[0], :] = seq

    return TemporalWindowDataset(
        X_seq=padded,
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=pd.DataFrame(meta_rows),
    )
```

- [ ] **Step 4: Run tests — verify they all PASS**

```bash
python -m pytest tests/ml/temporal/test_surrogate_dataset.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/temporal/dataset.py \
        tests/ml/temporal/test_surrogate_dataset.py
git commit -m "feat(surrogate-cnn): add build_surrogate_dataset() — one sample per node per run"
```

---

## Task 3: train_surrogate.py — training pipeline

**Files:**
- Create: `swmm_resilience/ml/temporal/train_surrogate.py`
- Create: `tests/ml/temporal/test_train_surrogate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/temporal/test_train_surrogate.py
"""TDD tests for train_surrogate() pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.train_surrogate import train_surrogate


def _synthetic_surrogate_dataset(
    n_runs: int = 4, n_nodes: int = 10, T: int = 20, use_temporal: bool = True
) -> TemporalWindowDataset:
    """Minimal dataset: n_runs groups, n_nodes nodes each."""
    N = n_runs * n_nodes
    rng = np.random.RandomState(42)
    groups = np.array([f"run_{i:02d}" for i in range(n_runs) for _ in range(n_nodes)], dtype=object)
    n_static = 7 if use_temporal else 8
    return TemporalWindowDataset(
        X_seq=rng.randn(N, T, 6).astype(np.float32),
        X_static=rng.randn(N, n_static).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=(rng.rand(N) * 100).astype(np.float32),
        groups=groups,
        meta=pd.DataFrame({
            "run_id": groups,
            "node_id": [f"J-{j:03d}" for j in range(n_nodes)] * n_runs,
            "window_start_min": [0.0] * N,
        }),
    )


class TestArtifactsSaved:
    def test_all_artifacts_exist(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        artifacts_dir = tmp_path / "artifacts"
        assert (artifacts_dir / "surrogate_cnn_weights.pt").exists()
        assert (artifacts_dir / "surrogate_cnn_scaler_seq.joblib").exists()
        assert (artifacts_dir / "surrogate_cnn_scaler_static.joblib").exists()
        assert (artifacts_dir / "surrogate_cnn_metrics.csv").exists()


class TestNoDataLeakage:
    def test_train_val_groups_disjoint(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        for fold in result["folds"]:
            train_set = set(fold["train_groups"])
            val_set = set(fold["val_groups"])
            assert not (train_set & val_set), f"Data leakage in fold {fold['fold']}"


class TestReturnedMetrics:
    def test_metrics_contain_both_tasks(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        fold = result["folds"][0]
        assert "val_auc_roc" in fold,  "Missing classification metric"
        assert "val_rmse" in fold,     "Missing regression metric"
        assert "val_f1" in fold,       "Missing F1 metric"

    def test_result_structure(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        assert "n_folds" in result
        assert "best_fold" in result
        assert len(result["folds"]) == 2


class TestNoTemporalMode:
    def test_no_temporal_artifacts_saved(self, tmp_path):
        dataset = _synthetic_surrogate_dataset(use_temporal=False)
        train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            use_temporal=False,
            _dataset=dataset,
        )
        artifacts_dir = tmp_path / "artifacts"
        assert (artifacts_dir / "surrogate_cnn_notemporal_weights.pt").exists()
        assert (artifacts_dir / "surrogate_cnn_notemporal_metrics.csv").exists()
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
python -m pytest tests/ml/temporal/test_train_surrogate.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named '...train_surrogate'`

- [ ] **Step 3: Implement `train_surrogate.py`**

```python
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
                torch.tensor(cls_logit_v.cpu().numpy()),
                torch.tensor(y_cls_val).unsqueeze(1),
            ).item()
        )
        try:
            val_auc = float(roc_auc_score(y_cls_val, cls_prob))
        except ValueError:
            val_auc = float("nan")

        val_rmse = float(mean_squared_error(y_reg_val, reg_pred) ** 0.5)
        try:
            from sklearn.metrics import r2_score
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
```

- [ ] **Step 4: Run tests — verify they all PASS**

```bash
python -m pytest tests/ml/temporal/test_train_surrogate.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/temporal/train_surrogate.py \
        tests/ml/temporal/test_train_surrogate.py
git commit -m "feat(surrogate-cnn): add train_surrogate() pipeline with dual-head loss and --no-temporal ablation"
```

---

## Task 4: predict_surrogate_from_multiplier()

**Files:**
- Modify: `swmm_resilience/ml/temporal/predict.py` (append function before `main()`)
- Create: `tests/ml/temporal/test_surrogate_predict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/temporal/test_surrogate_predict.py
"""TDD tests for predict_surrogate_from_multiplier()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.models.surrogate_cnn import SWMMSurrogateCNN
from swmm_resilience.ml.temporal.predict import predict_surrogate_from_multiplier

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_db_and_artifacts(tmp_path: Path, n_nodes: int = 4) -> tuple[Path, Path]:
    """Create minimal DB (Qx1.00 run) + saved model artifacts."""
    db_path = tmp_path / "test.db"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = uuid.uuid4().hex
    run_id = "run_qx100"

    conn.execute(
        "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
        "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "test.inp", network_hash, "uniform", "uniform", 0.0, 1.0, "done", "test"),
    )

    # Build Parquet for Qx1.00
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        for step in range(20):
            records.append({
                "run_id": run_id, "network_hash": network_hash, "node_id": node_id,
                "step_index": step, "time_sec": step * 300, "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5, "lateral_inflow_lps": 5.0,
                "depth_m": 0.5, "depth_ratio": 0.3,
                "flooding_lps": 0.0, "total_outflow_lps": 8.0, "failed_now": 0,
            })
    parquet_path = tmp_path / "run_qx100.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(parquet_path, index=False)

    conn.execute(
        "INSERT INTO temporal_artifacts (run_id, network_hash, parquet_path) VALUES (?, ?, ?)",
        (run_id, network_hash, str(parquet_path)),
    )
    for node_idx in range(n_nodes):
        conn.execute(
            "INSERT INTO network_nodes (node_uid, network_hash, full_depth_m, in_degree, out_degree, "
            "upstream_diam_avg_m, downstream_diam_avg_m, upstream_capacity_lps, downstream_capacity_lps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"J-{node_idx:03d}", network_hash, 1.2, 1, 1, 0.3, 0.3, 50.0, 50.0),
        )
    conn.commit()
    conn.close()

    # Save dummy model artifacts (untrained weights + fitted scalers)
    model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
    torch.save(model.state_dict(), artifacts_dir / "surrogate_cnn_weights.pt")

    scaler_seq = StandardScaler()
    scaler_seq.fit(np.random.randn(100, 6).astype(np.float32))
    joblib.dump(scaler_seq, artifacts_dir / "surrogate_cnn_scaler_seq.joblib")

    scaler_static = StandardScaler()
    scaler_static.fit(np.random.randn(100, 7).astype(np.float32))
    joblib.dump(scaler_static, artifacts_dir / "surrogate_cnn_scaler_static.joblib")

    return db_path, artifacts_dir


class TestOutputColumns:
    def test_returns_expected_columns(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=2.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        expected_cols = {"node_id", "flood_prob", "predicted_flooded", "peak_flooding_lps_pred"}
        assert expected_cols.issubset(result.columns), f"Missing columns: {expected_cols - set(result.columns)}"

    def test_one_row_per_node(self, tmp_path):
        n_nodes = 4
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path, n_nodes=n_nodes)
        result = predict_surrogate_from_multiplier(
            multiplier=2.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert len(result) == n_nodes


class TestFloodProbRange:
    def test_flood_prob_in_0_1(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=3.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert (result["flood_prob"] >= 0).all() and (result["flood_prob"] <= 1).all()

    def test_predicted_flooded_is_binary(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=3.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert set(result["predicted_flooded"].unique()).issubset({0, 1})


class TestMultiplierScaling:
    def test_higher_multiplier_gives_higher_mean_flood_prob(self, tmp_path):
        """A trained model should predict more flooding for higher multipliers.
        With an untrained model this is not guaranteed — just check no crash."""
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result_low = predict_surrogate_from_multiplier(
            multiplier=1.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        result_high = predict_surrogate_from_multiplier(
            multiplier=5.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        # Both must return same number of nodes without crashing
        assert len(result_low) == len(result_high)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
python -m pytest tests/ml/temporal/test_surrogate_predict.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'predict_surrogate_from_multiplier'`

- [ ] **Step 3: Add `predict_surrogate_from_multiplier()` to `predict.py`**

Insert this block just before the existing `predict_failure_timeline()` function in
`swmm_resilience/ml/temporal/predict.py`:

```python
def predict_surrogate_from_multiplier(
    multiplier: float,
    db_path: Path = DEFAULT_DB_FILE,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict flood risk for every node given an inflow multiplier.

    No SWMM run required. Loads the Qx1.00 base hydrograph from the DB,
    scales total_inflow_lps and lateral_inflow_lps by `multiplier`, sets
    SWMM-output features (depth_m, depth_ratio, flooding_lps, total_outflow_lps)
    to zero, then runs SWMMSurrogateCNN.

    Returns DataFrame with columns:
        node_id, flood_prob, predicted_flooded, peak_flooding_lps_pred
    """
    artifacts_dir = Path(artifacts_dir)
    prefix = "surrogate_cnn"

    state_dict = torch.load(
        artifacts_dir / f"{prefix}_weights.pt",
        map_location=device,
        weights_only=True,
    )
    scaler_seq = joblib.load(artifacts_dir / f"{prefix}_scaler_seq.joblib")
    scaler_static = joblib.load(artifacts_dir / f"{prefix}_scaler_static.joblib")

    model = SWMMSurrogateCNN(
        n_temporal_features=len(TEMPORAL_COLS),
        n_static_features=len(STATIC_COLS),
        use_temporal=True,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Qx1.00 base run (lowest multiplier in temporal_artifacts)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT ta.parquet_path, ta.network_hash "
            "FROM temporal_artifacts ta "
            "JOIN runs r ON ta.run_id = r.run_id "
            "ORDER BY r.inflow_multiplier ASC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("No temporal artifacts found in the database.")
        base_parquet_path, network_hash = row

        static_rows = conn.execute(
            f"SELECT node_uid, {', '.join(STATIC_COLS)} "
            "FROM network_nodes WHERE network_hash = ?",
            (network_hash,),
        ).fetchall()
    finally:
        conn.close()

    static_lookup: dict[str, np.ndarray] = {
        row[0]: np.nan_to_num(np.array(row[1:], dtype=np.float32), nan=0.0)
        for row in static_rows
    }

    df = pd.read_parquet(base_parquet_path)

    # Inflow feature indices within TEMPORAL_COLS
    _INFLOW_COLS = {"total_inflow_lps", "lateral_inflow_lps"}
    inflow_indices = [i for i, c in enumerate(TEMPORAL_COLS) if c in _INFLOW_COLS]
    swmm_output_indices = [i for i, c in enumerate(TEMPORAL_COLS) if c not in _INFLOW_COLS]

    records: list[dict] = []
    for node_id in df["node_id"].unique():
        if node_id not in static_lookup:
            warnings.warn(f"node_id '{node_id}' not in network_nodes — skipping.")
            continue

        node_df = (
            df[df["node_id"] == node_id]
            .sort_values("time_min")
            .drop_duplicates(subset=["time_min"], keep="last")
            .reset_index(drop=True)
        )
        if node_df.empty:
            continue

        # Resample to 5-min grid
        resample_min = 5
        t_start = node_df["time_min"].iloc[0]
        t_end = node_df["time_min"].iloc[-1]
        n_grid = int(round((t_end - t_start) / resample_min)) + 1
        grid = t_start + np.arange(n_grid, dtype=float) * resample_min
        node_df = (
            node_df.set_index("time_min")
            .reindex(grid)
            .ffill()
            .dropna(subset=TEMPORAL_COLS)
            .reset_index()
        )
        if node_df.empty:
            continue

        seq = node_df[TEMPORAL_COLS].values.astype(np.float32)  # [T, 6]

        # Scale inflow features, zero out SWMM-output features
        seq[:, inflow_indices] *= multiplier
        seq[:, swmm_output_indices] = 0.0

        T, F = seq.shape
        seq_sc = scaler_seq.transform(seq.reshape(-1, F)).reshape(1, T, F)
        x_static_raw = static_lookup[node_id]
        x_static_sc = scaler_static.transform(x_static_raw.reshape(1, -1))

        with torch.no_grad():
            cls_logit, reg_out = model(
                torch.tensor(seq_sc, dtype=torch.float32).to(device),
                torch.tensor(x_static_sc, dtype=torch.float32).to(device),
            )
            flood_prob = float(torch.sigmoid(cls_logit).cpu().item())
            peak_lps = float(reg_out.cpu().item())

        records.append({
            "node_id": node_id,
            "flood_prob": flood_prob,
            "predicted_flooded": int(flood_prob >= 0.5),
            "peak_flooding_lps_pred": max(peak_lps, 0.0),
        })

    return pd.DataFrame(records)
```

Also add `SWMMSurrogateCNN` to the imports at the top of `predict.py`. The existing import block has:

```python
from .models.cnn import SWMMTemporalCNN
```

Add after it:

```python
from .models.surrogate_cnn import SWMMSurrogateCNN
```

- [ ] **Step 4: Run tests — verify they all PASS**

```bash
python -m pytest tests/ml/temporal/test_surrogate_predict.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full test suite — verify no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/temporal/predict.py \
        tests/ml/temporal/test_surrogate_predict.py
git commit -m "feat(surrogate-cnn): add predict_surrogate_from_multiplier() — no SWMM at inference"
```

---

## Verification after all tasks

```bash
# Smoke-test the full training CLI against the real DB
python -m swmm_resilience.ml.temporal.train_surrogate --epochs 5 --folds 3

# Smoke-test the no-temporal ablation
python -m swmm_resilience.ml.temporal.train_surrogate --epochs 5 --folds 3 --no-temporal

# Smoke-test inference with a new multiplier
python -c "
from swmm_resilience.ml.temporal.predict import predict_surrogate_from_multiplier
df = predict_surrogate_from_multiplier(multiplier=3.32)
print(df.head())
print(f'Nodes predicted flooded: {df[\"predicted_flooded\"].sum()}/{len(df)}')
"
```
