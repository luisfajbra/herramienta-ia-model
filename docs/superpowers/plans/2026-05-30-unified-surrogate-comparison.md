# Unified Surrogate Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified dataset and comparison pipeline that trains XGBoost, CNN (full), and CNN (ablation) on identical features and identical GroupKFold splits — with no SWMM-output features — so their metrics are directly comparable.

**Architecture:** `build_unified_dataset()` reads the existing CSV (static features + targets) and joins inflow timeseries from Parquet files (one read per run). `compare_surrogate()` runs a single GroupKFold loop training all three models on the same splits and reporting a side-by-side metrics table.

**Tech Stack:** Python 3.11, PyTorch, xgboost, scikit-learn (GroupKFold, StandardScaler), pandas, SQLite, pytest.

---

## Background for agentic implementers

### Codebase conventions

- Package root: `swmm_resilience/`
- Temporal ML files: `swmm_resilience/ml/temporal/`
- Config: `swmm_resilience/config.py` — exports `DEFAULT_OUTPUT_CSV`, `DEFAULT_DB_FILE`, `DEFAULT_TEMPORAL_ARTIFACTS_DIR`
- Existing dataset builder: `swmm_resilience/ml/temporal/dataset.py` — append only, do not modify existing functions
- Schemas: `swmm_resilience/ml/temporal/schemas.py` — `TemporalWindowDataset` is a `@dataclass` with fields `X_seq, X_static, y_class, y_reg, groups, meta`
- Existing surrogate model: `swmm_resilience/ml/temporal/models/surrogate_cnn.py` — `SWMMSurrogateCNN(n_temporal_features, n_static_features, use_temporal)`
- Existing surrogate temporal cols: `SURROGATE_TEMPORAL_COLS = ["total_inflow_lps", "lateral_inflow_lps"]` in `dataset.py`
- Tests: `tests/ml/temporal/` — follow pattern in `test_surrogate_dataset.py`
- Python env for tests: `/opt/miniconda3/envs/py39/bin/pytest`

### Key data facts

- CSV (`DEFAULT_OUTPUT_CSV`): 2576 rows (16 runs × 161 nodes), one row per `(run_id, node_id)`
- `node_type` has 2 values: `junction` (majority), `outfall` — one-hot encodes to 1 binary column
- NaNs exist in `upstream_diam_avg_m`, `upstream_capacity_lps` (912 rows) and `downstream_diam_avg_m`, `downstream_capacity_lps` (16 rows) — fill with 0.0
- No NaNs in `flooded` or `peak_flooding_lps`
- Class balance: ~1712 non-flooded, ~864 flooded

### SWMM-output columns to drop from both models

```python
SWMM_OUTPUT_COLS = [
    "max_depth_m", "max_depth_ratio", "time_to_peak_min",
    "depth_rate_m_per_min", "max_total_outflow_lps",
    "time_to_peak_outflow_min", "downstream_link_peak_flows_lps_json",
]
```

### DB tables used

```sql
-- temporal_artifacts: run_id, parquet_path
```

---

## File map

| File | Action | Responsibility |
|------|--------|---------------|
| `swmm_resilience/ml/temporal/dataset.py` | **Modify** — append | `build_unified_dataset()` + `SWMM_OUTPUT_COLS` constant |
| `swmm_resilience/ml/temporal/compare_surrogate.py` | **Create** | Comparison pipeline + `main()` CLI |
| `tests/ml/temporal/test_unified_dataset.py` | **Create** | Dataset builder tests |
| `tests/ml/temporal/test_compare_surrogate.py` | **Create** | Comparison pipeline tests |

---

## Task 1: build_unified_dataset()

**Files:**
- Modify: `swmm_resilience/ml/temporal/dataset.py` (append at bottom, before `if __name__`)
- Create: `tests/ml/temporal/test_unified_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/temporal/test_unified_dataset.py`:

```python
"""TDD tests for build_unified_dataset()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import SWMM_OUTPUT_COLS, build_unified_dataset
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset

# ── helpers ──────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "run_id", "node_id", "network_hash", "network_file",
    "inflow_multiplier", "scenario_type", "spatial_pattern",
    "invert_elev_m", "full_depth_m", "base_inflow_lps", "node_type",
    "in_degree", "out_degree",
    "upstream_pipes_count", "upstream_diam_max_m", "upstream_diam_min_m",
    "upstream_diam_avg_m", "upstream_slope_avg", "upstream_slope_max",
    "upstream_capacity_lps",
    "downstream_pipes_count", "downstream_diam_max_m", "downstream_diam_min_m",
    "downstream_diam_avg_m", "downstream_slope_avg", "downstream_slope_max",
    "downstream_capacity_lps",
    "max_depth_m", "max_depth_ratio", "time_to_peak_min",
    "depth_rate_m_per_min", "max_total_outflow_lps", "time_to_peak_outflow_min",
    "downstream_link_peak_flows_lps_json",
    "flooded", "peak_flooding_lps", "flooding_duration_min",
]

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_csv(path: Path, n_runs: int = 2, n_nodes: int = 3) -> None:
    rows = []
    network_hash = uuid.uuid4().hex
    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        for j in range(n_nodes):
            node_id = f"J-{j:03d}"
            flooded = 1 if (j == 0 and i > 0) else 0
            rows.append({
                "run_id": run_id, "node_id": node_id,
                "network_hash": network_hash, "network_file": "test.inp",
                "inflow_multiplier": multiplier,
                "scenario_type": "uniform", "spatial_pattern": "uniform",
                "invert_elev_m": 10.0, "full_depth_m": 1.2,
                "base_inflow_lps": 5.0,
                "node_type": "outfall" if j == 0 else "junction",
                "in_degree": 1, "out_degree": 1,
                "upstream_pipes_count": 1,
                "upstream_diam_max_m": 0.3, "upstream_diam_min_m": 0.3,
                "upstream_diam_avg_m": 0.3,
                "upstream_slope_avg": 0.001, "upstream_slope_max": 0.002,
                "upstream_capacity_lps": 50.0,
                "downstream_pipes_count": 1,
                "downstream_diam_max_m": 0.3, "downstream_diam_min_m": 0.3,
                "downstream_diam_avg_m": 0.3,
                "downstream_slope_avg": 0.001, "downstream_slope_max": 0.002,
                "downstream_capacity_lps": 50.0,
                # SWMM outputs — should be dropped
                "max_depth_m": 0.9, "max_depth_ratio": 0.75,
                "time_to_peak_min": 30.0, "depth_rate_m_per_min": 0.01,
                "max_total_outflow_lps": 20.0, "time_to_peak_outflow_min": 35.0,
                "downstream_link_peak_flows_lps_json": "{}",
                # Targets
                "flooded": flooded, "peak_flooding_lps": 8.0 if flooded else 0.0,
                "flooding_duration_min": 15.0 if flooded else 0.0,
            })
    pd.DataFrame(rows, columns=_CSV_COLS).to_csv(path, index=False)


def _make_parquet(directory: Path, run_id: str, network_hash: str,
                  n_nodes: int = 3, n_steps: int = 10) -> Path:
    records = []
    for j in range(n_nodes):
        node_id = f"J-{j:03d}"
        for step in range(n_steps):
            records.append({
                "run_id": run_id, "network_hash": network_hash,
                "node_id": node_id, "step_index": step,
                "time_sec": step * 300, "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5, "depth_ratio": 0.3,
                "flooding_lps": 0.0, "total_outflow_lps": 8.0, "failed_now": 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _setup(tmp_path: Path, n_runs: int = 2, n_nodes: int = 3):
    csv_path = tmp_path / "dataset_ml.csv"
    _make_csv(csv_path, n_runs=n_runs, n_nodes=n_nodes)

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = pd.read_csv(csv_path)["network_hash"].iloc[0]
    parquet_dir = tmp_path / "parquets"

    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        conn.execute(
            "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
            "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, "test.inp", network_hash, "uniform", "uniform",
             0.0, multiplier, "done", "test"),
        )
        parquet_path = _make_parquet(parquet_dir, run_id, network_hash, n_nodes=n_nodes)
        conn.execute(
            "INSERT INTO temporal_artifacts (run_id, network_hash, parquet_path) "
            "VALUES (?, ?, ?)",
            (run_id, network_hash, str(parquet_path)),
        )
    conn.commit()
    conn.close()
    return csv_path, db_path


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSampleCount:
    def test_one_sample_per_node_per_run(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert ds.X_seq.shape[0] == 6, f"Expected 6, got {ds.X_seq.shape[0]}"

    def test_no_duplicates(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        pairs = list(zip(ds.meta["run_id"], ds.meta["node_id"]))
        assert len(pairs) == len(set(pairs))


class TestSwmmOutputsDropped:
    def test_swmm_output_cols_not_in_static(self, tmp_path):
        """X_static must not contain raw SWMM output values."""
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        # CSV has max_depth_m=0.9 for all rows; if it leaked, some column would be ~0.9
        # X_static columns from our CSV without SWMM outputs have values in [0, 50]
        # max_depth_m=0.9, max_depth_ratio=0.75 would appear clustered around these values
        # We verify the static feature count is correct (21 features)
        assert ds.X_static.shape[1] == 21, (
            f"Expected 21 static features, got {ds.X_static.shape[1]}. "
            "SWMM output columns may have leaked."
        )


class TestOutputShapes:
    def test_x_seq_shape(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        N, T, F = ds.X_seq.shape
        assert N == 6
        assert F == 2, f"Expected 2 temporal features (inflow only), got {F}"
        assert T >= 1

    def test_x_static_shape(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert ds.X_static.shape == (6, 21)

    def test_groups_are_run_ids(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert set(ds.groups) == {"run_000", "run_001"}

    def test_no_nans_in_static(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert not np.isnan(ds.X_static).any(), "NaNs found in X_static"


class TestLabels:
    def test_y_class_binary(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert set(ds.y_class.tolist()).issubset({0, 1})

    def test_flooded_node_labeled_correctly(self, tmp_path):
        """In the fixture, J-000 floods in run_001 (i>0)."""
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        mask = (ds.meta["node_id"] == "J-000") & (ds.meta["run_id"] == "run_001")
        assert ds.y_class[mask].all(), "J-000 in run_001 should be labeled flooded"

    def test_y_reg_nonnegative(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert (ds.y_reg >= 0).all()
```

- [ ] **Step 2: Run tests to verify they FAIL**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_unified_dataset.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SWMM_OUTPUT_COLS'` or `'build_unified_dataset'`

- [ ] **Step 3: Implement build_unified_dataset()**

Append the following at the bottom of `swmm_resilience/ml/temporal/dataset.py`,
**before** the `if __name__ == "__main__":` block:

```python
SWMM_OUTPUT_COLS: list[str] = [
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "max_total_outflow_lps",
    "time_to_peak_outflow_min",
    "downstream_link_peak_flows_lps_json",
]

_UNIFIED_META_COLS: list[str] = [
    "network_hash",
    "network_file",
    "scenario_type",
    "spatial_pattern",
    "flooding_duration_min",
]

_UNIFIED_TARGET_COLS: list[str] = ["flooded", "peak_flooding_lps"]


def build_unified_dataset(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    db_path: Path = DEFAULT_DB_FILE,
    resample_min: int = 5,
) -> TemporalWindowDataset:
    """Build one sample per (run_id, node_id) for the unified RF vs CNN comparison.

    Reads static features from the CSV (dropping SWMM-output columns that are not
    available before running SWMM). Joins inflow timeseries from the Parquet files
    registered in temporal_artifacts (one Parquet read per run).

    The returned dataset is shared by both models:
    - RF/XGBoost: uses X_static directly (21 inference-available features)
    - CNN: uses X_seq (inflow timeseries [T, 2]) + X_static

    Args:
        csv_path: Path to the ML dataset CSV (output of export_ml_dataset).
        db_path:  SQLite database path (for temporal_artifacts parquet lookup).
        resample_min: Temporal resampling interval in minutes.
    """
    csv_path = Path(csv_path)
    df_csv = pd.read_csv(csv_path)

    # ── static feature matrix ────────────────────────────────────────────────
    drop_cols = set(SWMM_OUTPUT_COLS) | set(_UNIFIED_META_COLS) | set(_UNIFIED_TARGET_COLS)
    feat_df = df_csv.drop(columns=[c for c in drop_cols if c in df_csv.columns])

    # One-hot encode node_type (junction/outfall → 1 binary column)
    if "node_type" in feat_df.columns:
        feat_df = pd.get_dummies(feat_df, columns=["node_type"], drop_first=True)

    # Fill NaNs (source nodes have no upstream; outfalls have no downstream pipes)
    feat_df = feat_df.fillna(0.0)

    feature_cols: list[str] = [
        c for c in feat_df.columns if c not in ("run_id", "node_id")
    ]

    # ── parquet lookup: run_id → path ────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    try:
        parquet_rows = conn.execute(
            "SELECT run_id, parquet_path FROM temporal_artifacts"
        ).fetchall()
    finally:
        conn.close()
    parquet_lookup: dict[str, str] = {rid: ppath for rid, ppath in parquet_rows}

    # ── build samples: one Parquet read per run ──────────────────────────────
    all_X_seq: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    for run_id, run_group in df_csv.groupby("run_id", sort=False):
        parquet_path = parquet_lookup.get(str(run_id))
        if parquet_path is None:
            warnings.warn(f"run_id '{run_id}' has no temporal_artifact — skipping.", stacklevel=2)
            continue

        parquet_df = pd.read_parquet(parquet_path)

        for csv_idx in run_group.index:
            csv_row = df_csv.loc[csv_idx]
            node_id = str(csv_row["node_id"])

            node_df = (
                parquet_df[parquet_df["node_id"] == node_id]
                .sort_values("time_min")
                .drop_duplicates(subset=["time_min"], keep="last")
                .reset_index(drop=True)
            )
            if node_df.empty:
                continue

            # Resample to regular grid via forward-fill
            t_start = node_df["time_min"].iloc[0]
            t_end = node_df["time_min"].iloc[-1]
            n_grid = int(round((t_end - t_start) / resample_min)) + 1
            grid = t_start + np.arange(n_grid, dtype=float) * resample_min
            node_df = (
                node_df.set_index("time_min")
                .reindex(grid)
                .ffill()
                .dropna(subset=SURROGATE_TEMPORAL_COLS)
                .reset_index()
            )
            if node_df.empty:
                continue

            seq = node_df[SURROGATE_TEMPORAL_COLS].values.astype(np.float32)
            x_static = feat_df.loc[csv_idx, feature_cols].values.astype(np.float32)

            all_X_seq.append(seq)
            all_X_static.append(x_static)
            all_y_class.append(int(csv_row["flooded"]))
            all_y_reg.append(float(csv_row["peak_flooding_lps"]))
            all_groups.append(str(run_id))
            meta_rows.append({"run_id": str(run_id), "node_id": node_id, "window_start_min": 0.0})

    if not all_X_seq:
        return TemporalWindowDataset(
            X_seq=np.empty((0, 1, len(SURROGATE_TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(feature_cols)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
        )

    # Zero-pad sequences to T_max
    T_max = max(s.shape[0] for s in all_X_seq)
    F = len(SURROGATE_TEMPORAL_COLS)
    padded = np.zeros((len(all_X_seq), T_max, F), dtype=np.float32)
    for i, seq in enumerate(all_X_seq):
        padded[i, : seq.shape[0], :] = seq

    meta_df = pd.DataFrame(meta_rows)
    meta_df.attrs["static_feature_names"] = feature_cols

    return TemporalWindowDataset(
        X_seq=padded,
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=meta_df,
    )
```

- [ ] **Step 4: Run tests to verify they PASS**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_unified_dataset.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Verify no regressions**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/ -q 2>&1 | tail -5
```

Expected: 40 passed (pre-existing) + 9 new = 49 passed.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/temporal/dataset.py tests/ml/temporal/test_unified_dataset.py
git commit -m "feat(comparison): add build_unified_dataset() — shared CSV+Parquet dataset for RF vs CNN comparison"
```

---

## Task 2: compare_surrogate.py — comparison pipeline

**Files:**
- Create: `swmm_resilience/ml/temporal/compare_surrogate.py`
- Create: `tests/ml/temporal/test_compare_surrogate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/temporal/test_compare_surrogate.py`:

```python
"""TDD tests for compare_surrogate() pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.compare_surrogate import compare_surrogate


def _synthetic_dataset(
    n_runs: int = 4, n_nodes: int = 12, T: int = 20, n_static: int = 21
) -> TemporalWindowDataset:
    """Minimal dataset: n_runs groups, n_nodes samples each."""
    N = n_runs * n_nodes
    rng = np.random.RandomState(0)
    groups = np.array(
        [f"run_{i:02d}" for i in range(n_runs) for _ in range(n_nodes)], dtype=object
    )
    meta = pd.DataFrame({
        "run_id": groups,
        "node_id": [f"J-{j:03d}" for j in range(n_nodes)] * n_runs,
        "window_start_min": [0.0] * N,
    })
    meta.attrs["static_feature_names"] = [f"feat_{i}" for i in range(n_static)]
    return TemporalWindowDataset(
        X_seq=rng.randn(N, T, 2).astype(np.float32),
        X_static=rng.randn(N, n_static).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=(rng.rand(N) * 50).astype(np.float32),
        groups=groups,
        meta=meta,
    )


class TestReturnShape:
    def test_returns_dataframe(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_fold(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert len(result) == 2


class TestMetricColumns:
    def test_xgb_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["xgb_auc_roc", "xgb_f1", "xgb_precision", "xgb_recall", "xgb_accuracy"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_cnn_full_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["cnn_auc_roc", "cnn_f1", "cnn_rmse"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_cnn_ablation_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["cnn_abl_auc_roc", "cnn_abl_f1", "cnn_abl_rmse"]:
            assert col in result.columns, f"Missing column: {col}"


class TestNoDataLeakage:
    def test_train_val_groups_disjoint(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for _, row in result.iterrows():
            train_set = set(row["train_groups"])
            val_set = set(row["val_groups"])
            assert not (train_set & val_set), f"Leakage in fold {row['fold']}"


class TestArtifactsSaved:
    def test_csv_saved(self, tmp_path):
        ds = _synthetic_dataset()
        artifacts_dir = tmp_path / "artifacts"
        compare_surrogate(
            artifacts_dir=artifacts_dir,
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert (artifacts_dir / "comparison_results.csv").exists()
```

- [ ] **Step 2: Run tests to verify they FAIL**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_compare_surrogate.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named '...compare_surrogate'`

- [ ] **Step 3: Implement compare_surrogate.py**

Create `swmm_resilience/ml/temporal/compare_surrogate.py`:

```python
# swmm_resilience/ml/temporal/compare_surrogate.py
"""Unified comparison: XGBoost vs CNN (full) vs CNN (ablation) — same data, same folds."""
from __future__ import annotations

import argparse
from pathlib import Path

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

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from ...config import DEFAULT_DB_FILE, DEFAULT_OUTPUT_CSV, DEFAULT_TEMPORAL_ARTIFACTS_DIR
from .dataset import build_unified_dataset
from .models.surrogate_cnn import SWMMSurrogateCNN
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


def _train_eval_cnn(
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

    model = SWMMSurrogateCNN(
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

    All models use the same inference-available features (no SWMM outputs).
    Returns a DataFrame with per-fold metrics for all three models.
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
        )
        xgb.fit(Xtr_sc, y_cls_tr.astype(int))
        xgb_prob = xgb.predict_proba(Xval_sc)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        for k, v in _cls_metrics(y_cls_val, xgb_pred, xgb_prob).items():
            row[f"xgb_{k}"] = v

        # ── CNN full ─────────────────────────────────────────────────────────
        cnn_m = _train_eval_cnn(
            X_seq_tr, X_static_tr, y_cls_tr, y_reg_tr,
            X_seq_val, X_static_val, y_cls_val, y_reg_val,
            use_temporal=True,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, device=device,
        )
        for k, v in cnn_m.items():
            row[f"cnn_{k}"] = v

        # ── CNN ablation ──────────────────────────────────────────────────────
        abl_m = _train_eval_cnn(
            X_seq_tr, X_static_tr, y_cls_tr, y_reg_tr,
            X_seq_val, X_static_val, y_cls_val, y_reg_val,
            use_temporal=False,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, device=device,
        )
        for k, v in abl_m.items():
            row[f"cnn_abl_{k}"] = v

        fold_rows.append(row)
        print(
            f"Fold {fold_i}: XGB F1={row['xgb_f1']:.3f}  "
            f"CNN F1={row['cnn_f1']:.3f}  Ablation F1={row['cnn_abl_f1']:.3f}"
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
    print(f"\n{'Metric':<14} {'XGBoost':>10} {'CNN Full':>10} {'CNN Ablation':>14}")
    print("-" * 52)
    for m in metrics:
        xgb = results[f"xgb_{m}"].mean()
        cnn = results[f"cnn_{m}"].mean()
        abl = results[f"cnn_abl_{m}"].mean()
        print(f"  {m:<12} {xgb:>10.4f} {cnn:>10.4f} {abl:>14.4f}")

    print(f"\nResults saved to comparison_results.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they PASS**

```bash
cd /Users/luis/herramienta-ia-model
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/test_compare_surrogate.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Verify no regressions**

```bash
/opt/miniconda3/envs/py39/bin/pytest tests/ml/temporal/ -q 2>&1 | tail -5
```

Expected: 49 + 8 = 57 passed.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/temporal/compare_surrogate.py \
        tests/ml/temporal/test_compare_surrogate.py
git commit -m "feat(comparison): add compare_surrogate() — XGBoost vs CNN on identical GroupKFold splits"
```

---

## Verification after all tasks

Run the full comparison on real data (takes ~10 min):

```bash
/opt/miniconda3/envs/py39/bin/python -m swmm_resilience.ml.temporal.compare_surrogate \
    --epochs 100 --folds 5
```

Expected output format:
```
=== Unified Surrogate Comparison ===
Fold 0: XGB F1=0.xxx  CNN F1=0.xxx  Ablation F1=0.xxx
...
Metric         XGBoost   CNN Full  CNN Ablation
----------------------------------------------------
  auc_roc         0.xxxx     0.xxxx         0.xxxx
  f1              0.xxxx     0.xxxx         0.xxxx
  ...
```
