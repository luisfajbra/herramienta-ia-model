# SP2 — Constructor de Ventanas Temporales — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `build_temporal_windows()` in `swmm_resilience/ml/temporal/dataset.py` to read all Parquets from `temporal_artifacts`, build sliding windows, join static features, and return a `TemporalWindowDataset` with numpy arrays ready for CNN/LSTM training.

**Architecture:** The function opens the SQLite DB, iterates over each Parquet registered in `temporal_artifacts`, resamples each node's time series to a regular 5-minute grid via forward-fill, applies a sliding window algorithm (window=4 steps, horizon=1 step, step=1 step), joins 7 static features per node from `network_nodes`, and returns a `TemporalWindowDataset` dataclass. Normalization is NOT done here — that belongs to the training pipeline.

**Tech Stack:** Python 3.11+, pandas, numpy, sqlite3 (stdlib), pyarrow (parquet), pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `tests/ml/__init__.py` | Create | Package marker |
| `tests/ml/temporal/__init__.py` | Create | Package marker |
| `tests/ml/temporal/test_window_builder.py` | Create | 6 TDD tests with in-memory fixtures |
| `swmm_resilience/ml/temporal/schemas.py` | Modify | Add `TemporalWindowDataset` dataclass + `resample_min` to `TemporalWindowSpec` |
| `swmm_resilience/ml/temporal/dataset.py` | Modify | Replace placeholder + update signature + add constants + CLI block |

---

## Key Schema Facts (read before implementing)

- `network_nodes` stores the static per-node features. The node identifier column is **`node_uid`** (not `node_id`).
- Parquet files store per-timestep records. The node identifier column is **`node_id`**.
- The join between Parquet and `network_nodes` is: `network_nodes.node_uid = parquet.node_id` scoped by `network_hash`.
- `temporal_artifacts` columns: `artifact_id`, `run_id`, `network_hash`, `parquet_path`, `node_count`, `step_count`, `created_at`.
- Parquet columns (from `REQUIRED_TIMESERIES_COLUMNS`): `run_id`, `network_hash`, `node_id`, `step_index`, `time_sec`, `time_min`, `total_inflow_lps`, `lateral_inflow_lps`, `depth_m`, `depth_ratio`, `flooding_lps`, `total_outflow_lps`, `failed_now`.

---

## Task 1: Test infrastructure + 6 failing tests

**Files:**
- Create: `tests/ml/__init__.py`
- Create: `tests/ml/temporal/__init__.py`
- Create: `tests/ml/temporal/test_window_builder.py`

- [ ] **Step 1: Create package markers**

```bash
touch /path/to/project/tests/ml/__init__.py
touch /path/to/project/tests/ml/temporal/__init__.py
```

Both files are empty.

- [ ] **Step 2: Write the test file**

Full content of `tests/ml/temporal/test_window_builder.py`:

```python
"""Tests for build_temporal_windows() — written before implementation (TDD)."""

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import build_temporal_windows
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset, TemporalWindowSpec

# ── shared fixture helpers ────────────────────────────────────────────────────

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
    n_nodes: int,
    n_steps: int,
    flooding_step: int | None = None,
) -> Path:
    """Synthetic Parquet: n_nodes × n_steps rows at 5-min intervals."""
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        for step in range(n_steps):
            flooding = 5.0 if flooding_step is not None and step == flooding_step else 0.0
            records.append({
                "run_id": run_id,
                "network_hash": network_hash,
                "node_id": node_id,
                "step_index": step,
                "time_sec": step * 300,
                "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.1,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5 + step * 0.01,
                "depth_ratio": 0.3,
                "flooding_lps": flooding,
                "total_outflow_lps": 8.0,
                "failed_now": 1 if flooding > 0 else 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _insert_run(conn: sqlite3.Connection, run_id: str, network_hash: str) -> None:
    conn.execute(
        """INSERT INTO runs
           (run_id, network_file, network_hash, scenario_type, spatial_pattern,
            delta_inflow_lps, inflow_multiplier, input_source, executed_at, status)
           VALUES (?, 'test.inp', ?, 'hydrograph', 'uniform', 0.0, 1.0,
                   'hydrograph', datetime('now'), 'completed')""",
        (run_id, network_hash),
    )


def _insert_artifact(
    conn: sqlite3.Connection,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    n_nodes: int,
    n_steps: int,
) -> None:
    conn.execute(
        """INSERT INTO temporal_artifacts
           (artifact_id, run_id, network_hash, parquet_path,
            node_count, step_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        (str(uuid.uuid4()), run_id, network_hash, str(parquet_path), n_nodes, n_steps),
    )


def _insert_nodes(conn: sqlite3.Connection, network_hash: str, n_nodes: int) -> None:
    for i in range(n_nodes):
        conn.execute(
            """INSERT INTO network_nodes
               (network_hash, node_uid, full_depth_m, in_degree, out_degree,
                upstream_diam_avg_m, downstream_diam_avg_m,
                upstream_capacity_lps, downstream_capacity_lps)
               VALUES (?, ?, 2.0, 2, 1, 0.3, 0.25, 50.0, 40.0)""",
            (network_hash, f"J-{i:03d}"),
        )


def _make_db(
    tmp_path: Path,
    *,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    n_nodes: int,
    n_steps: int,
) -> Path:
    """Single-run SQLite DB with temporal_artifact + network_nodes rows."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    _insert_run(conn, run_id, network_hash)
    _insert_artifact(conn, run_id, network_hash, parquet_path, n_nodes, n_steps)
    _insert_nodes(conn, network_hash, n_nodes)
    conn.commit()
    conn.close()
    return db_path


_SPEC = TemporalWindowSpec(window_min=20, horizon_min=5, step_min=5, resample_min=5)

# ── tests ─────────────────────────────────────────────────────────────────────


class TestWindowShape:
    def test_build_windows_produces_correct_shape(self, tmp_path):
        """2 nodes × 20 steps → X_seq is [N, 4, 6], X_static is [N, 7]."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_shape"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=2, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=2, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert ds.X_seq.ndim == 3
        assert ds.X_seq.shape[1] == 4   # window_min / resample_min = 20 / 5
        assert ds.X_seq.shape[2] == 6   # 6 temporal features
        assert ds.X_static.ndim == 2
        assert ds.X_static.shape[1] == 7  # 7 static features
        n = ds.X_seq.shape[0]
        assert ds.y_class.shape == (n,)
        assert ds.y_reg.shape == (n,)
        assert ds.groups.shape == (n,)
        assert len(ds.meta) == n


class TestNoLeakage:
    def test_no_leakage_between_runs(self, tmp_path):
        """Two run_ids → groups contains exactly 2 distinct values."""
        network_hash = "hash_leakage"
        run_id_1 = str(uuid.uuid4())
        run_id_2 = str(uuid.uuid4())
        pq1 = _make_parquet(tmp_path / "r1", run_id_1, network_hash, n_nodes=1, n_steps=20)
        pq2 = _make_parquet(tmp_path / "r2", run_id_2, network_hash, n_nodes=1, n_steps=20)

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        for run_id, pq in [(run_id_1, pq1), (run_id_2, pq2)]:
            _insert_run(conn, run_id, network_hash)
            _insert_artifact(conn, run_id, network_hash, pq, 1, 20)
        _insert_nodes(conn, network_hash, 1)
        conn.commit()
        conn.close()

        ds = build_temporal_windows(db_path=db_path, window_spec=_SPEC)

        assert ds.X_seq.shape[0] > 0
        assert len(set(ds.groups)) == 2


class TestIncompleteWindow:
    def test_incomplete_window_discarded(self, tmp_path):
        """3 steps × 5 min = 15 min < window_min=20 → 0 samples, no error."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_incomplete"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=3)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=3,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert ds.X_seq.shape[0] == 0


class TestHorizonLabel:
    def test_failure_within_horizon_label(self, tmp_path):
        """Flooding at step 4 → y_class[0] == 1 (first window sees it in horizon)."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_horizon_pos"
        # window = steps 0-3, horizon = step 4 (where flooding_step=4)
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20, flooding_step=4)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert int(ds.y_class[0]) == 1

    def test_no_failure_within_horizon_label(self, tmp_path):
        """No flooding anywhere → all y_class == 0."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_horizon_neg"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert all(int(y) == 0 for y in ds.y_class)


class TestStaticFeatures:
    def test_static_features_joined_correctly(self, tmp_path):
        """X_static values match what was inserted into network_nodes."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_static"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        # _insert_nodes sets: full_depth_m=2.0, in_degree=2, out_degree=1,
        # upstream_diam_avg_m=0.3, downstream_diam_avg_m=0.25,
        # upstream_capacity_lps=50.0, downstream_capacity_lps=40.0
        expected = np.array([2.0, 2.0, 1.0, 0.3, 0.25, 50.0, 40.0], dtype=np.float32)
        np.testing.assert_allclose(ds.X_static[0], expected)
```

- [ ] **Step 3: Run tests to confirm they all fail**

```bash
python -m pytest tests/ml/temporal/test_window_builder.py -v
```

Expected: All 6 tests FAIL. Acceptable failures:
- `ImportError: cannot import name 'TemporalWindowDataset'` (schemas not updated yet)
- `NotImplementedError` (placeholder still in place)

- [ ] **Step 4: Commit**

```bash
git add tests/ml/__init__.py tests/ml/temporal/__init__.py tests/ml/temporal/test_window_builder.py
git commit -m "test(sp2): add 6 failing TDD tests for build_temporal_windows"
```

---

## Task 2: Schema update — `TemporalWindowDataset` + `resample_min`

**Files:**
- Modify: `swmm_resilience/ml/temporal/schemas.py`

The current file does NOT import numpy or pandas. We add both and add the new dataclass plus the new field.

- [ ] **Step 1: Read the current schemas.py**

Read `swmm_resilience/ml/temporal/schemas.py` (already done during planning — see File Map above).

- [ ] **Step 2: Write the updated schemas.py**

Full replacement content:

```python
"""
Shared schemas for the planned temporal/CNN workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RESULTS_DIR,
    ML_TEMPORAL_HORIZON_MIN,
    ML_TEMPORAL_RESAMPLE_MIN,
    ML_TEMPORAL_STEP_MIN,
    ML_TEMPORAL_TARGET,
    ML_TEMPORAL_WINDOW_MIN,
)


@dataclass(frozen=True)
class TemporalWindowSpec:
    """Configuration for rolling temporal windows."""

    window_min: int = ML_TEMPORAL_WINDOW_MIN
    horizon_min: int = ML_TEMPORAL_HORIZON_MIN
    step_min: int = ML_TEMPORAL_STEP_MIN
    resample_min: int = ML_TEMPORAL_RESAMPLE_MIN
    target: str = ML_TEMPORAL_TARGET


@dataclass
class TemporalWindowDataset:
    """Output of build_temporal_windows().

    Arrays are raw (unscaled). Normalisation is the caller's responsibility.
    """

    X_seq: np.ndarray      # [N, timesteps, temporal_features]  float32
    X_static: np.ndarray   # [N, static_features]               float32
    y_class: np.ndarray    # [N]  int8  — failure_within_horizon (0 or 1)
    y_reg: np.ndarray      # [N]  float32 — peak_flooding_lps in horizon
    groups: np.ndarray     # [N]  object  — run_id string, for GroupKFold
    meta: pd.DataFrame     # columns: run_id, node_id, window_start_min


@dataclass(frozen=True)
class TemporalDatasetSpec:
    """Paths and metadata for temporal datasets."""

    source_csv: Path = DEFAULT_OUTPUT_CSV
    output_dir: Path = DEFAULT_RESULTS_DIR / "temporal"
    dataset_name: str = "temporal_windows"

    @property
    def output_csv(self) -> Path:
        return self.output_dir / f"{self.dataset_name}.csv"
```

- [ ] **Step 3: Run the failing tests again — error should change**

```bash
python -m pytest tests/ml/temporal/test_window_builder.py -v
```

Expected: Tests now fail with `NotImplementedError` (no longer `ImportError`). That confirms `TemporalWindowDataset` is importable.

- [ ] **Step 4: Commit**

```bash
git add swmm_resilience/ml/temporal/schemas.py
git commit -m "feat(sp2): add TemporalWindowDataset dataclass and resample_min to TemporalWindowSpec"
```

---

## Task 3: Core implementation of `build_temporal_windows()`

**Files:**
- Modify: `swmm_resilience/ml/temporal/dataset.py`

Replace the placeholder. The new function:
1. Queries `temporal_artifacts` for all Parquet paths (ordered by `created_at`).
2. For each Parquet, loads static features for the run's `network_hash` from `network_nodes`.
3. For each `node_id` in the Parquet, resamples to a regular `resample_min`-minute grid via forward-fill.
4. Slides a window of `window_steps` steps with stride `step_steps`, requires a full horizon of `horizon_steps` after each window.
5. Collects arrays and returns `TemporalWindowDataset`.

**Column order for static features** (must match `X_static.shape[1] == 7`):
`full_depth_m`, `in_degree`, `out_degree`, `upstream_diam_avg_m`, `downstream_diam_avg_m`, `upstream_capacity_lps`, `downstream_capacity_lps`

- [ ] **Step 1: Read the current dataset.py**

Read `swmm_resilience/ml/temporal/dataset.py` (already done during planning).

- [ ] **Step 2: Write the updated dataset.py**

Full replacement content:

```python
"""
Temporal dataset helpers for hydrograph/CNN experiments.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT_DB_FILE, NETWORKS_DIR
from .schemas import TemporalDatasetSpec, TemporalWindowDataset, TemporalWindowSpec


REQUIRED_TIMESERIES_COLUMNS = [
    "run_id",
    "network_hash",
    "node_id",
    "step_index",
    "time_sec",
    "time_min",
    "total_inflow_lps",
    "lateral_inflow_lps",
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
    "failed_now",
]

TEMPORAL_COLS = [
    "total_inflow_lps",
    "lateral_inflow_lps",
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
]

STATIC_COLS = [
    "full_depth_m",
    "in_degree",
    "out_degree",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "upstream_capacity_lps",
    "downstream_capacity_lps",
]


def expected_timeseries_columns() -> list[str]:
    """Return the columns the temporal dataset builder expects."""
    return REQUIRED_TIMESERIES_COLUMNS.copy()


def save_node_timeseries_parquet(records: list[dict], output_path: str | Path) -> Path:
    """Persist node-level timestep records to Parquet with a stable column order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame.from_records(records, columns=REQUIRED_TIMESERIES_COLUMNS)
    missing_columns = [
        column for column in REQUIRED_TIMESERIES_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias en node_timeseries: "
            + ", ".join(missing_columns)
        )

    try:
        dataframe.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "No se pudo guardar node_timeseries en Parquet. Instala 'pyarrow' "
            "o 'fastparquet' para habilitar este formato."
        ) from exc

    return output_path


def build_temporal_windows(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    window_spec: TemporalWindowSpec | None = None,
    dataset_spec: TemporalDatasetSpec | None = None,
) -> TemporalWindowDataset:
    """Build sliding temporal windows from all Parquets in temporal_artifacts.

    Reads each registered Parquet, resamples to resample_min-minute intervals
    via forward-fill, and produces a sliding window dataset. Static features
    are joined from network_nodes using (network_hash, node_id == node_uid).
    No normalisation is applied — that is the caller's responsibility.
    """
    window_spec = window_spec or TemporalWindowSpec()

    resample_min = window_spec.resample_min
    window_steps = window_spec.window_min // resample_min
    horizon_steps = window_spec.horizon_min // resample_min
    step_steps = window_spec.step_min // resample_min

    all_X_seq: list[np.ndarray] = []
    all_X_static: list[np.ndarray] = []
    all_y_class: list[int] = []
    all_y_reg: list[float] = []
    all_groups: list[str] = []
    meta_rows: list[dict] = []

    conn = sqlite3.connect(db_path)
    try:
        artifacts = conn.execute(
            "SELECT run_id, network_hash, parquet_path "
            "FROM temporal_artifacts ORDER BY created_at"
        ).fetchall()

        for run_id, network_hash, parquet_path in artifacts:
            df = pd.read_parquet(parquet_path)

            # Load static lookup: node_uid → float32 vector [7]
            static_rows = conn.execute(
                f"""SELECT node_uid, {', '.join(STATIC_COLS)}
                    FROM network_nodes
                    WHERE network_hash = ?""",
                (network_hash,),
            ).fetchall()
            static_lookup: dict[str, np.ndarray] = {
                row[0]: np.array(row[1:], dtype=np.float32)
                for row in static_rows
            }

            for node_id in df["node_id"].unique():
                if node_id not in static_lookup:
                    continue
                x_static = static_lookup[node_id]

                node_df = (
                    df[df["node_id"] == node_id]
                    .sort_values("time_min")
                    .reset_index(drop=True)
                )
                if node_df.empty:
                    continue

                # Resample to regular resample_min-minute grid via forward-fill
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

                n = len(node_df)
                i = 0
                while i + window_steps + horizon_steps <= n:
                    window = node_df.iloc[i : i + window_steps]
                    horizon = node_df.iloc[i + window_steps : i + window_steps + horizon_steps]

                    all_X_seq.append(window[TEMPORAL_COLS].values.astype(np.float32))
                    all_X_static.append(x_static)
                    all_y_class.append(int((horizon["flooding_lps"] > 0).any()))
                    all_y_reg.append(float(horizon["flooding_lps"].max()))
                    all_groups.append(run_id)
                    meta_rows.append({
                        "run_id": run_id,
                        "node_id": node_id,
                        "window_start_min": float(node_df["time_min"].iloc[i]),
                    })
                    i += step_steps
    finally:
        conn.close()

    if not all_X_seq:
        return TemporalWindowDataset(
            X_seq=np.empty((0, window_steps, len(TEMPORAL_COLS)), dtype=np.float32),
            X_static=np.empty((0, len(STATIC_COLS)), dtype=np.float32),
            y_class=np.empty(0, dtype=np.int8),
            y_reg=np.empty(0, dtype=np.float32),
            groups=np.empty(0, dtype=object),
            meta=pd.DataFrame(columns=["run_id", "node_id", "window_start_min"]),
        )

    return TemporalWindowDataset(
        X_seq=np.stack(all_X_seq),
        X_static=np.stack(all_X_static),
        y_class=np.array(all_y_class, dtype=np.int8),
        y_reg=np.array(all_y_reg, dtype=np.float32),
        groups=np.array(all_groups, dtype=object),
        meta=pd.DataFrame(meta_rows),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnosticar dataset temporal de ventanas.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_FILE)
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen del dataset.")
    args = parser.parse_args()

    if args.summary:
        _conn = sqlite3.connect(args.db)
        n_parquets = _conn.execute("SELECT COUNT(*) FROM temporal_artifacts").fetchone()[0]
        _conn.close()
        print(f"Parquets registrados: {n_parquets}")

        print("Construyendo ventanas...")
        _ds = build_temporal_windows(db_path=args.db)
        _n = _ds.X_seq.shape[0]
        print(f"Total de muestras: {_n}")
        if _n > 0:
            _pos = int(_ds.y_class.sum())
            _neg = _n - _pos
            print(f"  failure_within_horizon=1: {_pos} ({100 * _pos / _n:.1f}%)")
            print(f"  failure_within_horizon=0: {_neg} ({100 * _neg / _n:.1f}%)")
        else:
            print("No se generaron muestras.")
    else:
        parser.print_help()
```

- [ ] **Step 3: Run all 6 tests — they must all pass**

```bash
python -m pytest tests/ml/temporal/test_window_builder.py -v
```

Expected output:
```
PASSED tests/ml/temporal/test_window_builder.py::TestWindowShape::test_build_windows_produces_correct_shape
PASSED tests/ml/temporal/test_window_builder.py::TestNoLeakage::test_no_leakage_between_runs
PASSED tests/ml/temporal/test_window_builder.py::TestIncompleteWindow::test_incomplete_window_discarded
PASSED tests/ml/temporal/test_window_builder.py::TestHorizonLabel::test_failure_within_horizon_label
PASSED tests/ml/temporal/test_window_builder.py::TestHorizonLabel::test_no_failure_within_horizon_label
PASSED tests/ml/temporal/test_window_builder.py::TestStaticFeatures::test_static_features_joined_correctly
6 passed
```

- [ ] **Step 4: Run the full suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```

Expected: 14 passed (8 from SP1 + 6 from SP2). No failures.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/temporal/dataset.py
git commit -m "feat(sp2): implement build_temporal_windows() with sliding window + static join"
```

---

## Task 4: Smoke-test the CLI diagnostic

**Files:**
- No new files (CLI block is part of `dataset.py` written in Task 3)

- [ ] **Step 1: Verify the module runs without error on an empty or missing DB**

```bash
python -m swmm_resilience.ml.temporal.dataset --help
```

Expected output includes: `--db`, `--summary` options listed.

- [ ] **Step 2: Verify `--summary` on a fresh DB (no artifacts)**

```bash
python -c "
import sqlite3
from swmm_resilience.database.schema import create_schema
conn = sqlite3.connect('/tmp/sp2_smoke.db')
create_schema(conn)
conn.close()
print('DB created')
"
python -m swmm_resilience.ml.temporal.dataset --db /tmp/sp2_smoke.db --summary
```

Expected output:
```
Parquets registrados: 0
Construyendo ventanas...
Total de muestras: 0
No se generaron muestras.
```

- [ ] **Step 3: Commit if any changes were needed**

If `dataset.py` needed no changes from Task 3, skip this commit (it was already committed).

If a fix was needed:

```bash
git add swmm_resilience/ml/temporal/dataset.py
git commit -m "fix(sp2): patch CLI smoke-test issue"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|---|---|
| `build_temporal_windows()` replaces `NotImplementedError` | Task 3 |
| Signature: `db_path, networks_dir, window_spec, dataset_spec` | Task 3 |
| Reads `temporal_artifacts` via SQL (not `register_temporal_artifact`) | Task 3 |
| `resample_min` added to `TemporalWindowSpec` | Task 2 |
| `TemporalWindowDataset` dataclass with 6 fields | Task 2 |
| `X_seq [N, 4, 6]` — 6 temporal features | Task 3 (`TEMPORAL_COLS`) |
| `X_static [N, 7]` — 7 static features | Task 3 (`STATIC_COLS`) |
| `y_class` = `failure_within_horizon` (0/1) | Task 3 |
| `y_reg` = `peak_flooding_lps` in horizon | Task 3 |
| `groups` = `run_id` for GroupKFold | Task 3 |
| `meta` DataFrame: run_id, node_id, window_start_min | Task 3 |
| No normalisation in this function | Task 3 (confirmed) |
| Sliding window algorithm matching spec | Task 3 |
| Forward-fill resampling | Task 3 |
| Discard incomplete windows and horizons | Task 3 (`while i + window_steps + horizon_steps <= n`) |
| Static features joined via (network_hash, node_uid=node_id) | Task 3 |
| CLI `--summary` showing Parquet count, sample count, class balance | Task 3 + 4 |
| 6 tests in `tests/ml/temporal/test_window_builder.py` | Task 1 |
| test: correct shape | Task 1 (`TestWindowShape`) |
| test: no leakage between runs | Task 1 (`TestNoLeakage`) |
| test: incomplete window discarded | Task 1 (`TestIncompleteWindow`) |
| test: failure_within_horizon=1 | Task 1 (`TestHorizonLabel`) |
| test: failure_within_horizon=0 | Task 1 (`TestHorizonLabel`) |
| test: static features joined correctly | Task 1 (`TestStaticFeatures`) |

All spec requirements are covered. No gaps.

### Type Consistency

- `TemporalWindowSpec.resample_min` defined in Task 2, used in Task 3 as `window_spec.resample_min` ✓
- `TemporalWindowDataset` defined in Task 2, returned in Task 3 ✓
- `TEMPORAL_COLS` (6 items) and `STATIC_COLS` (7 items) defined as constants in Task 3, used consistently ✓
- Test fixture `_SPEC = TemporalWindowSpec(window_min=20, horizon_min=5, step_min=5, resample_min=5)` — `resample_min` must exist in `TemporalWindowSpec` (added in Task 2, tests written in Task 1 so will fail until Task 2 is done) ✓
