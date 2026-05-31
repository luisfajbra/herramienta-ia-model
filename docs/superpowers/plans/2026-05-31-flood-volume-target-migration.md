# Flood Volume Target Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-node total flood volume in cubic meters the primary hydraulic regression target, while keeping peak flooding flow in L/s as a diagnostic/candidate post-processor feature and summing node volumes into the run summary.

**Architecture:** Extend the SQLite schema and simulation extraction so `node_results.total_flood_volume_m3` is populated from the SWMM `.rpt` node flooding summary, with a deterministic fallback that integrates `flooding_lps` time series when `.rpt` volume is unavailable. Export the new target to the ML CSV, update tabular/temporal regression targets and manifests, and keep `peak_flooding_lps` exported as a post-SWMM diagnostic/candidate feature, but excluded from default no-SWMM prediction models to avoid leakage. Tests lead each boundary: `.rpt` parsing, DB schema/migration, simulation aggregation, dataset export, feature selection, and model target metadata.

**Tech Stack:** Python, pytest, pandas, SQLite, PySWMM/swmm-api, scikit-learn, PyTorch.

---

## File Structure

- Modify `swmm_resilience/config.py`: set `ML_TARGET_REGRESSION = "total_flood_volume_m3"`, add the volume target to dropped model-input columns, and keep `peak_flooding_lps` dropped for default no-SWMM prediction models.
- Modify `swmm_resilience/database/schema.py`: add `node_results.total_flood_volume_m3` and `run_summary.total_flood_volume_m3`; migrate legacy DBs without renaming old peak-flow columns.
- Modify `swmm_resilience/simulation/swmm_api_io.py`: keep parsing node flood volume from `.rpt` and clarify it is reported in m3.
- Modify `swmm_resilience/simulation/runner.py`: populate per-node volume from `.rpt`, fall back to integrated `flooding_lps`, and sum into `run_summary.total_flood_volume_m3`.
- Modify `swmm_resilience/database/repository.py`: include total flood volume in printed run summaries.
- Modify `swmm_resilience/analysis/dataset.py`: export `total_flood_volume_m3` and keep `peak_flooding_lps` as an exported post-SWMM diagnostic column.
- Modify `swmm_resilience/ml/temporal/dataset.py`: use `total_flood_volume_m3` as `y_reg` for surrogate/unified datasets while leaving time-series window regression on `peak_flooding_lps` unless explicitly migrated in a later temporal-window pass.
- Modify `swmm_resilience/ml/temporal/train_surrogate.py`: manifest regression target becomes `total_flood_volume_m3`; log transform remains `log1p`.
- Modify `swmm_resilience/ml/temporal/predict.py`: rename surrogate regression outputs to volume m3 columns for surrogate map/prediction paths.
- Modify `swmm_resilience/ml/predict_tabular.py` and `swmm_resilience/ml/predict_from_inp.py`: prediction output should expose `predicted_total_flood_volume_m3`; remove the old `predicted_peak_flooding_lps` output from volume-regression paths.
- Modify `swmm_resilience/desktop/app.py`: display and map the predicted volume target as m3.
- Add or update tests under `tests/simulation/`, `tests/database/`, `tests/ml/`, and `tests/ml/temporal/`.
- Do not delete data files. If stale artifacts need deletion, document them in `docs/model_hydraulic_prediction_audit_2026-05-31.md` instead.

---

## Task 1: Schema Columns And Legacy Migration

**Files:**
- Modify: `swmm_resilience/database/schema.py`
- Create: `tests/database/test_flood_volume_schema.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/database/test_flood_volume_schema.py`:

```python
import sqlite3

from swmm_resilience.database.schema import create_schema


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_schema_has_total_flood_volume_columns():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)

    assert "total_flood_volume_m3" in _columns(conn, "node_results")
    assert "total_flood_volume_m3" in _columns(conn, "run_summary")


def test_legacy_peak_column_is_not_renamed_to_volume_target():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE node_results (
            result_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            delta_inflow_lps REAL NOT NULL,
            inflow_multiplier REAL NOT NULL DEFAULT 1,
            node_id TEXT NOT NULL,
            flooded INTEGER NOT NULL DEFAULT 0,
            peak_flooding_lps REAL
        );
        """
    )

    create_schema(conn)
    columns = _columns(conn, "node_results")

    assert "peak_flooding_lps" in columns
    assert "total_flood_volume_m3" in columns
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/database/test_flood_volume_schema.py -v
```

Expected: first test fails because the new columns do not exist.

- [ ] **Step 3: Add columns to `SCHEMA_SQL`**

In `swmm_resilience/database/schema.py`, update `node_results`:

```python
    peak_flooding_lps       REAL,
    total_flood_volume_m3   REAL,
    flooding_duration_min   REAL,
```

Update `run_summary`:

```python
    total_peak_flooding_lps     REAL,
    total_flood_volume_m3       REAL,
    pct_flooded_nodes           REAL,
```

- [ ] **Step 4: Add migration entries to `REQUIRED_COLUMNS`**

Add:

```python
    "node_results": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
        "max_total_outflow_lps": "REAL",
        "time_to_peak_outflow_min": "REAL",
        "downstream_link_peak_flows_lps_json": "TEXT",
        "total_flood_volume_m3": "REAL",
    },
    "run_summary": {
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
        "failed_nodes_count": "INTEGER",
        "total_flood_volume_m3": "REAL",
    },
```

Keep the existing keys and values; only add the new `total_flood_volume_m3` entries.

- [ ] **Step 5: Replace the misleading legacy migration**

Replace `_migrate_node_results_peak_flooding()` with:

```python
def _migrate_legacy_node_flooding_volume(conn):
    """Preserve legacy volume columns without confusing them with peak flow."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(node_results)").fetchall()}
    if "flooding_volume_m3" in cols and "total_flood_volume_m3" not in cols:
        conn.execute("ALTER TABLE node_results RENAME COLUMN flooding_volume_m3 TO total_flood_volume_m3")
        conn.commit()
```

Then update `create_schema()` to call:

```python
    _migrate_legacy_node_flooding_volume(conn)
```

Do not rename `flooding_volume_m3` to `peak_flooding_lps`.

- [ ] **Step 6: Run schema tests**

Run:

```bash
pytest tests/database/test_flood_volume_schema.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add swmm_resilience/database/schema.py tests/database/test_flood_volume_schema.py
git commit -m "feat: add flood volume target schema"
```

---

## Task 2: Simulation Extraction And Run Summary Aggregation

**Files:**
- Modify: `swmm_resilience/simulation/runner.py`
- Modify: `swmm_resilience/simulation/swmm_api_io.py`
- Create: `tests/simulation/test_flood_volume_extraction.py`

- [ ] **Step 1: Write unit tests for volume merge and fallback math**

Create `tests/simulation/test_flood_volume_extraction.py`:

```python
import pandas as pd

from swmm_resilience.simulation.runner import (
    _flood_volume_from_timeseries_m3,
    _merge_rpt_flooding_metrics,
)


def test_flood_volume_from_timeseries_integrates_lps_to_m3():
    rows = [
        {"node_id": "J1", "time_sec": 60.0, "flooding_lps": 10.0},
        {"node_id": "J1", "time_sec": 120.0, "flooding_lps": 20.0},
        {"node_id": "J1", "time_sec": 180.0, "flooding_lps": 0.0},
        {"node_id": "J2", "time_sec": 60.0, "flooding_lps": 0.0},
        {"node_id": "J2", "time_sec": 120.0, "flooding_lps": 5.0},
    ]

    volumes = _flood_volume_from_timeseries_m3(rows)

    assert volumes["J1"] == 1.8
    assert volumes["J2"] == 0.3


def test_merge_rpt_flooding_metrics_prefers_rpt_volume_and_duration():
    node_records = [
        {
            "node_id": "J1",
            "peak_flooding_lps": 10.0,
            "total_flood_volume_m3": 1.8,
            "flooding_duration_min": 3.0,
            "flooded": 1,
        }
    ]
    rpt_df = pd.DataFrame(
        [
            {
                "node_id": "J1",
                "flooding_volume_m3": 2.5,
                "flooding_duration_min": 4.0,
            }
        ]
    )

    _merge_rpt_flooding_metrics(node_records, rpt_df)

    assert node_records[0]["total_flood_volume_m3"] == 2.5
    assert node_records[0]["flooding_duration_min"] == 4.0
    assert node_records[0]["flooded"] == 1
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/simulation/test_flood_volume_extraction.py -v
```

Expected: import errors because helper functions do not exist.

- [ ] **Step 3: Add fallback volume helper**

Add near the top of `swmm_resilience/simulation/runner.py`:

```python
def _flood_volume_from_timeseries_m3(node_timeseries_records: list[dict]) -> dict[str, float]:
    """Integrate per-node flooding_lps over time into m3."""
    by_node: dict[str, list[dict]] = defaultdict(list)
    for row in node_timeseries_records:
        by_node[str(row["node_id"])].append(row)

    volumes: dict[str, float] = {}
    for node_id, rows in by_node.items():
        rows = sorted(rows, key=lambda row: float(row.get("time_sec") or 0.0))
        prev_time = 0.0
        total_litres = 0.0
        for row in rows:
            current_time = float(row.get("time_sec") or prev_time)
            dt = max(0.0, current_time - prev_time)
            flooding_lps = float(row.get("flooding_lps") or 0.0)
            total_litres += flooding_lps * dt
            prev_time = current_time
        volumes[node_id] = round(total_litres / 1000.0, 6)
    return volumes
```

- [ ] **Step 4: Add RPT merge helper**

Add below `_flood_volume_from_timeseries_m3()`:

```python
def _merge_rpt_flooding_metrics(node_records: list[dict], rpt_df) -> None:
    """Overlay SWMM .rpt node flooding volume/duration onto node records."""
    if rpt_df is None:
        return
    rpt_lookup = {str(row["node_id"]): row for _, row in rpt_df.iterrows()}
    for record in node_records:
        rpt_row = rpt_lookup.get(str(record["node_id"]))
        if rpt_row is None:
            continue

        volume = rpt_row.get("flooding_volume_m3")
        try:
            fvol = float(volume)
            if fvol == fvol:
                record["total_flood_volume_m3"] = safe_round(fvol, 6)
        except (TypeError, ValueError):
            pass

        duration = rpt_row.get("flooding_duration_min")
        try:
            fdur = float(duration)
            if fdur == fdur:
                record["flooding_duration_min"] = safe_round(fdur, 2)
        except (TypeError, ValueError):
            pass

        record["flooded"] = int(
            (record.get("peak_flooding_lps") or 0) > 0
            or (record.get("total_flood_volume_m3") or 0) > 0
        )
```

- [ ] **Step 5: Populate fallback volumes in `run_simulation()`**

Before building `node_records`, after the simulation loop, add:

```python
        fallback_flood_volume_m3 = _flood_volume_from_timeseries_m3(node_timeseries_records)
```

When appending each node record, add:

```python
                "total_flood_volume_m3": safe_round(
                    fallback_flood_volume_m3.get(node_id, 0.0),
                    6,
                ),
```

Place it directly after `peak_flooding_lps`.

- [ ] **Step 6: Replace inline `.rpt` duration override**

Replace the existing `if USE_SWMM_API_RPT_RESULTS:` loop in `run_simulation()` with:

```python
    if USE_SWMM_API_RPT_RESULTS:
        _rpt_path = Path(simulation_inp).with_suffix(".rpt")
        _merge_rpt_flooding_metrics(node_records, read_node_flooding_summary(_rpt_path))
        total_flooded = sum(r["flooded"] for r in node_records)
```

- [ ] **Step 7: Add run summary volume**

After `total_peak_flooding_lps` is computed, add:

```python
    total_flood_volume_m3 = sum(record["total_flood_volume_m3"] or 0 for record in node_records)
```

In `summary`, add:

```python
        "total_flood_volume_m3": round(total_flood_volume_m3, 6),
```

directly after `total_peak_flooding_lps`.

- [ ] **Step 8: Update `.rpt` parser docstring**

In `swmm_resilience/simulation/swmm_api_io.py`, replace the outdated note with:

```python
    The .rpt reports flood volume as 10^6 litres. This function converts it
    to m3 so downstream code stores `total_flood_volume_m3`.
```

- [ ] **Step 9: Run tests**

Run:

```bash
pytest tests/simulation/test_flood_volume_extraction.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 10: Commit**

Run:

```bash
git add swmm_resilience/simulation/runner.py swmm_resilience/simulation/swmm_api_io.py tests/simulation/test_flood_volume_extraction.py
git commit -m "feat: extract node flood volume in m3"
```

---

## Task 3: Dataset Export And Tabular Target Contract

**Files:**
- Modify: `swmm_resilience/config.py`
- Modify: `swmm_resilience/analysis/dataset.py`
- Modify: `tests/ml/test_preprocessing_feature_contract.py`
- Create: `tests/analysis/test_dataset_flood_volume_export.py`

- [ ] **Step 1: Write export test**

Create `tests/analysis/test_dataset_flood_volume_export.py`:

```python
import sqlite3

import pandas as pd

from swmm_resilience.analysis.dataset import export_ml_dataset
from swmm_resilience.database.schema import create_schema


def test_export_ml_dataset_includes_volume_target_and_peak_feature(tmp_path):
    db_path = tmp_path / "runs.db"
    csv_path = tmp_path / "dataset.csv"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, network_file, network_hash, scenario_type, spatial_pattern,
            delta_inflow_lps, inflow_multiplier, status
        ) VALUES ('run-1', 'network.inp', 'hash-1', 'steady', 'uniform', 0.0, 2.0, 'completed')
        """
    )
    conn.execute(
        """
        INSERT INTO network_nodes (
            network_hash, node_uid, invert_elev_m, full_depth_m, base_inflow_lps,
            node_type, in_degree, out_degree
        ) VALUES ('hash-1', 'J1', 100.0, 2.0, 1.5, 'junction', 1, 1)
        """
    )
    conn.execute(
        """
        INSERT INTO node_results (
            result_id, run_id, delta_inflow_lps, inflow_multiplier, node_id,
            flooded, peak_flooding_lps, total_flood_volume_m3, flooding_duration_min
        ) VALUES ('res-1', 'run-1', 0.0, 2.0, 'J1', 1, 12.0, 3.5, 4.0)
        """
    )
    conn.commit()
    conn.close()

    df = export_ml_dataset(str(db_path), str(csv_path))

    assert "total_flood_volume_m3" in df.columns
    assert "peak_flooding_lps" in df.columns
    assert df.loc[0, "total_flood_volume_m3"] == 3.5
    assert df.loc[0, "peak_flooding_lps"] == 12.0
    assert pd.read_csv(csv_path).loc[0, "total_flood_volume_m3"] == 3.5
```

- [ ] **Step 2: Run failing export test**

Run:

```bash
pytest tests/analysis/test_dataset_flood_volume_export.py -v
```

Expected: fails because `total_flood_volume_m3` is not exported yet.

- [ ] **Step 3: Export the new column**

In `swmm_resilience/analysis/dataset.py`, add:

```sql
            nr.total_flood_volume_m3,
```

between `nr.peak_flooding_lps` and `nr.flooding_duration_min`.

- [ ] **Step 4: Change the regression target**

In `swmm_resilience/config.py`, change:

```python
ML_TARGET_REGRESSION = "peak_flooding_lps"
```

to:

```python
ML_TARGET_REGRESSION = "total_flood_volume_m3"
```

- [ ] **Step 5: Keep volume target out of model inputs and keep peak as post-SWMM diagnostic**

In `swmm_resilience/config.py`, keep `"peak_flooding_lps"` in `ML_DROP_COLUMNS` for the default no-SWMM prediction models. Although it remains exported in `dataset_ml.csv`, it is a post-SWMM value and would leak information if used to predict volume before running SWMM.

Add `"total_flood_volume_m3"` to `ML_DROP_COLUMNS` so the new regression target is never used as an input feature:

```python
    "total_flood_volume_m3",
```

Place it near `"flooded"`, `"peak_flooding_lps"`, and `"flooding_duration_min"`.

- [ ] **Step 6: Update preprocessing contract test**

In `tests/ml/test_preprocessing_feature_contract.py`, add `total_flood_volume_m3` to both fixtures and update assertions:

```python
            "total_flood_volume_m3": [3.5],
```

In `test_tabular_feature_selection_keeps_static_hydraulic_features()`, add explicit assertions that the exported post-SWMM columns are not default model inputs:

```python
    assert "peak_flooding_lps" not in features
    assert "total_flood_volume_m3" not in features
```

In `test_tabular_feature_selection_drops_result_and_metadata_columns()`, keep the exact selected feature set as:

```python
    assert set(features) == {
        "inflow_multiplier",
        "in_degree",
        "out_degree",
        "upstream_capacity_lps",
        "downstream_capacity_lps",
    }
```

Add the negative assertion:

```python
    assert "total_flood_volume_m3" not in features
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/analysis/test_dataset_flood_volume_export.py tests/ml/test_preprocessing_feature_contract.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add swmm_resilience/config.py swmm_resilience/analysis/dataset.py tests/analysis/test_dataset_flood_volume_export.py tests/ml/test_preprocessing_feature_contract.py
git commit -m "feat: use flood volume as tabular target"
```

---

## Task 4: Repository Summary And Persistence Output

**Files:**
- Modify: `swmm_resilience/database/repository.py`
- Modify: `swmm_resilience/main.py`
- Create: `tests/database/test_flood_volume_summary_output.py`

- [ ] **Step 1: Write focused summary query test**

Create `tests/database/test_flood_volume_summary_output.py`:

```python
import sqlite3

import pandas as pd

from swmm_resilience.database.schema import create_schema


def test_run_summary_stores_total_flood_volume_m3():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, network_file, network_hash, scenario_type, spatial_pattern,
            delta_inflow_lps, inflow_multiplier, status
        ) VALUES ('run-1', 'network.inp', 'hash-1', 'steady', 'uniform', 0.0, 2.0, 'completed')
        """
    )
    conn.execute(
        """
        INSERT INTO run_summary (
            summary_id, run_id, inflow_multiplier, total_nodes, failed_nodes_count,
            total_peak_flooding_lps, total_flood_volume_m3, pct_flooded_nodes,
            resilience_index
        ) VALUES ('sum-1', 'run-1', 2.0, 3, 1, 12.0, 3.5, 33.33, 0.6667)
        """
    )

    df = pd.read_sql(
        """
        SELECT s.total_flood_volume_m3
        FROM runs r
        LEFT JOIN run_summary s ON r.run_id = s.run_id
        WHERE r.status = 'completed'
        """,
        conn,
    )

    assert df.loc[0, "total_flood_volume_m3"] == 3.5
```

- [ ] **Step 2: Run test**

Run:

```bash
pytest tests/database/test_flood_volume_summary_output.py -v
```

Expected:

```text
1 passed
```

The schema task already makes this pass; this test pins summary semantics.

- [ ] **Step 3: Update `export_run_summary()` query**

In `swmm_resilience/database/repository.py`, include:

```sql
            s.total_flood_volume_m3,
```

after `s.total_peak_flooding_lps`.

- [ ] **Step 4: Update CLI run output wording**

In `swmm_resilience/main.py`, where run summary prints peak flooding, add:

```python
            print(f"    ok Vol flood : {summary['total_flood_volume_m3']:.3f} m3 (suma nodos)")
```

directly after the existing peak-flow line. Keep the peak line, but it must remain labeled as `lps`.

- [ ] **Step 5: Commit**

Run:

```bash
git add swmm_resilience/database/repository.py swmm_resilience/main.py tests/database/test_flood_volume_summary_output.py
git commit -m "feat: report flood volume summaries"
```

---

## Task 5: Temporal Surrogate Dataset And Manifest Target

**Files:**
- Modify: `swmm_resilience/ml/temporal/dataset.py`
- Modify: `swmm_resilience/ml/temporal/train_surrogate.py`
- Modify: `tests/ml/temporal/test_surrogate_dataset.py`
- Modify: `tests/ml/temporal/test_train_surrogate.py`

- [ ] **Step 1: Update surrogate dataset tests**

In `tests/ml/temporal/test_surrogate_dataset.py`, update helper CSV rows so they include:

```python
"total_flood_volume_m3": 2.5 if flooded else 0.0,
```

Update the regression-label test to assert volume:

```python
def test_y_reg_uses_total_flood_volume_m3(self, tmp_path):
    db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
    ds = build_surrogate_dataset(db_path=db_path)

    flooded_mask = ds.meta["node_id"] == "J-000"
    dry_mask = ds.meta["node_id"] == "J-002"

    assert (ds.y_reg[flooded_mask] == 2.5).all()
    assert (ds.y_reg[dry_mask] == 0.0).all()
```

Keep the existing nonnegative regression test.

- [ ] **Step 2: Update training manifest test**

In `tests/ml/temporal/test_train_surrogate.py`, change:

```python
assert manifest["regression_target"] == "peak_flooding_lps"
```

to:

```python
assert manifest["regression_target"] == "total_flood_volume_m3"
assert manifest["regression_units"] == "m3"
```

- [ ] **Step 3: Run failing temporal tests**

Run:

```bash
pytest tests/ml/temporal/test_surrogate_dataset.py tests/ml/temporal/test_train_surrogate.py -v
```

Expected: tests fail until dataset and manifest are updated.

- [ ] **Step 4: Update `build_surrogate_dataset()`**

In `swmm_resilience/ml/temporal/dataset.py`, replace the surrogate regression target source:

```python
y_reg.append(float(row.get("peak_flooding_lps", 0.0) or 0.0))
```

with:

```python
y_reg.append(float(row.get("total_flood_volume_m3", 0.0) or 0.0))
```

Only apply this to surrogate/unified full-run datasets. Do not change temporal-window horizon labels in this task.

- [ ] **Step 5: Update manifest metadata**

In `swmm_resilience/ml/temporal/train_surrogate.py`, update `_write_surrogate_manifest()` so manifest contains:

```python
        "regression_target": "total_flood_volume_m3",
        "regression_units": "m3",
        "regression_target_transform": "log1p",
```

- [ ] **Step 6: Run temporal tests**

Run:

```bash
pytest tests/ml/temporal/test_surrogate_dataset.py tests/ml/temporal/test_train_surrogate.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add swmm_resilience/ml/temporal/dataset.py swmm_resilience/ml/temporal/train_surrogate.py tests/ml/temporal/test_surrogate_dataset.py tests/ml/temporal/test_train_surrogate.py
git commit -m "feat: train surrogate on flood volume target"
```

---

## Task 6: Prediction Output Names And UI Mapping

**Files:**
- Modify: `swmm_resilience/ml/predict_tabular.py`
- Modify: `swmm_resilience/ml/predict_from_inp.py`
- Modify: `swmm_resilience/ml/temporal/predict.py`
- Modify: `swmm_resilience/desktop/app.py`
- Modify: `tests/ml/temporal/test_surrogate_predict.py`
- Create: `tests/ml/test_prediction_volume_output_schema.py`

- [ ] **Step 1: Add tabular output schema test**

Create `tests/ml/test_prediction_volume_output_schema.py`:

```python
def test_prediction_volume_column_name_contract():
    expected = "predicted_total_flood_volume_m3"
    legacy = "predicted_peak_flooding_lps"

    assert expected != legacy
```

This contract is intentionally tiny; the implementation code is covered by existing prediction tests and should be updated to use the expected name.

- [ ] **Step 2: Update tabular prediction output names**

In `swmm_resilience/ml/predict_tabular.py`, rename local variable:

```python
predicted_volume = reg_artifact.pipeline.predict(X_pred_reg)
predicted_volume = pd.Series(predicted_volume).clip(lower=0.0).to_numpy()
```

to:

```python
predicted_total_flood_volume_m3 = reg_artifact.pipeline.predict(X_pred_reg)
predicted_total_flood_volume_m3 = (
    pd.Series(predicted_total_flood_volume_m3).clip(lower=0.0).to_numpy()
)
```

Then output:

```python
"predicted_total_flood_volume_m3": predicted_total_flood_volume_m3,
```

Do not output `predicted_peak_flooding_lps` from the volume regressor.

- [ ] **Step 3: Update INP prediction output names**

Apply the same rename in `swmm_resilience/ml/predict_from_inp.py`.

- [ ] **Step 4: Update surrogate prediction output names**

In `swmm_resilience/ml/temporal/predict.py`, change surrogate output column:

```python
"peak_flooding_lps_pred"
```

to:

```python
"total_flood_volume_m3_pred"
```

Keep classification columns unchanged.

- [ ] **Step 5: Update surrogate map adapter**

In `plot_surrogate_map()` in `swmm_resilience/ml/temporal/predict.py`, map:

```python
"total_flood_volume_m3_pred": "total_flood_volume_m3"
```

and use that value for map intensity. Because `plot_flood_map()` currently
colors the generic magnitude column named `peak_flooding_lps`, pass a renamed
compatibility column inside `plot_surrogate_map()` only, with this comment:

```python
# plot_flood_map currently colors a generic magnitude column named peak_flooding_lps.
# The value supplied here is total flood volume in m3.
```

- [ ] **Step 6: Update desktop display mapping**

In `swmm_resilience/desktop/app.py`, replace display references to:

```python
predicted_peak_flooding_lps
```

with:

```python
predicted_total_flood_volume_m3
```

Use labels that include `m3`.

- [ ] **Step 7: Update tests**

In `tests/ml/temporal/test_surrogate_predict.py`, update expected columns:

```python
expected_cols = {"node_id", "flood_prob", "predicted_flooded", "total_flood_volume_m3_pred"}
```

Run:

```bash
pytest tests/ml/test_prediction_volume_output_schema.py tests/ml/temporal/test_surrogate_predict.py -v
```

Expected: tests pass after implementation.

- [ ] **Step 8: Commit**

Run:

```bash
git add swmm_resilience/ml/predict_tabular.py swmm_resilience/ml/predict_from_inp.py swmm_resilience/ml/temporal/predict.py swmm_resilience/desktop/app.py tests/ml/test_prediction_volume_output_schema.py tests/ml/temporal/test_surrogate_predict.py
git commit -m "feat: expose flood volume prediction outputs"
```

---

## Task 7: Regeneration And Verification

**Files:**
- Modify if needed: `docs/model_hydraulic_prediction_audit_2026-05-31.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/database/test_flood_volume_schema.py tests/simulation/test_flood_volume_extraction.py tests/analysis/test_dataset_flood_volume_export.py tests/database/test_flood_volume_summary_output.py tests/ml/test_preprocessing_feature_contract.py tests/ml/test_prediction_volume_output_schema.py tests/ml/temporal/test_surrogate_dataset.py tests/ml/temporal/test_train_surrogate.py tests/ml/temporal/test_surrogate_predict.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Regenerate DB-derived dataset after code changes**

Run the CLI regeneration for the current Chico Qx sweep:

```bash
python -c "from swmm_resilience.main import run_experiment; run_experiment(inflow_multipliers=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75], reset_db=True)"
```

This command recreates `data/training/swmm_resilience.db`, re-registers temporal
Parquet artifacts, and exports `data/networks/chico_hydro-qx1/results/dataset_ml.csv`
through `run_experiment()`.

After regeneration, verify the DB has non-null volume values:

```bash
sqlite3 data/training/swmm_resilience.db "SELECT COUNT(*), SUM(total_flood_volume_m3), MAX(total_flood_volume_m3) FROM node_results;"
```

Expected:

- count equals the number of node result rows
- sum is greater than `0` when flooding exists
- max is greater than `0` when flooding exists

- [ ] **Step 3: Verify ML CSV export**

The previous step exports `dataset_ml.csv`. Verify it has the new volume target:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/networks/chico_hydro-qx1/results/dataset_ml.csv")
print("total_flood_volume_m3" in df.columns)
print(df["total_flood_volume_m3"].describe())
PY
```

Expected: first line prints `True`.

- [ ] **Step 4: Retrain affected models**

Run:

```bash
python -m swmm_resilience.ml.train
python -m swmm_resilience.ml.temporal.train_surrogate --model cnn --epochs 100 --folds 5
python -m swmm_resilience.ml.temporal.train_surrogate --model lstm --epochs 100 --folds 5
```

On a CUDA machine, append:

```bash
--device cuda
```

to the surrogate commands.

- [ ] **Step 5: Regenerate maps**

Regenerate SWMM/ML maps:

```bash
python -m swmm_resilience.visualization.runner --source both --no-skip
```

Regenerate surrogate CNN maps for the same Qx sweep:

```bash
python -c "from pathlib import Path; from swmm_resilience.config import DEFAULT_INP_FILE; from swmm_resilience.ml.temporal.predict import predict_surrogate_from_multiplier, plot_surrogate_map; multipliers=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75]; [plot_surrogate_map(predict_surrogate_from_multiplier(multiplier=m, model_type='cnn'), Path(DEFAULT_INP_FILE), multiplier=m, model_type='cnn') for m in multipliers]"
```

Regenerate surrogate LSTM maps for the same Qx sweep:

```bash
python -c "from pathlib import Path; from swmm_resilience.config import DEFAULT_INP_FILE; from swmm_resilience.ml.temporal.predict import predict_surrogate_from_multiplier, plot_surrogate_map; multipliers=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75]; [plot_surrogate_map(predict_surrogate_from_multiplier(multiplier=m, model_type='lstm'), Path(DEFAULT_INP_FILE), multiplier=m, model_type='lstm') for m in multipliers]"
```

- [ ] **Step 6: Document stale artifacts instead of deleting**

If old artifacts still use `peak_flooding_lps` as the regression target, append a note to `docs/model_hydraulic_prediction_audit_2026-05-31.md`:

```markdown

## Flood volume target migration artifact note

After migrating the regression target to `total_flood_volume_m3`, old artifacts
trained on `peak_flooding_lps` should not be used for hydraulic volume
predictions. Do not compare old maps against regenerated volume-target maps
without checking each artifact manifest.
```

- [ ] **Step 7: Run broader verification**

Run:

```bash
pytest tests/ml tests/simulation tests/database tests/analysis -v
```

Expected: tests pass except known unrelated desktop/Tkinter aborts outside these paths.

- [ ] **Step 8: Commit docs note if added**

Run only if Step 6 modified the audit:

```bash
git add docs/model_hydraulic_prediction_audit_2026-05-31.md
git commit -m "docs: record flood volume artifact migration note"
```

---

## Final Notes For Implementers

- This migration changes model semantics. Existing trained artifacts that predict `peak_flooding_lps` are not compatible with volume-target predictions.
- Do not delete stale artifacts during this plan. Record stale-artifact risk in the audit instead.
- The new target is per-node `total_flood_volume_m3`. The per-run value is the sum of node volumes.
- `peak_flooding_lps` remains useful as a post-SWMM diagnostic and candidate post-processor feature after a SWMM-backed dataset export. It must stay out of default pre-SWMM prediction features unless a separate model mode explicitly predicts volume after SWMM outputs are already available.
