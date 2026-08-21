# SQLite V17 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the immutable 17-feature contract and the new SQLite schema/query foundation without switching production callers.

**Architecture:** Add a canonical contract module and a migration-driven SQLite boundary alongside the still-running legacy modules. The foundation exposes a flat `training_samples_v17` view while storing normalized network, scenario, run, feature, result, temporal, evaluation, and model data.

**Tech Stack:** Python 3.11, standard-library `sqlite3`, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`

## Global Constraints

- Contract ID is exactly `tabular_v3_17`.
- Feature order is normative and cannot be inferred from database or DataFrame order.
- Migrations are append-only and checksummed.
- Every SQLite connection enables foreign keys and a finite busy timeout.
- Existing `database/schema.py` and `database/repository.py` remain untouched until replacement consumers pass.
- Python 3.11 remains supported; dependency pins must install on it.

---

### Task 1: Repair The Reproducible Python 3.11 Baseline

**Files:**
- Modify: `requirements.txt:1-16`
- Create: `tests/test_dependency_contract.py`

**Interfaces:**
- Produces: an installable Python 3.11 dependency contract
- Consumes: none

- [ ] **Step 1: Write the failing dependency test**

```python
# tests/test_dependency_contract.py
from pathlib import Path


def test_python311_does_not_pin_incompatible_shap():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "shap==0.52.0" not in requirements
    assert "shap==0.51.0" in requirements
```

- [ ] **Step 2: Run it and observe the baseline failure**

```powershell
python -m pytest tests/test_dependency_contract.py -q
```

Expected: FAIL because `requirements.txt` currently contains `shap==0.52.0`.

- [ ] **Step 3: Pin the verified Python 3.11-compatible SHAP release**

Change only this line:

```text
shap==0.51.0
```

Do not change unrelated pins in this task.

- [ ] **Step 4: Verify Python 3.11 can resolve binary distributions**

```powershell
python --version
python -m pip install --dry-run --only-binary=:all: -r requirements.txt
```

Expected: Python reports 3.11.x and dependency resolution succeeds without a
source build.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/test_dependency_contract.py -q
git add requirements.txt tests/test_dependency_contract.py
git commit -m "build: keep SHAP compatible with Python 3.11"
```

Expected: one passing test.

### Task 2: Create The Canonical V17 Feature Contract

**Files:**
- Create: `swmm_resilience/ml/contracts.py`
- Create: `tests/ml/test_feature_contract_v17.py`
- Modify: `swmm_resilience/ml/trainer.py:13-27`
- Modify: `tests/conftest.py:10,47`

**Interfaces:**
- Produces: `FeatureContract`, `TABULAR_V3_17`, `FEATURE_COLUMNS_V17`, `validate_feature_frame(frame)`
- Consumes: pandas DataFrames

- [ ] **Step 1: Write contract tests before implementation**

```python
# tests/ml/test_feature_contract_v17.py
import pandas as pd
import pytest

from swmm_resilience.ml.contracts import (
    FEATURE_DEFINITIONS_V17,
    FEATURE_COLUMNS_V17,
    TARGET_DEFINITIONS_V17,
    TABULAR_V3_17,
    FeatureContractError,
)

EXPECTED = (
    "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps", "q_pico_nodo",
    "q_pico_acum_escalado", "duracion_horas", "tiempo_al_pico_h",
)


def valid_frame():
    return pd.DataFrame([{name: float(i + 1) for i, name in enumerate(EXPECTED)}])


def test_contract_has_exact_id_order_and_count():
    assert TABULAR_V3_17.contract_id == "tabular_v3_17"
    assert FEATURE_COLUMNS_V17 == EXPECTED
    assert TABULAR_V3_17.feature_count == 17
    assert TABULAR_V3_17.descriptor_sha256 == (
        "56af955cadb90dda63b79f48dcec18bccaabc8eb33bcce07f5bf1874dcfbca8a"
    )
    assert all(item.units and item.description for item in FEATURE_DEFINITIONS_V17)
    assert {item.name for item in FEATURE_DEFINITIONS_V17 if item.nullable} == {
        "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
        "dist_outfall_m", "upstream_capacity_lps",
    }
    assert [(item.name, item.task) for item in TARGET_DEFINITIONS_V17] == [
        ("inunda", "classification"),
        ("vol_inundacion_m3", "regression"),
    ]


@pytest.mark.parametrize("columns", [
    EXPECTED[:15],
    EXPECTED[:-1],
    EXPECTED + ("extra",),
    EXPECTED[::-1],
    EXPECTED[:-1] + ("renamed_feature",),
    EXPECTED[:-1] + (EXPECTED[-2],),
    EXPECTED + ("inunda",),
])
def test_contract_rejects_wrong_feature_sets(columns):
    frame = pd.DataFrame([[1.0] * len(columns)], columns=columns)
    with pytest.raises(FeatureContractError):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_non_finite_and_null_values():
    frame = valid_frame()
    frame.loc[0, "duracion_horas"] = float("nan")
    with pytest.raises(FeatureContractError, match="duracion_horas"):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_empty_and_numeric_strings():
    with pytest.raises(FeatureContractError, match="empty"):
        TABULAR_V3_17.validate_frame(valid_frame().iloc[0:0])
    frame = valid_frame()
    frame["prof_max"] = frame["prof_max"].astype(str)
    with pytest.raises(FeatureContractError, match="Non-numeric"):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_allows_only_physical_nullable_features():
    frame = valid_frame()
    frame.loc[0, "diam_max_in"] = float("nan")
    TABULAR_V3_17.validate_frame(frame)


def test_contract_allows_all_null_nullable_column_for_storage_validation():
    frame = valid_frame()
    frame["dist_outfall_m"] = None
    TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_text_even_in_nullable_feature():
    frame = valid_frame()
    frame.loc[0, "diam_max_in"] = "unknown"
    with pytest.raises(FeatureContractError, match="Non-numeric"):
        TABULAR_V3_17.validate_frame(frame)
```

- [ ] **Step 2: Run the test and verify import failure**

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py -q
```

Expected: collection fails because `ml.contracts` does not exist.

- [ ] **Step 3: Implement the immutable contract**

```python
# swmm_resilience/ml/contracts.py
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class FeatureContractError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    units: str
    description: str
    nullable: bool = False


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    task: str
    units: str
    description: str


FEATURE_DEFINITIONS_V17 = (
    FeatureDefinition("elev_fondo", "m", "Node invert elevation"),
    FeatureDefinition("prof_max", "m", "Node maximum available depth"),
    FeatureDefinition("n_tuberias_in", "count", "Incoming conduit count"),
    FeatureDefinition("n_tuberias_out", "count", "Outgoing conduit count"),
    FeatureDefinition("diam_max_in", "m", "Maximum incoming conduit diameter", True),
    FeatureDefinition("diam_max_out", "m", "Maximum outgoing conduit diameter", True),
    FeatureDefinition("pendiente_max_in", "m/m", "Maximum incoming conduit slope", True),
    FeatureDefinition("pendiente_out", "m/m", "Representative outgoing conduit slope", True),
    FeatureDefinition("base_inflow_lps", "L/s", "Persisted baseline node inflow"),
    FeatureDefinition("dist_outfall_m", "m", "Directed hydraulic distance to outfall", True),
    FeatureDefinition("n_nodos_aguas_arriba", "count", "Upstream node count"),
    FeatureDefinition("q_pico_acum_base", "L/s", "Accumulated baseline peak flow"),
    FeatureDefinition("upstream_capacity_lps", "L/s", "Aggregate upstream conveyance capacity", True),
    FeatureDefinition("q_pico_nodo", "L/s", "Scenario node peak inflow"),
    FeatureDefinition("q_pico_acum_escalado", "L/s", "Scenario-scaled accumulated peak flow"),
    FeatureDefinition("duracion_horas", "h", "Persisted scenario duration"),
    FeatureDefinition("tiempo_al_pico_h", "h", "Persisted scenario time to peak"),
)

TARGET_DEFINITIONS_V17 = (
    TargetDefinition(
        "inunda", "classification", "binary",
        "Whether the node floods during the run",
    ),
    TargetDefinition(
        "vol_inundacion_m3", "regression", "m3",
        "Total node flood volume during the run",
    ),
)

FEATURE_COLUMNS_V17 = tuple(item.name for item in FEATURE_DEFINITIONS_V17)
NULLABLE_FEATURE_COLUMNS_V17 = frozenset(
    item.name for item in FEATURE_DEFINITIONS_V17 if item.nullable
)


@dataclass(frozen=True)
class FeatureContract:
    contract_id: str
    definitions: tuple[FeatureDefinition, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def descriptor_sha256(self) -> str:
        payload = json.dumps(
            {
                "contract_id": self.contract_id,
                "features": [asdict(item) for item in self.definitions],
                "targets": [asdict(item) for item in TARGET_DEFINITIONS_V17],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        actual = tuple(frame.columns)
        if actual != self.feature_names:
            raise FeatureContractError(
                f"Expected ordered features {self.feature_names}; received {actual}"
            )
        if frame.empty:
            raise FeatureContractError("Feature frame is empty")
        bad_types = []
        for name in self.feature_names:
            series = frame[name]
            all_null_nullable = (
                name in NULLABLE_FEATURE_COLUMNS_V17 and series.isna().all()
            )
            if not all_null_nullable and (
                not is_numeric_dtype(series) or is_bool_dtype(series)
            ):
                bad_types.append(name)
        if bad_types:
            raise FeatureContractError(f"Non-numeric feature columns: {bad_types}")
        numeric = frame.astype(float)
        required = [name for name in self.feature_names if name not in NULLABLE_FEATURE_COLUMNS_V17]
        missing_required = [name for name in required if numeric[name].isna().any()]
        if missing_required:
            raise FeatureContractError(f"Null required feature values: {missing_required}")
        infinite = [name for name in self.feature_names if np.isinf(numeric[name].dropna()).any()]
        if infinite:
            raise FeatureContractError(f"Infinite feature values: {infinite}")
        return numeric


TABULAR_V3_17 = FeatureContract("tabular_v3_17", FEATURE_DEFINITIONS_V17)


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return TABULAR_V3_17.validate_frame(frame)
```

- [ ] **Step 4: Replace the duplicate trainer constant with an alias**

In `trainer.py` import the contract and retain a temporary compatibility alias:

```python
from .contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17

FEATURE_COLS = list(FEATURE_COLUMNS_V17)  # removed in Plan D after consumers migrate
```

Change `train_models()` to construct `X` in canonical order and validate it:

```python
X = TABULAR_V3_17.validate_frame(df.loc[:, FEATURE_COLUMNS_V17])
```

Replace the test fixture import in `tests/conftest.py` with the canonical
contract while preserving the fixture's list behavior:

```python
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

FEATURE_COLS = list(FEATURE_COLUMNS_V17)
```

- [ ] **Step 5: Run focused current and new tests**

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py tests/test_ml_trainer_predict.py tests/test_scenario_predict.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add swmm_resilience/ml/contracts.py swmm_resilience/ml/trainer.py tests/ml/test_feature_contract_v17.py tests/conftest.py
git commit -m "feat: define canonical 17-feature contract"
```

### Task 3: Add Migration And Connection Infrastructure

**Files:**
- Create: `swmm_resilience/database/connection.py`
- Create: `swmm_resilience/database/__init__.py`
- Create: `swmm_resilience/database/migrations.py`
- Create: `swmm_resilience/database/maintenance.py`
- Create: `swmm_resilience/database/sql/001_v17_initial.sql`
- Create: `tests/database/test_connection_v17.py`
- Create: `tests/database/test_migrations_v17.py`
- Create: `tests/database/test_maintenance_v17.py`

**Interfaces:**
- Produces: `connect_database(path) -> sqlite3.Connection`, `apply_migrations(conn, migration_dir=None) -> None`, `checkpoint_and_backup(conn, destination)`, `optimize_database(conn)`, `MigrationChecksumError`, `MigrationOrderError`
- Consumes: filesystem path

- [ ] **Step 1: Write connection tests**

```python
# tests/database/test_connection_v17.py
from swmm_resilience.database.connection import connect_database


def test_connection_enables_safety_pragmas(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()
```

- [ ] **Step 2: Write migration tests**

```python
# tests/database/test_migrations_v17.py
import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import (
    MigrationChecksumError,
    MigrationOrderError,
    apply_migrations,
)


EXPECTED_TABLES = {
    "schema_migrations", "networks", "nodes", "links", "scenarios",
    "scenario_inflows", "runs", "node_features", "node_results",
    "node_timeseries", "training_runs", "model_evaluations",
    "oof_predictions", "model_metrics", "trained_models",
}


def test_initial_migration_creates_expected_schema(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= tables
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1


def test_migrations_are_idempotent(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    apply_migrations(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1


def test_modified_applied_migration_is_rejected(tmp_path, monkeypatch):
    conn = connect_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    conn.execute("UPDATE schema_migrations SET checksum_sha256='bad'")
    conn.commit()
    with pytest.raises(MigrationChecksumError):
        apply_migrations(conn)


def test_broken_migration_rolls_back_all_its_objects(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "001_broken.sql").write_text(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, checksum_sha256 TEXT NOT NULL, applied_at_utc TEXT NOT NULL);\n"
        "CREATE TABLE partial_table(id INTEGER PRIMARY KEY);\n"
        "THIS IS INVALID SQL;\n",
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "broken.sqlite3")
    with pytest.raises(Exception):
        apply_migrations(conn, migration_dir=migration_dir)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "partial_table" not in names


def test_migration_versions_must_be_contiguous(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "002_gap.sql").write_text("SELECT 1;", encoding="utf-8")
    conn = connect_database(tmp_path / "gap.sqlite3")
    with pytest.raises(MigrationOrderError):
        apply_migrations(conn, migration_dir=migration_dir)
```

- [ ] **Step 3: Run tests and confirm missing modules fail**

```powershell
python -m pytest tests/database/test_connection_v17.py tests/database/test_migrations_v17.py -q
```

Expected: collection errors for new modules.

- [ ] **Step 4: Implement the connection boundary**

```python
# swmm_resilience/database/connection.py
from pathlib import Path
import sqlite3


def connect_database(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        conn.close()
        raise RuntimeError(f"SQLite WAL mode unavailable: {mode}")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
```

- [ ] **Step 5: Add the complete initial SQL migration**

Create `001_v17_initial.sql` with the complete schema below:

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    checksum_sha256 TEXT NOT NULL CHECK(length(checksum_sha256)=64),
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE networks (
    network_id INTEGER PRIMARY KEY,
    network_sha256 TEXT NOT NULL UNIQUE CHECK(length(network_sha256)=64),
    name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    inp_bytes BLOB NOT NULL,
    flow_units TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE nodes (
    node_pk INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    coord_x REAL,
    coord_y REAL,
    invert_elevation_m REAL,
    max_depth_m REAL,
    base_inflow_lps REAL,
    UNIQUE(network_id, node_id),
    UNIQUE(node_pk, network_id)
);

CREATE TABLE links (
    link_pk INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    link_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    from_node_pk INTEGER NOT NULL,
    to_node_pk INTEGER NOT NULL,
    length_m REAL,
    diameter_m REAL,
    slope_m_per_m REAL,
    roughness REAL,
    UNIQUE(network_id, link_id),
    FOREIGN KEY(from_node_pk, network_id) REFERENCES nodes(node_pk, network_id),
    FOREIGN KEY(to_node_pk, network_id) REFERENCES nodes(node_pk, network_id)
);

CREATE TABLE scenarios (
    scenario_id INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    scenario_key TEXT NOT NULL,
    scenario_kind TEXT NOT NULL,
    factor_mult REAL CHECK(factor_mult IS NULL OR factor_mult >= 0),
    shape_id TEXT,
    duracion_horas REAL NOT NULL CHECK(duracion_horas >= 0),
    tiempo_al_pico_h REAL NOT NULL CHECK(
        tiempo_al_pico_h >= 0 AND tiempo_al_pico_h <= duracion_horas
    ),
    config_json TEXT NOT NULL,
    config_sha256 TEXT NOT NULL CHECK(length(config_sha256)=64),
    UNIQUE(network_id, scenario_key, config_sha256),
    UNIQUE(scenario_id, network_id)
);

CREATE TABLE scenario_inflows (
    scenario_id INTEGER NOT NULL REFERENCES scenarios(scenario_id) ON DELETE CASCADE,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    step_index INTEGER NOT NULL CHECK(step_index >= 0),
    time_sec REAL NOT NULL CHECK(time_sec >= 0),
    inflow_lps REAL NOT NULL CHECK(inflow_lps >= 0),
    PRIMARY KEY(scenario_id, node_pk, step_index),
    FOREIGN KEY(scenario_id, network_id) REFERENCES scenarios(scenario_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id) REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    scenario_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    started_at_utc TEXT,
    completed_at_utc TEXT,
    swmm_version TEXT,
    config_sha256 TEXT NOT NULL CHECK(length(config_sha256)=64),
    continuity_error_pct REAL,
    failure_stage TEXT,
    failure_type TEXT,
    failure_message TEXT,
    node_count INTEGER,
    timestep_count INTEGER,
    UNIQUE(run_id, network_id),
    FOREIGN KEY(scenario_id, network_id) REFERENCES scenarios(scenario_id, network_id)
);

CREATE TABLE node_features (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    elev_fondo REAL NOT NULL, prof_max REAL NOT NULL CHECK(prof_max >= 0),
    n_tuberias_in INTEGER NOT NULL CHECK(n_tuberias_in >= 0),
    n_tuberias_out INTEGER NOT NULL CHECK(n_tuberias_out >= 0),
    diam_max_in REAL CHECK(diam_max_in IS NULL OR diam_max_in >= 0),
    diam_max_out REAL CHECK(diam_max_out IS NULL OR diam_max_out >= 0),
    pendiente_max_in REAL, pendiente_out REAL,
    base_inflow_lps REAL NOT NULL CHECK(base_inflow_lps >= 0),
    dist_outfall_m REAL CHECK(dist_outfall_m IS NULL OR dist_outfall_m >= 0),
    n_nodos_aguas_arriba INTEGER NOT NULL CHECK(n_nodos_aguas_arriba >= 0),
    q_pico_acum_base REAL NOT NULL CHECK(q_pico_acum_base >= 0),
    upstream_capacity_lps REAL CHECK(upstream_capacity_lps IS NULL OR upstream_capacity_lps >= 0),
    q_pico_nodo REAL NOT NULL CHECK(q_pico_nodo >= 0),
    q_pico_acum_escalado REAL NOT NULL CHECK(q_pico_acum_escalado >= 0),
    duracion_horas REAL NOT NULL CHECK(duracion_horas >= 0),
    tiempo_al_pico_h REAL NOT NULL CHECK(
        tiempo_al_pico_h >= 0 AND tiempo_al_pico_h <= duracion_horas
    ),
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    PRIMARY KEY(run_id, node_pk),
    FOREIGN KEY(run_id, network_id) REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id) REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE node_results (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    inunda INTEGER NOT NULL CHECK(inunda IN (0,1)),
    vol_inundacion_m3 REAL NOT NULL CHECK(vol_inundacion_m3 >= 0),
    peak_flooding_lps REAL CHECK(peak_flooding_lps IS NULL OR peak_flooding_lps >= 0),
    flooding_duration_min REAL CHECK(flooding_duration_min IS NULL OR flooding_duration_min >= 0),
    max_depth_m REAL CHECK(max_depth_m IS NULL OR max_depth_m >= 0),
    max_depth_ratio REAL CHECK(max_depth_ratio IS NULL OR max_depth_ratio >= 0),
    PRIMARY KEY(run_id, node_pk),
    FOREIGN KEY(run_id, network_id) REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id) REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE node_timeseries (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    step_index INTEGER NOT NULL CHECK(step_index >= 0),
    time_sec REAL NOT NULL CHECK(time_sec >= 0),
    total_inflow_lps REAL NOT NULL, lateral_inflow_lps REAL NOT NULL,
    depth_m REAL NOT NULL, depth_ratio REAL NOT NULL,
    flooding_lps REAL NOT NULL, total_outflow_lps REAL NOT NULL,
    failed_now INTEGER NOT NULL CHECK(failed_now IN (0,1)),
    PRIMARY KEY(run_id, node_pk, step_index),
    FOREIGN KEY(run_id, network_id) REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id) REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);
```

Append these exact evaluation and artifact tables to that SQL file:

```sql
CREATE TABLE training_runs (
    training_run_id INTEGER PRIMARY KEY,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    query_sql TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    included_run_ids_json TEXT NOT NULL,
    grouping_strategy TEXT NOT NULL CHECK(grouping_strategy IN ('group_kfold','loso')),
    fold_count INTEGER NOT NULL CHECK(fold_count >= 2),
    random_seed INTEGER NOT NULL,
    primary_metric TEXT NOT NULL,
    tie_breakers_json TEXT NOT NULL,
    python_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    started_at_utc TEXT,
    completed_at_utc TEXT,
    failure_type TEXT,
    failure_message TEXT
);

CREATE TABLE model_evaluations (
    evaluation_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    task TEXT NOT NULL CHECK(task IN ('classification','regression')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    fold_id INTEGER NOT NULL CHECK(fold_id >= 0),
    train_run_ids_json TEXT NOT NULL,
    validation_run_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    fit_seconds REAL NOT NULL CHECK(fit_seconds >= 0),
    predict_seconds REAL NOT NULL CHECK(predict_seconds >= 0),
    failure_type TEXT,
    failure_message TEXT,
    UNIQUE(training_run_id, task, algorithm, fold_id)
);

CREATE TABLE oof_predictions (
    evaluation_id INTEGER NOT NULL REFERENCES model_evaluations(evaluation_id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3')),
    observed REAL NOT NULL,
    predicted REAL NOT NULL,
    probability REAL,
    fold_id INTEGER NOT NULL CHECK(fold_id >= 0),
    PRIMARY KEY(evaluation_id, run_id, node_pk),
    FOREIGN KEY(run_id, node_pk) REFERENCES node_results(run_id, node_pk) ON DELETE CASCADE
);

CREATE TABLE model_metrics (
    metric_id INTEGER PRIMARY KEY,
    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('training_run','evaluation','model')),
    owner_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    reason TEXT,
    UNIQUE(owner_kind, owner_id, scope, metric_name)
);

CREATE TABLE trained_models (
    model_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    ordered_features_json TEXT NOT NULL,
    target_transform_json TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    included_run_ids_json TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    grouping_strategy TEXT NOT NULL,
    python_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    selected_metric TEXT NOT NULL,
    selected_value REAL NOT NULL,
    model_sha256 TEXT NOT NULL CHECK(length(model_sha256)=64),
    model_blob BLOB NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(training_run_id, target)
);

CREATE INDEX idx_runs_status_scenario ON runs(status, scenario_id);
CREATE INDEX idx_scenarios_network_kind ON scenarios(network_id, scenario_kind, factor_mult, shape_id);
CREATE INDEX idx_timeseries_run_time ON node_timeseries(run_id, time_sec);
CREATE INDEX idx_timeseries_run_node_step ON node_timeseries(run_id, node_pk, step_index);
CREATE INDEX idx_oof_evaluation ON oof_predictions(evaluation_id, fold_id);
CREATE INDEX idx_metrics_owner ON model_metrics(owner_kind, owner_id, metric_name);
CREATE INDEX idx_models_target_contract_metric ON trained_models(target, feature_contract_id, selected_metric, selected_value);
```

- [ ] **Step 6: Implement migration discovery and checksum enforcement**

```python
# swmm_resilience/database/migrations.py
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
from pathlib import Path
import re
import sqlite3


class MigrationChecksumError(RuntimeError):
    pass


class MigrationOrderError(RuntimeError):
    pass


MIGRATION_NAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


def _migration_entries(migration_dir: Path | None):
    root = migration_dir or files("swmm_resilience.database").joinpath("sql")
    entries = []
    for entry in root.iterdir():
        match = MIGRATION_NAME.fullmatch(entry.name)
        if match:
            entries.append((int(match.group(1)), match.group(2), entry))
    entries.sort()
    versions = [item[0] for item in entries]
    if versions != list(range(1, len(entries) + 1)):
        raise MigrationOrderError(f"Migrations must be contiguous from 001: {versions}")
    return entries


def apply_migrations(conn: sqlite3.Connection, migration_dir: Path | None = None) -> None:
    applied_any = False
    for version, name, sql_path in _migration_entries(migration_dir):
        sql = sql_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        row = None
        if exists:
            row = conn.execute(
                "SELECT name, checksum_sha256 FROM schema_migrations WHERE version=?",
                (version,),
            ).fetchone()
        if row:
            if row[0] != name or row[1] != checksum:
                raise MigrationChecksumError(
                    f"Applied migration {version:03d} identity/checksum differs"
                )
            continue
        stamp = datetime.now(timezone.utc).isoformat()
        record = (
            "INSERT INTO schema_migrations"
            "(version,name,checksum_sha256,applied_at_utc) VALUES"
            f"({version},'{name}','{checksum}','{stamp}');"
        )
        try:
            conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\n{record}\nCOMMIT;")
            applied_any = True
        except Exception:
            conn.rollback()
            raise
    if applied_any:
        conn.execute("PRAGMA optimize")
```

Migration names are constrained by `MIGRATION_NAME`, while checksums and UTC
timestamps contain no quotes, so the generated record is safe. The explicit
transaction and exception rollback make each migration atomic.

- [ ] **Step 7: Implement and test maintenance operations**

```python
# swmm_resilience/database/maintenance.py
from pathlib import Path
import sqlite3


def optimize_database(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA optimize")


def checkpoint_and_backup(
    conn: sqlite3.Connection, destination: str | Path
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    busy, _log_frames, _checkpointed = conn.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    ).fetchone()
    if busy:
        raise RuntimeError("Cannot back up SQLite database while WAL checkpoint is busy")
    backup_conn = sqlite3.connect(target)
    try:
        conn.backup(backup_conn)
    finally:
        backup_conn.close()
    return target
```

Tests insert committed rows after WAL activation, call
`checkpoint_and_backup()`, open the standalone destination without `-wal` or
`-shm`, and assert schema migration plus row counts match. A held write lock on
a second connection must produce the explicit busy failure. Test
`optimize_database()` executes without changing user rows.

Create the package boundary:

```python
# swmm_resilience/database/__init__.py
from .connection import connect_database
from .maintenance import checkpoint_and_backup, optimize_database
from .migrations import MigrationChecksumError, MigrationOrderError, apply_migrations

__all__ = [
    "MigrationChecksumError",
    "MigrationOrderError",
    "apply_migrations",
    "checkpoint_and_backup",
    "connect_database",
    "optimize_database",
]
```

- [ ] **Step 8: Verify and commit**

```powershell
python -m pytest tests/database/test_connection_v17.py tests/database/test_migrations_v17.py tests/database/test_maintenance_v17.py -q
git add swmm_resilience/database/__init__.py swmm_resilience/database/connection.py swmm_resilience/database/migrations.py swmm_resilience/database/maintenance.py swmm_resilience/database/sql/001_v17_initial.sql tests/database/test_connection_v17.py tests/database/test_migrations_v17.py tests/database/test_maintenance_v17.py
git commit -m "feat: add migration-driven SQLite v17 foundation"
```

### Task 4: Add The Canonical Training View And Query

**Files:**
- Modify: `swmm_resilience/database/sql/001_v17_initial.sql`
- Create: `swmm_resilience/database/training_queries.py`
- Create: `tests/database/test_training_view_v17.py`

**Interfaces:**
- Produces: `load_training_samples(conn, run_ids=None) -> pd.DataFrame`, `export_training_samples_csv(conn, output_path, run_ids=None) -> Path`
- Consumes: complete rows in normalized tables and `TABULAR_V3_17`

- [ ] **Step 1: Write tests for flat CSV-equivalent output**

```python
# tests/database/test_training_view_v17.py
import pandas as pd
import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.training_queries import (
    export_training_samples_csv,
    load_training_samples,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def test_training_view_is_flat_and_canonical(seed_complete_run):
    conn, run_id = seed_complete_run
    frame = load_training_samples(conn)
    expected_prefix = [
        "run_id", "network_id", "scenario_id", "scenario_key", "scenario_kind",
        "factor_mult", "shape_id", "node_id",
    ]
    assert frame.columns.tolist() == expected_prefix + list(FEATURE_COLUMNS_V17) + ["inunda", "vol_inundacion_m3"]
    assert len(frame) == 2


def test_training_view_excludes_failed_and_running_runs(seed_runs_by_status):
    conn = seed_runs_by_status
    frame = load_training_samples(conn)
    assert set(frame["run_id"]) == {seed_runs_by_status.complete_run_id}


def test_loader_rejects_missing_feature_rows(seed_incomplete_complete_run):
    conn = seed_incomplete_complete_run
    with pytest.raises(ValueError, match="feature/result row count"):
        load_training_samples(conn)


def test_csv_exists_only_after_explicit_export(seed_complete_run, tmp_path):
    conn, _run_id = seed_complete_run
    output = tmp_path / "export.csv"
    assert not output.exists()
    assert export_training_samples_csv(conn, output) == output
    assert pd.read_csv(output).columns.tolist() == load_training_samples(conn).columns.tolist()
```

Create fixtures in this test file or `tests/database/conftest.py`; do not couple
them to legacy schema fixtures.

- [ ] **Step 2: Add the explicit view**

Append to migration 001 before it is committed/applied anywhere outside tests:

```sql
CREATE VIEW training_samples_v17 AS
SELECT
    r.run_id,
    r.network_id,
    r.scenario_id,
    s.scenario_key,
    s.scenario_kind,
    s.factor_mult,
    s.shape_id,
    n.node_id,
    f.elev_fondo, f.prof_max, f.n_tuberias_in, f.n_tuberias_out,
    f.diam_max_in, f.diam_max_out, f.pendiente_max_in, f.pendiente_out,
    f.base_inflow_lps, f.dist_outfall_m, f.n_nodos_aguas_arriba,
    f.q_pico_acum_base, f.upstream_capacity_lps, f.q_pico_nodo,
    f.q_pico_acum_escalado, f.duracion_horas, f.tiempo_al_pico_h,
    o.inunda,
    o.vol_inundacion_m3
FROM runs AS r
JOIN scenarios AS s ON s.scenario_id = r.scenario_id
JOIN node_features AS f ON f.run_id = r.run_id AND f.network_id = r.network_id
JOIN node_results AS o ON o.run_id = f.run_id AND o.node_pk = f.node_pk AND o.network_id = f.network_id
JOIN nodes AS n ON n.node_pk = f.node_pk AND n.network_id = f.network_id
WHERE r.status = 'COMPLETE';
```

- [ ] **Step 3: Implement the strict query loader**

```python
# swmm_resilience/database/training_queries.py
from __future__ import annotations
from pathlib import Path
import sqlite3
import pandas as pd

from ..ml.contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17


IDENTITY_COLUMNS = (
    "run_id", "network_id", "scenario_id", "scenario_key", "scenario_kind",
    "factor_mult", "shape_id", "node_id",
)
TARGET_COLUMNS = ("inunda", "vol_inundacion_m3")


def load_training_samples(conn: sqlite3.Connection, run_ids: list[int] | None = None) -> pd.DataFrame:
    columns = IDENTITY_COLUMNS + FEATURE_COLUMNS_V17 + TARGET_COLUMNS
    sql = f"SELECT {', '.join(columns)} FROM training_samples_v17"
    params: list[int] = []
    if run_ids is not None:
        if not run_ids:
            raise ValueError("run_ids cannot be empty")
        sql += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
        params = list(run_ids)
    sql += " ORDER BY run_id, node_id"
    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        raise ValueError("No COMPLETE v17 training samples found")
    TABULAR_V3_17.validate_frame(frame.loc[:, FEATURE_COLUMNS_V17])
    return frame


def export_training_samples_csv(
    conn: sqlite3.Connection,
    output_path: str | Path,
    run_ids: list[int] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    load_training_samples(conn, run_ids=run_ids).to_csv(output, index=False)
    return output
```

Before returning, compare feature and result counts for selected complete runs
and raise on any mismatch. Do not silently drop incomplete nodes through the
inner join.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/database/test_training_view_v17.py tests/ml/test_feature_contract_v17.py -q
git add swmm_resilience/database/sql/001_v17_initial.sql swmm_resilience/database/training_queries.py tests/database/test_training_view_v17.py
git commit -m "feat: expose canonical v17 training view"
```

### Task 5: Add Foundation Scale And Query-Plan Gates

**Files:**
- Create: `tests/database/test_scale_v17.py`
- Create: `tests/database/test_query_plans_v17.py`

**Interfaces:**
- Consumes: migration schema and training query
- Produces: measured baseline evidence; no production API

- [ ] **Step 1: Add an opt-in million-row temporal load test**

Mark it `@pytest.mark.scale` and generate rows in batches without retaining one
million dictionaries in memory. Assert:

```python
assert inserted_rows == 1_000_000
assert elapsed_seconds < 120
assert db_path.stat().st_size < 500 * 1024 * 1024
```

The thresholds are guardrails for the development machine, not promises of
production latency. Record actual values in test output.

- [ ] **Step 2: Add critical query-plan tests**

Use `EXPLAIN QUERY PLAN` and assert the plans reference indexes for:

```sql
SELECT * FROM node_timeseries WHERE run_id=? ORDER BY time_sec;
SELECT * FROM node_timeseries WHERE run_id=? AND node_pk=? ORDER BY step_index;
SELECT * FROM training_samples_v17 WHERE run_id IN (?,?);
SELECT * FROM scenarios WHERE network_id=? AND scenario_kind=?;
SELECT * FROM model_metrics WHERE owner_kind=? AND owner_id=? AND metric_name=?;
SELECT model_id, selected_value FROM trained_models WHERE target=? AND feature_contract_id=? AND selected_metric=?;
```

Avoid asserting SQLite's full human-readable plan string. Assert the named
temporal, scenario, metric, and model-selection indexes appear in their query
details. For the view query, assert the planner uses keyed `SEARCH` operations
for `runs`, `node_features`, and `node_results` and does not perform a full
scan of either high-volume child table.

- [ ] **Step 3: Benchmark the temporal primary-key layout**

In `test_scale_v17.py`, insert the same generated rows into two isolated
databases: one using the migration's ordinary rowid `node_timeseries` table and
one using an otherwise identical `WITHOUT ROWID` table. Run five warmed reads
for `(run_id, node_pk, step_index)` windows and report median milliseconds plus
database bytes for both layouts. This plan keeps the ordinary rowid layout as
the fixed portable schema; the benchmark is recorded for the later temporal
design and must not mutate migrations based on the machine running tests.

- [ ] **Step 4: Run fast plans always and scale explicitly**

```powershell
python -m pytest tests/database/test_query_plans_v17.py -q
python -m pytest tests/database/test_scale_v17.py -m scale -q -s
```

Expected: both pass and print measured rows/seconds/bytes.

- [ ] **Step 5: Commit**

```powershell
git add tests/database/test_scale_v17.py tests/database/test_query_plans_v17.py pytest.ini
git commit -m "test: establish SQLite v17 scale gates"
```

If `scale` is added to `pytest.ini`, register it explicitly:

```ini
markers =
    scale: high-volume SQLite performance checks
```

### Task 6: Run The Foundation Phase Gate

**Files:**
- Verify only

**Interfaces:**
- Produces: approved foundation for Plan B
- Consumes: Tasks 1-5

- [ ] **Step 1: Run all foundation tests from a fresh command**

```powershell
python -m pytest tests/test_dependency_contract.py tests/ml/test_feature_contract_v17.py tests/database/test_connection_v17.py tests/database/test_migrations_v17.py tests/database/test_maintenance_v17.py tests/database/test_training_view_v17.py tests/database/test_query_plans_v17.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run compatibility tests for untouched production paths**

```powershell
python -m pytest tests/test_config.py tests/test_ml_trainer_predict.py tests/test_scenario_predict.py tests/database -q
```

Expected: zero failures, including legacy tests that will be retired only in Plan D.

- [ ] **Step 3: Inspect branch cleanliness**

```powershell
git diff --check
git status --short --branch
```

Expected: no uncommitted implementation files.
