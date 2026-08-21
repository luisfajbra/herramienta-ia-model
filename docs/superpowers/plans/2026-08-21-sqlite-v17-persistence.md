# SQLite V17 Simulation Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist networks, scenarios, aggregate results, exact v17 features, and every timestep directly into SQLite with recoverable run lifecycle semantics.

**Architecture:** Introduce repositories and a typed `SimulationStore` over the Plan A database primitives. A new `swmm_resilience/pipeline.py` composes existing extraction and SWMM functions, writes bounded batches, validates the row contract, and marks a run complete only after every count and feature check succeeds.

**Tech Stack:** Python 3.11, sqlite3, pandas, PySWMM, NetworkX, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`

## Global Constraints

- Requires Plan A commits and passing phase gate.
- SQLite is the only operational persistence target.
- SQL `NULL` is permitted only for the six contract-approved physically undefined features.
- `COMPLETE` is the only trainable run state.
- Timesteps are inserted in bounded batches; no full-run million-row DataFrame is required.
- Existing `swmm_resilience/main.py` remains until the new root path is verified.

---

### Task 1: Add Database Configuration Without Removing Legacy Keys

**Files:**
- Modify: `config.yaml:1-38`
- Modify: `swmm_resilience/config.py:182-285`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `DatabaseConfig(path: Path, batch_size: int, busy_timeout_ms: int)` and `Config.database`
- Consumes: YAML `database` mapping

- [ ] **Step 1: Extend the config test fixture and assertions**

Add to the YAML written by `tests/test_config.py`:

```yaml
database:
  path: "data/swmm_resilience.sqlite3"
  batch_size: 5000
  busy_timeout_ms: 5000
simulation:
  max_continuity_error_pct: 5.0
```

Assert:

```python
assert cfg.database.path == tmp_path / "data" / "swmm_resilience.sqlite3"
assert cfg.database.batch_size == 5000
assert cfg.database.busy_timeout_ms == 5000
assert cfg.simulation.max_continuity_error_pct == 5.0
```

Add rejection tests for `batch_size <= 0`, `busy_timeout_ms < 0`, and
`max_continuity_error_pct < 0`.

- [ ] **Step 2: Run the focused tests and observe failure**

```powershell
python -m pytest tests/test_config.py -q
```

Expected: failures because `Config.database` does not exist.

- [ ] **Step 3: Implement structured database config**

```python
@dataclass(frozen=True)
class DatabaseConfig:
    path: Path
    batch_size: int = 5000
    busy_timeout_ms: int = 5000
```

Parse it in `load_config()` and add it to `Config`. Update
`connect_database(path, busy_timeout_ms=...)` so the configured value controls
both the connection timeout and `PRAGMA busy_timeout`. Add
`max_continuity_error_pct: float = 5.0` to `SimulationConfig`.

- [ ] **Step 4: Update root YAML and verify**

Add the exact mapping shown in Step 1 to `config.yaml`.

```powershell
python -m pytest tests/test_config.py tests/database/test_connection_v17.py -q
git add config.yaml swmm_resilience/config.py swmm_resilience/database/connection.py tests/test_config.py tests/database/test_connection_v17.py
git commit -m "config: add SQLite v17 persistence settings"
```

### Task 2: Implement Network And Scenario Repositories

**Files:**
- Create: `swmm_resilience/database/repositories.py`
- Create: `tests/database/test_repositories_v17.py`

**Interfaces:**
- Produces:
  - `upsert_network(conn, inp_path, name, flow_units) -> int`
  - `load_verified_network_source(conn, network_id) -> bytes`
  - `replace_network_topology(conn, network_id, nodes, links) -> dict[str, int]`
  - `upsert_scenario(conn, network_id, scenario) -> int`
  - `replace_scenario_inflows(conn, scenario_id, rows, node_keys) -> int`
- Consumes: `.inp` bytes, topology rows, and canonical scenario inputs

- [ ] **Step 1: Write repository tests**

Tests must prove:

```python
assert upsert_network(conn, inp, "Chico", "LPS") == upsert_network(conn, inp, "Chico", "LPS")
assert conn.execute("SELECT inp_bytes FROM networks").fetchone()[0] == inp.read_bytes()
assert node_keys == {"J1": 1, "J2": 2}
assert inserted_inflows == len(inflow_rows)
```

Also mutate the `.inp` bytes and assert a new network ID/hash is produced.
Corrupt `inp_bytes` directly and assert `load_verified_network_source()`
raises a hash-mismatch error before returning bytes.
Insert duplicate `(scenario,node,step)` rows and assert the transaction rolls
back rather than overwriting them. Create a second network and assert foreign
keys reject its node in the first network's links, scenario inflows, features,
results, and timesteps.

- [ ] **Step 2: Implement canonical JSON and SHA helpers**

In `repositories.py`:

```python
def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
```

- [ ] **Step 3: Implement network upsert and topology replacement**

Use `INSERT ... ON CONFLICT(network_sha256) DO UPDATE SET name=excluded.name`
and then select the ID. Topology replacement must run inside the caller's
transaction and return a mapping from external `node_id` to integer `node_pk`.
Never match nodes globally without `network_id`.

Node records persist exactly `node_id`, `node_type`, coordinates, invert
elevation, maximum depth, and baseline inflow. Link records persist exactly
`link_id`, `link_type`, endpoint IDs, length, diameter, slope, and roughness;
SQL `NULL` represents attributes that do not apply to that SWMM object type.
`load_verified_network_source()` recomputes SHA-256 from `inp_bytes` and
compares it with `network_sha256` on every load.

- [ ] **Step 4: Implement scenario upsert and inflow batches**

Define a frozen input DTO:

```python
@dataclass(frozen=True)
class ScenarioRecord:
    scenario_key: str
    scenario_kind: str
    factor_mult: float | None
    shape_id: str | None
    duracion_horas: float
    tiempo_al_pico_h: float
    config: dict[str, object]
```

Reject negative flow/time, unknown node IDs, duplicate steps, and a
`tiempo_al_pico_h` greater than `duracion_horas`.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/database/test_repositories_v17.py -q
git add swmm_resilience/database/repositories.py tests/database/test_repositories_v17.py
git commit -m "feat: persist immutable networks and scenarios"
```

### Task 3: Implement Recoverable Run Lifecycle And Batched Writes

**Files:**
- Create: `swmm_resilience/database/simulation_store.py`
- Create: `tests/database/test_simulation_store.py`

**Interfaces:**
- Produces: `SimulationStore`
- Consumes: contract-valid feature/result/timestep rows

- [ ] **Step 1: Write lifecycle tests**

Cover this sequence:

```python
run_id = store.create_run(scenario_id, config_sha256="a" * 64)
assert store.status(run_id) == "PENDING"
store.mark_running(run_id, swmm_version="5.2")
store.write_features(run_id, feature_frame, node_keys)
store.write_results(run_id, result_frame, node_keys)
store.write_timeseries(run_id, iter(timestep_rows), node_keys, batch_size=2)
store.mark_complete(
    run_id,
    expected_nodes=2,
    expected_timesteps=6,
    continuity_error_pct=0.1,
    max_continuity_error_pct=5.0,
)
assert store.status(run_id) == "COMPLETE"
```

Add tests proving:

- `PENDING -> COMPLETE` is rejected;
- duplicate node features roll back;
- missing feature/result nodes prevent completion;
- a continuity error whose absolute value exceeds the configured maximum
  prevents completion;
- a bad 15-column frame fails before insertion;
- `mark_failed()` deletes children and persists failure details;
- `recover_abandoned_runs()` changes stale `RUNNING` rows to `FAILED` and
  deletes their children;
- rows from failed/running runs never appear in `training_samples_v17`.

- [ ] **Step 2: Define allowed transitions and DTO conversion**

```python
ALLOWED_TRANSITIONS = {
    "PENDING": {"RUNNING", "FAILED"},
    "RUNNING": {"COMPLETE", "FAILED"},
    "COMPLETE": set(),
    "FAILED": set(),
}
```

Feature writes call `TABULAR_V3_17.validate_frame()` before opening the write
transaction. Convert pandas missing values to Python `None`; never replace
them with zero.

- [ ] **Step 3: Implement bounded timestep insertion**

```python
def batched(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
```

Use one outer transaction for each logical `write_features`, `write_results`,
and `write_timeseries` call. `write_timeseries` streams bounded `executemany`
batches inside that single transaction, so Python memory stays bounded while a
failed call rolls back every timestep row. Leave the run `RUNNING` until all
writes and counts pass.

- [ ] **Step 4: Implement completion checks**

Before `COMPLETE`, query and compare:

```text
expected node count == node_features count == node_results count
expected timesteps == node_timeseries count
no duplicate keys
all required feature values present
all target constraints valid
absolute continuity error <= configured maximum
```

Persist the verified counts on `runs` in the same transaction as the status
transition.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/database/test_simulation_store.py tests/database/test_training_view_v17.py -q
git add swmm_resilience/database/simulation_store.py tests/database/test_simulation_store.py
git commit -m "feat: add recoverable SQLite run lifecycle"
```

### Task 4: Build One Persisted Simulation Pipeline

**Files:**
- Create: `swmm_resilience/pipeline.py`
- Create: `swmm_resilience/extraction/feature_builder.py`
- Create: `tests/test_pipeline_sqlite.py`
- Modify: `swmm_resilience/simulation/runner.py:356-645`
- Reuse: `swmm_resilience/extraction/static_features.py`
- Reuse: `swmm_resilience/extraction/topology.py`
- Reuse: `swmm_resilience/extraction/dynamic_features.py`

**Interfaces:**
- Produces: `build_feature_frame(conn, scenario_id) -> pd.DataFrame`, `run_scenario(config, scenario, *, pyswmm_api=None) -> int`
- Consumes: `Config`, a canonical scenario, existing SWMM runner output

- [ ] **Step 1: Write an integration test with a fake runner**

The test constructs two nodes, one scenario, four timesteps per node, and a
fake result matching the current `run_simulation()` keys. Assert:

```python
run_id = run_scenario(config, scenario, pyswmm_api=fake_api)
assert scalar(conn, "SELECT status FROM runs WHERE run_id=?", run_id) == "COMPLETE"
assert scalar(conn, "SELECT COUNT(*) FROM node_features WHERE run_id=?", run_id) == 2
assert scalar(conn, "SELECT COUNT(*) FROM node_results WHERE run_id=?", run_id) == 2
assert scalar(conn, "SELECT COUNT(*) FROM node_timeseries WHERE run_id=?", run_id) == 8
assert load_training_samples(conn)[list(FEATURE_COLUMNS_V17)].shape == (2, 17)
```

Add a fake runner exception and assert a `FAILED` run with zero child rows.

- [ ] **Step 2: Extract a result dataclass without changing values**

Add to `runner.py`:

```python
@dataclass
class SimulationResult:
    node_records: list[dict]
    link_records: list[dict]
    run_inputs: list[dict]
    node_timeseries_records: list[dict]
    summary: dict
```

Return this type from a new wrapper `run_simulation_result()` that adapts the
current dictionary. Keep `run_simulation()` unchanged temporarily for legacy
tests; remove the adapter in Plan D.

- [ ] **Step 3: Implement the orchestration in dependency order**

`run_scenario()` must:

1. connect and apply migrations;
2. ingest `.inp` bytes and topology;
3. persist scenario and inflow series;
4. create/mark the run;
5. invoke SWMM;
6. construct static+topology features;
7. call `build_feature_frame(conn, scenario_id)`, which reads persisted
   network/topology/scenario inflows and derives dynamic features with
   persisted duration/time-to-peak;
8. return exactly `node_id` plus `FEATURE_COLUMNS_V17` and validate the feature
   slice;
9. map current result names to `inunda` and `vol_inundacion_m3`;
10. insert aggregate and temporal data;
11. verify and mark complete;
12. on any exception, mark failed with stage metadata, clean children, and
    re-raise;
13. in `finally`, run `optimize_database()` after substantial writes and close
    the connection cleanly.

Do not call `assemble_dataset()` or any `to_csv()`/`to_parquet()` function.

- [ ] **Step 4: Prove training/inference feature parity**

Implement the builder in `extraction/feature_builder.py`; it must not accept a
result/target DataFrame. Add a test that calls the same public feature builder for one factor scenario
and one equivalent hydrograph scenario, then asserts both return columns
exactly equal to `FEATURE_COLUMNS_V17` and identical duration/time-to-peak
semantics. Mutate persisted `node_results` targets after building features and
assert rebuilding produces identical 17 values; this proves
`q_pico_nodo`/`q_pico_acum_escalado` come from scenario inflows and topology,
not post-SWMM flooding targets.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/test_pipeline_sqlite.py tests/test_dynamic_dataset_validation.py tests/test_scenario_predict.py tests/simulation/test_flood_volume_extraction.py -q
git add swmm_resilience/pipeline.py swmm_resilience/extraction/feature_builder.py swmm_resilience/simulation/runner.py tests/test_pipeline_sqlite.py
git commit -m "feat: persist complete SWMM runs to SQLite"
```

### Task 5: Cut Root Simulation Commands Over To SQLite

**Files:**
- Modify: `main.py:49-620`
- Modify: `tests/test_simulate_single_factor.py`
- Create: `tests/test_cli_sqlite.py`

**Interfaces:**
- Consumes: `run_scenario()` and `Config.database`
- Produces: supported `python main.py` and `python main.py --simulate --factor X` persistence behavior

- [ ] **Step 1: Write CLI tests first**

Assert root commands pass `config.database.path` to the new pipeline, print the
created `run_id`, and never create `dataset_final.csv`, `dataset_ml.csv`, or a
Parquet file in `tmp_path`.

- [ ] **Step 2: Replace the default extraction path**

The default pipeline iterates configured factors/shapes and calls
`run_scenario()`. Remove calls to `assemble_dataset()` from the operational
branch. Retain `--export-csv PATH` as the sole explicit flat-file export; it
queries the canonical view after persistence and never becomes a training
input.

- [ ] **Step 3: Preserve `--simulate` map behavior**

After persistence, build the SWMM map from rows queried by `run_id`. Keep
runtime annotations added on `main`. ML prediction remains on the old model
loader until Plan C switches it; tests must make that temporary boundary
explicit.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/test_cli_sqlite.py tests/test_simulate_single_factor.py tests/test_factor_comparison_cli.py -q
git add main.py tests/test_cli_sqlite.py tests/test_simulate_single_factor.py
git commit -m "feat: make root simulations SQLite-backed"
```

### Task 6: Run The Persistence Phase Gate

**Files:**
- Verify only

**Interfaces:**
- Produces: approved persisted pipeline for Plan C
- Consumes: Tasks 1-5

- [ ] **Step 1: Run focused persistence tests**

```powershell
python -m pytest tests/database/test_repositories_v17.py tests/database/test_simulation_store.py tests/test_pipeline_sqlite.py tests/test_cli_sqlite.py tests/test_simulate_single_factor.py -q
```

- [ ] **Step 2: Run simulation/extraction regression tests**

```powershell
python -m pytest tests/simulation tests/test_dynamic_dataset_validation.py tests/test_labels.py tests/test_timeseries_scenario.py tests/test_hydrograph_batch.py -q
```

- [ ] **Step 3: Confirm the new path has no mandatory flat-file writes**

```powershell
rg -n "assemble_dataset|to_csv|to_parquet|save_node_timeseries_parquet" swmm_resilience/pipeline.py main.py
```

Expected: no operational persistence matches. Optional export commands, if
retained, are named and isolated.

- [ ] **Step 4: Check clean branch state**

```powershell
git diff --check
git status --short --branch
```

Expected: no uncommitted implementation files.
