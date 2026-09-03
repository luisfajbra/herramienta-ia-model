# SQLite V17 Read Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every pipeline command read its training frame from
`training_samples_v17` instead of `data/training/dataset_final.csv`, while the
pipeline keeps writing the CSV unchanged.

**Architecture:** Add a `dataset.db_path` config field and one thin loader
wrapper (`load_training_frame`), then swap `pd.read_csv` for it at each of the
8 read sites, one command at a time, with the full suite green after each.
Nothing is deleted and nothing stops being written, so every task reverts with
a one-line change.

**Tech Stack:** Python 3.11+, SQLite (stdlib `sqlite3`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-sqlite-v17-training-frame-cutover-design.md`

## Global Constraints

- Feature contract ID is exactly `tabular_v3_17` with the 17 ordered features
  from `swmm_resilience/ml/contracts.py`. Do not add, remove, or reorder them.
- The loader frame has **27 columns**: 8 identity (`run_id`, `network_id`,
  `scenario_id`, `scenario_key`, `scenario_kind`, `factor_mult`, `shape_id`,
  `node_id`) + 17 features + 2 targets (`inunda`, `vol_inundacion_m3`). It does
  **not** contain `coord_x`/`coord_y`. This is intended (spec §4.6).
- Do not move or rename `swmm_resilience/database/training_queries.py` (spec §4.1).
- Do not add SQL-side filters to `load_training_samples` (spec §4.2). Consumers
  keep filtering in pandas.
- The pipeline keeps writing `dataset_final.csv` for the entire duration of this
  plan. Removing that write is a later phase and is currently blocked (see
  "Phase 2 blocker" at the end of this file).
- `--persist-sql` keeps using `pd.read_csv` in this plan. It needs
  `coord_x`/`coord_y` for `backfill_networks_and_runs`, which the loader frame
  does not carry.
- Run `python -m pytest -q` after every task. It must be green before commit.
- Do not refactor `tests/database/test_csv_backfill.py::_synthetic_dataset`.
  New tests get their fixtures from `tests/conftest.py` (Task 2).

---

### Task 1: `dataset.db_path` configuration field

**Files:**
- Modify: `config.yaml:11-13`
- Modify: `swmm_resilience/config.py:195-198` (`DatasetConfig`), `swmm_resilience/config.py:302-305` (`load_config`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `config.dataset.db_path: Path` — an absolute path, resolved
  relative to the directory holding `config.yaml`, defaulting to
  `<config dir>/outputs/training_v17.sqlite3` when the YAML key is absent.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_config.py`:

```python
def test_load_config_resolves_db_path(tmp_path):
    cfg_path = write_config(tmp_path)
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(
        text.replace(
            '  flood_threshold_m3: 0.0',
            '  db_path: "outputs/custom_v17.sqlite3"\n  flood_threshold_m3: 0.0',
            1,
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.dataset.db_path == tmp_path / "outputs" / "custom_v17.sqlite3"


def test_load_config_defaults_db_path_when_absent(tmp_path):
    cfg = load_config(write_config(tmp_path))

    assert cfg.dataset.db_path == tmp_path / "outputs" / "training_v17.sqlite3"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: both new tests FAIL with `AttributeError: 'DatasetConfig' object has no attribute 'db_path'`.

- [ ] **Step 3: Add the field to the dataclass**

In `swmm_resilience/config.py`, replace the `DatasetConfig` definition:

```python
@dataclass
class DatasetConfig:
    output_path: Path
    flood_threshold_m3: float
    db_path: Path
```

- [ ] **Step 4: Resolve the field in `load_config`**

In `swmm_resilience/config.py`, replace the `dataset=DatasetConfig(...)` block
inside `load_config`:

```python
        dataset=DatasetConfig(
            output_path=base_dir / ds["output_path"],
            flood_threshold_m3=float(ds["flood_threshold_m3"]),
            db_path=base_dir / ds.get("db_path", "outputs/training_v17.sqlite3"),
        ),
```

- [ ] **Step 5: Add the key to the real config**

In `config.yaml`, replace the `dataset:` block:

```yaml
dataset:
  output_path: "data/training/dataset_final.csv"
  db_path: "outputs/training_v17.sqlite3"
  flood_threshold_m3: 1.0
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_config.py tests/test_config_defaults.py -q`
Expected: PASS, including the two new tests.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures versus the pre-task baseline.

- [ ] **Step 8: Commit**

```bash
git add config.yaml swmm_resilience/config.py tests/test_config.py
git commit -m "feat: add dataset.db_path config field"
```

---

### Task 2: `load_training_frame` wrapper and shared SQL test fixtures

**Files:**
- Modify: `swmm_resilience/database/training_queries.py`
- Modify: `tests/conftest.py`
- Test: `tests/database/test_load_training_frame.py` (create)

**Interfaces:**
- Consumes: `config.dataset.db_path` (Task 1) — callers pass it in
- Produces:
  - `load_training_frame(db_path: str | Path) -> pd.DataFrame` in
    `swmm_resilience.database.training_queries` — returns the 27-column frame
    for every `COMPLETE` run, ordered by `(run_id, node_id)`; raises
    `ValueError("No COMPLETE v17 training samples found")` for an empty or
    absent database.
  - pytest fixture `csv_shaped_dataset` → a 24-row, 24-column
    `dataset_final.csv`-shaped `DataFrame` (2 shapes × 3 factors × 4 nodes,
    8 flooded rows).
  - pytest fixture `sql_training_db` → `Path` to a migrated SQLite v17
    database populated from `csv_shaped_dataset`.

- [ ] **Step 1: Add the shared fixtures**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def csv_shaped_dataset() -> pd.DataFrame:
    """A dataset_final.csv-shaped frame: the same 24 columns the assembler writes."""
    rows = []
    for shape_id in ("base", "storm_a"):
        for factor in (1.0, 2.0, 3.0):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "coord_x": float(node_idx),
                        "coord_y": float(node_idx * 2),
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "duracion_horas": 3.0,
                        "tiempo_al_pico_h": 0.7,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sql_training_db(tmp_path: Path, csv_shaped_dataset: pd.DataFrame) -> Path:
    """Path to a migrated v17 database holding csv_shaped_dataset's rows."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / "fixture_network.inp"
    inp_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")
    db_path = tmp_path / "training_v17.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(
            conn, csv_shaped_dataset, inp_path, "Fixture Network"
        )
    finally:
        conn.close()
    return db_path
```

- [ ] **Step 2: Write the failing tests**

Create `tests/database/test_load_training_frame.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.database.training_queries import (
    IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    load_training_frame,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def test_returns_canonical_27_column_frame(sql_training_db, csv_shaped_dataset):
    frame = load_training_frame(sql_training_db)

    assert frame.columns.tolist() == (
        list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS_V17) + list(TARGET_COLUMNS)
    )
    assert len(frame.columns) == 27
    assert len(frame) == len(csv_shaped_dataset)
    assert "coord_x" not in frame.columns
    assert "coord_y" not in frame.columns


def test_shared_columns_match_the_csv_shaped_source(
    sql_training_db, csv_shaped_dataset
):
    shared = [
        column
        for column in csv_shaped_dataset.columns
        if column not in ("coord_x", "coord_y")
    ]
    sort_keys = ["shape_id", "factor_mult", "node_id"]

    frame = load_training_frame(sql_training_db)

    actual = (
        frame[shared].sort_values(sort_keys).reset_index(drop=True)
    )
    expected = (
        csv_shaped_dataset[shared].sort_values(sort_keys).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_raises_for_a_database_with_no_complete_samples(tmp_path):
    with pytest.raises(ValueError, match="No COMPLETE v17 training samples found"):
        load_training_frame(tmp_path / "empty.sqlite3")


def test_leaves_no_open_connection_behind(sql_training_db):
    first = load_training_frame(sql_training_db)
    second = load_training_frame(sql_training_db)

    assert len(first) == len(second)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/database/test_load_training_frame.py -q`
Expected: FAIL with `ImportError: cannot import name 'load_training_frame'`.

- [ ] **Step 4: Implement the wrapper**

In `swmm_resilience/database/training_queries.py`, add these imports below the
existing `from ..ml.contracts import ...` line:

```python
from .connection import connect_managed_database
from .migrations import apply_migrations
```

Then append this function to the end of the file:

```python
def load_training_frame(db_path: str | Path) -> pd.DataFrame:
    """Load every COMPLETE v17 training sample from the database at ``db_path``.

    Convenience wrapper for CLI callers: opens a managed connection, applies
    pending migrations, reads the canonical frame, and closes the connection.
    Raises ``ValueError`` when the database holds no COMPLETE samples yet.

    The read leaves a transaction open (``load_training_samples`` reads inside
    a savepoint); closing discards it, which is safe because nothing here
    writes.
    """
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        return load_training_samples(conn)
    finally:
        conn.close()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/database/test_load_training_frame.py -q`
Expected: all 4 tests PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/database/training_queries.py tests/conftest.py tests/database/test_load_training_frame.py
git commit -m "feat: add load_training_frame wrapper for CLI callers"
```

---

### Task 3: `--resilience-curve` and `--flood-volume-curve` read from SQL

**Files:**
- Modify: `main.py:17` (imports), `main.py:214`, `main.py:225`
- Test: `tests/test_cli_sql_reads.py` (create)

**Interfaces:**
- Consumes: `load_training_frame` (Task 2), `config.dataset.db_path` (Task 1)
- Produces: `main.load_training_frame` exists as a module-level name, so tests
  monkeypatch it via `monkeypatch.setattr(main, "load_training_frame", ...)`.

Both commands get the identical change: `compute_resilience_curve` and
`compute_flood_volume_curve` already take a `DataFrame`, so only `main.py`'s
two call sites move.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_sql_reads.py`:

```python
import sys
from pathlib import Path

import pandas as pd

import main


def base_shape_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "node_id": ["N0", "N1", "N0", "N1"],
            "factor_mult": [1.0, 1.0, 2.0, 2.0],
            "shape_id": ["base", "base", "base", "base"],
            "inunda": [0, 1, 1, 1],
            "vol_inundacion_m3": [0.0, 5.0, 7.0, 9.0],
        }
    )


def test_resilience_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "resilience_swmm": [1.0] * len(factors),
                "resilience_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_resilience_curve", fake_curve)
    monkeypatch.setattr(main, "plot_resilience_curve", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--resilience-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]


def test_flood_volume_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "vol_total_swmm": [1.0] * len(factors),
                "vol_total_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_flood_volume_curve", fake_curve)
    monkeypatch.setattr(main, "plot_flood_volume_curve", lambda result, out: None)
    monkeypatch.setattr(main, "plot_flood_volume_combined", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--flood-volume-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_sql_reads.py -q`
Expected: FAIL with `AttributeError: <module 'main'> does not have the attribute 'load_training_frame'`.

- [ ] **Step 3: Import the loader in `main.py`**

Add this line to `main.py`'s import block, immediately after
`from swmm_resilience.dataset.validator import validate_dataset`:

```python
from swmm_resilience.database.training_queries import load_training_frame
```

- [ ] **Step 4: Switch both call sites**

In `main.py`, inside `if args.resilience_curve:`, replace:

```python
        df = base_shape_rows(pd.read_csv(config.dataset.output_path))
```

with:

```python
        df = base_shape_rows(load_training_frame(config.dataset.db_path))
```

Then inside `if args.flood_volume_curve:`, replace the identical
`pd.read_csv` line with the identical `load_training_frame` line.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli_sql_reads.py tests/test_resilience.py tests/test_flood_volume.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_cli_sql_reads.py
git commit -m "refactor: read resilience and volume curves from SQL"
```

---

### Task 4: `generate_factor_comparisons` takes a DataFrame

**Files:**
- Modify: `swmm_resilience/analysis/factor_comparison.py:17-24`
- Modify: `main.py:235-247`
- Test: `tests/analysis/test_factor_comparison.py:8-70`, `tests/test_factor_comparison_cli.py`

**Interfaces:**
- Consumes: `load_training_frame` (Task 2)
- Produces: `generate_factor_comparisons(frame: pd.DataFrame, config, models_dir: Path, output_dir: Path) -> list[Path]`
  — the `dataset_path` parameter is gone; callers load the frame themselves.

- [ ] **Step 1: Update the unit test to pass a frame**

In `tests/analysis/test_factor_comparison.py`, replace the CSV setup at the top
of `test_generate_factor_comparisons_processes_every_factor`:

```python
    frame = pd.DataFrame(
        {
            "node_id": ["1C", "2C", "1C", "2C"],
            "factor_mult": [1.0, 1.0, 2.0, 2.0],
            "vol_inundacion_m3": [1.0, 2.0, 3.0, 4.0],
            "inunda": [1, 1, 1, 1],
        }
    )
```

and replace the call at the bottom of the same test:

```python
    paths = factor_comparison.generate_factor_comparisons(
        frame=frame,
        config=config,
        models_dir=tmp_path / "models",
        output_dir=out_dir,
    )
```

- [ ] **Step 2: Update the CLI test**

Replace the body of `tests/test_factor_comparison_cli.py` with:

```python
import sys
from pathlib import Path

import pandas as pd

import main


def test_factor_comparison_cli_generates_plots_in_metrics_directory(
    monkeypatch, capsys
):
    calls = []
    load_calls = []
    expected_paths = [
        Path("outputs/metrics/factor_comparison/volume_by_node_factor_1.00.png"),
        Path("outputs/metrics/factor_comparison/parity_factor_1.00.png"),
    ]
    frame = pd.DataFrame(
        {
            "node_id": ["1C"],
            "factor_mult": [1.0],
            "shape_id": ["base"],
            "vol_inundacion_m3": [1.0],
            "inunda": [1],
        }
    )

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return frame

    def fake_generate(frame_arg, config, models_dir, output_dir):
        calls.append((frame_arg, config, models_dir, output_dir))
        return expected_paths

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "generate_factor_comparisons", fake_generate)
    monkeypatch.setattr(sys, "argv", ["main.py", "--factor-comparison"])

    main.main()

    assert len(calls) == 1
    frame_arg, config, models_dir, output_dir = calls[0]
    assert load_calls == [config.dataset.db_path]
    assert frame_arg.equals(frame)
    assert models_dir == main.MODELS_DIR
    assert output_dir == main.METRICS_DIR / "factor_comparison"
    output = capsys.readouterr().out
    assert all(str(path) in output for path in expected_paths)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/analysis/test_factor_comparison.py tests/test_factor_comparison_cli.py -q`
Expected: FAIL — `generate_factor_comparisons() got an unexpected keyword argument 'frame'`.

- [ ] **Step 4: Change the function signature**

In `swmm_resilience/analysis/factor_comparison.py`, replace the function header
and its first body line:

```python
def generate_factor_comparisons(
    frame: pd.DataFrame,
    config,
    models_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate node-volume and parity plots for every dataset factor."""
    dataset = base_shape_rows(frame)
```

- [ ] **Step 5: Update the CLI call site**

In `main.py`, inside `if args.factor_comparison:`, replace the
`generate_factor_comparisons(...)` call:

```python
        paths = generate_factor_comparisons(
            frame=load_training_frame(config.dataset.db_path),
            config=config,
            models_dir=MODELS_DIR,
            output_dir=output_dir,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/analysis/test_factor_comparison.py tests/test_factor_comparison_cli.py -q`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
git add swmm_resilience/analysis/factor_comparison.py main.py tests/analysis/test_factor_comparison.py tests/test_factor_comparison_cli.py
git commit -m "refactor: pass the training frame into generate_factor_comparisons"
```

---

### Task 5: `--only-maps` and the training read branch

**Files:**
- Modify: `main.py:614`, `main.py:744-747`
- Test: `tests/test_cli_sql_reads.py` (extend)

**Interfaces:**
- Consumes: `load_training_frame` (Task 2)
- Produces: no new interface — `--only-maps`, `--only-ml`, and
  `--skip-extraction` all source their frame from SQL.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_sql_reads.py`:

```python
def test_only_ml_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    train_calls = []
    frame = base_shape_frame()

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return frame

    def fake_train(df, config, models_dir):
        train_calls.append((df.copy(), config))
        return object(), object()

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "train_models", fake_train)
    monkeypatch.setattr(main, "evaluate_models", lambda df, config, out: {})
    monkeypatch.setattr(
        main, "generate_feature_importance_plots", lambda clf, reg, out: None
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--only-ml"])

    main.main()

    config = train_calls[0][1]
    assert load_calls == [config.dataset.db_path]
    assert train_calls[0][0].equals(frame)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli_sql_reads.py::test_only_ml_reads_the_frame_from_sql -q`
Expected: FAIL — the assertion on `load_calls` fails because `--only-ml` still
calls `pd.read_csv` (`load_calls` is empty).

- [ ] **Step 3: Switch the `--only-maps` read**

In `main.py`, inside `if args.only_maps:`, replace:

```python
        df_all = pd.read_csv(config.dataset.output_path)
```

with:

```python
        df_all = load_training_frame(config.dataset.db_path)
```

- [ ] **Step 4: Switch the training read branch**

In `main.py`, replace the `else:` branch that follows
`if not use_existing_dataset:` (the one printing "Leyendo dataset desde"):

```python
    else:
        print(f"\nLeyendo dataset desde {config.dataset.db_path}...")
        df = load_training_frame(config.dataset.db_path)
        print(f"  {df.shape[0]} filas × {df.shape[1]} cols")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_cli_sql_reads.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_cli_sql_reads.py
git commit -m "refactor: read maps and training frames from SQL"
```

---

### Task 6: `--analyze-features`, `--evaluate-shapes`, `--evaluate-generalization`

**Files:**
- Modify: `main.py:250-296` (`--analyze-features`), `main.py:298-331` (`--evaluate-shapes`), `main.py:375-406` (`--evaluate-generalization`)
- Test: `tests/test_cli_sql_reads.py` (extend)

**Interfaces:**
- Consumes: `load_training_frame` (Task 2)
- Produces: no new interface. All three commands replace their
  `config.dataset.output_path.exists()` precondition with a `try/except
  ValueError` around `load_training_frame`, reporting the same class of
  user-facing error through `parser.error`.

Keep each command's existing check order (dataset first, then models) so the
error messages users already see stay in the same sequence.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_sql_reads.py`:

```python
def test_analyze_features_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--analyze-features"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--analyze-features should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli_sql_reads.py::test_analyze_features_errors_when_sql_has_no_samples -q`
Expected: FAIL — the command still checks the CSV path, so it errors about
`dataset_final.csv` (or about missing models) instead of the database.

- [ ] **Step 3: Switch `--analyze-features`**

In `main.py`, inside `if args.analyze_features:`, replace the block that starts
with `dataset_path = Path(config.dataset.output_path)` and ends with the
`parser.error(...)` for a missing dataset:

```python
        try:
            df_analysis = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--analyze-features requiere datos en {config.dataset.db_path}; "
                f"ejecuta el pipeline completo primero ({load_error})"
            )
```

Then delete the now-duplicated read further down the same branch:

```python
        df_analysis = pd.read_csv(dataset_path)
```

- [ ] **Step 4: Switch `--evaluate-shapes`**

In `main.py`, inside `if args.evaluate_shapes:`, replace the precondition:

```python
        try:
            _df = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--evaluate-shapes requiere datos en {config.dataset.db_path}; "
                f"ejecuta el pipeline completo primero ({load_error})"
            )
```

Then delete the later read in the same branch:

```python
        _df = pd.read_csv(config.dataset.output_path)
```

- [ ] **Step 5: Switch `--evaluate-generalization`**

In `main.py`, inside `if args.evaluate_generalization:`, replace the
precondition:

```python
        try:
            _df = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--evaluate-generalization requiere datos en {config.dataset.db_path}; "
                f"ejecuta el pipeline completo primero ({load_error})"
            )
```

Then delete the later read in the same branch:

```python
        _df = pd.read_csv(config.dataset.output_path)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_cli_sql_reads.py -q`
Expected: PASS.

- [ ] **Step 7: Confirm only `--persist-sql` still reads the CSV**

Run: `rg -n "read_csv" main.py`
Expected: exactly one match, inside the `if args.persist_sql:` branch. That one
is intentional (see Global Constraints).

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
git add main.py tests/test_cli_sql_reads.py
git commit -m "refactor: read analysis and shape-evaluation frames from SQL"
```

---

### Task 7: Document the finished read cutover

**Files:**
- Modify: `docs/FLUJO_ACTUAL.md` (§12.3, §12.8 step 3, §12.9)
- Modify: `COMANDOS.md` (architecture summary and "Qué NO está conectado todavía")

**Interfaces:**
- Consumes: the completed Tasks 1-6
- Produces: documentation matching the code

- [ ] **Step 1: Record the real test count**

Run: `python -m pytest -q`
Copy the exact summary line (for example `512 passed, 3 deselected`). Both
documents currently carry an unverified count with a note saying so; replace
the note with this measured number in both files.

- [ ] **Step 2: Update `docs/FLUJO_ACTUAL.md`**

In §12.3, replace the closing paragraph that says no consumer calls
`load_training_samples` with a statement that every read path now goes through
`load_training_frame`, and that `--persist-sql` is the only remaining
`pd.read_csv` caller. In §12.8, mark step 3 as done, listing the eight
migrated call sites. In §12.9, tick "Ningún módulo del pipeline llama a
`pd.read_csv` sobre el dataset" with the `--persist-sql` exception noted.

- [ ] **Step 3: Update `COMANDOS.md`**

In the architecture summary, replace the sentence stating that nothing in the
pipeline calls the loader: reads now come from
`outputs/training_v17.sqlite3`, while writes still produce both the CSV and
(from the later phase) SQL. Add `dataset.db_path` to the configuration notes.

- [ ] **Step 4: Commit**

```bash
git add docs/FLUJO_ACTUAL.md COMANDOS.md
git commit -m "docs: record the completed SQL read cutover"
```

---

## Phase 2 blocker — resolve before planning the write cutover

Writing the assembler's SQL path (spec §6) cannot be planned concretely yet.
Verified while writing this plan:

`scenarios` carries `UNIQUE(network_id, scenario_key, config_sha256)`
(`swmm_resilience/database/sql/001_v17_initial.sql:60`), and
`backfill_networks_and_runs` always `INSERT`s a scenario row, deriving
`scenario_key` as `f"{shape_id}__f{factor:.2f}"` and `config_sha256` from a
deterministic payload. Its own docstring says it is "NOT idempotent on
scenarios/runs — call this once per dataset snapshot."

Therefore, once `assemble_dataset` writes to SQL, a **second** `python main.py`
run against the same database raises `sqlite3.IntegrityError: UNIQUE constraint
failed` at the write step — after having already spent the full SWMM sweep.
Re-running the pipeline is routine, so this must be designed, not patched.

Two sub-questions, both needing a decision:

1. **Scenario reuse.** The schema's intent appears to be that `scenarios` are
   deduplicated configurations while `runs` are executions (many runs per
   scenario, no unique constraint on `runs.scenario_id`). Making
   `backfill_networks_and_runs` SELECT-or-INSERT the scenario — exactly as it
   already does for `networks` and `nodes` — and always insert a fresh `runs`
   row would make re-runs work naturally. This changes a function that
   currently has passing tests asserting insert counts
   (`tests/database/test_csv_backfill.py:71-89`).
2. **Which runs to train on.** After two pipeline runs, every run is
   `COMPLETE`, so `load_training_frame` returns both snapshots — roughly
   double the rows, which silently changes training and breaks
   `validate_dataset`'s `n_nodes × n_simulations` arithmetic. Options: have
   the pipeline pass the `run_ids` it just inserted (already available from
   `backfill_networks_and_runs`'s `run_id_by_key`, and `load_training_samples`
   already accepts `run_ids`); or add a "latest run per scenario" resolver for
   the `--only-ml`/`--skip-extraction` path; or introduce an explicit training
   snapshot concept.

Phase 1 (Tasks 1-7) is unaffected: the database is populated by `--persist-sql`
exactly as it is today, and each of those tasks is independently revertible.

## Self-review notes

- **Spec coverage:** this plan implements spec §5 (Phase 1) and the §4.1-4.4,
  §4.6 decisions. Spec §6 (Phase 2), §7 (Phase 3), and the §6.4 parity gate are
  deliberately not planned here — see the blocker above. Spec §4.5
  (`--persist-sql` narrowing) and §4.7 (`validate_dataset`) belong to Phase 3
  and depend on the blocker's resolution.
- **Interface consistency:** `load_training_frame(db_path)` is used with that
  exact single-argument signature in Tasks 3, 5, and 6, and monkeypatched with
  the matching `fake_load(db_path)` shape in every test.
  `generate_factor_comparisons` is called with the keyword `frame=` in both
  `main.py` and both of its tests after Task 4.
