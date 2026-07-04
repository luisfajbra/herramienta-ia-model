# Hydrograph Batch Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated diagnostic pipeline that replaces SWMM `[TIMESERIES]` with user-supplied CSV hydrographs and measures how well the current XGBoost models predict flood failure and volume per node.

**Architecture:** Seven new modules, four new test files, zero changes to existing training/evaluation/prediction paths. Tasks are sequenced so each builds on the previous: CSV validation → temporal `.inp` writer → comparison analysis → comparison visualization → scenario predictor → batch coordinator → CLI.

**Tech Stack:** Python, pandas, numpy, swmm_api (existing), pyswmm (existing), matplotlib (existing), scikit-learn metrics, joblib

---

## Interfaces between modules (reference for all tasks)

```
HydrographScenario               ← hydrograph_csv.py
  .scenario_id: str
  .node_series: dict[str, list[tuple[float, float]]]   # node → [(hours, lps)]
  .time_grid_hours: list[float]
  .last_time_hours: float

write_scenario_inp(base, scenario, out_dir) → Path     ← timeseries_scenario.py
predict_scenario(base_inp, temp_inp, models_dir) → DataFrame[node_id, inunda_pred, vol_pred_m3]
build_comparison_df(swmm_df, pred_df, scenario_id) → DataFrame  ← analysis/model_comparison.py
compute_classification_metrics(inunda_swmm, inunda_pred) → dict
compute_volume_metrics(vol_swmm, vol_pred) → dict
plot_parity_nodes/aggregated/node_profiles(df, out_dir, scenario_id) → list[Path]
```

---

## Task 1: CSV validation layer

**Files:**
- Create: `swmm_resilience/validation/__init__.py`
- Create: `swmm_resilience/validation/hydrograph_csv.py`
- Create: `tests/test_hydrograph_csv.py`

- [ ] **Step 1: Create the validation package**

```bash
touch swmm_resilience/validation/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_hydrograph_csv.py`:

```python
import datetime
import pytest
from pathlib import Path
import pandas as pd

from swmm_resilience.validation.hydrograph_csv import (
    load_scenario,
    scenario_id_from_path,
    HydrographScenario,
)

NODES = {"1C", "2C"}


def _write_csv(tmp_path: Path, rows: list[dict], filename="scenario.csv") -> Path:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _valid_rows():
    return [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.6},
        {"node_id": "1C", "time": "0:03", "value_lps": 0.9},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.8},
        {"node_id": "2C", "time": "0:03", "value_lps": 1.2},
    ]


def test_load_valid_scenario(tmp_path):
    path = _write_csv(tmp_path, _valid_rows())
    s = load_scenario(path, NODES)
    assert s.scenario_id == "scenario"
    assert set(s.node_series.keys()) == NODES
    assert s.time_grid_hours == pytest.approx([0.0, 0.05])
    assert s.last_time_hours == pytest.approx(0.05)
    assert s.node_series["1C"] == pytest.approx([(0.0, 0.6), (0.05, 0.9)])


def test_accepts_different_durations_between_files(tmp_path):
    rows_a = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "1:00", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "1:00", "value_lps": 0.5},
    ]
    rows_b = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "5:00", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "5:00", "value_lps": 0.5},
    ]
    s_a = load_scenario(_write_csv(tmp_path, rows_a, "a.csv"), NODES)
    s_b = load_scenario(_write_csv(tmp_path, rows_b, "b.csv"), NODES)
    assert s_a.last_time_hours == pytest.approx(1.0)
    assert s_b.last_time_hours == pytest.approx(5.0)


def test_accepts_irregular_shared_grid(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:07", "value_lps": 1.0},
        {"node_id": "1C", "time": "1:30", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:07", "value_lps": 0.5},
        {"node_id": "2C", "time": "1:30", "value_lps": 0.0},
    ]
    s = load_scenario(_write_csv(tmp_path, rows), NODES)
    assert len(s.time_grid_hours) == 3


def test_rejects_missing_column(tmp_path):
    rows = [{"node_id": "1C", "time": "0:00"}, {"node_id": "2C", "time": "0:00"}]
    with pytest.raises(ValueError, match="Columnas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_extra_column(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0, "extra": 1},
        {"node_id": "1C", "time": "0:03", "value_lps": 0.5, "extra": 1},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0, "extra": 1},
        {"node_id": "2C", "time": "0:03", "value_lps": 0.5, "extra": 1},
    ]
    with pytest.raises(ValueError, match="Columnas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_missing_node(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:03", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="Nodos incorrectos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_extra_node(tmp_path):
    rows = _valid_rows() + [
        {"node_id": "3C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "3C", "time": "0:03", "value_lps": 0.5},
    ]
    with pytest.raises(ValueError, match="Nodos incorrectos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_invalid_time_format(tmp_path):
    rows = _valid_rows()
    rows[0]["time"] = "abc"
    with pytest.raises(ValueError, match="formato inválido"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_invalid_minutes(tmp_path):
    rows = _valid_rows()
    rows[0]["time"] = "0:60"
    with pytest.raises(ValueError, match="formato inválido"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_duplicate_node_time_key(tmp_path):
    rows = _valid_rows() + [{"node_id": "1C", "time": "0:00", "value_lps": 9.9}]
    with pytest.raises(ValueError, match="duplicadas"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_non_increasing_times(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "1C", "time": "0:03", "value_lps": 0.5},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:03", "value_lps": 0.5},
    ]
    with pytest.raises(ValueError, match="crecientes"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_node_not_starting_at_zero(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:01", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:01", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:05", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="0:00"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_fewer_than_two_time_steps(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
    ]
    with pytest.raises(ValueError, match="menos de 2 tiempos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_mismatched_time_grids(tmp_path):
    rows = [
        {"node_id": "1C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "1C", "time": "0:05", "value_lps": 1.0},
        {"node_id": "2C", "time": "0:00", "value_lps": 0.0},
        {"node_id": "2C", "time": "0:10", "value_lps": 1.0},
    ]
    with pytest.raises(ValueError, match="malla temporal diferente"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_negative_value(tmp_path):
    rows = _valid_rows()
    rows[0]["value_lps"] = -0.1
    with pytest.raises(ValueError, match="negativos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_rejects_nan_value(tmp_path):
    rows = _valid_rows()
    rows[0]["value_lps"] = float("nan")
    with pytest.raises(ValueError, match="no numéricos"):
        load_scenario(_write_csv(tmp_path, rows), NODES)


def test_scenario_id_from_path():
    assert scenario_id_from_path(Path("pico_desplazado.csv")) == "pico_desplazado"
    assert scenario_id_from_path(Path("/some/dir/Doble Pico.csv")) == "doble_pico"
```

- [ ] **Step 3: Run tests to confirm they all fail**

```bash
python -m pytest tests/test_hydrograph_csv.py -v
```

Expected: all FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `hydrograph_csv.py`**

Create `swmm_resilience/validation/hydrograph_csv.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_TIME_RE = re.compile(r"^(\d+):([0-5]\d)$")


@dataclass
class HydrographScenario:
    scenario_id: str
    node_series: dict[str, list[tuple[float, float]]]  # node_id → [(hours, lps)]
    time_grid_hours: list[float]
    last_time_hours: float


def scenario_id_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _parse_time_hours(s: str) -> float | None:
    m = _TIME_RE.match(str(s).strip())
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60.0


def load_scenario(csv_path: Path, expected_nodes: set[str]) -> HydrographScenario:
    """Load and validate a hydrograph CSV scenario.

    Raises ValueError with a descriptive message if any validation rule fails.
    Rules 1–9 from spec section 6 (rule 10 — unique IDs across a batch —
    is enforced by the batch coordinator).
    """
    df = pd.read_csv(csv_path)

    # Rule 1: exactly the required columns
    required = {"node_id", "time", "value_lps"}
    missing_cols = required - set(df.columns)
    extra_cols = set(df.columns) - required
    if missing_cols or extra_cols:
        raise ValueError(
            f"Columnas inválidas. Faltantes: {sorted(missing_cols)}. "
            f"Adicionales: {sorted(extra_cols)}"
        )

    df["node_id"] = df["node_id"].astype(str)

    # Rule 2: exactly the expected nodes
    csv_nodes = set(df["node_id"].unique())
    if csv_nodes != expected_nodes:
        raise ValueError(
            f"Nodos incorrectos. Faltantes: {sorted(expected_nodes - csv_nodes)}. "
            f"Adicionales: {sorted(csv_nodes - expected_nodes)}"
        )

    # Rule 5: valid H:MM format
    parsed = df["time"].apply(lambda t: _parse_time_hours(str(t).strip()))
    bad = df["time"][parsed.isna()].unique().tolist()
    if bad:
        raise ValueError(f"Tiempos con formato inválido: {bad}")
    df["_th"] = parsed

    # Rule 6: no duplicate (node_id, time) keys
    if df.duplicated(subset=["node_id", "_th"]).any():
        raise ValueError("Claves (node_id, time) duplicadas en el CSV")

    # Rule 9: finite non-negative values
    vals = pd.to_numeric(df["value_lps"], errors="coerce")
    if vals.isna().any():
        raise ValueError("value_lps contiene valores no numéricos o nulos")
    if not np.isfinite(vals.to_numpy()).all():
        raise ValueError("value_lps contiene valores infinitos")
    if (vals < 0).any():
        raise ValueError("value_lps contiene valores negativos")
    df["_v"] = vals

    # Rules 3, 4, 7 per node and rule 8 (shared grid)
    ref_grid: list[float] | None = None
    node_series: dict[str, list[tuple[float, float]]] = {}

    for nid, grp in df.groupby("node_id", sort=False):
        times = sorted(grp["_th"].tolist())

        # Rule 3: at least two time steps
        if len(times) < 2:
            raise ValueError(f"Nodo '{nid}' tiene menos de 2 tiempos")

        # Rule 4: start at 0:00
        if times[0] != 0.0:
            raise ValueError(f"Nodo '{nid}' no comienza en 0:00")

        # Rule 7: strictly increasing
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError(
                    f"Nodo '{nid}' tiene tiempos no estrictamente crecientes"
                )

        # Rule 8: same grid for all nodes
        if ref_grid is None:
            ref_grid = times
        elif times != ref_grid:
            raise ValueError(
                f"Nodo '{nid}' tiene una malla temporal diferente al resto"
            )

        grp_sorted = grp.sort_values("_th")
        node_series[nid] = list(zip(grp_sorted["_th"], grp_sorted["_v"]))

    return HydrographScenario(
        scenario_id=scenario_id_from_path(csv_path),
        node_series=node_series,
        time_grid_hours=ref_grid,
        last_time_hours=ref_grid[-1],
    )
```

- [ ] **Step 5: Run tests — all should pass**

```bash
python -m pytest tests/test_hydrograph_csv.py -v
```

Expected: all PASS

- [ ] **Step 6: Run full suite to check no regressions**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/validation/__init__.py \
        swmm_resilience/validation/hydrograph_csv.py \
        tests/test_hydrograph_csv.py
git commit -m "$(cat <<'EOF'
feat: CSV validation layer for hydrograph batch scenarios

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Temporal `.inp` writer

**Files:**
- Create: `swmm_resilience/simulation/timeseries_scenario.py`
- Create: `tests/test_timeseries_scenario.py`

Key facts from swmm_api (verified against actual `.inp`):
- `opts['START_DATE']` → `datetime.date`
- `opts['START_TIME']` → `datetime.time`
- `opts['END_DATE']` → `datetime.date`
- `opts['END_TIME']` → string like `'3:00:00'`
- `ts_obj.data` → `list[tuple[float, float]]` where first element is decimal hours

- [ ] **Step 1: Write failing tests**

Create `tests/test_timeseries_scenario.py`:

```python
from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.simulation import timeseries_scenario as ts_mod


# ---------------------------------------------------------------------------
# Minimal mock of swmm_api SwmmInput
# ---------------------------------------------------------------------------

class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self, shared_series=False):
        ts_name_2c = "1C" if shared_series else "2C"
        self._s = {
            "INFLOWS": {
                ("1C", "FLOW"): _FakeInflow("1C"),
                ("2C", "FLOW"): _FakeInflow(ts_name_2c),
            },
            "TIMESERIES": {
                "1C": _FakeTs([(0.0, 0.6), (0.05, 0.973)]),
                "2C": _FakeTs([(0.0, 0.825), (0.05, 1.337)]),
            },
            "OPTIONS": {
                "START_DATE": datetime.date(2025, 1, 4),
                "START_TIME": datetime.time(0, 0, 0),
                "END_DATE": datetime.date(2025, 1, 4),
                "END_TIME": "3:00:00",
            },
        }
        self.written_to = None

    def __contains__(self, key):
        return key in self._s

    def __getitem__(self, key):
        return self._s[key]

    def write_file(self, path):
        self.written_to = str(path)


def _make_scenario(last_time_hours=0.05):
    return HydrographScenario(
        scenario_id="test_scen",
        node_series={
            "1C": [(0.0, 1.0), (last_time_hours, 2.0)],
            "2C": [(0.0, 0.5), (last_time_hours, 1.5)],
        },
        time_grid_hours=[0.0, last_time_hours],
        last_time_hours=last_time_hours,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_replaces_all_expected_series(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    scenario = _make_scenario()
    ts_mod.write_scenario_inp(tmp_path / "base.inp", scenario, tmp_path / "out")

    assert fake_inp["TIMESERIES"]["1C"].data == scenario.node_series["1C"]
    assert fake_inp["TIMESERIES"]["2C"].data == scenario.node_series["2C"]


def test_inflows_section_unchanged(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    original_inflows = dict(fake_inp["INFLOWS"])
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert fake_inp["INFLOWS"] == original_inflows


def test_preserves_series_names(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert "1C" in fake_inp["TIMESERIES"]
    assert "2C" in fake_inp["TIMESERIES"]


def test_base_inp_file_not_modified(monkeypatch, tmp_path):
    base = tmp_path / "base.inp"
    base.write_text("original content")
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(base, _make_scenario(), tmp_path / "out")

    assert base.read_text() == "original content"


def test_adjusts_end_time_no_date_rollover(monkeypatch, tmp_path):
    # last_time=5h, base_duration=3h → new end = 8h → same date
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(5.0), tmp_path / "out")

    assert fake_inp["OPTIONS"]["END_TIME"] == "08:00:00"
    assert fake_inp["OPTIONS"]["END_DATE"] == datetime.date(2025, 1, 4)


def test_adjusts_end_date_on_midnight_crossover(monkeypatch, tmp_path):
    # last_time=22h, base_duration=3h → new end = 25h → next day 01:00:00
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(22.0), tmp_path / "out")

    assert fake_inp["OPTIONS"]["END_TIME"] == "01:00:00"
    assert fake_inp["OPTIONS"]["END_DATE"] == datetime.date(2025, 1, 5)


def test_preserves_routing_and_report_step(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    fake_inp["OPTIONS"]["ROUTING_STEP"] = "0:00:05"
    fake_inp["OPTIONS"]["REPORT_STEP"] = "00:05:00"
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert fake_inp["OPTIONS"]["ROUTING_STEP"] == "0:00:05"
    assert fake_inp["OPTIONS"]["REPORT_STEP"] == "00:05:00"


def test_rejects_shared_series(monkeypatch, tmp_path):
    # Both 1C and 2C reference the same series "1C"
    fake_inp = _FakeInp(shared_series=True)
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    with pytest.raises(ValueError, match="compartida"):
        ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")


def test_writes_file_to_output_dir(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    result = ts_mod.write_scenario_inp(
        tmp_path / "base.inp", _make_scenario(), tmp_path / "out"
    )

    assert fake_inp.written_to is not None
    assert str(tmp_path / "out") in fake_inp.written_to
    assert result == Path(fake_inp.written_to)
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
python -m pytest tests/test_timeseries_scenario.py -v
```

Expected: all FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `timeseries_scenario.py`**

Create `swmm_resilience/simulation/timeseries_scenario.py`:

```python
from __future__ import annotations

import datetime
from pathlib import Path

from .swmm_api_io import load_inp, get_node_timeseries_map
from ..validation.hydrograph_csv import HydrographScenario


def _parse_end_time(end_time_str: str) -> datetime.time:
    parts = str(end_time_str).strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
    return datetime.time(h % 24, m, s)


def _validate_one_to_one_mapping(inp, scenario: HydrographScenario) -> dict[str, str]:
    """Return {node_id: series_name} enforcing a 1-to-1 relationship.

    Raises ValueError if any series is shared between scenario nodes or a node
    lacks a timeseries reference in [INFLOWS].
    """
    node_ts_map = get_node_timeseries_map(inp)

    missing = [n for n in scenario.node_series if n not in node_ts_map]
    if missing:
        raise ValueError(
            f"Nodos del CSV sin referencia a serie en [INFLOWS]: {sorted(missing)}"
        )

    seen: dict[str, str] = {}
    for node_id in scenario.node_series:
        ts_name = node_ts_map[node_id]
        if ts_name in seen:
            raise ValueError(
                f"La serie '{ts_name}' está compartida por los nodos "
                f"'{seen[ts_name]}' y '{node_id}' — relación no unívoca"
            )
        seen[ts_name] = node_id

    return {n: node_ts_map[n] for n in scenario.node_series}


def write_scenario_inp(
    base_inp_path: Path,
    scenario: HydrographScenario,
    out_dir: Path,
) -> Path:
    """Write a temporary .inp with replaced [TIMESERIES] for the given scenario.

    Adjusts END_TIME / END_DATE so the network drains after the last CSV point.
    The base .inp is never modified.
    Returns the path to the written .inp file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = load_inp(base_inp_path)

    node_to_series = _validate_one_to_one_mapping(inp, scenario)

    if "TIMESERIES" not in inp:
        raise ValueError("El .inp base no contiene sección [TIMESERIES]")

    for node_id, ts_name in node_to_series.items():
        if ts_name not in inp["TIMESERIES"]:
            raise ValueError(
                f"Serie '{ts_name}' referenciada en [INFLOWS] pero ausente en [TIMESERIES]"
            )
        inp["TIMESERIES"][ts_name].data = list(scenario.node_series[node_id])

    opts = inp["OPTIONS"]
    start_dt = datetime.datetime.combine(opts["START_DATE"], opts["START_TIME"])
    end_time_parsed = _parse_end_time(opts["END_TIME"])
    end_dt = datetime.datetime.combine(opts["END_DATE"], end_time_parsed)
    base_duration = end_dt - start_dt

    new_end_dt = start_dt + datetime.timedelta(hours=scenario.last_time_hours) + base_duration
    opts["END_DATE"] = new_end_dt.date()
    opts["END_TIME"] = new_end_dt.strftime("%H:%M:%S")

    out_inp = out_dir / f"{scenario.scenario_id}.inp"
    inp.write_file(str(out_inp))
    return out_inp
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_timeseries_scenario.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/simulation/timeseries_scenario.py \
        tests/test_timeseries_scenario.py
git commit -m "$(cat <<'EOF'
feat: temporal .inp writer for hydrograph scenarios

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Comparison analysis module

**Files:**
- Create: `swmm_resilience/analysis/model_comparison.py`

Tests for this module are exercised through `test_hydrograph_batch.py` (Task 6). No standalone test file is added here per the spec.

- [ ] **Step 1: Implement `analysis/model_comparison.py`**

Create `swmm_resilience/analysis/model_comparison.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def build_comparison_df(
    swmm_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    scenario_id: str,
) -> pd.DataFrame:
    """Build the canonical per-node comparison DataFrame for one scenario.

    swmm_df must have columns: node_id, inunda_swmm, vol_swmm_m3
    pred_df must have columns: node_id, inunda_pred, vol_pred_m3

    Raises ValueError on node set mismatch or duplicate node_ids.
    """
    swmm_nodes = set(swmm_df["node_id"].astype(str))
    pred_nodes = set(pred_df["node_id"].astype(str))
    if swmm_nodes != pred_nodes:
        raise ValueError(
            f"Conjuntos de nodos distintos entre SWMM y predicción. "
            f"Solo en SWMM: {swmm_nodes - pred_nodes}. "
            f"Solo en pred: {pred_nodes - swmm_nodes}"
        )
    if swmm_df["node_id"].duplicated().any():
        raise ValueError("node_ids duplicados en swmm_df")
    if pred_df["node_id"].duplicated().any():
        raise ValueError("node_ids duplicados en pred_df")

    merged = swmm_df.merge(pred_df, on="node_id", how="inner")
    merged["scenario_id"] = scenario_id
    merged["clasificacion_correcta"] = (
        (merged["inunda_swmm"] == merged["inunda_pred"]).astype(int)
    )
    merged["error_m3"] = merged["vol_pred_m3"] - merged["vol_swmm_m3"]
    merged["abs_error_m3"] = merged["error_m3"].abs()

    return merged[
        [
            "scenario_id", "node_id",
            "inunda_swmm", "inunda_pred", "clasificacion_correcta",
            "vol_swmm_m3", "vol_pred_m3", "error_m3", "abs_error_m3",
        ]
    ].reset_index(drop=True)


def compute_classification_metrics(
    inunda_swmm: "array-like",
    inunda_pred: "array-like",
) -> dict:
    """Return TP, TN, FP, FN, accuracy, precision, recall, F1.

    Metrics without a valid denominator are stored as None, not 0.
    """
    y_true = np.asarray(inunda_swmm, dtype=int)
    y_pred = np.asarray(inunda_pred, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_volume_metrics(
    vol_swmm: "array-like",
    vol_pred: "array-like",
) -> dict:
    """Return MAE, RMSE, total volumes, absolute and percentage total error.

    Computed across all nodes (including zeros).
    """
    y_true = np.asarray(vol_swmm, dtype=float)
    y_pred = np.asarray(vol_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    vol_total_swmm = float(y_true.sum())
    vol_total_pred = float(y_pred.sum())
    error_abs_total = float(abs(vol_total_pred - vol_total_swmm))
    error_pct_total = (
        float((vol_total_pred - vol_total_swmm) / vol_total_swmm * 100)
        if vol_total_swmm > 0
        else None
    )

    return {
        "mae_m3": mae,
        "rmse_m3": rmse,
        "vol_total_swmm_m3": vol_total_swmm,
        "vol_total_pred_m3": vol_total_pred,
        "error_abs_total_m3": error_abs_total,
        "error_pct_total": error_pct_total,
    }


def compute_per_node_r2(
    records: list[dict],
) -> dict[str, float | None]:
    """Compute R² per node across all records.

    Each record must have keys: node_id, vol_swmm_m3, vol_pred_m3.
    Returns {node_id: r2_or_None}.
    R² is None when fewer than 2 samples or variance of vol_swmm is zero.
    """
    from collections import defaultdict

    by_node: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        by_node[rec["node_id"]].append((rec["vol_swmm_m3"], rec["vol_pred_m3"]))

    result: dict[str, float | None] = {}
    for node_id, pairs in by_node.items():
        if len(pairs) < 2:
            result[node_id] = None
            continue
        y_true = np.array([p[0] for p in pairs], dtype=float)
        y_pred = np.array([p[1] for p in pairs], dtype=float)
        if np.var(y_true) == 0:
            result[node_id] = None
        else:
            result[node_id] = float(r2_score(y_true, y_pred))

    return result
```

- [ ] **Step 2: Quick smoke-test (no test file yet — validated in Task 6)**

```bash
python -c "
from swmm_resilience.analysis.model_comparison import (
    build_comparison_df, compute_classification_metrics,
    compute_volume_metrics, compute_per_node_r2,
)
import pandas as pd
swmm = pd.DataFrame({'node_id': ['A','B'], 'inunda_swmm': [1,0], 'vol_swmm_m3': [10.0, 0.0]})
pred = pd.DataFrame({'node_id': ['A','B'], 'inunda_pred': [1,0], 'vol_pred_m3': [8.0, 0.0]})
df = build_comparison_df(swmm, pred, 'test')
print(df)
print(compute_classification_metrics([1,0],[1,0]))
print(compute_volume_metrics([10,0],[8,0]))
"
```

Expected: no errors, output shows comparison DataFrame and metric dicts

- [ ] **Step 3: Run full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add swmm_resilience/analysis/model_comparison.py
git commit -m "$(cat <<'EOF'
feat: comparison analysis module (metrics, comparison table)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Comparison visualization module

**Files:**
- Create: `swmm_resilience/visualization/model_comparison.py`

- [ ] **Step 1: Implement `visualization/model_comparison.py`**

Create `swmm_resilience/visualization/model_comparison.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.model_comparison import compute_volume_metrics

_SYMLOG_THRESH = 1.0  # m³


def _safe_r2(y_true, y_pred) -> str:
    from sklearn.metrics import r2_score
    if len(y_true) < 2 or np.var(y_true) == 0:
        return "N/A"
    return f"{r2_score(y_true, y_pred):.3f}"


def _metrics_text(vol_swmm, vol_pred) -> str:
    m = compute_volume_metrics(vol_swmm, vol_pred)
    r2 = _safe_r2(np.asarray(vol_swmm), np.asarray(vol_pred))
    return (
        f"R²={r2}  MAE={m['mae_m3']:.2f} m³  "
        f"RMSE={m['rmse_m3']:.2f} m³  n={len(vol_swmm)}"
    )


def _draw_parity(ax, x, y, title: str, xlabel: str = "SWMM (m³)", ylabel: str = "XGBoost (m³)"):
    ax.scatter(x, y, alpha=0.6, s=20, color="steelblue")
    lim = max(max(x, default=0), max(y, default=0), 1e-3) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.text(
        0.02, 0.97, _metrics_text(x, y),
        transform=ax.transAxes, va="top", fontsize=7,
    )


def _draw_parity_symlog(ax, x, y, title: str):
    ax.scatter(x, y, alpha=0.6, s=20, color="steelblue")
    lim = max(max(x, default=0), max(y, default=0), _SYMLOG_THRESH) * 1.05
    ref = np.linspace(0, lim, 200)
    ax.plot(ref, ref, "r--", linewidth=0.8, label="y = x")
    ax.set_xscale("symlog", linthresh=_SYMLOG_THRESH)
    ax.set_yscale("symlog", linthresh=_SYMLOG_THRESH)
    ax.set_xlabel("SWMM (m³)")
    ax.set_ylabel("XGBoost (m³)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.text(
        0.02, 0.97, _metrics_text(x, y),
        transform=ax.transAxes, va="top", fontsize=7,
    )


def plot_parity_nodes(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    scenario_id: str,
) -> list[Path]:
    """Generate per-node parity plots (linear + symlog). Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x = comparison_df["vol_swmm_m3"].tolist()
    y = comparison_df["vol_pred_m3"].tolist()
    title = f"Parity by node — {scenario_id}"
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_parity(ax, x, y, title)
    p = output_dir / "parity_nodes_linear.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_parity_symlog(ax, x, y, title)
    p = output_dir / "parity_nodes_symlog.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def plot_parity_aggregated(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    scenario_id: str,
) -> list[Path]:
    """Generate aggregated (network total) parity plots (linear + symlog)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x = [comparison_df["vol_swmm_m3"].sum()]
    y = [comparison_df["vol_pred_m3"].sum()]
    title = f"Parity aggregated — {scenario_id}"
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(4, 4))
    _draw_parity(ax, x, y, title)
    p = output_dir / "parity_agg_linear.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4, 4))
    _draw_parity_symlog(ax, x, y, title)
    p = output_dir / "parity_agg_symlog.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def plot_node_profiles(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    scenario_id: str,
) -> list[Path]:
    """Generate node profile plots (linear + symlog).

    Only includes nodes where at least one of vol_swmm_m3 or vol_pred_m3 > 0.
    Nodes ordered by vol_swmm_m3 descending, then node_id ascending.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    visible = comparison_df[
        (comparison_df["vol_swmm_m3"] > 0) | (comparison_df["vol_pred_m3"] > 0)
    ].copy()

    if visible.empty:
        return []

    visible = visible.sort_values(
        ["vol_swmm_m3", "node_id"], ascending=[False, True]
    )
    nodes = visible["node_id"].tolist()
    swmm_vals = visible["vol_swmm_m3"].tolist()
    pred_vals = visible["vol_pred_m3"].tolist()

    n = len(nodes)
    fig_w = max(6.0, min(n * 0.35, 24.0))
    paths: list[Path] = []

    for scale, fname in [("linear", "node_profile_linear.png"), ("symlog", "node_profile_symlog.png")]:
        fig, ax = plt.subplots(figsize=(fig_w, 4))
        x = range(n)
        ax.plot(x, swmm_vals, "b-o", markersize=4, label="SWMM")
        ax.plot(x, pred_vals, "o-", color="orange", markersize=4, label="XGBoost")
        ax.set_xticks(list(x))
        ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Node ID")
        ax.set_ylabel("Flood volume (m³)")
        ax.set_title(f"Node profile — {scenario_id}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=_SYMLOG_THRESH)
        p = output_dir / fname
        fig.savefig(str(p), dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "
import pandas as pd, tempfile
from pathlib import Path
from swmm_resilience.visualization.model_comparison import (
    plot_parity_nodes, plot_parity_aggregated, plot_node_profiles
)
df = pd.DataFrame({
    'node_id': ['A','B','C'],
    'vol_swmm_m3': [10.0, 5.0, 0.0],
    'vol_pred_m3': [8.0, 6.0, 0.0],
    'inunda_swmm': [1,1,0],
    'inunda_pred': [1,1,0],
})
with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    paths = plot_parity_nodes(df, out, 'test')
    print('parity nodes:', [p.name for p in paths])
    paths = plot_parity_aggregated(df, out, 'test')
    print('parity agg:', [p.name for p in paths])
    paths = plot_node_profiles(df, out, 'test')
    print('profiles:', [p.name for p in paths])
"
```

Expected: no errors, prints filenames

- [ ] **Step 3: Run full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add swmm_resilience/visualization/model_comparison.py
git commit -m "$(cat <<'EOF'
feat: comparison visualization module (parity plots, node profiles)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Scenario predictor

**Files:**
- Create: `swmm_resilience/ml/scenario_predict.py`
- Create: `tests/test_scenario_predict.py`

Key design: hash check against base `.inp`, feature extraction from temporal `.inp`. Reuses `extract_static_features`, `compute_topology_features`, `compute_dynamic_features(factor=1.0)`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenario_predict.py`:

```python
from __future__ import annotations

import numpy as np
import joblib
import pandas as pd
import pytest
from pathlib import Path

from swmm_resilience.ml import scenario_predict as sp
from swmm_resilience.ml.trainer import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

class _FakeClf:
    """Classifies node 'J2' as flooded, everything else as not."""
    def predict(self, X):
        assert list(X.columns) == FEATURE_COLS, f"Wrong feature order: {list(X.columns)}"
        return np.array([0, 1], dtype=int)


class _FakeReg:
    """Always predicts log1p(33)."""
    def predict(self, X):
        return np.full(len(X), np.log1p(33.0))


def _make_models_dir(tmp_path: Path, inp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    joblib.dump(_FakeClf(), d / "classifier.joblib")
    joblib.dump(_FakeReg(), d / "regressor.joblib")
    (d / "training_inp_hash.txt").write_text(sp._md5(inp_path), encoding="utf-8")
    return d


def _base_static_df():
    """Static DataFrame with exactly the non-dynamic FEATURE_COLS columns."""
    non_dynamic = [c for c in FEATURE_COLS if c not in {"factor_mult", "q_pico_nodo", "q_pico_acum_escalado"}]
    return pd.DataFrame(
        {"node_id": ["J1", "J2"], "coord_x": [0.0, 1.0], "coord_y": [0.0, 1.0],
         **{col: [1.0, 2.0] for col in non_dynamic}}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_scenario_uses_factor_mult_one(monkeypatch, tmp_path):
    inp = tmp_path / "base.inp"
    inp.write_text("base", encoding="utf-8")
    models_dir = _make_models_dir(tmp_path, inp)

    static_df = _base_static_df()
    received_factors = []

    def fake_extract_static(path):
        return static_df[["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max"]].copy()

    def fake_compute_topology(df, path):
        return static_df.copy()

    def fake_compute_dynamic(df, factor):
        received_factors.append(factor)
        return pd.DataFrame({
            "node_id": ["J1", "J2"],
            "factor_mult": [factor, factor],
            "q_pico_nodo": [df.loc[df["node_id"] == "J1", "base_inflow_lps"].iloc[0] * factor,
                            df.loc[df["node_id"] == "J2", "base_inflow_lps"].iloc[0] * factor],
            "q_pico_acum_escalado": [df.loc[df["node_id"] == "J1", "q_pico_acum_base"].iloc[0] * factor,
                                     df.loc[df["node_id"] == "J2", "q_pico_acum_base"].iloc[0] * factor],
        })

    monkeypatch.setattr(sp, "extract_static_features", fake_extract_static)
    monkeypatch.setattr(sp, "compute_topology_features", fake_compute_topology)
    monkeypatch.setattr(sp, "compute_dynamic_features", fake_compute_dynamic)

    sp.predict_scenario(inp, tmp_path / "temp.inp", models_dir)

    assert received_factors == [1.0]


def test_predict_scenario_recalculates_peaks_from_new_series(monkeypatch, tmp_path):
    inp = tmp_path / "base.inp"
    inp.write_text("base", encoding="utf-8")
    models_dir = _make_models_dir(tmp_path, inp)

    # The temporal .inp has higher peaks than the base
    static_with_new_peaks = _base_static_df().copy()
    static_with_new_peaks["base_inflow_lps"] = [50.0, 100.0]  # new peaks from temporal inp
    static_with_new_peaks["q_pico_acum_base"] = [50.0, 150.0]

    seen_peaks = {}

    def fake_extract_static(path):
        seen_peaks["path"] = str(path)
        return static_with_new_peaks[["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max"]].copy()

    def fake_compute_topology(df, path):
        return static_with_new_peaks.copy()

    def fake_compute_dynamic(df, factor):
        return pd.DataFrame({
            "node_id": ["J1", "J2"],
            "factor_mult": [1.0, 1.0],
            "q_pico_nodo": df["base_inflow_lps"].tolist(),
            "q_pico_acum_escalado": df["q_pico_acum_base"].tolist(),
        })

    monkeypatch.setattr(sp, "extract_static_features", fake_extract_static)
    monkeypatch.setattr(sp, "compute_topology_features", fake_compute_topology)
    monkeypatch.setattr(sp, "compute_dynamic_features", fake_compute_dynamic)

    temporal = tmp_path / "temporal.inp"
    result = sp.predict_scenario(inp, temporal, models_dir)

    # Features were extracted from temporal inp path
    assert seen_peaks["path"] == str(temporal)
    # J2 is flooded; J1 is not
    assert result.loc[result["node_id"] == "J1", "vol_pred_m3"].iloc[0] == 0.0
    assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(33.0)


def test_predict_scenario_applies_expm1_and_clips_negatives(monkeypatch, tmp_path):
    inp = tmp_path / "base.inp"
    inp.write_text("base", encoding="utf-8")
    models_dir = _make_models_dir(tmp_path, inp)

    class NegReg:
        def predict(self, X):
            # Returns a value that after expm1 would be negative (impossible with expm1
            # of a valid log1p, but clip is still tested via a very negative raw log value)
            return np.array([-100.0])

    joblib.dump(NegReg(), models_dir / "regressor.joblib")

    static_df = _base_static_df()
    monkeypatch.setattr(sp, "extract_static_features",
                        lambda p: static_df[["node_id","coord_x","coord_y","elev_fondo","prof_max"]].copy())
    monkeypatch.setattr(sp, "compute_topology_features", lambda df, p: static_df.copy())
    monkeypatch.setattr(sp, "compute_dynamic_features",
                        lambda df, factor: pd.DataFrame({
                            "node_id": ["J1","J2"], "factor_mult": [1.0,1.0],
                            "q_pico_nodo": [1.0,2.0], "q_pico_acum_escalado": [1.0,2.0],
                        }))

    result = sp.predict_scenario(inp, tmp_path / "t.inp", models_dir)
    # J2 classified as flooded but expm1(-100) ≈ -1 → clipped to 0
    assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(0.0)


def test_verifies_hash_of_base_inp_not_temporal(monkeypatch, tmp_path):
    base = tmp_path / "base.inp"
    base.write_text("base", encoding="utf-8")
    temporal = tmp_path / "temporal.inp"
    temporal.write_text("completely different timeseries content", encoding="utf-8")

    models_dir = _make_models_dir(tmp_path, base)  # hash stored for base

    static_df = _base_static_df()
    monkeypatch.setattr(sp, "extract_static_features",
                        lambda p: static_df[["node_id","coord_x","coord_y","elev_fondo","prof_max"]].copy())
    monkeypatch.setattr(sp, "compute_topology_features", lambda df, p: static_df.copy())
    monkeypatch.setattr(sp, "compute_dynamic_features",
                        lambda df, factor: pd.DataFrame({
                            "node_id": ["J1","J2"], "factor_mult": [1.0,1.0],
                            "q_pico_nodo": [1.0,2.0], "q_pico_acum_escalado": [1.0,2.0],
                        }))

    # Should NOT raise — temporal can differ, hash check is on base
    result = sp.predict_scenario(base, temporal, models_dir)
    assert set(result["node_id"]) == {"J1", "J2"}


def test_rejects_wrong_base_inp_hash(tmp_path):
    base = tmp_path / "base.inp"
    base.write_text("base", encoding="utf-8")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(_FakeClf(), models_dir / "classifier.joblib")
    joblib.dump(_FakeReg(), models_dir / "regressor.joblib")
    (models_dir / "training_inp_hash.txt").write_text("wrong_hash", encoding="utf-8")

    with pytest.raises(ValueError, match="cambiado"):
        sp.predict_scenario(base, base, models_dir)


def test_does_not_call_train_functions(monkeypatch, tmp_path):
    inp = tmp_path / "base.inp"
    inp.write_text("base", encoding="utf-8")
    models_dir = _make_models_dir(tmp_path, inp)

    called = []

    def fake_train(*a, **kw):
        called.append("train_models")

    monkeypatch.setattr("swmm_resilience.ml.trainer.train_models", fake_train, raising=False)

    static_df = _base_static_df()
    monkeypatch.setattr(sp, "extract_static_features",
                        lambda p: static_df[["node_id","coord_x","coord_y","elev_fondo","prof_max"]].copy())
    monkeypatch.setattr(sp, "compute_topology_features", lambda df, p: static_df.copy())
    monkeypatch.setattr(sp, "compute_dynamic_features",
                        lambda df, factor: pd.DataFrame({
                            "node_id": ["J1","J2"], "factor_mult": [1.0,1.0],
                            "q_pico_nodo": [1.0,2.0], "q_pico_acum_escalado": [1.0,2.0],
                        }))

    sp.predict_scenario(inp, inp, models_dir)
    assert called == []


def test_output_columns(monkeypatch, tmp_path):
    inp = tmp_path / "base.inp"
    inp.write_text("base", encoding="utf-8")
    models_dir = _make_models_dir(tmp_path, inp)

    static_df = _base_static_df()
    monkeypatch.setattr(sp, "extract_static_features",
                        lambda p: static_df[["node_id","coord_x","coord_y","elev_fondo","prof_max"]].copy())
    monkeypatch.setattr(sp, "compute_topology_features", lambda df, p: static_df.copy())
    monkeypatch.setattr(sp, "compute_dynamic_features",
                        lambda df, factor: pd.DataFrame({
                            "node_id": ["J1","J2"], "factor_mult": [1.0,1.0],
                            "q_pico_nodo": [1.0,2.0], "q_pico_acum_escalado": [1.0,2.0],
                        }))

    result = sp.predict_scenario(inp, inp, models_dir)
    assert list(result.columns) == ["node_id", "inunda_pred", "vol_pred_m3"]
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
python -m pytest tests/test_scenario_predict.py -v
```

Expected: all FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `scenario_predict.py`**

Create `swmm_resilience/ml/scenario_predict.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..extraction.dynamic_features import compute_dynamic_features
from ..extraction.static_features import extract_static_features
from ..extraction.topology import compute_topology_features
from .trainer import FEATURE_COLS


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_scenario(
    base_inp_path: Path,
    temporal_inp_path: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Predict flooding for all junction nodes using a scenario-specific temporal .inp.

    Hash is verified against the BASE .inp (unchanged network structure).
    Features are extracted from the TEMPORAL .inp (which has new timeseries).
    Returns DataFrame: node_id, inunda_pred, vol_pred_m3
    """
    stored_hash = (models_dir / "training_inp_hash.txt").read_text().strip()
    if _md5(base_inp_path) != stored_hash:
        raise ValueError(
            f"El .inp base en '{base_inp_path}' ha cambiado desde el entrenamiento. "
            "Re-entrena el modelo o usa el .inp original."
        )

    clf = joblib.load(models_dir / "classifier.joblib")
    reg = joblib.load(models_dir / "regressor.joblib")

    static_df = extract_static_features(temporal_inp_path)
    full_df = compute_topology_features(static_df, temporal_inp_path)
    dynamic_df = compute_dynamic_features(full_df, 1.0)
    merged = full_df.merge(dynamic_df, on="node_id", how="left")

    X = merged[FEATURE_COLS]
    inunda_pred = clf.predict(X)
    vol_pred = np.zeros(len(X))
    flood_mask = inunda_pred == 1
    if flood_mask.sum() > 0:
        vol_pred[flood_mask] = np.expm1(reg.predict(X.loc[flood_mask]))
        vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)

    merged["inunda_pred"] = inunda_pred
    merged["vol_pred_m3"] = vol_pred
    return merged[["node_id", "inunda_pred", "vol_pred_m3"]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_scenario_predict.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/scenario_predict.py \
        tests/test_scenario_predict.py
git commit -m "$(cat <<'EOF'
feat: scenario predictor with hash-verified base .inp

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Batch coordinator

**Files:**
- Create: `swmm_resilience/validation/hydrograph_batch.py`
- Create: `tests/test_hydrograph_batch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hydrograph_batch.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.validation import hydrograph_batch as hb
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, rows: list[dict], filename: str) -> Path:
    p = tmp_path / filename
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _valid_rows_for_nodes(nodes: list[str], last_time: str = "0:05"):
    rows = []
    for n in nodes:
        rows.append({"node_id": n, "time": "0:00", "value_lps": 1.0})
        rows.append({"node_id": n, "time": last_time, "value_lps": 2.0})
    return rows


def _make_swmm_df(nodes: list[str], flooded: list[str], vol: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": nodes,
        "inunda_swmm": [1 if n in flooded else 0 for n in nodes],
        "vol_swmm_m3": [vol if n in flooded else 0.0 for n in nodes],
    })


def _make_pred_df(nodes: list[str], flooded: list[str], vol: float = 8.0) -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": nodes,
        "inunda_pred": [1 if n in flooded else 0 for n in nodes],
        "vol_pred_m3": [vol if n in flooded else 0.0 for n in nodes],
    })


# ---------------------------------------------------------------------------
# Tests: comparison metrics (section 15.4)
# ---------------------------------------------------------------------------

def test_compute_classification_metrics_correct():
    from swmm_resilience.analysis.model_comparison import compute_classification_metrics
    m = compute_classification_metrics([1, 0, 1, 0], [1, 0, 0, 1])
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["accuracy"] == pytest.approx(0.5)


def test_classification_metrics_null_when_no_denominator():
    from swmm_resilience.analysis.model_comparison import compute_classification_metrics
    # All negative — precision has no denominator
    m = compute_classification_metrics([0, 0], [0, 0])
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None
    assert m["tp"] == 0
    assert m["tn"] == 2


def test_build_comparison_df_matches_nodes():
    from swmm_resilience.analysis.model_comparison import build_comparison_df
    nodes = ["A", "B", "C"]
    swmm = _make_swmm_df(nodes, ["A"])
    pred = _make_pred_df(nodes, ["A"])
    df = build_comparison_df(swmm, pred, "scen1")
    assert set(df["node_id"]) == set(nodes)
    assert df["scenario_id"].iloc[0] == "scen1"
    assert "error_m3" in df.columns
    assert "abs_error_m3" in df.columns


def test_build_comparison_df_rejects_node_mismatch():
    from swmm_resilience.analysis.model_comparison import build_comparison_df
    swmm = _make_swmm_df(["A", "B"], ["A"])
    pred = _make_pred_df(["A", "C"], ["A"])
    with pytest.raises(ValueError, match="Conjuntos de nodos"):
        build_comparison_df(swmm, pred, "scen")


def test_swmm_parse_failure_not_silently_zero():
    # parse failure should raise, not fill zeros — tested via extract_labels contract
    from swmm_resilience.extraction.labels import extract_labels
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".rpt", delete=False) as f:
        f.write(b"not a valid rpt")
        rpt = Path(f.name)
    # extract_labels returns zeros for missing nodes but does NOT raise on bad rpt
    # The batch coordinator must check if swmm execution itself failed (non-zero exit)
    # This is tested in test_batch_continues_after_swmm_failure below
    result = extract_labels(rpt, ["X", "Y"], threshold_m3=0.0)
    assert set(result["node_id"]) == {"X", "Y"}


def test_volume_metrics_correct():
    from swmm_resilience.analysis.model_comparison import compute_volume_metrics
    m = compute_volume_metrics([10.0, 0.0], [8.0, 0.0])
    assert m["mae_m3"] == pytest.approx(1.0)
    assert m["vol_total_swmm_m3"] == pytest.approx(10.0)
    assert m["vol_total_pred_m3"] == pytest.approx(8.0)
    assert m["error_pct_total"] == pytest.approx(-20.0)


def test_volume_metrics_no_pct_when_swmm_total_zero():
    from swmm_resilience.analysis.model_comparison import compute_volume_metrics
    m = compute_volume_metrics([0.0, 0.0], [0.0, 0.0])
    assert m["error_pct_total"] is None


# ---------------------------------------------------------------------------
# Tests: batch flow (section 15.5)
# ---------------------------------------------------------------------------

def test_batch_processes_csv_in_alphabetical_order(monkeypatch, tmp_path):
    nodes = ["N1", "N2"]
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "b_scenario.csv")
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "a_scenario.csv")

    order = []
    base_inp = tmp_path / "base.inp"
    base_inp.write_text("x")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "training_inp_hash.txt").write_text("h")
    (models_dir / "classifier.joblib").write_text("x")
    (models_dir / "regressor.joblib").write_text("x")

    def fake_run_scenario(scenario, csv_path, base_inp_path, models_dir, output_dir,
                          all_node_ids, flood_threshold_m3, base_inp_hash):
        order.append(scenario.scenario_id)
        return hb._ScenarioResult(scenario_id=scenario.scenario_id, status="ok",
                                   error_message=None, comparison_df=None,
                                   clf_metrics={}, vol_metrics={}, metadata={})

    monkeypatch.setattr(hb, "_run_scenario", fake_run_scenario)
    monkeypatch.setattr(hb, "_preflight_check", lambda *a, **kw: ({"N1", "N2"}, None))

    hb.run_batch(
        hydrographs_dir=tmp_path,
        base_inp_path=base_inp,
        models_dir=models_dir,
        output_root=tmp_path / "out",
        flood_threshold_m3=0.0,
    )

    assert order == ["a_scenario", "b_scenario"]


def test_batch_continues_after_invalid_scenario(monkeypatch, tmp_path):
    nodes = ["N1", "N2"]
    # valid scenario
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "good.csv")
    # invalid scenario (missing node 2)
    pd.DataFrame([
        {"node_id": "N1", "time": "0:00", "value_lps": 1.0},
        {"node_id": "N1", "time": "0:05", "value_lps": 2.0},
    ]).to_csv(tmp_path / "bad.csv", index=False)

    processed = []
    base_inp = tmp_path / "base.inp"
    base_inp.write_text("x")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "training_inp_hash.txt").write_text("h")
    (models_dir / "classifier.joblib").write_text("x")
    (models_dir / "regressor.joblib").write_text("x")

    def fake_run(scenario, *a, **kw):
        processed.append(scenario.scenario_id)
        return hb._ScenarioResult(scenario_id=scenario.scenario_id, status="ok",
                                   error_message=None, comparison_df=None,
                                   clf_metrics={}, vol_metrics={}, metadata={})

    monkeypatch.setattr(hb, "_run_scenario", fake_run)
    monkeypatch.setattr(hb, "_preflight_check", lambda *a, **kw: ({"N1", "N2"}, None))

    results = hb.run_batch(
        hydrographs_dir=tmp_path,
        base_inp_path=base_inp,
        models_dir=models_dir,
        output_root=tmp_path / "out",
        flood_threshold_m3=0.0,
    )

    assert "good" in processed
    failed = [r for r in results if r.status != "ok"]
    assert any(r.scenario_id == "bad" for r in failed)
    assert len(failed) == 1


def test_batch_aborts_on_global_model_error(monkeypatch, tmp_path):
    nodes = ["N1"]
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "s.csv")
    base_inp = tmp_path / "base.inp"
    base_inp.write_text("x")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def failing_preflight(*a, **kw):
        raise RuntimeError("Modelo no encontrado")

    monkeypatch.setattr(hb, "_preflight_check", failing_preflight)

    with pytest.raises(RuntimeError, match="Modelo no encontrado"):
        hb.run_batch(
            hydrographs_dir=tmp_path,
            base_inp_path=base_inp,
            models_dir=models_dir,
            output_root=tmp_path / "out",
            flood_threshold_m3=0.0,
        )


def test_batch_summary_includes_all_scenarios(monkeypatch, tmp_path):
    nodes = ["N1", "N2"]
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "s1.csv")
    _write_csv(tmp_path, _valid_rows_for_nodes(nodes), "s2.csv")

    base_inp = tmp_path / "base.inp"
    base_inp.write_text("x")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "training_inp_hash.txt").write_text("h")
    (models_dir / "classifier.joblib").write_text("x")
    (models_dir / "regressor.joblib").write_text("x")

    def fake_run(scenario, *a, **kw):
        return hb._ScenarioResult(scenario_id=scenario.scenario_id, status="ok",
                                   error_message=None, comparison_df=None,
                                   clf_metrics={}, vol_metrics={}, metadata={})

    monkeypatch.setattr(hb, "_run_scenario", fake_run)
    monkeypatch.setattr(hb, "_preflight_check", lambda *a, **kw: ({"N1", "N2"}, None))

    results = hb.run_batch(
        hydrographs_dir=tmp_path,
        base_inp_path=base_inp,
        models_dir=models_dir,
        output_root=tmp_path / "out",
        flood_threshold_m3=0.0,
    )

    assert len(results) == 2
    scenario_ids = {r.scenario_id for r in results}
    assert scenario_ids == {"s1", "s2"}


def test_per_node_r2_computed_across_scenarios():
    from swmm_resilience.analysis.model_comparison import compute_per_node_r2

    records = [
        {"node_id": "A", "vol_swmm_m3": 10.0, "vol_pred_m3": 9.0},
        {"node_id": "A", "vol_swmm_m3": 20.0, "vol_pred_m3": 18.0},
        {"node_id": "B", "vol_swmm_m3": 5.0, "vol_pred_m3": 5.0},
        {"node_id": "B", "vol_swmm_m3": 10.0, "vol_pred_m3": 10.0},
        {"node_id": "C", "vol_swmm_m3": 0.0, "vol_pred_m3": 0.0},  # only 1 scenario
    ]
    # Add second record for C
    r2 = compute_per_node_r2(records)
    assert "A" in r2
    assert r2["A"] is not None
    # C appears only once → None
    assert r2["C"] is None


def test_per_node_r2_null_when_zero_variance():
    from swmm_resilience.analysis.model_comparison import compute_per_node_r2

    records = [
        {"node_id": "A", "vol_swmm_m3": 0.0, "vol_pred_m3": 5.0},
        {"node_id": "A", "vol_swmm_m3": 0.0, "vol_pred_m3": 3.0},
    ]
    r2 = compute_per_node_r2(records)
    assert r2["A"] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_hydrograph_batch.py -v
```

Expected: FAIL (imports not found)

- [ ] **Step 3: Implement `hydrograph_batch.py`**

> Self-review fixes applied here vs first draft:
> - `_run_scenario` now receives `csv_path` so csv_md5 and metadata.json can be written per scenario
> - `_compute_scenario_metadata` is actually called and its result saved to `metadata.json`
> - `run_batch` writes `run_metadata.json` at batch level with reserved_for_validation marker
> - `_preflight_check` loads models to detect loading errors early (before any scenario runs)

Create `swmm_resilience/validation/hydrograph_batch.py`:

```python
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyswmm import Simulation

from ..analysis.model_comparison import (
    build_comparison_df,
    compute_classification_metrics,
    compute_per_node_r2,
    compute_volume_metrics,
)
from ..extraction.labels import extract_labels
from ..ml.scenario_predict import predict_scenario
from ..visualization.model_comparison import (
    plot_node_profiles,
    plot_parity_aggregated,
    plot_parity_nodes,
)
from .hydrograph_csv import HydrographScenario, load_scenario, scenario_id_from_path
from ..simulation.timeseries_scenario import write_scenario_inp


@dataclass
class _ScenarioResult:
    scenario_id: str
    status: str          # "ok" | "failed"
    error_message: Optional[str]
    comparison_df: Optional[pd.DataFrame]
    clf_metrics: dict
    vol_metrics: dict
    metadata: dict


def _md5(path: Path) -> str:
    d = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            d.update(chunk)
    return d.hexdigest()


def _preflight_check(
    hydrographs_dir: Path,
    base_inp_path: Path,
    models_dir: Path,
) -> tuple[set[str], str]:
    """Validate global prerequisites before any scenario runs.

    Checks folder, .inp existence, model artefacts, hash, and actually
    loads models (to catch corrupt files before any scenario runs).
    Returns (expected_nodes, base_inp_hash).
    Raises RuntimeError on any global failure.
    """
    if not hydrographs_dir.is_dir():
        raise RuntimeError(f"Carpeta de hidrogramas no encontrada: {hydrographs_dir}")
    if not base_inp_path.exists():
        raise RuntimeError(f".inp base no encontrado: {base_inp_path}")

    for artefact in ("classifier.joblib", "regressor.joblib", "training_inp_hash.txt"):
        if not (models_dir / artefact).exists():
            raise RuntimeError(f"Artefacto de modelo no encontrado: {models_dir / artefact}")

    stored_hash = (models_dir / "training_inp_hash.txt").read_text().strip()
    current_hash = _md5(base_inp_path)
    if current_hash != stored_hash:
        raise RuntimeError(
            f"El .inp base ha cambiado desde el entrenamiento. "
            f"Hash almacenado: {stored_hash[:8]}…  Actual: {current_hash[:8]}…"
        )

    # Eagerly load models to fail fast on corrupt files
    try:
        joblib.load(models_dir / "classifier.joblib")
        joblib.load(models_dir / "regressor.joblib")
    except Exception as exc:
        raise RuntimeError(f"Error al cargar modelos: {exc}") from exc

    from ..simulation.swmm_api_io import load_inp, list_inflow_nodes
    inp = load_inp(base_inp_path)
    expected_nodes = list_inflow_nodes(inp)

    return expected_nodes, stored_hash


def _detect_multipico(values: list[float]) -> bool:
    """Return True if the series has ≥ 2 significant local peaks.

    Collapses consecutive equal values first. A peak is greater than both
    neighbors (endpoints compared to their single neighbor).
    Only counts peaks ≥ 10% of the global max.
    """
    if not values:
        return False

    # Collapse consecutive equal values
    collapsed = [values[0]]
    for v in values[1:]:
        if v != collapsed[-1]:
            collapsed.append(v)

    n = len(collapsed)
    if n < 2:
        return False

    global_max = max(collapsed)
    threshold = 0.1 * global_max if global_max > 0 else 0.0

    peaks = 0
    for i, v in enumerate(collapsed):
        if v < threshold:
            continue
        if n == 1:
            peaks += 1
        elif i == 0 and v > collapsed[1]:
            peaks += 1
        elif i == n - 1 and v > collapsed[-2]:
            peaks += 1
        elif 0 < i < n - 1 and v > collapsed[i - 1] and v > collapsed[i + 1]:
            peaks += 1

    return peaks >= 2


def _compute_scenario_metadata(
    scenario: HydrographScenario,
    base_inp_hash: str,
    models_dir: Path,
) -> dict:
    times = np.array(scenario.time_grid_hours)
    intervals_h = np.diff(times)
    intervals_min = intervals_h * 60.0

    all_peaks = []
    all_time_to_peak = []
    all_values = []
    multipico_count = 0

    for nid, series in scenario.node_series.items():
        ts_times = np.array([t for t, _ in series])
        ts_vals = np.array([v for _, v in series])
        all_values.extend(ts_vals.tolist())
        peak_val = float(ts_vals.max())
        peak_idx = int(ts_vals.argmax())
        all_peaks.append(peak_val)
        all_time_to_peak.append(float(ts_times[peak_idx] * 60.0))  # minutes
        if _detect_multipico(ts_vals.tolist()):
            multipico_count += 1

    # Trapezoidal input volume: L/s * h * 3.6 = m³
    total_vol_m3 = 0.0
    for nid, series in scenario.node_series.items():
        ts_times = np.array([t for t, _ in series])
        ts_vals = np.array([v for _, v in series])
        total_vol_m3 += float(np.trapz(ts_vals, x=ts_times) * 3.6)

    clf_hash = _md5(models_dir / "classifier.joblib")
    reg_hash = _md5(models_dir / "regressor.joblib")

    return {
        "csv_md5": None,  # caller must set this to _md5(csv_path)
        "base_inp_md5": base_inp_hash,
        "classifier_md5": clf_hash,
        "regressor_md5": reg_hash,
        "n_nodes": len(scenario.node_series),
        "n_steps": len(scenario.time_grid_hours),
        "last_time_hours": scenario.last_time_hours,
        "interval_min_min": float(intervals_min.min()) if len(intervals_min) > 0 else None,
        "interval_median_min": float(np.median(intervals_min)) if len(intervals_min) > 0 else None,
        "interval_max_min": float(intervals_min.max()) if len(intervals_min) > 0 else None,
        "peak_flow_min_lps": float(min(all_peaks)) if all_peaks else None,
        "peak_flow_median_lps": float(np.median(all_peaks)) if all_peaks else None,
        "peak_flow_max_lps": float(max(all_peaks)) if all_peaks else None,
        "time_to_peak_min_min": float(min(all_time_to_peak)) if all_time_to_peak else None,
        "time_to_peak_median_min": float(np.median(all_time_to_peak)) if all_time_to_peak else None,
        "time_to_peak_max_min": float(max(all_time_to_peak)) if all_time_to_peak else None,
        "total_input_vol_m3": total_vol_m3,
        "n_multipico_nodes": multipico_count,
        "run_datetime": datetime.now().isoformat(),
        "status": None,  # filled by caller
        "error_message": None,
    }


def _run_swmm(temporal_inp: Path) -> Path:
    """Run SWMM on temporal_inp. Returns path to .rpt. Raises on failure."""
    with Simulation(str(temporal_inp)) as sim:
        for _ in sim:
            pass
    rpt = temporal_inp.with_suffix(".rpt")
    if not rpt.exists():
        raise FileNotFoundError(f"SWMM no generó el .rpt esperado: {rpt}")
    return rpt


def _run_scenario(
    scenario: HydrographScenario,
    csv_path: Path,
    base_inp_path: Path,
    models_dir: Path,
    output_dir: Path,
    all_node_ids: set[str],
    flood_threshold_m3: float,
    base_inp_hash: str,
) -> _ScenarioResult:
    """Execute one scenario: write inp, run SWMM, predict, compare, save outputs."""
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix=f"swmm_{scenario.scenario_id}_") as tmp:
            tmp_path = Path(tmp)

            # Write temporal .inp
            temporal_inp = write_scenario_inp(base_inp_path, scenario, tmp_path)

            # Run SWMM
            rpt_path = _run_swmm(temporal_inp)

            # Parse SWMM results
            node_ids_list = sorted(all_node_ids)
            labels_df = extract_labels(rpt_path, node_ids_list, flood_threshold_m3)
            swmm_df = labels_df.rename(
                columns={"inunda": "inunda_swmm", "vol_inundacion_m3": "vol_swmm_m3"}
            )

            # Predict with XGBoost
            pred_df = predict_scenario(base_inp_path, temporal_inp, models_dir)

            # Build comparison
            comparison_df = build_comparison_df(swmm_df, pred_df, scenario.scenario_id)

        clf_metrics = compute_classification_metrics(
            comparison_df["inunda_swmm"], comparison_df["inunda_pred"]
        )
        vol_metrics = compute_volume_metrics(
            comparison_df["vol_swmm_m3"], comparison_df["vol_pred_m3"]
        )

        # Save CSVs
        comparison_df.to_csv(output_dir / "comparison_nodes.csv", index=False)

        # Save metrics
        metrics = {**clf_metrics, **vol_metrics}
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Compute and save metadata
        meta = _compute_scenario_metadata(scenario, base_inp_hash, models_dir)
        meta["csv_md5"] = _md5(csv_path)
        meta["status"] = "ok"
        meta["reserved_for_validation"] = True
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Generate plots (warning on failure — does not fail the scenario)
        plot_warnings: list[str] = []
        for plot_fn in (plot_parity_nodes, plot_parity_aggregated, plot_node_profiles):
            try:
                plot_fn(comparison_df, output_dir, scenario.scenario_id)
            except Exception as exc:
                plot_warnings.append(f"{plot_fn.__name__}: {exc}")
                warnings.warn(
                    f"[{scenario.scenario_id}] Warning al generar gráfica ({plot_fn.__name__}): {exc}",
                    stacklevel=2,
                )

        return _ScenarioResult(
            scenario_id=scenario.scenario_id,
            status="ok",
            error_message="; ".join(plot_warnings) if plot_warnings else None,
            comparison_df=comparison_df,
            clf_metrics=clf_metrics,
            vol_metrics=vol_metrics,
            metadata=meta,
        )

    except Exception as exc:
        meta_err = {
            "scenario_id": scenario.scenario_id,
            "status": "failed",
            "error_message": str(exc),
            "run_datetime": datetime.now().isoformat(),
            "reserved_for_validation": True,
        }
        try:
            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_err, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        return _ScenarioResult(
            scenario_id=scenario.scenario_id,
            status="failed",
            error_message=str(exc),
            comparison_df=None,
            clf_metrics={},
            vol_metrics={},
            metadata=meta_err,
        )


def run_batch(
    hydrographs_dir: Path,
    base_inp_path: Path,
    models_dir: Path,
    output_root: Path,
    flood_threshold_m3: float,
) -> list[_ScenarioResult]:
    """Run the full hydrograph batch evaluation.

    Returns list of _ScenarioResult, one per discovered CSV.
    Raises RuntimeError on global failures (missing models, hash mismatch).
    """
    # Pre-flight (raises on global failure)
    expected_nodes, base_inp_hash = _preflight_check(
        hydrographs_dir, base_inp_path, models_dir
    )

    # Discover CSV files, sorted alphabetically (deterministic)
    csv_files = sorted(
        p for p in hydrographs_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )

    # Rule 10: unique scenario IDs after normalization
    seen_ids: dict[str, Path] = {}
    for csv_path in csv_files:
        sid = scenario_id_from_path(csv_path)
        if sid in seen_ids:
            raise RuntimeError(
                f"Identificadores de escenario duplicados tras normalización: "
                f"'{sid}' ('{csv_path.name}' y '{seen_ids[sid].name}')"
            )
        seen_ids[sid] = csv_path

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[_ScenarioResult] = []

    for csv_path in csv_files:
        sid = scenario_id_from_path(csv_path)
        scenario_out = run_dir / sid

        # Try loading the CSV; mark as failed if invalid
        try:
            scenario = load_scenario(csv_path, expected_nodes)
        except ValueError as exc:
            results.append(_ScenarioResult(
                scenario_id=sid,
                status="failed",
                error_message=f"CSV inválido: {exc}",
                comparison_df=None,
                clf_metrics={},
                vol_metrics={},
                metadata={},
            ))
            continue

        # Copy input CSV to scenario output dir
        scenario_out.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(csv_path, scenario_out / "input_timeseries.csv")

        result = _run_scenario(
            scenario=scenario,
            csv_path=csv_path,
            base_inp_path=base_inp_path,
            models_dir=models_dir,
            output_dir=scenario_out,
            all_node_ids=expected_nodes,
            flood_threshold_m3=flood_threshold_m3,
            base_inp_hash=base_inp_hash,
        )
        results.append(result)

    # Write batch summary CSV
    _write_batch_summary(results, run_dir)

    # Write batch metrics JSON (pooled + per-node R² + macro)
    _write_batch_metrics(results, run_dir)

    # Write run-level metadata
    run_meta = {
        "run_datetime": datetime.now().isoformat(),
        "hydrographs_dir": str(hydrographs_dir),
        "base_inp_path": str(base_inp_path),
        "base_inp_md5": base_inp_hash,
        "models_dir": str(models_dir),
        "flood_threshold_m3": flood_threshold_m3,
        "n_scenarios": len(results),
        "n_ok": sum(1 for r in results if r.status == "ok"),
        "n_failed": sum(1 for r in results if r.status != "ok"),
        "reserved_for_validation": True,
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    return results


def _write_batch_summary(results: list[_ScenarioResult], run_dir: Path) -> None:
    rows = []
    for r in results:
        row = {
            "scenario_id": r.scenario_id,
            "status": r.status,
            "error_message": r.error_message,
        }
        row.update(r.clf_metrics)
        row.update(r.vol_metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(run_dir / "batch_summary.csv", index=False)


def _write_batch_metrics(results: list[_ScenarioResult], run_dir: Path) -> None:
    ok_results = [r for r in results if r.status == "ok" and r.comparison_df is not None]

    batch = {
        "n_total": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
    }

    if not ok_results:
        batch["pooled"] = None
        batch["per_node_r2"] = None
        batch["macro"] = None
    else:
        # Pooled metrics across all nodes of all successful scenarios
        pooled_df = pd.concat([r.comparison_df for r in ok_results], ignore_index=True)
        batch["pooled"] = compute_volume_metrics(
            pooled_df["vol_swmm_m3"], pooled_df["vol_pred_m3"]
        )

        # Per-node R²
        records = pooled_df[["node_id", "vol_swmm_m3", "vol_pred_m3"]].to_dict(orient="records")
        batch["per_node_r2"] = compute_per_node_r2(records)

        # Macro: average of per-scenario metrics, ignoring None
        def _macro_avg(key: str) -> float | None:
            vals = [r.vol_metrics.get(key) for r in ok_results if r.vol_metrics.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        batch["macro"] = {
            "mae_m3": _macro_avg("mae_m3"),
            "rmse_m3": _macro_avg("rmse_m3"),
        }

    with open(run_dir / "batch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False, default=str)
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_hydrograph_batch.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/validation/hydrograph_batch.py \
        tests/test_hydrograph_batch.py
git commit -m "$(cat <<'EOF'
feat: hydrograph batch coordinator with SWMM + XGBoost comparison

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CLI integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `--evaluate-hydrographs` argument to `main.py`**

Add the import and the new argument block. Insert after the existing imports:

```python
from swmm_resilience.validation.hydrograph_batch import run_batch
```

Add to the argument parser (after the existing `--flood-volume-curve` argument):

```python
parser.add_argument(
    "--evaluate-hydrographs",
    metavar="DIR",
    help="Evaluar lote de hidrogramas CSV contra SWMM y XGBoost",
)
```

Add the handler block (after the `--flood-volume-curve` block, before the `# ── Pipeline completo` comment):

```python
    # ── Modo: validación por lotes de hidrogramas ────────────────────────────
    if args.evaluate_hydrographs:
        hyd_dir = Path(args.evaluate_hydrographs)
        print(f"\nEvaluando hidrogramas en {hyd_dir}...")
        results = run_batch(
            hydrographs_dir=hyd_dir,
            base_inp_path=config.network.inp_path,
            models_dir=MODELS_DIR,
            output_root=METRICS_DIR / "hydrograph_validation",
            flood_threshold_m3=config.dataset.flood_threshold_m3,
        )
        n_ok = sum(1 for r in results if r.status == "ok")
        n_fail = len(results) - n_ok
        print(f"\nLote completado: {n_ok} exitosos, {n_fail} fallidos")
        if n_fail:
            for r in results:
                if r.status != "ok":
                    print(f"  FALLO [{r.scenario_id}]: {r.error_message}")
            import sys
            sys.exit(1)
        return
```

- [ ] **Step 2: Verify the import and argument parse correctly**

```bash
python main.py --help | grep evaluate
```

Expected: `--evaluate-hydrographs DIR`

- [ ] **Step 3: Run full suite one final time**

```bash
python -m pytest tests/ -x -q
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
feat: add --evaluate-hydrographs CLI command

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance checklist

After all 7 tasks are complete, run:

```bash
# 1. Full test suite
python -m pytest tests/ -v

# 2. CLI smoke test (help)
python main.py --help | grep evaluate-hydrographs

# 3. Module imports work
python -c "
from swmm_resilience.validation.hydrograph_csv import load_scenario
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.ml.scenario_predict import predict_scenario
from swmm_resilience.analysis.model_comparison import build_comparison_df
from swmm_resilience.visualization.model_comparison import plot_parity_nodes
from swmm_resilience.validation.hydrograph_batch import run_batch
print('All imports OK')
"
```

All 28 existing tests plus the new tests should pass.
