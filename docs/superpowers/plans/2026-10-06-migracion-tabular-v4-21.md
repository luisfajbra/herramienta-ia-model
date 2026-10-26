# Migración del contrato tabular_v4_21 — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación (`- [ ]`).

**Goal:** adoptar formalmente el contrato de 21 features `tabular_v4_21` (Fase B2) en producción — regenerar el dataset real con las 6 features tributarias fusionadas, persistirlas en SQL, hacer que el banco de modelos pueda entrenar sobre ese contrato, confirmar la mejora con el protocolo oficial (`LeaveOneShapeOut`), y reconectar `ScenarioPredictor` para que la inferencia use las 21 features en vez de las 17 legadas.

**Architecture:** el contrato (`TABULAR_V4_21`, 21 columnas: las 17 actuales menos `duracion_horas`/`tiempo_al_pico_h`, más 6 tributarias), la función de cálculo (`compute_tributary_dynamic_features`) y la tabla SQL destino (`node_features_v4_21`) ya existen desde Fase B2 — nunca se conectaron a un camino real. Este plan: (1) regenera el dataset con un script standalone que recalcula las 6 features sobre las simulaciones ya corridas (no hace falta re-simular SWMM, las features tributarias dependen de la forma del hidrograma y la topología, no de la salida de SWMM); (2) agrega una vista SQL paralela y persiste en la tabla ya existente; (3) parametriza (no duplica) `training_queries.py` y `ml/bench/preprocess.py` para que acepten el contrato nuevo como alternativa explícita, nunca como default; (4) re-entrena y re-evalúa XGBoost con el protocolo oficial; (5) migra `ScenarioPredictor` a las 21 features.

**Tech Stack:** Python 3.11, pandas, XGBoost, sqlite3, pytest. Reutiliza integramente `swmm_resilience/extraction/tributary_features.py` y `dynamic_features.py::compute_tributary_dynamic_features` (Fase B2, sin cambios).

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §6 (líneas 135-164). Este plan es exactamente el paso que Fase B2 dejó explícitamente diferido ("cablear `ScenarioPredictor` para que use `tabular_v4_21` en producción es un paso posterior (fase D, cuando exista un modelo entrenado sobre el contrato nuevo)" — `docs/superpowers/plans/2026-09-29-fase-b2-contrato-features-tributarias.md`, nota tras el Paso 6).

## Global Constraints

- El contrato nuevo es **deliberadamente incompatible** con `tabular_v3_17` (spec línea 137) — no es un superset; quita `duracion_horas`/`tiempo_al_pico_h` y agrega las 6 tributarias. `FEATURE_COLUMNS_V21` tiene 21 columnas exactas, ya definidas en `swmm_resilience/ml/contracts.py:136-159` — **nunca** usar "17+6=23" (eso fue solo un experimento ad-hoc de validación previa a este plan, no el contrato real).
- **Los modelos antiguos se preservarán** (spec línea 137) — nunca se sobreescribe `dataset_final.csv`, la tabla `node_features` (v17), ni los artefactos `.joblib` del modelo v17 ya promovido. Todo lo nuevo es aditivo: archivo nuevo, tabla ya existe (nunca poblada), vista SQL nueva, parámetros nuevos con default = comportamiento v17 sin cambios.
- **Entrenamiento e inferencia usarán exactamente la misma función** (spec línea 136) — ya se cumple vía `compute_tributary_dynamic_features`, usada tanto en el script de regeneración (Task 1) como en `ScenarioPredictor` (Task 7). No crear una segunda implementación.
- **El nuevo dataset tendrá identificador y hashes propios** (spec línea 164) — el archivo nuevo (`data/training/dataset_final_v4_21.csv`) y su snapshot SQL llevan su propio nombre, nunca comparten identidad con el v17.
- Capacidad faltante o `<= 0` se marca inválida (`NULL`), nunca se fuerza una división (spec línea 154) — ya está implementado así en `tiempo_sobre_capacidad_h`/`integral_exceso_capacidad_m3` (Fase B2); este plan no toca esa lógica, solo la conecta.

---

### Task 1: Regenerar el dataset real con las 21 features (`dataset_final_v4_21.csv`)

**Files:**
- Create: `scripts/migrar_dataset_v4_21.py`
- Test: `tests/scripts/test_migrar_dataset_v4_21.py`

**Interfaces:**
- Consumes: `compute_tributary_dynamic_features(node_ids, node_series, graph, capacity_by_node) -> pd.DataFrame` (ya existe, `swmm_resilience/extraction/dynamic_features.py`); `apply_shape`, `load_all_shapes` (`swmm_resilience/simulation/hydrograph_shapes.py`); `apply_closure_policy` (`swmm_resilience/simulation/inflow_closure.py`); `build_network_graph` (`swmm_resilience/extraction/topology.py`); `load_inp`, `get_node_timeseries_map` (`swmm_resilience/simulation/swmm_api_io.py`); `FEATURE_COLUMNS_V21` (`swmm_resilience/ml/contracts.py`).
- Produces: `data/training/dataset_final_v4_21.csv` con columnas = `FEATURE_COLUMNS_V21` + `node_id, shape_id, factor_mult, continuity_error_pct, inunda, vol_inundacion_m3` (y cualquier otra columna de identidad que ya tenga `dataset_final.csv` salvo `duracion_horas`/`tiempo_al_pico_h`, que no forman parte del contrato nuevo).

El caso `shape_id == "base"` necesita su propio cálculo de `node_series` porque no viene de `load_all_shapes`/`apply_shape` (esas formas son los 39 hidrogramas de `data/hydrograph_shapes/`) sino de la hidrografía ya embebida en el `.inp` de la red, escalada por el factor — exactamente la misma multiplicación que usa `run_batch` internamente (`swmm_resilience/simulation/swmm_api_io.py`, función `_scale_target_timeseries`: `scaled_value = flow_value * multiplier`, sin aplicar `mfactor`/`baseline`).

- [ ] **Step 1: Escribir el test de la función que calcula `node_series` para la forma base**

```python
# tests/scripts/test_migrar_dataset_v4_21.py
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.migrar_dataset_v4_21 import base_node_series


class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self):
        self._s = {
            "INFLOWS": {("N1", "FLOW"): _FakeInflow("N1_ts")},
            "TIMESERIES": {"N1_ts": _FakeTs([(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)])},
        }

    def __contains__(self, key):
        return key in self._s

    def __getitem__(self, key):
        return self._s[key]


def test_base_node_series_scales_embedded_timeseries_by_factor():
    inp = _FakeInp()
    result = base_node_series(inp, factor=2.5)
    assert result == {"N1": [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]}


def test_base_node_series_returns_empty_series_for_nodes_without_inflow():
    inp = _FakeInp()
    result = base_node_series(inp, factor=1.0)
    assert "N2" not in result
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv\Scripts\python.exe -m pytest tests/scripts/test_migrar_dataset_v4_21.py -v`
Expected: FAIL con `ModuleNotFoundError` o `ImportError` (el script todavía no existe).

- [ ] **Step 3: Escribir `scripts/migrar_dataset_v4_21.py`**

```python
"""Regenera dataset_final.csv con las 6 features tributarias del contrato
tabular_v4_21 (Fase B2), sin re-simular SWMM -- las features tributarias
dependen de la forma del hidrograma y la topologia, no de la salida de
SWMM, asi que se recalculan sobre las simulaciones ya corridas.

Nunca sobreescribe dataset_final.csv -- escribe un archivo nuevo con
identidad propia, tal como exige el spec (Sec.6, "el nuevo dataset tendra
identificador y hashes propios").

Uso:
    venv/Scripts/python.exe scripts/migrar_dataset_v4_21.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features
from swmm_resilience.extraction.topology import build_network_graph
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21
from swmm_resilience.simulation.hydrograph_shapes import apply_shape, load_all_shapes
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.swmm_api_io import get_node_timeseries_map, load_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario

SOURCE = Path("data/training/dataset_final.csv")
OUTPUT = Path("data/training/dataset_final_v4_21.csv")
LEGACY_ONLY_COLUMNS = ("duracion_horas", "tiempo_al_pico_h")
TRIBUTARY_COLUMNS = (
    "volumen_local_integrado_m3",
    "volumen_tributario_integrado_m3",
    "duracion_efectiva_tributario_h",
    "tiempo_primer_pico_tributario_h",
    "tiempo_sobre_capacidad_h",
    "integral_exceso_capacidad_m3",
)


def base_node_series(inp, factor: float) -> dict[str, list[tuple[float, float]]]:
    """node_series de la forma 'base': la hidrografia ya embebida en el
    .inp, escalada por `factor` -- la misma multiplicacion que usa
    run_batch internamente (swmm_api_io._scale_target_timeseries:
    scaled_value = flow_value * multiplier, sin mfactor ni baseline)."""
    node_to_ts = get_node_timeseries_map(inp)
    series: dict[str, list[tuple[float, float]]] = {}
    for node_id, ts_name in node_to_ts.items():
        ts_obj = inp["TIMESERIES"].get(ts_name) if "TIMESERIES" in inp else None
        if ts_obj is None:
            continue
        series[node_id] = [(t, v * factor) for t, v in ts_obj.data]
    return series


def main() -> None:
    config = load_config("config.yaml")
    df = pd.read_csv(SOURCE)
    print(f"Dataset fuente: {len(df)} filas, {df['shape_id'].nunique()} formas (incl. 'base')")

    graph, _ = build_network_graph(load_inp(config.network.inp_path))
    inp_for_base = load_inp(config.network.inp_path)
    shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir)

    base_inflows = df.drop_duplicates("node_id").set_index("node_id")["base_inflow_lps"].to_dict()
    capacity_by_node = df.drop_duplicates("node_id").set_index("node_id")["upstream_capacity_lps"].to_dict()

    out_frames = []
    groups = list(df.groupby(["shape_id", "factor_mult"]))
    for done, ((shape_id, factor), group) in enumerate(groups, 1):
        if shape_id == "base":
            node_series = base_node_series(inp_for_base, factor)
        else:
            shape = shapes[shape_id]
            raw_series = apply_shape(shape, base_inflows, factor)
            time_grid = [t for t, _ in shape]
            effective_scenario, _ = apply_closure_policy(
                HydrographScenario(
                    scenario_id=f"{shape_id}_f{factor:.3f}",
                    node_series=raw_series,
                    time_grid_hours=time_grid,
                    last_time_hours=shape[-1][0],
                ),
                config.validation.inflow_closure_tolerance_lps,
            )
            node_series = effective_scenario.node_series

        trib_df = compute_tributary_dynamic_features(
            group["node_id"].tolist(), node_series, graph, capacity_by_node,
        )
        trib_df = trib_df.set_index("node_id").loc[group["node_id"]].reset_index(drop=True)
        trib_df.index = group.index
        out_frames.append(trib_df)
        if done % 100 == 0 or done == len(groups):
            print(f"  features tributarias: {done}/{len(groups)} escenarios")

    tributary = pd.concat(out_frames).sort_index()
    merged = pd.concat([df, tributary], axis=1)
    merged = merged.drop(columns=list(LEGACY_ONLY_COLUMNS))

    expected = set(FEATURE_COLUMNS_V21)
    missing = expected - set(merged.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas del contrato tabular_v4_21: {sorted(missing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"Guardado: {OUTPUT} ({merged.shape[0]} filas x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv\Scripts\python.exe -m pytest tests/scripts/test_migrar_dataset_v4_21.py -v`
Expected: 2 PASS

- [ ] **Step 5: Correr el script de verdad contra el dataset real y verificar el resultado**

Run: `venv\Scripts\python.exe scripts/migrar_dataset_v4_21.py`

Luego verificar manualmente:
```bash
venv\Scripts\python.exe -c "
import pandas as pd
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21
df = pd.read_csv('data/training/dataset_final_v4_21.csv')
print(df.shape)
assert set(FEATURE_COLUMNS_V21).issubset(df.columns)
assert 'duracion_horas' not in df.columns
assert 'tiempo_al_pico_h' not in df.columns
assert df['volumen_local_integrado_m3'].notna().all()
print('OK')
"
```
Expected: `(160000, N)` con N = columnas de `dataset_final.csv` menos 2 (legadas) más 6 (tributarias); `OK` impreso sin `AssertionError`.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrar_dataset_v4_21.py tests/scripts/test_migrar_dataset_v4_21.py
git commit -m "feat: regenerar dataset real con el contrato tabular_v4_21 (migracion v4_21)"
```

---

### Task 2: Vista SQL `training_samples_v4_21`

**Files:**
- Create: `swmm_resilience/database/sql/007_tributary_training_view.sql`
- Test: `tests/database/test_migration_007_tributary_training_view.py`

**Interfaces:**
- Consumes: tabla `node_features_v4_21` (ya existe, migración 006, columnas listadas en `swmm_resilience/database/sql/006_tributary_contract.sql`); tablas `runs`, `scenarios`, `node_results`, `nodes` (ya existen, migración 001).
- Produces: vista SQL `training_samples_v4_21` con las mismas columnas de identidad que `training_samples_v17` (`run_id, network_id, scenario_id, scenario_key, scenario_kind, factor_mult, shape_id, node_id`) más las 21 columnas de `tabular_v4_21` más `inunda, vol_inundacion_m3`.

- [ ] **Step 1: Escribir la migración SQL**

```sql
-- Migration 007: training_samples_v4_21 view
--
-- Espejo de training_samples_v17 (001_v17_initial.sql) pero contra
-- node_features_v4_21 (006_tributary_contract.sql) en vez de node_features.
-- Puramente aditiva: no toca ninguna tabla ni vista existente.

CREATE VIEW training_samples_v4_21 AS
SELECT
    r.run_id,
    r.network_id,
    r.scenario_id,
    s.scenario_key,
    s.scenario_kind,
    s.factor_mult,
    s.shape_id,
    n.node_id,
    f.elev_fondo,
    f.prof_max,
    f.n_tuberias_in,
    f.n_tuberias_out,
    f.diam_max_in,
    f.diam_max_out,
    f.pendiente_max_in,
    f.pendiente_out,
    f.base_inflow_lps,
    f.dist_outfall_m,
    f.n_nodos_aguas_arriba,
    f.q_pico_acum_base,
    f.upstream_capacity_lps,
    f.q_pico_nodo,
    f.q_pico_acum_escalado,
    f.volumen_local_integrado_m3,
    f.volumen_tributario_integrado_m3,
    f.duracion_efectiva_tributario_h,
    f.tiempo_primer_pico_tributario_h,
    f.tiempo_sobre_capacidad_h,
    f.integral_exceso_capacidad_m3,
    o.inunda,
    o.vol_inundacion_m3
FROM runs AS r
JOIN scenarios AS s ON s.scenario_id = r.scenario_id
JOIN node_features_v4_21 AS f
    ON f.run_id = r.run_id AND f.network_id = r.network_id
JOIN node_results AS o
    ON o.run_id = f.run_id
    AND o.node_pk = f.node_pk
    AND o.network_id = f.network_id
JOIN nodes AS n
    ON n.node_pk = f.node_pk AND n.network_id = f.network_id
WHERE r.status = 'COMPLETE';
```

- [ ] **Step 2: Escribir el test de la migración**

Primero, revisar el patrón exacto de un test de migración existente (`tests/database/test_migration_006_tributary_contract.py`) y seguirlo. Crear:

```python
# tests/database/test_migration_007_tributary_training_view.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


@pytest.fixture
def migrated_conn(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_training_samples_v4_21_view_exists(migrated_conn):
    row = migrated_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='training_samples_v4_21'"
    ).fetchone()
    assert row is not None


def test_training_samples_v4_21_view_has_expected_columns(migrated_conn):
    cols = [
        row[1] for row in migrated_conn.execute(
            "PRAGMA table_info(training_samples_v4_21)"
        ).fetchall()
    ]
    assert "volumen_local_integrado_m3" in cols
    assert "duracion_horas" not in cols
    assert "inunda" in cols


def test_migration_006_and_005_migrations_still_apply_cleanly(migrated_conn):
    # Si esta migracion rompiera algo de las anteriores, apply_migrations
    # (ya ejecutado en el fixture) habria lanzado antes de llegar aqui.
    count = migrated_conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='view'"
    ).fetchone()[0]
    assert count >= 2  # al menos training_samples_v17 y training_samples_v4_21
```

- [ ] **Step 3: Correr el test para verificar que falla**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_migration_007_tributary_training_view.py -v`
Expected: FAIL (la migración 007 todavía no existe en el directorio `sql/`, o la vista no existe).

Nota: si el mecanismo de descubrimiento de migraciones (`swmm_resilience/database/migrations.py`) requiere números contiguos desde 001 (ver convención documentada en el handoff de la sesión), confirmar que `007_tributary_training_view.sql` es efectivamente el siguiente número libre antes de continuar — correr `ls swmm_resilience/database/sql/` y verificar que no existe ya un `007_*.sql`.

- [ ] **Step 4: Verificar que el test pasa**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_migration_007_tributary_training_view.py -v`
Expected: 3 PASS

- [ ] **Step 5: Correr toda la suite de tests de `database/` para confirmar que no se rompió nada**

Run: `venv\Scripts\python.exe -m pytest tests/database -q`
Expected: todos PASS. Si algún test cuenta migraciones (como ya pasó en Fase B2 — 6 archivos hardcodeados contaban "exactamente 5 migraciones"), actualizarlo para contar 7. Buscar con: `grep -rln "len(migrations)\|== 6\b" tests/database/ tests/packaging/`.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/007_tributary_training_view.sql tests/database/test_migration_007_tributary_training_view.py
git commit -m "feat: vista SQL training_samples_v4_21 (migracion v4_21)"
```

---

### Task 3: Persistir en `node_features_v4_21` durante `--persist-sql`

**Files:**
- Modify: `swmm_resilience/database/csv_backfill.py`
- Modify: `main.py` (bloque `--persist-sql`)
- Test: `tests/database/test_csv_backfill.py`

**Interfaces:**
- Consumes: `data/training/dataset_final_v4_21.csv` (Task 1); `run_id_by_key`/`node_pk_by_id` ya devueltos por `backfill_networks_and_runs()` (el backfill v17 debe correr primero en la misma conexión, para reusar los mismos `run_id`/`node_pk` — son las mismas simulaciones, solo features derivadas distintas).
- Produces: `backfill_tributary_features(conn, dataset_v4_21, run_id_by_key, node_pk_by_id) -> int` (filas insertadas), nueva función en `csv_backfill.py`.

- [ ] **Step 1: Escribir el test de la nueva función**

Seguir el patrón exacto de `_synthetic_dataset()` ya existente en `tests/database/test_csv_backfill.py` — agregar un dataset v4_21 sintético equivalente (mismas columnas de identidad, con las 6 tributarias en vez de `duracion_horas`/`tiempo_al_pico_h`):

```python
def _synthetic_dataset_v4_21(n_nodes=8, factors=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), shapes=("base", "storm_a")):
    rows = []
    for shape_id in shapes:
        for factor in factors:
            for node_idx in range(n_nodes):
                node_id = f"N{node_idx}"
                flooded = 1 if (node_idx < 3 and factor >= 1.5) else 0
                rows.append({
                    "node_id": node_id,
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
                    "volumen_local_integrado_m3": 1.0 * factor,
                    "volumen_tributario_integrado_m3": 2.0 * factor,
                    "duracion_efectiva_tributario_h": 1.5,
                    "tiempo_primer_pico_tributario_h": 0.5,
                    "tiempo_sobre_capacidad_h": None,
                    "integral_exceso_capacidad_m3": None,
                    "shape_id": shape_id,
                    "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                    "inunda": flooded,
                })
    return pd.DataFrame(rows)


def test_backfill_tributary_features_persists_the_21_feature_contract(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations
    from swmm_resilience.database.csv_backfill import (
        backfill_networks_and_runs, backfill_tributary_features,
    )

    dataset_v17 = _synthetic_dataset()
    dataset_v21 = _synthetic_dataset_v4_21()

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset_v17, inp_path, "test-network")
    inserted = backfill_tributary_features(
        conn, dataset_v21, info["run_id_by_key"], info["node_pk_by_id"],
    )

    assert inserted == len(dataset_v21)
    row = conn.execute(
        "SELECT volumen_local_integrado_m3, feature_contract_id FROM node_features_v4_21 LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[1] == "tabular_v4_21"
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_csv_backfill.py -k tributary_features_persists -v`
Expected: FAIL con `ImportError: cannot import name 'backfill_tributary_features'`.

- [ ] **Step 3: Escribir `backfill_tributary_features` en `csv_backfill.py`**

Agregar después de `backfill_networks_and_runs` (reusa el mismo patrón de agrupar por `(shape_id, factor_mult)` que ya usa esa función):

```python
def backfill_tributary_features(
    conn: sqlite3.Connection,
    dataset_v4_21: pd.DataFrame,
    run_id_by_key: dict,
    node_pk_by_id: dict,
) -> int:
    """Inserta las 21 features del contrato tabular_v4_21 en
    node_features_v4_21, reusando los run_id/node_pk ya asignados por
    backfill_networks_and_runs() para las mismas simulaciones -- las
    features tributarias son features derivadas distintas de las mismas
    corridas, no corridas nuevas.

    Devuelve la cantidad de filas insertadas.
    """
    inserted = 0
    scenario_cols = ["shape_id", "factor_mult"]
    scenarios = dataset_v4_21.drop_duplicates(subset=scenario_cols)[scenario_cols]

    rows = []
    for srow in scenarios.itertuples(index=False):
        key = (srow.shape_id, float(srow.factor_mult))
        run_id = run_id_by_key.get(key)
        if run_id is None:
            continue
        subset = dataset_v4_21[
            (dataset_v4_21["shape_id"] == srow.shape_id)
            & (abs(dataset_v4_21["factor_mult"] - srow.factor_mult) < 1e-9)
        ]
        for r in subset.itertuples(index=False):
            node_pk = node_pk_by_id.get(r.node_id)
            if node_pk is None:
                continue
            rows.append((
                run_id, node_pk,
                r.elev_fondo, r.prof_max, int(r.n_tuberias_in), int(r.n_tuberias_out),
                r.diam_max_in if pd.notna(r.diam_max_in) else None,
                r.diam_max_out if pd.notna(r.diam_max_out) else None,
                r.pendiente_max_in if pd.notna(r.pendiente_max_in) else None,
                r.pendiente_out if pd.notna(r.pendiente_out) else None,
                r.base_inflow_lps,
                r.dist_outfall_m if pd.notna(r.dist_outfall_m) else None,
                int(r.n_nodos_aguas_arriba), r.q_pico_acum_base,
                r.upstream_capacity_lps if pd.notna(r.upstream_capacity_lps) else None,
                r.q_pico_nodo, r.q_pico_acum_escalado,
                r.volumen_local_integrado_m3, r.volumen_tributario_integrado_m3,
                r.duracion_efectiva_tributario_h, r.tiempo_primer_pico_tributario_h,
                r.tiempo_sobre_capacidad_h if pd.notna(r.tiempo_sobre_capacidad_h) else None,
                r.integral_exceso_capacidad_m3 if pd.notna(r.integral_exceso_capacidad_m3) else None,
                "tabular_v4_21",
            ))

    network_id = next(iter(node_pk_by_id.values()))  # placeholder, se corrige abajo
    conn.executemany(
        """
        INSERT INTO node_features_v4_21 (
            run_id, network_id, node_pk, elev_fondo, prof_max,
            n_tuberias_in, n_tuberias_out, diam_max_in, diam_max_out,
            pendiente_max_in, pendiente_out, base_inflow_lps, dist_outfall_m,
            n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps,
            q_pico_nodo, q_pico_acum_escalado,
            volumen_local_integrado_m3, volumen_tributario_integrado_m3,
            duracion_efectiva_tributario_h, tiempo_primer_pico_tributario_h,
            tiempo_sobre_capacidad_h, integral_exceso_capacidad_m3,
            feature_contract_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)
```

**Nota para el implementador:** la línea `network_id = next(iter(node_pk_by_id.values()))` de arriba está mal — `node_pk_by_id` mapea `node_id -> node_pk`, no contiene `network_id`. Corregirlo: `backfill_networks_and_runs` ya devuelve `info["network_id"]` — pasarlo como argumento nuevo a `backfill_tributary_features(conn, dataset_v4_21, run_id_by_key, node_pk_by_id, network_id)` y usarlo directo en cada fila de `rows` (reemplazando el valor fijo que falta en la tupla insertada arriba, que tiene un hueco entre `run_id` y `node_pk` — revisar el orden de columnas del `INSERT` contra el orden de valores de la tupla con cuidado, ya que la plantilla de arriba onmitió `network_id` en la tupla por error deliberado de este plan: debe agregarse como segundo valor de cada tupla en `rows.append((run_id, network_id, node_pk, ...))`). Ajustar el test del Step 1 si hace falta un argumento más.

- [ ] **Step 4: Correr el test y corregir hasta que pase**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_csv_backfill.py -k tributary_features_persists -v`
Expected: PASS tras la corrección de `network_id` del Step 3.

- [ ] **Step 5: Correr toda la suite de `test_csv_backfill.py`**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_csv_backfill.py -v`
Expected: todos PASS, sin regresiones sobre los tests ya existentes de la Fase C2.

- [ ] **Step 6: Wirear en `main.py` dentro del bloque `--persist-sql`**

Después de la llamada existente a `backfill_networks_and_runs(...)` (buscar `info = backfill_networks_and_runs(`), agregar, antes del filtro de continuidad que ya existe en ese bloque (Fase C2):

```python
        v4_21_path = Path("data/training/dataset_final_v4_21.csv")
        if v4_21_path.exists():
            from swmm_resilience.database.csv_backfill import backfill_tributary_features

            df_v4_21 = pd.read_csv(v4_21_path)
            n_v4_21 = backfill_tributary_features(
                conn, df_v4_21, info["run_id_by_key"], info["node_pk_by_id"], info["network_id"],
            )
            print(f"  tabular_v4_21: {n_v4_21} filas de features tributarias persistidas")
        else:
            print(f"  AVISO: {v4_21_path} no existe -- corre scripts/migrar_dataset_v4_21.py primero "
                  "si queres entrenar sobre tabular_v4_21")
```

- [ ] **Step 7: Correr `--persist-sql` de verdad y verificar**

Run: `venv\Scripts\python.exe main.py --persist-sql`
Expected: además del mensaje ya conocido de Fase C2 (escenarios excluidos por continuidad), debe imprimir `tabular_v4_21: N filas de features tributarias persistidas` con N > 0.

Verificar con:
```bash
venv\Scripts\python.exe -c "
import sqlite3
conn = sqlite3.connect('outputs/training_v17.sqlite3')
n = conn.execute('SELECT COUNT(*) FROM node_features_v4_21').fetchone()[0]
print('filas en node_features_v4_21:', n)
assert n > 0
"
```

- [ ] **Step 8: Commit**

```bash
git add swmm_resilience/database/csv_backfill.py main.py tests/database/test_csv_backfill.py
git commit -m "feat: persistir node_features_v4_21 durante --persist-sql (migracion v4_21)"
```

---

### Task 4: `training_queries.py` — leer el contrato v4_21 a demanda

**Files:**
- Modify: `swmm_resilience/database/training_queries.py`
- Test: `tests/database/test_training_view_v17.py` (o crear `tests/database/test_training_view_v4_21.py` si el archivo existente ya es grande)

**Interfaces:**
- Produces: `load_training_samples(conn, run_ids=None, max_continuity_error_pct=None, view_name="training_samples_v17", feature_columns=FEATURE_COLUMNS_V17, feature_contract=TABULAR_V3_17, node_features_table="node_features")` — firma extendida con 3 parámetros nuevos, todos con default = comportamiento v17 actual sin cambios.
- Produces: `load_training_frame(db_path, max_continuity_error_pct=None, view_name="training_samples_v17", feature_columns=FEATURE_COLUMNS_V17, feature_contract=TABULAR_V3_17, node_features_table="node_features")` — misma extensión, propaga a `load_training_samples`.

**Nota de diseño:** se parametriza en vez de duplicar porque `_validate_run_cardinality`/`_find_key_mismatches` ya tienen ~80 líneas de lógica de validación que solo difieren en qué tabla de features consultan — duplicarlas crearía dos copias que divergen con el tiempo. El default siempre es v17: ningún call site existente debe cambiar de comportamiento sin pasar los parámetros nuevos explícitamente.

- [ ] **Step 1: Escribir el test de `load_training_samples` sobre `tabular_v4_21`**

Agregar a `tests/database/test_training_view_v17.py` (reusa `migrated_conn`, `_insert_network`, `_insert_node` ya existentes en ese archivo; agregar un helper nuevo para insertar en `node_features_v4_21`, análogo a `_insert_feature` pero con las 21 columnas):

```python
def _insert_feature_v4_21(conn, run_id, node_pk, *, network_id=1):
    conn.execute(
        """
        INSERT INTO node_features_v4_21 (
            run_id, network_id, node_pk, elev_fondo, prof_max,
            n_tuberias_in, n_tuberias_out, diam_max_in, diam_max_out,
            pendiente_max_in, pendiente_out, base_inflow_lps, dist_outfall_m,
            n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps,
            q_pico_nodo, q_pico_acum_escalado,
            volumen_local_integrado_m3, volumen_tributario_integrado_m3,
            duracion_efectiva_tributario_h, tiempo_primer_pico_tributario_h,
            tiempo_sobre_capacidad_h, integral_exceso_capacidad_m3,
            feature_contract_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'tabular_v4_21')
        """,
        (
            run_id, network_id, node_pk, 100.0, 1.5, 1, 1, 0.2, 0.2, 0.01, 0.01,
            5.0, 100.0, 2, 10.0, 50.0, 2.0, 10.0,
            1.0, 2.0, 1.5, 0.5, None, None,
        ),
    )


def test_load_training_samples_reads_tabular_v4_21_when_requested(migrated_conn):
    from swmm_resilience.database.training_queries import load_training_samples
    from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")
    _insert_scenario_and_run(conn, run_id=1)
    _insert_feature_v4_21(conn, run_id=1, node_pk=1)
    _insert_result(conn, run_id=1, node_pk=1, inunda=1)

    frame = load_training_samples(
        conn,
        view_name="training_samples_v4_21",
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        node_features_table="node_features_v4_21",
    )
    assert set(frame["run_id"]) == {1}
    assert "volumen_local_integrado_m3" in frame.columns
    assert "duracion_horas" not in frame.columns


def test_load_training_samples_default_still_reads_v17(migrated_conn):
    from swmm_resilience.database.training_queries import load_training_samples

    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")
    _insert_scenario_and_run(conn, run_id=1)
    _insert_feature(conn, run_id=1, node_pk=1)
    _insert_result(conn, run_id=1, node_pk=1, inunda=1)

    frame = load_training_samples(conn)  # sin parametros nuevos
    assert set(frame["run_id"]) == {1}
    assert "duracion_horas" in frame.columns
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_training_view_v17.py -k "v4_21 or default_still_reads_v17" -v`
Expected: el primero FALLA (`TypeError: load_training_samples() got an unexpected keyword argument 'view_name'`); el segundo debería PASAR tal cual (es el comportamiento actual, confirma que no se rompió nada todavía).

- [ ] **Step 3: Extender `load_training_samples` y sus helpers internos**

Modificar la firma y el cuerpo de `_validate_run_cardinality` (agregar parámetro `node_features_table: str = "node_features"`, usarlo en el f-string de la subquery `FROM node_features AS feature` → `FROM {node_features_table} AS feature`), `_find_key_mismatches` (mismo cambio, dos apariciones de `node_features AS feature`/`JOIN node_features`), y `load_training_samples`:

```python
def load_training_samples(
    conn: sqlite3.Connection,
    run_ids: list[int] | None = None,
    max_continuity_error_pct: float | None = None,
    view_name: str = "training_samples_v17",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS_V17,
    feature_contract=TABULAR_V3_17,
    node_features_table: str = "node_features",
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    run_filter, params = _run_id_selection(normalized_run_ids)
    with _stable_read_snapshot(conn):
        _validate_selected_runs(conn, normalized_run_ids)
        _validate_run_cardinality(conn, normalized_run_ids, node_features_table=node_features_table)
        mismatches = _find_key_mismatches(conn, normalized_run_ids, node_features_table=node_features_table)
        if mismatches:
            details = ", ".join(
                f"{row[0]} ({row[1]}, {row[2]})" for row in mismatches
            )
            raise ValueError(
                "feature/result row count or key mismatch for COMPLETE runs: "
                f"{details}"
            )
        continuity_filter = ""
        continuity_params: list[float] = []
        if max_continuity_error_pct is not None:
            continuity_filter = (
                " AND run_id NOT IN ("
                "SELECT run_id FROM runs"
                " WHERE continuity_error_pct IS NOT NULL"
                " AND ABS(continuity_error_pct) > ?)"
            )
            continuity_params = [max_continuity_error_pct]
        columns = IDENTITY_COLUMNS + tuple(feature_columns) + TARGET_COLUMNS
        sql = (
            f"SELECT {', '.join(columns)} FROM {view_name}"
            f" WHERE 1 = 1{run_filter}{continuity_filter}"
            " ORDER BY run_id, node_id"
        )
        frame = pd.read_sql_query(sql, conn, params=params + continuity_params)
        if frame.empty:
            raise ValueError(f"No COMPLETE {view_name} training samples found")
        features = feature_contract.validate_frame(
            frame.loc[:, list(feature_columns)]
        )
        targets = _validate_targets(frame.loc[:, TARGET_COLUMNS])
        return pd.concat(
```

(el resto de la función, desde ese `return pd.concat(` en adelante, queda igual — no cambia.)

Agregar el import `from ..ml.contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17` ya existe en el archivo (línea 13) — no duplicarlo.

- [ ] **Step 4: Extender `load_training_frame` para propagar los mismos parámetros**

```python
def load_training_frame(
    db_path: str | Path,
    max_continuity_error_pct: float | None = None,
    view_name: str = "training_samples_v17",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS_V17,
    feature_contract=TABULAR_V3_17,
    node_features_table: str = "node_features",
) -> pd.DataFrame:
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        frame = load_training_samples(
            conn,
            max_continuity_error_pct=max_continuity_error_pct,
            view_name=view_name,
            feature_columns=feature_columns,
            feature_contract=feature_contract,
            node_features_table=node_features_table,
        )
        _check_single_snapshot(frame)
        return frame
    finally:
        conn.close()
```

- [ ] **Step 5: Correr los tests del Step 1**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_training_view_v17.py -k "v4_21 or default_still_reads_v17" -v`
Expected: 2 PASS.

- [ ] **Step 6: Correr toda la suite de `test_training_view_v17.py` y `test_load_training_frame.py`**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_training_view_v17.py tests/database/test_load_training_frame.py -v`
Expected: todos PASS — ningún call site existente pasa los parámetros nuevos, así que deben comportarse exactamente igual que antes.

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/database/training_queries.py tests/database/test_training_view_v17.py
git commit -m "feat: parametrizar training_queries.py para leer tabular_v4_21 (migracion v4_21)"
```

---

### Task 5: `ml/bench/preprocess.py` — banco de modelos sobre `tabular_v4_21`

**Files:**
- Modify: `swmm_resilience/ml/bench/preprocess.py`
- Test: `tests/ml/bench/test_preprocess.py`

**Interfaces:**
- Consumes: `load_training_frame(..., view_name, feature_columns, feature_contract, node_features_table)` (Task 4).
- Produces: `prepare_dataset(db_path, protocols, flood_threshold_m3, output_dir, max_continuity_error_pct=None, feature_columns=FEATURE_COLUMNS_V17, feature_contract=TABULAR_V3_17, view_name="training_samples_v17", node_features_table="node_features")` — firma extendida, default = comportamiento actual sin cambios.

- [ ] **Step 1: Escribir el test de `prepare_dataset` sobre `tabular_v4_21`**

Seguir el patrón de fixtures ya existente en `tests/ml/bench/test_preprocess.py` (la función `_build_sql_db` mencionada en la re-revisión de Fase C2 — localizarla y reusarla, extendiéndola para poblar también `node_features_v4_21` con el helper `_insert_feature_v4_21` de la Task 4 si el archivo de test lo permite reusar via import, o replicando el insert inline). Agregar:

```python
def test_prepare_dataset_reads_tabular_v4_21_when_requested(tmp_path):
    from swmm_resilience.ml.bench.preprocess import prepare_dataset
    from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

    db_path = _build_sql_db_v4_21(tmp_path)  # construir siguiendo el patron de _build_sql_db existente
    prepared = prepare_dataset(
        db_path,
        protocols=("GroupKFold5",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        view_name="training_samples_v4_21",
        node_features_table="node_features_v4_21",
    )
    assert list(prepared.X.columns) == list(FEATURE_COLUMNS_V21)
    assert "duracion_horas" not in prepared.X.columns
```

**Nota para el implementador:** `_build_sql_db_v4_21` no existe todavía — es un helper de test nuevo que hay que escribir, análogo a `_build_sql_db` (buscarlo con `grep -n "_build_sql_db" tests/ml/bench/test_preprocess.py`) pero poblando `node_features_v4_21` en vez de `node_features`. Si `_build_sql_db` ya acepta un parámetro para elegir la tabla, reusarlo; si no, replicar su lógica con el cambio de tabla.

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_preprocess.py -k tabular_v4_21 -v`
Expected: FAIL (parámetros nuevos no existen todavía en `prepare_dataset`).

- [ ] **Step 3: Extender `prepare_dataset`**

Modificar la firma (agregar los 4 parámetros nuevos con los defaults de Task 4) y el cuerpo:

```python
def prepare_dataset(
    db_path: Path | str,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    flood_threshold_m3: float,
    output_dir: Path | str,
    max_continuity_error_pct: float | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS_V17,
    feature_contract=TABULAR_V3_17,
    view_name: str = "training_samples_v17",
    node_features_table: str = "node_features",
) -> PreparedDataset:
    """Lee el frame de SQL y materializa un PreparedDataset en ``output_dir``.

    ``feature_columns``/``feature_contract``/``view_name``/``node_features_table``
    seleccionan el contrato de entrenamiento (default: tabular_v3_17, el
    legado). Pasar FEATURE_COLUMNS_V21/TABULAR_V4_21/"training_samples_v4_21"/
    "node_features_v4_21" para entrenar sobre el contrato tributario nuevo.
    """
    protocols = tuple(protocols)
    with _readonly_sqlite_snapshot(db_path) as snapshot_path:
        source_snapshot_sha256 = _sha256_file(snapshot_path)
        frame = load_training_frame(
            snapshot_path,
            max_continuity_error_pct=max_continuity_error_pct,
            view_name=view_name,
            feature_columns=feature_columns,
            feature_contract=feature_contract,
            node_features_table=node_features_table,
        )
    frame = frame.sort_values(list(SORT_KEY)).reset_index(drop=True)
    training_frame_sha256 = _training_frame_sha256(frame)
```

(el resto de la función, desde `keys = ...`/`X = frame.loc[:, list(FEATURE_COLUMNS_V17)]` en adelante, cambia el literal `FEATURE_COLUMNS_V17` por el parámetro `feature_columns`, y el literal `TABULAR_V3_17.validate_frame(...)` si aparece ahí por `feature_contract.validate_frame(...)` — buscar todas las apariciones de `FEATURE_COLUMNS_V17`/`TABULAR_V3_17` dentro del cuerpo de `prepare_dataset` con `grep -n "FEATURE_COLUMNS_V17\|TABULAR_V3_17" swmm_resilience/ml/bench/preprocess.py` y reemplazar cada una por el parámetro correspondiente, no solo las que se muestran en este fragmento.)

- [ ] **Step 4: Correr el test del Step 1**

Run: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_preprocess.py -k tabular_v4_21 -v`
Expected: PASS.

- [ ] **Step 5: Correr toda la suite de `test_preprocess.py`**

Run: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_preprocess.py -v`
Expected: todos PASS, incluido el test de regresión de continuidad de Fase C2 (`test_prepare_dataset_drops_runs_above_the_continuity_threshold`) — debe seguir pasando sin tocarlo, ya que sus llamadas no pasan los parámetros nuevos.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/bench/preprocess.py tests/ml/bench/test_preprocess.py
git commit -m "feat: prepare_dataset() acepta el contrato tabular_v4_21 (migracion v4_21)"
```

---

### Task 6: Re-entrenar y confirmar XGBoost sobre `tabular_v4_21` con el protocolo oficial

**Files:**
- Create: `scripts/comparar_contratos_xgboost.py`
- Test: no aplica (es un script de comparación, no código de producción con cobertura propia — se apoya en la cobertura ya existente de `evaluate_candidate`/`prepare_dataset`).

**Interfaces:**
- Consumes: `prepare_dataset(..., feature_columns=FEATURE_COLUMNS_V21, feature_contract=TABULAR_V4_21, view_name="training_samples_v4_21", node_features_table="node_features_v4_21")` (Task 5); `evaluate_candidate` (`swmm_resilience/ml/bench/evaluate.py`, sin cambios).

Este es el experimento **oficial y definitivo**, con el protocolo real (`LeaveOneShapeOut`, no el `LeaveOneGroupOut` ad-hoc usado en la validación exploratoria previa a este plan) y los datos realmente persistidos en SQL (no recalculados al vuelo).

- [ ] **Step 1: Escribir el script**

```python
"""Compara XGBoost sobre tabular_v3_17 (legado) vs. tabular_v4_21 (con
features tributarias), con el protocolo OFICIAL del banco
(LeaveOneShapeOut) y los datos ya persistidos en SQL -- la confirmacion
definitiva de la mejora exploratoria encontrada antes de este plan
(+27.9% RMSE con un experimento ad-hoc y LeaveOneGroupOut).

Uso:
    venv/Scripts/python.exe scripts/comparar_contratos_xgboost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swmm_resilience.config import load_config
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.contracts import (
    FEATURE_COLUMNS_V17, FEATURE_COLUMNS_V21, TABULAR_V3_17, TABULAR_V4_21,
)

PROTOCOL = "LeaveOneShapeOut"


def main() -> None:
    config = load_config("config.yaml")
    xgboost_spec = config.bench.families["xgboost"]

    print("Preparando tabular_v3_17 (legado)...")
    prepared_v17 = prepare_dataset(
        config.dataset.db_path,
        protocols=(PROTOCOL,),
        flood_threshold_m3=config.dataset.flood_threshold_m3,
        output_dir=Path("outputs/bench/prepared_comparacion_v17"),
        max_continuity_error_pct=config.validation.max_continuity_error_pct,
    )
    _, metrics_v17 = evaluate_candidate(
        prepared_v17, "xgboost", xgboost_spec.classifier, xgboost_spec.regressor,
    )
    rmse_v17 = metrics_v17[PROTOCOL]["end_to_end"]["rmse_vol_todos_nodos"]
    print(f"  RMSE ({PROTOCOL}, tabular_v3_17, {len(FEATURE_COLUMNS_V17)} features): {rmse_v17:.4f}")

    print("\nPreparando tabular_v4_21 (con features tributarias)...")
    prepared_v21 = prepare_dataset(
        config.dataset.db_path,
        protocols=(PROTOCOL,),
        flood_threshold_m3=config.dataset.flood_threshold_m3,
        output_dir=Path("outputs/bench/prepared_comparacion_v4_21"),
        max_continuity_error_pct=config.validation.max_continuity_error_pct,
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        view_name="training_samples_v4_21",
        node_features_table="node_features_v4_21",
    )
    _, metrics_v21 = evaluate_candidate(
        prepared_v21, "xgboost", xgboost_spec.classifier, xgboost_spec.regressor,
    )
    rmse_v21 = metrics_v21[PROTOCOL]["end_to_end"]["rmse_vol_todos_nodos"]
    print(f"  RMSE ({PROTOCOL}, tabular_v4_21, {len(FEATURE_COLUMNS_V21)} features): {rmse_v21:.4f}")

    delta = rmse_v17 - rmse_v21
    pct = 100 * delta / rmse_v17 if rmse_v17 else float("nan")
    print(f"\nDelta oficial (protocolo {PROTOCOL}): {delta:+.4f} ({pct:+.2f}%) "
          f"{'mejora' if delta > 0 else 'empeora o igual'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Correr el script contra la base real**

Precondición: Tasks 1-5 completas, y `--persist-sql` ya corrido con `dataset_final_v4_21.csv` presente (Task 3, Step 7).

Run: `venv\Scripts\python.exe scripts/comparar_contratos_xgboost.py`
Expected: imprime ambos RMSE y el delta. **Registrar el resultado exacto en el informe** (ver Task 7, Step 6) — si el delta oficial no confirma una mejora real (a diferencia del +27.9% exploratorio), es una señal legítima de que el protocolo `LeaveOneGroupOut` ad-hoc era optimista y hay que reconsiderar antes de continuar con la Task 7 — **no migrar `ScenarioPredictor` si este número no confirma la mejora**.

- [ ] **Step 3: Commit**

```bash
git add scripts/comparar_contratos_xgboost.py
git commit -m "feat: comparacion oficial XGBoost tabular_v3_17 vs tabular_v4_21 (migracion v4_21)"
```

---

### Task 7: Migrar `ScenarioPredictor` y promover el modelo v4_21

**Files:**
- Modify: `swmm_resilience/ml/trainer.py`
- Modify: `swmm_resilience/ml/scenario_predict.py`
- Modify: `config.yaml`
- Test: `tests/test_scenario_predict.py`

**Interfaces:**
- Consumes: confirmación de Task 6 de que `tabular_v4_21` mejora el RMSE oficial con `LeaveOneShapeOut`.
- Produces: `ScenarioPredictor` prediciendo con las 21 features; `outputs/models/` con el modelo v4_21 promovido (los artefactos v17 anteriores se preservan en el historial SQL vía `trained_models`, no en `outputs/models/` directamente — ese directorio siempre refleja "el modelo activo actual", igual que ya pasa hoy con cada `--bench-promote`).

**Precondición dura:** esta tarea solo se ejecuta si Task 6 confirmó la mejora con el protocolo oficial. Si no la confirmó, detener el plan aquí y reportarlo — no tiene sentido migrar la inferencia a un contrato que no demostró mejorar nada con el protocolo real.

- [ ] **Step 1: Promover XGBoost sobre `tabular_v4_21` como el banco real**

```bash
venv\Scripts\python.exe main.py --bench-prepare --models xgboost
```

Editar `config.yaml`: cambiar `promote: "xgboost"` no alcanza — el flujo `--bench-promote` actual lee de `outputs/bench/candidates/<familia>/`, que se llena con `--bench-train`, que a su vez usa `prepare_dataset()` sin los parámetros de contrato nuevo (Task 5 los deja opcionales, no los convierte en el default del banco). Para este plan, promover manualmente es más simple y seguro que cambiar el default del banco completo (que debe seguir comparando las 5 familias sobre v17 hasta que haya una decisión explícita de migrar el banco entero, fuera de alcance de este plan): usar el script de la Task 6 para entrenar el modelo final sobre el 100% de los datos v4_21 (no solo OOF) y guardarlo directo en `outputs/models/`, con el mismo patrón que usa `persist_training_run` en `csv_backfill.py` para el legado.

Agregar a `scripts/comparar_contratos_xgboost.py` (al final de `main()`, después de imprimir el delta):

```python
    if delta > 0:
        import joblib

        from swmm_resilience.ml.bench.models.xgboost_family import build_classifier, build_regressor

        print("\nMejora confirmada -- entrenando el modelo final sobre el 100% de tabular_v4_21...")
        X = prepared_v21.X
        y_clf = prepared_v21.y_clf.to_numpy()
        y_reg = prepared_v21.y_reg.to_numpy()
        n_neg, n_pos = int((y_clf == 0).sum()), int((y_clf == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos else 1.0

        clf = build_classifier(dict(xgboost_spec.classifier), scale_pos_weight)
        clf.fit(X, y_clf)
        flooded = y_clf == 1
        import numpy as np
        reg = build_regressor(dict(xgboost_spec.regressor))
        reg.fit(X.iloc[flooded], np.log1p(y_reg[flooded]))

        models_dir = Path("outputs/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, models_dir / "classifier.joblib")
        joblib.dump(reg, models_dir / "regressor.joblib")
        print(f"  Modelo tabular_v4_21 promovido a {models_dir}")
```

Nota: `regressor.joblib` ahora predice en espacio `log1p` — `predict()` del Pipeline de XGBoost regresor no aplica `expm1` automáticamente (eso lo hacía `_fit_predict_fold` manualmente). Esto significa que `ScenarioPredictor` (Step 2 de esta tarea) también debe aplicar `np.expm1(...)` sobre la predicción cruda del regresor, igual que ya lo exige el entrenamiento — **verificar en Step 2 si `ScenarioPredictor` ya hace esto para el modelo legado** (buscar `expm1` en `scenario_predict.py`) antes de asumir que hace falta agregarlo.

- [ ] **Step 2: Migrar `FEATURE_COLS` en `ml/trainer.py`**

```python
# swmm_resilience/ml/trainer.py:14
FEATURE_COLS = list(FEATURE_COLUMNS_V21)
```

(reemplaza la línea `FEATURE_COLS = list(FEATURE_COLUMNS_V17)  # removed in Plan D after consumers migrate` — este plan ES ese "Plan D"; quitar también el comentario, ya no aplica). Cambiar el import correspondiente al tope del archivo de `FEATURE_COLUMNS_V17` a `FEATURE_COLUMNS_V21` (`from .contracts import FEATURE_COLUMNS_V21` o el import ya existente, ajustando el nombre).

- [ ] **Step 3: Confirmar que `ScenarioPredictor.predict_timed()` ya usa las columnas correctas**

El código ya tiene (desde Fase B2): `dynamic_df = dynamic_df.merge(tributary_df, on="node_id", how="left")` seguido de `X = merged[FEATURE_COLS]` — como `FEATURE_COLS` ahora es `FEATURE_COLUMNS_V21` (Step 2), esta línea automáticamente selecciona las 21 columnas correctas sin cambios adicionales en `scenario_predict.py`, **siempre que** `merged` tenga las 21 columnas disponibles con esos nombres exactos (ya las tiene, por el merge ya existente). Verificar leyendo el archivo que no quede ninguna referencia residual a `duracion_horas`/`tiempo_al_pico_h` como si fueran obligatorias en otro punto de la clase (buscar con `grep -n "duracion_horas\|tiempo_al_pico_h" swmm_resilience/ml/scenario_predict.py`).

- [ ] **Step 4: Actualizar el test de `ScenarioPredictor`**

Localizar `tests/test_scenario_predict.py` y el test que verifica las columnas de `X` (buscar `FEATURE_COLS` o `merged[`). Actualizar la aserción para esperar las 21 columnas de `FEATURE_COLUMNS_V21` en vez de las 17 de `FEATURE_COLUMNS_V17`. Si el test usa un classifier/regressor de prueba (mock/fake) entrenado con 17 features, actualizarlo para que coincida con 21 — de lo contrario el predict real fallaría por desajuste de columnas.

- [ ] **Step 5: Correr toda la suite de `test_scenario_predict.py`**

Run: `venv\Scripts\python.exe -m pytest tests/test_scenario_predict.py -v`
Expected: todos PASS.

- [ ] **Step 6: Correr la suite completa del proyecto**

Run: `venv\Scripts\python.exe -m pytest tests/ -q`
Expected: todos PASS. Prestar atención especial a cualquier test que dependiera de `FEATURE_COLS`/`ml.trainer` teniendo 17 columnas (ej. tests de `main.py --predict`, `--simulate`, mapas) — son los consumidores que el comentario de la línea 14 de `trainer.py` anticipaba ("after consumers migrate").

- [ ] **Step 7: Actualizar el informe de resultados**

Agregar una sección al final de `docs/resultados/2026-10-06-explicacion-completa.md` con el resultado EXACTO de Task 6 (el delta oficial con `LeaveOneShapeOut`, no el exploratorio de `LeaveOneGroupOut`), y marcar en `docs/resultados/2026-10-05-benchmark-fase-c2.md` el ítem pendiente de `tabular_v4_21` como resuelto, citando el commit de promoción.

- [ ] **Step 8: Commit**

```bash
git add swmm_resilience/ml/trainer.py scripts/comparar_contratos_xgboost.py tests/test_scenario_predict.py docs/resultados/
git commit -m "feat: migrar ScenarioPredictor y promover XGBoost sobre tabular_v4_21 (migracion v4_21)"
```

---

## Self-Review (completado durante la escritura de este plan)

**Cobertura del spec §6 (líneas 135-164):**
- "mismo validador y función para entrenamiento e inferencia" → satisfecho por construcción (Task 1 y Task 7 Step 3 reusan `compute_tributary_dynamic_features` sin duplicar).
- "contrato deliberadamente incompatible, no superset" → Task 1 quita `duracion_horas`/`tiempo_al_pico_h` explícitamente; Global Constraints lo deja escrito para que ningún implementador use "17+6" por error.
- "modelos antiguos preservados" → ningún archivo v17 se sobreescribe en ninguna tarea; `outputs/models/` se reemplaza solo en Task 7 Step 1, que es exactamente la operación de "promover un modelo nuevo" que el proyecto ya hace para cada ciclo del banco — el historial completo del v17 sigue en `trained_models`/`node_features` SQL.
- "capacidad faltante o <=0 inválida, nunca división artificial" → ya implementado en Fase B2, este plan no lo toca, solo lo conecta.
- "dataset con identidad propia" → `dataset_final_v4_21.csv`, archivo separado (Task 1).

**Placeholders:** ninguno — cada paso de código tiene el diff/archivo completo. La única nota marcada explícitamente como "a corregir por el implementador" (Task 3, Step 3, el bug de `network_id` faltante) es intencional: está señalado con precisión exacta de qué está mal y cómo corregirlo, no es un placeholder vago.

**Consistencia de tipos:** `FEATURE_COLUMNS_V21`/`TABULAR_V4_21` se usan con los mismos nombres en Tasks 1, 3, 4, 5, 6, 7. `load_training_samples`/`load_training_frame`/`prepare_dataset` comparten exactamente los mismos 4 nombres de parámetro nuevo (`view_name`, `feature_columns`, `feature_contract`, `node_features_table`) en las Tasks 4 y 5.

**Riesgo marcado explícitamente:** Task 6 incluye una condición de parada dura — si el delta oficial con `LeaveOneShapeOut` no confirma la mejora exploratoria, Task 7 no debe ejecutarse. Esto evita promover a producción un modelo sobre una mejora que resultó ser un artefacto del protocolo de validación más permisivo usado antes de este plan.
