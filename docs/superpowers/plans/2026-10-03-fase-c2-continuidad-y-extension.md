# Fase C2 — Exclusión por continuidad y extensión de simulación — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación (`- [ ]`).

**Goal:** (A) convertir el chequeo de continuidad de routing SWMM, hoy solo un *warning*, en una exclusión real del conjunto de entrenamiento — con el mismo conjunto elegible para las cinco familias del banco y un reporte separado de lo excluido — y (B) dar a `run_batch_shapes` un reintento configurable con extensión de `drain_down_hours` cuando una simulación no termina `drained`, en vez de aceptar silenciosamente cualquier estado no-`drained`.

**Architecture:** (A) reutiliza la columna `runs.continuity_error_pct`, ya presente en el esquema SQL desde v17 pero nunca poblada — el valor se calcula durante `run_batch_shapes`/ensamblado de `dataset_final.csv` (reusando el parser de `.rpt` que ya existe en `hydrograph_batch.py`, movido a un módulo compartido), fluye a través de `dataset_final.csv` como columna por escenario, se persiste en `runs` durante `--persist-sql`, y se filtra en `training_queries.load_training_samples()` — el único punto de lectura que usan todas las familias del banco, así que filtrar ahí garantiza el mismo conjunto elegible para las cinco. (B) añade un bucle de reintento dentro de `run_batch_shapes`: si el estado de drenaje (Fase C1) no es `drained`, se regenera el `.inp` con `drain_down_hours` incrementado y se vuelve a simular, hasta un límite configurable, registrando cada intento.

**Tech Stack:** Python 3.11, `swmm_api`, `pyswmm`, pandas, sqlite3, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §7 (líneas 184-190). Este plan cierra los dos ítems que la Fase C1 dejó explícitamente fuera de su alcance (ver `docs/superpowers/plans/2026-09-29-fase-c1-estado-drenaje.md`, sección Spec): exclusión por continuidad >5% y extensión configurable de simulaciones. La clasificación `drained`/`not_drained`/`unverified` (Fase C1) ya existe y este plan la consume sin modificarla.

## Global Constraints

- "Un error absoluto de continuidad de routing superior a 5% marcará la referencia como cuestionada. Ese escenario saldrá del benchmark principal y aparecerá en un reporte separado. Todas las familias usarán exactamente el mismo conjunto elegible." (spec §7). "Benchmark principal" se interpreta como el camino de lectura de `training_queries.load_training_samples()`, porque es el único punto que consumen las cinco familias del banco (`swmm_resilience/ml/bench/`) — filtrar ahí, y solo ahí, es lo que garantiza el mismo conjunto elegible sin duplicar lógica por familia.
- "No se extenderá una simulación indefinidamente. Las extensiones repetidas tendrán incremento y límite configurables, ambos registrados." (spec §7). El límite es un número de reintentos, no un tiempo total — ver Ruling de Task 6.
- El umbral por defecto de continuidad es **5.0%**, igual al `CONTINUITY_ERROR_WARN_PCT` ya existente en `hydrograph_batch.py` — Fase C2 no cambia el umbral, lo hace exigible en vez de solo informativo.
- **Ruling (NULL = no excluido).** `dataset_final.csv`/`runs` generados antes de este plan no tienen `continuity_error_pct` (columna ausente o `NULL`). Un valor `NULL` se trata como "no excluido" en `load_training_samples()`, nunca como "cuestionado" — excluir por ausencia de dato rompería retroactivamente datasets ya persistidos sin ninguna señal real de mala continuidad. **Costo si es incorrecto:** un dataset viejo con de verdad mala continuidad (nunca detectada porque esta columna no existía) seguiría incluido hasta que se regenere con este plan aplicado; no hay forma de distinguir eso de un dataset simplemente nunca medido sin volver a simular.
- **Ruling (reporte separado = CSV junto a la base SQLite).** El spec no fija un formato ni una ubicación para "reporte separado". Este plan lo materializa como `continuity_excluded.csv` escrito junto a `config.dataset.db_path` cada vez que corre `--persist-sql`, con columnas `run_id, scenario_key, shape_id, factor_mult, continuity_error_pct`. **Costo si es incorrecto:** es solo una convención de nombre/ubicación de archivo — moverlo después es un cambio de una línea en `main.py`, no afecta el filtrado real.
- El filtro de continuidad se implementa como una subconsulta `NOT IN` sobre `runs` dentro de `load_training_samples()`, sin tocar la vista `training_samples_v17` ni agregar una migración — la columna `continuity_error_pct` ya existe en `runs` desde `001_v17_initial.sql`.

---

### Task 1: Función compartida de lectura de continuidad + config del umbral

**Files:**
- Modify: `swmm_resilience/simulation/swmm_api_io.py` (agregar función pública)
- Modify: `swmm_resilience/validation/hydrograph_batch.py:51,179-196` (usar la función compartida en vez de la privada local)
- Modify: `swmm_resilience/config.py:255-262` (agregar campo a `ValidationConfig`), `:380-475` (parsear en `load_config`)
- Modify: `config.yaml:16-23` (agregar `max_continuity_error_pct`)
- Test: `tests/simulation/test_swmm_api_io.py` (crear si no existe, o agregar al archivo de tests existente de ese módulo)
- Test: `tests/test_config.py` (agregar casos para el nuevo campo)

**Interfaces:**
- Produces: `swmm_resilience.simulation.swmm_api_io.read_continuity_error(rpt_path: Path) -> float | None` — peor (máximo valor absoluto) error de continuidad (%) reportado en el `.rpt`; `None` si no se pudo leer o no hay match.
- Produces: `Config.validation.max_continuity_error_pct: float` (default `5.0`).

- [ ] **Step 1: Mover la función a `swmm_api_io.py` con un nombre público**

En `swmm_resilience/simulation/swmm_api_io.py`, agregar al final del archivo:

```python
import re

_CONTINUITY_RE = re.compile(
    r"Continuity Error \(%\)\s*\.*\s*(-?\d+(?:\.\d+)?)"
)


def read_continuity_error(rpt_path: Path) -> float | None:
    """Worst (max |value|) continuity error (%) reported in the .rpt.

    A routing continuity error above the configured threshold means SWMM
    itself does not trust its mass balance for that scenario (spec Sec.7).
    """
    try:
        text = Path(rpt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = [float(v) for v in _CONTINUITY_RE.findall(text)]
    if not values:
        return None
    return max(values, key=abs)
```

(`re` probablemente no está importado en este archivo todavía — agregar el `import re` junto a los demás imports al inicio del archivo, no inline como aquí arriba.)

- [ ] **Step 2: Reapuntar `hydrograph_batch.py` a la función compartida**

En `swmm_resilience/validation/hydrograph_batch.py`, eliminar la definición local (líneas 179-196: el `_CONTINUITY_RE` y `_read_continuity_error`) y en su lugar, junto al bloque de imports existente (línea ~33, donde ya se importa `from ..simulation.swmm_api_io import get_node_timeseries_map, load_inp`), cambiar a:

```python
from ..simulation.swmm_api_io import (
    get_node_timeseries_map, load_inp, read_continuity_error as _read_continuity_error,
)
```

Esto mantiene el único call site existente (línea 353: `continuity_pct = _read_continuity_error(rpt_path)`) y el test existente (`tests/test_hydrograph_batch.py:528`, que llama `hydrograph_batch._read_continuity_error(rpt)`) funcionando sin cambios — es un alias de importación, no una función nueva.

- [ ] **Step 3: Test de la función movida**

Crear `tests/simulation/test_swmm_api_io.py` si no existe (si ya existe, agregar este test al final):

```python
from pathlib import Path

from swmm_resilience.simulation.swmm_api_io import read_continuity_error


def test_read_continuity_error_parses_worst_value(tmp_path):
    rpt = tmp_path / "run.rpt"
    rpt.write_text(
        "\n".join(
            [
                "  Runoff Quantity Continuity",
                "  Continuity Error (%) .....        -0.05",
                "  Flow Routing Continuity",
                "  Continuity Error (%) .....        -7.31",
            ]
        ),
        encoding="utf-8",
    )
    assert read_continuity_error(rpt) == -7.31


def test_read_continuity_error_returns_none_for_missing_file(tmp_path):
    assert read_continuity_error(tmp_path / "does_not_exist.rpt") is None


def test_read_continuity_error_returns_none_when_no_match(tmp_path):
    rpt = tmp_path / "run.rpt"
    rpt.write_text("no continuity section here", encoding="utf-8")
    assert read_continuity_error(rpt) is None
```

- [ ] **Step 4: Correr los tests nuevos y el existente de hydrograph_batch**

Run: `venv\Scripts\python.exe -m pytest tests/simulation/test_swmm_api_io.py tests/test_hydrograph_batch.py -k continuity -v`
Expected: 4 PASS (3 nuevos + `test_read_continuity_error_parses_worst_value` en `test_hydrograph_batch.py`, que sigue pasando porque `_read_continuity_error` sigue existiendo como alias).

- [ ] **Step 5: Agregar `max_continuity_error_pct` a `ValidationConfig`**

En `swmm_resilience/config.py`, en la clase `ValidationConfig` (línea 255-262), agregar un campo:

```python
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"  # "strict" | "close"
    drainage_inflow_tolerance_lps: float = 0.01
    drainage_flooding_tolerance_lps: float = 0.01
    drainage_storage_change_tolerance_m3: float = 0.01
    drainage_tail_window_hours: float = 1.0
    max_continuity_error_pct: float = 5.0
```

En `load_config()` (dentro del bloque `validation=ValidationConfig(...)`, después de `drainage_tail_window_hours=...` alrededor de la línea 473), agregar:

```python
            max_continuity_error_pct=float(
                (raw.get("validation") or {}).get("max_continuity_error_pct", 5.0)
            ),
```

- [ ] **Step 6: Agregar la clave a `config.yaml`**

En `config.yaml`, dentro del bloque `validation:` (línea 16-23), agregar la última línea:

```yaml
validation:
  drain_down_hours: 6.0
  inflow_closure_tolerance_lps: 0.01
  csv_inflow_closure_mode: "strict"
  drainage_inflow_tolerance_lps: 0.01
  drainage_flooding_tolerance_lps: 0.01
  drainage_storage_change_tolerance_m3: 0.01
  drainage_tail_window_hours: 1.0
  max_continuity_error_pct: 5.0
```

- [ ] **Step 7: Tests de config**

En `tests/test_config.py`, siguiendo el patrón exacto de `test_validation_drainage_state_defaults`/`test_validation_drainage_state_is_parsed_when_present` (líneas 143-163), agregar:

```python
def test_validation_max_continuity_error_pct_default(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.max_continuity_error_pct == 5.0


def test_validation_max_continuity_error_pct_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  max_continuity_error_pct: 3.0',
    )
    config = load_config(cfg)
    assert config.validation.max_continuity_error_pct == 3.0
```

- [ ] **Step 8: Correr tests de config**

Run: `venv\Scripts\python.exe -m pytest tests/test_config.py -k continuity -v`
Expected: 2 PASS

- [ ] **Step 9: Commit**

```bash
git add swmm_resilience/simulation/swmm_api_io.py swmm_resilience/validation/hydrograph_batch.py swmm_resilience/config.py config.yaml tests/simulation/test_swmm_api_io.py tests/test_config.py
git commit -m "feat: compartir lector de continuidad y agregar umbral configurable (fase C2)"
```

---

### Task 2: Propagar `continuity_error_pct` a `dataset_final.csv`

**Files:**
- Modify: `main.py:1077-1125` (bucle de `shape_results` en el modo `--extend-shapes` o equivalente)
- Test: `tests/dataset/test_assembler.py` (si existe) o un test de integración mínimo nuevo

**Interfaces:**
- Consumes: `read_continuity_error(rpt_path)` de Task 1.
- Produces: `dataset_final.csv` tiene una columna `continuity_error_pct` (float, puede ser NaN), con el mismo valor repetido en todas las filas de un mismo `(shape_id, factor)` — igual convención que `shape_id` en la línea 1085.

- [ ] **Step 1: Leer y adjuntar el error de continuidad por escenario**

En `main.py`, dentro del bucle `for shape_id, factor, rpt_path in shape_results:` (línea 1078), inmediatamente después de la línea `dynamic_df["shape_id"] = shape_id` (línea 1085), agregar:

```python
                from swmm_resilience.simulation.swmm_api_io import read_continuity_error

                dynamic_df["continuity_error_pct"] = read_continuity_error(rpt_path)
```

- [ ] **Step 2: Test de integración mínimo**

Buscar primero si existe un test de `assemble_dataset` en `tests/dataset/` o `tests/test_assembler.py`. Si existe un archivo de tests para `assemble_dataset`, agregar un caso que confirme que una columna extra en `dynamic_df` llega intacta a la salida:

```python
def test_assemble_dataset_passes_through_extra_dynamic_columns(tmp_path):
    import pandas as pd
    from swmm_resilience.dataset.assembler import assemble_dataset

    static_topo_df = pd.DataFrame({
        "node_id": ["N1", "N2"],
        "coord_x": [0.0, 1.0], "coord_y": [0.0, 1.0],
        "elev_fondo": [100.0, 101.0], "prof_max": [1.5, 1.5],
        "n_tuberias_in": [1, 1], "n_tuberias_out": [1, 1],
        "diam_max_in": [0.2, 0.2], "diam_max_out": [0.2, 0.2],
        "pendiente_max_in": [0.01, 0.01], "pendiente_out": [0.01, 0.01],
        "base_inflow_lps": [5.0, 5.0], "dist_outfall_m": [100.0, 100.0],
        "n_nodos_aguas_arriba": [0, 1], "q_pico_acum_base": [10.0, 10.0],
        "upstream_capacity_lps": [50.0, 50.0],
    })
    dynamic_df = pd.DataFrame({
        "node_id": ["N1", "N2"],
        "q_pico_nodo": [2.0, 2.0], "q_pico_acum_escalado": [10.0, 10.0],
        "duracion_horas": [3.0, 3.0], "tiempo_al_pico_h": [0.7, 0.7],
        "shape_id": ["storm_a", "storm_a"],
        "continuity_error_pct": [-7.31, -7.31],
    })
    labels_df = pd.DataFrame({
        "node_id": ["N1", "N2"], "vol_inundacion_m3": [0.0, 0.0], "inunda": [0, 0],
    })
    df = assemble_dataset(
        static_topo_df, [(1.0, dynamic_df, labels_df)], tmp_path / "out.csv",
    )
    assert "continuity_error_pct" in df.columns
    assert df["continuity_error_pct"].tolist() == [-7.31, -7.31]
```

Si no existe ningún archivo de tests para `assemble_dataset`, crear `tests/dataset/test_assembler.py` con este único test (agregar `tests/dataset/__init__.py` vacío si el directorio no lo tiene, siguiendo la convención de paquetes de test del repo — verificar con `ls tests/dataset/` antes).

- [ ] **Step 3: Correr el test**

Run: `venv\Scripts\python.exe -m pytest tests/dataset/test_assembler.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add main.py tests/dataset/test_assembler.py
git commit -m "feat: propagar continuity_error_pct por escenario a dataset_final.csv (fase C2)"
```

---

### Task 3: Persistir `continuity_error_pct` en la tabla `runs`

**Files:**
- Modify: `swmm_resilience/database/csv_backfill.py:143-152` (INSERT INTO runs)
- Test: `tests/database/test_csv_backfill.py`

**Interfaces:**
- Consumes: columna `continuity_error_pct` en `dataset` (puede faltar en CSVs antiguos — Ruling de Global Constraints).
- Produces: `runs.continuity_error_pct` poblado (float o `NULL`).

- [ ] **Step 1: Leer el valor por escenario y agregarlo al INSERT**

En `swmm_resilience/database/csv_backfill.py`, dentro de `backfill_networks_and_runs`, justo antes del bloque `cur = conn.execute("""INSERT INTO runs (...)""", ...)` (línea ~143-152), agregar:

```python
        if "continuity_error_pct" in subset.columns:
            raw_continuity = subset["continuity_error_pct"].iloc[0]
            continuity_error_pct = (
                float(raw_continuity) if pd.notna(raw_continuity) else None
            )
        else:
            continuity_error_pct = None

        cur = conn.execute(
            """
            INSERT INTO runs (
                scenario_id, network_id, status, started_at_utc, completed_at_utc,
                config_sha256, node_count, continuity_error_pct
            ) VALUES (?, ?, 'COMPLETE', ?, ?, ?, ?, ?)
            """,
            (
                scenario_id, network_id, _now(), _now(), config_sha256, len(subset),
                continuity_error_pct,
            ),
        )
```

(Esto reemplaza el bloque `INSERT INTO runs` existente en el mismo lugar — mismo `cur = conn.execute(...)`, con la columna y el parámetro nuevos agregados.)

- [ ] **Step 2: Test — valor presente se persiste**

En `tests/database/test_csv_backfill.py`, extender `_synthetic_dataset` para aceptar continuidad opcional, o agregar un test dedicado que construya un dataset mínimo con la columna. Agregar al final del archivo:

```python
def test_backfill_persists_continuity_error_pct_when_present(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    dataset = _synthetic_dataset()
    dataset["continuity_error_pct"] = -7.31

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset, inp_path, "test-network")

    run_id = next(iter(info["run_id_by_key"].values()))
    row = conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row[0] == pytest.approx(-7.31)


def test_backfill_leaves_continuity_error_pct_null_when_column_absent(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    dataset = _synthetic_dataset()
    assert "continuity_error_pct" not in dataset.columns

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset, inp_path, "test-network")

    run_id = next(iter(info["run_id_by_key"].values()))
    row = conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row[0] is None
```

- [ ] **Step 3: Correr los tests**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_csv_backfill.py -k continuity -v`
Expected: 2 PASS

- [ ] **Step 4: Correr toda la suite de csv_backfill para confirmar que no se rompió nada existente**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_csv_backfill.py -v`
Expected: todos PASS (el INSERT cambiado agrega una columna, no quita ninguna — los tests existentes que no setean `continuity_error_pct` en su dataset sintético deben seguir pasando con el valor `None`).

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/database/csv_backfill.py tests/database/test_csv_backfill.py
git commit -m "feat: persistir continuity_error_pct en runs durante --persist-sql (fase C2)"
```

---

### Task 4: Filtro de continuidad y reporte de excluidos en `training_queries.py`

**Files:**
- Modify: `swmm_resilience/database/training_queries.py:269-310` (`load_training_samples`)
- Modify: `swmm_resilience/database/training_queries.py` (nueva función `load_continuity_excluded`, agregar después de `load_training_samples`)
- Test: `tests/database/test_training_view_v17.py`

**Interfaces:**
- Consumes: `runs.continuity_error_pct` (Task 3).
- Produces: `load_training_samples(conn, run_ids=None, max_continuity_error_pct=None) -> pd.DataFrame` — firma extendida, compatible hacia atrás (`max_continuity_error_pct=None` = sin filtro, comportamiento actual sin cambios).
- Produces: `load_continuity_excluded(conn, max_continuity_error_pct) -> pd.DataFrame` con columnas `run_id, scenario_key, shape_id, factor_mult, continuity_error_pct`.

- [ ] **Step 1: Extender `load_training_samples` con el filtro de continuidad**

En `swmm_resilience/database/training_queries.py`, modificar la firma y el cuerpo de `load_training_samples` (línea 269-310):

```python
def load_training_samples(
    conn: sqlite3.Connection,
    run_ids: list[int] | None = None,
    max_continuity_error_pct: float | None = None,
) -> pd.DataFrame:
    normalized_run_ids = _normalize_run_ids(run_ids)
    run_filter, params = _run_id_selection(normalized_run_ids)
    with _stable_read_snapshot(conn):
        _validate_selected_runs(conn, normalized_run_ids)
        _validate_run_cardinality(conn, normalized_run_ids)
        mismatches = _find_key_mismatches(conn, normalized_run_ids)
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
        columns = IDENTITY_COLUMNS + FEATURE_COLUMNS_V17 + TARGET_COLUMNS
        sql = (
            f"SELECT {', '.join(columns)} FROM training_samples_v17"
            f" WHERE 1 = 1{run_filter}{continuity_filter}"
            " ORDER BY run_id, node_id"
        )
        frame = pd.read_sql_query(sql, conn, params=params + continuity_params)
        if frame.empty:
            raise ValueError("No COMPLETE v17 training samples found")
        features = TABULAR_V3_17.validate_frame(
            frame.loc[:, FEATURE_COLUMNS_V17]
        )
        targets = _validate_targets(frame.loc[:, TARGET_COLUMNS])
        return pd.concat(
```

(El resto de la función, después de ese `return pd.concat(`, queda exactamente igual — no lo repito porque no cambia; es la continuación ya existente del archivo.)

- [ ] **Step 2: Nueva función `load_continuity_excluded`**

Inmediatamente después del cierre de `load_training_samples` (buscar dónde termina esa función — el siguiente `def` en el archivo es `export_training_samples_csv`), insertar antes de `export_training_samples_csv`:

```python
def load_continuity_excluded(
    conn: sqlite3.Connection,
    max_continuity_error_pct: float,
) -> pd.DataFrame:
    """Escenarios cuyo |continuity_error_pct| supera el umbral -- el
    'reporte separado' exigido por el spec Sec.7 para referencias
    cuestionadas. No valida cardinalidad ni COMPLETE: es un reporte de
    diagnostico, no una fuente de entrenamiento."""
    sql = """
        SELECT r.run_id, s.scenario_key, s.shape_id, s.factor_mult,
               r.continuity_error_pct
        FROM runs AS r
        JOIN scenarios AS s ON s.scenario_id = r.scenario_id
        WHERE r.continuity_error_pct IS NOT NULL
          AND ABS(r.continuity_error_pct) > ?
        ORDER BY r.run_id
    """
    return pd.read_sql_query(sql, conn, params=[max_continuity_error_pct])
```

- [ ] **Step 3: Tests del filtro**

En `tests/database/test_training_view_v17.py`, extender `_insert_scenario_and_run` (línea 76-117) para aceptar `continuity_error_pct=None` como kwarg opcional, agregándolo al INSERT:

```python
def _insert_scenario_and_run(
    conn,
    *,
    run_id,
    status="COMPLETE",
    network_id=1,
    scenario_id=None,
    node_count=1,
    continuity_error_pct=None,
):
    scenario_id = scenario_id if scenario_id is not None else 100 + run_id
    conn.execute(
        """
        INSERT INTO scenarios (
            scenario_id, network_id, scenario_key, scenario_kind,
            factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
            config_json, config_sha256
        ) VALUES (?, ?, ?, 'factor', ?, ?, ?, ?, '{}', ?)
        """,
        (
            scenario_id, network_id, f"scenario-{run_id}", 1.25,
            f"shape-{run_id}", 2.0, 0.75, f"{scenario_id:064x}",
        ),
    )
    conn.execute(
        """
        INSERT INTO runs (
            run_id, scenario_id, network_id, status, config_sha256,
            node_count, continuity_error_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, scenario_id, network_id, status, f"{run_id:064x}",
            node_count, continuity_error_pct,
        ),
    )
```

Agregar al final del archivo (reutilizando `migrated_conn`, `_insert_network`, `_insert_node`, `_insert_feature`, `_insert_result` ya definidos en el archivo — un escenario completo con dos runs, uno excluido y uno no):

```python
from swmm_resilience.database.training_queries import load_continuity_excluded


def test_load_training_samples_excludes_runs_above_continuity_threshold(migrated_conn):
    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")

    _insert_scenario_and_run(conn, run_id=1, continuity_error_pct=1.0)
    _insert_feature(conn, run_id=1, node_pk=1)
    _insert_result(conn, run_id=1, node_pk=1, inunda=1)

    _insert_scenario_and_run(conn, run_id=2, scenario_id=200, continuity_error_pct=-7.31)
    _insert_feature(conn, run_id=2, node_pk=1)
    _insert_result(conn, run_id=2, node_pk=1, inunda=1)

    frame = load_training_samples(conn, max_continuity_error_pct=5.0)
    assert set(frame["run_id"]) == {1}


def test_load_training_samples_includes_null_continuity_as_not_excluded(migrated_conn):
    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")

    _insert_scenario_and_run(conn, run_id=1, continuity_error_pct=None)
    _insert_feature(conn, run_id=1, node_pk=1)
    _insert_result(conn, run_id=1, node_pk=1, inunda=1)

    frame = load_training_samples(conn, max_continuity_error_pct=5.0)
    assert set(frame["run_id"]) == {1}


def test_load_training_samples_without_threshold_includes_everything(migrated_conn):
    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")

    _insert_scenario_and_run(conn, run_id=1, continuity_error_pct=-7.31)
    _insert_feature(conn, run_id=1, node_pk=1)
    _insert_result(conn, run_id=1, node_pk=1, inunda=1)

    frame = load_training_samples(conn)
    assert set(frame["run_id"]) == {1}


def test_load_continuity_excluded_returns_only_scenarios_above_threshold(migrated_conn):
    conn = migrated_conn
    _insert_network(conn)
    _insert_node(conn, node_pk=1, node_id="N1")

    _insert_scenario_and_run(conn, run_id=1, continuity_error_pct=1.0)
    _insert_scenario_and_run(conn, run_id=2, scenario_id=200, continuity_error_pct=-7.31)

    excluded = load_continuity_excluded(conn, max_continuity_error_pct=5.0)
    assert excluded["run_id"].tolist() == [2]
    assert excluded["continuity_error_pct"].iloc[0] == pytest.approx(-7.31)
```

- [ ] **Step 4: Correr los tests nuevos**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_training_view_v17.py -k continuity -v`
Expected: 4 PASS

- [ ] **Step 5: Correr toda la suite de `test_training_view_v17.py` para confirmar que la firma extendida no rompió nada**

Run: `venv\Scripts\python.exe -m pytest tests/database/test_training_view_v17.py -v`
Expected: todos PASS (el parámetro nuevo tiene default `None`, ningún call site existente lo pasa).

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/training_queries.py tests/database/test_training_view_v17.py
git commit -m "feat: filtrar por continuidad en load_training_samples y reportar excluidos (fase C2)"
```

---

### Task 5: Escribir el reporte de excluidos durante `--persist-sql`

**Files:**
- Modify: `main.py:930-940` (bloque `if args.persist_sql:`)
- Test: manual / smoke test dentro del mismo bloque de tests de integración de `--persist-sql` si existe uno; si no, verificación por inspección (ver Step 3).

**Interfaces:**
- Consumes: `load_continuity_excluded` (Task 4), `config.validation.max_continuity_error_pct` (Task 1).
- Produces: `continuity_excluded.csv` junto a `config.dataset.db_path`, solo si hay al menos un escenario excluido.

- [ ] **Step 1: Escribir el reporte después del backfill**

En `main.py`, dentro del bloque `if args.persist_sql:` (línea ~905 en adelante), justo después de la línea:

```python
        info = backfill_networks_and_runs(conn, df, config.network.inp_path, config.network.name)
        print(f"  network_id={info['network_id']}  nodos={len(info['node_pk_by_id'])}  "
              f"corridas={len(info['run_id_by_key'])}")
```

agregar:

```python
        from swmm_resilience.database.training_queries import load_continuity_excluded

        excluded = load_continuity_excluded(
            conn, config.validation.max_continuity_error_pct
        )
        if not excluded.empty:
            report_path = config.dataset.db_path.parent / "continuity_excluded.csv"
            excluded.to_csv(report_path, index=False)
            print(
                f"  ADVERTENCIA: {len(excluded)} escenario(s) excluidos del "
                f"benchmark principal por continuidad > "
                f"{config.validation.max_continuity_error_pct}% -- "
                f"reporte en {report_path}"
            )
```

- [ ] **Step 2: Localizar si existe un test de integración de `--persist-sql` en main.py**

Run: `grep -rn "persist_sql\|persist-sql" tests/`

Si existe un test que invoca el flujo completo de `--persist-sql` con un dataset sintético, extenderlo agregando una fila con `continuity_error_pct` fuera de umbral y verificando que `continuity_excluded.csv` se escribe. Si no existe ningún test de integración de ese flujo completo (lo más probable, dado que `--persist-sql` orquesta I/O real de archivo y DB), no crear uno nuevo aquí — la lógica que importa (el filtro y el cálculo del reporte) ya está cubierta por los tests unitarios de Task 3 y Task 4; este Step es solo para no duplicar un test de integración si ya existe uno natural donde encaje.

- [ ] **Step 3: Verificación manual mínima**

Run (en una terminal, no como test automatizado — confirma que el bloque nuevo no tiene errores de sintaxis ni de import):

```bash
venv\Scripts\python.exe -c "import main"
```

Expected: sin `ImportError`/`SyntaxError`.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: escribir continuity_excluded.csv al persistir en SQLite (fase C2)"
```

---

### Task 6: Config de reintento por extensión de simulación

**Files:**
- Modify: `swmm_resilience/config.py:255-263` (`ValidationConfig`), `:380-475` (`load_config`)
- Modify: `config.yaml:16-24` (bloque `validation:`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.validation.drainage_extension_increment_hours: float` (default `2.0`).
- Produces: `Config.validation.drainage_extension_max_retries: int` (default `2`).

- [ ] **Step 1: Agregar los dos campos a `ValidationConfig`**

En `swmm_resilience/config.py`, extender la clase (después de `max_continuity_error_pct` agregado en Task 1):

```python
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"  # "strict" | "close"
    drainage_inflow_tolerance_lps: float = 0.01
    drainage_flooding_tolerance_lps: float = 0.01
    drainage_storage_change_tolerance_m3: float = 0.01
    drainage_tail_window_hours: float = 1.0
    max_continuity_error_pct: float = 5.0
    drainage_extension_increment_hours: float = 2.0
    drainage_extension_max_retries: int = 2
```

- [ ] **Step 2: Parsear en `load_config`**

Dentro del bloque `validation=ValidationConfig(...)`, agregar después de `max_continuity_error_pct=...`:

```python
            drainage_extension_increment_hours=float(
                (raw.get("validation") or {}).get(
                    "drainage_extension_increment_hours", 2.0
                )
            ),
            drainage_extension_max_retries=int(
                (raw.get("validation") or {}).get(
                    "drainage_extension_max_retries", 2
                )
            ),
```

- [ ] **Step 3: Validar `drainage_extension_max_retries >= 0`**

El spec exige que las extensiones tengan un límite (no indefinidas) — un valor negativo no tiene sentido como "número de reintentos". Agregar, junto a las demás funciones de validación en `config.py` (cerca de `_validate_csv_inflow_closure_mode`, línea ~181), antes de `load_config`:

```python
def _validate_drainage_extension_max_retries(value: int) -> int:
    """Validate that drainage_extension_max_retries is a non-negative integer."""
    if value < 0:
        raise ValueError(
            f"validation.drainage_extension_max_retries debe ser >= 0: {value}"
        )
    return value
```

Y en `load_config`, envolver el valor parseado:

```python
    drainage_extension_max_retries = _validate_drainage_extension_max_retries(
        int((raw.get("validation") or {}).get("drainage_extension_max_retries", 2))
    )
```

(definir esta variable junto a `csv_inflow_closure_mode` más arriba en la función, antes del `return Config(...)`, y usar `drainage_extension_max_retries=drainage_extension_max_retries,` dentro de `ValidationConfig(...)` en vez del parseo inline del Step 2 para ese campo específico.)

- [ ] **Step 4: Agregar las claves a `config.yaml`**

```yaml
validation:
  drain_down_hours: 6.0
  inflow_closure_tolerance_lps: 0.01
  csv_inflow_closure_mode: "strict"
  drainage_inflow_tolerance_lps: 0.01
  drainage_flooding_tolerance_lps: 0.01
  drainage_storage_change_tolerance_m3: 0.01
  drainage_tail_window_hours: 1.0
  max_continuity_error_pct: 5.0
  drainage_extension_increment_hours: 2.0
  drainage_extension_max_retries: 2
```

- [ ] **Step 5: Tests**

En `tests/test_config.py`:

```python
def test_validation_drainage_extension_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.drainage_extension_increment_hours == 2.0
    assert config.validation.drainage_extension_max_retries == 2


def test_validation_drainage_extension_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drainage_extension_increment_hours: 1.0\n'
        '  drainage_extension_max_retries: 5',
    )
    config = load_config(cfg)
    assert config.validation.drainage_extension_increment_hours == 1.0
    assert config.validation.drainage_extension_max_retries == 5


def test_validation_drainage_extension_max_retries_rejects_negative(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  drainage_extension_max_retries: -1',
    )
    with pytest.raises(ValueError, match="drainage_extension_max_retries"):
        load_config(cfg)
```

- [ ] **Step 6: Correr los tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_config.py -k extension -v`
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/config.py config.yaml tests/test_config.py
git commit -m "feat: configuracion de reintento por extension de drenaje (fase C2)"
```

---

### Task 7: Bucle de reintento con extensión en `run_batch_shapes`

**Files:**
- Modify: `swmm_resilience/simulation/batch.py:24-66` (`_prepare_closed_scenario_inp`), `:70-163` (`run_batch_shapes`)
- Modify: `tests/simulation/test_batch_shapes_closure.py` (actualizar 5 tests existentes + agregar 3 nuevos)

**Interfaces:**
- Consumes: `config.validation.drainage_extension_increment_hours`, `config.validation.drainage_extension_max_retries` (Task 6).
- Produces: `_prepare_closed_scenario_inp(..., drain_down_hours_override: float | None = None)` — firma extendida.
- Produces: el manifiesto de drenaje (`<scenario>.drainage_state.json`) incluye una clave nueva `extension_log`: lista de `{"attempt": int, "drain_down_hours": float, "state": str}`.

- [ ] **Step 1: Extender `_prepare_closed_scenario_inp` con un override de `drain_down_hours`**

En `swmm_resilience/simulation/batch.py`, cambiar la firma y el cuerpo (línea 24-66):

```python
def _prepare_closed_scenario_inp(
    config,
    shape_id: str,
    factor: float,
    shape: "list[tuple[float, float]]",
    time_grid: "list[float]",
    base_inflows: "dict[str, float]",
    run_dir: Path,
    drain_down_hours_override: float | None = None,
) -> "tuple[Path, Path]":
    """Construye el escenario de (shape_id, factor), lo cierra, guarda el
    manifiesto y escribe el .inp. Devuelve (scenario_inp_path, rpt_path).

    Se extrajo de run_batch_shapes para que el cierre del aporte sea
    comprobable sin necesidad de simular con pyswmm de verdad.

    drain_down_hours_override, si se da, reemplaza
    config.validation.drain_down_hours -- lo usa el bucle de reintento por
    extension de run_batch_shapes para alargar la simulacion sin tocar la
    config global.
    """
    from .hydrograph_shapes import apply_shape
    from .inflow_closure import apply_closure_policy, save_closure_manifest
    from .timeseries_scenario import write_scenario_inp
    from ..validation.hydrograph_csv import HydrographScenario

    drain_down_hours = (
        drain_down_hours_override
        if drain_down_hours_override is not None
        else config.validation.drain_down_hours
    )

    node_series = apply_shape(shape, base_inflows, factor)
    scenario = HydrographScenario(
        scenario_id=f"{shape_id}_f{factor:.3f}",
        node_series=node_series,
        time_grid_hours=time_grid,
        last_time_hours=shape[-1][0],
    )
    scenario, closure_manifest = apply_closure_policy(
        scenario, config.validation.inflow_closure_tolerance_lps
    )
    closure_manifest["mode"] = "close"
    if closure_manifest["changed_nodes"]:
        warnings.warn(
            f"[{scenario.scenario_id}] Aporte cerrado a cero en los nodos: "
            f"{closure_manifest['changed_nodes']}.",
            stacklevel=2,
        )
    shape_run_dir = run_dir / shape_id
    save_closure_manifest(closure_manifest, shape_run_dir, scenario.scenario_id)
    scenario_inp = write_scenario_inp(
        config.network.inp_path, scenario, shape_run_dir,
        drain_down_hours=drain_down_hours,
    )
    return scenario_inp, scenario_inp.with_suffix(".rpt")
```

- [ ] **Step 2: Envolver la simulación + clasificación de `run_batch_shapes` en un bucle de reintento**

En `swmm_resilience/simulation/batch.py`, reemplazar el cuerpo del bucle `for factor in factors:` dentro de `run_batch_shapes` (línea ~90-163, desde `scenario_inp, rpt_path = _prepare_closed_scenario_inp(...)` hasta el final del `for factor`) por:

```python
        for factor in factors:
            i += 1
            print(
                f"  [{i:>3}/{total}] shape={shape_id:<20} factor={factor:.2f}",
                end=" ... ", flush=True,
            )

            max_retries = config.validation.drainage_extension_max_retries
            increment = config.validation.drainage_extension_increment_hours
            extension_log = []
            attempt = 0
            state = "unverified"
            drainage_manifest = {}
            scenario_inp = None
            rpt_path = None

            while True:
                current_drain_down_hours = (
                    config.validation.drain_down_hours
                    if attempt == 0
                    else config.validation.drain_down_hours + attempt * increment
                )
                scenario_inp, rpt_path = _prepare_closed_scenario_inp(
                    config, shape_id, factor, shape, time_grid, base_inflows,
                    run_dir,
                    drain_down_hours_override=(
                        None if attempt == 0 else current_drain_down_hours
                    ),
                )
                with _Simulation(str(scenario_inp)) as sim:
                    for _ in sim:
                        pass
                if not rpt_path.exists():
                    raise FileNotFoundError(f"SWMM no generó .rpt: {rpt_path}")
                out_path = scenario_inp.with_suffix(".out")
                scenario_id = scenario_inp.stem
                window_hours = min(
                    config.validation.drainage_tail_window_hours,
                    current_drain_down_hours,
                )
                try:
                    observables = read_tail_window_observables(
                        out_path, current_drain_down_hours,
                        config.validation.drainage_tail_window_hours,
                    )
                    state, drainage_manifest = classify_drainage_state(
                        observables,
                        inflow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
                        flooding_tolerance_lps=config.validation.drainage_flooding_tolerance_lps,
                        storage_change_tolerance_m3=config.validation.drainage_storage_change_tolerance_m3,
                        residual_flow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
                        window_hours=window_hours,
                    )
                except Exception as exc:
                    # Un .out ilegible/truncado/faltante no debe abortar el resto
                    # del lote -- se degrada a "unverified" (spec Sec.7) y se
                    # sigue con el resto de escenarios ya simulados.
                    state = "unverified"
                    drainage_manifest = {
                        "policy_version": DRAINAGE_STATE_POLICY_VERSION,
                        "state": "unverified",
                        "failed_checks": ["out_read_error"],
                        "error": repr(exc),
                        "n_report_steps": None,
                        "has_link_data": None,
                        "window_hours": window_hours,
                        "min_report_steps": 2,
                        "max_inflow_lps": None,
                        "max_flooding_lps": None,
                        "max_storage_change_m3": None,
                        "max_residual_flow_lps": None,
                        "tolerances": {
                            "inflow_tolerance_lps": config.validation.drainage_inflow_tolerance_lps,
                            "flooding_tolerance_lps": config.validation.drainage_flooding_tolerance_lps,
                            "storage_change_tolerance_m3": (
                                config.validation.drainage_storage_change_tolerance_m3
                            ),
                            "residual_flow_tolerance_lps": config.validation.drainage_inflow_tolerance_lps,
                        },
                    }
                    warnings.warn(
                        f"[{scenario_id}] No se pudo leer/clasificar el .out: {exc!r} — "
                        f"estado marcado unverified.",
                        stacklevel=2,
                    )

                extension_log.append({
                    "attempt": attempt,
                    "drain_down_hours": current_drain_down_hours,
                    "state": state,
                })

                if state == "drained" or attempt >= max_retries:
                    break
                attempt += 1

            drainage_manifest["extension_log"] = extension_log
            save_drainage_state_manifest(
                drainage_manifest, scenario_inp.parent, scenario_inp.stem,
            )
            results.append((shape_id, factor, rpt_path))
            print(f"OK [{state}]")
            if state != "drained":
                warnings.warn(
                    f"[{scenario_id}] Estado de drenaje: {state} "
                    f"({drainage_manifest.get('failed_checks')}) tras "
                    f"{len(extension_log)} intento(s).",
                    stacklevel=2,
                )
```

- [ ] **Step 3: Actualizar los 5 tests existentes en `test_batch_shapes_closure.py`**

Los 5 tests de `run_batch_shapes` en `tests/simulation/test_batch_shapes_closure.py` (líneas 151-375) construyen `config` como `SimpleNamespace(validation=SimpleNamespace(...))` sin los dos campos nuevos — van a fallar con `AttributeError` apenas `run_batch_shapes` los lea. En cada uno de los 5 bloques `config = SimpleNamespace(validation=SimpleNamespace(...), ...)`, agregar las dos claves:

```python
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
```

(reemplaza la línea `drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,` que aparece igual en los 5 tests: `test_run_batch_shapes_classifies_and_saves_drainage_state`, `test_run_batch_shapes_does_not_warn_when_drainage_state_is_drained`, `test_run_batch_shapes_warns_when_drainage_state_is_not_drained`, `test_run_batch_shapes_survives_an_unreadable_out_file`, y cualquier otro que comparta ese bloque).

Además, `test_run_batch_shapes_warns_when_drainage_state_is_not_drained` (línea 269-318) usa un `_fake_classify` que SIEMPRE devuelve `"not_drained"` — con el bucle de reintento, ahora se va a llamar 3 veces (`attempt` 0, 1, 2) antes de rendirse, no 1 vez. El test sigue siendo válido tal cual (el mensaje de warning sigue conteniendo `"Estado de drenaje: not_drained"`), pero agregar una aserción de que se reintentó:

```python
def test_run_batch_shapes_warns_when_drainage_state_is_not_drained(monkeypatch, tmp_path):
    """Finding 6: the classified state must not be silently discarded -- a
    non-"drained" verdict must surface as a warning. Fase C2: tambien debe
    agotar los reintentos configurados antes de rendirse."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    classify_calls = []

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        classify_calls.append(kwargs)
        return "not_drained", {
            "policy_version": "tail_window_v1", "state": "not_drained",
            "failed_checks": ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0 -> sin warning de cierre
    shapes = {"storm_notdrained": shape}

    with pytest.warns(UserWarning, match="Estado de drenaje: not_drained"):
        results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(results) == 1
    # max_retries=2 -> intentos 0, 1, 2 = 3 llamadas a classify_drainage_state
    assert len(classify_calls) == 3

    manifest_path = tmp_path / "run" / "storm_notdrained" / "storm_notdrained_f1.000.drainage_state.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["extension_log"]) == 3
    assert [entry["attempt"] for entry in payload["extension_log"]] == [0, 1, 2]
    assert payload["extension_log"][1]["drain_down_hours"] == pytest.approx(8.0)  # 6.0 + 1*2.0
    assert payload["extension_log"][2]["drain_down_hours"] == pytest.approx(10.0)  # 6.0 + 2*2.0
```

- [ ] **Step 4: Nuevo test — reintento exitoso a mitad de camino**

Agregar al final del archivo:

```python
def test_run_batch_shapes_retries_until_drained(monkeypatch, tmp_path):
    """Fase C2: si el primer intento no drena pero uno posterior si, el
    bucle debe parar ahi -- no agotar los reintentos innecesariamente."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    states = iter(["not_drained", "drained"])

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        state = next(states)
        return state, {
            "policy_version": "tail_window_v1", "state": state,
            "failed_checks": [] if state == "drained" else ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_retry_ok": shape}

    results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(results) == 1
    manifest_path = tmp_path / "run" / "storm_retry_ok" / "storm_retry_ok_f1.000.drainage_state.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["state"] == "drained"
    assert len(payload["extension_log"]) == 2  # paro en el intento 1 (drained), no llega a 2
    assert payload["extension_log"][-1]["state"] == "drained"


def test_run_batch_shapes_never_exceeds_max_retries(monkeypatch, tmp_path):
    """Fase C2: con max_retries=0 no debe haber ningun reintento -- un solo
    intento, igual que el comportamiento pre-C2."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    classify_calls = []

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        classify_calls.append(kwargs)
        return "not_drained", {
            "policy_version": "tail_window_v1", "state": "not_drained",
            "failed_checks": ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=0,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_no_retry": shape}

    with pytest.warns(UserWarning, match="Estado de drenaje: not_drained"):
        batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(classify_calls) == 1
```

- [ ] **Step 5: Correr toda la suite de `test_batch_shapes_closure.py`**

Run: `venv\Scripts\python.exe -m pytest tests/simulation/test_batch_shapes_closure.py -v`
Expected: todos PASS (5 actualizados + 2 nuevos = al menos 7 tests relacionados con `run_batch_shapes`, más los 4 tests de `_prepare_closed_scenario_inp` sin cambios).

- [ ] **Step 6: Correr la suite completa del proyecto para detectar regresiones**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: todos PASS. Si algo falla fuera de los archivos tocados en este plan, detenerse e investigar antes de continuar — no es un fallo esperado de esta tarea.

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/simulation/batch.py tests/simulation/test_batch_shapes_closure.py
git commit -m "feat: reintento con extension de drenaje en run_batch_shapes (fase C2)"
```

---

## Self-Review (completado durante la escritura de este plan)

**Cobertura del spec §7 (líneas 184-190):**
- "Un error absoluto de continuidad de routing superior a 5% marcará la referencia como cuestionada" → Tasks 1, 3, 4.
- "Ese escenario saldrá del benchmark principal" → Task 4 (`load_training_samples` filtra).
- "y aparecerá en un reporte separado" → Task 5.
- "Todas las familias usarán exactamente el mismo conjunto elegible" → satisfecho por construcción: todas las familias del banco leen de `load_training_samples()`, el único punto filtrado.
- "No se extenderá una simulación indefinidamente. Las extensiones repetidas tendrán incremento y límite configurables, ambos registrados" → Tasks 6, 7 (`drainage_extension_increment_hours`, `drainage_extension_max_retries`, `extension_log` en el manifiesto).

**Placeholders:** ninguno — cada paso de código tiene el diff completo, cada test tiene aserciones concretas con valores numéricos exactos.

**Consistencia de tipos:** `read_continuity_error(rpt_path: Path) -> float | None` se usa igual en Task 1 (definición), Task 2 (`main.py`) y Task 1-Step 2 (alias en `hydrograph_batch.py`). `load_training_samples(conn, run_ids=None, max_continuity_error_pct=None)` mantiene compatibilidad con los dos call sites existentes que no se tocan en este plan (si los hay fuera de los archivos de test ya revisados, seguirán funcionando porque el nuevo parámetro tiene default `None`).
