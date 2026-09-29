# Fase B1 — Política de cierre del aporte — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación (`- [ ]`).

**Goal:** separar fin-de-aporte de fin-de-simulación en toda generación de hidrograma (formas generadas y
CSV de usuario), cerrando linealmente a cero cualquier aporte que quede positivo, en vez de usar el
colchón de drenaje para cerrarlo implícitamente.

**Architecture:** un módulo nuevo, puro y sin dependencias de SWMM, decide si una serie ya está cerrada y,
si no, le añade un punto de cierre en el siguiente paso de malla. Los dos productores de escenarios
(`batch.run_batch_shapes` para formas generadas, `hydrograph_batch` para CSV de usuario) aplican esa
política ANTES de llamar a `write_scenario_inp`, que no cambia: ya solo recibe series cerradas. Cada
corrida registra un manifiesto de cierre junto al `.inp` escrito.

**Tech Stack:** Python 3.11, pytest. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §6 (primera
mitad: cierre del aporte). §6 completo también cubre el contrato de features nuevo — eso es la Fase B2,
un plan separado; este plan no lo toca.

## Global Constraints

- Nunca se altera una serie de CSV subida por el usuario en silencio: el modo estricto (default) rechaza
  con `ValueError`; el modo de cierre solo se activa si `config.yaml` lo pide explícitamente.
- El colchón de drenaje (`drain_down_hours`) nunca vuelve a usarse para cerrar gradualmente un aporte
  positivo — `write_scenario_inp` no cambia en este plan porque ya no necesita hacerlo: sus llamadores
  siempre le pasan una serie ya cerrada.
- La tolerancia de cierre se expresa en L/s y es explícita en `config.yaml`
  (`validation.inflow_closure_tolerance_lps`), nunca un porcentaje del pico.
- **Ruling (tolerancia por defecto):** `0.01` L/s. Es un valor conservador (mucho menor que cualquier
  `base_inflow_lps` real de la red) elegido porque no hay evidencia previa que fije uno mejor; el spec
  exige que sea explícito y versionado, no que sea óptimo desde el día uno. Costo si es incorrecto: hay
  que ajustar un solo número en `config.yaml`, sin tocar código, y las corridas antiguas quedan
  identificadas por su propio manifiesto de cierre (que registra qué tolerancia se usó).
- **Ruling (persistencia del manifiesto):** JSON junto al `.inp` de cada escenario (mismo patrón que
  `PreparedDataset.manifest`), no una tabla SQL nueva. Fase B2 decidirá si el manifiesto de cierre entra
  a la identidad del dataset nuevo cuando regenere la verdad SWMM; este plan no reabre el esquema de base
  de datos. Costo si es incorrecto: reprocesar los JSON a SQL es mecánico, no hay pérdida de información.
- **Fuera de alcance, deliberadamente:** volver a simular el histórico completo (175 combinaciones
  forma×factor) con la política nueva. Este plan construye y prueba el mecanismo, y corre un smoke de UNA
  sola simulación real para probar que funciona end-to-end — no relanza el dataset completo. Esa decisión
  de cómputo queda para cuando Fase B2 (contrato de features) esté lista, así la red solo se resimula una
  vez para ambos cambios.
- No hacer commit fuera de las tareas de este plan; se ejecuta en la misma rama/worktree que ya trae fase
  A y la búsqueda de hiperparámetros (`worktree-optuna-tuning-familia-ganadora`), construyendo sobre ella.

---

## Mapa de archivos

- `swmm_resilience/simulation/inflow_closure.py` (nuevo): política de cierre pura — `close_series_to_zero`,
  `apply_closure_policy`, `enforce_strict_zero_endpoint`, `save_closure_manifest`.
- `swmm_resilience/config.py`: `ValidationConfig` gana `inflow_closure_tolerance_lps` y
  `csv_inflow_closure_mode`.
- `config.yaml`: bloque `validation:` gana las dos claves nuevas.
- `swmm_resilience/simulation/batch.py`: `run_batch_shapes` aplica la política antes de escribir el `.inp`.
- `swmm_resilience/validation/hydrograph_batch.py`: el bucle de validación aplica la política (estricta o
  de cierre, según config) antes de escribir el `.inp`.
- `tests/simulation/test_inflow_closure.py` (nuevo), `tests/test_bench_config.py`-style tests en
  `tests/test_config.py`, `tests/simulation/test_batch_shapes.py` (o el archivo de tests de batch que ya
  exista), `tests/validation/test_hydrograph_batch.py` (o su equivalente).

## Interfaces que produce el plan

```python
# inflow_closure.py
INFLOW_CLOSURE_POLICY_VERSION: str = "linear_v1"

def close_series_to_zero(
    series: list[tuple[float, float]], tolerance_lps: float,
) -> tuple[list[tuple[float, float]], bool]: ...

def apply_closure_policy(
    scenario: HydrographScenario, tolerance_lps: float,
) -> tuple[HydrographScenario, dict]: ...

def enforce_strict_zero_endpoint(
    scenario: HydrographScenario, tolerance_lps: float,
) -> None: ...  # raises ValueError, never modifies

def save_closure_manifest(manifest: dict, out_dir: Path, scenario_id: str) -> Path: ...

# config.py
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"  # "strict" | "close"

# simulation/batch.py -- new helper, extracted from run_batch_shapes so the
# closure wiring is testable without mocking pyswmm.Simulation
def _prepare_closed_scenario_inp(
    config, shape_id: str, factor: float, shape: list[tuple[float, float]],
    time_grid: list[float], base_inflows: dict[str, float], run_dir: Path,
) -> tuple[Path, Path]: ...  # (scenario_inp_path, rpt_path)

# validation/hydrograph_batch.py -- run_batch_validation gains two params
def run_batch_validation(
    ..., csv_inflow_closure_mode: str = "strict",
    inflow_closure_tolerance_lps: float = 0.01,
) -> dict: ...
```

---

### Task 1: Política de cierre pura

**Files:**
- Crear: `swmm_resilience/simulation/inflow_closure.py`
- Test: `tests/simulation/test_inflow_closure.py`
- Modificar: `swmm_resilience/config.py`, `config.yaml`
- Test: `tests/test_config.py`

**Consume:** `HydrographScenario` (`swmm_resilience/validation/hydrograph_csv.py`, ya existente, sin
cambios).

**Produce:** las cuatro funciones de `inflow_closure.py` listadas arriba; `ValidationConfig` con los dos
campos nuevos.

- [ ] **Paso 1: escribir las pruebas de la política pura.**

Crear `tests/simulation/test_inflow_closure.py`:

```python
import json

import pytest

from swmm_resilience.simulation.inflow_closure import (
    INFLOW_CLOSURE_POLICY_VERSION,
    apply_closure_policy,
    close_series_to_zero,
    enforce_strict_zero_endpoint,
    save_closure_manifest,
)
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


def test_close_series_to_zero_leaves_an_already_closed_series_untouched():
    series = [(0.0, 5.0), (1.0, 2.0), (2.0, 0.0)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert closed == series
    assert changed is False


def test_close_series_to_zero_appends_one_point_at_the_next_grid_step():
    series = [(0.0, 5.0), (1.0, 8.0), (2.0, 3.0)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert changed is True
    assert closed[:-1] == series
    assert closed[-1] == (3.0, 0.0)  # step = 2.0 - 1.0 = 1.0


def test_close_series_to_zero_uses_the_last_interval_as_the_step():
    series = [(0.0, 5.0), (1.5, 8.0), (2.0, 3.0)]  # irregular grid, last step = 0.5
    closed, _ = close_series_to_zero(series, tolerance_lps=0.01)
    assert closed[-1] == (2.5, 0.0)


def test_close_series_to_zero_treats_a_value_within_tolerance_as_already_closed():
    series = [(0.0, 5.0), (1.0, 0.005)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert changed is False
    assert closed == series


def test_close_series_to_zero_rejects_a_series_with_fewer_than_two_points():
    with pytest.raises(ValueError, match="2 puntos"):
        close_series_to_zero([(0.0, 5.0)], tolerance_lps=0.01)


def _scenario(node_series, time_grid_hours=None, last_time_hours=None):
    grid = time_grid_hours or sorted({t for s in node_series.values() for t, _ in s})
    last = last_time_hours if last_time_hours is not None else max(grid)
    return HydrographScenario(
        scenario_id="s1", node_series=node_series,
        time_grid_hours=grid, last_time_hours=last,
    )


def test_apply_closure_policy_closes_every_node_that_needs_it():
    scenario = _scenario({
        "A": [(0.0, 5.0), (1.0, 3.0)],   # needs closing
        "B": [(0.0, 5.0), (1.0, 0.0)],   # already closed
    })
    closed_scenario, manifest = apply_closure_policy(scenario, tolerance_lps=0.01)

    assert closed_scenario.node_series["A"][-1] == (2.0, 0.0)
    assert closed_scenario.node_series["B"] == scenario.node_series["B"]
    assert closed_scenario.last_time_hours == 2.0
    assert manifest["changed_nodes"] == ["A"]
    assert manifest["policy_version"] == INFLOW_CLOSURE_POLICY_VERSION
    assert manifest["tolerance_lps"] == 0.01
    assert manifest["source_scenario_id"] == "s1"
    assert manifest["original_last_time_hours"] == 1.0
    assert manifest["effective_last_time_hours"] == 2.0


def test_apply_closure_policy_is_a_no_op_when_nothing_needs_closing():
    scenario = _scenario({"A": [(0.0, 5.0), (1.0, 0.0)]})
    closed_scenario, manifest = apply_closure_policy(scenario, tolerance_lps=0.01)

    assert closed_scenario is scenario
    assert manifest["changed_nodes"] == []
    assert manifest["effective_last_time_hours"] == manifest["original_last_time_hours"]


def test_apply_closure_policy_never_mutates_the_input_scenario():
    original_series = [(0.0, 5.0), (1.0, 3.0)]
    scenario = _scenario({"A": list(original_series)})
    apply_closure_policy(scenario, tolerance_lps=0.01)
    assert scenario.node_series["A"] == original_series


def test_enforce_strict_zero_endpoint_accepts_an_already_closed_scenario():
    scenario = _scenario({"A": [(0.0, 5.0), (1.0, 0.0)]})
    enforce_strict_zero_endpoint(scenario, tolerance_lps=0.01)  # must not raise


def test_enforce_strict_zero_endpoint_rejects_and_never_modifies():
    original_series = [(0.0, 5.0), (1.0, 3.0)]
    scenario = _scenario({"A": list(original_series)})
    with pytest.raises(ValueError, match="A"):
        enforce_strict_zero_endpoint(scenario, tolerance_lps=0.01)
    assert scenario.node_series["A"] == original_series


def test_save_closure_manifest_writes_readable_json(tmp_path):
    manifest = {
        "policy_version": INFLOW_CLOSURE_POLICY_VERSION, "tolerance_lps": 0.01,
        "source_scenario_id": "s1", "changed_nodes": ["A"],
        "original_last_time_hours": 1.0, "effective_last_time_hours": 2.0,
    }
    path = save_closure_manifest(manifest, tmp_path, "s1")
    assert path.name == "s1.closure_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
```

- [ ] **Paso 2: ejecutar la prueba y comprobar que falla por ausencia del módulo.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/simulation/test_inflow_closure.py -q`

Esperado: `ModuleNotFoundError: No module named 'swmm_resilience.simulation.inflow_closure'`

- [ ] **Paso 3: implementar `inflow_closure.py`.**

```python
"""Política de cierre del aporte: separa fin-de-aporte de fin-de-simulación.

El fin del aporte (cuándo deja de entrar agua a la red) y el fin de la
simulación (cuánto tiempo se deja drenar después) son conceptos distintos.
Antes de este módulo, un aporte generado o subido por el usuario que
terminaba en un valor positivo se "cerraba" implícitamente: write_scenario_inp
insertaba un único punto en cero al final del colchón de drenaje, así que
SWMM interpolaba linealmente TODO ese colchón desde el último valor positivo
hasta cero -- el drenaje terminaba cerrando el aporte en vez de solo drenar
la red ya cerrada (spec §6 del diseño de cierre de auditoría).

Este módulo decide, antes de escribir ningún .inp, si cada serie de nodo ya
termina en (casi) cero. Si no, añade UN punto de cierre en el siguiente paso
de malla -- un cierre lineal rápido, versionado y explícito -- y deja que el
colchón de drenaje empiece ya en cero. Nunca sustituye ese colchón: solo dejar
de necesitarlo para cerrar el aporte.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..validation.hydrograph_csv import HydrographScenario

INFLOW_CLOSURE_POLICY_VERSION = "linear_v1"


def close_series_to_zero(
    series: list[tuple[float, float]], tolerance_lps: float,
) -> tuple[list[tuple[float, float]], bool]:
    """Cierra ``series`` a cero si su último valor supera ``tolerance_lps``.

    Si ``|series[-1][1]| <= tolerance_lps`` la serie ya está cerrada: se
    devuelve intacta y ``changed=False``. En otro caso se añade un único
    punto nuevo en ``series[-1][0] + paso``, valor 0.0, donde ``paso`` es el
    intervalo entre los dos últimos puntos de ``series``. Nunca trunca,
    reordena ni modifica los puntos existentes.
    """
    if len(series) < 2:
        raise ValueError(
            "La serie necesita al menos 2 puntos para inferir el paso de malla"
        )
    last_t, last_v = series[-1]
    if abs(last_v) <= tolerance_lps:
        return list(series), False
    step = last_t - series[-2][0]
    if step <= 0:
        raise ValueError(f"Paso de malla no positivo entre los dos últimos puntos: {step}")
    return list(series) + [(last_t + step, 0.0)], True


def apply_closure_policy(
    scenario: HydrographScenario, tolerance_lps: float,
) -> tuple[HydrographScenario, dict]:
    """Aplica ``close_series_to_zero`` a cada nodo de ``scenario``.

    Devuelve ``(escenario_efectivo, manifest)``. Si ningún nodo necesitó
    cierre, ``escenario_efectivo is scenario`` (mismo objeto, sin copiar) y
    el manifest documenta que no hubo cambios. ``scenario`` nunca se muta.
    """
    changed_nodes: list[str] = []
    new_node_series: dict[str, list[tuple[float, float]]] = {}
    new_last_time = scenario.last_time_hours

    for node_id, series in scenario.node_series.items():
        closed, changed = close_series_to_zero(series, tolerance_lps)
        new_node_series[node_id] = closed
        if changed:
            changed_nodes.append(node_id)
            new_last_time = max(new_last_time, closed[-1][0])

    manifest = {
        "policy_version": INFLOW_CLOSURE_POLICY_VERSION,
        "tolerance_lps": tolerance_lps,
        "source_scenario_id": scenario.scenario_id,
        "changed_nodes": sorted(changed_nodes),
        "original_last_time_hours": scenario.last_time_hours,
        "effective_last_time_hours": new_last_time,
    }

    if not changed_nodes:
        return scenario, manifest

    new_time_grid = sorted(set(scenario.time_grid_hours) | {new_last_time})
    closed_scenario = HydrographScenario(
        scenario_id=scenario.scenario_id,
        node_series=new_node_series,
        time_grid_hours=new_time_grid,
        last_time_hours=new_last_time,
    )
    return closed_scenario, manifest


def enforce_strict_zero_endpoint(scenario: HydrographScenario, tolerance_lps: float) -> None:
    """Rechaza ``scenario`` si algún nodo no termina en (casi) cero.

    Modo estricto para CSV de usuario: nunca modifica la serie. No se
    infiere ningún cierre; el usuario corrige el CSV o pide explícitamente
    el modo de cierre (``apply_closure_policy``).
    """
    bad = sorted(
        node_id for node_id, series in scenario.node_series.items()
        if abs(series[-1][1]) > tolerance_lps
    )
    if bad:
        raise ValueError(
            f"El escenario '{scenario.scenario_id}' tiene caudal final distinto de cero "
            f"(> {tolerance_lps} L/s) en los nodos: {bad}. Modo estricto (default): corrige "
            "el CSV para que termine en cero, o configura "
            "validation.csv_inflow_closure_mode: \"close\" para cerrarlo automáticamente."
        )


def save_closure_manifest(manifest: dict, out_dir: Path | str, scenario_id: str) -> Path:
    """Escribe ``manifest`` en ``<out_dir>/<scenario_id>.closure_manifest.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.closure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

Añadir a `swmm_resilience/config.py`, en `ValidationConfig`:

```python
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"  # "strict" | "close"
```

En el bloque `_parse_config`/`load_config` donde se construye `ValidationConfig(...)` (ver
`swmm_resilience/config.py` alrededor de la línea 417, junto a `drain_down_hours`):

```python
        validation=ValidationConfig(
            drain_down_hours=float(
                (raw.get("validation") or {}).get("drain_down_hours", 6.0)
            ),
            inflow_closure_tolerance_lps=float(
                (raw.get("validation") or {}).get("inflow_closure_tolerance_lps", 0.01)
            ),
            csv_inflow_closure_mode=str(
                (raw.get("validation") or {}).get("csv_inflow_closure_mode", "strict")
            ),
        ),
```

Añadir validación: si `csv_inflow_closure_mode` no está en `{"strict", "close"}`, `raise ValueError` con
un mensaje que nombre las dos opciones válidas — buscar el patrón de validación ya usado para
`bench.protocols` en el mismo archivo y seguirlo.

En `config.yaml`, dentro del bloque `validation:` existente (que hoy solo tiene `drain_down_hours`):

```yaml
validation:
  drain_down_hours: 6.0
  inflow_closure_tolerance_lps: 0.01
  csv_inflow_closure_mode: "strict"
```

- [ ] **Paso 4: pruebas de configuración.**

`tests/test_config.py` ya tiene un helper `write_config(tmp_path, inp_name="network.inp",
factor_step=0.2)` que escribe un `config.yaml` mínimo SIN bloque `validation:` (así que
`test_load_config_resolves_paths_and_factors` y las demás pruebas existentes de ese archivo ya ejercitan
los defaults de `ValidationConfig` implícitamente). Añadir, al final de `tests/test_config.py`:

```python
def test_validation_inflow_closure_defaults(tmp_path):
    """write_config() no incluye bloque validation: -- deben aplicar los defaults."""
    config = load_config(write_config(tmp_path))
    assert config.validation.inflow_closure_tolerance_lps == 0.01
    assert config.validation.csv_inflow_closure_mode == "strict"


def _write_config_with_validation_block(tmp_path: Path, validation_yaml: str) -> Path:
    """Como write_config(), pero añade un bloque validation: al final del YAML."""
    cfg = write_config(tmp_path)
    text = cfg.read_text(encoding="utf-8")
    cfg.write_text(text + f"\nvalidation:\n{validation_yaml}\n", encoding="utf-8")
    return cfg


def test_validation_inflow_closure_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drain_down_hours: 6.0\n  inflow_closure_tolerance_lps: 0.05\n'
        '  csv_inflow_closure_mode: "close"',
    )
    config = load_config(cfg)
    assert config.validation.inflow_closure_tolerance_lps == 0.05
    assert config.validation.csv_inflow_closure_mode == "close"


def test_validation_rejects_an_unknown_csv_inflow_closure_mode(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path, '  csv_inflow_closure_mode: "auto"',
    )
    with pytest.raises(ValueError, match="strict.*close|close.*strict"):
        load_config(cfg)
```

`write_config()`'s YAML ya termina con el bloque `visualization:`; añadir un bloque `validation:` nuevo
al final del texto es válido YAML siempre que no exista ya uno (no existe, se comprobó arriba).

- [ ] **Paso 5: ejecutar todo y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/simulation/test_inflow_closure.py tests/test_config.py -q`

Esperado: todos pasan.

- [ ] **Paso 6: commit.**

```bash
git add swmm_resilience/simulation/inflow_closure.py tests/simulation/test_inflow_closure.py swmm_resilience/config.py config.yaml tests/test_config.py
git commit -m "feat: politica de cierre lineal del aporte (fase B1)"
```

---

### Task 2: Cerrar formas generadas antes de simular

**Files:**
- Modificar: `swmm_resilience/simulation/batch.py`
- Test: crear `tests/simulation/test_batch_shapes_closure.py`

**Consume:** `apply_closure_policy`, `save_closure_manifest` (Task 1);
`config.validation.inflow_closure_tolerance_lps`, `config.validation.drain_down_hours` (Task 1).

**Produce:** `_prepare_closed_scenario_inp()`, una función nueva extraída de `run_batch_shapes` para que
el cierre sea comprobable sin necesidad de simular con `pyswmm` de verdad. `run_batch_shapes` pasa a
llamarla en vez de repetir su lógica inline.

`swmm_resilience/simulation/batch.py` no tiene tests propios hoy (se comprobó: no existe ningún
`tests/simulation/test_batch*.py`), así que esta tarea también es la primera cobertura de esta función.

- [ ] **Paso 1: escribir las pruebas.**

Crear `tests/simulation/test_batch_shapes_closure.py`:

```python
"""Tests for the inflow-closure wiring inside swmm_resilience.simulation.batch."""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

from swmm_resilience.simulation import batch as batch_mod
from swmm_resilience.simulation import timeseries_scenario as ts_mod


class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self):
        self._s = {
            "INFLOWS": {("1C", "FLOW"): _FakeInflow("1C")},
            "TIMESERIES": {"1C": _FakeTs([(0.0, 0.0)])},
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


def _make_config(tmp_path):
    return SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
    )


def test_prepare_closed_scenario_inp_closes_a_shape_that_does_not_end_at_zero(
    monkeypatch, tmp_path,
):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.3)]  # q_norm no termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    scenario_inp, rpt_path = batch_mod._prepare_closed_scenario_inp(
        config, "storm_x", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
    )

    data = fake_inp["TIMESERIES"]["1C"].data
    assert data[2] == (0.1, 0.6)     # ultimo punto original: 0.3 * 2.0 * 1.0
    assert data[3] == (0.15, 0.0)    # punto de cierre, paso de malla 0.05
    assert data[4] == (6.15, 0.0)    # punto de drenaje, plano en cero
    assert len(data) == 5
    assert rpt_path == scenario_inp.with_suffix(".rpt")

    manifest_path = tmp_path / "run" / "storm_x" / "storm_x_f1.000.closure_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["changed_nodes"] == ["1C"]


def test_prepare_closed_scenario_inp_leaves_an_already_closed_shape_untouched(
    monkeypatch, tmp_path,
):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    batch_mod._prepare_closed_scenario_inp(
        config, "storm_y", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
    )

    data = fake_inp["TIMESERIES"]["1C"].data
    assert data[2] == (0.1, 0.0)   # sin punto de cierre añadido
    assert data[3] == (6.1, 0.0)   # solo el punto de drenaje
    assert len(data) == 4

    manifest_path = tmp_path / "run" / "storm_y" / "storm_y_f1.000.closure_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["changed_nodes"] == []
```

- [ ] **Paso 2: ejecutar y comprobar que falla.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/simulation/test_batch_shapes_closure.py -q`

Esperado: `AttributeError: module 'swmm_resilience.simulation.batch' has no attribute
'_prepare_closed_scenario_inp'`.

- [ ] **Paso 3: extraer `_prepare_closed_scenario_inp()` y aplicar la política en `run_batch_shapes`.**

Reemplazar el contenido de `swmm_resilience/simulation/batch.py` desde `def run_batch_shapes` hasta el
final del archivo por:

```python
def _prepare_closed_scenario_inp(
    config,
    shape_id: str,
    factor: float,
    shape: "list[tuple[float, float]]",
    time_grid: "list[float]",
    base_inflows: "dict[str, float]",
    run_dir: Path,
) -> "tuple[Path, Path]":
    """Construye el escenario de (shape_id, factor), lo cierra, guarda el
    manifiesto y escribe el .inp. Devuelve (scenario_inp_path, rpt_path).

    Se extrajo de run_batch_shapes para que el cierre del aporte sea
    comprobable sin necesidad de simular con pyswmm de verdad.
    """
    from .hydrograph_shapes import apply_shape
    from .inflow_closure import apply_closure_policy, save_closure_manifest
    from .timeseries_scenario import write_scenario_inp
    from ..validation.hydrograph_csv import HydrographScenario

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
    shape_run_dir = run_dir / shape_id
    save_closure_manifest(closure_manifest, shape_run_dir, scenario.scenario_id)
    scenario_inp = write_scenario_inp(
        config.network.inp_path, scenario, shape_run_dir,
        drain_down_hours=config.validation.drain_down_hours,
    )
    return scenario_inp, scenario_inp.with_suffix(".rpt")


def run_batch_shapes(
    config: Config,
    shapes: "dict[str, list[tuple[float, float]]]",
    base_inflows: "dict[str, float]",
    run_dir: Path,
) -> "list[tuple[str, float, Path]]":
    """Run SWMM for every (shape, factor) combination.

    shapes: {shape_id: [(time_h, q_norm), ...]}
    base_inflows: {node_id: base_inflow_lps}
    Returns list of (shape_id, factor, rpt_path).
    """
    from pyswmm import Simulation as _Simulation

    factors = config.factors()
    total = len(shapes) * len(factors)
    results = []
    i = 0

    for shape_id, shape in shapes.items():
        time_grid = [t for t, _ in shape]

        for factor in factors:
            i += 1
            print(
                f"  [{i:>3}/{total}] shape={shape_id:<20} factor={factor:.2f}",
                end=" ... ", flush=True,
            )
            scenario_inp, rpt_path = _prepare_closed_scenario_inp(
                config, shape_id, factor, shape, time_grid, base_inflows, run_dir,
            )
            with _Simulation(str(scenario_inp)) as sim:
                for _ in sim:
                    pass
            if not rpt_path.exists():
                raise FileNotFoundError(f"SWMM no generó .rpt: {rpt_path}")
            results.append((shape_id, factor, rpt_path))
            print("OK")

    return results
```

No cambia nada antes de `def run_batch_shapes` en el archivo (el import de `run_simulation_simple` y
`run_batch` arriba quedan intactos).

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/simulation/test_batch_shapes_closure.py -q`

Esperado: 2 passed.

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/simulation/batch.py tests/simulation/test_batch_shapes_closure.py
git commit -m "feat: cerrar formas generadas antes de escribir el .inp (fase B1)"
```

---

### Task 3: Cerrar o rechazar CSV de usuario según el modo configurado

**Files:**
- Modificar: `swmm_resilience/validation/hydrograph_batch.py`
- Modificar: `main.py` (los 4 call sites de `run_batch_validation`)
- Test: `tests/test_hydrograph_batch.py`

**Consume:** `apply_closure_policy`, `enforce_strict_zero_endpoint`, `save_closure_manifest` (Task 1).

**Produce:** `run_batch_validation()` gana dos parámetros (`csv_inflow_closure_mode`,
`inflow_closure_tolerance_lps`) y aplica la política antes de escribir el `.inp` de cada escenario CSV.

**Contexto ya verificado en este archivo (`tests/test_hydrograph_batch.py`):** ya existen los helpers
`_make_csv`, `_patch_all` (monkeypatchea `load_scenario` para devolver siempre `_FAKE_SCENARIO`, y
`write_scenario_inp` para devolver un `.inp`/`.rpt` falsos sin inspeccionar el escenario recibido) y `_run`
(llama a `run_batch_validation` con valores por defecto salvo los `**kwargs` que se le pasen).
`_FAKE_SCENARIO` ya termina en valores no nulos (`J1: (1.0, 50.0)`, `J2: (1.0, 80.0)`) — es exactamente el
caso que este plan necesita ejercitar, sin tener que construir un escenario nuevo.

- [ ] **Paso 1: escribir las pruebas, en `tests/test_hydrograph_batch.py`.**

Añadir `import json` a los imports del archivo si no está. Añadir, después de las pruebas de
`TestRunBatchValidationNScenarios`:

```python
def test_strict_mode_rejects_a_csv_that_does_not_end_at_zero(monkeypatch, tmp_path):
    """csv_inflow_closure_mode='strict' (default) aborta con un mensaje claro
    y nunca llega a escribir un .inp."""
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    _patch_all(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="J1|J2"):
        _run(tmp_path, csv_dir, csv_inflow_closure_mode="strict")


def test_close_mode_closes_the_scenario_before_writing_the_inp(monkeypatch, tmp_path):
    """csv_inflow_closure_mode='close' cierra el escenario y deja un manifiesto
    junto al .inp, en vez de rechazarlo."""
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    _make_csv(csv_dir)
    fake_inp, _ = _patch_all(monkeypatch, tmp_path)

    received: dict = {}

    def _capture_write_scenario_inp(base_inp_path, scenario, out_dir, drain_down_hours=6.0):
        received["scenario"] = scenario
        return fake_inp

    monkeypatch.setattr(hydrograph_batch, "write_scenario_inp", _capture_write_scenario_inp)

    _run(tmp_path, csv_dir, csv_inflow_closure_mode="close")

    closed = received["scenario"]
    assert closed.node_series["J1"][-1] == (2.0, 0.0)  # paso de malla 1.0-0.0=1.0
    assert closed.node_series["J2"][-1] == (2.0, 0.0)

    manifest_path = tmp_path / "out" / "inp" / "sc001.closure_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sorted(payload["changed_nodes"]) == ["J1", "J2"]
```

- [ ] **Paso 2: ejecutar y comprobar que fallan.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/test_hydrograph_batch.py -k closure -q`

Esperado: `TypeError: run_batch_validation() got an unexpected keyword argument 'csv_inflow_closure_mode'`
en ambas.

- [ ] **Paso 3: extender `run_batch_validation()`.**

En `swmm_resilience/validation/hydrograph_batch.py`, añadir al bloque de imports (junto a
`from ..simulation.timeseries_scenario import write_scenario_inp`):

```python
from ..simulation.inflow_closure import (
    apply_closure_policy, enforce_strict_zero_endpoint, save_closure_manifest,
)
```

Extender la firma de `run_batch_validation` (línea 216), añadiendo los dos parámetros nuevos al final:

```python
def run_batch_validation(
    csv_dir: Path,
    base_inp_path: Path,
    clf_path: Path,
    reg_path: Path,
    flood_threshold_m3: float,
    out_dir: Path,
    expected_nodes: set[str] | None = None,
    drain_down_hours: float = 6.0,
    allow_inp_mismatch: bool = False,
    factor_range: tuple[float, float] | None = None,
    csv_inflow_closure_mode: str = "strict",
    inflow_closure_tolerance_lps: float = 0.01,
) -> dict:
```

Documentar los dos parámetros nuevos en el docstring, junto a `drain_down_hours`, con el mismo estilo.

En el bucle (línea ~295), insertar entre `load_scenario` y `write_scenario_inp`:

```python
    for csv_path in csv_files:
        # Step a: load scenario
        scenario = load_scenario(csv_path, expected_nodes)

        # Step a.5: enforce/apply the inflow closure policy
        if csv_inflow_closure_mode == "close":
            scenario, closure_manifest = apply_closure_policy(
                scenario, inflow_closure_tolerance_lps
            )
            save_closure_manifest(closure_manifest, inp_dir, scenario.scenario_id)
        else:
            enforce_strict_zero_endpoint(scenario, inflow_closure_tolerance_lps)

        # Step b: write scenario-specific .inp (with drain-down buffer)
        t0 = time.perf_counter()
        scenario_inp_path = write_scenario_inp(
            base_inp_path, scenario, inp_dir, drain_down_hours=drain_down_hours
        )
```

- [ ] **Paso 4: propagar los dos parámetros nuevos desde `main.py`.**

En los 4 call sites de `run_batch_validation(` en `main.py` (localizarlos con
`grep -n "run_batch_validation(" main.py`), añadir a cada uno, junto a la línea
`drain_down_hours=config.validation.drain_down_hours,` (o, en el cuarto call site —
`--evaluate-hydrographs` — junto a `drain_down_hours=drain_down_hours,`):

```python
                csv_inflow_closure_mode=config.validation.csv_inflow_closure_mode,
                inflow_closure_tolerance_lps=config.validation.inflow_closure_tolerance_lps,
```

(usar la indentación exacta de cada call site; los cuatro ya usan `config.validation.drain_down_hours` o
una variable local `drain_down_hours` derivada de `config.validation.drain_down_hours` — confirmar esto
último en el cuarto call site antes de escribir la línea, leyendo el código unas líneas arriba de esa
llamada).

- [ ] **Paso 5: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/test_hydrograph_batch.py -q`

Esperado: todos los tests de este archivo en verde, incluidos los preexistentes.

- [ ] **Paso 6: commit.**

```bash
git add swmm_resilience/validation/hydrograph_batch.py main.py tests/test_hydrograph_batch.py
git commit -m "feat: aplicar politica de cierre a CSV de usuario (fase B1)"
```

---

### Task 4: Regresión end-to-end del cierre (sin SWMM real)

**Files:**
- Modificar: `tests/test_timeseries_scenario.py`

**Consume:** `apply_closure_policy` (Task 1); `write_scenario_inp` (ya existente, sin cambios de código
en este plan).

**Produce:** una prueba que encadena `apply_closure_policy` → `write_scenario_inp` usando el mismo
`_FakeInp` que ya usa este archivo, y verifica que la `TIMESERIES` final tiene el punto de cierre correcto
seguido de un tramo plano en cero durante el colchón de drenaje — la garantía completa que exige el spec,
en un solo test rápido sin SWMM.

- [ ] **Paso 1: escribir la prueba.**

```python
def test_a_closed_scenario_produces_a_flat_zero_drain_tail(monkeypatch, tmp_path):
    """apply_closure_policy + write_scenario_inp: el cierre queda en un solo
    paso de malla, y el colchon de drenaje se mantiene plano en cero despues
    -- nunca una rampa larga desde un valor positivo."""
    from swmm_resilience.simulation.inflow_closure import apply_closure_policy

    unclosed = HydrographScenario(
        scenario_id="storm_x_f1.000",
        node_series={"1C": [(0.0, 0.6), (0.05, 0.973)]},  # no termina en cero
        time_grid_hours=[0.0, 0.05],
        last_time_hours=0.05,
    )
    closed, manifest = apply_closure_policy(unclosed, tolerance_lps=0.01)
    assert manifest["changed_nodes"] == ["1C"]

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(
        Path("fake.inp"), closed, tmp_path, drain_down_hours=6.0,
    )

    data = fake_inp["TIMESERIES"]["1C"].data
    # El cierre inserta (0.1, 0.0) -- paso 0.05 tras el ultimo punto real.
    assert data[2] == (0.1, 0.0)
    # El drenaje añade un punto plano en cero al final del colchon.
    assert data[-1] == (0.1 + 6.0, 0.0)
    # Ningun punto entre el cierre y el final del drenaje queda por encima
    # de la tolerancia: el tramo es plano, no una rampa.
    assert all(v == 0.0 for _, v in data[2:])
```

`_FakeInp` (definida al inicio de este archivo) ya trae por defecto `TIMESERIES["1C"]` e
`INFLOWS[("1C","FLOW")]` referenciando esa misma serie, así que no hace falta construirla con parámetros
distintos para este test — el `write_scenario_inp` real sobrescribe `fake_inp["TIMESERIES"]["1C"].data`
con la serie de `closed`, que es lo que la prueba verifica.

- [ ] **Paso 2: ejecutar y comprobar que pasa** (este test no debería fallar por RED/GREEN de una
  implementación nueva — `write_scenario_inp` no cambia en este plan; el test existe para fijar como
  regresión permanente el comportamiento correcto de principio a fin). Si falla, es evidencia de que el
  wiring de las Tasks 1-3 tiene un defecto, no que falte código nuevo aquí: investigar antes de tocar
  nada.

Ejecutar: `venv\Scripts\python.exe -m pytest tests/test_timeseries_scenario.py -q`

Esperado: pasa, junto con todos los tests preexistentes de este archivo.

- [ ] **Paso 3: commit.**

```bash
git add tests/test_timeseries_scenario.py
git commit -m "test: regresion end-to-end del cierre de aporte (fase B1)"
```

---

### Task 5: Verificación de fase B1

**Files:**
- No crear archivos de producción adicionales.

**Consume:** las cuatro tareas anteriores.

**Produce:** evidencia de que la suite completa sigue verde y de que el mecanismo funciona contra SWMM
real en un único escenario (no el histórico completo).

- [ ] **Paso 1: suite completa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests -q`

Esperado: código 0, sin regresiones sobre el estado anterior a este plan.

- [ ] **Paso 2: smoke de UNA simulación real con un shape que necesite cierre.**

Elegir (o construir, si ninguna de las formas de `data/hydrograph_shapes/` lo necesita ya) un shape cuyo
último `q_norm` no sea 0, correr `run_batch_shapes` para exactamente esa forma con un solo factor, y
confirmar:
- el `.inp` generado simula sin error en SWMM real;
- el `.closure_manifest.json` junto a él tiene `changed_nodes` no vacío;
- el `.rpt` resultante no muestra un salto de caudal extraño en el tramo de drenaje (revisar
  manualmente el resumen de `Node Flooding` o el continuity error, igual que ya hace
  `hydrograph_batch.py` para otras corridas).

No correr el resto de formas ni factores — eso es una decisión de cómputo explícitamente fuera de
alcance de este plan (ver Global Constraints).

- [ ] **Paso 3: diff y estado de git.**

Ejecutar `git diff --check` y `git status --short`. Esperado: sin errores de espacios; los cambios
corresponden solo al mapa de archivos de este plan.

## Autorrevisión del plan

- **Cobertura:** spec §6 primera mitad (cierre del aporte, los dos modos para CSV de usuario, tolerancia
  explícita en L/s, manifiesto por corrida, regeneración de verdad SWMM) queda cubierta por las Tasks 1-4;
  la regeneración del histórico completo queda explícitamente diferida (Global Constraints) porque
  depende de una decisión de cómputo del usuario, no de código faltante.
- **Consistencia:** `apply_closure_policy`, `enforce_strict_zero_endpoint`, `save_closure_manifest`,
  `INFLOW_CLOSURE_POLICY_VERSION` y los dos campos nuevos de `ValidationConfig` se usan con el mismo
  nombre y firma en las cuatro tareas que los consumen.
- **Sin placeholders:** todas las fixtures reales que consumen las Tasks 2-4 (`_patch_all`/`_run` de
  `tests/test_hydrograph_batch.py`, `_FakeInp`/`_FakeTs`/`_FakeInflow` de `tests/test_timeseries_scenario.py`,
  `write_config()` de `tests/test_config.py`, la firma real de `run_batch_validation` y sus 4 call sites en
  `main.py`) se leyeron antes de escribir este plan; cada paso trae código completo y ejecutable, no una
  descripción de qué probar.
