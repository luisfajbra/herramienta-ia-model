# Fase C1 — Estado de drenaje desde `.out` — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación (`- [ ]`).

**Goal:** clasificar cada escenario como `drained`, `not_drained` o `unverified` leyendo observables reales
del `.out` binario de SWMM en la última ventana de la simulación — nunca `END_TIME + 6h` ni profundidad
cero aislada, que es lo que el pipeline actual implícitamente asume.

**Architecture:** un lector puro envuelve `swmm_api.SwmmOutput` y extrae la ventana final (nodos: aporte,
inundación, profundidad, volumen; enlaces: flujo) con al menos dos reportes. Un clasificador puro aplica
la regla operativa del spec sobre esos observables y tolerancias configurables. Se integra al mismo punto
donde ya se generan `.rpt`/`.out` (`run_batch_shapes` de fase B1), persistiendo un manifiesto sidecar por
escenario — mismo patrón que `closure_manifest.json`.

**Tech Stack:** Python 3.11, `swmm_api` (ya en uso para `.inp`, ahora también para `.out`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §7. Este plan
cubre la clasificación `drained`/`not_drained`/`unverified`; la exclusión de escenarios por continuidad
>5% (`ya parcialmente implementada como aviso en hydrograph_batch.py: CONTINUITY_ERROR_WARN_PCT`) y la
extensión configurable de simulaciones quedan para un plan C2 separado.

## Global Constraints

- Nunca se declara `drained` solo por `END_TIME + drain_down_hours` ni por profundidad cero aislada — el
  estado sale de observables reales del `.out`, con al menos dos reportes en la ventana final.
- Si el `.out` solo ofrece volumen nodal (sin datos de enlace, aporte o inundación), el estado es
  `unverified` — nunca se declara estabilidad ni vaciado con datos incompletos.
- Las tolerancias son explícitas en `config.yaml` y aparecen en el manifiesto de cada escenario.
- **Ruling (estado basal simplificado).** El spec exige comparar el flujo residual de nodos/enlaces
  contra "su estado basal". Calcular un basal real (una corrida de referencia del mismo `.inp` sin
  aporte externo) es una decisión de ingeniería que excede el alcance de una sola tarea de este plan —
  requeriría su propio piloto. Este plan usa **cero** como basal (residual = |flujo final| directamente,
  sin restar una corrida de referencia): es el caso degenerado correcto para una red de tormenta sin
  caudal base permanente, y dado que el basal exacto se congelará en un piloto físico junto con las demás
  tolerancias (spec §7, "congelada en el piloto"), esta simplificación es honesta con el estado actual del
  proyecto y no falsea el resultado — subestima ligeramente el residual real si la red tuviera un caudal
  base propio, lo que solo puede hacer que un escenario se marque `drained` de forma optimista, nunca
  pesimista. **Costo si es incorrecto:** un basal real futuro es un cambio de una función (Task 2, no de
  las demás); ningún dato persistido cambia de forma, solo el criterio de aprobación.
- **Ruling (tolerancias por defecto, pendientes de piloto).** `0.01` L/s para aporte/inundación residual
  (mismo valor que la tolerancia de cierre de fase B1, por consistencia), `0.01` m³ para cambio de
  almacenamiento entre reportes consecutivos, ventana final `min(1h, drain_down_hours)`, mínimo 2
  reportes. El spec exige explícitamente que las tolerancias físicas se congelen en un piloto — estos son
  valores conservadores de arranque, no el resultado de ese piloto. **Costo si son incorrectos:** son
  cuatro números en `config.yaml`; ningún escenario pierde su manifiesto si cambian, solo se re-clasifica.
- No hacer commit fuera de las tareas de este plan; se ejecuta en la misma rama/worktree que ya trae fase
  A, la búsqueda de hiperparámetros y fases B1-B2 (`worktree-optuna-tuning-familia-ganadora`).

---

## Mapa de archivos

- `swmm_resilience/validation/drainage_state.py` (nuevo): lector de `.out` y clasificador puro.
- `tests/validation/test_drainage_state.py` (nuevo).
- `swmm_resilience/config.py`: `ValidationConfig` gana los cuatro campos de tolerancia/ventana.
- `config.yaml`: bloque `validation:` gana esas cuatro claves.
- `swmm_resilience/simulation/batch.py`: `_prepare_closed_scenario_inp`'s caller (`run_batch_shapes`)
  clasifica y persiste el estado tras cada corrida real.
- `tests/simulation/test_batch_shapes_closure.py` (existente, de fase B1): se extiende con la nueva
  aserción de wiring.

## Interfaces que produce el plan

```python
# validation/drainage_state.py
DRAINAGE_STATE_POLICY_VERSION: str = "tail_window_v1"

def read_tail_window_observables(
    out_path: Path, drain_down_hours: float, tail_window_hours: float = 1.0,
) -> dict: ...
# {"n_report_steps": int, "has_link_data": bool,
#  "node_total_inflow_lps": {node_id: [v, ...]}, "node_flooding_lps": {node_id: [v, ...]},
#  "node_volume_m3": {node_id: [v, ...]}, "link_flow_lps": {link_id: [v, ...]}}

def classify_drainage_state(
    observables: dict,
    inflow_tolerance_lps: float, flooding_tolerance_lps: float,
    storage_change_tolerance_m3: float, residual_flow_tolerance_lps: float,
    min_report_steps: int = 2,
) -> tuple[str, dict]: ...  # (state in {"drained","not_drained","unverified"}, diagnostics dict)

def save_drainage_state_manifest(manifest: dict, out_dir: Path, scenario_id: str) -> Path: ...

# config.py
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"
    drainage_inflow_tolerance_lps: float = 0.01
    drainage_flooding_tolerance_lps: float = 0.01
    drainage_storage_change_tolerance_m3: float = 0.01
    drainage_tail_window_hours: float = 1.0
```

---

### Task 1: Lector de la ventana final del `.out`

**Files:**
- Crear: `swmm_resilience/validation/drainage_state.py`
- Test: `tests/validation/test_drainage_state.py`

**Consume:** `swmm_api.SwmmOutput` (dependencia ya instalada, usada hoy solo para `.inp`).

**Produce:** `read_tail_window_observables()` — abre un `.out`, calcula la ventana final
`min(tail_window_hours, drain_down_hours)`, y devuelve los observables de nodo/enlace en esa ventana con
al menos los reportes disponibles ahí.

- [ ] **Paso 1: escribir las pruebas usando un `.out` real minúsculo.**

Antes de escribir el test, generar un `.out` real de pocos pasos correndo SWMM sobre un `.inp` mínimo (2-3
nodos, unos pocos minutos, report_step corto) — esto es más confiable que intentar construir un `.out`
binario sintético a mano. Usar el patrón ya existente en el repo para escribir un `.inp` de prueba y
correr `pyswmm.Simulation` (ver `swmm_resilience/simulation/batch.py::run_batch_shapes` para el patrón de
invocación), guardando el `.out` resultante en un fixture de `tests/validation/` reutilizable entre tests
(por ejemplo, generarlo una vez en un fixture de módulo `@pytest.fixture(scope="module")` que corre SWMM
sobre un `.inp` sintético de 2 nodos y limpia después).

Crear `tests/validation/test_drainage_state.py`:

```python
"""Tests for swmm_resilience.validation.drainage_state, against a real .out file."""

from __future__ import annotations

from pathlib import Path

import pytest

from swmm_resilience.validation.drainage_state import read_tail_window_observables


@pytest.fixture(scope="module")
def tiny_out_file(tmp_path_factory):
    """Corre SWMM sobre un .inp minusculo y devuelve la ruta al .out resultante."""
    from pyswmm import Simulation

    out_dir = tmp_path_factory.mktemp("drainage_state_out")
    inp_path = out_dir / "tiny.inp"
    inp_path.write_text(
        """
[TITLE]
tiny network for drainage_state tests

[OPTIONS]
FLOW_UNITS           LPS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             00:20:00
REPORT_STEP          00:01:00
WET_STEP             00:01:00
DRY_STEP             00:01:00
ROUTING_STEP         0:00:30

[JUNCTIONS]
;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
J1      0.0        5.0       0.0        0.0       0
J2      -1.0       5.0       0.0        0.0       0

[OUTFALLS]
;;Name  Elevation  Type      StageData  Gated
OUT1    -2.0       FREE                 NO

[CONDUITS]
;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset
C1      J1        J2      100     0.01       0          0
C2      J2        OUT1    100     0.01       0          0

[XSECTIONS]
;;Link  Shape       Geom1  Geom2  Geom3  Geom4
C1      CIRCULAR    0.5    0      0      0
C2      CIRCULAR    0.5    0      0      0

[INFLOWS]
;;Node  Constituent  TimeSeries  Type   Mfactor  Sfactor  Baseline  Pattern
J1      FLOW         inflow1     FLOW   1.0      1.0

[TIMESERIES]
;;Name    Date  Time     Value
inflow1         0:00     5.0
inflow1         0:05     5.0
inflow1         0:06     0.0
inflow1         0:20     0.0

[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS  NONE
NODES      ALL
LINKS      ALL

[TAGS]
""",
        encoding="utf-8",
    )

    with Simulation(str(inp_path)) as sim:
        for _ in sim:
            pass

    out_path = inp_path.with_suffix(".out")
    assert out_path.exists(), "SWMM no genero el .out esperado"
    return out_path


def test_read_tail_window_observables_has_at_least_two_report_steps(tiny_out_file):
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=0.25, tail_window_hours=1.0)
    assert result["n_report_steps"] >= 2


def test_read_tail_window_observables_uses_the_shorter_of_tail_window_and_drain_down(tiny_out_file):
    # drain_down_hours=0.05h (3 min) es mas corto que tail_window_hours=1.0 -> la ventana
    # efectiva son los ultimos 3 minutos, unos pocos reportes de 1 min cada uno.
    short = read_tail_window_observables(tiny_out_file, drain_down_hours=0.05, tail_window_hours=1.0)
    long = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert short["n_report_steps"] <= long["n_report_steps"]


def test_read_tail_window_observables_includes_all_expected_series(tiny_out_file):
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert result["has_link_data"] is True
    assert set(result["node_total_inflow_lps"]) == {"J1", "J2"}
    assert set(result["node_flooding_lps"]) == {"J1", "J2"}
    assert set(result["node_volume_m3"]) == {"J1", "J2"}
    assert set(result["link_flow_lps"]) == {"C1", "C2"}
    for series in result["node_total_inflow_lps"].values():
        assert len(series) == result["n_report_steps"]


def test_read_tail_window_observables_flow_settles_toward_zero_at_the_end(tiny_out_file):
    """El inflow del .inp de prueba termina en 0.0 desde el minuto 6; a los 20 min
    (fin de simulacion) el aporte de J1 en la ventana final debe ser ~0."""
    result = read_tail_window_observables(tiny_out_file, drain_down_hours=1.0, tail_window_hours=1.0)
    assert result["node_total_inflow_lps"]["J1"][-1] == pytest.approx(0.0, abs=0.5)
```

(El valor `abs=0.5` en la última prueba es una tolerancia generosa: el objetivo es demostrar que el lector
trae datos reales y coherentes con el escenario simulado, no fijar un número exacto de SWMM que dependería
de la versión del solver.)

- [ ] **Paso 2: ejecutar la prueba y comprobar que falla por ausencia del módulo.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/validation/test_drainage_state.py -q`

Esperado: `ModuleNotFoundError: No module named 'swmm_resilience.validation.drainage_state'`

- [ ] **Paso 3: implementar el lector.**

Crear `swmm_resilience/validation/drainage_state.py`:

```python
"""Lectura de la ventana final del .out y clasificacion del estado de drenaje.

Antes de este modulo, el pipeline asumia que una simulacion habia drenado
solo por haber corrido END_TIME + drain_down_hours, o por profundidad cero
aislada en un unico paso -- ninguna de las dos demuestra drenaje real (spec
Sec.7 del diseno de cierre de auditoria). Este modulo lee observables reales
del .out binario de SWMM (nunca antes leido en este proyecto -- todo el resto
del pipeline usa el .rpt de texto) en la ultima ventana de la simulacion, con
al menos dos reportes, y aplica la regla operativa del spec: ausencia de
aporte e inundacion sobre tolerancia, cambio de almacenamiento estabilizado,
y flujo residual de nodos y enlaces dentro de tolerancia.
"""

from __future__ import annotations

from pathlib import Path


def read_tail_window_observables(
    out_path: Path | str,
    drain_down_hours: float,
    tail_window_hours: float = 1.0,
) -> dict:
    """Lee nodos y enlaces en la ventana final de `out_path`.

    La ventana es `min(tail_window_hours, drain_down_hours)` horas antes del
    final de la simulacion. Devuelve un dict con `n_report_steps`,
    `has_link_data`, y las series por objeto de aporte/inundacion/volumen
    (nodos) y flujo (enlaces).
    """
    from swmm_api import SwmmOutput

    window_hours = min(tail_window_hours, drain_down_hours)

    with SwmmOutput(str(out_path)) as out:
        end_time = out.index[-1]
        start_time = end_time - __import__("datetime").timedelta(hours=window_hours)

        node_inflow = out.get_part(kind="node", variable="total_inflow", start=start_time, end=end_time)
        node_flooding = out.get_part(kind="node", variable="flooding", start=start_time, end=end_time)
        node_volume = out.get_part(kind="node", variable="volume", start=start_time, end=end_time)
        try:
            link_flow = out.get_part(kind="link", variable="flow", start=start_time, end=end_time)
            has_link_data = link_flow is not None and not link_flow.empty
        except Exception:
            link_flow = None
            has_link_data = False

    n_report_steps = len(node_inflow.index) if node_inflow is not None else 0

    return {
        "n_report_steps": n_report_steps,
        "has_link_data": has_link_data,
        "node_total_inflow_lps": {
            str(col): node_inflow[col].tolist() for col in node_inflow.columns
        },
        "node_flooding_lps": {
            str(col): node_flooding[col].tolist() for col in node_flooding.columns
        },
        "node_volume_m3": {
            str(col): node_volume[col].tolist() for col in node_volume.columns
        },
        "link_flow_lps": (
            {str(col): link_flow[col].tolist() for col in link_flow.columns}
            if has_link_data else {}
        ),
    }
```

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/validation/test_drainage_state.py -q`

Esperado: 4 passed. Si `SwmmOutput.get_part(...)`'s exact keyword names (`start`/`end`, o el nombre de la
variable `"total_inflow"`/`"flooding"`/`"volume"`/`"flow"`) no coinciden con la versión de `swmm_api`
instalada, el error de `TypeError`/`KeyError` lo dirá explícitamente — ajustar a la firma real reportada
por el propio error, no adivinar; `swmm_api.output_file.definitions.NODE_VARIABLES`/`LINK_VARIABLES` (ya
explorados: `NODE_VARIABLES` tiene `DEPTH, FLOODING, HEAD, LATERAL_INFLOW, TOTAL_INFLOW, VOLUME`;
`LINK_VARIABLES` tiene `CAPACITY, DEPTH, FLOW, VELOCITY, VOLUME`) confirman que los nombres de variable en
minúsculas (`"total_inflow"`, `"flooding"`, `"volume"`, `"flow"`) son válidos según la ayuda de
`SwmmOutput.get_part` ya leída al escribir este plan.

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/validation/drainage_state.py tests/validation/test_drainage_state.py
git commit -m "feat: lector de la ventana final del .out (fase C1)"
```

---

### Task 2: Clasificador puro del estado de drenaje

**Files:**
- Modificar: `swmm_resilience/validation/drainage_state.py`
- Modificar: `tests/validation/test_drainage_state.py`

**Consume:** el dict que produce `read_tail_window_observables()` (Task 1) como entrada — este task no
depende de `.out` real, opera sobre dicts construidos a mano en los tests.

**Produce:** `classify_drainage_state()` y `save_drainage_state_manifest()`.

- [ ] **Paso 1: escribir las pruebas con observables sintéticos.**

Añadir a `tests/validation/test_drainage_state.py`:

```python
import json

from swmm_resilience.validation.drainage_state import (
    DRAINAGE_STATE_POLICY_VERSION,
    classify_drainage_state,
    save_drainage_state_manifest,
)

_TOL = dict(
    inflow_tolerance_lps=0.01,
    flooding_tolerance_lps=0.01,
    storage_change_tolerance_m3=0.01,
    residual_flow_tolerance_lps=0.01,
)


def _obs(**overrides):
    base = {
        "n_report_steps": 3,
        "has_link_data": True,
        "node_total_inflow_lps": {"A": [0.0, 0.0, 0.0]},
        "node_flooding_lps": {"A": [0.0, 0.0, 0.0]},
        "node_volume_m3": {"A": [1.0, 1.0, 1.0]},
        "link_flow_lps": {"L1": [0.0, 0.0, 0.0]},
    }
    base.update(overrides)
    return base


def test_classify_drained_when_everything_is_within_tolerance():
    state, diag = classify_drainage_state(_obs(), **_TOL, min_report_steps=2)
    assert state == "drained"
    assert diag["policy_version"] == DRAINAGE_STATE_POLICY_VERSION


def test_classify_not_drained_when_inflow_exceeds_tolerance():
    obs = _obs(node_total_inflow_lps={"A": [0.0, 0.0, 5.0]})
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "not_drained"
    assert "inflow" in diag["failed_checks"]


def test_classify_not_drained_when_flooding_exceeds_tolerance():
    obs = _obs(node_flooding_lps={"A": [0.0, 0.2, 0.0]})
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "not_drained"
    assert "flooding" in diag["failed_checks"]


def test_classify_not_drained_when_storage_change_between_steps_exceeds_tolerance():
    obs = _obs(node_volume_m3={"A": [1.0, 5.0, 5.0]})  # salto de 4.0 m3 entre pasos
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "not_drained"
    assert "storage_change" in diag["failed_checks"]


def test_classify_not_drained_when_residual_link_flow_exceeds_tolerance():
    obs = _obs(link_flow_lps={"L1": [0.0, 0.0, 3.0]})
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "not_drained"
    assert "residual_flow" in diag["failed_checks"]


def test_classify_unverified_when_too_few_report_steps():
    obs = _obs(n_report_steps=1)
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "unverified"
    assert "insufficient_report_steps" in diag["failed_checks"]


def test_classify_unverified_when_link_data_is_missing():
    """Spec: si el .out solo ofrece volumen nodal, nunca se declara drenaje."""
    obs = _obs(has_link_data=False, link_flow_lps={})
    state, diag = classify_drainage_state(obs, **_TOL, min_report_steps=2)
    assert state == "unverified"
    assert "missing_link_data" in diag["failed_checks"]


def test_save_drainage_state_manifest_writes_readable_json(tmp_path):
    _, diag = classify_drainage_state(_obs(), **_TOL, min_report_steps=2)
    path = save_drainage_state_manifest(diag, tmp_path, "scenario_1")
    assert path.name == "scenario_1.drainage_state.json"
    assert json.loads(path.read_text(encoding="utf-8")) == diag
```

- [ ] **Paso 2: ejecutar y comprobar que fallan por `ImportError`.**

- [ ] **Paso 3: implementar el clasificador.**

Añadir a `swmm_resilience/validation/drainage_state.py`:

```python
import json

DRAINAGE_STATE_POLICY_VERSION = "tail_window_v1"


def classify_drainage_state(
    observables: dict,
    inflow_tolerance_lps: float,
    flooding_tolerance_lps: float,
    storage_change_tolerance_m3: float,
    residual_flow_tolerance_lps: float,
    min_report_steps: int = 2,
) -> tuple[str, dict]:
    """Aplica la regla operativa del spec Sec.7 sobre `observables`.

    "drained" exige: suficientes reportes, datos de enlace presentes,
    aporte e inundacion bajo tolerancia en toda la ventana, cambio de
    almacenamiento estabilizado entre pasos consecutivos, y flujo residual
    de nodos y enlaces bajo tolerancia (basal = cero, ver Global
    Constraints de este plan). Cualquier dato insuficiente -> "unverified",
    nunca "drained" ni "not_drained" por omision.
    """
    failed_checks: list[str] = []

    if observables["n_report_steps"] < min_report_steps:
        failed_checks.append("insufficient_report_steps")
    if not observables["has_link_data"]:
        failed_checks.append("missing_link_data")

    if failed_checks:
        state = "unverified"
    else:
        if any(
            abs(v) > inflow_tolerance_lps
            for series in observables["node_total_inflow_lps"].values()
            for v in series
        ):
            failed_checks.append("inflow")
        if any(
            abs(v) > flooding_tolerance_lps
            for series in observables["node_flooding_lps"].values()
            for v in series
        ):
            failed_checks.append("flooding")
        if any(
            abs(series[i + 1] - series[i]) > storage_change_tolerance_m3
            for series in observables["node_volume_m3"].values()
            for i in range(len(series) - 1)
        ):
            failed_checks.append("storage_change")
        if any(
            abs(v) > residual_flow_tolerance_lps
            for series in observables["link_flow_lps"].values()
            for v in series
        ):
            failed_checks.append("residual_flow")

        state = "drained" if not failed_checks else "not_drained"

    diagnostics = {
        "policy_version": DRAINAGE_STATE_POLICY_VERSION,
        "state": state,
        "failed_checks": failed_checks,
        "n_report_steps": observables["n_report_steps"],
        "has_link_data": observables["has_link_data"],
        "tolerances": {
            "inflow_tolerance_lps": inflow_tolerance_lps,
            "flooding_tolerance_lps": flooding_tolerance_lps,
            "storage_change_tolerance_m3": storage_change_tolerance_m3,
            "residual_flow_tolerance_lps": residual_flow_tolerance_lps,
        },
    }
    return state, diagnostics


def save_drainage_state_manifest(manifest: dict, out_dir: Path | str, scenario_id: str) -> Path:
    """Escribe `manifest` en `<out_dir>/<scenario_id>.drainage_state.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.drainage_state.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

Añadir `import json` y mantener `from pathlib import Path` ya presente de Task 1 en el bloque de imports
del archivo (no duplicar el import de `Path`).

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/validation/test_drainage_state.py -q`

Esperado: los 4 tests de Task 1 más los 8 de Task 2, todos en verde (12 en total).

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/validation/drainage_state.py tests/validation/test_drainage_state.py
git commit -m "feat: clasificador puro del estado de drenaje (fase C1)"
```

---

### Task 3: Configuración de tolerancias

**Files:**
- Modificar: `swmm_resilience/config.py`
- Modificar: `config.yaml`
- Modificar: `tests/test_config.py`

**Consume:** nada de las tareas anteriores directamente.

**Produce:** los cuatro campos nuevos en `ValidationConfig`, con los defaults del Global Constraint de
este plan.

- [ ] **Paso 1: escribir las pruebas.**

`tests/test_config.py`'s `write_config()` no incluye bloque `validation:` (confirmado en el plan de fase
B1, ya en este branch). Añadir, siguiendo el mismo patrón que B1 ya estableció en este archivo
(`_write_config_with_validation_block`, ya presente):

```python
def test_validation_drainage_state_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.validation.drainage_inflow_tolerance_lps == 0.01
    assert config.validation.drainage_flooding_tolerance_lps == 0.01
    assert config.validation.drainage_storage_change_tolerance_m3 == 0.01
    assert config.validation.drainage_tail_window_hours == 1.0


def test_validation_drainage_state_is_parsed_when_present(tmp_path):
    cfg = _write_config_with_validation_block(
        tmp_path,
        '  drainage_inflow_tolerance_lps: 0.05\n'
        '  drainage_flooding_tolerance_lps: 0.05\n'
        '  drainage_storage_change_tolerance_m3: 0.1\n'
        '  drainage_tail_window_hours: 2.0',
    )
    config = load_config(cfg)
    assert config.validation.drainage_inflow_tolerance_lps == 0.05
    assert config.validation.drainage_flooding_tolerance_lps == 0.05
    assert config.validation.drainage_storage_change_tolerance_m3 == 0.1
    assert config.validation.drainage_tail_window_hours == 2.0
```

- [ ] **Paso 2: ejecutar y comprobar que fallan** (`AttributeError` sobre los campos nuevos).

- [ ] **Paso 3: extender `ValidationConfig`.**

En `swmm_resilience/config.py`, `ValidationConfig` (ya tiene `drain_down_hours`,
`inflow_closure_tolerance_lps`, `csv_inflow_closure_mode` de fase B1):

```python
@dataclass
class ValidationConfig:
    drain_down_hours: float = 6.0
    inflow_closure_tolerance_lps: float = 0.01
    csv_inflow_closure_mode: str = "strict"
    drainage_inflow_tolerance_lps: float = 0.01
    drainage_flooding_tolerance_lps: float = 0.01
    drainage_storage_change_tolerance_m3: float = 0.01
    drainage_tail_window_hours: float = 1.0
```

Extender el bloque de `load_config()` que construye `ValidationConfig(...)` (mismo patrón que las tres
claves de fase B1, buscando `(raw.get("validation") or {}).get(...)` como referencia exacta) para leer
las cuatro claves nuevas con esos defaults. No hace falta un `min_report_steps` configurable en esta
tarea — el plan lo deja como argumento de `classify_drainage_state()` con default `2` (Task 2), sin
exponerlo todavía en `config.yaml`; añadirlo después es un cambio de una línea si se necesita.

En `config.yaml`, dentro del bloque `validation:` ya existente:

```yaml
validation:
  drain_down_hours: 6.0
  inflow_closure_tolerance_lps: 0.01
  csv_inflow_closure_mode: "strict"
  drainage_inflow_tolerance_lps: 0.01
  drainage_flooding_tolerance_lps: 0.01
  drainage_storage_change_tolerance_m3: 0.01
  drainage_tail_window_hours: 1.0
```

- [ ] **Paso 4: ejecutar y comprobar verde.**

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/config.py config.yaml tests/test_config.py
git commit -m "feat: tolerancias de estado de drenaje en config (fase C1)"
```

---

### Task 4: Wiring en `run_batch_shapes`

**Files:**
- Modificar: `swmm_resilience/simulation/batch.py`
- Modificar: `tests/simulation/test_batch_shapes_closure.py`

**Consume:** `read_tail_window_observables`, `classify_drainage_state`, `save_drainage_state_manifest`
(Tasks 1-2); los cuatro campos de `config.validation.drainage_*` (Task 3).

**Produce:** `run_batch_shapes` clasifica y persiste el estado de drenaje de cada escenario justo después
de confirmar que el `.rpt` existe.

- [ ] **Paso 1: escribir la prueba de wiring.**

`tests/simulation/test_batch_shapes_closure.py` (de fase B1) prueba `_prepare_closed_scenario_inp`
directamente, sin `pyswmm.Simulation` real. Este wiring vive en `run_batch_shapes` (la función que SÍ
llama a `pyswmm.Simulation`), así que su prueba necesita mockear `_Simulation` — seguir el patrón de mock
descrito en la Task 5 del plan de fase B1 (`monkeypatch.setattr(pyswmm, "Simulation", FakeSimulation)`,
con `FakeSimulation` creando también un `.rpt`/`.out` falsos junto al `.inp` en `__enter__`). Dado que este
plan no puede fijar el contenido exacto de un `.out` falso sin haber leído primero cómo
`read_tail_window_observables` (Task 1) se comporta ante un archivo inválido, la prueba de esta tarea NO
intenta un `.out` binario real: en su lugar, monkeypatchea `swmm_resilience.simulation.batch.classify_drainage_state`
y `swmm_resilience.simulation.batch.read_tail_window_observables` para verificar SOLO el wiring (que se
llaman con los argumentos correctos y que el resultado se persiste), dejando la prueba de la lectura real
del `.out` para la Task 5.

Añadir a `tests/simulation/test_batch_shapes_closure.py`:

```python
def test_run_batch_shapes_classifies_and_saves_drainage_state(monkeypatch, tmp_path):
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    calls = {}

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        calls["read_args"] = (out_path, drain_down_hours, tail_window_hours)
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        calls["classify_kwargs"] = kwargs
        return "drained", {"policy_version": "tail_window_v1", "state": "drained", "failed_checks": []}

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
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.01,
            drainage_storage_change_tolerance_m3=0.01, drainage_tail_window_hours=1.0,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_z": shape}

    batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert calls["read_args"][1] == 6.0  # drain_down_hours
    assert calls["read_args"][2] == 1.0  # drainage_tail_window_hours
    assert calls["classify_kwargs"]["inflow_tolerance_lps"] == 0.01

    manifest_path = tmp_path / "run" / "storm_z" / "storm_z_f1.000.drainage_state.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["state"] == "drained"
```

Añadir `import json`, `from pathlib import Path`, `from types import SimpleNamespace` a los imports del
archivo de test si no están ya presentes (Task 2 de fase B1 ya los usa en este mismo archivo — revisar el
bloque de imports existente antes de duplicar).

- [ ] **Paso 2: ejecutar y comprobar que falla** (el wiring todavía no existe).

- [ ] **Paso 3: añadir el wiring en `run_batch_shapes`.**

En `swmm_resilience/simulation/batch.py`, añadir al bloque de imports (junto a los que ya trae de fase B1
para `apply_closure_policy`/`save_closure_manifest`):

```python
from ..validation.drainage_state import (
    classify_drainage_state, read_tail_window_observables, save_drainage_state_manifest,
)
```

Dentro de `run_batch_shapes`, justo después del bloque existente:

```python
            if not rpt_path.exists():
                raise FileNotFoundError(f"SWMM no generó .rpt: {rpt_path}")
```

y antes de `results.append(...)`, insertar:

```python
            out_path = scenario_inp.with_suffix(".out")
            observables = read_tail_window_observables(
                out_path, config.validation.drain_down_hours,
                config.validation.drainage_tail_window_hours,
            )
            _, drainage_manifest = classify_drainage_state(
                observables,
                inflow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
                flooding_tolerance_lps=config.validation.drainage_flooding_tolerance_lps,
                storage_change_tolerance_m3=config.validation.drainage_storage_change_tolerance_m3,
                residual_flow_tolerance_lps=config.validation.drainage_inflow_tolerance_lps,
            )
            save_drainage_state_manifest(
                drainage_manifest, scenario_inp.parent, scenario_inp.stem,
            )
```

(`residual_flow_tolerance_lps` reutiliza `drainage_inflow_tolerance_lps` en esta tarea — ambos son
tolerancias de flujo residual en L/s bajo el mismo Global Constraint por defecto; separarlas en
`config.yaml` es un cambio de una clave si un piloto futuro decide que deben diferir.)

`scenario_inp.parent` es `shape_run_dir` (ya construido por `_prepare_closed_scenario_inp`);
`scenario_inp.stem` es el `scenario_id` (`f"{shape_id}_f{factor:.3f}"`, ya confirmado en fase B1).

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/simulation/test_batch_shapes_closure.py -q`

Esperado: todos los tests de este archivo (los 2 de fase B1 más el nuevo de esta tarea) en verde.

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/simulation/batch.py tests/simulation/test_batch_shapes_closure.py
git commit -m "feat: clasificar y persistir el estado de drenaje en run_batch_shapes (fase C1)"
```

---

### Task 5: Verificación de fase C1

**Files:**
- No crear archivos de producción adicionales.

**Consume:** las cuatro tareas anteriores.

**Produce:** evidencia de que la suite completa sigue verde y de que el mecanismo funciona contra un
`.out` real de la red del proyecto (no el `.inp` minúsculo sintético de Task 1).

- [ ] **Paso 1: suite completa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests -q`

Esperado: código 0, sin regresiones sobre el estado anterior a este plan.

- [ ] **Paso 2: smoke real contra la red del proyecto.**

Reutilizar el mismo shape real ya usado en la verificación de fases B1/B2
(`data/hydrograph_shapes/long_moderate.csv`, factor=1.0) y correr `run_batch_shapes` para exactamente esa
combinación contra la red real (`config.yaml`'s `network.inp_path`), en un directorio temporal. Confirmar:
- se genera `<scenario_id>.drainage_state.json` junto al `.inp`/`.rpt`/`.out` reales;
- el manifiesto tiene una de las tres claves de estado válidas (`drained`, `not_drained`, `unverified`) y,
  si no es `drained`, `failed_checks` explica por qué;
- si el estado es `unverified`, confirmar la razón (típicamente `has_link_data` si la red no reporta
  enlaces, o pocos reportes si `drain_down_hours` es muy corto) — cualquier resultado es válido aquí,
  este smoke prueba que el mecanismo corre sobre datos reales sin excepción, no que el resultado sea
  `drained`.

No correr el resto de formas/factores ni regenerar ningún dataset — fuera de alcance de este plan.

- [ ] **Paso 3: diff y estado de git.**

Ejecutar `git diff --check` y `git status --short`. Esperado: sin errores de espacios; los cambios
corresponden solo al mapa de archivos de este plan.

## Autorrevisión del plan

- **Cobertura:** spec §7 (estados `drained`/`not_drained`/`unverified`, ventana final con al menos dos
  reportes, nunca `END_TIME+6h` ni profundidad aislada, tolerancias explícitas en config y manifiesto,
  `unverified` cuando falta el dato de enlace) queda cubierta por las Tasks 1-4; la exclusión por
  continuidad >5% y la extensión configurable de simulaciones quedan explícitamente para un plan C2
  separado (Global Constraints); el basal real (no cero) queda como ruling documentado, no como
  pendiente silencioso.
- **Consistencia:** `read_tail_window_observables`, `classify_drainage_state`,
  `save_drainage_state_manifest`, `DRAINAGE_STATE_POLICY_VERSION` y los cuatro campos nuevos de
  `ValidationConfig` se usan con el mismo nombre y firma en todas las tareas que los consumen.
- **Sin placeholders:** cada paso trae código completo; Task 1's `.inp` sintético está completo y
  ejecutable, no descrito en abstracto.
