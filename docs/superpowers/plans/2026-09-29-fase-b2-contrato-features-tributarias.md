# Fase B2 — Contrato de features tributarias — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación (`- [ ]`).

**Goal:** calcular seis features nuevas por nodo a partir de `Q_tributario` (la serie del nodo más sus
ancestros, sobre la serie efectivamente enviada a SWMM tras el cierre de Fase B1), declarar un contrato
de features nuevo e incompatible con el actual de 17, y usar la misma función para entrenamiento e
inferencia.

**Architecture:** un módulo puro nuevo calcula `Q_tributario` (suma de series sobre una malla común, con
interpolación lineal explícita) y las seis features (integrales y duraciones con cortes exactos en los
cruces de umbral). Ese módulo no sabe de SWMM ni de bases de datos. Un contrato nuevo (`contracts.py`)
declara las 21 columnas resultantes. Una función de extracción compartida la consume tanto en el bulk de
entrenamiento (reconstruyendo la serie efectiva con `apply_shape` + `apply_closure_policy` de Fase B1,
sin volver a simular) como en `ScenarioPredictor` de inferencia. Una migración SQL nueva persiste el
contrato nuevo en una tabla separada, sin tocar `node_features`/`tabular_v3_17`.

**Tech Stack:** Python 3.11, pandas, numpy, networkx (ya en uso), pytest, SQLite.

**Spec:** `docs/superpowers/specs/2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §6 (segunda
mitad: contrato de features). La primera mitad (cierre del aporte) es Fase B1, ya implementada y
mergeada en este mismo branch; este plan depende de ella (`apply_closure_policy`,
`apply_shape`).

## Global Constraints

- El contrato nuevo es deliberadamente incompatible con `tabular_v3_17`. Los modelos y el contrato
  antiguos se preservan intactos — este plan nunca modifica `swmm_resilience/ml/contracts.py`'s
  `TABULAR_V3_17`, ni la tabla SQL `node_features`, ni ningún archivo bajo `swmm_resilience/ml/bench/`.
- `Q_tributario` suma las series del nodo y sus ancestros únicos sobre una misma malla, con
  interpolación lineal explícita. Nunca se sustituye la suma temporal por una suma de máximos (eso es lo
  que ya hace `q_pico_acum_escalado`, que se conserva sin cambios como proxy documentado — no se
  rebautiza).
- Las integrales usan cortes exactos en los cruces de umbral por segmento — nunca solo los puntos de la
  malla original. La conversión de una integral en L/s·h a m³ es multiplicar por 3.6.
- Una capacidad (`upstream_capacity_lps`) faltante o ≤ 0 marca inválidas únicamente las dos features que
  dependen de capacidad (`tiempo_sobre_capacidad_h`, `integral_exceso_capacidad_m3`); las otras cuatro
  features de un nodo con capacidad inválida se calculan igual. No se hacen divisiones artificiales.
- **Ruling (alcance: solo escenarios por forma).** Fase B1 solo aplicó el cierre del aporte a
  `run_batch_shapes` (formas) y a `hydrograph_batch.run_batch_validation` (CSV de usuario) — el barrido
  de un solo factor sobre la forma base del `.inp` (`run_batch`, invocado en `main.py` como
  `shape_id="base"`) usa un mecanismo distinto (`run_simulation_simple`, sin `HydrographScenario` ni
  política de cierre) y quedó fuera de fase B1. Este plan calcula el contrato nuevo únicamente para filas
  con `shape_id != "base"` — las únicas que tienen una serie efectiva post-cierre bien definida. Las filas
  `"base"` no entran al contrato nuevo. Costo si es incorrecto: el dataset nuevo excluye 25 filas por
  nodo (una por factor de la forma base); el protocolo principal ya es `LeaveOneShapeOut`, que necesita
  diversidad de formas y nunca dependió de `"base"`, así que esta exclusión no afecta el objetivo
  principal del banco.
- **Ruling (identificador del contrato).** `"tabular_v4_21"` — versión 4 (incompatible con v3), 21
  columnas (15 de las 17 actuales, menos `duracion_horas`/`tiempo_al_pico_h` que quedan solo en el
  contrato legado, más las 6 nuevas). Costo si es incorrecto: renombrar una constante, sin efecto en
  datos ya persistidos bajo `tabular_v3_17`.
- **Ruling (no se re-simula ni se regenera el histórico completo).** Las seis features nuevas se derivan
  de la serie efectiva, que es una función pura y determinista de (forma, factor, `base_inflows`,
  tolerancia de cierre) vía `apply_shape` + `apply_closure_policy` — no hace falta volver a correr SWMM
  para calcularlas. Este plan construye y prueba esa función, y corre un smoke sobre un puñado de
  combinaciones reales, pero no reprocesa las ~4000 filas del dataset actual. La regeneración completa
  del dataset (con las labels SWMM ya recalculadas por el cierre de fase B1 para los escenarios que
  cambiaron) es una decisión de cómputo separada, igual que en fase B1.
- No hacer commit fuera de las tareas de este plan; se ejecuta en la misma rama/worktree que ya trae fase
  A, la búsqueda de hiperparámetros y fase B1 (`worktree-optuna-tuning-familia-ganadora`).

---

## Mapa de archivos

- `swmm_resilience/extraction/tributary_features.py` (nuevo): `Q_tributario`, interpolación lineal, y las
  seis funciones de feature.
- `tests/extraction/test_tributary_features.py` (nuevo).
- `swmm_resilience/ml/contracts.py`: `FEATURE_DEFINITIONS_V21`, `TABULAR_V4_21`.
- `tests/ml/test_contracts.py` (o el archivo existente que prueba `contracts.py`).
- `swmm_resilience/extraction/dynamic_features.py`: función compartida
  `compute_tributary_dynamic_features()` que usan entrenamiento e inferencia.
- `main.py`: el bloque de "Additional shapes" (pipeline completo) calcula y persiste las columnas nuevas
  junto a `dynamic_df` para filas `shape_id != "base"`.
- `swmm_resilience/ml/scenario_predict.py`: `ScenarioPredictor.predict_timed()` calcula las mismas
  columnas para inferencia.
- `swmm_resilience/database/sql/006_tributary_contract.sql` (nuevo): tabla `node_features_v4_21`.
- `tests/database/test_migration_006_tributary_contract.py` (nuevo, o seguir el patrón de
  `tests/database/test_migration_005_*.py` ya existentes).

## Interfaces que produce el plan

```python
# extraction/tributary_features.py
def compute_q_tributario(
    node_id: str,
    node_series: dict[str, list[tuple[float, float]]],
    graph: "nx.DiGraph",
) -> list[tuple[float, float]]: ...

def volumen_local_integrado_m3(series: list[tuple[float, float]]) -> float: ...
def volumen_tributario_integrado_m3(q_tributario: list[tuple[float, float]]) -> float: ...
def duracion_efectiva_tributario_h(q_tributario: list[tuple[float, float]]) -> float: ...
def tiempo_primer_pico_tributario_h(q_tributario: list[tuple[float, float]]) -> float: ...
def tiempo_sobre_capacidad_h(
    q_tributario: list[tuple[float, float]], capacity_lps: float | None,
) -> float | None: ...
def integral_exceso_capacidad_m3(
    q_tributario: list[tuple[float, float]], capacity_lps: float | None,
) -> float | None: ...

def compute_node_tributary_features(
    node_id: str,
    node_series: dict[str, list[tuple[float, float]]],
    graph: "nx.DiGraph",
    capacity_lps: float | None,
) -> dict: ...  # las 6 columnas, con las 2 dependientes de capacidad en None si invalida

# ml/contracts.py
TABULAR_V4_21: FeatureContract  # contract_id == "tabular_v4_21"

# extraction/dynamic_features.py
def compute_tributary_dynamic_features(
    node_ids: list[str],
    node_series: dict[str, list[tuple[float, float]]],
    graph: "nx.DiGraph",
    capacity_by_node: dict[str, float | None],
) -> pd.DataFrame: ...  # una fila por node_id, columnas = las 6 features nuevas
```

`held_out_groups`-style, `q_tributario` nunca se persiste column-a-column: solo las seis features
derivadas entran al contrato.

---

### Task 1: `Q_tributario` e interpolación lineal

**Files:**
- Crear: `swmm_resilience/extraction/tributary_features.py`
- Test: `tests/extraction/test_tributary_features.py`

**Consume:** nada del banco. Reutiliza el patrón de `nx.ancestors(graph, node_id)` que ya usa
`compute_scenario_dynamic_features` en `dynamic_features.py` (leído antes de escribir este plan).

**Produce:** `compute_q_tributario()` — la serie sumada del nodo y sus ancestros únicos, sobre la unión de
sus mallas temporales, con interpolación lineal explícita en los puntos que no comparte cada serie
individual.

- [ ] **Paso 1: escribir las pruebas.**

Crear `tests/extraction/test_tributary_features.py`:

```python
import networkx as nx
import pytest

from swmm_resilience.extraction.tributary_features import compute_q_tributario


def _linear_graph():
    """upstream -> mid -> downstream (upstream y mid son ancestros de downstream)."""
    g = nx.DiGraph()
    g.add_edge("upstream", "mid")
    g.add_edge("mid", "downstream")
    return g


def test_q_tributario_sums_node_and_ancestors_on_shared_grid():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 1.0), (1.0, 3.0)],
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }
    q = compute_q_tributario("downstream", node_series, graph)
    assert q == [(0.0, 3.5), (1.0, 7.5)]  # 1.0+2.0+0.5, 3.0+4.0+0.5


def test_q_tributario_interpolates_a_series_with_a_different_grid():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 0.0), (2.0, 4.0)],  # solo 2 puntos, malla distinta
        "mid": [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)],
        "downstream": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
    }
    q = compute_q_tributario("downstream", node_series, graph)
    # malla union: {0.0, 1.0, 2.0}. upstream en t=1.0 se interpola: 0.0 + (4.0-0.0)*(1.0/2.0) = 2.0
    assert q == [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]


def test_q_tributario_ignores_a_node_absent_from_node_series():
    """Un nodo sin aporte propio (base_inflow == 0) nunca entra a node_series;
    no debe romper la suma de sus descendientes."""
    graph = _linear_graph()
    node_series = {
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }  # "upstream" ausente
    q = compute_q_tributario("downstream", node_series, graph)
    assert q == [(0.0, 2.5), (1.0, 4.5)]


def test_q_tributario_of_an_isolated_node_is_just_its_own_series():
    graph = nx.DiGraph()
    graph.add_node("solo")
    node_series = {"solo": [(0.0, 1.0), (1.0, 2.0)]}
    assert compute_q_tributario("solo", node_series, graph) == [(0.0, 1.0), (1.0, 2.0)]


def test_q_tributario_of_a_node_not_in_the_graph_falls_back_to_its_own_series():
    graph = nx.DiGraph()
    node_series = {"orphan": [(0.0, 1.0), (1.0, 2.0)]}
    assert compute_q_tributario("orphan", node_series, graph) == [(0.0, 1.0), (1.0, 2.0)]
```

- [ ] **Paso 2: ejecutar la prueba y comprobar que falla por ausencia del módulo.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/extraction/test_tributary_features.py -q`

Esperado: `ModuleNotFoundError: No module named 'swmm_resilience.extraction.tributary_features'`

- [ ] **Paso 3: implementar el módulo.**

Crear `swmm_resilience/extraction/tributary_features.py`:

```python
"""Q_tributario y las seis features de forma tributaria (spec §6, contrato nuevo).

Q_tributario(t) para un nodo es la suma de su propia serie de aporte y la de
todos sus ancestros únicos en la red, sobre una malla temporal común. Se
calcula por interpolación lineal explícita -- nunca sustituyendo la suma
temporal por una suma de picos, que es justamente lo que ya hace
q_pico_acum_escalado (contrato legado, se conserva como proxy documentado).

Este módulo es puro: no sabe de SWMM, de la base de datos ni de la política
de cierre de fase B1. Recibe la serie ya efectiva (post-cierre) de cada
nodo y el grafo de la red, y no hace I/O.
"""

from __future__ import annotations

import networkx as nx

_Point = tuple[float, float]
_Series = list[_Point]


def _interp_at(series: _Series, t: float) -> float:
    """Interpolación lineal de `series` (ordenada por t) en el instante `t`.

    Fuera del rango de `series` se sostiene el valor del extremo más
    cercano (clamp), consistente con cómo write_scenario_inp trata el
    tramo de drenaje: una serie nunca "empieza" antes de su primer punto
    ni "sigue cambiando" después del último.
    """
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            frac = (t - t0) / (t1 - t0)
            return v0 + (v1 - v0) * frac
    return series[-1][1]


def compute_q_tributario(
    node_id: str,
    node_series: dict[str, _Series],
    graph: "nx.DiGraph",
) -> _Series:
    """Suma la serie de `node_id` y la de sus ancestros únicos, malla común.

    Un nodo sin aporte propio (base_inflow_lps == 0) nunca aparece en
    node_series -- se ignora, no contribuye. Un nodo ausente del grafo
    (o sin ancestros) devuelve solo su propia serie.
    """
    ancestors = set(nx.ancestors(graph, node_id)) if graph.has_node(node_id) else set()
    contributing = sorted((ancestors | {node_id}) & set(node_series))
    if not contributing:
        return []

    grid = sorted({t for nid in contributing for t, _ in node_series[nid]})
    return [
        (t, sum(_interp_at(node_series[nid], t) for nid in contributing))
        for t in grid
    ]
```

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/extraction/test_tributary_features.py -q`

Esperado: 5 passed.

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/extraction/tributary_features.py tests/extraction/test_tributary_features.py
git commit -m "feat: Q_tributario con interpolacion lineal explicita (fase B2)"
```

---

### Task 2: Las seis features (integrales y duraciones con cortes exactos)

**Files:**
- Modificar: `swmm_resilience/extraction/tributary_features.py`
- Modificar: `tests/extraction/test_tributary_features.py`

**Consume:** `compute_q_tributario()` (Task 1).

**Produce:** `volumen_local_integrado_m3`, `volumen_tributario_integrado_m3`,
`duracion_efectiva_tributario_h`, `tiempo_primer_pico_tributario_h`, `tiempo_sobre_capacidad_h`,
`integral_exceso_capacidad_m3`, y `compute_node_tributary_features()` que las agrupa.

- [ ] **Paso 1: escribir las pruebas.**

Añadir a `tests/extraction/test_tributary_features.py`:

```python
from swmm_resilience.extraction.tributary_features import (
    compute_node_tributary_features,
    duracion_efectiva_tributario_h,
    integral_exceso_capacidad_m3,
    tiempo_primer_pico_tributario_h,
    tiempo_sobre_capacidad_h,
    volumen_local_integrado_m3,
    volumen_tributario_integrado_m3,
)


def test_volumen_integrado_uses_trapezoidal_rule_times_3_6():
    # trapecio: (0,0)-(1,10) -> area = 0.5*1*10 = 5.0 L/s*h -> *3.6 = 18.0 m3
    assert volumen_local_integrado_m3([(0.0, 0.0), (1.0, 10.0)]) == pytest.approx(18.0)


def test_volumen_tributario_integrado_matches_volumen_local_for_a_single_series():
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)]
    assert volumen_tributario_integrado_m3(series) == pytest.approx(
        volumen_local_integrado_m3(series)
    )


def test_duracion_efectiva_counts_only_time_above_one_percent_of_peak():
    # pico = 10.0, 1% = 0.1. Serie sube de 0 a 10 en [0,1], baja de 10 a 0 en [1,2].
    # cruza 0.1 en t=0.01 (subida) y en t=1.99 (bajada) -> duracion ~= 1.98h
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)]
    assert duracion_efectiva_tributario_h(series) == pytest.approx(1.98, abs=1e-6)


def test_duracion_efectiva_is_zero_for_a_null_series():
    assert duracion_efectiva_tributario_h([(0.0, 0.0), (1.0, 0.0)]) == 0.0


def test_tiempo_primer_pico_returns_the_time_of_the_first_maximum():
    # dos picos iguales a 5.0, en t=1.0 y t=3.0 -> el PRIMERO es t=1.0
    series = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
    assert tiempo_primer_pico_tributario_h(series) == 1.0


def test_tiempo_primer_pico_is_zero_for_a_null_series():
    assert tiempo_primer_pico_tributario_h([(0.0, 0.0), (1.0, 0.0)]) == 0.0


def test_tiempo_sobre_capacidad_counts_time_above_the_exact_threshold():
    # capacidad = 5.0. Serie sube 0->10 en [0,1]: cruza 5.0 en t=0.5.
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert tiempo_sobre_capacidad_h(series, capacity_lps=5.0) == pytest.approx(0.5)


def test_tiempo_sobre_capacidad_is_none_when_capacity_is_missing_or_non_positive():
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert tiempo_sobre_capacidad_h(series, capacity_lps=None) is None
    assert tiempo_sobre_capacidad_h(series, capacity_lps=0.0) is None
    assert tiempo_sobre_capacidad_h(series, capacity_lps=-1.0) is None


def test_integral_exceso_capacidad_uses_exact_segment_crossings():
    # capacidad = 5.0. Serie sube 0->10 en [0,1] (cruza en t=0.5), luego
    # 10->10 en [1,2] (constante). Exceso: triangulo [0.5,1] area=0.5*0.5*5=1.25
    # mas rectangulo [1,2] area=1.0*5=5.0 -> total 6.25 L/s*h -> *3.6 = 22.5 m3
    series = [(0.0, 0.0), (1.0, 10.0), (2.0, 10.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=5.0) == pytest.approx(22.5)


def test_integral_exceso_capacidad_is_zero_when_never_exceeded():
    series = [(0.0, 0.0), (1.0, 3.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=5.0) == pytest.approx(0.0)


def test_integral_exceso_capacidad_is_none_when_capacity_is_missing_or_non_positive():
    series = [(0.0, 0.0), (1.0, 10.0)]
    assert integral_exceso_capacidad_m3(series, capacity_lps=None) is None
    assert integral_exceso_capacidad_m3(series, capacity_lps=0.0) is None


def test_compute_node_tributary_features_returns_all_six_keys():
    graph = _linear_graph()
    node_series = {
        "upstream": [(0.0, 1.0), (1.0, 3.0)],
        "mid": [(0.0, 2.0), (1.0, 4.0)],
        "downstream": [(0.0, 0.5), (1.0, 0.5)],
    }
    result = compute_node_tributary_features("downstream", node_series, graph, capacity_lps=6.0)
    assert set(result) == {
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }


def test_compute_node_tributary_features_nulls_only_the_capacity_dependent_pair(): 
    graph = _linear_graph()
    node_series = {"downstream": [(0.0, 0.0), (1.0, 10.0)]}
    result = compute_node_tributary_features(
        "downstream", node_series, graph, capacity_lps=None,
    )
    assert result["tiempo_sobre_capacidad_h"] is None
    assert result["integral_exceso_capacidad_m3"] is None
    assert result["volumen_local_integrado_m3"] is not None
    assert result["volumen_tributario_integrado_m3"] is not None
    assert result["duracion_efectiva_tributario_h"] is not None
    assert result["tiempo_primer_pico_tributario_h"] is not None
```

- [ ] **Paso 2: ejecutar y comprobar que fallan por `ImportError`.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/extraction/test_tributary_features.py -q`

Esperado: `ImportError: cannot import name 'volumen_local_integrado_m3'`.

- [ ] **Paso 3: implementar las seis funciones y el agregador.**

Añadir a `swmm_resilience/extraction/tributary_features.py`:

```python
def _trapezoid_lps_h(series: _Series) -> float:
    """Integral trapezoidal de `series` (t en horas, q en L/s), en L/s*h."""
    return sum(
        (t1 - t0) * (v0 + v1) / 2.0
        for (t0, v0), (t1, v1) in zip(series, series[1:])
    )


def _segment_duration_above(t0: float, v0: float, t1: float, v1: float, threshold: float) -> float:
    """Duracion exacta donde v(t) > threshold sobre el segmento lineal [t0,t1]."""
    e0, e1 = v0 - threshold, v1 - threshold
    if e0 <= 0 and e1 <= 0:
        return 0.0
    if e0 > 0 and e1 > 0:
        return t1 - t0
    t_cross = t0 + (t1 - t0) * (e0 / (e0 - e1))
    return (t_cross - t0) if e0 > 0 else (t1 - t_cross)


def _segment_excess_integral(t0: float, v0: float, t1: float, v1: float, cap: float) -> float:
    """Integral exacta de max(v(t)-cap, 0) sobre el segmento lineal [t0,t1], en L/s*h."""
    e0, e1 = v0 - cap, v1 - cap
    if e0 <= 0 and e1 <= 0:
        return 0.0
    if e0 >= 0 and e1 >= 0:
        return (t1 - t0) * (e0 + e1) / 2.0
    t_cross = t0 + (t1 - t0) * (e0 / (e0 - e1))
    if e0 > 0:
        return (t_cross - t0) * e0 / 2.0
    return (t1 - t_cross) * e1 / 2.0


def volumen_local_integrado_m3(series: _Series) -> float:
    """Integral de la serie propia del nodo, en m3 (spec §6)."""
    return _trapezoid_lps_h(series) * 3.6


def volumen_tributario_integrado_m3(q_tributario: _Series) -> float:
    """Integral de Q_tributario, en m3."""
    return _trapezoid_lps_h(q_tributario) * 3.6


def duracion_efectiva_tributario_h(q_tributario: _Series) -> float:
    """Duracion en que Q_tributario supera el 1% de su propio pico."""
    if not q_tributario:
        return 0.0
    peak = max(v for _, v in q_tributario)
    if peak <= 0:
        return 0.0
    threshold = 0.01 * peak
    return sum(
        _segment_duration_above(t0, v0, t1, v1, threshold)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )


def tiempo_primer_pico_tributario_h(q_tributario: _Series) -> float:
    """Instante del primer maximo de Q_tributario; cero para una serie nula."""
    if not q_tributario:
        return 0.0
    peak = max(v for _, v in q_tributario)
    if peak <= 0:
        return 0.0
    return next(t for t, v in q_tributario if v == peak)


def tiempo_sobre_capacidad_h(q_tributario: _Series, capacity_lps: float | None) -> float | None:
    """Duracion en que Q_tributario supera upstream_capacity_lps.

    None si la capacidad falta o es <= 0 (spec §6: sin divisiones artificiales).
    """
    if capacity_lps is None or capacity_lps <= 0:
        return None
    return sum(
        _segment_duration_above(t0, v0, t1, v1, capacity_lps)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )


def integral_exceso_capacidad_m3(q_tributario: _Series, capacity_lps: float | None) -> float | None:
    """Integral de max(Q_tributario - upstream_capacity_lps, 0), en m3.

    None si la capacidad falta o es <= 0.
    """
    if capacity_lps is None or capacity_lps <= 0:
        return None
    total = sum(
        _segment_excess_integral(t0, v0, t1, v1, capacity_lps)
        for (t0, v0), (t1, v1) in zip(q_tributario, q_tributario[1:])
    )
    return total * 3.6


def compute_node_tributary_features(
    node_id: str,
    node_series: dict[str, _Series],
    graph: "nx.DiGraph",
    capacity_lps: float | None,
) -> dict:
    """Agrupa las seis features tributarias de un nodo."""
    own_series = node_series.get(node_id, [(0.0, 0.0), (0.0, 0.0)])
    q_trib = compute_q_tributario(node_id, node_series, graph)
    return {
        "volumen_local_integrado_m3": volumen_local_integrado_m3(own_series),
        "volumen_tributario_integrado_m3": volumen_tributario_integrado_m3(q_trib),
        "duracion_efectiva_tributario_h": duracion_efectiva_tributario_h(q_trib),
        "tiempo_primer_pico_tributario_h": tiempo_primer_pico_tributario_h(q_trib),
        "tiempo_sobre_capacidad_h": tiempo_sobre_capacidad_h(q_trib, capacity_lps),
        "integral_exceso_capacidad_m3": integral_exceso_capacidad_m3(q_trib, capacity_lps),
    }
```

Añadir `import pytest` al principio de `tests/extraction/test_tributary_features.py` si no está ya (los
tests de Task 2 usan `pytest.approx`).

- [ ] **Paso 4: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/extraction/test_tributary_features.py -q`

Esperado: todos los tests de Task 1 y Task 2 en verde (17 casos entre ambas).

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/extraction/tributary_features.py tests/extraction/test_tributary_features.py
git commit -m "feat: seis features tributarias con cortes exactos de umbral (fase B2)"
```

---

### Task 3: Contrato `tabular_v4_21`

**Files:**
- Modificar: `swmm_resilience/ml/contracts.py`
- Test: `tests/ml/test_contracts.py` (o el archivo existente que prueba `contracts.py`; buscar primero con
  `grep -rl "TABULAR_V3_17\|FEATURE_COLUMNS_V17" tests/` para confirmar el archivo real antes de escribir)

**Consume:** nada del resto del plan; solo declara el esquema.

**Produce:** `TABULAR_V4_21`, un `FeatureContract` con 21 columnas: las 15 features estáticas/escaladas
que se conservan de `tabular_v3_17` (todas menos `duracion_horas` y `tiempo_al_pico_h`, que quedan solo
en el contrato legado) más las 6 nuevas de las Tasks 1-2.

- [ ] **Paso 1: localizar el archivo de test real de `contracts.py` y escribir las pruebas.**

Ejecutar primero: `grep -rl "TABULAR_V3_17" tests/` para confirmar dónde viven las pruebas de contrato
existentes, y leer su estilo antes de añadir. Añadir, en ese archivo (o crear
`tests/ml/test_contracts_v4.py` si no existe ninguno dedicado a contratos):

```python
from swmm_resilience.ml.contracts import TABULAR_V4_21


def test_tabular_v4_21_has_21_ordered_features():
    assert TABULAR_V4_21.contract_id == "tabular_v4_21"
    assert TABULAR_V4_21.feature_count == 21


def test_tabular_v4_21_drops_duracion_and_tiempo_al_pico():
    names = TABULAR_V4_21.feature_names
    assert "duracion_horas" not in names
    assert "tiempo_al_pico_h" not in names


def test_tabular_v4_21_keeps_q_pico_acum_escalado_as_a_proxy():
    assert "q_pico_acum_escalado" in TABULAR_V4_21.feature_names


def test_tabular_v4_21_includes_the_six_new_tributary_features():
    names = set(TABULAR_V4_21.feature_names)
    assert names.issuperset({
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    })


def test_tabular_v4_21_marks_the_capacity_dependent_features_nullable():
    from swmm_resilience.ml.contracts import NULLABLE_FEATURE_COLUMNS_V4_21
    assert "tiempo_sobre_capacidad_h" in NULLABLE_FEATURE_COLUMNS_V4_21
    assert "integral_exceso_capacidad_m3" in NULLABLE_FEATURE_COLUMNS_V4_21


def test_tabular_v3_17_is_unchanged():
    """No-regresion: este plan nunca toca el contrato legado."""
    from swmm_resilience.ml.contracts import TABULAR_V3_17
    assert TABULAR_V3_17.contract_id == "tabular_v3_17"
    assert TABULAR_V3_17.feature_count == 17
```

- [ ] **Paso 2: ejecutar y comprobar que falla.**

Esperado: `ImportError: cannot import name 'TABULAR_V4_21'`.

- [ ] **Paso 3: declarar el contrato nuevo en `contracts.py`.**

Añadir, después de `TABULAR_V3_17 = FeatureContract(...)`:

```python
FEATURE_DEFINITIONS_V21 = tuple(
    item for item in FEATURE_DEFINITIONS_V17
    if item.name not in {"duracion_horas", "tiempo_al_pico_h"}
) + (
    FeatureDefinition("volumen_local_integrado_m3", "m3", "Node's own integrated inflow volume"),
    FeatureDefinition("volumen_tributario_integrado_m3", "m3", "Integrated Q_tributario volume"),
    FeatureDefinition("duracion_efectiva_tributario_h", "h", "Time Q_tributario stays above 1% of its own peak"),
    FeatureDefinition("tiempo_primer_pico_tributario_h", "h", "Time to the first Q_tributario maximum"),
    FeatureDefinition(
        "tiempo_sobre_capacidad_h", "h",
        "Time Q_tributario exceeds upstream_capacity_lps", nullable=True,
    ),
    FeatureDefinition(
        "integral_exceso_capacidad_m3", "m3",
        "Integral of max(Q_tributario - upstream_capacity_lps, 0)", nullable=True,
    ),
)

FEATURE_COLUMNS_V21 = tuple(item.name for item in FEATURE_DEFINITIONS_V21)
NULLABLE_FEATURE_COLUMNS_V4_21 = frozenset(
    item.name for item in FEATURE_DEFINITIONS_V21 if item.nullable
)

TABULAR_V4_21 = FeatureContract("tabular_v4_21", FEATURE_DEFINITIONS_V21)
```

No modificar nada de `TABULAR_V3_17`, `FEATURE_DEFINITIONS_V17`, `FEATURE_COLUMNS_V17` ni
`NULLABLE_FEATURE_COLUMNS_V17` — el contrato legado queda intacto.

- [ ] **Paso 4: ejecutar y comprobar verde.**

Esperado: todos los tests nuevos pasan, y ningún test preexistente de `contracts.py`/`TABULAR_V3_17`
cambia de resultado.

- [ ] **Paso 5: commit.**

```bash
git add swmm_resilience/ml/contracts.py <archivo de test correspondiente>
git commit -m "feat: contrato tabular_v4_21 con features tributarias (fase B2)"
```

---

### Task 4: Función de extracción compartida (entrenamiento + inferencia)

**Files:**
- Modificar: `swmm_resilience/extraction/dynamic_features.py`
- Modificar: `main.py` (bloque "Additional shapes" del pipeline completo)
- Modificar: `swmm_resilience/ml/scenario_predict.py`
- Test: `tests/extraction/test_dynamic_features.py` (o el archivo existente; buscar con
  `grep -rl "compute_scenario_dynamic_features" tests/`)
- Test: el archivo existente de `tests/ml/test_scenario_predict.py` (buscar con
  `grep -rl "ScenarioPredictor" tests/`)

**Consume:** `compute_node_tributary_features` (Tasks 1-2); `apply_shape`, `apply_closure_policy` (fase
B1, ya en el branch); `TABULAR_V4_21` (Task 3, solo para nombres de columna, no para escribir el CSV
final todavía — esta tarea calcula las columnas, no persiste el dataset completo).

**Produce:** `compute_tributary_dynamic_features()` en `dynamic_features.py`, usada por igual desde el
bloque de entrenamiento en `main.py` y desde `ScenarioPredictor.predict_timed()`.

- [ ] **Paso 1: escribir las pruebas de la función compartida.**

Localizar primero el archivo de test real de `dynamic_features.py`
(`grep -rl "compute_scenario_dynamic_features" tests/`) y seguir su estilo de fixtures (probablemente ya
construye un `nx.DiGraph` pequeño de forma similar a las Tasks 1-2). Añadir:

```python
def test_compute_tributary_dynamic_features_returns_one_row_per_node():
    import networkx as nx
    from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features

    graph = nx.DiGraph()
    graph.add_edge("A", "B")
    node_series = {
        "A": [(0.0, 1.0), (1.0, 2.0)],
        "B": [(0.0, 0.5), (1.0, 0.5)],
    }
    capacity_by_node = {"A": 10.0, "B": None}

    df = compute_tributary_dynamic_features(["A", "B"], node_series, graph, capacity_by_node)

    assert sorted(df["node_id"]) == ["A", "B"]
    assert {
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }.issubset(df.columns)
    row_b = df.loc[df["node_id"] == "B"].iloc[0]
    assert row_b["tiempo_sobre_capacidad_h"] is None or pd.isna(row_b["tiempo_sobre_capacidad_h"])


def test_compute_tributary_dynamic_features_handles_a_node_with_no_own_series():
    """Un nodo con base_inflow == 0 nunca aparece en node_series; debe seguir
    recibiendo una fila (con volumen local 0), no romper la extraccion."""
    import networkx as nx
    from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features

    graph = nx.DiGraph()
    graph.add_node("no_inflow")
    df = compute_tributary_dynamic_features(
        ["no_inflow"], {}, graph, {"no_inflow": 5.0},
    )
    assert len(df) == 1
    assert df.iloc[0]["volumen_local_integrado_m3"] == 0.0
```

(Ajustar el import `import pandas as pd` en el archivo de test si hace falta.)

- [ ] **Paso 2: ejecutar y comprobar que falla por `ImportError`.**

- [ ] **Paso 3: implementar `compute_tributary_dynamic_features()` en `dynamic_features.py`.**

Añadir a `swmm_resilience/extraction/dynamic_features.py` (junto a las dos funciones existentes,
mismo estilo de docstring):

```python
def compute_tributary_dynamic_features(
    node_ids: list[str],
    node_series: dict[str, list[tuple[float, float]]],
    graph: "nx.DiGraph",
    capacity_by_node: dict[str, float | None],
) -> pd.DataFrame:
    """Calcula las seis features tributarias (contrato tabular_v4_21) por nodo.

    Usada por igual en el bulk de entrenamiento (con la serie efectiva
    reconstruida via apply_shape + apply_closure_policy de fase B1) y en
    ScenarioPredictor de inferencia (con scenario.node_series directo) --
    es la MISMA función en ambos casos, tal como exige la spec §6.
    """
    from ..extraction.tributary_features import compute_node_tributary_features

    rows = []
    for node_id in node_ids:
        features = compute_node_tributary_features(
            node_id, node_series, graph, capacity_by_node.get(node_id)
        )
        rows.append({"node_id": node_id, **features})
    return pd.DataFrame(rows)
```

Añadir `import networkx as nx` a los imports del archivo si no está ya presente (revisar el bloque de
imports actual antes de añadirlo, puede que ya lo use `compute_scenario_dynamic_features`).

- [ ] **Paso 4: ejecutar y comprobar verde.**

Esperado: todos los tests de esta tarea pasan.

- [ ] **Paso 5: llamar la función desde el bulk de entrenamiento en `main.py`.**

En el bloque "Additional shapes" de `main.py` (dentro de `for shape_id, factor, rpt_path in
shape_results:`), después de construir `dynamic_df` con `compute_dynamic_features(...)` y antes de
`simulation_results.append(...)`, añadir el cálculo de las columnas nuevas reconstruyendo la serie
efectiva con las funciones de fase B1:

```python
                from swmm_resilience.simulation.hydrograph_shapes import apply_shape
                from swmm_resilience.simulation.inflow_closure import apply_closure_policy
                from swmm_resilience.validation.hydrograph_csv import HydrographScenario
                from swmm_resilience.extraction.dynamic_features import (
                    compute_tributary_dynamic_features,
                )

                node_series_for_features = apply_shape(shapes[shape_id], base_inflows, factor)
                time_grid = [t for t, _ in shapes[shape_id]]
                effective_scenario, _ = apply_closure_policy(
                    HydrographScenario(
                        scenario_id=f"{shape_id}_f{factor:.3f}",
                        node_series=node_series_for_features,
                        time_grid_hours=time_grid,
                        last_time_hours=shapes[shape_id][-1][0],
                    ),
                    config.validation.inflow_closure_tolerance_lps,
                )
                capacity_by_node = dict(
                    zip(
                        static_topo_df["node_id"].astype(str),
                        static_topo_df["upstream_capacity_lps"],
                    )
                )
                tributary_df = compute_tributary_dynamic_features(
                    all_node_ids, effective_scenario.node_series, graph, capacity_by_node,
                )
                dynamic_df = dynamic_df.merge(tributary_df, on="node_id", how="left")
```

`graph` debe construirse una sola vez antes del bucle de formas (no dentro de él) —
`swmm_resilience.extraction.topology.build_network_graph(load_inp(config.network.inp_path))`, siguiendo
exactamente el mismo patrón que ya usa `ScenarioPredictor.__init__` (`scenario_predict.py`, línea ~66).
Añadir esa construcción una sola vez, antes del `for shape_id, factor, rpt_path in shape_results:`.

Esto no cambia qué columnas se persisten hoy (`assemble_dataset` sigue escribiendo lo mismo que antes;
el CSV final de este plan no adopta aún el contrato nuevo como esquema oficial — eso es la Task 6). El
objetivo de este paso es solo probar que la función compartida corre sin error contra datos reales del
pipeline de entrenamiento.

- [ ] **Paso 6: llamar la misma función desde `ScenarioPredictor.predict_timed()`.**

En `swmm_resilience/ml/scenario_predict.py`, dentro de `predict_timed()`, después de construir
`dynamic_df = compute_scenario_dynamic_features(...)` y antes de `static_base = ...`, añadir:

```python
        capacity_by_node = dict(
            zip(self.full_df["node_id"].astype(str), self.full_df["upstream_capacity_lps"])
        )
        tributary_df = compute_tributary_dynamic_features(
            list(self.full_df["node_id"].astype(str)),
            scenario.node_series, self.graph, capacity_by_node,
        )
        dynamic_df = dynamic_df.merge(tributary_df, on="node_id", how="left")
```

Añadir `compute_tributary_dynamic_features` al import ya existente de `..extraction.dynamic_features` en
este archivo. No cambiar `X = merged[FEATURE_COLS]` — `FEATURE_COLS` sigue siendo el contrato legado
(`tabular_v3_17`) hasta que exista un modelo entrenado sobre `tabular_v4_21`; las columnas nuevas quedan
calculadas y disponibles en `merged`, pero el modelo activo (legado) simplemente no las usa todavía. Esto
es intencional: cablear `ScenarioPredictor` para que use `tabular_v4_21` en producción es un paso
posterior (fase D, cuando exista un modelo entrenado sobre el contrato nuevo), no de este plan.

- [ ] **Paso 7: ejecutar las suites de ambos módulos y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/extraction tests/ml -q -k "not bench"`

Esperado: sin regresiones. Si `-k "not bench"` excluye demasiado o muy poco, ajustar el filtro tras
revisar qué archivos existen bajo `tests/ml/` — el objetivo es correr las pruebas de `dynamic_features.py`
y `scenario_predict.py` sin pagar el costo del banco completo, que no cambió en esta tarea.

- [ ] **Paso 8: commit.**

```bash
git add swmm_resilience/extraction/dynamic_features.py main.py swmm_resilience/ml/scenario_predict.py <archivos de test tocados>
git commit -m "feat: funcion de extraccion tributaria compartida entre entrenamiento e inferencia (fase B2)"
```

---

### Task 5: Migración SQL para `tabular_v4_21`

**Files:**
- Crear: `swmm_resilience/database/sql/006_tributary_contract.sql`
- Test: crear `tests/database/test_migration_006_tributary_contract.py`, siguiendo el patrón de
  `tests/database/test_migration_005_*.py` (leer al menos uno de esos archivos antes de escribir el
  nuevo, para replicar su convención de fixtures de conexión/migraciones exactamente)

**Consume:** nada de las tareas anteriores directamente — es un esquema paralelo, aditivo.

**Produce:** tabla `node_features_v4_21`, espejo de `node_features` pero con las 21 columnas del
contrato nuevo y `CHECK(feature_contract_id='tabular_v4_21')`. No modifica `node_features` ni ninguna
tabla existente.

- [ ] **Paso 1: leer un migration file existente para replicar la convención exacta.**

Leer `swmm_resilience/database/sql/001_v17_initial.sql` (ya citado en este plan, sección `node_features`)
y al menos uno de `tests/database/test_migration_005_*.py` antes de escribir esta tarea — confirmar el
estilo de `PRAGMA`, `FOREIGN KEY`, y cómo los tests existentes abren una conexión y aplican migraciones
(`connect_managed_database`, `apply_migrations`, ya usados en `tests/conftest.py`'s
`sql_training_db`/`sql_training_db_six_shapes`, leídos en fases anteriores de este mismo proyecto).

- [ ] **Paso 2: escribir la prueba de la migración.**

Crear `tests/database/test_migration_006_tributary_contract.py` con al menos:

```python
def test_migration_006_creates_node_features_v4_21(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(node_features_v4_21)").fetchall()
    }
    expected = {
        "run_id", "network_id", "node_pk", "feature_contract_id",
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }
    assert expected.issubset(columns)
    conn.close()


def test_migration_006_does_not_alter_the_legacy_node_features_table(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(node_features)").fetchall()}
    assert "volumen_local_integrado_m3" not in columns
    conn.close()


def test_migration_006_rejects_a_row_with_the_wrong_contract_id(tmp_path):
    import sqlite3
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    # Insertar con un feature_contract_id incorrecto debe violar el CHECK.
    # (Completar el resto de columnas NOT NULL según el esquema real leido
    # en el Paso 1 -- requiere filas existentes en networks/nodes/runs para
    # las foreign keys; seguir el patron de fixtures de
    # tests/database/test_migration_005_*.py para poblarlas.)
    ...
```

El tercer test requiere construir filas válidas en `networks`/`nodes`/`runs` primero para satisfacer las
foreign keys — completar su cuerpo replicando el fixture de inserción que ya usa
`tests/database/test_migration_005_*.py` para una tabla con las mismas FKs (`node_features` u otra
tabla hija de `runs`/`nodes`), en vez de inventar una construcción nueva.

- [ ] **Paso 3: ejecutar y comprobar que falla** (la tabla no existe: `sqlite3.OperationalError: no such
  table: node_features_v4_21`).

- [ ] **Paso 4: escribir la migración.**

Crear `swmm_resilience/database/sql/006_tributary_contract.sql`, espejando exactamente la estructura de
`node_features` (mismas columnas estáticas/escaladas que sobreviven en el contrato nuevo, mismos
`FOREIGN KEY`, mismo `PRIMARY KEY(run_id, node_pk)`) pero:
- sin `duracion_horas` ni `tiempo_al_pico_h`;
- con las seis columnas nuevas, `REAL` y `NULL`-ables para `tiempo_sobre_capacidad_h` e
  `integral_exceso_capacidad_m3` (sin `CHECK(... >= 0)` en esas dos, ya que `None` es un valor válido
  cuando la capacidad falta; las otras cuatro sí pueden llevar `CHECK(... >= 0)` como sus análogas en
  `node_features`);
- `CHECK(feature_contract_id='tabular_v4_21')` en vez de `'tabular_v3_17'`.

Usar el número `006` y el nombre de archivo exacto `006_tributary_contract.sql` — el descubridor de
migraciones (`MIGRATION_NAME` regex en `migrations.py`, ya leído en este plan) exige contigüidad desde
`001`.

- [ ] **Paso 5: ejecutar y comprobar verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/database/test_migration_006_tributary_contract.py tests/database -q`

Esperado: todos los tests de migración (incluidos los cinco archivos preexistentes) en verde — una
migración nueva no debe romper ninguna de las anteriores.

- [ ] **Paso 6: commit.**

```bash
git add swmm_resilience/database/sql/006_tributary_contract.sql tests/database/test_migration_006_tributary_contract.py
git commit -m "feat: tabla node_features_v4_21 para el contrato tributario (fase B2)"
```

---

### Task 6: Verificación de fase B2

**Files:**
- No crear archivos de producción adicionales.

**Consume:** las cinco tareas anteriores.

**Produce:** evidencia de que la suite completa sigue verde, de que el contrato legado (`tabular_v3_17`,
`node_features`, todo `ml/bench/`) no cambió, y de que las seis features nuevas se calculan
correctamente contra datos reales (una forma, unos pocos nodos) sin volver a simular ni regenerar el
histórico.

- [ ] **Paso 1: suite completa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests -q`

Esperado: código 0, sin regresiones sobre el estado anterior a este plan.

- [ ] **Paso 2: smoke real de las features tributarias, sin regenerar el dataset.**

Usando la misma forma real identificada en la verificación de fase B1
(`data/hydrograph_shapes/long_moderate.csv`, que termina en `q_norm=0.35` y por tanto necesita cierre),
reconstruir su serie efectiva con `apply_shape` + `apply_closure_policy` (fase B1, sin volver a correr
SWMM), construir el grafo real de la red con `build_network_graph`, y llamar
`compute_tributary_dynamic_features()` para todos los nodos. Confirmar:
- no lanza ninguna excepción;
- las columnas nuevas son numéricas y finitas para los nodos con capacidad válida;
- `tiempo_sobre_capacidad_h`/`integral_exceso_capacidad_m3` son `None`/`NaN` exactamente para los nodos
  cuyo `upstream_capacity_lps` (de `static_topo_df`) es nulo o ≤ 0 — comparar el conteo contra
  `static_topo_df["upstream_capacity_lps"].isna().sum()` u otro chequeo equivalente;
- `volumen_tributario_integrado_m3` de un nodo sin ancestros (raíz de la red) es igual a su
  `volumen_local_integrado_m3` (dentro de una tolerancia de punto flotante).

No persistir nada en la base de datos real ni escribir sobre `outputs/` — este smoke corre en memoria o
en un directorio temporal.

- [ ] **Paso 3: confirmar que el contrato legado no cambió.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench -q`

Esperado: mismo resultado que al cierre de fase A/Optuna (258+ passed) — este plan nunca tocó
`swmm_resilience/ml/bench/`, así que cualquier cambio aquí sería evidencia de una regresión no intencional.

- [ ] **Paso 4: diff y estado de git.**

Ejecutar `git diff --check` y `git status --short`. Esperado: sin errores de espacios; los cambios
corresponden solo al mapa de archivos de este plan.

## Autorrevisión del plan

- **Cobertura:** spec §6 segunda mitad (Q_tributario por interpolación lineal, las seis features con
  cortes exactos, capacidad inválida como nulo sin divisiones artificiales, contrato nuevo incompatible
  con modelos legados preservados, misma función en entrenamiento e inferencia, dataset con identidad
  propia) queda cubierta por las Tasks 1-5; la regeneración completa del histórico y el entrenamiento de
  un modelo nuevo sobre `tabular_v4_21` quedan explícitamente diferidos (Global Constraints), consistente
  con cómo fase B1 difirió la re-simulación completa.
- **Consistencia:** `compute_q_tributario`, `compute_node_tributary_features`,
  `compute_tributary_dynamic_features`, `TABULAR_V4_21` y los nombres de las seis columnas se usan
  idénticos en todas las tareas que los consumen.
- **Sin placeholders:** cada paso trae código completo; los dos pasos que dependen de archivos de test
  existentes cuyo contenido exacto no se leyó al escribir este plan (Task 3 y Task 5) instruyen
  explícitamente localizar y leer ese archivo primero, con el comando `grep` exacto para encontrarlo,
  antes de escribir sus pruebas — no se adivina ningún nombre de fixture.
