# Plan: resolver los hallazgos de la auditoría hidráulica y de ML

> **Para agentes:** SUB-SKILL REQUERIDA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para ejecutar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para el seguimiento.

**Objetivo:** atender los hallazgos 1, 2, 4, 5, 6 y 7 de la auditoría del 8 de noviembre. Para eso se construyen análisis post hoc reproducibles sobre la reserva sellada, una comprobación de cobertura para futuras particiones, un respaldo verificable de los artefactos sellados y las correcciones de documentación. No se reentrena ni se vuelve a seleccionar ningún modelo.

**Arquitectura:** tres módulos pequeños de funciones puras (`DataFrame` de entrada, `DataFrame`/`dict` de salida) en `swmm_resilience/ml/bench/`. Un script los ejecuta en solo lectura sobre los artefactos sellados y escribe un `audit.json` nuevo con creación exclusiva. Un segundo script empaqueta y verifica los artefactos sellados. La documentación se actualiza al final, con las cifras que produce el script.

**Tecnología:** Python 3.12 (`venv/Scripts/python.exe`), pandas 2.3.3, numpy, scikit-learn 1.8.0, pytest (sin `pytest-xdist`), SQLite en modo solo lectura y `zipfile`.

**Spec:** [docs/superpowers/reports/2026-11-08-auditoria-hidraulica-ml.md](../reports/2026-11-08-auditoria-hidraulica-ml.md).

## Restricciones globales

- **Fuera de alcance, por decisión del investigador (8 de noviembre):** el hallazgo 3 (cola de drenaje e inyección de caudal en los nodos). Se considera parte del diseño de la red y queda como está. No se modifica `swmm_resilience/simulation/timeseries_scenario.py`, `batch.py` ni `frozen_pair_runner.py`. Tampoco se tocan las condiciones iniciales (`InitFlow`) de los INP.
- **Artefactos sellados, nunca se escriben:**
  - `outputs/experiments/joint_chico_sur_norte_20261104_v1/`
  - `outputs/experiments/evaluacion_reserva_conjunta_20261108_v1/`
  - `outputs/experiments/final_joint_candidates_20261108_v1/`
  - `outputs/experiments/particion_chico_sur_norte_20261104.json`
  - `outputs/experiments/reserva_formas_chico_sur_norte_20261104.json`
  - todo `registration.json` bajo `outputs/`

  Se abren en solo lectura y las salidas nuevas van a directorios nuevos con creación exclusiva.
- **Scripts ya ejecutados, no se modifican:** `scripts/*_20261104.py` y `scripts/*_20261108.py`. Documentan lo que se corrió. La única excepción es el texto de la presentación (Tarea 6, paso 5), y solo con autorización explícita del investigador.
- **Todo resultado nuevo sobre la reserva es post hoc.** Se rotula `post_hoc_audit_not_model_selection` y nunca reemplaza las métricas oficiales de `scores.json`. La métrica principal sigue siendo `macro_network_rmse`.
- Constantes de la evaluación oficial:
  - umbral de inundación: `1.0` m³
  - bandas de volumen: `(24., 108., 344.)` m³
  - umbral de probabilidad: `0.5`
  - SHA-256 de la partición: `b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4`
- Agregar archivos `.py` bajo `swmm_resilience/` cambia el conjunto de hashes del contrato de los estudios Optuna (`implementation_sha256`). Los estudios ya terminaron. Cualquier intento de reanudarlos debe usar la versión congelada del código, no esta.
- **Tests:** `venv/Scripts/python.exe -m pytest ... -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-<tarea>`. El `--basetemp` debe ser corto, porque Windows tiene `LongPathsEnabled=0` (límite de 260 caracteres). No usar `-n`.
- Código e identificadores en inglés; informes en español.
- Un commit por tarea. **No hacer push** sin autorización explícita.
- Al final de cada commit agregar: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Antes de empezar

- Rama: `fix/test-suite-contracts-2026-10-27`. La Tarea 6 modifica el informe de auditoría, así que debe estar commiteado antes. Si `git status` lo muestra sin seguimiento, commitearlo junto con este plan:
  `git add docs/superpowers/reports/2026-11-08-auditoria-hidraulica-ml.md docs/superpowers/plans/2026-11-08-resolver-hallazgos-auditoria.md && git commit -m "docs: add hydraulic and ML audit and remediation plan"`
- Orden obligatorio: las Tareas 1, 2 y 3 son independientes entre sí. La Tarea 4 depende de las tres. El paso 5 de la Tarea 5 depende de la Tarea 4. La Tarea 6 depende de las Tareas 4 y 5.

## Mapa de archivos

| Archivo | Responsabilidad | Hallazgo |
|---|---|---|
| `swmm_resilience/ml/bench/holdout_audit.py` (nuevo) | RMSE macro, RMSE por forma y red, macro excluyendo una forma, subconjunto por continuidad | 1, 5 |
| `swmm_resilience/ml/bench/physical_checks.py` (nuevo) | Cota por volumen tributario, recorte de sensibilidad, monotonía por factor | 4 |
| `swmm_resilience/ml/bench/partition_coverage.py` (nuevo) | Volumen de entrada por corrida, cobertura reserva frente a desarrollo, sorteo estratificado para futuras reservas | 2 |
| `scripts/auditar_reserva_conjunta.py` (nuevo) | Verifica sellos, ejecuta los tres módulos y escribe `audit.json` | 1, 2, 4, 5 |
| `scripts/respaldar_artefactos_sellados.py` (nuevo) | ZIP con SHA-256 por archivo y verificación | 6 |
| `tests/ml/bench/test_holdout_audit.py`, `test_physical_checks.py`, `test_partition_coverage.py`, `tests/test_auditar_reserva_conjunta.py`, `tests/test_respaldar_artefactos_sellados.py` (nuevos) | Pruebas | — |
| `docs/superpowers/reports/2026-11-08-evaluacion-reserva-conjunta.md`, `2026-11-08-auditoria-hidraulica-ml.md`, `README.md` | Documentación | 1, 2, 4, 5, 6, 7 |

---

### Tarea 1: sensibilidad de la reserva por forma y por continuidad (hallazgos 1 y 5)

**Archivos:**
- Crear: `swmm_resilience/ml/bench/holdout_audit.py`
- Test: `tests/ml/bench/test_holdout_audit.py`

**Interfaces:**
- Consume: nada de otras tareas.
- Produce:
  - `macro_network_rmse(frame: pd.DataFrame) -> float`
  - `rmse_by_shape_network(frame) -> pd.DataFrame`, con columnas `network_id, shape_id, n_pairs, rmse_m3, sse_share`
  - `leave_one_shape_out_macro(frame) -> pd.DataFrame`, con columnas `excluded_shape, macro_network_rmse_m3`; la primera fila tiene `excluded_shape=None` y corresponde a todas las formas
  - `continuity_subset(frame, continuity_pct_by_run: pd.Series, max_abs_pct: float) -> pd.DataFrame`
  - `frame` debe tener `network_id, run_id, node_id, shape_id, factor_mult, vol_inundacion_m3, volume_routed_m3`

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/ml/bench/test_holdout_audit.py
import math

import pandas as pd
import pytest

from swmm_resilience.ml.bench.holdout_audit import (
    continuity_subset, leave_one_shape_out_macro, macro_network_rmse, rmse_by_shape_network)


def frame_with_errors():
    # Errors: net1/A=3, net1/B=0, net2/A=4, net2/B=0 (truth 10 m3 everywhere).
    rows = [(1, 1, "n1", "A", 1.0, 3.0), (1, 2, "n1", "B", 1.0, 0.0),
            (2, 3, "m1", "A", 1.0, 4.0), (2, 4, "m1", "B", 1.0, 0.0)]
    return pd.DataFrame([{"network_id": n, "run_id": r, "node_id": node, "shape_id": s,
                          "factor_mult": f, "vol_inundacion_m3": 10.0, "volume_routed_m3": 10.0 + e}
                         for n, r, node, s, f, e in rows])


def test_macro_rmse_averages_network_rmse():
    assert math.isclose(macro_network_rmse(frame_with_errors()),
                        (math.sqrt(4.5) + math.sqrt(8.0)) / 2, rel_tol=1e-12)


def test_rmse_by_shape_network_reports_share_of_squared_error():
    table = rmse_by_shape_network(frame_with_errors()).set_index(["network_id", "shape_id"])
    assert table.loc[(1, "A"), "rmse_m3"] == 3.0
    assert math.isclose(table.loc[(1, "A"), "sse_share"], 9 / 25)
    assert math.isclose(table.loc[(2, "A"), "sse_share"], 16 / 25)
    assert table.loc[(2, "B"), "sse_share"] == 0.0
    assert table.n_pairs.tolist() == [1, 1, 1, 1]


def test_leave_one_shape_out_reports_all_then_each_exclusion():
    table = leave_one_shape_out_macro(frame_with_errors())
    assert table.excluded_shape.tolist() == [None, "A", "B"]
    assert table.macro_network_rmse_m3.iloc[1] == 0.0
    assert table.macro_network_rmse_m3.iloc[2] == 3.5


def test_leave_one_shape_out_rejects_dropping_a_whole_network():
    frame = frame_with_errors()
    frame.loc[frame.network_id == 2, "shape_id"] = "only_net2"
    with pytest.raises(ValueError, match="whole network"):
        leave_one_shape_out_macro(frame)


def test_continuity_subset_keeps_runs_within_absolute_limit():
    continuity = pd.Series({1: 0.5, 2: 3.0, 3: -1.0, 4: 4.9})
    subset = continuity_subset(frame_with_errors(), continuity, 2.0)
    assert sorted(subset.run_id.unique()) == [1, 3]


def test_continuity_subset_rejects_missing_runs_and_bad_limit():
    with pytest.raises(ValueError, match="missing"):
        continuity_subset(frame_with_errors(), pd.Series({1: 0.5}), 2.0)
    with pytest.raises(ValueError, match="positive"):
        continuity_subset(frame_with_errors(), pd.Series({1: 0, 2: 0, 3: 0, 4: 0}), 0.0)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="lacks columns"):
        macro_network_rmse(frame_with_errors().drop(columns=["shape_id"]))
```

- [ ] **Paso 2: comprobar que fallan**

Correr: `venv/Scripts/python.exe -m pytest tests/ml/bench/test_holdout_audit.py -v -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-t1`
Esperado: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.holdout_audit'`.

- [ ] **Paso 3: implementar**

```python
# swmm_resilience/ml/bench/holdout_audit.py
"""Post hoc sensitivity of a sealed holdout evaluation; never re-selects models.

RMSE here is computed on routed volumes (dry prediction -> 0 m3), the same
definition as ``rmse_m3`` in the official holdout ``scores.json``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("network_id", "run_id", "node_id", "shape_id", "factor_mult",
                    "vol_inundacion_m3", "volume_routed_m3")


def _require(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"holdout frame lacks columns: {missing}")
    if frame.empty:
        raise ValueError("holdout frame is empty")


def _squared_error(frame: pd.DataFrame) -> np.ndarray:
    return (frame["volume_routed_m3"].to_numpy(float) - frame["vol_inundacion_m3"].to_numpy(float)) ** 2


def _rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(_squared_error(frame))))


def macro_network_rmse(frame: pd.DataFrame) -> float:
    """Mean of per-network RMSE, as in the agreed primary metric."""
    _require(frame)
    return float(np.mean([_rmse(group) for _, group in frame.groupby("network_id", sort=True)]))


def rmse_by_shape_network(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame)
    total = float(_squared_error(frame).sum())
    rows = []
    for (network, shape), group in frame.groupby(["network_id", "shape_id"], sort=True):
        sse = float(_squared_error(group).sum())
        rows.append({"network_id": int(network), "shape_id": str(shape), "n_pairs": int(len(group)),
                     "rmse_m3": _rmse(group), "sse_share": sse / total if total else 0.0})
    return pd.DataFrame(rows, columns=["network_id", "shape_id", "n_pairs", "rmse_m3", "sse_share"])


def leave_one_shape_out_macro(frame: pd.DataFrame) -> pd.DataFrame:
    """Macro RMSE with all shapes, then with each shape removed (descriptive only)."""
    _require(frame)
    networks = frame["network_id"].nunique()
    rows = [{"excluded_shape": None, "macro_network_rmse_m3": macro_network_rmse(frame)}]
    for shape in sorted(frame["shape_id"].unique()):
        rest = frame[frame["shape_id"] != shape]
        if rest.empty or rest["network_id"].nunique() != networks:
            raise ValueError(f"excluding {shape!r} removes a whole network")
        rows.append({"excluded_shape": str(shape), "macro_network_rmse_m3": macro_network_rmse(rest)})
    return pd.DataFrame(rows, columns=["excluded_shape", "macro_network_rmse_m3"])


def continuity_subset(frame: pd.DataFrame, continuity_pct_by_run: pd.Series,
                      max_abs_pct: float) -> pd.DataFrame:
    """Keep runs whose |continuity error| is within ``max_abs_pct`` percent."""
    _require(frame)
    if not (isinstance(max_abs_pct, (int, float)) and max_abs_pct > 0):
        raise ValueError("max_abs_pct must be positive")
    runs = sorted(int(run) for run in frame["run_id"].unique())
    known = continuity_pct_by_run.reindex(runs)
    if known.isna().any():
        raise ValueError("continuity is missing for holdout runs")
    keep = set(known[known.abs() <= max_abs_pct].index)
    return frame[frame["run_id"].isin(keep)]
```

- [ ] **Paso 4: comprobar que pasan**

Correr el mismo comando del paso 2. Esperado: 7 passed.

- [ ] **Paso 5: commit**

```bash
git add swmm_resilience/ml/bench/holdout_audit.py tests/ml/bench/test_holdout_audit.py
git commit -m "feat: add post hoc holdout sensitivity by shape and continuity"
```

---

### Tarea 2: comprobaciones físicas de las predicciones (hallazgo 4)

**Archivos:**
- Crear: `swmm_resilience/ml/bench/physical_checks.py`
- Test: `tests/ml/bench/test_physical_checks.py`

**Interfaces:**
- Consume: nada.
- Produce:
  - `tributary_bound_violations(frame, tolerance_m3=1.0) -> pd.DataFrame`, con las filas que violan la cota más la columna `ratio`
  - `cap_at_tributary_volume(volume, tributary) -> np.ndarray`
  - `total_monotonicity_violations(frame, value_column, truth_column="vol_inundacion_m3", tolerance_m3=1e-6) -> pd.DataFrame`, con columnas `network_id, shape_id, factor_mult, predicted_drop_m3`
  - `node_monotonicity_violations(frame, value_column, truth_column="vol_inundacion_m3", tolerance_m3=1.0) -> dict`, con claves `violations` y `steps`

**Fundamento:** el volumen inundado de un nodo no puede superar el agua que llega a él desde sus aportes aguas arriba (`volumen_tributario_integrado_m3`, que no admite nulos en `tabular_v4_21`). En la reserva, SWMM llega como máximo al 0,888 de esa cota y no la supera nunca. XGBoost la supera 0 veces y MLP 46 veces (hasta 2,27 veces). Las redes son árboles: la cota omite solo el almacenamiento inicial, por eso se usa una tolerancia.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/ml/bench/test_physical_checks.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.physical_checks import (
    cap_at_tributary_volume, node_monotonicity_violations,
    total_monotonicity_violations, tributary_bound_violations)


def series_frame(predicted, truth, factors=(1.0, 2.0, 3.0), network=1, shape="A", node="n1"):
    return pd.DataFrame({"network_id": network, "run_id": range(len(factors)), "node_id": node,
                         "shape_id": shape, "factor_mult": list(factors),
                         "volume_routed_m3": predicted, "vol_inundacion_m3": truth,
                         "volumen_tributario_integrado_m3": 100.0})


def test_bound_violations_use_tolerance_and_report_ratio():
    frame = series_frame([50.0, 100.5, 250.0], [0.0, 0.0, 0.0])
    found = tributary_bound_violations(frame, tolerance_m3=1.0)
    assert found.factor_mult.tolist() == [3.0]
    assert found.ratio.tolist() == [2.5]


def test_cap_never_exceeds_tributary_and_rejects_invalid_bound():
    assert cap_at_tributary_volume([5.0, 300.0], [100.0, 100.0]).tolist() == [5.0, 100.0]
    with pytest.raises(ValueError, match="finite"):
        cap_at_tributary_volume([1.0], [np.nan])
    with pytest.raises(ValueError, match="negative"):
        cap_at_tributary_volume([1.0], [-1.0])
    with pytest.raises(ValueError, match="shape"):
        cap_at_tributary_volume([1.0, 2.0], [1.0])


def test_total_monotonicity_flags_drop_only_when_truth_does_not_drop():
    frame = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 15.0])
    found = total_monotonicity_violations(frame, "volume_routed_m3")
    assert found.factor_mult.tolist() == [3.0]
    assert found.predicted_drop_m3.tolist() == [-10.0]
    truth_drops = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 8.0])
    assert total_monotonicity_violations(truth_drops, "volume_routed_m3").empty


def test_node_monotonicity_counts_steps_per_node_and_shape():
    a = series_frame([10.0, 30.0, 20.0], [5.0, 10.0, 15.0], node="n1")
    b = series_frame([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], node="n2")
    result = node_monotonicity_violations(pd.concat([a, b], ignore_index=True), "volume_routed_m3")
    assert result == {"violations": 1, "steps": 4}
```

- [ ] **Paso 2: comprobar que fallan**

Correr: `venv/Scripts/python.exe -m pytest tests/ml/bench/test_physical_checks.py -v -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-t2`
Esperado: FAIL con `ModuleNotFoundError`.

- [ ] **Paso 3: implementar**

```python
# swmm_resilience/ml/bench/physical_checks.py
"""Physical sanity checks for routed node flood volumes (diagnostic only).

Networks in this study are trees, so a node cannot flood more water than the
integrated inflow of itself and its upstream nodes; only the small initial
storage is outside that bound, hence the tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_KEYS = ["network_id", "run_id", "node_id", "shape_id", "factor_mult"]


def tributary_bound_violations(frame: pd.DataFrame, tolerance_m3: float = 1.0) -> pd.DataFrame:
    tributary = frame["volumen_tributario_integrado_m3"].astype(float)
    volume = frame["volume_routed_m3"].astype(float)
    mask = volume > tributary + tolerance_m3
    found = frame.loc[mask, _KEYS + ["volume_routed_m3", "volumen_tributario_integrado_m3"]].copy()
    found["ratio"] = volume[mask] / tributary[mask]
    return found.reset_index(drop=True)


def cap_at_tributary_volume(volume, tributary) -> np.ndarray:
    volume, tributary = np.asarray(volume, dtype=float), np.asarray(tributary, dtype=float)
    if volume.shape != tributary.shape:
        raise ValueError("volume and tributary must have the same shape")
    if not (np.isfinite(volume).all() and np.isfinite(tributary).all()):
        raise ValueError("volume and tributary must be finite")
    if (tributary < 0).any():
        raise ValueError("tributary volume cannot be negative")
    return np.minimum(volume, tributary)


def total_monotonicity_violations(frame: pd.DataFrame, value_column: str,
                                  truth_column: str = "vol_inundacion_m3",
                                  tolerance_m3: float = 1e-6) -> pd.DataFrame:
    """Factor steps where the predicted network total drops while SWMM does not."""
    totals = (frame.groupby(["network_id", "shape_id", "factor_mult"], sort=True)
              .agg(predicted=(value_column, "sum"), truth=(truth_column, "sum")).reset_index())
    grouped = totals.groupby(["network_id", "shape_id"], sort=False)
    totals["predicted_drop_m3"] = grouped["predicted"].diff()
    truth_step = grouped["truth"].diff()
    mask = (totals["predicted_drop_m3"] < -tolerance_m3) & (truth_step >= 0)
    return totals.loc[mask, ["network_id", "shape_id", "factor_mult", "predicted_drop_m3"]].reset_index(drop=True)


def node_monotonicity_violations(frame: pd.DataFrame, value_column: str,
                                 truth_column: str = "vol_inundacion_m3",
                                 tolerance_m3: float = 1.0) -> dict:
    """Count node-level factor steps where prediction drops and SWMM does not."""
    ordered = frame.sort_values(["network_id", "shape_id", "node_id", "factor_mult"])
    grouped = ordered.groupby(["network_id", "shape_id", "node_id"], sort=False)
    predicted_step = grouped[value_column].diff()
    truth_step = grouped[truth_column].diff()
    valid = predicted_step.notna()
    violations = valid & (predicted_step < -tolerance_m3) & (truth_step >= 0)
    return {"violations": int(violations.sum()), "steps": int(valid.sum())}
```

- [ ] **Paso 4: comprobar que pasan**

Correr el comando del paso 2. Esperado: 4 passed.

- [ ] **Paso 5: commit**

```bash
git add swmm_resilience/ml/bench/physical_checks.py tests/ml/bench/test_physical_checks.py
git commit -m "feat: add tributary bound and monotonicity diagnostics"
```

---

### Tarea 3: cobertura de volumen de entrada y sorteo estratificado (hallazgo 2)

**Archivos:**
- Crear: `swmm_resilience/ml/bench/partition_coverage.py`
- Test: `tests/ml/bench/test_partition_coverage.py`

**Interfaces:**
- Consume: nada.
- Produce:
  - `run_inflow_volumes(features) -> pd.DataFrame`, con columnas `network_id, run_id, shape_id, factor_mult, total_inflow_m3`. La entrada son filas nodo–corrida con `volumen_local_integrado_m3`.
  - `inflow_coverage(run_volumes, development_run_ids, holdout_run_ids) -> pd.DataFrame`, una fila por red con columnas `network_id, development_max_inflow_m3, development_max_shape, holdout_max_inflow_m3, holdout_max_shape, holdout_runs_above_development_max, max_ratio_to_development_max`.
  - `shape_representative_volume(run_volumes) -> pd.Series`, con índice `shape_id` y valor en [0, 1]: el máximo, sobre redes y factores, del volumen de entrada dividido por el máximo de su red.
  - `stratified_shape_draw(shape_volumes, k, seed, keep_top_in_development=1) -> list[str]`

**Fundamento:** la suma de `volumen_local_integrado_m3` por corrida coincide con el "External Inflow" de SWMM en las corridas elegibles (diferencia de 0,18 % o menos; auditoría, sección 8). Así no hace falta leer los RPT. El sorteo estratificado es **solo para un diseño futuro**: la reserva actual ya fue consultada y no se vuelve a sortear.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/ml/bench/test_partition_coverage.py
import pandas as pd
import pytest

from swmm_resilience.ml.bench.partition_coverage import (
    inflow_coverage, run_inflow_volumes, shape_representative_volume, stratified_shape_draw)


def node_rows():
    rows = []
    for run, network, shape, factor, per_node in [(1, 1, "dev_a", 1.0, 10.0), (2, 1, "dev_b", 1.0, 30.0),
                                                   (3, 1, "held", 1.0, 40.0), (4, 2, "dev_a", 1.0, 5.0),
                                                   (5, 2, "held", 1.0, 4.0)]:
        for node in ("x", "y"):
            rows.append({"network_id": network, "run_id": run, "node_id": node, "shape_id": shape,
                         "factor_mult": factor, "volumen_local_integrado_m3": per_node})
    return pd.DataFrame(rows)


def test_run_inflow_volumes_sums_nodes_per_run():
    volumes = run_inflow_volumes(node_rows()).set_index("run_id")
    assert volumes.loc[2, "total_inflow_m3"] == 60.0
    assert volumes.loc[2, "shape_id"] == "dev_b"


def test_run_inflow_volumes_rejects_inconsistent_run_identity_or_missing_volume():
    rows = node_rows()
    rows.loc[0, "shape_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        run_inflow_volumes(rows)
    rows = node_rows()
    rows.loc[0, "volumen_local_integrado_m3"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_inflow_volumes(rows)


def test_inflow_coverage_flags_holdout_above_development_max():
    table = inflow_coverage(run_inflow_volumes(node_rows()), [1, 2, 4], [3, 5]).set_index("network_id")
    assert table.loc[1, "development_max_inflow_m3"] == 60.0
    assert table.loc[1, "holdout_max_shape"] == "held"
    assert table.loc[1, "holdout_runs_above_development_max"] == 1
    assert table.loc[1, "max_ratio_to_development_max"] == pytest.approx(80 / 60)
    assert table.loc[2, "holdout_runs_above_development_max"] == 0


def test_inflow_coverage_rejects_overlapping_or_unknown_runs():
    volumes = run_inflow_volumes(node_rows())
    with pytest.raises(ValueError, match="overlap"):
        inflow_coverage(volumes, [1, 3], [3])
    with pytest.raises(ValueError, match="unknown"):
        inflow_coverage(volumes, [1, 99], [3])


def test_shape_representative_volume_is_normalised_per_network():
    representative = shape_representative_volume(run_inflow_volumes(node_rows()))
    assert representative["held"] == 1.0  # network 1 maximum
    assert representative["dev_a"] == 1.0  # network 2 maximum
    assert representative["dev_b"] == pytest.approx(60 / 80)


def test_stratified_draw_keeps_largest_in_development_and_is_reproducible():
    volumes = pd.Series({f"s{i}": float(i) for i in range(1, 11)})
    drawn = stratified_shape_draw(volumes, k=3, seed=42)
    assert drawn == stratified_shape_draw(volumes, k=3, seed=42)
    assert "s10" not in drawn
    strata = [{"s1", "s2", "s3"}, {"s4", "s5", "s6"}, {"s7", "s8", "s9"}]
    assert all(len(set(drawn) & stratum) == 1 for stratum in strata)


def test_stratified_draw_rejects_invalid_requests():
    with pytest.raises(ValueError, match="not enough"):
        stratified_shape_draw(pd.Series({"a": 1.0, "b": 2.0}), k=2, seed=1)
    with pytest.raises(ValueError, match="finite"):
        stratified_shape_draw(pd.Series({"a": float("nan"), "b": 2.0, "c": 3.0}), k=1, seed=1)
```

- [ ] **Paso 2: comprobar que fallan**

Correr: `venv/Scripts/python.exe -m pytest tests/ml/bench/test_partition_coverage.py -v -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-t3`
Esperado: FAIL con `ModuleNotFoundError`.

- [ ] **Paso 3: implementar**

```python
# swmm_resilience/ml/bench/partition_coverage.py
"""Inflow-volume coverage between development and held-out runs.

``stratified_shape_draw`` is for designing a *future* reservation; it must not
be used to redraw a holdout that has already been scored.
"""
from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import pandas as pd

_RUN_COLUMNS = ["network_id", "run_id", "shape_id", "factor_mult", "total_inflow_m3"]


def run_inflow_volumes(features: pd.DataFrame) -> pd.DataFrame:
    """Sum each run's local integrated node inflow volumes (m3)."""
    volume = features["volumen_local_integrado_m3"].astype(float)
    if not np.isfinite(volume).all():
        raise ValueError("volumen_local_integrado_m3 must be finite")
    grouped = features.assign(volumen_local_integrado_m3=volume).groupby("run_id", sort=True)
    identity = grouped[["network_id", "shape_id", "factor_mult"]].nunique()
    if (identity != 1).any().any():
        raise ValueError("each run must have a single network, shape and factor identity")
    first = grouped[["network_id", "shape_id", "factor_mult"]].first()
    result = first.assign(total_inflow_m3=grouped["volumen_local_integrado_m3"].sum()).reset_index()
    result["network_id"] = result["network_id"].astype(int)
    result["run_id"] = result["run_id"].astype(int)
    return result.loc[:, _RUN_COLUMNS]


def inflow_coverage(run_volumes: pd.DataFrame, development_run_ids: Iterable[int],
                    holdout_run_ids: Iterable[int]) -> pd.DataFrame:
    development, holdout = {int(r) for r in development_run_ids}, {int(r) for r in holdout_run_ids}
    if development & holdout:
        raise ValueError("development and holdout runs overlap")
    known = set(run_volumes["run_id"].astype(int))
    if not (development | holdout) <= known:
        raise ValueError("unknown run ids in development or holdout")
    rows = []
    for network, group in run_volumes.groupby("network_id", sort=True):
        dev = group[group["run_id"].isin(development)]
        held = group[group["run_id"].isin(holdout)]
        if dev.empty or held.empty:
            raise ValueError(f"network {network} lacks development or holdout runs")
        dev_top, held_top = dev.loc[dev["total_inflow_m3"].idxmax()], held.loc[held["total_inflow_m3"].idxmax()]
        dev_max = float(dev_top["total_inflow_m3"])
        rows.append({"network_id": int(network), "development_max_inflow_m3": dev_max,
                     "development_max_shape": str(dev_top["shape_id"]),
                     "holdout_max_inflow_m3": float(held_top["total_inflow_m3"]),
                     "holdout_max_shape": str(held_top["shape_id"]),
                     "holdout_runs_above_development_max": int((held["total_inflow_m3"] > dev_max).sum()),
                     "max_ratio_to_development_max": float(held_top["total_inflow_m3"]) / dev_max})
    return pd.DataFrame(rows)


def shape_representative_volume(run_volumes: pd.DataFrame) -> pd.Series:
    """Max over networks/factors of a shape's inflow, normalised by its network maximum."""
    network_max = run_volumes.groupby("network_id")["total_inflow_m3"].transform("max")
    normalised = run_volumes.assign(normalised=run_volumes["total_inflow_m3"] / network_max)
    return normalised.groupby("shape_id")["normalised"].max().astype(float)


def stratified_shape_draw(shape_volumes: pd.Series, k: int, seed: int,
                          keep_top_in_development: int = 1) -> list[str]:
    """Draw one shape per volume stratum, always keeping the largest shapes in development."""
    if k < 1 or keep_top_in_development < 0:
        raise ValueError("k must be >= 1 and keep_top_in_development >= 0")
    volumes = shape_volumes.astype(float)
    if volumes.index.duplicated().any() or not np.isfinite(volumes.to_numpy()).all():
        raise ValueError("shape volumes must be unique and finite")
    ordered = sorted(volumes.index, key=lambda shape: (volumes[shape], str(shape)))
    candidates = ordered[:len(ordered) - keep_top_in_development]
    if len(candidates) < k:
        raise ValueError("not enough candidate shapes for the requested draw")
    rng = random.Random(seed)
    strata = np.array_split(np.array(candidates, dtype=object), k)
    return sorted(str(rng.choice(list(stratum))) for stratum in strata)
```

- [ ] **Paso 4: comprobar que pasan**

Correr el comando del paso 2. Esperado: 7 passed.

- [ ] **Paso 5: commit**

```bash
git add swmm_resilience/ml/bench/partition_coverage.py tests/ml/bench/test_partition_coverage.py
git commit -m "feat: add inflow coverage check and stratified shape draw for future holdouts"
```

---

### Tarea 4: script de auditoría post hoc sobre la reserva sellada (hallazgos 1, 2, 4 y 5)

**Archivos:**
- Crear: `scripts/auditar_reserva_conjunta.py`
- Test: `tests/test_auditar_reserva_conjunta.py`

**Interfaces:**
- Consume:
  - de la Tarea 1: `macro_network_rmse`, `rmse_by_shape_network`, `leave_one_shape_out_macro`, `continuity_subset`
  - de la Tarea 2: `tributary_bound_violations`, `cap_at_tributary_volume`, `total_monotonicity_violations`, `node_monotonicity_violations`
  - de la Tarea 3: `run_inflow_volumes`, `inflow_coverage`
  - del repositorio: `scripts.prepare_joint_dataset.development_run_ids(package, partition) -> list[int]`, `verify_partition_hash(path, expected) -> None` y `swmm_resilience.ml.bench.evaluate._nested_reporting_entry(y_true, y_pred, y_prob, vol_true, vol_pred, bands, threshold) -> dict`
- Produce:
  - `verify_sealed_evaluation(directory: Path) -> dict`, con claves `status, scores, seal`
  - `assemble_family_frame(inputs, truth, predictions) -> pd.DataFrame`
  - el archivo `<output>/audit.json`, con `schema: "joint_holdout_post_hoc_audit_v1"`

**Formatos sellados que lee** (`outputs/experiments/evaluacion_reserva_conjunta_20261108_v1/`):
- `status.json`: `{"status": "holdout_scored", "scores_sha256": ...}`
- `scores.json`: `seal_sha256`, `truth_sha256` y `models.<familia>.macro_network_rmse_m3`
- `prediction_seal.json`: `inputs_sha256`, `source_database_sha256`, `holdout_run_ids` y `models.<familia>.predictions_sha256`
- `holdout_inputs.parquet`: `KEY_COLUMNS` más las features, incluidas `shape_id`, `factor_mult` y `volumen_*`
- `truth.parquet`: `network_id, run_id, node_id, inunda, vol_inundacion_m3`
- `<familia>_predictions.parquet`: `KEY_COLUMNS` más `y_prob_clf, y_pred_clf, volume_routed_m3`

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_auditar_reserva_conjunta.py
import hashlib
import json

import pandas as pd
import pytest

KEYS = ["network_id", "run_id", "node_id"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_directory(tmp_path):
    inputs = pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "shape_id": ["s"],
                           "factor_mult": [1.0], "volumen_tributario_integrado_m3": [50.0],
                           "volumen_local_integrado_m3": [5.0]})
    truth = pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "inunda": [1],
                          "vol_inundacion_m3": [10.0]})
    inputs.to_parquet(tmp_path / "holdout_inputs.parquet", index=False)
    truth.to_parquet(tmp_path / "truth.parquet", index=False)
    models = {}
    for family in ("xgboost", "mlp"):
        path = tmp_path / f"{family}_predictions.parquet"
        pd.DataFrame({"network_id": [1], "run_id": [7], "node_id": ["a"], "shape_id": ["s"],
                      "factor_mult": [1.0], "y_prob_clf": [0.9], "y_pred_clf": [1],
                      "volume_routed_m3": [12.0]}).to_parquet(path, index=False)
        models[family] = {"predictions_sha256": _sha(path)}
    (tmp_path / "prediction_seal.json").write_text(json.dumps(
        {"inputs_sha256": _sha(tmp_path / "holdout_inputs.parquet"), "models": models}), encoding="utf-8")
    (tmp_path / "scores.json").write_text(json.dumps(
        {"seal_sha256": _sha(tmp_path / "prediction_seal.json"),
         "truth_sha256": _sha(tmp_path / "truth.parquet")}), encoding="utf-8")
    (tmp_path / "status.json").write_text(json.dumps(
        {"status": "holdout_scored", "scores_sha256": _sha(tmp_path / "scores.json")}), encoding="utf-8")
    return tmp_path


def test_verify_sealed_evaluation_accepts_intact_files(tmp_path):
    from scripts.auditar_reserva_conjunta import verify_sealed_evaluation
    sealed = verify_sealed_evaluation(sealed_directory(tmp_path))
    assert set(sealed) == {"status", "scores", "seal"}


@pytest.mark.parametrize("name, message", [
    ("scores.json", "scores"), ("truth.parquet", "truth"),
    ("holdout_inputs.parquet", "inputs"), ("mlp_predictions.parquet", "mlp predictions")])
def test_verify_sealed_evaluation_detects_tampering(tmp_path, name, message):
    from scripts.auditar_reserva_conjunta import verify_sealed_evaluation
    directory = sealed_directory(tmp_path)
    with (directory / name).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match=message):
        verify_sealed_evaluation(directory)


def test_assemble_family_frame_joins_one_to_one_and_rejects_partial_coverage(tmp_path):
    from scripts.auditar_reserva_conjunta import assemble_family_frame
    directory = sealed_directory(tmp_path)
    inputs = pd.read_parquet(directory / "holdout_inputs.parquet")
    truth = pd.read_parquet(directory / "truth.parquet")
    predictions = pd.read_parquet(directory / "xgboost_predictions.parquet")
    frame = assemble_family_frame(inputs, truth, predictions)
    assert frame.loc[0, "volume_routed_m3"] == 12.0 and frame.loc[0, "vol_inundacion_m3"] == 10.0
    assert frame.loc[0, "shape_id"] == "s"
    extra = pd.concat([truth, truth.assign(node_id="b")], ignore_index=True)
    with pytest.raises(ValueError, match="same pairs"):
        assemble_family_frame(inputs, extra, predictions)
```

- [ ] **Paso 2: comprobar que fallan**

Correr: `venv/Scripts/python.exe -m pytest tests/test_auditar_reserva_conjunta.py -v -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-t4`
Esperado: FAIL con `ModuleNotFoundError: No module named 'scripts.auditar_reserva_conjunta'`.

- [ ] **Paso 3: implementar**

Nota sobre el orden de verificación: cada mensaje de error nombra el archivo alterado. Si cambian `scores.json`, `status.json` lo detecta. Si cambia `truth.parquet`, lo detecta `scores.json`. Si cambian `holdout_inputs.parquet` o las predicciones, lo detecta `prediction_seal.json`.

```python
# scripts/auditar_reserva_conjunta.py
"""Post hoc audit of the sealed joint holdout; never refits, reselects or rescores officially."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.ml.bench.evaluate import _nested_reporting_entry
from swmm_resilience.ml.bench.holdout_audit import (
    continuity_subset, leave_one_shape_out_macro, macro_network_rmse, rmse_by_shape_network)
from swmm_resilience.ml.bench.partition_coverage import inflow_coverage, run_inflow_volumes
from swmm_resilience.ml.bench.physical_checks import (
    cap_at_tributary_volume, node_monotonicity_violations, total_monotonicity_violations,
    tributary_bound_violations)

KEYS = ["network_id", "run_id", "node_id"]
FAMILIES = ("xgboost", "mlp")
BANDS = (24.0, 108.0, 344.0)
FLOOD_THRESHOLD_M3 = 1.0
CONTINUITY_LIMITS_PCT = (2.0, 1.0)


def sha256(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def verify_sealed_evaluation(directory: Path) -> dict:
    directory = Path(directory)
    status = _read_json(directory / "status.json")
    if status.get("status") != "holdout_scored" or sha256(directory / "scores.json") != status.get("scores_sha256"):
        raise ValueError("sealed scores changed")
    scores = _read_json(directory / "scores.json")
    if sha256(directory / "prediction_seal.json") != scores.get("seal_sha256"):
        raise ValueError("sealed prediction seal changed")
    if sha256(directory / "truth.parquet") != scores.get("truth_sha256"):
        raise ValueError("sealed truth changed")
    seal = _read_json(directory / "prediction_seal.json")
    if sha256(directory / "holdout_inputs.parquet") != seal.get("inputs_sha256"):
        raise ValueError("sealed inputs changed")
    for family in FAMILIES:
        if sha256(directory / f"{family}_predictions.parquet") != seal["models"][family]["predictions_sha256"]:
            raise ValueError(f"sealed {family} predictions changed")
    return {"status": status, "scores": scores, "seal": seal}


def assemble_family_frame(inputs: pd.DataFrame, truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if not (len(inputs) == len(truth) == len(predictions)):
        raise ValueError("sealed inputs, predictions and truth do not cover the same pairs")
    frame = (predictions.loc[:, KEYS + ["y_prob_clf", "y_pred_clf", "volume_routed_m3"]]
             .merge(inputs.loc[:, KEYS + ["shape_id", "factor_mult", "volumen_tributario_integrado_m3"]],
                    on=KEYS, how="inner", validate="one_to_one")
             .merge(truth.loc[:, KEYS + ["inunda", "vol_inundacion_m3"]],
                    on=KEYS, how="inner", validate="one_to_one"))
    if len(frame) != len(predictions):
        raise ValueError("sealed inputs, predictions and truth do not cover the same pairs")
    return frame


def _metrics_by_network(frame: pd.DataFrame) -> dict:
    return {str(network): _nested_reporting_entry(group.inunda, group.y_pred_clf, group.y_prob_clf,
                                                  group.vol_inundacion_m3, group.volume_routed_m3,
                                                  BANDS, FLOOD_THRESHOLD_M3)
            for network, group in frame.groupby("network_id", sort=True)}


def audit_family(frame: pd.DataFrame, official_macro: float, continuity: pd.Series) -> dict:
    recomputed = macro_network_rmse(frame)
    if not np.isclose(recomputed, official_macro, rtol=1e-9, atol=1e-9):
        raise ValueError("macro RMSE does not reproduce the sealed score")
    violations = tributary_bound_violations(frame)
    capped = frame.assign(volume_routed_m3=cap_at_tributary_volume(
        frame["volume_routed_m3"], frame["volumen_tributario_integrado_m3"]))
    sensitivity = {}
    for limit in CONTINUITY_LIMITS_PCT:
        subset = continuity_subset(frame, continuity, limit)
        sensitivity[f"max_abs_{limit:g}_pct"] = {
            "runs": int(subset["run_id"].nunique()), "pairs": int(len(subset)),
            "macro_network_rmse_m3": macro_network_rmse(subset), "by_network": _metrics_by_network(subset)}
    return {"macro_network_rmse_m3": recomputed,
            "by_shape_network": _records(rmse_by_shape_network(frame)),
            "leave_one_shape_out": _records(leave_one_shape_out_macro(frame)),
            "continuity_sensitivity": sensitivity,
            "tributary_bound": {"tolerance_m3": 1.0, "violations": int(len(violations)),
                                "max_ratio": float(violations["ratio"].max()) if len(violations) else None,
                                "capped_macro_network_rmse_m3": macro_network_rmse(capped)},
            "total_monotonicity_violations": _records(total_monotonicity_violations(frame, "volume_routed_m3")),
            "node_monotonicity": node_monotonicity_violations(frame, "volume_routed_m3")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("evaluation", "package", "partition", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--partition-sha256", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for frozen in (args.evaluation.resolve(), args.package.resolve()):
        if output == frozen or frozen in output.parents:
            raise ValueError("output must be outside sealed inputs")
    sealed = verify_sealed_evaluation(args.evaluation)
    verify_partition_hash(args.partition, args.partition_sha256)
    database = args.package / "joint.sqlite3"
    if sha256(database) != sealed["seal"]["source_database_sha256"]:
        raise ValueError("joint database differs from the sealed evaluation")
    development = development_run_ids(args.package, args.partition)
    holdout = [int(run) for run in sealed["seal"]["holdout_run_ids"]]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        marks = ",".join("?" for _ in holdout)
        continuity = pd.read_sql_query(
            f"SELECT run_id, continuity_error_pct FROM runs WHERE run_id IN ({marks})",
            connection, params=holdout).set_index("run_id")["continuity_error_pct"]
        dev_marks = ",".join("?" for _ in development)
        dev_features = pd.read_sql_query(
            "SELECT network_id, run_id, shape_id, factor_mult, volumen_local_integrado_m3 "
            f"FROM training_samples_v4_21 WHERE run_id IN ({dev_marks})", connection, params=development)
    inputs = pd.read_parquet(args.evaluation / "holdout_inputs.parquet")
    truth = pd.read_parquet(args.evaluation / "truth.parquet")
    report = {"schema": "joint_holdout_post_hoc_audit_v1",
              "status": "post_hoc_audit_not_model_selection",
              "scope": "descriptive sensitivity of sealed predictions; official metrics stay in scores.json",
              "evaluation_scores_sha256": sealed["status"]["scores_sha256"],
              "source_database_sha256": sealed["seal"]["source_database_sha256"],
              "partition_sha256": args.partition_sha256, "models": {}}
    for family in FAMILIES:
        predictions = pd.read_parquet(args.evaluation / f"{family}_predictions.parquet")
        frame = assemble_family_frame(inputs, truth, predictions)
        official = float(sealed["scores"]["models"][family]["macro_network_rmse_m3"])
        report["models"][family] = audit_family(frame, official, continuity)
    volumes = pd.concat([run_inflow_volumes(dev_features), run_inflow_volumes(inputs)], ignore_index=True)
    report["inflow_coverage"] = _records(inflow_coverage(volumes, development, holdout))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "audit.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output / "audit.json")}))


if __name__ == "__main__":
    main()
```

- [ ] **Paso 4: comprobar que pasan los tests**

Correr el comando del paso 2. Esperado: 6 passed.

- [ ] **Paso 5: ejecutar sobre los artefactos reales**

```bash
venv/Scripts/python.exe scripts/auditar_reserva_conjunta.py \
  --evaluation outputs/experiments/evaluacion_reserva_conjunta_20261108_v1 \
  --package outputs/experiments/joint_chico_sur_norte_20261104_v1 \
  --partition outputs/experiments/particion_chico_sur_norte_20261104.json \
  --partition-sha256 b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4 \
  --output outputs/experiments/auditoria_reserva_conjunta_20261108_v1
```

Esperado: `{"status": "post_hoc_audit_not_model_selection", ...}`. Comprobar que `audit.json` reproduce las cifras de la auditoría (si alguna difiere, **detenerse** y reportar, sin ajustar el código para forzarla):

| Clave | XGBoost | MLP |
|---|---:|---:|
| `macro_network_rmse_m3` | 234,54 | 184,63 |
| `leave_one_shape_out` con `excluded_shape="shape_quasi_uniform_block"` | 66,68 | 87,23 |
| `tributary_bound.violations` | 0 | 46 |
| `total_monotonicity_violations` (filas) | 0 | 1 (red 1, factor 5,0) |
| `node_monotonicity.violations` / `steps` | 1.242 / 101.972 | 63 / 101.972 |

En `inflow_coverage`, la red 1 debe tener `holdout_max_shape="shape_quasi_uniform_block"`, `development_max_shape="shape_sustained_plateau_14h"` y `max_ratio_to_development_max` ≈ 1,16. La red 2 debe dar ≈ 1,16.

`outputs/` está en `.gitignore`: el `audit.json` no se commitea, y su respaldo se trata en la Tarea 5.

- [ ] **Paso 6: commit**

```bash
git add scripts/auditar_reserva_conjunta.py tests/test_auditar_reserva_conjunta.py
git commit -m "feat: add post hoc audit script for the sealed joint holdout"
```

---

### Tarea 5: respaldo verificable de los artefactos sellados (hallazgo 6)

**Archivos:**
- Crear: `scripts/respaldar_artefactos_sellados.py`
- Test: `tests/test_respaldar_artefactos_sellados.py`

**Interfaces:**
- Consume: nada.
- Produce:
  - `build_archive(sources: list[Path], archive: Path) -> dict`. Escribe `archive` y `<archive sin .zip>.manifest.json` con creación exclusiva.
  - `verify_archive(archive: Path, manifest: dict) -> None`

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_respaldar_artefactos_sellados.py
import json

import pytest


def sources(tmp_path):
    folder = tmp_path / "sealed_eval"
    (folder / "nested").mkdir(parents=True)
    (folder / "scores.json").write_text('{"a": 1}', encoding="utf-8")
    (folder / "nested" / "truth.parquet").write_bytes(b"\x00\x01binary")
    single = tmp_path / "partition.json"
    single.write_text('{"p": 2}', encoding="utf-8")
    return [folder, single]


def test_build_and_verify_archive_round_trip(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive, verify_archive
    archive = tmp_path / "out" / "backup.zip"
    archive.parent.mkdir()
    manifest = build_archive(sources(tmp_path), archive)
    assert sorted(entry["path"] for entry in manifest["files"]) == [
        "partition.json", "sealed_eval/nested/truth.parquet", "sealed_eval/scores.json"]
    saved = json.loads((tmp_path / "out" / "backup.manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    verify_archive(archive, manifest)


def test_verify_detects_corrupted_archive_and_member_mismatch(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive, verify_archive
    archive = tmp_path / "backup.zip"
    manifest = build_archive(sources(tmp_path), archive)
    altered = json.loads(json.dumps(manifest))
    altered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member hash"):
        verify_archive(archive, altered)
    with archive.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="archive hash"):
        verify_archive(archive, manifest)


def test_build_refuses_existing_output_wal_and_duplicate_names(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive
    paths = sources(tmp_path)
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"")
    with pytest.raises(FileExistsError):
        build_archive(paths, archive)
    (paths[0] / "joint.sqlite3-wal").write_bytes(b"pending")
    with pytest.raises(ValueError, match="WAL"):
        build_archive(paths, tmp_path / "other.zip")
    (paths[0] / "joint.sqlite3-wal").unlink()
    with pytest.raises(ValueError, match="distinct"):
        build_archive([paths[0], paths[0]], tmp_path / "third.zip")
```

- [ ] **Paso 2: comprobar que fallan**

Correr: `venv/Scripts/python.exe -m pytest tests/test_respaldar_artefactos_sellados.py -v -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-t5`
Esperado: FAIL con `ModuleNotFoundError`.

- [ ] **Paso 3: implementar**

```python
# scripts/respaldar_artefactos_sellados.py
"""Archive sealed artifacts with per-file SHA-256 and verify the archive; never modifies sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return _sha256_stream(handle)


def _members(source: Path) -> list[tuple[Path, str]]:
    if source.is_file():
        members = [(source, source.name)]
    else:
        members = [(path, f"{source.name}/{path.relative_to(source).as_posix()}")
                   for path in sorted(source.rglob("*")) if path.is_file()]
    for path, _ in members:
        if path.name.endswith("-wal") and path.stat().st_size:
            raise ValueError(f"nonempty SQLite WAL cannot be archived by hash: {path}")
    return members


def _manifest_path(archive: Path) -> Path:
    return archive.with_suffix(".manifest.json")


def build_archive(sources: list[Path], archive: Path) -> dict:
    archive = Path(archive)
    if archive.exists() or _manifest_path(archive).exists():
        raise FileExistsError(archive)
    resolved = [Path(source).resolve(strict=True) for source in sources]
    if len({source.name for source in resolved}) != len(resolved):
        raise ValueError("source names must be distinct inside the archive")
    members = [member for source in resolved for member in _members(source)]
    entries = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for path, name in members:
            before = _sha256(path)
            bundle.write(path, name)
            if _sha256(path) != before:
                raise ValueError(f"source changed while archiving: {path}")
            entries.append({"path": name, "sha256": before, "bytes": path.stat().st_size})
    manifest = {"schema": "sealed_artifact_backup_v1", "archive": archive.name,
                "archive_sha256": _sha256(archive), "files": entries}
    with _manifest_path(archive).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def verify_archive(archive: Path, manifest: dict) -> None:
    archive = Path(archive)
    if _sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(entry["path"] for entry in manifest["files"]):
            raise ValueError("archive member list mismatch")
        for entry in manifest["files"]:
            with bundle.open(entry["path"]) as member:
                if _sha256_stream(member) != entry["sha256"]:
                    raise ValueError(f"member hash mismatch: {entry['path']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_archive(args.source, args.archive)
    verify_archive(args.archive, manifest)
    print(json.dumps({"archive": str(args.archive), "archive_sha256": manifest["archive_sha256"],
                      "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Paso 4: comprobar que pasan**

Correr el comando del paso 2. Esperado: 3 passed.

- [ ] **Paso 5: crear el respaldo real** (después de la Tarea 4, para incluir `audit.json`)

```bash
mkdir -p outputs/backups
venv/Scripts/python.exe scripts/respaldar_artefactos_sellados.py \
  --source outputs/experiments/evaluacion_reserva_conjunta_20261108_v1 \
  --source outputs/experiments/joint_chico_sur_norte_20261104_v1 \
  --source outputs/experiments/particion_chico_sur_norte_20261104.json \
  --source outputs/experiments/reserva_formas_chico_sur_norte_20261104.json \
  --source outputs/experiments/auditoria_reserva_conjunta_20261108_v1 \
  --archive outputs/backups/33_reserva_conjunta_y_base_20261108.zip
```

Esperado: un JSON con `archive_sha256` y `files` > 0, y la verificación sin errores. Anotar el SHA-256 para la Tarea 6.

- [ ] **Paso 6: subida a Drive (lo hace el investigador)**

Pedir al investigador que suba `33_reserva_conjunta_y_base_20261108.zip` y su `.manifest.json` a la carpeta privada de Drive, junto a los respaldos `27_`, `29_` y `32_`, y que confirme. No marcar el hallazgo 6 como resuelto sin esa confirmación.

- [ ] **Paso 7: commit**

```bash
git add scripts/respaldar_artefactos_sellados.py tests/test_respaldar_artefactos_sellados.py
git commit -m "feat: add verifiable archive script for sealed artifacts"
```

---

### Tarea 6: documentación (hallazgos 1, 2, 4, 5, 6 y 7)

**Archivos:**
- Modificar: `docs/superpowers/reports/2026-11-08-evaluacion-reserva-conjunta.md`
- Modificar: `docs/superpowers/reports/2026-11-08-auditoria-hidraulica-ml.md`
- Modificar: `README.md` (sección `## Verificacion`)
- Crear: `docs/superpowers/reports/2026-11-08-resolucion-hallazgos-auditoria.md` (documento de resolución por hallazgo, pedido expresamente por el investigador)

**Interfaces:**
- Consume: `outputs/experiments/auditoria_reserva_conjunta_20261108_v1/audit.json` (Tarea 4), el SHA-256 del ZIP (Tarea 5), los commits de las Tareas 1 a 5 (`git log --oneline`) y el registro de avance `.superpowers/sdd/2026-11-08-resolver-hallazgos-auditoria/progress.md` (las líneas `Ruling:`).

- [ ] **Paso 1: corregir la conclusión del informe de la reserva (hallazgo 1)**

En `2026-11-08-evaluacion-reserva-conjunta.md`, reemplazar:

```
En esta reserva MLP tiene menor error de volumen; XGBoost obtiene mayor precisión y recall de clasificación. Son métricas distintas y no establecen un ganador universal para todos los objetivos.
```

por:

```
En esta reserva MLP tiene menor RMSE macro de volumen y XGBoost obtiene mayor precisión y recall de clasificación. **La ventaja de MLP en volumen depende de una sola forma:** sin `shape_quasi_uniform_block`, el RMSE macro es 66,68 m³ para XGBoost y 87,23 m³ para MLP. Por forma y red el resultado está repartido (tabla siguiente). Este análisis se hizo después de ver los resultados: es descriptivo, no cambia la métrica principal acordada ni selecciona un modelo.
```

Inmediatamente después, insertar la tabla de RMSE por forma y red copiada de `audit.json` → `models.<familia>.by_shape_network` (columnas: forma, XGB Sur, XGB Norte, MLP Sur, MLP Norte; un decimal, coma decimal), y la línea: `Fuente: outputs/experiments/auditoria_reserva_conjunta_20261108_v1/audit.json (scripts/auditar_reserva_conjunta.py).`

- [ ] **Paso 2: añadir al mismo informe la sección "Sensibilidad post hoc" (hallazgos 2, 4 y 5)**

Agregar antes de `## Figuras` una sección `## Sensibilidad post hoc` con tres párrafos y cifras tomadas de `audit.json`:

1. **Cobertura:** de `inflow_coverage`, por red, la forma con más entrada en desarrollo y en la reserva, y `max_ratio_to_development_max`. Indicar que el sorteo fue correcto y fijado de antemano, que por azar dejó en la reserva la forma de mayor volumen, y que `stratified_shape_draw` queda disponible para un diseño futuro, sin usarse para volver a sortear esta reserva.
2. **Coherencia física:** `tributary_bound.violations` y `max_ratio` por modelo; `capped_macro_network_rmse_m3` rotulado explícitamente como "recorte post hoc, no adoptado"; las violaciones de monotonía total y nodal. Explicar en una frase que el regresor aprende `log1p(volumen)`, de modo que el MLP extrapola de forma exponencial en m³ y XGBoost se satura.
3. **Continuidad:** de `continuity_sensitivity`, el número de corridas y el RMSE macro con |continuidad| ≤ 2 % y ≤ 1 %, más precisión y recall por red. Indicar que la QA admite hasta 5 %.

- [ ] **Paso 3: registrar el estado en la auditoría**

En `2026-11-08-auditoria-hidraulica-ml.md`, agregar debajo del título una sección `## Estado (actualizado)` con una línea por hallazgo:

- 1, 2, 4 y 5: atendidos con análisis post hoc (`scripts/auditar_reserva_conjunta.py`) y documentados en el informe de la reserva. El reentrenamiento queda pendiente de decisión (ver "Decisiones pendientes" en el plan).
- 3: **fuera de alcance por decisión del investigador (8 de noviembre)**. Se considera parte del diseño de la red; se mantiene como está.
- 6: respaldo creado (`33_reserva_conjunta_y_base_20261108.zip`, con el SHA-256 de la Tarea 5). Indicar si la subida a Drive está confirmada o pendiente.
- 7: se retira el punto de `with sqlite3.connect(...)`. Al revisarlo, la conexión vive dentro de una función o un bucle y CPython la cierra al liberar el objeto, así que no queda abierta de forma persistente y no requiere cambios. Los demás puntos ya estaban documentados. La longitud de ruta de los tests se documenta en el README.

- [ ] **Paso 3b: documento de resolución por hallazgo**

Crear `docs/superpowers/reports/2026-11-08-resolucion-hallazgos-auditoria.md`, en español y escrito para el siguiente agente o para el investigador. Debe empezar con un párrafo de alcance (qué auditoría, qué plan y qué rama) y una tabla resumen (hallazgo, estado, commits y dónde está la evidencia). Después, una sección por hallazgo (1, 2, 3, 4, 5, 6 y 7), cada una con exactamente estos cuatro subtítulos:

1. **Problema:** qué se observó, con las cifras de la auditoría, y **por qué ocurre** (el mecanismo). Ejemplos: el RMSE macro es una suma de errores cuadrados dominada por una forma; ReLU sobre `log1p` extrapola de forma exponencial en m³; los árboles se saturan.
2. **Plan:** qué se decidió hacer y qué no, y por qué. Incluir las restricciones: todo es post hoc, la reserva no se reutiliza ni se vuelve a sortear, y no hay reentrenamiento.
3. **Solución aplicada:** archivos y funciones creados (con enlaces markdown relativos), los tests que la cubren, el comando para reproducirla y las cifras resultantes de `audit.json` (o el SHA-256 del ZIP para el hallazgo 6).
4. **Qué queda pendiente y cómo continuar:** decisiones abiertas y pasos concretos para quien retome.

El hallazgo 3 se documenta con "Problema" completo y su mecanismo (punto cero agregado en `last_time + drain_down_hours`, interpolación lineal de SWMM, 175 corridas con 16 % a 41 % más entrada, todas excluidas por la QA). Su "Plan" es: **fuera de alcance por decisión del investigador del 8 de noviembre; se considera diseño de la red y se mantiene**. En "Pendiente" va la corrección propuesta en la auditoría, por si se retoma. El hallazgo 7 explica por qué se retiró el punto de SQLite. Terminar con una sección "Decisiones tomadas durante la ejecución" que copie, tal cual, las líneas `Ruling:` del registro de avance, y con la sección "Decisiones pendientes" de este plan.

- [ ] **Paso 4: nota de tests en el README (hallazgo 7)**

En `README.md`, reemplazar:

````
## Verificacion

```bash
python -m pytest tests -v
python -m compileall main.py swmm_resilience
```
````

por:

````
## Verificacion

```bash
python -m pytest tests -v --basetemp C:/Users/<usuario>/AppData/Local/Temp/pt
python -m compileall main.py swmm_resilience
```

En Windows con `LongPathsEnabled=0`, usar un `--basetemp` corto. Algunos tests generan archivos `source-<sha256>.bin` y una ruta temporal larga supera el límite de 260 caracteres (`FileNotFoundError [WinError 3]`). La alternativa es habilitar las rutas largas en Windows, lo que requiere permisos de administrador. `pytest-xdist` no está instalado: no usar `-n`.
````

- [ ] **Paso 5: revisar la presentación (no regenerarla sin permiso)**

`scripts/build_weekly_2026_11_08_joint_deck.py:182` contiene el texto `'MLP logró menor RMSE de volumen en esta reserva.'`. **Preguntar al investigador** si quiere corregir esa diapositiva y regenerar el `.pptx` (es un archivo binario versionado de unos 15 MB). No editar el script ni regenerar sin una respuesta afirmativa. Si la respuesta es sí, cambiar el texto a `'MLP logró menor RMSE macro, pero por una sola forma; sin ella gana XGBoost.'`, regenerar y registrar el nuevo SHA-256 en `2026-11-08-presentacion-y-diagnostico.md`.

- [ ] **Paso 6: suite completa**

Correr: `venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp C:/Users/lf.fajardo/AppData/Local/Temp/pt-full`
Esperado: 0 failed. Hay 1.706 tests aprobados antes del plan, más los 27 nuevos. Si algo falla, usar superpowers:systematic-debugging antes de tocar nada.

- [ ] **Paso 7: commit**

```bash
git add docs/superpowers/reports/2026-11-08-evaluacion-reserva-conjunta.md docs/superpowers/reports/2026-11-08-auditoria-hidraulica-ml.md docs/superpowers/reports/2026-11-08-resolucion-hallazgos-auditoria.md README.md
git commit -m "docs: report post hoc holdout sensitivity and audit status"
```

---

## Decisiones pendientes (no son tareas de este plan)

Cambian modelos o protocolo y requieren aprobación explícita del investigador antes de planificarse:

1. **Adoptar el recorte por volumen tributario en la inferencia** (`cap_at_tributary_volume`). Cambiaría las predicciones oficiales de los modelos futuros. No debe aplicarse con efecto retroactivo a la reserva ya consultada.
2. **Reentrenar con restricciones de monotonía** (`monotone_constraints` en XGBoost sobre las variables de volumen) o con otra representación del volumen para el MLP. Requiere una prueba independiente nueva, porque la reserva actual ya se consultó.
3. **Diseñar una reserva nueva** con `stratified_shape_draw` y `inflow_coverage` comprobados antes del sorteo, y registrar el procedimiento antes de ejecutarlo, como se hizo el 4 de noviembre.
4. **Endurecer la QA de continuidad** (por ejemplo, al 2 %). Cambia la población de entrenamiento y obliga a preparar de nuevo los datos.
