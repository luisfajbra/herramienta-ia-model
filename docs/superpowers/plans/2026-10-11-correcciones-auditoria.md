# Correcciones de la auditoría (pasos 1 a 6) — Plan de implementación

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación.

**Objetivo:** que la evaluación del banco deje de sobreestimar o confundir resultados (agregación de métricas, referencias sin aprendizaje, desglose por forma, matriz de confusión, predicciones fuera de fold persistidas) y que las figuras, el modelo promovido y los scripts peligrosos sean trazables y seguros; y dejar documentado el spike sobre la resolución de la etiqueta.

**Arquitectura:** `score_predictions()` (`ml/bench/evaluate.py`) sigue siendo la única capa que puntúa predicciones fuera de fold. Se le añade un conjunto de métricas agregadas, el desglose por forma y un parámetro `protocols`. Las referencias sin aprendizaje (`ml/bench/baselines.py`) producen el mismo frame OOF y se puntúan con esa misma función. `main.py` persiste OOF, métricas y proveniencia. Verificación de promoción y rótulos "en muestra" se añaden en los puntos de consumo.

**Tecnologías:** Python 3.12 (venv del proyecto), pandas, numpy, scikit-learn, pytest, matplotlib, pyarrow (parquet).

**Spec:** `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md`
**Auditoría de origen:** `docs/auditoria/2026-10-10-auditoria-completa.md`

## Restricciones globales

- Trabajar en el worktree `C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model\.claude\worktrees\optuna-tuning-familia-ganadora`, rama `worktree-optuna-tuning-familia-ganadora` (publicada como `fix/cierre-auditoria-hidrogramas-20260927`). **No** hacer `cd` al checkout principal ni editar archivos allí.
- Intérprete: `..\..\..\venv\Scripts\python.exe` (relativo al worktree). Todos los comandos `pytest` se ejecutan como `..\..\..\venv\Scripts\python.exe -m pytest <ruta> -q -p no:cacheprovider --basetemp=<carpeta temporal fuera del repo>`.
- TDD: cada prueba nueva debe fallar por la ausencia del comportamiento antes de modificar producción.
- Línea base de la suite completa: **1.049 passed, 0 failed** (commit `1ecca50`). Al cerrar cada tarea, la suite completa debe seguir en verde (los tests que cambien deliberadamente se listan en la tarea).
- No cambiar `promote: "xgboost"` de `config.yaml`. No escribir en `outputs/models/`, ni borrar o sobrescribir artefactos de `outputs/` existentes. Los resultados nuevos van a rutas nuevas.
- No tocar `main` ni el checkout principal. Un commit por tarea. El push a `origin fix/cierre-auditoria-hidrogramas-20260927` lo hace el coordinador al cerrar cada bloque, no los subagentes.
- Convenciones del banco: mensajes y comentarios en español; nombres de código en el estilo del módulo vecino. Cada commit termina con la línea `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Umbral de inundación canónico: `volumen_m3 >= flood_threshold_m3` (Fase A).

## Corrección a la spec (descubierta al leer el código)

La spec §2.3 proponía añadir `vol_total_error_pct` a `end_to_end`. El test de paridad exige que `end_to_end` tenga **las mismas claves** que el evaluador legado, así que esas métricas agregadas van en un bloque nuevo `end_to_end_pooled`. `by_factor` no cambia. Esta tarea actualiza la spec (Task 2, paso 7).

## Mapa de archivos

| Archivo | Responsabilidad |
|---|---|
| `swmm_resilience/ml/bench/metrics.py` | Funciones puras de métricas (se añaden las agregadas, por fold y por forma) |
| `swmm_resilience/ml/bench/evaluate.py` | `score_predictions` (usa las métricas nuevas y el parámetro `protocols`) |
| `swmm_resilience/ml/bench/baselines.py` | **Nuevo.** Referencias sin aprendizaje |
| `swmm_resilience/ml/bench/schemas.py` | `PROTOCOL_INTERPRETATION` |
| `swmm_resilience/ml/bench/reports.py` | `write_metrics_files`, `write_oof`, `write_provenance` |
| `swmm_resilience/ml/bench/promote.py` | `verify_promotion`, `assert_promotion_consistent` |
| `swmm_resilience/ml/predict.py` | Llama a `assert_promotion_consistent` antes de cargar modelos |
| `swmm_resilience/visualization/labels.py` | `IN_SAMPLE_NOTE`, `add_in_sample_note` |
| `swmm_resilience/visualization/{resilience_curve,flood_volume_curve,model_comparison}.py` | Nota "en muestra" |
| `swmm_resilience/visualization/runner.py` | `provenance.json` junto a las figuras |
| `main.py` | OOF + referencias + proveniencia en `--bench-evaluate`; aviso en `--evaluate-shapes` |
| `halve_timeseries.py` | Función con guarda `__main__` |
| `scripts/informe_evaluacion.py` | **Nuevo.** Tablas del informe a partir de OOF y datos preparados |
| `tests/ml/bench/prepared_factory.py` | **Nuevo.** Fabrica de `PreparedDataset` mínimos para pruebas |
| `docs/auditoria/correcciones/` | Documentación para la investigación |

---

### Task 1: Métricas agregadas, por fold y por forma (funciones puras)

**Files:**
- Modify: `swmm_resilience/ml/bench/metrics.py` (añadir al final)
- Test: `tests/ml/bench/test_metrics.py` (añadir al final)

**Interfaces:**
- Produces (usadas por la Task 2):
  - `confusion_counts(y_true, y_pred) -> dict` con claves `tn, fp, fn, tp` (enteros)
  - `pooled_classifier_metrics(true_parts, pred_parts, prob_parts) -> dict` con `precision, recall, f1, auc_roc, specificity` (floats)
  - `fold_classifier_metrics(y_true, y_pred) -> dict` con `precision, recall, f1`; `NaN` donde no está definido
  - `summarize_fold_metrics(folds: list) -> dict` por métrica: `mean, min, median, max, n_undefined`
  - `end_to_end_pooled_metrics(vol_true_parts, vol_routed_parts) -> dict`
  - `shape_breakdown_entry(yc_true, yc_pred, vol_true, vol_routed) -> dict`

- [ ] **Step 1: Escribir las pruebas que fallan**

Añadir al final de `tests/ml/bench/test_metrics.py` (y añadir los nombres nuevos al bloque `from swmm_resilience.ml.bench.metrics import (...)` del inicio del archivo):

```python
from swmm_resilience.ml.bench.metrics import (
    confusion_counts,
    end_to_end_pooled_metrics,
    fold_classifier_metrics,
    pooled_classifier_metrics,
    shape_breakdown_entry,
    summarize_fold_metrics,
)


def test_confusion_counts_on_a_known_case():
    counts = confusion_counts([1, 1, 0, 0], [1, 0, 0, 1])
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_pooled_classifier_metrics_is_perfect_even_when_a_part_has_no_positives():
    """El caso de M-03: un fold sin inundaciones no debe contar como F1 = 0."""
    true_parts = [np.array([0, 0]), np.array([1, 0])]
    pred_parts = [np.array([0, 0]), np.array([1, 0])]
    prob_parts = [np.array([0.1, 0.1]), np.array([0.9, 0.1])]

    result = pooled_classifier_metrics(true_parts, pred_parts, prob_parts)

    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["auc_roc"] == pytest.approx(1.0)


def test_pooled_classifier_metrics_of_no_parts_is_empty():
    assert pooled_classifier_metrics([], [], []) == {}


def test_pooled_specificity_is_nan_without_negatives():
    result = pooled_classifier_metrics(
        [np.array([1, 1])], [np.array([1, 0])], [np.array([0.9, 0.4])]
    )
    assert np.isnan(result["specificity"])


def test_fold_metrics_are_nan_when_the_fold_has_no_real_positives():
    result = fold_classifier_metrics(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(result["precision"])
    assert np.isnan(result["recall"])
    assert np.isnan(result["f1"])


def test_fold_precision_is_zero_but_recall_nan_with_only_false_positives():
    result = fold_classifier_metrics(np.array([0, 0]), np.array([1, 0]))
    assert result["precision"] == pytest.approx(0.0)
    assert np.isnan(result["recall"])
    assert np.isnan(result["f1"])


def test_fold_metrics_match_the_textbook_values():
    result = fold_classifier_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_summarize_fold_metrics_ignores_undefined_folds_and_counts_them():
    folds = [
        {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3},
        {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    ]
    summary = summarize_fold_metrics(folds)

    assert summary["f1"]["n_undefined"] == 1
    assert summary["f1"]["mean"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert summary["f1"]["min"] == pytest.approx(2 / 3)
    assert summary["f1"]["max"] == pytest.approx(1.0)
    assert summary["f1"]["median"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_summarize_fold_metrics_of_all_undefined_folds_is_nan():
    folds = [{"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}]
    summary = summarize_fold_metrics(folds)
    assert np.isnan(summary["f1"]["mean"])
    assert summary["f1"]["n_undefined"] == 1


def test_end_to_end_pooled_metrics_known_values():
    true_parts = [np.array([0.0, 10.0]), np.array([40.0, 50.0])]
    routed_parts = [np.array([0.0, 10.0]), np.array([80.0, 100.0])]

    result = end_to_end_pooled_metrics(true_parts, routed_parts)

    assert result["vol_total_real_m3"] == pytest.approx(100.0)
    assert result["vol_total_pred_m3"] == pytest.approx(190.0)
    assert result["vol_total_error_pct"] == pytest.approx(90.0)
    assert result["rmse_vol_todos_nodos"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))
    assert result["mae_vol_todos_nodos"] == pytest.approx((40 + 50) / 4)


def test_end_to_end_pooled_error_pct_is_nan_without_real_volume():
    result = end_to_end_pooled_metrics([np.array([0.0])], [np.array([5.0])])
    assert np.isnan(result["vol_total_error_pct"])


def test_shape_breakdown_entry_known_values():
    entry = shape_breakdown_entry(
        yc_true=np.array([0, 0, 1, 1]),
        yc_pred=np.array([0, 0, 1, 1]),
        vol_true=np.array([0.0, 0.0, 40.0, 50.0]),
        vol_routed=np.array([0.0, 0.0, 80.0, 100.0]),
    )
    assert entry["n_rows"] == 4
    assert entry["n_positives"] == 2
    assert entry["f1"] == pytest.approx(1.0)
    assert entry["vol_real_m3"] == pytest.approx(90.0)
    assert entry["vol_pred_m3"] == pytest.approx(180.0)
    assert entry["error_vol_pct"] == pytest.approx(100.0)
    assert entry["rmse_vol"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))


def test_shape_breakdown_entry_f1_and_error_are_nan_without_positives():
    entry = shape_breakdown_entry(
        yc_true=np.array([0, 0]),
        yc_pred=np.array([0, 0]),
        vol_true=np.array([0.0, 0.0]),
        vol_routed=np.array([0.0, 0.0]),
    )
    assert np.isnan(entry["f1"])
    assert np.isnan(entry["error_vol_pct"])
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_metrics.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t1`
Expected: FAIL con `ImportError: cannot import name 'confusion_counts'`

- [ ] **Step 3: Implementar**

Añadir al final de `swmm_resilience/ml/bench/metrics.py`:

```python
def confusion_counts(y_true, y_pred) -> dict:
    """Conteos de la matriz de confusión; la clase positiva es ``inunda`` = 1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def pooled_classifier_metrics(true_parts: list, pred_parts: list, prob_parts: list) -> dict:
    """Nivel 1 agregado: concatena todas las filas de test del protocolo.

    Es la cifra principal del clasificador. Promediar por fold cuenta como 0
    los folds sin inundaciones reales (``zero_division=0``), lo que rebaja el
    F1 incluso de un predictor perfecto (hallazgo M-03 de la auditoría).
    """
    if not true_parts:
        return {}
    y_true = np.concatenate([np.asarray(part) for part in true_parts])
    y_pred = np.concatenate([np.asarray(part) for part in pred_parts])
    y_prob = np.concatenate([np.asarray(part) for part in prob_parts])
    result = classifier_metrics(y_true, y_pred, y_prob)
    counts = confusion_counts(y_true, y_pred)
    negatives = counts["tn"] + counts["fp"]
    result["specificity"] = counts["tn"] / negatives if negatives else float("nan")
    return result


def fold_classifier_metrics(y_true, y_pred) -> dict:
    """Precisión, recall y F1 de un fold; NaN (no 0) donde no están definidos.

    Recall y F1 no existen sin positivos reales; la precisión no existe sin
    positivos predichos.
    """
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    positives = tp + fn
    predicted = tp + fp
    nan = float("nan")
    return {
        "precision": tp / predicted if predicted else nan,
        "recall": tp / positives if positives else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if positives else nan,
    }


def summarize_fold_metrics(folds: list) -> dict:
    """Dispersión entre folds de precisión, recall y F1, ignorando los NaN."""
    summary = {}
    for key in ("precision", "recall", "f1"):
        values = np.array([fold[key] for fold in folds], dtype=float)
        valid = values[~np.isnan(values)]
        nan = float("nan")
        summary[key] = {
            "mean": float(valid.mean()) if valid.size else nan,
            "min": float(valid.min()) if valid.size else nan,
            "median": float(np.median(valid)) if valid.size else nan,
            "max": float(valid.max()) if valid.size else nan,
            "n_undefined": int(np.isnan(values).sum()),
        }
    return summary


def _relative_error_pct(real: float, predicted: float) -> float:
    return 100.0 * (predicted - real) / real if real > 0 else float("nan")


def end_to_end_pooled_metrics(vol_true_parts: list, vol_routed_parts: list) -> dict:
    """Nivel 3 sobre el pool de todos los folds, con el error relativo del volumen total."""
    if not vol_true_parts:
        return {}
    true = np.concatenate([np.asarray(part, dtype=float) for part in vol_true_parts])
    routed = np.concatenate([np.asarray(part, dtype=float) for part in vol_routed_parts])
    real, predicted = float(true.sum()), float(routed.sum())
    return {
        "rmse_vol_todos_nodos": float(np.sqrt(mean_squared_error(true, routed))),
        "mae_vol_todos_nodos": float(mean_absolute_error(true, routed)),
        "vol_total_real_m3": real,
        "vol_total_pred_m3": predicted,
        "vol_total_error_pct": _relative_error_pct(real, predicted),
    }


def shape_breakdown_entry(yc_true, yc_pred, vol_true, vol_routed) -> dict:
    """Métricas de una forma de hidrograma (sus filas de test concatenadas)."""
    yc_true = np.asarray(yc_true).astype(int)
    vol_true = np.asarray(vol_true, dtype=float)
    vol_routed = np.asarray(vol_routed, dtype=float)
    real, predicted = float(vol_true.sum()), float(vol_routed.sum())
    return {
        "n_rows": int(len(yc_true)),
        "n_positives": int(yc_true.sum()),
        "f1": fold_classifier_metrics(yc_true, yc_pred)["f1"],
        "rmse_vol": float(np.sqrt(np.mean((vol_true - vol_routed) ** 2))),
        "vol_real_m3": real,
        "vol_pred_m3": predicted,
        "error_vol_pct": _relative_error_pct(real, predicted),
    }
```

- [ ] **Step 4: Verificar que pasan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_metrics.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t1`
Expected: todos pasan (los preexistentes y los nuevos).

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/metrics.py tests/ml/bench/test_metrics.py
git commit -m "feat(bench): metricas agregadas, por fold (NaN) y por forma (hallazgos M-03 y M-02)"
```

---

### Task 2: `score_predictions` con agregación honesta, matriz de confusión y desglose por forma

**Files:**
- Create: `tests/ml/bench/prepared_factory.py`
- Modify: `swmm_resilience/ml/bench/evaluate.py` (`_validate_oof`, `score_predictions`)
- Modify: `tests/ml/bench/test_score_predictions.py`, `tests/ml/bench/test_parity_with_legacy_evaluator.py`
- Modify: `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md` (§2.3 y §2.4)

**Interfaces:**
- Consumes (Task 1): `pooled_classifier_metrics`, `fold_classifier_metrics`, `summarize_fold_metrics`, `confusion_counts`, `end_to_end_pooled_metrics`, `shape_breakdown_entry`.
- Produces (Tasks 3 y 4):
  - `score_predictions(prepared, oof, provenance, protocols=None) -> dict`. Por protocolo, el dict contiene las claves: `classifier` (pooled: `precision, recall, f1, auc_roc, specificity`), `confusion_matrix` (`tn, fp, fn, tp`), `classifier_per_fold`, `classifier_legacy_mean_zero_fill` (la media por fold antigua, con las mismas claves que el evaluador legado), `regressor_oracle`, `end_to_end` (sin cambios), `end_to_end_pooled`, `by_factor`, `by_shape`.
  - `tests/ml/bench/prepared_factory.make_prepared(rows, protocols, threshold=1.0) -> PreparedDataset`

- [ ] **Step 1: Crear la fábrica de datasets de prueba**

Crear `tests/ml/bench/prepared_factory.py`:

```python
"""Fábrica mínima de PreparedDataset para pruebas del banco (sin SQL)."""

from __future__ import annotations

import pandas as pd

from swmm_resilience.ml.bench.folds import build_all_folds
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS, PreparedDataset
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

PREP_ID = "0123456789abcdef"


def make_prepared(rows: list[dict], protocols, *, threshold: float = 1.0) -> PreparedDataset:
    """Construye un PreparedDataset a partir de filas ya expandidas.

    Cada fila: ``shape_id, node_id, factor_mult, vol, duracion_horas, tiempo_al_pico_h``.
    ``inunda`` = ``vol >= threshold``. Las demás features valen 0.0.
    """
    frame = pd.DataFrame(rows).reset_index(drop=True)
    pairs = sorted(set(zip(frame["shape_id"], frame["factor_mult"])))
    scenario_id = {pair: index + 1 for index, pair in enumerate(pairs)}
    ids = [scenario_id[(s, f)] for s, f in zip(frame["shape_id"], frame["factor_mult"])]
    keys = pd.DataFrame(
        {
            "run_id": ids,
            "network_id": 1,
            "scenario_id": ids,
            "scenario_key": [
                f"{s}@{f}" for s, f in zip(frame["shape_id"], frame["factor_mult"])
            ],
            "scenario_kind": "base",
            "node_id": frame["node_id"],
            "factor_mult": frame["factor_mult"].astype(float),
            "shape_id": frame["shape_id"],
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(0.0, index=range(len(frame)), columns=list(FEATURE_COLUMNS_V17))
    X["duracion_horas"] = frame["duracion_horas"].to_numpy(dtype=float)
    X["tiempo_al_pico_h"] = frame["tiempo_al_pico_h"].to_numpy(dtype=float)
    volume = frame["vol"].to_numpy(dtype=float)
    folds = build_all_folds(keys, list(protocols))
    return PreparedDataset(
        prep_id=PREP_ID,
        keys=keys,
        X=X,
        y_clf=pd.Series((volume >= threshold).astype(int), name="inunda"),
        y_reg=pd.Series(volume, name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": PREP_ID, "protocols": list(protocols)},
        quality={},
    )


def perfect_oof(prepared: PreparedDataset) -> pd.DataFrame:
    """Predicciones fuera de fold perfectas para todos los folds de test."""
    tests = prepared.folds[prepared.folds["split"] == "test"]
    idx = tests["sample_idx"].to_numpy()
    return pd.DataFrame(
        {
            "sample_idx": idx,
            "protocol": tests["protocol"].to_numpy(),
            "fold_id": tests["fold_id"].to_numpy(),
            "y_pred_clf": prepared.y_clf.iloc[idx].to_numpy(),
            "y_prob_clf": prepared.y_clf.iloc[idx].to_numpy().astype(float),
            "y_pred_reg": prepared.y_reg.iloc[idx].to_numpy(),
        }
    )
```

Verificar que se importa: `..\..\..\venv\Scripts\python.exe -c "from tests.ml.bench.prepared_factory import make_prepared"` (si falla por rutas, usar `pytest.ini pythonpath = .` y ejecutar desde la raíz del worktree).

- [ ] **Step 2: Escribir las pruebas que fallan**

Añadir al final de `tests/ml/bench/test_score_predictions.py`:

```python
from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


def _loso_with_an_empty_fold():
    """3 factores en LOSO; el factor 1.0 no tiene ninguna fila inundada."""
    rows = []
    volumes = {
        1.0: [0.0, 0.0, 0.0],
        2.0: [0.0, 10.0, 20.0],
        3.0: [30.0, 40.0, 50.0],
    }
    for factor, vols in volumes.items():
        for node, vol in zip(["N0", "N1", "N2"], vols):
            rows.append(
                dict(shape_id="base", node_id=node, factor_mult=factor, vol=vol,
                     duracion_horas=3.0, tiempo_al_pico_h=0.7)
            )
    return make_prepared(rows, ["LOSO"])


def test_perfect_predictions_score_f1_one_even_with_a_fold_without_flooding():
    """M-03: el valor agregado no se rebaja por un fold sin positivos."""
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["classifier"]["f1"] == pytest.approx(1.0)
    assert result["classifier"]["specificity"] == pytest.approx(1.0)


def test_the_legacy_zero_fill_mean_is_kept_under_an_explicit_name():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    # folds: factor 1.0 (sin positivos) -> 0 ; factores 2.0 y 3.0 -> 1.0
    assert result["classifier_legacy_mean_zero_fill"]["f1"] == pytest.approx(2 / 3)
    assert set(result["classifier_legacy_mean_zero_fill"]) == {
        "precision", "recall", "f1", "auc_roc"
    }


def test_per_fold_dispersion_reports_the_undefined_fold():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["classifier_per_fold"]["f1"]["n_undefined"] == 1
    assert result["classifier_per_fold"]["f1"]["min"] == pytest.approx(1.0)


def test_confusion_matrix_counts_every_test_row():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 5}


def test_a_false_negative_and_a_false_positive_show_up_in_the_confusion_matrix():
    prepared = _loso_with_an_empty_fold()
    oof = perfect_oof(prepared)
    oof.loc[oof["sample_idx"] == 4, "y_pred_clf"] = 0   # falso negativo (N1, factor 2.0)
    oof.loc[oof["sample_idx"] == 0, "y_pred_clf"] = 1   # falso positivo (N0, factor 1.0)

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})["LOSO"]

    assert result["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 4}
    assert result["classifier"]["recall"] == pytest.approx(4 / 5)
    assert result["classifier"]["precision"] == pytest.approx(4 / 5)


def test_score_predictions_can_be_restricted_to_some_protocols():
    prepared = _loso_with_an_empty_fold()
    oof = perfect_oof(prepared)
    # el frame solo cubre LOSO, pero el dataset declara LOSO únicamente: sin restricción pasa
    assert "LOSO" in score_predictions(prepared, oof, {"prep_id": prepared.prep_id})
    restricted = score_predictions(
        prepared, oof, {"prep_id": prepared.prep_id}, protocols=("LOSO",)
    )
    assert set(restricted) == {"LOSO"}


def test_by_shape_and_pooled_end_to_end_use_the_routed_volume():
    rows = []
    spec = {
        "A": {("N0", 1.0): 0.0, ("N1", 1.0): 10.0, ("N0", 2.0): 0.0, ("N1", 2.0): 20.0},
        "B": {("N0", 1.0): 0.0, ("N1", 1.0): 0.0, ("N0", 2.0): 40.0, ("N1", 2.0): 50.0},
    }
    for shape, cells in spec.items():
        for (node, factor), vol in cells.items():
            rows.append(
                dict(shape_id=shape, node_id=node, factor_mult=factor, vol=vol,
                     duracion_horas=3.0, tiempo_al_pico_h=0.7)
            )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    oof = perfect_oof(prepared)
    shape_b = prepared.keys.loc[oof["sample_idx"], "shape_id"].to_numpy() == "B"
    oof.loc[shape_b, "y_pred_reg"] = oof.loc[shape_b, "y_pred_reg"] * 2.0  # B se sobreestima 2x

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})["LeaveOneShapeOut"]

    assert result["by_shape"]["A"]["error_vol_pct"] == pytest.approx(0.0)
    assert result["by_shape"]["B"]["error_vol_pct"] == pytest.approx(100.0)
    assert result["by_shape"]["B"]["rmse_vol"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))
    assert result["end_to_end_pooled"]["vol_total_real_m3"] == pytest.approx(120.0)
    assert result["end_to_end_pooled"]["vol_total_pred_m3"] == pytest.approx(210.0)
    assert result["end_to_end_pooled"]["vol_total_error_pct"] == pytest.approx(75.0)
    assert result["end_to_end_pooled"]["rmse_vol_todos_nodos"] == pytest.approx(
        np.sqrt((40**2 + 50**2) / 8)
    )
```

Y actualizar la prueba existente `test_result_has_the_three_levels_and_the_factor_breakdown` (cambiar la aserción de igualdad por inclusión):

```python
    assert set(result["LOSO"]) >= {
        "classifier", "regressor_oracle", "end_to_end", "by_factor",
        "confusion_matrix", "classifier_per_fold",
        "classifier_legacy_mean_zero_fill", "end_to_end_pooled", "by_shape",
    }
```

- [ ] **Step 3: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_score_predictions.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t2`
Expected: FAIL (`KeyError: 'confusion_matrix'`, `TypeError ... unexpected keyword argument 'protocols'`, etc.)

- [ ] **Step 4: Implementar**

En `swmm_resilience/ml/bench/evaluate.py`:

1. Reemplazar el bloque `from .metrics import (...)` por:

```python
from .metrics import (
    classifier_metrics,
    confusion_counts,
    end_to_end_metrics,
    end_to_end_pooled_metrics,
    fold_classifier_metrics,
    mean_metrics,
    pooled_classifier_metrics,
    pooled_regressor_metrics,
    shape_breakdown_entry,
    summarize_fold_metrics,
)
```

2. Reemplazar `_validate_oof` por:

```python
def _validate_oof(
    prepared: PreparedDataset, oof: pd.DataFrame, protocols=None
) -> None:
    missing = [column for column in OOF_COLUMNS if column not in oof.columns]
    if missing:
        raise ValueError(f"Al frame OOF le faltan columnas: {missing}")

    duplicated = oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).sum()
    if duplicated:
        raise ValueError(
            f"El frame OOF tiene {duplicated} predicción(es) duplicadas para la misma "
            "(protocol, fold_id, sample_idx)."
        )

    expected = prepared.folds[prepared.folds["split"] == "test"]
    if protocols is not None:
        expected = expected[expected["protocol"].isin(list(protocols))]
    expected_keys = set(
        map(tuple, expected[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    got_keys = set(
        map(tuple, oof[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    if expected_keys != got_keys:
        raise ValueError(
            "La cobertura del frame OOF no coincide con las filas de test de los folds: "
            f"faltan {len(expected_keys - got_keys)}, sobran {len(got_keys - expected_keys)}."
        )
```

3. Reemplazar `score_predictions` completa por:

```python
def score_predictions(
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    provenance: dict,
    protocols=None,
) -> dict:
    """Puntúa predicciones out-of-fold contra ``prepared``.

    ``provenance`` debe traer el ``prep_id`` del dataset con el que se
    generaron. Sin esa verificación se podrían comparar números de datasets
    distintos sin notarlo, que es el modo de fallo silencioso que este
    diseño existe para cerrar.

    ``protocols`` restringe la cobertura exigida al frame OOF (una referencia
    sin aprendizaje puede aplicar solo a algunos protocolos). Por defecto se
    exigen todos los del PreparedDataset.

    Por protocolo devuelve:
    - ``classifier``: métricas AGREGADAS sobre todas las filas de test (cifra
      principal) y ``confusion_matrix`` agregada.
    - ``classifier_per_fold``: dispersión entre folds, con NaN donde no está definido.
    - ``classifier_legacy_mean_zero_fill``: la media por fold con ``zero_division=0``
      (definición anterior, conservada para citar el antes y el después).
    - ``regressor_oracle``, ``end_to_end``, ``by_factor``: sin cambios.
    - ``end_to_end_pooled`` y ``by_shape``: agregado de nivel 3 y desglose por forma.
    """
    if provenance.get("prep_id") != prepared.prep_id:
        raise ProvenanceMismatchError(
            f"prep_id de las predicciones ({provenance.get('prep_id')!r}) no coincide "
            f"con el del dataset ({prepared.prep_id!r}). No son comparables."
        )
    _validate_oof(prepared, oof, protocols)

    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()
    factors = prepared.keys["factor_mult"].to_numpy()
    shapes = prepared.keys["shape_id"].astype(str).to_numpy()

    results: dict = {}
    for protocol, protocol_oof in oof.groupby("protocol"):
        classifier_folds, fold_stats, e2e_folds = [], [], []
        oracle_true, oracle_pred = [], []
        pool_true, pool_pred, pool_prob = [], [], []
        pool_vol_true, pool_vol_routed = [], []
        by_factor: dict[str, list] = {}
        by_shape_parts: dict[str, dict[str, list]] = {}

        for _, fold_oof in protocol_oof.groupby("fold_id"):
            idx = fold_oof["sample_idx"].to_numpy()
            yc_true, yr_true = y_clf[idx], y_reg[idx]
            yc_pred = fold_oof["y_pred_clf"].to_numpy()
            yc_prob = fold_oof["y_prob_clf"].to_numpy()
            yr_pred = fold_oof["y_pred_reg"].to_numpy()

            classifier_folds.append(classifier_metrics(yc_true, yc_pred, yc_prob))
            fold_stats.append(fold_classifier_metrics(yc_true, yc_pred))
            pool_true.append(yc_true)
            pool_pred.append(yc_pred)
            pool_prob.append(yc_prob)

            flooded = yc_true == 1
            if flooded.sum():
                oracle_true.append(yr_true[flooded])
                oracle_pred.append(yr_pred[flooded])

            routed = np.where(yc_pred == 1, yr_pred, 0.0)
            e2e_folds.append(end_to_end_metrics(yr_true, routed, yc_true, yc_pred))
            pool_vol_true.append(yr_true)
            pool_vol_routed.append(routed)

            for factor in np.unique(factors[idx]):
                mask = factors[idx] == factor
                key = f"{factor:.2f}"
                by_factor.setdefault(key, []).append(
                    {
                        "f1": classifier_metrics(
                            yc_true[mask], yc_pred[mask], yc_prob[mask]
                        )["f1"],
                        "rmse_vol": end_to_end_metrics(
                            yr_true[mask], routed[mask], yc_true[mask], yc_pred[mask]
                        )["rmse_vol_todos_nodos"],
                    }
                )

            for shape in np.unique(shapes[idx]):
                mask = shapes[idx] == shape
                parts = by_shape_parts.setdefault(
                    str(shape),
                    {"yc_true": [], "yc_pred": [], "vol_true": [], "vol_routed": []},
                )
                parts["yc_true"].append(yc_true[mask])
                parts["yc_pred"].append(yc_pred[mask])
                parts["vol_true"].append(yr_true[mask])
                parts["vol_routed"].append(routed[mask])

        all_true = np.concatenate(pool_true)
        all_pred = np.concatenate(pool_pred)
        results[protocol] = {
            "classifier": pooled_classifier_metrics(pool_true, pool_pred, pool_prob),
            "confusion_matrix": confusion_counts(all_true, all_pred),
            "classifier_per_fold": summarize_fold_metrics(fold_stats),
            "classifier_legacy_mean_zero_fill": mean_metrics(classifier_folds),
            "regressor_oracle": pooled_regressor_metrics(oracle_true, oracle_pred),
            "end_to_end": mean_metrics(e2e_folds),
            "end_to_end_pooled": end_to_end_pooled_metrics(pool_vol_true, pool_vol_routed),
            "by_factor": {key: mean_metrics(value) for key, value in by_factor.items()},
            "by_shape": {
                shape: shape_breakdown_entry(
                    np.concatenate(parts["yc_true"]),
                    np.concatenate(parts["yc_pred"]),
                    np.concatenate(parts["vol_true"]),
                    np.concatenate(parts["vol_routed"]),
                )
                for shape, parts in by_shape_parts.items()
            },
        }
    return results
```

- [ ] **Step 5: Ajustar el test de paridad con el evaluador legado**

En `tests/ml/bench/test_parity_with_legacy_evaluator.py`:

(a) Reemplazar la aserción de claves del banco en `_assert_top_level_shape` por inclusión, y reemplazar la comprobación del banco vacío para que mire el campo legado. Sustituir el cuerpo de `_assert_top_level_shape` por:

```python
def _assert_top_level_shape(legacy: dict, bench: dict) -> None:
    """Las claves del legado están en el banco y ninguna sub-dict viene vacía.

    El banco añade claves nuevas (matriz de confusión, agregados, por forma);
    la paridad se exige sobre las del legado. La métrica de clasificador del
    legado se compara contra ``classifier_legacy_mean_zero_fill``.
    """
    assert set(legacy) == TOP_LEVEL_KEYS, (
        f"claves de primer nivel del legacy: {sorted(legacy)}"
    )
    assert set(bench) >= TOP_LEVEL_KEYS, (
        f"claves de primer nivel del banco: {sorted(bench)}"
    )
    for level in sorted(TOP_LEVEL_KEYS):
        assert legacy[level], (
            f"legacy[{level!r}] esta vacio: la evaluacion legada no produjo nada "
            "y la paridad no significaria nada."
        )
        assert bench[level], (
            f"banco[{level!r}] esta vacio: la evaluacion del banco no produjo nada "
            "y la paridad no significaria nada."
        )
    assert bench["classifier_legacy_mean_zero_fill"], (
        "classifier_legacy_mean_zero_fill vacio: no hay con que comparar el clasificador legado."
    )
```

(b) En los tres tests, reemplazar la línea
`_assert_metrics_match(legacy["classifier"], bench["classifier"], "classifier")`
por
`_assert_metrics_match(legacy["classifier"], bench["classifier_legacy_mean_zero_fill"], "classifier")`.

(c) Añadir al final del archivo una prueba que fija la diferencia buscada:

```python
def test_the_new_classifier_value_is_pooled_and_differs_from_the_legacy_mean(
    sql_training_db, tmp_path
):
    """El valor principal ya no es la media por fold: es el agregado del protocolo."""
    _, bench = _run_both_sides(sql_training_db, tmp_path, "LOSO")

    matrix = bench["confusion_matrix"]
    assert matrix["tp"] + matrix["fn"] > 0, "el fixture debe tener filas inundadas"
    expected_f1 = 2 * matrix["tp"] / (2 * matrix["tp"] + matrix["fp"] + matrix["fn"])
    assert bench["classifier"]["f1"] == pytest.approx(expected_f1)
```

- [ ] **Step 6: Ejecutar el paquete del banco completo y reparar solo lo que cambia a propósito**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench -q -p no:cacheprovider --basetemp=%TEMP%\pt_t2b`
Expected: todo en verde. Si un test preexistente falla únicamente porque (i) comparaba el conjunto exacto de claves de `score_predictions` o (ii) esperaba la media por fold en `classifier`, ajustarlo a las nuevas claves y dejar constancia en el mensaje del commit de cuáles cambiaron. Cualquier otro fallo es una regresión real: investigar, no relajar tolerancias.

- [ ] **Step 7: Actualizar la spec**

En `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md`, §2.3, reemplazar la frase "Se añade a `end_to_end` el error relativo del volumen total (`vol_total_error_pct`)." por: "Se añade un bloque nuevo `end_to_end_pooled` (RMSE, MAE, volumen total real y predicho y `vol_total_error_pct`, calculados sobre el pool de todos los folds); `end_to_end` conserva exactamente las claves del evaluador legado porque el test de paridad lo exige." Y en §2.3 la lista de claves de `classifier` se complementa con `confusion_matrix` como clave de primer nivel separada (no anidada en `classifier`).

- [ ] **Step 8: Suite completa y commit**

Run: `..\..\..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=%TEMP%\pt_t2c -o addopts="-m 'not scale' -q"`
Expected: sin fallos.

```bash
git add tests/ml/bench/prepared_factory.py swmm_resilience/ml/bench/evaluate.py tests/ml/bench/test_score_predictions.py tests/ml/bench/test_parity_with_legacy_evaluator.py docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md
git commit -m "feat(bench): clasificador agregado, matriz de confusion y desglose por forma (M-03, M-02)"
```

---

### Task 3: Referencias sin aprendizaje (M-01)

**Files:**
- Create: `swmm_resilience/ml/bench/baselines.py`
- Create: `tests/ml/bench/test_baselines.py`
- Modify: `swmm_resilience/ml/bench/schemas.py` (añadir `PROTOCOL_INTERPRETATION` tras `PROTOCOL_GROUP_COLUMNS`)

**Interfaces:**
- Consumes (Task 2): `score_predictions(prepared, oof, provenance, protocols=...)`, `make_prepared`, `perfect_oof`.
- Produces (Task 4): constantes `INTERPOLACION_FACTOR = "baseline_interpolacion_factor"`, `FORMA_MAS_CERCANA = "baseline_forma_mas_cercana"`, `BASELINES`; excepción `BaselineNotApplicable(ValueError)`; `baseline_oof(prepared, name, flood_threshold_m3) -> pd.DataFrame` (columnas `OOF_COLUMNS`); `evaluate_baseline(prepared, name, flood_threshold_m3) -> tuple[pd.DataFrame, dict]`; `schemas.PROTOCOL_INTERPRETATION: dict[str, str]`.

- [ ] **Step 1: Escribir las pruebas que fallan**

Crear `tests/ml/bench/test_baselines.py`:

```python
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.baselines import (
    BASELINES,
    FORMA_MAS_CERCANA,
    INTERPOLACION_FACTOR,
    BaselineNotApplicable,
    baseline_oof,
    evaluate_baseline,
)
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PROTOCOL_INTERPRETATION
from tests.ml.bench.prepared_factory import make_prepared


def _linear_rows(shape="s", dur=1.0, tp=0.5, factors=(1.0, 2.0, 3.0, 4.0, 5.0), scale=1.0):
    rows = []
    for factor in factors:
        for node, per_factor in (("N0", 10.0), ("N1", 20.0)):
            rows.append(
                dict(shape_id=shape, node_id=node, factor_mult=factor,
                     vol=per_factor * factor * scale,
                     duracion_horas=dur, tiempo_al_pico_h=tp)
            )
    return rows


def _predictions(prepared, oof):
    keys = prepared.keys.loc[oof["sample_idx"]]
    out = oof.assign(node=keys["node_id"].to_numpy(), factor=keys["factor_mult"].to_numpy())
    return out.set_index(["node", "factor"])["y_pred_reg"]


def test_interpolation_reproduces_a_linear_function_in_the_interior():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)
    predicted = _predictions(prepared, oof)

    assert predicted[("N0", 3.0)] == pytest.approx(30.0)
    assert predicted[("N1", 3.0)] == pytest.approx(60.0)
    assert predicted[("N0", 2.0)] == pytest.approx(20.0)


def test_interpolation_uses_the_nearest_neighbour_at_the_extremes():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    predicted = _predictions(prepared, baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0))

    assert predicted[("N0", 1.0)] == pytest.approx(20.0)   # vecino f=2.0
    assert predicted[("N0", 5.0)] == pytest.approx(40.0)   # vecino f=4.0


def test_interpolation_label_follows_the_threshold():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 35.0)
    keys = prepared.keys.loc[oof["sample_idx"]]
    labels = oof.assign(node=keys["node_id"].to_numpy(), factor=keys["factor_mult"].to_numpy())
    labels = labels.set_index(["node", "factor"])["y_pred_clf"]

    assert labels[("N0", 3.0)] == 0     # 30 < 35
    assert labels[("N1", 3.0)] == 1     # 60 >= 35


def test_baseline_oof_has_the_oof_contract_and_is_scored_by_score_predictions():
    prepared = make_prepared(_linear_rows(), ["LOSO"])

    oof, metrics = evaluate_baseline(prepared, INTERPOLACION_FACTOR, 1.0)

    assert list(oof.columns) == list(OOF_COLUMNS)
    assert set(metrics) == {"LOSO"}
    assert metrics["LOSO"]["classifier"]["f1"] == pytest.approx(1.0)


def test_interpolation_does_not_apply_to_shape_protocols_and_is_omitted():
    rows = _linear_rows("A") + _linear_rows("B", scale=2.0)
    prepared = make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])

    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)

    assert set(oof["protocol"]) == {"LOSO"}


def test_nearest_shape_only_applies_to_shape_protocols():
    rows = _linear_rows("A") + _linear_rows("B", scale=2.0)
    prepared = make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])

    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)

    assert set(oof["protocol"]) == {"LeaveOneShapeOut"}


def test_nearest_shape_borrows_the_volume_of_the_closest_training_shape():
    rows = (
        _linear_rows("A", dur=1.0, tp=0.2, factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", dur=1.1, tp=0.2, factors=(1.0, 2.0), scale=7.0)
        + _linear_rows("C", dur=10.0, tp=5.0, factors=(1.0, 2.0), scale=100.0)
    )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)

    keys = prepared.keys.loc[oof["sample_idx"]]
    table = oof.assign(
        shape=keys["shape_id"].to_numpy(),
        node=keys["node_id"].to_numpy(),
        factor=keys["factor_mult"].to_numpy(),
    ).set_index(["shape", "node", "factor"])["y_pred_reg"]

    # B se parece a A, A se parece a B, C se parece a B (la más cercana en duración/pico)
    assert table[("B", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 5.0)    # volumen de A
    assert table[("A", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)    # volumen de B
    assert table[("C", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)    # volumen de B


def test_nearest_shape_baseline_is_scored_with_the_shape_breakdown():
    rows = (
        _linear_rows("A", dur=1.0, tp=0.2, factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", dur=1.1, tp=0.2, factors=(1.0, 2.0), scale=7.0)
    )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])

    _, metrics = evaluate_baseline(prepared, FORMA_MAS_CERCANA, 1.0)

    assert set(metrics["LeaveOneShapeOut"]["by_shape"]) == {"A", "B"}


def test_a_baseline_with_no_applicable_protocol_raises_the_dedicated_error():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    with pytest.raises(BaselineNotApplicable, match="LOSO"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)


def test_unknown_baseline_name_is_rejected():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    with pytest.raises(ValueError, match="desconocida"):
        baseline_oof(prepared, "baseline_inventada", 1.0)


def test_every_protocol_has_an_interpretation():
    for protocol in ("LOSO", "GroupKFold5", "LeaveOneShapeOut"):
        assert PROTOCOL_INTERPRETATION[protocol]
    assert set(BASELINES) == {INTERPOLACION_FACTOR, FORMA_MAS_CERCANA}
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_baselines.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t3`
Expected: FAIL con `ModuleNotFoundError: ...baselines`

- [ ] **Step 3: Implementar**

En `swmm_resilience/ml/bench/schemas.py`, justo después de la definición de `PROTOCOL_GROUP_COLUMNS`:

```python
PROTOCOL_INTERPRETATION = {
    "LOSO": (
        "Interpolación entre factores: se retiene un factor de intensidad completo; "
        "todas las formas de hidrograma siguen presentes en el entrenamiento."
    ),
    "GroupKFold5": (
        "Interpolación entre factores en 5 bloques retenidos; todas las formas de "
        "hidrograma siguen presentes en el entrenamiento."
    ),
    "LeaveOneShapeOut": (
        "Generalización a formas de hidrograma no vistas: se retiene una forma completa."
    ),
}
```

Crear `swmm_resilience/ml/bench/baselines.py`:

```python
"""Referencias sin aprendizaje para el banco (hallazgo M-01).

Una referencia produce un frame OOF con el mismo contrato que
``evaluate_candidate`` y se puntúa con ``score_predictions``. Sirve para
saber cuánto de un resultado se debe a aprender y cuánto a que la tarea es
fácil: LOSO y GroupKFold5 retienen factores de intensidad, y una tabla de
consulta que interpola entre factores vecinos ya rinde muy bien.

- ``baseline_interpolacion_factor``: para una fila (forma, nodo, factor)
  retenida, interpola linealmente el volumen de la misma forma y nodo entre
  los factores vecinos del entrenamiento (en los extremos, el vecino más
  cercano). Aplica a los protocolos agrupados por ``factor_mult``.
- ``baseline_forma_mas_cercana``: para una fila (forma retenida, nodo,
  factor), toma el volumen del mismo nodo y factor en la forma de
  entrenamiento más cercana en (``duracion_horas``, ``tiempo_al_pico_h``)
  estandarizados. Aplica a los protocolos agrupados por ``shape_id``.

La etiqueta predicha es ``volumen_predicho >= flood_threshold_m3`` y la
"probabilidad" es esa misma etiqueta (0/1): el AUC-ROC de una referencia sin
probabilidades reales no es interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import score_predictions
from .schemas import OOF_COLUMNS, PreparedDataset, group_column_for

INTERPOLACION_FACTOR = "baseline_interpolacion_factor"
FORMA_MAS_CERCANA = "baseline_forma_mas_cercana"
BASELINES = (INTERPOLACION_FACTOR, FORMA_MAS_CERCANA)

_SHAPE_DESCRIPTORS = ("duracion_horas", "tiempo_al_pico_h")
_FACTOR_DECIMALS = 6


class BaselineNotApplicable(ValueError):
    """La referencia no aplica a ningún protocolo del PreparedDataset."""


def _applicable_protocols(prepared: PreparedDataset, name: str) -> list[str]:
    wanted = "factor_mult" if name == INTERPOLACION_FACTOR else "shape_id"
    protocols = sorted(prepared.folds["protocol"].unique())
    return [protocol for protocol in protocols if group_column_for(protocol) == wanted]


def _node_ids(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["node_id"].astype(str).to_numpy()


def _shape_ids(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["shape_id"].astype(str).to_numpy()


def _factors(prepared: PreparedDataset) -> np.ndarray:
    return prepared.keys["factor_mult"].to_numpy(dtype=float).round(_FACTOR_DECIMALS)


def _interpolacion_factor_fold(
    prepared: PreparedDataset, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    series = np.char.add(np.char.add(_shape_ids(prepared), "|"), _node_ids(prepared))
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    train_factors = np.unique(factors[train_idx])
    table = pd.DataFrame(
        {"series": series[train_idx], "factor": factors[train_idx], "vol": volumes[train_idx]}
    )
    wide = table.pivot(index="series", columns="factor", values="vol").reindex(
        columns=train_factors
    )
    if wide.isna().any().any():
        raise ValueError(
            "baseline_interpolacion_factor: hay series (forma, nodo) sin todos los "
            "factores de entrenamiento; no se puede interpolar."
        )
    row_of = pd.Series(np.arange(len(wide)), index=wide.index)
    rows = row_of.reindex(series[test_idx])
    if rows.isna().any():
        raise ValueError(
            "baseline_interpolacion_factor: una serie (forma, nodo) del test no está en el train."
        )
    rows = rows.to_numpy(dtype=int)
    matrix = wide.to_numpy(dtype=float)

    f_test = factors[test_idx]
    position = np.searchsorted(train_factors, f_test)
    low = np.clip(position - 1, 0, len(train_factors) - 1)
    high = np.clip(position, 0, len(train_factors) - 1)
    f_low, f_high = train_factors[low], train_factors[high]
    span = f_high - f_low
    weight = np.divide(f_test - f_low, span, out=np.zeros_like(f_test), where=span > 0)
    return matrix[rows, low] * (1.0 - weight) + matrix[rows, high] * weight


def _forma_mas_cercana_fold(
    prepared: PreparedDataset, train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    shapes = _shape_ids(prepared)
    nodes = _node_ids(prepared)
    factors = _factors(prepared)
    volumes = prepared.y_reg.to_numpy(dtype=float)

    descriptors = pd.DataFrame(
        {"shape": shapes, **{c: prepared.X[c].to_numpy(dtype=float) for c in _SHAPE_DESCRIPTORS}}
    )
    grouped = descriptors.groupby("shape")[list(_SHAPE_DESCRIPTORS)]
    if (grouped.nunique() > 1).any().any():
        raise ValueError(
            "baseline_forma_mas_cercana: las features de forma no son constantes dentro de una forma."
        )
    table = grouped.first()

    train_shapes = sorted(set(shapes[train_idx]))
    test_shapes = sorted(set(shapes[test_idx]))
    train_table = table.loc[train_shapes]
    mean = train_table.mean()
    std = train_table.std(ddof=0).replace(0.0, 1.0)
    z_train = (train_table - mean) / std

    nearest = {}
    for shape in test_shapes:
        z_shape = (table.loc[shape] - mean) / std
        distance = np.sqrt(((z_train - z_shape) ** 2).sum(axis=1))
        nearest[shape] = distance.idxmin()

    index = pd.MultiIndex.from_arrays(
        [shapes[train_idx], nodes[train_idx], factors[train_idx]]
    )
    lookup_table = pd.Series(volumes[train_idx], index=index)
    if lookup_table.index.has_duplicates:
        raise ValueError(
            "baseline_forma_mas_cercana: hay (forma, nodo, factor) duplicados en el train."
        )
    wanted = pd.MultiIndex.from_arrays(
        [[nearest[shape] for shape in shapes[test_idx]], nodes[test_idx], factors[test_idx]]
    )
    predicted = lookup_table.reindex(wanted)
    if predicted.isna().any():
        raise ValueError(
            "baseline_forma_mas_cercana: la forma vecina no tiene todos los (nodo, factor) del test."
        )
    return predicted.to_numpy(dtype=float)


def baseline_oof(
    prepared: PreparedDataset, name: str, flood_threshold_m3: float
) -> pd.DataFrame:
    """Frame OOF de la referencia ``name`` para los protocolos donde aplica."""
    if name not in BASELINES:
        raise ValueError(f"Referencia desconocida: {name!r}. Opciones: {BASELINES}")
    protocols = _applicable_protocols(prepared, name)
    if not protocols:
        available = ", ".join(sorted(prepared.folds["protocol"].unique()))
        raise BaselineNotApplicable(
            f"{name} no aplica a ningún protocolo de este PreparedDataset ({available})."
        )
    predict_fold = (
        _interpolacion_factor_fold if name == INTERPOLACION_FACTOR else _forma_mas_cercana_fold
    )

    records = []
    for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
        if protocol not in protocols:
            continue
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        volume = predict_fold(prepared, train_idx, test_idx)
        flooded = (volume >= flood_threshold_m3).astype(int)
        records.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "y_pred_clf": flooded,
                    "y_prob_clf": flooded.astype(float),
                    "y_pred_reg": volume.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )
    return pd.concat(records, ignore_index=True)


def evaluate_baseline(
    prepared: PreparedDataset, name: str, flood_threshold_m3: float
) -> tuple[pd.DataFrame, dict]:
    """Produce el OOF de la referencia y lo puntúa como a cualquier candidato."""
    oof = baseline_oof(prepared, name, flood_threshold_m3)
    protocols = tuple(sorted(oof["protocol"].unique()))
    metrics = score_predictions(
        prepared, oof, {"prep_id": prepared.prep_id}, protocols=protocols
    )
    return oof, metrics
```

- [ ] **Step 4: Verificar que pasan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_baselines.py tests/ml/bench/test_schemas.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t3`
Expected: todo en verde. Si `test_nearest_shape_borrows_the_volume_of_the_closest_training_shape` falla, revisar primero los números esperados del enunciado (A↔B se parecen; C se parece a B) antes de tocar la implementación.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/baselines.py swmm_resilience/ml/bench/schemas.py tests/ml/bench/test_baselines.py
git commit -m "feat(bench): referencias sin aprendizaje (interpolacion de factor y forma mas cercana) y etiqueta de protocolos (M-01)"
```

---

### Task 4: Persistir OOF, referencias y proveniencia en `--bench-evaluate`

**Files:**
- Modify: `swmm_resilience/ml/bench/reports.py`
- Modify: `main.py` (`_run_bench`, bloque de evaluación)
- Modify: `tests/ml/bench/test_reports.py`, `tests/ml/bench/test_bench_cli.py`

**Interfaces:**
- Consumes: `evaluate_baseline`, `BASELINES`, `BaselineNotApplicable` (Task 3); `PROTOCOL_INTERPRETATION` (Task 3); `evaluate_candidate` devuelve `(oof, metrics)`.
- Produces: `reports.write_metrics_files(metrics_by_name, output_dir, prefix="metrics_") -> dict[str, Path]`; `reports.write_oof(oof, root, prep_id, name) -> Path` (`<root>/<prep_id>/<name>.parquet`); `reports.write_provenance(output_dir, prepared, families, baselines) -> Path` (`provenance.json`). `write_reports` conserva su firma y comportamiento.

- [ ] **Step 1: Escribir las pruebas que fallan**

Añadir al final de `tests/ml/bench/test_reports.py`:

```python
import json

from swmm_resilience.ml.bench.reports import (
    write_metrics_files,
    write_oof,
    write_provenance,
)
from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


def _tiny_prepared():
    rows = [
        dict(shape_id="s", node_id="N0", factor_mult=f, vol=10.0 * f,
             duracion_horas=1.0, tiempo_al_pico_h=0.5)
        for f in (1.0, 2.0, 3.0)
    ]
    return make_prepared(rows, ["LOSO"])


def test_write_metrics_files_uses_the_given_prefix_and_strict_json(tmp_path):
    metrics = {"baseline_x": {"LOSO": {"classifier": {"f1": float("nan")}}}}

    paths = write_metrics_files(metrics, tmp_path)

    written = json.loads((tmp_path / "metrics_baseline_x.json").read_text(encoding="utf-8"))
    assert written["LOSO"]["classifier"]["f1"] is None
    assert paths["metrics_baseline_x"] == tmp_path / "metrics_baseline_x.json"


def test_write_oof_stores_one_parquet_per_name_under_the_prep_id(tmp_path):
    prepared = _tiny_prepared()
    oof = perfect_oof(prepared)

    path = write_oof(oof, tmp_path, prepared.prep_id, "xgboost")

    assert path == tmp_path / prepared.prep_id / "xgboost.parquet"
    assert pd.read_parquet(path).shape == oof.shape


def test_write_provenance_records_prep_id_params_baselines_and_protocol_meaning(tmp_path):
    prepared = _tiny_prepared()
    families = {"xgboost": {"classifier": {"max_depth": 3}, "regressor": {"max_depth": 4}}}

    path = write_provenance(tmp_path, prepared, families, ["baseline_interpolacion_factor"])

    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "provenance.json"
    assert record["prep_id"] == prepared.prep_id
    assert record["families"]["xgboost"]["classifier"] == {"max_depth": 3}
    assert record["baselines"] == ["baseline_interpolacion_factor"]
    assert "generated_at_utc" in record
    assert "LOSO" in record["protocol_interpretation"]
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_reports.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t4`
Expected: FAIL con `ImportError: cannot import name 'write_metrics_files'`

- [ ] **Step 3: Implementar en `reports.py`**

Añadir a los imports: `from datetime import datetime, timezone` y `from .schemas import PROTOCOL_INTERPRETATION, PreparedDataset`.

Añadir estas funciones y hacer que `write_reports` use `write_metrics_files` (conservando su firma y los nombres de archivo):

```python
def write_metrics_files(
    metrics_by_name: dict[str, dict], output_dir: Path | str, prefix: str = "metrics_"
) -> dict[str, Path]:
    """Escribe un JSON estricto por nombre (familia o referencia)."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, metrics in metrics_by_name.items():
        path = directory / f"{prefix}{name}.json"
        path.write_text(
            json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        written[f"{prefix}{name}"] = path
    return written


def write_oof(oof: pd.DataFrame, root: Path | str, prep_id: str, name: str) -> Path:
    """Guarda las predicciones fuera de fold en ``<root>/<prep_id>/<name>.parquet``."""
    directory = Path(root) / prep_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    oof.to_parquet(path, index=False)
    return path


def write_provenance(
    output_dir: Path | str,
    prepared: PreparedDataset,
    families: dict[str, dict],
    baselines: list[str],
) -> Path:
    """``provenance.json``: qué dataset, qué hiperparámetros y qué mide cada protocolo."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "prep_id": prepared.prep_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_interpretation": dict(PROTOCOL_INTERPRETATION),
        "families": families,
        "baselines": list(baselines),
    }
    path = directory / "provenance.json"
    path.write_text(
        json.dumps(_json_safe(record), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
```

Y dentro de `write_reports`, sustituir el bucle que escribe `metrics_<family>.json` por:

```python
    written: dict[str, Path] = write_metrics_files(metrics_by_family, directory)
```

(mantener el resto: `ranking.json`, `ranking.csv`).

- [ ] **Step 4: Verificar que pasan las pruebas de reportes**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_reports.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t4`
Expected: PASS.

- [ ] **Step 5: Escribir la prueba de CLI que falla**

Añadir a `tests/ml/bench/test_bench_cli.py`, junto a `test_full_bench_run_writes_ranking_without_promoting`:

```python
def test_bench_evaluate_persists_oof_baselines_and_provenance(
    sql_training_db, tmp_path, monkeypatch
):
    import main

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )

    main._run_bench(_make_args(bench=True, models="xgboost"), config)

    bench_root = tmp_path / "outputs" / "bench"
    prep_id = next((bench_root / "prepared").iterdir()).name
    assert (bench_root / "oof" / prep_id / "xgboost.parquet").exists()
    assert (bench_root / "oof" / prep_id / "baseline_interpolacion_factor.parquet").exists()
    assert (bench_root / "reports" / "metrics_baseline_interpolacion_factor.json").exists()
    # LOSO por factor: la referencia por forma no aplica y se omite
    assert not (bench_root / "oof" / prep_id / "baseline_forma_mas_cercana.parquet").exists()

    provenance = json.loads((bench_root / "reports" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["prep_id"] == prep_id
    assert provenance["families"]["xgboost"]["classifier"] == TINY
    assert provenance["baselines"] == ["baseline_interpolacion_factor"]
    assert not (tmp_path / "outputs" / "models").exists()
```

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_bench_cli.py::test_bench_evaluate_persists_oof_baselines_and_provenance -q -p no:cacheprovider --basetemp=%TEMP%\pt_t4b`
Expected: FAIL (`AssertionError` por archivos inexistentes).

- [ ] **Step 6: Implementar en `main.py`**

En `_run_bench`, añadir a los imports locales:

```python
    from swmm_resilience.ml.bench.baselines import (
        BASELINES,
        BaselineNotApplicable,
        evaluate_baseline,
    )
    from swmm_resilience.ml.bench.reports import (
        write_metrics_files,
        write_oof,
        write_provenance,
        write_reports,
    )
```
(sustituyendo el import existente de `write_reports`), y definir `oof_dir = BENCH_ROOT / "oof"` junto a `reports_dir`. Reemplazar el bucle de evaluación y las líneas siguientes hasta `write_reports(...)` por:

```python
    if run_all or args.bench_evaluate:
        metrics_by_family = {}
        for family in families:
            spec = config.bench.families[family]
            oof, metrics = evaluate_candidate(prepared, family, spec.classifier, spec.regressor)
            metrics_by_family[family] = metrics
            write_oof(oof, oof_dir, prepared.prep_id, family)
            print(f"  {family} evaluado")

        baseline_metrics = {}
        for baseline in BASELINES:
            try:
                baseline_oof_frame, metrics = evaluate_baseline(
                    prepared, baseline, config.dataset.flood_threshold_m3
                )
            except BaselineNotApplicable as exc:
                print(f"  {baseline}: omitida ({exc})")
                continue
            baseline_metrics[baseline] = metrics
            write_oof(baseline_oof_frame, oof_dir, prepared.prep_id, baseline)
            print(f"  {baseline} evaluada")

        criterion = RankingCriterion(
            primary_metric=config.bench.ranking.primary_metric,
            primary_direction=config.bench.ranking.primary_direction,
            tie_breakers=config.bench.ranking.tie_breakers,
        )
        ranking_protocol = config.bench.protocols[0]
        ranking = rank_candidates(metrics_by_family, criterion, ranking_protocol)
        write_reports(metrics_by_family, ranking, reports_dir)
        write_metrics_files(baseline_metrics, reports_dir)
        write_provenance(
            reports_dir,
            prepared,
            {
                family: {
                    "classifier": config.bench.families[family].classifier,
                    "regressor": config.bench.families[family].regressor,
                }
                for family in families
            },
            sorted(baseline_metrics),
        )
        print(f"Ranking calculado sobre el protocolo: {ranking_protocol}")
        print(ranking.to_string(index=False))
```

- [ ] **Step 7: Verificar y commit**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench -q -p no:cacheprovider --basetemp=%TEMP%\pt_t4c`
Expected: todo en verde (incluida `test_full_bench_run_writes_ranking_without_promoting`, que sigue exigiendo que no se cree `outputs/models`).

```bash
git add swmm_resilience/ml/bench/reports.py main.py tests/ml/bench/test_reports.py tests/ml/bench/test_bench_cli.py
git commit -m "feat(bench): persistir OOF, referencias y proveniencia en --bench-evaluate (M-08)"
```

---

### Task 5: Rotular como "en muestra" las figuras y salidas que lo son (V-01)

**Files:**
- Modify: `swmm_resilience/visualization/labels.py`, `resilience_curve.py`, `flood_volume_curve.py`, `model_comparison.py` (solo `plot_factor_comparison`)
- Modify: `main.py` (inicio del bloque `--evaluate-shapes`)
- Create: `tests/visualization/test_in_sample_note.py`

**Interfaces:**
- Produces: `labels.IN_SAMPLE_NOTE: str`, `labels.IN_SAMPLE_BANNER: str`, `labels.add_in_sample_note(fig) -> None`.

- [ ] **Step 1: Escribir las pruebas que fallan**

Crear `tests/visualization/test_in_sample_note.py`:

```python
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from swmm_resilience.visualization.flood_volume_curve import (
    plot_flood_volume_combined,
    plot_flood_volume_curves,
)
from swmm_resilience.visualization.labels import IN_SAMPLE_NOTE, add_in_sample_note
from swmm_resilience.visualization.model_comparison import plot_factor_comparison
from swmm_resilience.visualization.resilience_curve import plot_resilience_curve


@pytest.fixture
def captured_figure_texts(monkeypatch):
    seen = []
    original = Figure.savefig

    def spy(self, *args, **kwargs):
        seen.append((args[0] if args else None, [t.get_text() for t in self.texts]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return seen


def _has_note(seen, name_part):
    return any(name_part in str(path) and IN_SAMPLE_NOTE in texts for path, texts in seen)


def test_add_in_sample_note_writes_the_note_into_the_figure():
    fig = Figure()
    add_in_sample_note(fig)
    assert IN_SAMPLE_NOTE in [t.get_text() for t in fig.texts]


def test_resilience_ml_figure_carries_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"factor": [1.0, 2.0], "resilience_swmm": [1.0, 0.5], "resilience_ml": [1.0, 0.6]})
    plot_resilience_curve(df, tmp_path)
    assert _has_note(captured_figure_texts, "resilience_ml")


def test_flood_volume_figures_carry_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"factor": [1.0, 2.0], "vol_total_swmm": [10.0, 20.0], "vol_total_ml": [11.0, 19.0]})
    plot_flood_volume_curves(df, tmp_path)
    plot_flood_volume_combined(df, tmp_path)
    assert _has_note(captured_figure_texts, "flood_volume_ml")
    assert _has_note(captured_figure_texts, "flood_volume_combined")


def test_factor_comparison_figures_carry_the_in_sample_note(tmp_path, captured_figure_texts):
    df = pd.DataFrame({"node_id": ["1I", "2O"], "vol_swmm_m3": [12.0, 3.0], "vol_pred_m3": [10.0, 5.0]})
    plot_factor_comparison(df, tmp_path, 1.0)
    assert _has_note(captured_figure_texts, "volume_by_node_factor")
    assert _has_note(captured_figure_texts, "parity_factor")


def test_the_evaluate_shapes_branch_prints_the_in_sample_banner():
    from pathlib import Path

    source = Path("main.py").read_text(encoding="utf-8")
    assert "IN_SAMPLE_BANNER" in source
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/visualization/test_in_sample_note.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t5`
Expected: FAIL con `ImportError: cannot import name 'IN_SAMPLE_NOTE'`. Nota: si el nombre de la función de flood_volume (`plot_flood_volume_curves`) no coincide con el real, abrir `flood_volume_curve.py` y usar el nombre público que genera `flood_volume_swmm.png` y `flood_volume_ml.png`.

- [ ] **Step 3: Implementar**

En `swmm_resilience/visualization/labels.py`, añadir tras los imports:

```python
IN_SAMPLE_NOTE = (
    "In-sample: the model was trained on these scenarios; this is not an independent validation."
)
IN_SAMPLE_BANNER = (
    "[EN MUESTRA] Estos escenarios (formas y factores) se usaron para entrenar el modelo; "
    "las métricas no son una validación independiente."
)


def add_in_sample_note(fig) -> None:
    """Pie de figura que avisa que el modelo ya vio estos escenarios."""
    fig.text(
        0.5, 0.005, IN_SAMPLE_NOTE,
        ha="center", va="bottom", fontsize=8, color="#7a7a7a", style="italic",
    )
```

En `resilience_curve.py`: importar `from .labels import add_in_sample_note` y, en `plot_resilience_curve`, justo antes de `fig.savefig(path_ml, ...)` añadir `add_in_sample_note(fig)`.

En `flood_volume_curve.py`: importar igual y añadir `add_in_sample_note(fig)` justo antes de `fig.savefig(path_ml, ...)` y antes de `fig.savefig(output_path, ...)` en `plot_flood_volume_combined`.

En `model_comparison.py`, dentro de `plot_factor_comparison`: añadir `add_in_sample_note(fig)` justo antes de cada uno de los dos `fig.savefig(...)` (el de `profile_path` y el de `parity_path`); importar `add_in_sample_note` desde `.labels` junto a `format_node_label`.

En `main.py`, en el bloque `if args.evaluate_shapes:`, justo después del `try/except` de `load_training_frame` y antes de `from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes`, añadir:

```python
        from swmm_resilience.visualization.labels import IN_SAMPLE_BANNER

        print(f"\n{IN_SAMPLE_BANNER}")
```

- [ ] **Step 4: Verificar y suite de visualización**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/visualization tests/test_visualization.py tests/test_resilience.py tests/test_flood_volume.py tests/test_factor_comparison_cli.py tests/test_hydrograph_batch.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t5b`
Expected: todo en verde (los títulos de los mapas de validación no se tocaron).

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/visualization swmm_resilience/analysis main.py tests/visualization/test_in_sample_note.py
git commit -m "fix(viz): rotular como en muestra las curvas y comparaciones evaluadas sobre datos de entrenamiento (V-01)"
```

---

### Task 6: `halve_timeseries.py` sin efectos al importarse (V-07)

**Files:**
- Modify: `halve_timeseries.py` (reescritura)
- Create: `tests/scripts/test_halve_timeseries.py` (la carpeta `tests/scripts` ya existe)

**Interfaces:**
- Produces: `halve_timeseries.halve_timeseries(src, dst, force=False) -> int` (número de valores divididos) y `main(argv=None)`.

- [ ] **Step 1: Escribir las pruebas que fallan**

Crear `tests/scripts/test_halve_timeseries.py`:

```python
import importlib
from pathlib import Path

import pytest

INP = """[TIMESERIES]
;;Name  Time  Value
TS1     0:00  10.000
TS1     1:00  20.000

[OTHER]
KEEP    5.0
"""


def test_importing_the_module_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("halve_timeseries")
    importlib.reload(module)
    assert not any(tmp_path.iterdir())


def test_halves_only_the_timeseries_section(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    dst = tmp_path / "out.inp"
    src.write_text(INP, encoding="utf-8")

    count = halve_timeseries(src, dst)

    out = dst.read_text(encoding="utf-8")
    assert count == 2
    assert "5.000" in out and "10.000" in out
    assert "KEEP    5.0" in out


def test_refuses_to_overwrite_an_existing_destination_without_force(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    dst = tmp_path / "out.inp"
    src.write_text(INP, encoding="utf-8")
    dst.write_text("PRECIOUS", encoding="utf-8")

    with pytest.raises(FileExistsError):
        halve_timeseries(src, dst)
    assert dst.read_text(encoding="utf-8") == "PRECIOUS"

    halve_timeseries(src, dst, force=True)
    assert dst.read_text(encoding="utf-8") != "PRECIOUS"


def test_refuses_to_use_the_source_as_destination(tmp_path):
    from halve_timeseries import halve_timeseries

    src = tmp_path / "in.inp"
    src.write_text(INP, encoding="utf-8")
    with pytest.raises(ValueError, match="mismo archivo"):
        halve_timeseries(src, src, force=True)
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/scripts/test_halve_timeseries.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t6`
Expected: FAIL. Atención: el primer test **importa el módulo actual, que escribe el `.inp` base al importarse**. Para no dañar datos reales, antes de ejecutar este paso hacer `git status data/` y confirmar que `data/networks/chico_hydro-qx1/` no tiene cambios después; si el import modificara el archivo, restaurar con `git checkout -- data/networks/chico_hydro-qx1/`. (Esta es justamente la razón de V-07.) Alternativa más segura: ejecutar el paso 2 con `monkeypatch.chdir(tmp_path)` ya incluido en la primera prueba: las rutas relativas del script no existirán desde `tmp_path` y el import fallará con `FileNotFoundError` sin tocar nada.

- [ ] **Step 3: Implementar**

Reemplazar `halve_timeseries.py` por:

```python
"""Divide por 2 los valores de [TIMESERIES] de un .inp de SWMM.

Uso (explícito, nunca al importar):
    python halve_timeseries.py --src <entrada.inp> --dst <salida.inp> [--force]

Se usó una vez para derivar la red Qx1 a partir de Qx2. El destino NO se
sobrescribe sin ``--force``: una versión anterior de este script escribía sobre
el .inp base de producción al importarse (hallazgo V-07 de la auditoría).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def halve_timeseries(src, dst, force: bool = False) -> int:
    """Escribe en ``dst`` una copia de ``src`` con los valores de [TIMESERIES] a la mitad.

    Devuelve cuántos valores se dividieron.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("El origen y el destino son el mismo archivo.")
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} ya existe; usa force=True (--force) para sobrescribirlo.")

    in_timeseries = False
    out_lines: list[str] = []
    halved = 0
    with open(src, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if re.match(r"^\[", stripped):
                in_timeseries = stripped.strip() == "[TIMESERIES]"
                out_lines.append(stripped + "\n")
                continue
            if in_timeseries and stripped and not stripped.startswith(";;"):
                parts = stripped.split()
                try:
                    value = float(parts[-1]) / 2
                    last_token = stripped.rsplit(None, 1)[-1]
                    prefix = stripped[: stripped.rfind(last_token)]
                    stripped = prefix + f"{value:.3f}"
                    halved += 1
                except ValueError:
                    pass  # la última columna no es un número: la línea queda igual
            out_lines.append(stripped + "\n")

    with open(dst, "w", encoding="utf-8") as handle:
        handle.writelines(out_lines)
    return halved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    count = halve_timeseries(args.src, args.dst, force=args.force)
    print(f"Listo: {count} valores divididos. Escrito en: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verificar y commit**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/scripts -q -p no:cacheprovider --basetemp=%TEMP%\pt_t6b` y `git status --short data/` (debe estar vacío).
Expected: verde y sin cambios en `data/`.

```bash
git add halve_timeseries.py tests/scripts/test_halve_timeseries.py
git commit -m "fix: halve_timeseries sin efectos al importarse y sin sobrescribir sin --force (V-07)"
```

---

### Task 7: Coherencia modelo–`promotion.json` y procedencia de las figuras (M-07/V-04)

**Files:**
- Modify: `swmm_resilience/ml/bench/promote.py`, `swmm_resilience/ml/predict.py`, `swmm_resilience/visualization/runner.py`
- Create: `tests/ml/bench/test_verify_promotion.py`
- Modify: `tests/test_ml_trainer_predict.py` (solo si un test existente carga modelos sin `promotion.json`; ver Step 5)

**Interfaces:**
- Produces: `promote.ModelProvenanceError(RuntimeError)`; `promote.verify_promotion(models_dir) -> dict` con `status` ∈ {`verified`, `mismatch`, `no_record`}, `ok` (`True`/`False`/`None`) y `discrepancies` (lista de `{artifact, expected, actual}`); `promote.assert_promotion_consistent(models_dir) -> dict`; `promote.model_provenance(models_dir) -> dict` (`family`, `feature_contract_id`, `classifier_sha256`, `regressor_sha256`, `prep_id`).

- [ ] **Step 1: Escribir las pruebas que fallan**

Crear `tests/ml/bench/test_verify_promotion.py`:

```python
import hashlib
import json

import pytest

from swmm_resilience.ml.bench.promote import (
    ModelProvenanceError,
    assert_promotion_consistent,
    model_provenance,
    verify_promotion,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _models_dir(tmp_path, *, classifier=b"clf", regressor=b"reg", record=True, tamper=False):
    (tmp_path / "classifier.joblib").write_bytes(classifier)
    (tmp_path / "regressor.joblib").write_bytes(regressor)
    if record:
        payload = {
            "family": "xgboost",
            "prep_id": "abc",
            "feature_contract_id": "tabular_v4_21",
            "classifier_sha256": _sha(b"clf"),
            "regressor_sha256": _sha(b"reg"),
        }
        (tmp_path / "promotion.json").write_text(json.dumps(payload), encoding="utf-8")
    if tamper:
        (tmp_path / "classifier.joblib").write_bytes(b"otro modelo")
    return tmp_path


def test_matching_hashes_are_verified(tmp_path):
    result = verify_promotion(_models_dir(tmp_path))
    assert result["status"] == "verified" and result["ok"] is True
    assert result["discrepancies"] == []


def test_a_replaced_model_is_reported_with_both_hashes(tmp_path):
    result = verify_promotion(_models_dir(tmp_path, tamper=True))

    assert result["status"] == "mismatch" and result["ok"] is False
    discrepancy = result["discrepancies"][0]
    assert discrepancy["artifact"] == "classifier.joblib"
    assert discrepancy["expected"] == _sha(b"clf")
    assert discrepancy["actual"] == _sha(b"otro modelo")


def test_a_missing_artifact_is_a_discrepancy(tmp_path):
    models = _models_dir(tmp_path)
    (models / "regressor.joblib").unlink()
    result = verify_promotion(models)
    assert result["status"] == "mismatch"
    assert result["discrepancies"][0]["actual"] is None


def test_no_promotion_record_is_reported_not_raised(tmp_path):
    result = verify_promotion(_models_dir(tmp_path, record=False))
    assert result["status"] == "no_record" and result["ok"] is None


def test_assert_raises_on_mismatch_naming_the_hashes(tmp_path):
    with pytest.raises(ModelProvenanceError, match="classifier.joblib"):
        assert_promotion_consistent(_models_dir(tmp_path, tamper=True))


def test_assert_warns_but_continues_without_a_record(tmp_path):
    with pytest.warns(UserWarning, match="promotion.json"):
        result = assert_promotion_consistent(_models_dir(tmp_path, record=False))
    assert result["status"] == "no_record"


def test_assert_returns_silently_when_consistent(tmp_path):
    assert assert_promotion_consistent(_models_dir(tmp_path))["status"] == "verified"


def test_model_provenance_summarises_the_active_model(tmp_path):
    info = model_provenance(_models_dir(tmp_path))
    assert info["family"] == "xgboost"
    assert info["feature_contract_id"] == "tabular_v4_21"
    assert info["classifier_sha256"] == _sha(b"clf")
    assert info["prep_id"] == "abc"


def test_model_provenance_without_a_record_is_marked_unknown(tmp_path):
    info = model_provenance(_models_dir(tmp_path, record=False))
    assert info["family"] == "desconocida"
```

- [ ] **Step 2: Verificar que fallan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_verify_promotion.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t7`
Expected: FAIL con `ImportError: cannot import name 'ModelProvenanceError'`

- [ ] **Step 3: Implementar en `promote.py`**

Añadir `import warnings` a los imports y estas definiciones:

```python
class ModelProvenanceError(RuntimeError):
    """Los modelos de ``models_dir`` no coinciden con lo declarado en promotion.json."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PROMOTION_ARTIFACTS = (
    ("classifier.joblib", "classifier_sha256"),
    ("regressor.joblib", "regressor_sha256"),
)


def verify_promotion(models_dir: Path | str) -> dict:
    """Compara los SHA-256 de los modelos con los de ``promotion.json``.

    ``status``: ``verified`` (coinciden), ``mismatch`` (alguno difiere o falta) o
    ``no_record`` (no hay ``promotion.json``; no se puede afirmar ni negar nada).
    """
    target = Path(models_dir)
    record_path = target / "promotion.json"
    if not record_path.exists():
        return {"status": "no_record", "ok": None, "discrepancies": []}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    discrepancies = []
    for artifact, field in _PROMOTION_ARTIFACTS:
        path = target / artifact
        actual = _sha256(path) if path.exists() else None
        expected = record.get(field)
        if actual != expected:
            discrepancies.append({"artifact": artifact, "expected": expected, "actual": actual})
    status = "mismatch" if discrepancies else "verified"
    return {"status": status, "ok": not discrepancies, "discrepancies": discrepancies}


def assert_promotion_consistent(models_dir: Path | str) -> dict:
    """Falla si los modelos no son los declarados; avisa si no hay registro."""
    result = verify_promotion(models_dir)
    if result["status"] == "mismatch":
        detail = "; ".join(
            f"{d['artifact']}: esperado {d['expected']}, actual {d['actual']}"
            for d in result["discrepancies"]
        )
        raise ModelProvenanceError(
            f"Los modelos en {models_dir} no coinciden con promotion.json ({detail}). "
            "Los resultados no se pueden atribuir al candidato promovido. Vuelve a "
            "promover o regenera los modelos antes de predecir."
        )
    if result["status"] == "no_record":
        warnings.warn(
            f"No hay promotion.json en {models_dir}: no se puede verificar la procedencia del modelo.",
            UserWarning,
            stacklevel=2,
        )
    return result


def model_provenance(models_dir: Path | str) -> dict:
    """Resumen de procedencia del modelo activo, para estamparlo junto a las figuras."""
    record_path = Path(models_dir) / "promotion.json"
    if not record_path.exists():
        return {
            "family": "desconocida", "feature_contract_id": None,
            "classifier_sha256": None, "regressor_sha256": None, "prep_id": None,
        }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return {
        "family": record.get("family"),
        "feature_contract_id": record.get("feature_contract_id"),
        "classifier_sha256": record.get("classifier_sha256"),
        "regressor_sha256": record.get("regressor_sha256"),
        "prep_id": record.get("prep_id"),
    }
```

- [ ] **Step 4: Verificar que pasan**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_verify_promotion.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t7`
Expected: PASS.

- [ ] **Step 5: Conectar `predict_network` y probar la conexión**

En `swmm_resilience/ml/predict.py`, añadir `from .bench.promote import assert_promotion_consistent` a los imports (verificar que no hay importación circular: `promote` importa `.train`; si el import circular aparece, hacer el import dentro de la función) y, como primera línea de `predict_network` (antes de `joblib.load`):

```python
    assert_promotion_consistent(models_dir)
```

Añadir a `tests/ml/bench/test_verify_promotion.py` una prueba de conexión:

```python
def test_predict_network_refuses_models_that_do_not_match_promotion_json(tmp_path):
    from types import SimpleNamespace
    from swmm_resilience.ml.predict import predict_network

    models = _models_dir(tmp_path, tamper=True)
    with pytest.raises(ModelProvenanceError):
        predict_network(1.0, SimpleNamespace(), models)
```

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/ml/bench/test_verify_promotion.py tests/test_ml_trainer_predict.py tests/test_predict_shape_features.py tests/test_scenario_predict.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t7b`
Expected: verde. Si un test preexistente de `predict_network` falla porque sus modelos de prueba tienen un `promotion.json` incoherente, corregir el fixture del test (no la verificación). Si falla por `UserWarning` convertido en error, comprobar `pytest.ini` (no hay `filterwarnings = error` hoy).

- [ ] **Step 6: Estampar la procedencia junto a las figuras**

En `swmm_resilience/visualization/runner.py`, localizar la función que genera mapas/figuras con el modelo (el punto donde se llama a `predict_network` o `generate_ml_map`) y, al terminar de escribir las figuras de un directorio de salida, escribir `provenance.json` con:

```python
from ..ml.bench.promote import model_provenance

def write_figure_provenance(output_dir, models_dir) -> Path:
    """Registra qué modelo produjo las figuras de este directorio."""
    import json
    from datetime import datetime, timezone

    record = {**model_provenance(models_dir), "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    path = Path(output_dir) / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

Definir `write_figure_provenance` en `runner.py` y llamarla al final de la rutina que produce las figuras ML (leer el archivo para ubicar el punto exacto; `models_dir` es el directorio de modelos que ese código ya usa). Añadir a `tests/visualization/test_runtime_caption.py` o a un archivo nuevo `tests/visualization/test_figure_provenance.py`:

```python
import json

from swmm_resilience.visualization.runner import write_figure_provenance


def test_figure_provenance_records_the_active_model(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "promotion.json").write_text(
        json.dumps({"family": "xgboost", "classifier_sha256": "aa", "regressor_sha256": "bb",
                    "feature_contract_id": "tabular_v4_21", "prep_id": "p"}),
        encoding="utf-8",
    )

    path = write_figure_provenance(tmp_path / "figs", models)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["classifier_sha256"] == "aa"
    assert record["family"] == "xgboost"
    assert "generated_at_utc" in record
```

- [ ] **Step 7: Suite completa y commit**

Run: `..\..\..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=%TEMP%\pt_t7c -o addopts="-m 'not scale' -q"`
Expected: sin fallos.

```bash
git add swmm_resilience/ml/bench/promote.py swmm_resilience/ml/predict.py swmm_resilience/visualization/runner.py tests/ml/bench/test_verify_promotion.py tests/visualization
git commit -m "feat: verificar modelo contra promotion.json antes de predecir y estampar procedencia en las figuras (M-07, V-04)"
```

---

### Task 8: Medición real y script del informe (M-01, M-02, M-03, matriz de confusión)

Esta tarea **no cambia el código del paquete**: ejecuta el banco sobre los datos reales y genera tablas reproducibles. No se escribe en `outputs/models`.

**Files:**
- Create: `scripts/informe_evaluacion.py`
- Create: `tests/scripts/test_informe_evaluacion.py`
- Create (salida): `docs/auditoria/correcciones/datos/*.csv`

**Interfaces:**
- Consumes: parquet de OOF (`outputs/bench/oof/<prep_id>/*.parquet`, Task 4) y el PreparedDataset (`outputs/bench/prepared/<prep_id>/`).
- Produces: `informe_evaluacion.build_tables(prepared, oof_by_name, flood_threshold_m3) -> dict[str, pd.DataFrame]` con las tablas `comparacion_protocolos`, `matriz_confusion`, `banda_limitrofe`, `por_forma`; y un `main()` CLI.

- [ ] **Step 1: Prueba que falla**

Crear `tests/scripts/test_informe_evaluacion.py`:

```python
import numpy as np
import pandas as pd
import pytest

from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


def _prepared():
    rows = []
    for shape, scale in (("A", 1.0), ("B", 2.0)):
        for factor in (1.0, 2.0, 3.0):
            rows.append(dict(shape_id=shape, node_id="N0", factor_mult=factor, vol=10.0 * factor * scale,
                             duracion_horas=1.0, tiempo_al_pico_h=0.5))
    return make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])


def test_build_tables_has_the_four_tables_and_a_confusion_row_per_model_and_protocol():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()
    tables = build_tables(prepared, {"modelo": perfect_oof(prepared)}, flood_threshold_m3=1.0)

    assert set(tables) == {"comparacion_protocolos", "matriz_confusion", "banda_limitrofe", "por_forma"}
    confusion = tables["matriz_confusion"]
    assert set(confusion["modelo"]) == {"modelo"}
    assert set(confusion["protocolo"]) == {"LOSO", "LeaveOneShapeOut"}
    row = confusion[confusion["protocolo"] == "LOSO"].iloc[0]
    assert row["tp"] + row["fn"] == int(prepared.y_clf.sum())


def test_boundary_band_splits_false_negatives_by_true_volume():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()
    oof = perfect_oof(prepared)
    oof.loc[oof["sample_idx"] == 0, "y_pred_clf"] = 0   # nodo con vol 10 -> FN fuera de la banda 1-2

    tables = build_tables(prepared, {"modelo": oof}, flood_threshold_m3=1.0)

    band = tables["banda_limitrofe"]
    assert {"modelo", "protocolo", "fn_total", "fn_en_banda_1_2m3", "fn_fuera_de_banda"} <= set(band.columns)
    assert int(band.iloc[0]["fn_total"]) == int(band["fn_en_banda_1_2m3"].iloc[0] + band["fn_fuera_de_banda"].iloc[0])
```

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/scripts/test_informe_evaluacion.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t8`
Expected: FAIL con `ModuleNotFoundError: scripts.informe_evaluacion` (verificar que `scripts/` es importable; si no existe `scripts/__init__.py`, crearlo vacío en este mismo paso).

- [ ] **Step 2: Implementar `scripts/informe_evaluacion.py`**

```python
"""Tablas del informe de evaluación a partir de OOF persistidos.

Uso:
    python scripts/informe_evaluacion.py --prep-id <ID> [--bench-root outputs/bench]
        [--out docs/auditoria/correcciones/datos] [--flood-threshold 1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from swmm_resilience.ml.bench.evaluate import score_predictions
from swmm_resilience.ml.bench.preprocess import resolve_prepared
from swmm_resilience.ml.bench.schemas import PROTOCOLS, PROTOCOL_INTERPRETATION

BAND_MAX_M3 = 2.0


def _scored(prepared, oof):
    protocols = tuple(sorted(oof["protocol"].unique()))
    return score_predictions(prepared, oof, {"prep_id": prepared.prep_id}, protocols=protocols)


def build_tables(prepared, oof_by_name: dict, flood_threshold_m3: float) -> dict:
    y_true = prepared.y_clf.to_numpy()
    volume = prepared.y_reg.to_numpy()
    comparison, confusion, band, by_shape = [], [], [], []

    for name, oof in oof_by_name.items():
        scored = _scored(prepared, oof)
        for protocol, block in scored.items():
            comparison.append(
                {
                    "modelo": name,
                    "protocolo": protocol,
                    "que_mide": PROTOCOL_INTERPRETATION.get(protocol, ""),
                    "f1_agregado": block["classifier"].get("f1"),
                    "recall_agregado": block["classifier"].get("recall"),
                    "precision_agregada": block["classifier"].get("precision"),
                    "f1_media_por_fold_con_ceros": block["classifier_legacy_mean_zero_fill"].get("f1"),
                    "folds_f1_indefinido": block["classifier_per_fold"]["f1"]["n_undefined"],
                    "rmse_vol_agregado_m3": block["end_to_end_pooled"].get("rmse_vol_todos_nodos"),
                    "rmse_vol_media_por_fold_m3": block["end_to_end"].get("rmse_vol_todos_nodos"),
                    "error_vol_total_pct": block["end_to_end_pooled"].get("vol_total_error_pct"),
                    "nse_regresor_oracle": block["regressor_oracle"].get("nse"),
                }
            )
            confusion.append({"modelo": name, "protocolo": protocol, **block["confusion_matrix"]})
            for shape, entry in block["by_shape"].items():
                by_shape.append({"modelo": name, "protocolo": protocol, "forma": shape, **entry})

            subset = oof[oof["protocol"] == protocol]
            idx = subset["sample_idx"].to_numpy()
            false_neg = (y_true[idx] == 1) & (subset["y_pred_clf"].to_numpy() == 0)
            in_band = false_neg & (volume[idx] < BAND_MAX_M3)
            band.append(
                {
                    "modelo": name,
                    "protocolo": protocol,
                    "fn_total": int(false_neg.sum()),
                    "fn_en_banda_1_2m3": int(in_band.sum()),
                    "fn_fuera_de_banda": int((false_neg & ~in_band).sum()),
                    "fn_con_vol_ge_100m3": int((false_neg & (volume[idx] >= 100.0)).sum()),
                }
            )

    return {
        "comparacion_protocolos": pd.DataFrame(comparison),
        "matriz_confusion": pd.DataFrame(confusion),
        "banda_limitrofe": pd.DataFrame(band),
        "por_forma": pd.DataFrame(by_shape),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-id", required=True)
    parser.add_argument("--bench-root", type=Path, default=Path("outputs/bench"))
    parser.add_argument("--out", type=Path, default=Path("docs/auditoria/correcciones/datos"))
    parser.add_argument("--flood-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    prepared = resolve_prepared(args.bench_root / "prepared", args.prep_id, PROTOCOLS)
    oof_dir = args.bench_root / "oof" / args.prep_id
    oof_by_name = {path.stem: pd.read_parquet(path) for path in sorted(oof_dir.glob("*.parquet"))}
    tables = build_tables(prepared, oof_by_name, args.flood_threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} filas -> {args.out / (name + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Nota para el implementador: `resolve_prepared` exige que el PreparedDataset cubra los protocolos pedidos; si el prepared real solo contiene un subconjunto de `PROTOCOLS`, pasar `tuple(prepared_protocols)` leído del manifiesto en vez de `PROTOCOLS`.

- [ ] **Step 3: Verificar las pruebas**

Run: `..\..\..\venv\Scripts\python.exe -m pytest tests/scripts/test_informe_evaluacion.py -q -p no:cacheprovider --basetemp=%TEMP%\pt_t8`
Expected: PASS.

- [ ] **Step 4: Ejecución real (solo XGBoost y referencias)**

Esta ejecución usa los datos reales y tarda varios minutos. Ejecutarla en segundo plano, redirigiendo la salida a un archivo en el scratchpad. **No** correr el MLP ni las demás familias aquí.

```
..\..\..\venv\Scripts\python.exe main.py --bench-evaluate --models xgboost --prep-id <PREP_ID_V4_21>
```

`<PREP_ID_V4_21>` es el PreparedDataset vigente del contrato `tabular_v4_21`: es el directorio de `outputs/bench/prepared/` cuyo `manifest` declara `feature_contract_id` = `tabular_v4_21` y los tres protocolos de `config.yaml`. Si ese directorio no existe o no cubre `bench.protocols`, ejecutar antes `main.py --bench-prepare` (escribe solo en `outputs/bench/prepared/`) y usar el `prep_id` que imprime. Verificar que `--bench-evaluate` NO crea ni modifica nada en `outputs/models/` (comparar `git status outputs/models` antes y después).

- [ ] **Step 5: Generar las tablas**

```
..\..\..\venv\Scripts\python.exe scripts\informe_evaluacion.py --prep-id <PREP_ID_V4_21>
```

Expected: cuatro CSV en `docs/auditoria/correcciones/datos/`. Revisar que la fila `baseline_interpolacion_factor` / `LOSO` existe y que `xgboost` / `LeaveOneShapeOut` trae las 40 formas en `por_forma.csv`.

- [ ] **Step 6: Commit**

```bash
git add scripts/informe_evaluacion.py scripts/__init__.py tests/scripts/test_informe_evaluacion.py docs/auditoria/correcciones/datos
git commit -m "feat: script del informe de evaluacion y tablas de XGBoost y referencias sobre tabular_v4_21"
```

---

### Task 9: Spike de la resolución de la etiqueta (D-04/H-06) — informe, sin cambios al paquete

Es un spike: su producto es un documento con números y una recomendación. Todo el código de diagnóstico vive en el scratchpad de la sesión; no se modifica nada del paquete ni de `outputs/`.

**Files:**
- Create: `docs/auditoria/correcciones/03-resolucion-de-la-etiqueta.md`

- [ ] **Step 1: Entender cómo se lee hoy el volumen**

Leer `swmm_resilience/extraction/labels.py` y `swmm_resilience/simulation/swmm_api_io.py` (la lectura del reporte `.rpt`, "Flooding Summary" / "Total Flood Volume", unidades de 10^6 L con 3 decimales) y `swmm_resilience/simulation/runner.py` para ver cómo se invoca SWMM por escenario. Documentar en el informe la fórmula exacta y la resolución resultante (1 m³).

- [ ] **Step 2: Medir si el volumen se puede obtener con mayor resolución**

Sobre COPIAS del `.inp` en el scratchpad, simular 6 combinaciones forma/factor de control (incluir `base`, una forma corta, una larga y factores 0,2, 1,0, 3,0 y 5,0) y comparar, nodo por nodo:
(a) el volumen del `.rpt` (resolución actual);
(b) el volumen obtenido integrando en el tiempo la serie de caudal de inundación del archivo binario `.out` (o `pyswmm`/`swmm_api` `Output`), con la convención de unidades de la red (LPS);
(c) si SWMM expone el volumen de inundación acumulado por nodo con más decimales por otra vía.
Reportar la diferencia máxima, la mediana y cuántos nodos cambian de etiqueta con el umbral actual de 1,0 m³ y con un umbral continuo.

- [ ] **Step 3: Estimar el costo de re-simular**

Cronometrar las 6 simulaciones y extrapolar a las 1.000 corridas del dataset (40 formas × 25 factores). Estimar también el costo posterior: regeneración del dataset, nuevo `prep_id`, recálculo de features v4_21 y re-ejecución del banco.

- [ ] **Step 4: Cuantificar el efecto sobre la matriz de confusión**

Con los OOF de XGBoost (Task 8) reportar cuántos falsos negativos y falsos positivos caen en la banda de 1 a 2 m³ y qué pasaría con el F1 agregado si esa banda se excluyera de la evaluación (camino A) frente a usar etiquetas de resolución completa (camino B, estimado a partir de la muestra de control).

- [ ] **Step 5: Escribir el informe del spike**

Documento con: fórmula actual, resolución, tabla de comparación `.rpt` vs `.out`, costo estimado, efecto sobre la matriz de confusión, y una recomendación explícita entre:
- **A:** conservar la resolución y el umbral actuales, declararlos como supuesto y reportar la matriz con la banda limítrofe separada;
- **B:** re-simular y regenerar el dataset.
Terminar con la sección "Decisión pendiente del autor". **No ejecutar B.**

- [ ] **Step 6: Commit**

```bash
git add docs/auditoria/correcciones/03-resolucion-de-la-etiqueta.md
git commit -m "docs: spike de la resolucion de la etiqueta de inundacion (D-04, H-06)"
```

---

### Task 10: Documentación para la investigación

**Files:**
- Create: `docs/auditoria/correcciones/00-indice.md`
- Create: `docs/auditoria/correcciones/01-evaluacion-honesta.md`
- Create: `docs/auditoria/correcciones/02-trazabilidad-y-seguridad.md`
- Create: `docs/auditoria/correcciones/04-informe-consolidado.md`

Cada hallazgo se escribe con esta estructura exacta: **Problema y evidencia** · **Causa** · **Qué cambió** (archivos y hash de commit) · **Verificación** (cifras antes y después, con la fuente de cada número) · **Límites que quedan** · **Texto para la tesis** (párrafo listo para pegar, en español académico, sin jerga de implementación).

- [ ] **Step 1: `01-evaluacion-honesta.md`** — M-01 (referencias sin aprendizaje: qué miden LOSO/GroupKFold5 frente a LeaveOneShapeOut), M-02 (medición con v4_21: NSE y error por forma, concentración del error, comparación con el 0,814 previo), M-03 (valor agregado frente a media por fold: 0,9656 frente a 0,7408 en LOSO con la definición anterior), persistencia de OOF y matriz de confusión (tabla de `matriz_confusion.csv` y de `banda_limitrofe.csv`). Todas las cifras salen de `docs/auditoria/correcciones/datos/*.csv`; citar el archivo.

- [ ] **Step 2: `02-trazabilidad-y-seguridad.md`** — V-01 (qué figuras y salidas son en muestra), V-07 (`halve_timeseries`), M-07/V-04 (`verify_promotion`, `provenance.json`, y la nota de que en el worktree los SHA ya coincidían, mientras que el checkout atrasado tenía `660be918…` frente a `e2e7925e…`).

- [ ] **Step 3: `04-informe-consolidado.md`** — tabla hallazgo → estado → commit → documento; tabla "antes y después" de las métricas clave (F1, RMSE, NSE, matriz de confusión) para XGBoost y las dos referencias en los tres protocolos; limitaciones; y un resumen de una página para el informe de la investigación. Incluir la sección "Pendiente": D-04/H-06 (decisión A/B) y la Parte 2.

- [ ] **Step 4: `00-indice.md`** — índice con enlaces relativos a los documentos y a los CSV.

- [ ] **Step 5: Revisión de cifras**

Para cada cifra del texto, comprobar que aparece en un CSV de `datos/` o en la salida de un test citado. Reemplazar cualquier número que no se pueda rastrear. Ejecutar `..\..\..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=%TEMP%\pt_t10 -o addopts="-m 'not scale' -q"` y copiar el conteo final de tests al informe.

- [ ] **Step 6: Commit**

```bash
git add docs/auditoria/correcciones
git commit -m "docs: correcciones de la auditoria documentadas para la investigacion"
```

---

## Autorrevisión del plan

**Cobertura de la spec:**
- §2.2 referencias → Task 3 · §2.3 agregación → Tasks 1–2 · §2.4 `by_shape` → Tasks 1–2 · §2.5 OOF y proveniencia → Task 4 · §2.6 en muestra → Task 5 · §2.7 `halve_timeseries` → Task 6 · §2.8 `verify_promotion` y figuras → Task 7 · §2.9 spike → Task 9 · §5 documentación → Task 10 · medición real (M-02) → Task 8.
- Test de contrato de `outputs/models`: ya existe en `test_full_bench_run_writes_ranking_without_promoting` (Task 4 la conserva y la nueva prueba de CLI repite la aserción).

**Escaneo de marcadores vacíos:** sin pendientes de redacción. Dos puntos donde el implementador debe leer el código real antes de editar, ambos señalados con su criterio de decisión: el nombre público de la función de `flood_volume_curve.py` (Task 5, Step 2) y el punto de llamada en `runner.py` (Task 7, Step 6).

**Consistencia de tipos:** `score_predictions(prepared, oof, provenance, protocols=None)` se define en la Task 2 y se usa con `protocols=` en las Tasks 3 y 8. Las claves `classifier`, `confusion_matrix`, `classifier_per_fold`, `classifier_legacy_mean_zero_fill`, `end_to_end_pooled` y `by_shape` se definen en la Task 2 y se leen con los mismos nombres en la Task 8. `PROTOCOL_INTERPRETATION` se define en la Task 3 y se usa en las Tasks 4 y 8. `write_oof`, `write_metrics_files` y `write_provenance` se definen y se prueban en la Task 4 con las firmas que usa `main.py`.
