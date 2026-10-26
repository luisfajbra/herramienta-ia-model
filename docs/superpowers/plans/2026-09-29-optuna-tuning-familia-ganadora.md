# Búsqueda automática de hiperparámetros (Optuna) — Implementation Plan

> **Para agentes:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan casillas de verificación.

**Goal:** dar al banco de modelos un mecanismo de búsqueda automática de hiperparámetros con Optuna, aislado del camino de comparación de familias (`--bench`), invocable manualmente vía `--bench-tune` sobre una sola familia a la vez.

**Architecture:** módulo nuevo `swmm_resilience/ml/bench/tune.py` que reutiliza el `PreparedDataset` existente y un helper de fit/predict extraído de `evaluate.py`; genera sus propios folds internos (`GroupKFold(5)` por `shape_id`, efímeros, nunca persistidos como protocolo del banco) y corre un `optuna.Study` que minimiza `end_to_end.rmse_vol_todos_nodos`. El resultado se guarda como JSON etiquetado explícitamente como estimación exploratoria (no hay holdout de formas separado).

**Tech Stack:** Python 3.11, Optuna (nuevo), scikit-learn `GroupKFold`, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-optuna-tuning-familia-ganadora-design.md`

## Global Constraints

- `--bench-tune` es siempre un flag explícito; `--bench` (`run_all`) nunca lo activa implícitamente.
- `--bench-tune` exige exactamente una familia en `--models`; cero o más de una es `SystemExit`.
- Ningún estudio de tuning escribe en `outputs/models/` ni en `outputs/bench/reports/ranking.csv`.
- Todo `study.json` lleva `"status": "estimacion_exploratoria_condicionada"` y el `status_reason` fijo del spec §4.4, porque no hay holdout de formas separado ni selección de familia anidada.
- Los folds internos de tuning (`GroupKFold(5)` por `shape_id`) nunca se declaran como protocolo del banco: no tocan `PROTOCOL_GROUP_COLUMNS`, `group_column_for()` ni el manifiesto de `PreparedDataset`.
- La semilla del estudio reutiliza `ML_RANDOM_STATE` de `swmm_resilience/config.py`; no se declara una semilla nueva en `config.yaml`.
- `optuna` se importa perezosamente (dentro de `tune.py`, y `tune.py` se importa perezosamente dentro de `_run_bench()`), igual que las demás etapas del banco, para no pagar ese costo cuando `--bench-tune` no se usa.
- Espacios de búsqueda declarados en código (`SEARCH_SPACES` en `tune.py`), no en YAML — exactos a los de la tabla del spec §6.
- No hacer commit salvo que se indique explícitamente entre tareas (el usuario controla la integración en esta rama).

---

## Mapa de archivos

- `requirements.txt`: añade `optuna` con versión fijada.
- `tests/conftest.py`: fixture nueva `sql_training_db_six_shapes` (≥5 `shape_id` distintos), sibling de `sql_training_db_six_factors`.
- `swmm_resilience/ml/bench/evaluate.py`: extrae `_fit_predict_fold()` desde `evaluate_candidate()`.
- `swmm_resilience/ml/bench/tune.py` (nuevo): `build_tuning_folds()`, `SEARCH_SPACES`, `_sample_params()`, `_objective()`, `TuneResult`, `run_study()`, `save_tune_result()`.
- `swmm_resilience/config.py`: `BenchTuningConfig` dataclass, extensión de `_parse_bench()` y `BenchConfig`.
- `config.yaml`: bloque opcional `bench.tuning`.
- `main.py`: flag `--bench-tune`, validación de una sola familia, resolución de `PreparedDataset`, invocación de `run_study()`/`save_tune_result()`.
- Tests nuevos: `tests/ml/bench/test_tune.py`. Tests modificados: `tests/ml/bench/test_bench_config.py`, `tests/ml/bench/test_bench_cli.py`.

## Interfaces que produce el plan

```python
# evaluate.py
def _fit_predict_fold(
    prepared: PreparedDataset, family_name: str,
    classifier_params: dict, regressor_params: dict,
    train_idx: np.ndarray, test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # (y_pred_clf, y_prob_clf, y_pred_reg)

# tune.py
def build_tuning_folds(keys: pd.DataFrame, n_splits: int = 5, seed: int = ML_RANDOM_STATE) -> pd.DataFrame: ...
SEARCH_SPACES: dict[str, dict[str, dict[str, tuple]]]
def run_study(
    prepared: PreparedDataset, family_name: str,
    n_trials: int = 50, n_splits: int = 5, seed: int = ML_RANDOM_STATE,
) -> TuneResult: ...
def save_tune_result(result: TuneResult, output_dir: Path) -> Path: ...

# config.py
@dataclass
class BenchTuningConfig:
    n_trials: int
    n_splits: int
# BenchConfig gana el campo: tuning: BenchTuningConfig
```

---

### Task 1: Instalar y fijar la dependencia de Optuna

**Files:**
- Modificar: `requirements.txt`

**Consume:** nada.

**Produce:** `optuna` importable desde `venv`, versión exacta fijada en `requirements.txt`.

- [ ] **Paso 1: instalar Optuna en el venv del proyecto y ver qué versión resuelve.**

Ejecutar: `venv\Scripts\python.exe -m pip install optuna`

- [ ] **Paso 2: capturar la versión instalada y fijarla en `requirements.txt`.**

Ejecutar: `venv\Scripts\python.exe -c "import optuna; print(optuna.__version__)"`

Añadir a `requirements.txt` (mismo estilo `paquete==version` que el resto del archivo) una línea nueva:

```text
optuna==<version impresa en el paso anterior>
```

- [ ] **Paso 3: verificar el import y que no rompe nada del resto de la suite.**

Ejecutar: `venv\Scripts\python.exe -c "import optuna; from optuna.samplers import TPESampler; print('ok')"`

Esperado: imprime `ok` sin traceback.

---

### Task 2: Fixture de base de datos con seis formas para tuning

**Files:**
- Modificar: `tests/conftest.py`

**Consume:** nada nuevo (sigue el patrón exacto de `csv_shaped_dataset_six_factors` / `sql_training_db_six_factors`, ya presentes en el archivo).

**Produce:** `sql_training_db_six_shapes`, una base con 6 `shape_id` distintos (necesaria porque `GroupKFold(5)` exige al menos 5 grupos, y las fixtures actuales solo tienen 2 formas).

- [ ] **Paso 1: escribir la fixture nueva, sibling de las de seis factores.**

Añadir en `tests/conftest.py`, después de `sql_training_db_six_factors`:

```python
@pytest.fixture
def csv_shaped_dataset_six_shapes() -> pd.DataFrame:
    """Like csv_shaped_dataset but with 6 distinct shape_id values.

    GroupKFold(5) para tuning necesita al menos 5 grupos de shape_id; las
    2 formas de csv_shaped_dataset no alcanzan. Sibling explicito, igual
    que csv_shaped_dataset_six_factors: las fixtures existentes no cambian.
    """
    rows = []
    shapes = ("base", "storm_a", "storm_b", "storm_c", "storm_d", "storm_e")
    for shape_id in shapes:
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
def sql_training_db_six_shapes(
    tmp_path: Path, csv_shaped_dataset_six_shapes: pd.DataFrame
) -> Path:
    """Path to a migrated v17 database with 6 shape_id groups (GroupKFold(5)-by-shape-capable)."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / "fixture_network_shapes.inp"
    inp_path.write_text("[TITLE]\nfixture network shapes\n", encoding="utf-8")
    db_path = tmp_path / "training_v17_shapes.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(
            conn, csv_shaped_dataset_six_shapes, inp_path, "Fixture Network Shapes"
        )
    finally:
        conn.close()
    return db_path
```

- [ ] **Paso 2: verificar que la fixture nueva es utilizable de forma aislada.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_preprocess.py -q --basetemp C:\Windows\Temp\pytest-optuna-task2 -p no:cacheprovider`

Esperado: la suite existente sigue en verde (la fixture nueva no se usa aún, pero no debe romper la colección de conftest.py).

---

### Task 3: Extraer `_fit_predict_fold()` de `evaluate_candidate()`

**Files:**
- Modificar: `swmm_resilience/ml/bench/evaluate.py`
- Test: `tests/ml/bench/test_evaluate_candidate.py` (no se modifica — debe seguir en verde sin cambios, como prueba de paridad del refactor)

**Consume:** nada nuevo del banco; solo reordena código existente.

**Produce:** `_fit_predict_fold()`, la única función que decide fit/predict de un fold, compartida más adelante por `tune.py`.

- [ ] **Paso 1: confirmar el comportamiento actual como referencia (ya está en verde).**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parity_with_legacy_evaluator.py -q --basetemp C:\Windows\Temp\pytest-optuna-task3-before -p no:cacheprovider`

Esperado: todos pasan (esta es la foto "antes" contra la que se compara el refactor).

- [ ] **Paso 2: extraer el helper dentro de `evaluate.py`.**

En `swmm_resilience/ml/bench/evaluate.py`, añadir antes de `evaluate_candidate`:

```python
def _fit_predict_fold(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta clasificador+regresor SOLO con train_idx, predice sobre test_idx.

    Única función que decide fit/predict de un fold: evaluate_candidate()
    (folds exteriores de comparación) y tune.py (folds interiores de
    búsqueda de hiperparámetros) la comparten, así que la garantía de que
    ningún test toca .fit() vive en un solo lugar.

    Devuelve (y_pred_clf, y_prob_clf, y_pred_reg).
    """
    from .registry import get_family

    family = get_family(family_name)
    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train = y_clf[train_idx]
    yr_train = y_reg[train_idx]

    n_negative, n_positive = int((yc_train == 0).sum()), int((yc_train == 1).sum())
    scale_pos_weight = n_negative / n_positive if n_positive else 1.0

    classifier = family.build_classifier(classifier_params, scale_pos_weight)
    classifier.fit(X_train, yc_train)
    yc_pred = classifier.predict(X_test)
    yc_prob = classifier.predict_proba(X_test)[:, 1]

    flooded_train = yc_train == 1
    if flooded_train.sum():
        regressor = family.build_regressor(regressor_params)
        regressor.fit(X_train.iloc[flooded_train], np.log1p(yr_train[flooded_train]))
        yr_pred = np.clip(np.expm1(regressor.predict(X_test)), a_min=0.0, a_max=None)
    else:
        # Ver el comentario original en evaluate_candidate: divergencia
        # deliberada de _run_cv cuando el fold no tiene filas inundadas de
        # entrenamiento -- se predice volumen cero en vez de reventar con
        # NotFittedError.
        yr_pred = np.zeros(len(test_idx), dtype=float)

    return yc_pred.astype(int), yc_prob.astype(float), yr_pred.astype(float)
```

Reemplazar el cuerpo del bucle `for (protocol, fold_id), chunk in prepared.folds.groupby(...)` dentro de `evaluate_candidate()` para que llame a `_fit_predict_fold(prepared, family_name, classifier_params, regressor_params, train_idx, test_idx)` en vez de repetir la lógica inline, conservando exactamente la construcción del `records.append(pd.DataFrame({...}))` que ya existe (mismas columnas, mismo orden).

- [ ] **Paso 3: confirmar que el refactor no cambió ningún resultado.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parity_with_legacy_evaluator.py tests/ml/bench/test_score_predictions.py -q --basetemp C:\Windows\Temp\pytest-optuna-task3-after -p no:cacheprovider`

Esperado: exactamente los mismos tests en verde que en el Paso 1, sin ninguno modificado — es la prueba de que `_fit_predict_fold()` produce resultados idénticos a la lógica que reemplaza (incluye `test_running_it_twice_gives_identical_predictions`, `test_scale_pos_weight_is_computed_per_fold_not_globally` y `test_the_scaler_is_fitted_inside_the_fold_not_on_the_whole_dataset`, que ejercen exactamente las garantías anti-fuga que este helper debe preservar).

---

### Task 4: `build_tuning_folds()` — GroupKFold(5) interno por `shape_id`

**Files:**
- Crear: `swmm_resilience/ml/bench/tune.py`
- Test: `tests/ml/bench/test_tune.py` (nuevo)

**Consume:** nada del banco todavía (función pura sobre un `pd.DataFrame` de `keys`).

**Produce:** `build_tuning_folds(keys, n_splits=5, seed=ML_RANDOM_STATE) -> pd.DataFrame` con columnas `('fold_id', 'split', 'sample_idx')`.

- [ ] **Paso 1: escribir la prueba de partición completa sin fuga.**

Crear `tests/ml/bench/test_tune.py`:

```python
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.tune import build_tuning_folds


def _keys_six_shapes() -> pd.DataFrame:
    """6 formas x 5 factores, para que GroupKFold(5) tenga margen."""
    shapes = ["s0", "s1", "s2", "s3", "s4", "s5"]
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = [
        {"shape_id": shape, "factor_mult": factor}
        for shape in shapes
        for factor in factors
    ]
    return pd.DataFrame(rows)


def test_build_tuning_folds_has_the_declared_columns():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    assert tuple(folds.columns) == ("fold_id", "split", "sample_idx")
    assert folds["fold_id"].nunique() == 5


def test_build_tuning_folds_partitions_without_shape_leakage():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    for _, chunk in folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        train_shapes = set(keys["shape_id"].iloc[train_idx])
        test_shapes = set(keys["shape_id"].iloc[test_idx])
        assert train_shapes.isdisjoint(test_shapes)
        assert len(train_idx) + len(test_idx) == len(keys)


def test_build_tuning_folds_covers_every_row_across_folds_as_test_exactly_once():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    test_rows = folds.loc[folds["split"] == "test", "sample_idx"]
    assert sorted(test_rows.tolist()) == list(range(len(keys)))
    assert not test_rows.duplicated().any()


def test_build_tuning_folds_is_deterministic_given_the_same_seed():
    keys = _keys_six_shapes()
    first = build_tuning_folds(keys, n_splits=5, seed=7)
    second = build_tuning_folds(keys, n_splits=5, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_build_tuning_folds_rejects_too_few_shapes_for_n_splits():
    keys = pd.DataFrame({"shape_id": ["a", "b", "c"], "factor_mult": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="5"):
        build_tuning_folds(keys, n_splits=5)
```

- [ ] **Paso 2: ejecutar la prueba y comprobar que falla porque `tune.py` no existe.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task4-red -p no:cacheprovider`

Esperado: falla con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.tune'`.

- [ ] **Paso 3: crear `tune.py` con `build_tuning_folds()`.**

Crear `swmm_resilience/ml/bench/tune.py`:

```python
"""Búsqueda automática de hiperparámetros (Optuna), aislada del banco.

Este módulo NUNCA se invoca desde --bench: solo desde --bench-tune,
explícito, sobre una sola familia. Los folds que genera son internos y
efímeros -- GroupKFold(5) agrupado por shape_id sobre todo el
PreparedDataset -- y nunca se declaran como protocolo del banco: no tocan
PROTOCOL_GROUP_COLUMNS, group_column_for() ni el manifiesto persistido de
PreparedDataset (ver spec docs/superpowers/specs/2026-09-29-optuna-tuning-
familia-ganadora-design.md §4.2).

Al no reservar un holdout de formas separado, todo resultado de un estudio
se etiqueta "estimacion_exploratoria_condicionada" (spec §12 del diseño de
cierre de auditoría): no debe presentarse como confirmatorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ...config import ML_RANDOM_STATE


def build_tuning_folds(
    keys: pd.DataFrame, n_splits: int = 5, seed: int = ML_RANDOM_STATE
) -> pd.DataFrame:
    """GroupKFold(n_splits) agrupado por shape_id sobre todo `keys`.

    Devuelve un frame con columnas ('fold_id', 'split', 'sample_idx'), sin
    'protocol': estos folds son efímeros, por-trial, y nunca se escriben en
    el manifiesto de PreparedDataset.
    """
    groups = keys["shape_id"]
    distinct = int(groups.nunique())
    if distinct < n_splits:
        raise ValueError(
            f"GroupKFold({n_splits}) para tuning necesita al menos {n_splits} "
            f"formas (shape_id) distintas; el dataset tiene {distinct}."
        )
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {"fold_id": fold_id, "split": split_name, "sample_idx": sample_idx}
                )
    return pd.DataFrame.from_records(records, columns=["fold_id", "split", "sample_idx"])
```

- [ ] **Paso 4: ejecutar la prueba y comprobar que pasa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task4-green -p no:cacheprovider`

Esperado: 6 passed.

---

### Task 5: Espacios de búsqueda y muestreo de parámetros

**Files:**
- Modificar: `swmm_resilience/ml/bench/tune.py`
- Modificar: `tests/ml/bench/test_tune.py`

**Consume:** nada nuevo de otras tareas.

**Produce:** `SEARCH_SPACES` (dict por familia/tarea/parámetro) y `_sample_params(trial, space, prefix)`.

- [ ] **Paso 1: escribir las pruebas de muestreo para las cinco familias.**

Añadir a `tests/ml/bench/test_tune.py`:

```python
import optuna

from swmm_resilience.ml.bench.tune import SEARCH_SPACES, _sample_params


@pytest.mark.parametrize("family_name", ["xgboost", "random_forest", "linear", "svm", "mlp"])
@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_sample_params_returns_only_declared_keys_within_range(family_name, task):
    space = SEARCH_SPACES[family_name][task]
    study = optuna.create_study()
    trial = study.ask()

    params = _sample_params(trial, space, prefix=f"{family_name}_{task}_")

    assert set(params) == set(space)
    for name, spec in space.items():
        kind = spec[0]
        if kind in ("int", "float", "float_log"):
            low, high = spec[1], spec[2]
            assert low <= params[name] <= high
        elif kind == "categorical":
            assert params[name] in spec[1]


def test_sample_params_does_not_collide_between_classifier_and_regressor_in_the_same_trial():
    """El prefijo evita que Optuna vea el mismo nombre de parametro dos veces
    en un trial con rangos distintos (p.ej. xgboost.max_depth en ambas tareas)."""
    study = optuna.create_study()
    trial = study.ask()

    clf_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["classifier"], prefix="xgboost_classifier_"
    )
    reg_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["regressor"], prefix="xgboost_regressor_"
    )

    assert set(clf_params) == set(SEARCH_SPACES["xgboost"]["classifier"])
    assert set(reg_params) == set(SEARCH_SPACES["xgboost"]["regressor"])
```

- [ ] **Paso 2: ejecutar y comprobar que falla por ausencia de `SEARCH_SPACES`/`_sample_params`.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task5-red -p no:cacheprovider`

Esperado: `ImportError: cannot import name 'SEARCH_SPACES'`.

- [ ] **Paso 3: añadir `SEARCH_SPACES` y `_sample_params()` a `tune.py`.**

Añadir a `swmm_resilience/ml/bench/tune.py` (después de `build_tuning_folds`):

```python
# Notación por parámetro: ("int", low, high) | ("float", low, high) |
# ("float_log", low, high) muestreado log-uniforme | ("categorical", [...]).
# Rangos centrados en los defaults actuales de config.yaml (ver spec §6).
SEARCH_SPACES: dict[str, dict[str, dict[str, tuple]]] = {
    "xgboost": {
        "classifier": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
        "regressor": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
    },
    "random_forest": {
        "classifier": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
        "regressor": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
    },
    "linear": {
        "classifier": {"C": ("float_log", 1e-3, 100.0)},
        "regressor": {"alpha": ("float_log", 1e-3, 100.0)},
    },
    "svm": {
        "classifier": {
            "C": ("float_log", 1e-2, 100.0),
            "gamma": ("categorical", ["scale", "auto"]),
        },
        "regressor": {
            "C": ("float_log", 1e-2, 100.0),
            "epsilon": ("float", 0.01, 1.0),
        },
    },
    "mlp": {
        "classifier": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
        "regressor": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
    },
}


def _sample_params(trial, space: dict, *, prefix: str) -> dict:
    """Muestrea un dict de hiperparámetros desde `space` usando `trial`.

    `prefix` evita colisiones de nombre entre clasificador y regresor
    dentro del mismo trial (p.ej. "xgboost_classifier_max_depth" vs
    "xgboost_regressor_max_depth"): Optuna identifica cada valor muestreado
    por su nombre completo dentro del estudio.
    """
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        full_name = f"{prefix}{name}"
        if kind == "int":
            params[name] = trial.suggest_int(full_name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(full_name, spec[1])
        else:
            raise ValueError(f"Tipo de espacio desconocido: {kind!r}")
    return params
```

Añadir `import optuna` al bloque de imports de `tune.py` (junto a `from sklearn.model_selection import GroupKFold`).

- [ ] **Paso 4: ejecutar y comprobar que pasa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task5-green -p no:cacheprovider`

Esperado: todos los tests de la Task 4 y la Task 5 en verde (16 casos entre ambas, contando la parametrización 5 familias × 2 tareas).

---

### Task 6: `run_study()`, `TuneResult` y `save_tune_result()`

**Files:**
- Modificar: `swmm_resilience/ml/bench/tune.py`
- Modificar: `tests/ml/bench/test_tune.py`

**Consume:** `_fit_predict_fold()` (Task 3), `build_tuning_folds()` y `SEARCH_SPACES`/`_sample_params()` (Tasks 4–5), `sql_training_db_six_shapes` (Task 2).

**Produce:** `run_study(prepared, family_name, n_trials, n_splits, seed) -> TuneResult`, `save_tune_result(result, output_dir) -> Path`.

- [ ] **Paso 1: escribir las pruebas de estudio completo y persistencia.**

Añadir a `tests/ml/bench/test_tune.py`:

```python
import json

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.tune import run_study, save_tune_result


@pytest.fixture
def prepared_six_shapes(sql_training_db_six_shapes, tmp_path):
    return prepare_dataset(
        sql_training_db_six_shapes,
        protocols=("LeaveOneShapeOut",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_run_study_returns_a_finite_best_value_and_matching_param_keys(prepared_six_shapes):
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)

    assert result.family == "xgboost"
    assert result.prep_id == prepared_six_shapes.prep_id
    assert result.n_trials == 2
    assert len(result.trials) == 2
    assert np.isfinite(result.best_value)
    assert set(result.best_classifier_params) == set(SEARCH_SPACES["xgboost"]["classifier"])
    assert set(result.best_regressor_params) == set(SEARCH_SPACES["xgboost"]["regressor"])
    assert result.status == "estimacion_exploratoria_condicionada"
    assert "holdout" in result.status_reason


def test_run_study_rejects_a_family_with_too_few_shapes_for_the_inner_cv(prepared_six_shapes):
    with pytest.raises(ValueError, match="5"):
        run_study(prepared_six_shapes, "xgboost", n_trials=1, n_splits=99)


def test_save_tune_result_writes_the_declared_schema(prepared_six_shapes, tmp_path):
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)
    path = save_tune_result(result, tmp_path / "tuning")

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "family", "prep_id", "inner_cv", "objective_metric", "direction",
        "n_trials", "seed", "optuna_version", "started_at_utc", "finished_at_utc",
        "duration_s", "trials", "best_classifier_params", "best_regressor_params",
        "best_value", "status", "status_reason",
    ):
        assert key in payload, f"falta {key} en study.json"
    assert payload["inner_cv"] == {
        "method": "GroupKFold", "n_splits": 5, "group_column": "shape_id"
    }
    assert payload["status"] == "estimacion_exploratoria_condicionada"


def test_save_tune_result_never_touches_outputs_models(prepared_six_shapes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)
    save_tune_result(result, tmp_path / "outputs" / "bench" / "tuning")

    assert not (tmp_path / "outputs" / "models").exists()
```

- [ ] **Paso 2: ejecutar y comprobar que falla por ausencia de `run_study`/`save_tune_result`.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task6-red -p no:cacheprovider`

Esperado: `ImportError: cannot import name 'run_study'`.

- [ ] **Paso 3: implementar el objetivo, `run_study()`, `TuneResult` y `save_tune_result()`.**

Añadir a `swmm_resilience/ml/bench/tune.py` (imports adicionales al principio: `import json`, `import time`, `from dataclasses import dataclass, field`, `from datetime import datetime, timezone`, `from importlib.metadata import version as _package_version`, `from pathlib import Path`, `from .evaluate import _fit_predict_fold`, `from .metrics import end_to_end_metrics`):

```python
_STATUS_EXPLORATORY = "estimacion_exploratoria_condicionada"
_STATUS_REASON_EXPLORATORY = (
    "No se reservó un holdout de formas separado ni se aplicó selección de "
    "familia anidada (spec §12 del diseño de cierre de auditoría, sección "
    "'Optimización posterior'); este resultado no debe presentarse como "
    "confirmatorio."
)


def _objective(trial, prepared, family_name: str, tuning_folds: pd.DataFrame) -> float:
    classifier_params = _sample_params(
        trial, SEARCH_SPACES[family_name]["classifier"], prefix=f"{family_name}_classifier_"
    )
    regressor_params = _sample_params(
        trial, SEARCH_SPACES[family_name]["regressor"], prefix=f"{family_name}_regressor_"
    )
    trial.set_user_attr("classifier_params", classifier_params)
    trial.set_user_attr("regressor_params", regressor_params)

    y_clf_all = prepared.y_clf.to_numpy()
    y_reg_all = prepared.y_reg.to_numpy()

    fold_values: list[float] = []
    for _, chunk in tuning_folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()

        yc_pred, _, yr_pred = _fit_predict_fold(
            prepared, family_name, classifier_params, regressor_params, train_idx, test_idx
        )
        routed = np.where(yc_pred == 1, yr_pred, 0.0)
        fold_values.append(
            end_to_end_metrics(
                y_reg_all[test_idx], routed, y_clf_all[test_idx], yc_pred,
            )["rmse_vol_todos_nodos"]
        )
    return float(np.mean(fold_values))


@dataclass(frozen=True)
class TuneResult:
    family: str
    prep_id: str
    inner_cv: dict
    objective_metric: str
    direction: str
    n_trials: int
    seed: int
    optuna_version: str
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    trials: list = field(default_factory=list)
    best_classifier_params: dict = field(default_factory=dict)
    best_regressor_params: dict = field(default_factory=dict)
    best_value: float = float("nan")
    status: str = _STATUS_EXPLORATORY
    status_reason: str = _STATUS_REASON_EXPLORATORY


def run_study(
    prepared: PreparedDataset,
    family_name: str,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = ML_RANDOM_STATE,
) -> TuneResult:
    """Corre un estudio Optuna sobre `family_name`, inner GroupKFold(n_splits) por shape_id.

    Etiqueta el resultado como estimación exploratoria: no hay holdout de
    formas separado del que usa esta búsqueda (spec §12).
    """
    if family_name not in SEARCH_SPACES:
        raise ValueError(
            f"Familia sin espacio de búsqueda declarado: {family_name!r}. "
            f"Disponibles: {', '.join(sorted(SEARCH_SPACES))}"
        )

    tuning_folds = build_tuning_folds(prepared.keys, n_splits=n_splits, seed=seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    started = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    study.optimize(
        lambda trial: _objective(trial, prepared, family_name, tuning_folds),
        n_trials=n_trials,
    )
    duration_s = time.perf_counter() - start_perf
    finished = datetime.now(timezone.utc)

    trials = [
        {
            "number": t.number,
            "classifier_params": t.user_attrs["classifier_params"],
            "regressor_params": t.user_attrs["regressor_params"],
            "value": t.value,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        for t in study.trials
    ]

    return TuneResult(
        family=family_name,
        prep_id=prepared.prep_id,
        inner_cv={"method": "GroupKFold", "n_splits": n_splits, "group_column": "shape_id"},
        objective_metric="end_to_end.rmse_vol_todos_nodos",
        direction="minimize",
        n_trials=n_trials,
        seed=seed,
        optuna_version=_package_version("optuna"),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_s=duration_s,
        trials=trials,
        best_classifier_params=study.best_trial.user_attrs["classifier_params"],
        best_regressor_params=study.best_trial.user_attrs["regressor_params"],
        best_value=float(study.best_value),
    )


def save_tune_result(result: TuneResult, output_dir: Path | str) -> Path:
    """Escribe `result` en <output_dir>/<family>/<prep_id>_<timestamp>/study.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_dir) / result.family / f"{result.prep_id}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "study.json"
    payload = {
        "family": result.family,
        "prep_id": result.prep_id,
        "inner_cv": result.inner_cv,
        "objective_metric": result.objective_metric,
        "direction": result.direction,
        "n_trials": result.n_trials,
        "seed": result.seed,
        "optuna_version": result.optuna_version,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "duration_s": result.duration_s,
        "trials": result.trials,
        "best_classifier_params": result.best_classifier_params,
        "best_regressor_params": result.best_regressor_params,
        "best_value": result.best_value,
        "status": result.status,
        "status_reason": result.status_reason,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

Añadir también `from .schemas import PreparedDataset` a los imports de `tune.py` (para la anotación de tipo de `run_study`).

- [ ] **Paso 4: ejecutar y comprobar que pasa.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_tune.py -q --basetemp C:\Windows\Temp\pytest-optuna-task6-green -p no:cacheprovider`

Esperado: todos los tests de `test_tune.py` en verde (Tasks 4–6 combinadas).

---

### Task 7: `BenchTuningConfig` en `config.py` y `config.yaml`

**Files:**
- Modificar: `swmm_resilience/config.py`
- Modificar: `config.yaml`
- Modificar: `tests/ml/bench/test_bench_config.py`

**Consume:** nada de las tareas anteriores (paralelo a ellas en la capa de configuración).

**Produce:** `BenchTuningConfig`, campo `BenchConfig.tuning`, parseo con defaults `n_trials=50`, `n_splits=5` cuando el bloque falta.

- [ ] **Paso 1: escribir las pruebas de parseo.**

Añadir a `tests/ml/bench/test_bench_config.py` (junto a los tests existentes de `_BENCH_YAML`):

```python
def test_bench_tuning_defaults_when_the_block_is_absent(tmp_path):
    config = load_config(_write_config(tmp_path, _BENCH_YAML))
    assert config.bench.tuning.n_trials == 50
    assert config.bench.tuning.n_splits == 5


def test_bench_tuning_block_is_parsed_when_present(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 10, n_splits: 3}',
    )
    config = load_config(_write_config(tmp_path, with_tuning))
    assert config.bench.tuning.n_trials == 10
    assert config.bench.tuning.n_splits == 3


def test_bench_tuning_rejects_non_positive_n_trials(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 0, n_splits: 5}',
    )
    with pytest.raises(ValueError, match="n_trials"):
        load_config(_write_config(tmp_path, with_tuning))


def test_bench_tuning_rejects_non_positive_n_splits(tmp_path):
    with_tuning = _BENCH_YAML.replace(
        'promote: "none"',
        'promote: "none"\n  tuning: {n_trials: 50, n_splits: 0}',
    )
    with pytest.raises(ValueError, match="n_splits"):
        load_config(_write_config(tmp_path, with_tuning))
```

Cada test construye su propio bloque `tuning` completo en un solo `.replace()` (un solo valor inválido, el otro campo en su default), así que no hay riesgo de claves YAML duplicadas.

- [ ] **Paso 2: ejecutar y comprobar que falla porque `BenchConfig` no tiene `tuning`.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_bench_config.py -q --basetemp C:\Windows\Temp\pytest-optuna-task7-red -p no:cacheprovider`

Esperado: `AttributeError: 'BenchConfig' object has no attribute 'tuning'`.

- [ ] **Paso 3: añadir `BenchTuningConfig` y extender `_parse_bench()`.**

En `swmm_resilience/config.py`, junto a `BenchRankingConfig`:

```python
@dataclass
class BenchTuningConfig:
    n_trials: int
    n_splits: int
```

Modificar `BenchConfig` para añadir el campo:

```python
@dataclass
class BenchConfig:
    protocols: list
    ranking: BenchRankingConfig
    promote: str
    families: dict
    tuning: BenchTuningConfig
    ...
```

En `_parse_bench()`, antes del `return BenchConfig(...)`:

```python
    tuning_raw = raw_bench.get("tuning") or {}
    n_trials = int(tuning_raw.get("n_trials", 50))
    n_splits = int(tuning_raw.get("n_splits", 5))
    if n_trials <= 0:
        raise ValueError(f"bench.tuning.n_trials debe ser positivo; recibido {n_trials}")
    if n_splits <= 0:
        raise ValueError(f"bench.tuning.n_splits debe ser positivo; recibido {n_splits}")
```

Y añadir `tuning=BenchTuningConfig(n_trials=n_trials, n_splits=n_splits)` a la construcción de `BenchConfig(...)`.

En `config.yaml`, dentro del bloque `bench:` (después de `promote: "none"`):

```yaml
  tuning:
    n_trials: 50
    n_splits: 5
```

- [ ] **Paso 4: ejecutar y comprobar que pasa, y que el resto de la config sigue en verde.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_bench_config.py -q --basetemp C:\Windows\Temp\pytest-optuna-task7-green -p no:cacheprovider`

Esperado: todos los tests de `test_bench_config.py` en verde, incluidos los preexistentes.

---

### Task 8: CLI `--bench-tune`

**Files:**
- Modificar: `main.py`
- Modificar: `tests/ml/bench/test_bench_cli.py`

**Consume:** `run_study()`/`save_tune_result()` (Task 6), `BenchConfig.tuning` (Task 7).

**Produce:** flag `--bench-tune` que exige una sola familia, resuelve el `PreparedDataset` igual que `--bench-train`, corre el estudio y persiste el resultado, sin tocar `outputs/models/` ni el ranking.

- [ ] **Paso 1: escribir las pruebas de CLI.**

Primero extender los dos helpers existentes en `tests/ml/bench/test_bench_cli.py`.

Añadir `BenchTuningConfig` a los imports de `swmm_resilience.config` al inicio del archivo (junto a `BenchConfig, BenchFamilyConfig, BenchRankingConfig, Config, DatasetConfig, NetworkConfig`).

Modificar `_make_config`:

```python
def _make_config(
    *, inp_path, db_path, families, promote="auto", protocols=("LOSO",), tuning=None
):
    """Builds a Config with only the fields _run_bench actually reads.

    ``simulation``/``ml``/``evaluation``/``visualization`` are left as None:
    _run_bench never touches them, and Config is a plain dataclass with no
    runtime type enforcement.
    """
    family_configs = {
        name: BenchFamilyConfig(enabled=True, classifier=dict(TINY), regressor=dict(TINY))
        for name in families
    }
    bench = BenchConfig(
        protocols=list(protocols),
        ranking=BenchRankingConfig(
            primary_metric="end_to_end.rmse_vol_todos_nodos",
            primary_direction="minimize",
            tie_breakers=(("classifier.f1", "maximize"),),
        ),
        promote=promote,
        families=family_configs,
        tuning=tuning or BenchTuningConfig(n_trials=2, n_splits=5),
    )
```

(el resto del cuerpo de `_make_config` no cambia). `n_trials=2` mantiene los tests de CLI rápidos; `n_splits=5` combina con la fixture `sql_training_db_six_shapes` de la Task 2, que trae 6 formas.

Modificar `_make_args`, añadiendo `bench_tune=False` al dict `base`:

```python
def _make_args(**overrides):
    base = dict(
        bench=False,
        bench_prepare=False,
        bench_train=False,
        bench_evaluate=False,
        bench_promote=False,
        bench_tune=False,
        models=None,
        prep_id=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)
```

Con los dos helpers extendidos, añadir los tests de CLI:

```python
def test_bench_tune_requires_exactly_one_family(sql_training_db, tmp_path):
    import main

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost", "random_forest"],
    )
    args = _make_args(bench_tune=True)  # sin --models: toma las dos habilitadas

    with pytest.raises(SystemExit, match="exactamente una familia"):
        main._run_bench(args, config)


def test_bench_tune_rejects_a_prepared_dataset_with_stale_protocol_semantics(
    sql_training_db_six_shapes, tmp_path, monkeypatch
):
    import main

    monkeypatch.chdir(tmp_path)
    prepared = prepare_dataset(
        sql_training_db_six_shapes, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )
    manifest_path = Path("outputs/bench/prepared") / prepared.prep_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_definitions"]["LOSO"]["group_column"] = "shape_id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db_six_shapes,
        families=["xgboost"], protocols=("LOSO",),
    )
    args = _make_args(bench_tune=True, models="xgboost")

    with pytest.raises(SystemExit) as excinfo:
        main._run_bench(args, config)
    assert "--bench-prepare" in str(excinfo.value)


def test_bench_tune_writes_a_study_and_touches_neither_models_nor_ranking(
    sql_training_db_six_shapes, tmp_path, monkeypatch
):
    import main

    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db_six_shapes,
        families=["xgboost"], protocols=("LeaveOneShapeOut",),
    )
    main._run_bench(_make_args(bench_prepare=True), config)
    main._run_bench(_make_args(bench_tune=True, models="xgboost"), config)

    study_files = list((tmp_path / "outputs" / "bench" / "tuning").rglob("study.json"))
    assert len(study_files) == 1
    payload = json.loads(study_files[0].read_text(encoding="utf-8"))
    assert payload["family"] == "xgboost"
    assert payload["status"] == "estimacion_exploratoria_condicionada"

    assert not (tmp_path / "outputs" / "models").exists()
    assert not (tmp_path / "outputs" / "bench" / "reports" / "ranking.csv").exists()
```

- [ ] **Paso 2: ejecutar y comprobar que falla porque `--bench-tune` no existe.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_bench_cli.py -q --basetemp C:\Windows\Temp\pytest-optuna-task8-red -p no:cacheprovider`

Esperado: falla con `TypeError: __init__() got an unexpected keyword argument 'bench_tune'` o `AttributeError: 'Namespace' object has no attribute 'bench_tune'`.

- [ ] **Paso 3: añadir el flag y la lógica en `main.py`.**

Añadir el argumento, junto a `--bench-promote`:

```python
    parser.add_argument("--bench-tune", action="store_true",
                        help="Busqueda Optuna de hiperparametros sobre UNA familia (exploratoria)")
```

En `_run_bench()`, justo después del bloque `unknown_families` (antes de `if run_all or args.bench_prepare:`):

```python
    if args.bench_tune and len(families) != 1:
        raise SystemExit(
            "--bench-tune requiere exactamente una familia en --models "
            f"(recibidas {len(families)}: {', '.join(families) or '(ninguna)'})."
        )
```

Modificar la resolución de `prepared` para incluir `args.bench_tune`:

```python
    elif args.bench_train or args.bench_evaluate or args.bench_tune:
        try:
            prepared = resolve_prepared(
                prepared_dir, args.prep_id, tuple(config.bench.protocols)
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
```

Añadir, después del bloque `if run_all or args.bench_evaluate:` y antes de `if args.bench_promote:`:

```python
    if args.bench_tune:
        from swmm_resilience.ml.bench.tune import run_study, save_tune_result

        tuning_cfg = config.bench.tuning
        result = run_study(
            prepared, families[0],
            n_trials=tuning_cfg.n_trials, n_splits=tuning_cfg.n_splits,
        )
        study_path = save_tune_result(result, BENCH_ROOT / "tuning")
        print(f"Estudio Optuna guardado en {study_path}")
        print(f"status={result.status}: {result.status_reason}")
        print(
            f"Mejor valor ({result.objective_metric}): {result.best_value:.4f} "
            f"tras {result.n_trials} trials"
        )
```

Añadir `args.bench_tune` a la condición de entrada a `_run_bench()`:

```python
    if (args.bench or args.bench_prepare or args.bench_train or args.bench_evaluate
            or args.bench_promote or args.bench_tune):
        _run_bench(args, config)
```

- [ ] **Paso 4: ejecutar y comprobar que pasa, junto al resto del CLI del banco.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench/test_bench_cli.py -q --basetemp C:\Windows\Temp\pytest-optuna-task8-green -p no:cacheprovider`

Esperado: todos los tests de `test_bench_cli.py` en verde, incluidos los preexistentes (25 antes de esta tarea, más los nuevos de tuning).

---

### Task 9: Verificación de cierre

**Files:**
- No crear archivos de producción adicionales.
- Modificar solo pruebas/documentación si una prueba existente contradice la semántica nueva.

**Consume:** las ocho tareas anteriores.

**Produce:** evidencia repetible de que `--bench-tune` es aditivo (no cambia el comportamiento de `--bench`), de que Optuna nunca se ejecuta implícitamente, y de que la suite completa sigue en verde.

- [ ] **Paso 1: ejecutar toda la batería del banco.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests/ml/bench -q --basetemp C:\Windows\Temp\pytest-optuna-task9-bench -p no:cacheprovider`

Esperado: salida con código 0, sin regresiones sobre los 258 tests que ya pasaban antes de este plan (más los nuevos de `test_tune.py` y los añadidos a `test_bench_cli.py`/`test_bench_config.py`).

- [ ] **Paso 2: ejecutar la suite completa del repositorio.**

Ejecutar: `venv\Scripts\python.exe -m pytest tests -q --basetemp C:\Windows\Temp\pytest-optuna-task9-full -p no:cacheprovider`

Esperado: código 0, sin regresiones sobre los 881 tests previos.

- [ ] **Paso 3: smoke manual de `--bench-tune` contra la base real, con presupuesto reducido.**

Ejecutar (usa el `PreparedDataset` ya preparado en fase A; no vuelve a llamar `--bench-prepare`):

```bash
venv\Scripts\python.exe main.py --bench-tune --models xgboost
```

Con `config.yaml` en su `bench.tuning` por defecto (`n_trials: 50`), esta corrida puede tardar; si el piloto de tiempos de fase D aún no existe, ejecutar antes una variante acotada para verificar que el camino funciona sin comprometer minutos de cómputo:

```bash
venv\Scripts\python.exe -c "
from swmm_resilience.ml.bench.preprocess import resolve_prepared
from swmm_resilience.ml.bench.tune import run_study, save_tune_result
prepared = resolve_prepared('outputs/bench/prepared', None, ('LeaveOneShapeOut','LOSO','GroupKFold5'))
result = run_study(prepared, 'xgboost', n_trials=3, n_splits=5)
path = save_tune_result(result, 'outputs/bench/tuning')
print(path, result.status, result.best_value)
"
```

Esperado: imprime la ruta del `study.json`, `status=estimacion_exploratoria_condicionada` y un `best_value` finito. Verificar después:

```bash
git status --short outputs/models  # debe no reportar nada (directorio sin cambios)
```

- [ ] **Paso 4: revisar el diff y el estado de Git.**

Ejecutar: `git diff --check` y `git status --short`.

Esperado: sin errores de espacios; las modificaciones corresponden solo al mapa de archivos de este plan más lo pendiente de fase A. No hacer commit — el usuario controla la integración en esta rama.

## Autorrevisión del plan

- **Cobertura del spec:** §4.1 (helper compartido) → Task 3; §4.2 (folds internos) → Task 4; §4.3/§4.4 (objetivo, `TuneResult`, `run_study`) → Task 6; §5 (persistencia) → Task 6; §6 (espacios de búsqueda) → Task 5; §7 (CLI/config) → Tasks 7–8; §8 (dependencia) → Task 1; §9 (pruebas) → todas las tareas llevan su propio TDD; §10 (fuera de alcance) no genera tareas, por diseño.
- **Consistencia:** `run_study`, `TuneResult`, `save_tune_result`, `build_tuning_folds`, `SEARCH_SPACES`, `_sample_params` y `_fit_predict_fold` se usan con la misma firma en todas las tareas que los consumen.
- **Sin placeholders:** cada paso trae el código completo a escribir; los valores de rangos, defaults y mensajes de error están definidos, no descritos.
