# Búsqueda de hiperparámetros con Optuna Implementation Plan

> Estado: **plan pendiente; no implementado**. Antes de ejecutarlo, actualizar
> las tareas al contrato activo de 17 features y al resultado de la limpieza de
> la pipeline legacy. La presencia de este archivo no habilita Optuna.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar los hiperparámetros fijos de XGBoost en `config.yaml` (clasificador y regresor de la pipeline activa) por valores encontrados con Optuna, aplicándolos automáticamente si igualan o mejoran el LOSO actual.

**Architecture:** Un módulo nuevo, `swmm_resilience/ml/hyperparam_search.py`, reutiliza `trainer.make_classifier`/`make_regressor` y la lógica de pooling de `evaluator.py` (mismo dataset, mismas columnas, mismo agrupamiento por `factor_mult`). Dos estudios de Optuna independientes (clasificador, regresor), cada uno con `GroupKFold(5)` como proxy de búsqueda y pruning por fold, y un reporte final con `LeaveOneGroupOut` que decide si el resultado se escribe de vuelta a `config.yaml`.

**Tech Stack:** Python, Optuna (TPESampler + MedianPruner), scikit-learn (GroupKFold, LeaveOneGroupOut), pandas, PyYAML.

## Global Constraints

- Solo se optimizan los 4 hiperparámetros existentes de `ClassifierConfig`/`RegressorConfig` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`) — no se agregan campos nuevos al esquema.
- Algoritmo fijo en el que ya esté configurado en `config.yaml` (`xgboost`) — no se optimiza `random_forest` en este trabajo.
- Dos estudios independientes (clasificador, regresor) — no un estudio conjunto end-to-end.
- Búsqueda (cada trial): `GroupKFold(5)` agrupado por `factor_mult`. Reporte final (una vez, con los mejores hiperparámetros): `LeaveOneGroupOut` agrupado por `factor_mult`.
- Aplicación automática a `config.yaml` con backup con timestamp y salvaguarda de no-regresión (no sobrescribe si el nuevo score es peor que el actual, evaluado con los mismos folds).
- No se toca `swmm_resilience/ml/train.py` ni el resto de "Pipeline A" (legacy, ver spec).
- No se paraleliza la ejecución de trials (`study.optimize` corre secuencial).
- Nueva dependencia: `optuna` en `requirements.txt`.
- Todo el trabajo en la branch `feature/optuna-hyperparam-search`.

Spec completo: `docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`.

---

## Task 1: Helpers de datos y de configuración por trial

**Files:**
- Create: `swmm_resilience/ml/hyperparam_search.py`
- Test: `tests/ml/test_hyperparam_search.py`

**Interfaces:**
- Produces:
  - `_prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` — retorna `(X, y_clf, y_reg, groups)` desde las columnas `FEATURE_COLS`, `inunda`, `vol_inundacion_m3`, `factor_mult`.
  - `_with_updated_ml_component(config, component: str, params: dict)` — copia superficial de `config` con `config.ml.<component>` actualizado con `params`. Funciona tanto con el `Config` real (`swmm_resilience/config.py`) como con el doble de prueba `SimpleNamespace` de `tests/conftest.py::tiny_config_factory`, porque solo usa `copy.copy` + `setattr` (no `dataclasses.replace`).
  - `_classifier_config_from_params(config, params: dict)` / `_regressor_config_from_params(config, params: dict)`.
  - `_fold_score_classifier(config, X_tr, yc_tr, X_te, yc_te) -> float` — F1.
  - `_fold_score_regressor(config, X_tr, yc_tr, yr_tr, X_te, yc_te, yr_te) -> float | None` — RMSE en espacio original sobre nodos inundados; `None` si no hay nodos inundados en train o test de ese fold.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_hyperparam_search.py
import copy

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.hyperparam_search import (
    _classifier_config_from_params,
    _fold_score_classifier,
    _fold_score_regressor,
    _prepare_arrays,
    _regressor_config_from_params,
)
from swmm_resilience.ml.trainer import FEATURE_COLS


def hyperparam_search_df() -> pd.DataFrame:
    rows = []
    for factor in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for node_idx in range(4):
            row = {col: float(node_idx + 1) for col in FEATURE_COLS}
            row["factor_mult"] = factor
            row["q_pico_nodo"] = factor * (node_idx + 1)
            row["q_pico_acum_escalado"] = factor * (node_idx + 2)
            row["inunda"] = 1 if factor >= 2.0 and node_idx >= 2 else 0
            row["vol_inundacion_m3"] = factor * 10.0 if row["inunda"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_arrays_shapes():
    df = hyperparam_search_df()
    X, y_clf, y_reg, groups = _prepare_arrays(df)

    assert X.shape == (len(df), len(FEATURE_COLS))
    assert y_clf.shape == (len(df),)
    assert y_reg.shape == (len(df),)
    assert groups.shape == (len(df),)
    assert set(np.unique(groups)) == {1.0, 1.5, 2.0, 2.5, 3.0}


def test_config_component_override_does_not_mutate_original(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    original_n_estimators = config.ml.classifier.n_estimators

    updated = _classifier_config_from_params(config, {"n_estimators": 999, "max_depth": 7})

    assert updated.ml.classifier.n_estimators == 999
    assert updated.ml.classifier.max_depth == 7
    assert config.ml.classifier.n_estimators == original_n_estimators
    assert updated.ml.regressor is config.ml.regressor  # solo se copia el componente tocado


def test_fold_score_classifier_returns_f1_in_unit_range(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    df = hyperparam_search_df()
    X, y_clf, _, groups = _prepare_arrays(df)
    train_mask = groups != 2.0
    test_mask = groups == 2.0

    score = _fold_score_classifier(config, X[train_mask], y_clf[train_mask], X[test_mask], y_clf[test_mask])

    assert 0.0 <= score <= 1.0


def test_fold_score_regressor_returns_none_when_no_flooded_test_rows(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    df = hyperparam_search_df()
    X, y_clf, y_reg, groups = _prepare_arrays(df)
    train_mask = groups != 1.0
    test_mask = groups == 1.0  # factor 1.0 no tiene nodos inundados

    score = _fold_score_regressor(
        config, X[train_mask], y_clf[train_mask], y_reg[train_mask],
        X[test_mask], y_clf[test_mask], y_reg[test_mask],
    )

    assert score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swmm_resilience.ml.hyperparam_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/hyperparam_search.py
"""Búsqueda de hiperparámetros de XGBoost (pipeline activa) con Optuna."""

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ..config import ML_RANDOM_STATE, load_config
from .evaluator import _mean_metrics, _pooled_regressor_oracle_metrics
from .trainer import FEATURE_COLS, make_classifier, make_regressor


def _prepare_arrays(df: pd.DataFrame):
    """Return (X, y_clf, y_reg, groups) matching evaluator._run_cv's contract."""
    X = df[FEATURE_COLS].values
    y_clf = df["inunda"].values
    y_reg = df["vol_inundacion_m3"].values
    groups = df["factor_mult"].values
    return X, y_clf, y_reg, groups


def _with_updated_ml_component(config, component: str, params: dict):
    """Shallow-copy config with config.ml.<component> updated with params.

    Uses copy.copy + setattr (not dataclasses.replace) so it works with both
    the real Config dataclass and SimpleNamespace-based test doubles.
    """
    new_component = copy.copy(getattr(config.ml, component))
    for key, value in params.items():
        setattr(new_component, key, value)
    new_ml = copy.copy(config.ml)
    setattr(new_ml, component, new_component)
    new_config = copy.copy(config)
    setattr(new_config, "ml", new_ml)
    return new_config


def _classifier_config_from_params(config, params: dict):
    return _with_updated_ml_component(config, "classifier", params)


def _regressor_config_from_params(config, params: dict):
    return _with_updated_ml_component(config, "regressor", params)


def _fold_score_classifier(config, X_tr, yc_tr, X_te, yc_te) -> float:
    n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    clf = make_classifier(config, spw)
    clf.fit(X_tr, yc_tr)
    yc_pred = clf.predict(X_te)
    return float(f1_score(yc_te, yc_pred, zero_division=0))


def _fold_score_regressor(config, X_tr, yc_tr, yr_tr, X_te, yc_te, yr_te):
    flooded_tr = yc_tr == 1
    flooded_te = yc_te == 1
    if flooded_tr.sum() == 0 or flooded_te.sum() == 0:
        return None
    reg = make_regressor(config)
    reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))
    yr_pred = np.expm1(reg.predict(X_te[flooded_te]))
    yr_pred = np.clip(yr_pred, a_min=0.0, a_max=None)
    return float(np.sqrt(mean_squared_error(yr_te[flooded_te], yr_pred)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/hyperparam_search.py tests/ml/test_hyperparam_search.py
git commit -m "feat: add data/config helpers for hyperparameter search"
```

---

## Task 2: `build_objective` y `run_study` (búsqueda con pruning)

**Files:**
- Modify: `swmm_resilience/ml/hyperparam_search.py`
- Modify: `requirements.txt`
- Test: `tests/ml/test_hyperparam_search.py`

**Interfaces:**
- Consumes: `_prepare_arrays`, `_classifier_config_from_params`, `_regressor_config_from_params`, `_fold_score_classifier`, `_fold_score_regressor` (Task 1).
- Produces:
  - `build_objective(task: str, df: pd.DataFrame, config) -> Callable[[optuna.Trial], float]`. `task` es `"classifier"` o `"regressor"`; lanza `ValueError` en cualquier otro valor.
  - `run_study(task: str, df: pd.DataFrame, config, timeout_sec: int, seed: int = ML_RANDOM_STATE) -> optuna.Study`.

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/ml/test_hyperparam_search.py
import optuna

from swmm_resilience.ml.hyperparam_search import build_objective, run_study


def test_build_objective_rejects_unknown_task(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    with pytest.raises(ValueError):
        build_objective("unknown", hyperparam_search_df(), config)


def test_build_objective_classifier_runs_n_trials(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    objective = build_objective("classifier", hyperparam_search_df(), config)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert 0.0 <= study.best_value <= 1.0
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample"):
        assert key in study.best_params


def test_build_objective_regressor_minimizes_rmse(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    objective = build_objective("regressor", hyperparam_search_df(), config)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert study.best_value >= 0.0


def test_run_study_uses_timeout_and_returns_all_expected_params(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    study = run_study("classifier", hyperparam_search_df(), config, timeout_sec=3, seed=0)

    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert len(study.trials) >= 1
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample"):
        assert key in study.best_params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install optuna` (agregar `optuna>=3.0` a `requirements.txt` primero), luego
`pytest tests/ml/test_hyperparam_search.py -v -k "objective or run_study"`
Expected: FAIL with `ImportError: cannot import name 'build_objective'`

- [ ] **Step 3: Write minimal implementation**

En `requirements.txt`, agregar una línea después de `xgboost>=1.7.0`:

```diff
 xgboost>=1.7.0
+optuna>=3.0
 swmm-api>=0.4
```

```python
# agregar a swmm_resilience/ml/hyperparam_search.py

def build_objective(task: str, df: pd.DataFrame, config) -> Callable[[optuna.Trial], float]:
    if task not in {"classifier", "regressor"}:
        raise ValueError(f"task debe ser 'classifier' o 'regressor', recibido: {task!r}")

    config_from_params = _classifier_config_from_params if task == "classifier" else _regressor_config_from_params

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        trial_config = config_from_params(config, params)

        X, y_clf, y_reg, groups = _prepare_arrays(df)
        cv = GroupKFold(n_splits=5)
        fold_scores = []
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y_clf, groups)):
            X_tr, X_te = X[train_idx], X[test_idx]
            yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
            if task == "classifier":
                score = _fold_score_classifier(trial_config, X_tr, yc_tr, X_te, yc_te)
            else:
                yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]
                score = _fold_score_regressor(trial_config, X_tr, yc_tr, yr_tr, X_te, yc_te, yr_te)
                if score is None:
                    continue

            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_scores:
            raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    return objective


def run_study(task: str, df: pd.DataFrame, config, timeout_sec: int, seed: int = ML_RANDOM_STATE) -> optuna.Study:
    direction = "maximize" if task == "classifier" else "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(build_objective(task, df, config), timeout=timeout_sec)
    return study
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: PASS (todos los tests, incluidos los de Task 1)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/hyperparam_search.py requirements.txt tests/ml/test_hyperparam_search.py
git commit -m "feat: add Optuna objective and study runner with pruning"
```

---

## Task 3: `evaluate_best_params_loso` (reporte final con LOSO)

**Files:**
- Modify: `swmm_resilience/ml/hyperparam_search.py`
- Test: `tests/ml/test_hyperparam_search.py`

**Interfaces:**
- Consumes: `_prepare_arrays`, `_classifier_config_from_params`, `_regressor_config_from_params` (Task 1); `_mean_metrics`, `_pooled_regressor_oracle_metrics` de `swmm_resilience.ml.evaluator` (ya existen, sin cambios).
- Produces: `evaluate_best_params_loso(task: str, params: dict, df: pd.DataFrame, config) -> dict`.
  - `task="classifier"` → `{"precision", "recall", "f1", "auc_roc"}` (mismas claves que el nivel `classifier` de `evaluator._run_cv`).
  - `task="regressor"` → `{"nse", "log_nse", "rmse", "mae", "r2"}` (mismas claves que el nivel `regressor_oracle` de `evaluator._run_cv`).
  - Lanza `ValueError` si ningún fold de LOSO produjo datos válidos para esa tarea.

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/ml/test_hyperparam_search.py
from swmm_resilience.ml.hyperparam_search import evaluate_best_params_loso

DEFAULT_PARAMS = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0}


def test_evaluate_best_params_loso_classifier_keys(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    metrics = evaluate_best_params_loso("classifier", DEFAULT_PARAMS, hyperparam_search_df(), config)

    assert set(metrics.keys()) == {"precision", "recall", "f1", "auc_roc"}
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_best_params_loso_regressor_keys(tiny_config_factory):
    config = tiny_config_factory(algorithm="random_forest")
    metrics = evaluate_best_params_loso("regressor", DEFAULT_PARAMS, hyperparam_search_df(), config)

    assert set(metrics.keys()) == {"nse", "log_nse", "rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_hyperparam_search.py -v -k evaluate_best_params_loso`
Expected: FAIL with `ImportError: cannot import name 'evaluate_best_params_loso'`

- [ ] **Step 3: Write minimal implementation**

```python
# agregar a swmm_resilience/ml/hyperparam_search.py

def evaluate_best_params_loso(task: str, params: dict, df: pd.DataFrame, config) -> dict:
    if task not in {"classifier", "regressor"}:
        raise ValueError(f"task debe ser 'classifier' o 'regressor', recibido: {task!r}")

    config_from_params = _classifier_config_from_params if task == "classifier" else _regressor_config_from_params
    trial_config = config_from_params(config, params)
    X, y_clf, y_reg, groups = _prepare_arrays(df)
    cv = LeaveOneGroupOut()

    if task == "classifier":
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y_clf, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
            n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0
            clf = make_classifier(trial_config, spw)
            clf.fit(X_tr, yc_tr)
            yc_pred = clf.predict(X_te)
            yc_prob = clf.predict_proba(X_te)[:, 1]
            has_both_classes = yc_te.sum() > 0 and (1 - yc_te).sum() > 0
            fold_metrics.append({
                "precision": float(precision_score(yc_te, yc_pred, zero_division=0)),
                "recall": float(recall_score(yc_te, yc_pred, zero_division=0)),
                "f1": float(f1_score(yc_te, yc_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(yc_te, yc_prob)) if has_both_classes else float("nan"),
            })
        if not fold_metrics:
            raise ValueError("LOSO no produjo folds validos para task='classifier'.")
        return _mean_metrics(fold_metrics)

    reg_true_parts, reg_pred_parts = [], []
    for train_idx, test_idx in cv.split(X, y_clf, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
        yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]
        flooded_tr = yc_tr == 1
        flooded_te = yc_te == 1
        if flooded_tr.sum() == 0 or flooded_te.sum() == 0:
            continue
        reg = make_regressor(trial_config)
        reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))
        yr_pred = np.clip(np.expm1(reg.predict(X_te[flooded_te])), a_min=0.0, a_max=None)
        reg_true_parts.append(yr_te[flooded_te])
        reg_pred_parts.append(yr_pred)

    if not reg_true_parts:
        raise ValueError("LOSO no produjo folds validos para task='regressor'.")
    return _pooled_regressor_oracle_metrics(reg_true_parts, reg_pred_parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: PASS (todos los tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/hyperparam_search.py tests/ml/test_hyperparam_search.py
git commit -m "feat: add LOSO evaluation for best hyperparameters"
```

---

## Task 4: Aplicación automática con salvaguarda de no-regresión

**Files:**
- Modify: `swmm_resilience/ml/hyperparam_search.py`
- Test: `tests/ml/test_hyperparam_search.py`

**Interfaces:**
- Consumes: `evaluate_best_params_loso` (Task 3).
- Produces:
  - `_current_params(task: str, config) -> dict` — lee `n_estimators`/`max_depth`/`learning_rate`/`subsample` de `config.ml.classifier` o `config.ml.regressor`.
  - `_write_ml_params(config_yaml_path: Path, task: str, params: dict) -> None` — reescribe solo `ml.classifier` o `ml.regressor` dentro del YAML, preservando el resto.
  - `apply_if_better(task: str, best_params: dict, df: pd.DataFrame, config, config_yaml_path: Path) -> dict` — retorna `{"applied": bool, "reason": str | None, "loso_<metric>_previous": float, "loso_<metric>_new": float, "backup_path": str | None}`, donde `<metric>` es `"f1"` para clasificador y `"rmse"` para regresor.

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/ml/test_hyperparam_search.py
import yaml

from swmm_resilience.ml.hyperparam_search import _current_params, _write_ml_params, apply_if_better


SAMPLE_CONFIG_YAML = """
ml:
  classifier:
    algorithm: xgboost
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    scale_pos_weight: auto
  regressor:
    algorithm: xgboost
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
  use_scaler: false
evaluation:
  methods: [LOSO, GroupKFold5]
  stratify_by_factor: true
"""


def test_current_params_reads_existing_config_values(tiny_config_factory):
    config = tiny_config_factory(algorithm="xgboost")
    params = _current_params("classifier", config)
    assert params == {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0}


def test_write_ml_params_updates_only_targeted_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_CONFIG_YAML, encoding="utf-8")

    _write_ml_params(config_path, "classifier", {"n_estimators": 999, "max_depth": 9})

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["ml"]["classifier"]["n_estimators"] == 999
    assert raw["ml"]["classifier"]["max_depth"] == 9
    assert raw["ml"]["classifier"]["learning_rate"] == 0.05  # no tocado
    assert raw["ml"]["regressor"]["n_estimators"] == 200  # no tocado
    assert raw["evaluation"]["methods"] == ["LOSO", "GroupKFold5"]  # secciones ajenas intactas


def test_apply_if_better_writes_backup_and_updates_when_score_improves(tmp_path, tiny_config_factory, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_CONFIG_YAML, encoding="utf-8")
    config = tiny_config_factory(algorithm="random_forest")

    scores = iter([{"f1": 0.5}, {"f1": 0.9}])  # previous, new
    monkeypatch.setattr(
        "swmm_resilience.ml.hyperparam_search.evaluate_best_params_loso",
        lambda task, params, df, cfg: next(scores),
    )

    result = apply_if_better("classifier", {"n_estimators": 999}, hyperparam_search_df(), config, config_path)

    assert result["applied"] is True
    assert result["reason"] is None
    assert result["loso_f1_previous"] == 0.5
    assert result["loso_f1_new"] == 0.9
    backups = list(tmp_path.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["ml"]["classifier"]["n_estimators"] == 999


def test_apply_if_better_leaves_config_untouched_when_score_worsens(tmp_path, tiny_config_factory, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SAMPLE_CONFIG_YAML, encoding="utf-8")
    config = tiny_config_factory(algorithm="random_forest")

    scores = iter([{"rmse": 1.0}, {"rmse": 5.0}])  # previous, new (peor: rmse mas alto)
    monkeypatch.setattr(
        "swmm_resilience.ml.hyperparam_search.evaluate_best_params_loso",
        lambda task, params, df, cfg: next(scores),
    )

    result = apply_if_better("regressor", {"n_estimators": 999}, hyperparam_search_df(), config, config_path)

    assert result["applied"] is False
    assert result["reason"] == "regression_detected"
    assert list(tmp_path.glob("config.yaml.bak.*")) == []
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["ml"]["regressor"]["n_estimators"] == 200  # config.yaml intacto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_hyperparam_search.py -v -k "current_params or write_ml_params or apply_if_better"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# agregar a swmm_resilience/ml/hyperparam_search.py

def _current_params(task: str, config) -> dict:
    component = config.ml.classifier if task == "classifier" else config.ml.regressor
    return {
        "n_estimators": component.n_estimators,
        "max_depth": component.max_depth,
        "learning_rate": component.learning_rate,
        "subsample": component.subsample,
    }


def _write_ml_params(config_yaml_path: Path, task: str, params: dict) -> None:
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    section = "classifier" if task == "classifier" else "regressor"
    raw["ml"][section].update(params)
    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)


def apply_if_better(task: str, best_params: dict, df: pd.DataFrame, config, config_yaml_path: Path) -> dict:
    metric_key = "f1" if task == "classifier" else "rmse"
    current_metrics = evaluate_best_params_loso(task, _current_params(task, config), df, config)
    new_metrics = evaluate_best_params_loso(task, best_params, df, config)

    if task == "classifier":
        better = new_metrics["f1"] >= current_metrics["f1"]
    else:
        better = new_metrics["rmse"] <= current_metrics["rmse"]

    backup_path = None
    if better:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = config_yaml_path.with_name(f"{config_yaml_path.name}.bak.{timestamp}")
        shutil.copy2(config_yaml_path, backup_path)
        _write_ml_params(config_yaml_path, task, best_params)

    return {
        "applied": better,
        "reason": None if better else "regression_detected",
        f"loso_{metric_key}_previous": current_metrics[metric_key],
        f"loso_{metric_key}_new": new_metrics[metric_key],
        "backup_path": str(backup_path) if backup_path is not None else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: PASS (todos los tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/hyperparam_search.py tests/ml/test_hyperparam_search.py
git commit -m "feat: auto-apply best hyperparameters to config.yaml with regression guard"
```

---

## Task 5: Orquestación `run_hyperparam_search` + reporte

**Files:**
- Modify: `swmm_resilience/ml/hyperparam_search.py`
- Test: `tests/ml/test_hyperparam_search.py`

**Interfaces:**
- Consumes: `run_study` (Task 2), `apply_if_better` (Task 4), `load_config` (`swmm_resilience.config`).
- Produces: `run_hyperparam_search(config_path: str = "config.yaml", timeout_per_study: int = 600, tasks: list[str] | None = None, dataset_csv: str | None = None) -> dict`. Escribe `outputs/metrics/hyperparam_search_report.json` e imprime una tabla en consola. `tasks=None` corre `["classifier", "regressor"]`.

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/ml/test_hyperparam_search.py
from swmm_resilience.ml.hyperparam_search import run_hyperparam_search


FULL_CONFIG_YAML_TEMPLATE = """
network:
  inp_path: "{inp_name}"
  name: "Test Network"
simulation:
  factor_min: 0.2
  factor_max: 1.0
  factor_step: 0.2
dataset:
  output_path: "dataset_final.csv"
  flood_threshold_m3: 1.0
ml:
  classifier:
    algorithm: random_forest
    n_estimators: 5
    max_depth: 2
    learning_rate: 0.1
    subsample: 1.0
    scale_pos_weight: auto
  regressor:
    algorithm: random_forest
    n_estimators: 5
    max_depth: 2
    learning_rate: 0.1
    subsample: 1.0
  use_scaler: false
evaluation:
  methods: [LOSO, GroupKFold5]
  stratify_by_factor: true
visualization:
  factors_to_plot: [1.0]
  colormap: "RdYlBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
"""


def test_run_hyperparam_search_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        FULL_CONFIG_YAML_TEMPLATE.format(inp_name=inp_path.name), encoding="utf-8"
    )

    csv_path = tmp_path / "dataset_final.csv"
    hyperparam_search_df().to_csv(csv_path, index=False)

    report = run_hyperparam_search(
        config_path=str(config_path),
        timeout_per_study=2,
        dataset_csv=str(csv_path),
    )

    assert set(report.keys()) >= {"generated_at", "dataset_csv", "dataset_rows", "classifier", "regressor"}
    assert report["dataset_rows"] == len(hyperparam_search_df())
    for task in ("classifier", "regressor"):
        assert "params" in report[task]
        assert "applied" in report[task]
        assert "n_trials" in report[task]

    report_path = tmp_path / "outputs" / "metrics" / "hyperparam_search_report.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["dataset_rows"] == len(hyperparam_search_df())

    captured = capsys.readouterr()
    assert "BUSQUEDA DE HIPERPARAMETROS" in captured.out


def test_run_hyperparam_search_raises_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        FULL_CONFIG_YAML_TEMPLATE.format(inp_name=inp_path.name), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError):
        run_hyperparam_search(config_path=str(config_path), dataset_csv=str(tmp_path / "no_existe.csv"))
```

Nota: `swmm_resilience.config.load_config` (`swmm_resilience/config.py:265-333`) valida `network.inp_path` (debe existir en disco), `simulation`, `dataset`, `ml`, `evaluation` y `visualization` completos — por eso `FULL_CONFIG_YAML_TEMPLATE` cubre las 6 secciones en vez de reusar el `SAMPLE_CONFIG_YAML` más chico de la Tarea 4 (ese solo cubre `ml`/`evaluation`, suficiente para `apply_if_better` porque esa función no llama a `load_config`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_hyperparam_search.py -v -k run_hyperparam_search`
Expected: FAIL with `ImportError: cannot import name 'run_hyperparam_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# agregar a swmm_resilience/ml/hyperparam_search.py

def _print_report_table(report: dict) -> None:
    print("\n" + "=" * 90)
    print("BUSQUEDA DE HIPERPARAMETROS (OPTUNA)")
    print("=" * 90)
    for task in ("classifier", "regressor"):
        if task not in report:
            continue
        entry = report[task]
        metric_key = "f1" if task == "classifier" else "rmse"
        print(f"\n[{task}]")
        print(f"  params            : {entry['params']}")
        print(f"  search_score      : {entry['search_score']:.4f}")
        print(f"  loso_{metric_key}_previous : {entry[f'loso_{metric_key}_previous']:.4f}")
        print(f"  loso_{metric_key}_new      : {entry[f'loso_{metric_key}_new']:.4f}")
        print(f"  n_trials/n_pruned : {entry['n_trials']}/{entry['n_pruned']}")
        applied_note = f" ({entry['reason']})" if entry["reason"] else ""
        print(f"  applied           : {entry['applied']}{applied_note}")
    print("=" * 90)


def run_hyperparam_search(
    config_path: str = "config.yaml",
    timeout_per_study: int = 600,
    tasks: list[str] | None = None,
    dataset_csv: str | None = None,
) -> dict:
    config = load_config(config_path)
    csv_path = Path(dataset_csv) if dataset_csv is not None else config.dataset.output_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {csv_path}. Corre el pipeline completo o "
            "--skip-extraction antes de --tune-hyperparams."
        )
    df = pd.read_csv(csv_path)

    tasks = tasks or ["classifier", "regressor"]
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_csv": str(csv_path),
        "dataset_rows": int(len(df)),
    }

    for task in tasks:
        study = run_study(task, df, config, timeout_sec=timeout_per_study)
        best_params = study.best_params
        apply_result = apply_if_better(task, best_params, df, config, Path(config_path))
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        report[task] = {
            "params": best_params,
            "search_score": study.best_value,
            **apply_result,
            "n_trials": len(study.trials),
            "n_pruned": n_pruned,
        }
        if apply_result["applied"]:
            config = load_config(config_path)  # siguiente tarea ve el config.yaml ya actualizado

    report_path = Path("outputs/metrics/hyperparam_search_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    _print_report_table(report)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ml/test_hyperparam_search.py -v`
Expected: PASS (todos los tests). Si `test_run_hyperparam_search_smoke` falla por secciones faltantes de `load_config`, completar `SAMPLE_CONFIG_YAML` en el test según la nota del Step 1 y volver a correr.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/hyperparam_search.py tests/ml/test_hyperparam_search.py
git commit -m "feat: add run_hyperparam_search orchestration and JSON report"
```

---

## Task 6: CLI `--tune-hyperparams` en `main.py`

**Files:**
- Modify: `main.py:11-44` (imports), `main.py:86-96` (argparse), `main.py:~97` (dispatch, justo después de `config = load_config("config.yaml")` y antes de `if args.predict:`)
- Test: `tests/test_main_tune_hyperparams_cli.py`

**Interfaces:**
- Consumes: `run_hyperparam_search` (Task 5).
- Produces: flags `--tune-hyperparams`, `--timeout-per-study` (default 600), `--only-classifier`, `--only-regressor` en el parser de `main.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main_tune_hyperparams_cli.py
import sys
from unittest.mock import MagicMock

import pytest


def test_tune_hyperparams_flag_calls_run_hyperparam_search(monkeypatch):
    import main as main_module

    mock_run = MagicMock(return_value={})
    monkeypatch.setattr(main_module, "run_hyperparam_search", mock_run)
    monkeypatch.setattr(sys, "argv", ["main.py", "--tune-hyperparams", "--timeout-per-study", "5"])

    main_module.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["timeout_per_study"] == 5
    assert kwargs["tasks"] is None


def test_tune_hyperparams_only_classifier_sets_tasks(monkeypatch):
    import main as main_module

    mock_run = MagicMock(return_value={})
    monkeypatch.setattr(main_module, "run_hyperparam_search", mock_run)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--tune-hyperparams", "--only-classifier"]
    )

    main_module.main()

    _, kwargs = mock_run.call_args
    assert kwargs["tasks"] == ["classifier"]


def test_tune_hyperparams_rejects_both_only_flags(monkeypatch, capsys):
    import main as main_module

    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "--tune-hyperparams", "--only-classifier", "--only-regressor"],
    )

    with pytest.raises(SystemExit):
        main_module.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_tune_hyperparams_cli.py -v`
Expected: FAIL with `AttributeError: module 'main' has no attribute 'run_hyperparam_search'` (o `error: unrecognized arguments: --tune-hyperparams`)

- [ ] **Step 3: Write minimal implementation**

En `main.py`, junto a los demás imports de `swmm_resilience.ml.*` (cerca de la línea 26-29):

```python
from swmm_resilience.ml.hyperparam_search import run_hyperparam_search
```

En el bloque de `argparse` (junto a los demás `parser.add_argument`, cerca de la línea 86-91, antes de `args = parser.parse_args()`):

```python
    parser.add_argument("--tune-hyperparams", action="store_true",
                        help="Buscar hiperparametros de XGBoost con Optuna y aplicarlos a config.yaml si mejoran")
    parser.add_argument("--timeout-per-study", type=int, default=600,
                        help="Segundos maximos por estudio de Optuna (default: 600)")
    parser.add_argument("--only-classifier", action="store_true",
                        help="Con --tune-hyperparams, optimizar solo el clasificador")
    parser.add_argument("--only-regressor", action="store_true",
                        help="Con --tune-hyperparams, optimizar solo el regresor")
```

Justo después de `config = load_config("config.yaml")` (línea ~97), antes del bloque `if args.predict:`:

```python
    # ── Modo: busqueda de hiperparametros con Optuna ──────────────────────────
    if args.tune_hyperparams:
        if args.only_classifier and args.only_regressor:
            parser.error("--only-classifier y --only-regressor son mutuamente excluyentes")
        tasks = None
        if args.only_classifier:
            tasks = ["classifier"]
        elif args.only_regressor:
            tasks = ["regressor"]
        run_hyperparam_search(
            config_path="config.yaml",
            timeout_per_study=args.timeout_per_study,
            tasks=tasks,
        )
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main_tune_hyperparams_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main_tune_hyperparams_cli.py
git commit -m "feat: wire --tune-hyperparams into main.py CLI"
```

---

## Task 7: Suite completa y verificación final

**Files:**
- No se crean archivos nuevos.

- [ ] **Step 1: Correr toda la suite de tests del proyecto**

Run: `pytest -v`
Expected: PASS (todos los tests existentes siguen pasando, más los 15+ nuevos de `tests/ml/test_hyperparam_search.py` y `tests/test_main_tune_hyperparams_cli.py`).

- [ ] **Step 2: Correr `--tune-hyperparams` contra el dataset real, con timeout bajo, para verificar el camino end-to-end fuera de mocks**

Run (requiere `data/training/dataset_final.csv` ya generado; si no existe, correr antes `python main.py --skip-extraction --only-ml` o el pipeline completo una vez):

```bash
python main.py --tune-hyperparams --timeout-per-study 15
```

Expected: imprime la tabla `BUSQUEDA DE HIPERPARAMETROS (OPTUNA)` con una fila `[classifier]` y una `[regressor]`, genera `outputs/metrics/hyperparam_search_report.json`, y si algún estudio mejora el LOSO actual, crea `config.yaml.bak.<timestamp>` y actualiza la sección `ml:` correspondiente en `config.yaml`.

- [ ] **Step 3: Revisar `config.yaml` y el backup manualmente**

Confirmar que si se aplicó un cambio, `config.yaml.bak.<timestamp>` contiene los valores anteriores y `config.yaml` los nuevos; si no se aplicó, confirmar que `config.yaml` no cambió (`git diff config.yaml` vacío).

- [ ] **Step 4: Commit final si Step 2 dejó cambios de datos que se quieran conservar**

```bash
git status
# Si config.yaml cambio y se quiere conservar el resultado de esta corrida real:
git add config.yaml
git commit -m "chore: apply Optuna-tuned hyperparameters from local run"
# Si fue solo una verificacion y no se quiere conservar el cambio:
git checkout -- config.yaml
rm config.yaml.bak.*
```

---

## Execution Handoff

Ver la sección "Execution Handoff" en la skill `writing-plans` para las dos opciones de ejecución (subagent-driven vs. inline). Este plan asume que se ejecuta tarea por tarea, con commit al final de cada una.
