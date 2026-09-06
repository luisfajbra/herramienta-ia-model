# Banco ML — Plan 3: retiro del legacy y documentación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retirar los dos stacks de entrenamiento tabular que el banco reemplaza, sin romper ninguno de los consumidores que sobreviven, y dejar la documentación describiendo el estado real.

**Architecture:** Dos bloques de borrado con distinto nivel de riesgo. El bloque 1 (stack A + parte ML de la GUI) está desconectado de `main.py` y se puede borrar en cuanto el banco exista. El bloque 2 (stack B) sólo se borra con el test de paridad verde. Antes de ambos, un desacople mecánico que reduce el radio de impacto a casi cero.

**Tech Stack:** Python 3.12, pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-05-ml-bench-preproc-train-eval-design.md` §12, §15

**Depende de:** Planes 1 y 2 completos. **El test de paridad (`tests/ml/bench/test_parity_with_legacy_evaluator.py`) debe estar verde antes de la Task 5.** Si no lo está, este plan se detiene ahí.

## Global Constraints

- **Nada se borra sin que la suite esté verde antes y después.** Cada task es un commit pequeño con la suite corriendo entre medias.
- **Consumidores que deben seguir funcionando sin modificarse:** `main.py --predict`, `--simulate`, `--only-maps`, `--evaluate-hydrographs`, `--analyze-features`, `--evaluate-shapes`, `--evaluate-generalization`; `ml/predict.py`, `ml/scenario_predict.py`, `ml/feature_analysis.py`, `ml/feature_importance.py`, `validation/hydrograph_batch.py`. Todos leen `outputs/models/{classifier,regressor}.joblib`, que `--bench-promote` produce en el mismo formato.
- **La GUI conserva simulación y visor de resultados.** Sólo se retira su parte ML (entrenar, predecir, mapas ML).
- `swmm_resilience/ml/temporal/` **no se toca**, ni sus tests, ni `DEFAULT_DB_FILE`, ni `DEFAULT_OUTPUT_CSV`.
- Comando de test: `./venv/Scripts/python.exe -m pytest -q`.

---

## Corrección al alcance heredado de la spec

La spec §12.2 dice que `ml/preprocessing.py` se borra porque «sólo lo usaba `train.py`». **Eso es falso**, verificado el 2026-09-05:

```
desktop/app.py  →  swmm_resilience/main.py::run_experiment      (la parte SWMM: SOBREVIVE)
                        └─> analysis/eda.py::run_dataset_review      (swmm_resilience/main.py:309)
                              ├─> ml/preprocessing.py  (get_feature_columns, dataset_info)
                              └─> ML_DROP_COLUMNS, ML_TARGET_CLASSIFICATION, ML_TARGET_REGRESSION
```

`analysis/eda.py` (877 líneas) se invoca al final de **cada corrida de simulación de la GUI** y escribe `dataset_review_tables.xlsx`. Borrar `ml/preprocessing.py` o esas tres constantes rompe esa corrida.

**Resolución adoptada en este plan (Task 3):** no se borran — se **mueven** a `analysis/`, junto a su único consumidor real. Así `ml/` queda limpio para el banco y la GUI sigue funcionando. Es la opción conservadora: no elimina ninguna salida que el usuario pueda estar usando.

**Alternativa si prefieres adelgazar más:** quitar la llamada a `run_dataset_review` de `run_experiment` y borrar `analysis/eda.py` entero (877 líneas + `ml/preprocessing.py` + las tres constantes). Eso elimina el Excel de revisión del dataset. **Es una decisión tuya, no mía** — está marcada en la Task 3 como punto de parada.

---

## Task 1: Desacoplar `FEATURE_COLS` de `trainer.py`

**Files:**
- Modify: `swmm_resilience/ml/predict.py:11`, `ml/feature_importance.py:5`, `ml/feature_analysis.py:12`, `ml/scenario_predict.py:22`, `database/csv_backfill.py:40`
- Modify: `tests/ml/test_feature_importance_labels.py`, `tests/ml/test_feature_analysis.py`, `tests/test_evaluator.py`, `tests/test_scenario_predict.py`

**Interfaces:**
- Produces: cero dependencias sobre `ml/trainer.py` fuera de `main.py` y `csv_backfill.py`.

**Por qué primero:** seis módulos importan `FEATURE_COLS` de `trainer.py`, cuando la constante real vive en `ml/contracts.py` y `trainer.py` sólo la re-exporta — con un comentario que ya anticipa esta migración: `# removed in Plan D after consumers migrate`. Es mecánico, sin cambio de comportamiento, y después borrar `trainer.py` no afecta a nadie.

- [ ] **Step 1: Confirmar el estado de partida**

Run: `./venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3`
Anota el número. **Es la línea base de todo este plan.** La última registrada (`531 passed, 3 deselected`) es de HEAD `7a5ce36`, varios commits atrás y antes de los Planes 1 y 2.

- [ ] **Step 2: Write the failing test**

```python
# tests/ml/test_no_trainer_imports.py
"""Ningún módulo debe importar FEATURE_COLS de trainer.py.

La constante vive en ml/contracts.py; trainer.py sólo la re-exporta, y va a
desaparecer en la Task 5. Este test es el que hace ese borrado seguro.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# main.py y csv_backfill.py siguen usando trainer.py hasta la Task 5.
ALLOWED = {"main.py", "csv_backfill.py", "trainer.py", "evaluator.py"}


def _modules_importing_trainer(directory: Path) -> list[str]:
    offenders = []
    for path in directory.rglob("*.py"):
        if path.name in ALLOWED or "temporal" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.endswith("trainer") or node.module.endswith("ml.trainer"):
                    offenders.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom) and node.level and node.module == "trainer":
                offenders.append(str(path.relative_to(ROOT)))
    return offenders


def test_no_package_module_imports_trainer():
    offenders = _modules_importing_trainer(ROOT / "swmm_resilience")
    assert not offenders, (
        f"Estos módulos siguen importando trainer.py: {offenders}. "
        "Importa FEATURE_COLUMNS_V17 de ml/contracts.py."
    )


def test_no_test_module_imports_trainer():
    offenders = _modules_importing_trainer(ROOT / "tests")
    assert not offenders, f"Estos tests siguen importando trainer.py: {offenders}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/test_no_trainer_imports.py -q`
Expected: FAIL listando `predict.py`, `feature_importance.py`, `feature_analysis.py`, `scenario_predict.py` y los cuatro tests.

- [ ] **Step 4: Aplicar el cambio mecánico**

En cada módulo, sustituir:

```python
from .trainer import FEATURE_COLS
```
por
```python
from .contracts import FEATURE_COLUMNS_V17

FEATURE_COLS = list(FEATURE_COLUMNS_V17)
```

Ajusta el nivel relativo según el módulo (`from ..ml.contracts import ...` en `csv_backfill.py`, `from swmm_resilience.ml.contracts import ...` en los tests). En `feature_analysis.py` la línea importa además `make_classifier, make_regressor`: **déjalos** apuntando a `trainer.py` por ahora; se resuelven en la Task 5.

En los tests, sustituir `from swmm_resilience.ml.trainer import FEATURE_COLS` por `from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17` más `FEATURE_COLS = list(FEATURE_COLUMNS_V17)`, o usar directamente la fixture `FEATURE_COLS` que `tests/conftest.py` ya define a partir del contrato.

> `feature_analysis.py` seguirá importando `trainer` por `make_classifier`/`make_regressor`, así que quedará en la lista de infractores. Añádelo temporalmente a `ALLOWED` con un comentario `# hasta la Task 5` y quítalo entonces. No relajes el test de otra forma.

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest -q`
Expected: la línea base del Step 1, más 2 tests nuevos. **Ni uno menos.**

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/ swmm_resilience/database/csv_backfill.py tests/
git commit -m "refactor: importar FEATURE_COLUMNS_V17 del contrato, no de trainer"
```

---

## Task 2: Retirar la parte ML de la GUI

**Files:**
- Modify: `swmm_resilience/desktop/app.py` (quitar la pestaña ML, `_train_models`, `_predict_with_ml`, los mapas ML y sus variables)
- Modify: `swmm_resilience/visualization/runner.py` (quitar las ramas `--source ml`, `generate_ml_map`, `_global_vmax` si queda huérfano)
- Modify: `swmm_resilience/visualization/loaders.py` (quitar `load_from_ml`, `load_all_ml`)
- Modify: `swmm_resilience/visualization/__init__.py`
- Modify: `tests/desktop/test_results_tab.py` (recortar lo que cubra la pestaña ML)

**Interfaces:**
- Produces: una GUI que sólo hace simulación SWMM y visor de resultados. `visualization/loaders.py` conserva `load_from_swmm`, `load_all_swmm_runs` y `_standardize`.

- [ ] **Step 1: Mapear la superficie exacta antes de tocar nada**

```bash
grep -n "ml_train\|artifacts_dir_var\|predict_regressor_var\|predict_classifier_var\|generate_ml_map\|predict_steady_flows" swmm_resilience/desktop/app.py
grep -n "load_from_ml\|load_all_ml\|source.*ml\|artifacts_dir" swmm_resilience/visualization/runner.py swmm_resilience/visualization/loaders.py swmm_resilience/visualization/__init__.py
```

Lista cada método, variable de Tk y widget. La pestaña ML se construye en `_build_ml_tab` (app.py:275); sus consumidores conocidos están en las líneas 129, 143-145, 349, 397, 926, 954, 1060-1122.

- [ ] **Step 2: Write the failing test**

```python
# tests/desktop/test_ml_tab_removed.py
"""La GUI conserva simulación y visor de resultados; su parte ML se retiró.

El banco (python main.py --bench-*) es el único camino de entrenamiento.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "swmm_resilience" / "desktop" / "app.py"


def _source() -> str:
    return APP.read_text(encoding="utf-8")


def _defined_functions() -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(_source()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_the_ml_tab_builder_is_gone():
    assert "_build_ml_tab" not in _defined_functions()


def test_the_training_and_prediction_handlers_are_gone():
    defined = _defined_functions()
    assert "_train_models" not in defined
    assert "_predict_with_ml" not in defined


def test_the_app_does_not_import_the_legacy_training_module():
    assert "ml import train" not in _source()
    assert "ml_train" not in _source()


def test_the_app_does_not_import_the_legacy_prediction_modules():
    source = _source()
    assert "predict_tabular" not in source
    assert "predict_from_inp" not in source
    assert "generate_ml_map" not in source


def test_the_simulation_entry_point_survives():
    """run_experiment es la parte SWMM de la GUI: NO se retira."""
    assert "run_experiment" in _source()


def test_the_results_viewer_survives():
    assert "_refresh_results_tree" in _defined_functions()
```

```python
# tests/visualization/test_ml_loaders_removed.py
from swmm_resilience.visualization import loaders


def test_ml_loaders_are_gone():
    assert not hasattr(loaders, "load_from_ml")
    assert not hasattr(loaders, "load_all_ml")


def test_swmm_loaders_survive():
    assert hasattr(loaders, "load_from_swmm")
    assert hasattr(loaders, "load_all_swmm_runs")
    assert hasattr(loaders, "_standardize")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/desktop/test_ml_tab_removed.py tests/visualization/test_ml_loaders_removed.py -q`
Expected: FAIL — todo eso todavía existe.

- [ ] **Step 4: Borrar, en este orden**

1. `visualization/loaders.py`: eliminar `load_from_ml` y `load_all_ml`; quitar `DEFAULT_MODEL_ARTIFACTS_DIR` del import de `..config` si queda sin uso.
2. `visualization/runner.py`: eliminar `generate_ml_map`, la rama `--source ml`/`both`, el argumento `--artifacts-dir` y el import de `load_all_ml`/`load_from_ml`. Deja `--source swmm` como único valor y actualiza el docstring del módulo (líneas 1-25), que hoy promete `flood_map_qx*_ml.png`.
3. `visualization/__init__.py`: quitar `load_from_ml` de las reexportaciones.
4. `desktop/app.py`: eliminar `_build_ml_tab`, `_train_models`, `_predict_with_ml`, la llamada que añade la pestaña al notebook, las variables Tk (`artifacts_dir_var`, `predict_regressor_var`, `predict_classifier_var`, `predict_source_var`, `predict_flows_var`, `artifacts_dir_entry`), el log `append_ml_log` si queda huérfano, y los imports de `ml_train`, `predict_steady_flows_from_inp`, `predict_tabular` y `generate_ml_map`.
5. `tests/desktop/test_results_tab.py`: recortar lo que cubra la pestaña ML.

> Después de cada archivo: `./venv/Scripts/python.exe -m pytest -q`. Si algo se rompe, es de ese archivo, no de los cinco.

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest -q` y, además, abrir la GUI a mano:
```bash
./venv/Scripts/python.exe -m swmm_resilience.desktop.app
```
Expected: la ventana abre, no hay pestaña ML, la pestaña de simulación y el visor de resultados funcionan.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/desktop/app.py swmm_resilience/visualization/ tests/desktop/ tests/visualization/
git commit -m "refactor: retirar la parte ML de la GUI de escritorio"
```

---

## Task 3: Mover los helpers de EDA fuera de `ml/`

**Files:**
- Create: `swmm_resilience/analysis/eda_columns.py` (recibe `get_feature_columns`, `dataset_info`, `ML_DROP_COLUMNS`, `ML_TARGET_CLASSIFICATION`, `ML_TARGET_REGRESSION`)
- Modify: `swmm_resilience/analysis/eda.py` (importar de ahí)
- Delete: `swmm_resilience/ml/preprocessing.py`
- Modify: `swmm_resilience/config.py` (quitar las tres constantes migradas)
- Delete: `tests/ml/test_preprocessing_feature_contract.py` → mover su cobertura útil a `tests/analysis/test_eda_columns.py`

**Interfaces:**
- Produces: `ml/` sin `preprocessing.py`. `analysis/eda.py` sigue funcionando igual.

### ⚠️ Punto de parada — decisión del dueño del proyecto

Este task **conserva** el informe EDA (`dataset_review_tables.xlsx`) que se genera al final de cada corrida de simulación de la GUI, moviendo sus helpers en vez de borrarlos. Es lo conservador.

**Si prefieres eliminarlo del todo** (quitar `run_dataset_review` de `swmm_resilience/main.py:309` y borrar `analysis/eda.py` completo, 877 líneas), dilo antes de ejecutar este task y se sustituye por un borrado directo. **No lo decidas por tu cuenta si estás ejecutando este plan como agente.**

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_eda_columns.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.analysis.eda_columns import (
    ML_DROP_COLUMNS,
    ML_TARGET_CLASSIFICATION,
    ML_TARGET_REGRESSION,
    dataset_info,
    get_feature_columns,
)


@pytest.fixture
def review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 2, 3],
            "node_id": ["N0", "N1", "N2"],
            "elev_fondo": [100.0, 101.0, np.nan],
            "prof_max": [1.5, 1.6, 1.7],
            "vol_inundacion_m3": [0.0, 10.0, 20.0],
            "inunda": [0, 1, 1],
        }
    )


def test_get_feature_columns_drops_identifiers_and_the_target(review_frame):
    columns = get_feature_columns(review_frame, target="vol_inundacion_m3")

    assert "elev_fondo" in columns
    assert "run_id" not in columns
    assert "node_id" not in columns
    assert "vol_inundacion_m3" not in columns


def test_get_feature_columns_keeps_only_numeric(review_frame):
    assert "node_id" not in get_feature_columns(review_frame)


def test_dataset_info_counts_rows_and_missing_values(review_frame):
    info = dataset_info(review_frame)

    assert info["total_rows"] == 3
    assert info["missing_per_column"]["elev_fondo"] == 1
    assert info["missing_total"] == 1


def test_the_target_constants_moved_with_their_consumer():
    assert ML_TARGET_CLASSIFICATION == "inunda"
    assert ML_TARGET_REGRESSION == "vol_inundacion_m3"
    assert "run_id" in ML_DROP_COLUMNS
```

```python
# tests/ml/test_preprocessing_module_removed.py
def test_the_legacy_preprocessing_module_is_gone():
    """Su lista negra la sustituye la lista blanca del contrato (bench/preprocess.py)."""
    import importlib

    try:
        importlib.import_module("swmm_resilience.ml.preprocessing")
    except ModuleNotFoundError:
        return
    raise AssertionError("ml/preprocessing.py sigue existiendo")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/analysis/test_eda_columns.py tests/ml/test_preprocessing_module_removed.py -q`
Expected: `ModuleNotFoundError: ...analysis.eda_columns`

- [ ] **Step 3: Mover el código**

Crea `swmm_resilience/analysis/eda_columns.py` con:
- las tres constantes copiadas literalmente de `config.py` (`ML_DROP_COLUMNS`, `ML_TARGET_CLASSIFICATION`, `ML_TARGET_REGRESSION`),
- `get_feature_columns` y `dataset_info` copiadas literalmente de `ml/preprocessing.py`,
- un docstring que diga qué es esto y qué no:

```python
"""Helpers de la revisión exploratoria del dataset (EDA).

Vienen de ml/preprocessing.py, que se retiró con el stack legacy. Se
conservan aquí porque analysis/eda.py los usa para generar
dataset_review_tables.xlsx al final de cada corrida de simulación.

NO son el preprocesamiento del entrenamiento. Ese vive en
ml/bench/preprocess.py y usa la lista BLANCA ordenada del contrato de 17
features, no la lista negra de abajo. La diferencia importa: una lista negra
deja entrar cualquier columna numérica nueva sin que nadie lo decida.
"""
```

`select_features_for_model` **no se mueve**: sólo la usaba `ml/train.py`, que se borra en la Task 4.

Después: actualiza el import de `analysis/eda.py:25-32`, borra `ml/preprocessing.py`, y quita las tres constantes de `config.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest -q`
Expected: línea base + los tests nuevos. Verifica también que la corrida de la GUI sigue generando el Excel.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/analysis/ swmm_resilience/config.py tests/analysis/ tests/ml/
git rm swmm_resilience/ml/preprocessing.py tests/ml/test_preprocessing_feature_contract.py
git commit -m "refactor: mover los helpers de EDA de ml/ a analysis/"
```

---

## Task 4: Borrar el stack A

**Files:**
- Delete: `swmm_resilience/ml/train.py` (1116 líneas), `ml/predict_tabular.py`, `ml/predict_from_inp.py`
- Delete: `tests/ml/test_prediction_volume_output_schema.py`
- Modify: `swmm_resilience/config.py` (quitar `ML_MODEL_CONFIGS`, `ML_USE_PCA`, `ML_PCA_COMPONENTS`, `ML_PCA_SVD_SOLVER`, `ML_TEST_SIZE`, `ML_SPLIT_STRATEGY`, `ML_CV_FOLDS`, `ML_GROUP_COLUMN`, `DEFAULT_MODEL_ARTIFACTS_DIR`)

**Interfaces:**
- Produces: `ml/` sin la comparación legacy de 7 modelos. Su función la cumple el banco.

- [ ] **Step 1: Verificar que nadie los usa**

```bash
grep -rn "ml.train\|ml import train\|predict_tabular\|predict_from_inp" --include=*.py swmm_resilience tests main.py
```
Expected: **cero resultados** (las Tasks 2 y 3 quitaron los últimos). Si aparece algo, resuélvelo antes de borrar.

```bash
for name in ML_MODEL_CONFIGS ML_USE_PCA ML_PCA_COMPONENTS ML_PCA_SVD_SOLVER ML_TEST_SIZE ML_SPLIT_STRATEGY ML_CV_FOLDS ML_GROUP_COLUMN DEFAULT_MODEL_ARTIFACTS_DIR; do
  echo "--- $name"; grep -rn "$name" --include=*.py swmm_resilience tests main.py | grep -v "config.py:"
done
```
Expected: vacío para cada uno. **Cualquiera que tenga consumidores no se borra** — anótalo y sigue con el resto.

- [ ] **Step 2: Write the failing test**

```python
# tests/ml/test_legacy_stack_removed.py
"""El stack legacy de comparación de 7 modelos se retiró.

Su función —comparar familias de modelos— la cumple ahora
swmm_resilience/ml/bench/, sobre SQL, con folds y métricas compartidas.
"""

import importlib

import pytest

REMOVED_MODULES = (
    "swmm_resilience.ml.train",
    "swmm_resilience.ml.predict_tabular",
    "swmm_resilience.ml.predict_from_inp",
)

REMOVED_CONSTANTS = (
    "ML_MODEL_CONFIGS", "ML_USE_PCA", "ML_PCA_COMPONENTS", "ML_PCA_SVD_SOLVER",
    "ML_TEST_SIZE", "ML_SPLIT_STRATEGY", "ML_CV_FOLDS", "ML_GROUP_COLUMN",
    "DEFAULT_MODEL_ARTIFACTS_DIR",
)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_legacy_module_is_gone(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("name", REMOVED_CONSTANTS)
def test_legacy_constant_is_gone(name):
    from swmm_resilience import config

    assert not hasattr(config, name), f"config.{name} sigue definido"


def test_the_bench_replaces_them():
    from swmm_resilience.ml.bench.registry import available_families

    assert len(available_families()) >= 4


def test_constants_the_temporal_stack_needs_survive():
    """ml/temporal/ no se toca en este plan."""
    from swmm_resilience import config

    assert hasattr(config, "DEFAULT_DB_FILE")
    assert hasattr(config, "DEFAULT_OUTPUT_CSV")
    assert hasattr(config, "DEFAULT_TEMPORAL_ARTIFACTS_DIR")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/test_legacy_stack_removed.py -q`
Expected: FAIL — los módulos y constantes siguen ahí.

- [ ] **Step 4: Borrar**

```bash
git rm swmm_resilience/ml/train.py swmm_resilience/ml/predict_tabular.py swmm_resilience/ml/predict_from_inp.py tests/ml/test_prediction_volume_output_schema.py
```

Y quitar de `config.py` sólo las constantes que el Step 1 confirmó sin consumidores.

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest -q`
Expected: línea base menos los tests borrados, más los nuevos. Comprueba también que `./venv/Scripts/python.exe -m swmm_resilience.desktop.app` sigue abriendo.

- [ ] **Step 6: Commit**

```bash
git add -A swmm_resilience/config.py tests/ml/
git commit -m "refactor: borrar el stack legacy de comparacion de 7 modelos"
```

---

## Task 5: Borrar el stack B — **requiere paridad verde**

**Files:**
- Delete: `swmm_resilience/ml/trainer.py`, `swmm_resilience/ml/evaluator.py`, `tests/test_evaluator.py`, `tests/ml/bench/test_parity_with_legacy_evaluator.py`
- Modify: `main.py` (quitar los imports y las llamadas a `train_models`/`evaluate_models`; `--only-ml` y `--skip-extraction` pasan a delegar en el banco)
- Modify: `swmm_resilience/database/csv_backfill.py` (retirar `persist_training_run` y su import de `trainer`)
- Modify: `swmm_resilience/ml/feature_analysis.py` (usar el registro del banco en vez de `make_classifier`/`make_regressor`)
- Modify: `tests/ml/test_no_trainer_imports.py` (vaciar `ALLOWED`)

**Interfaces:**
- Produces: un solo camino de entrenamiento en todo el repo.

- [ ] **Step 1: PUERTA — verificar la paridad**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_parity_with_legacy_evaluator.py -q`
Expected: **2 passed.**

> **Si falla, este task se detiene aquí.** No se borra nada. No se sube la tolerancia. Hay una diferencia real entre el banco y `evaluator.py` que no identificamos, y borrar la referencia haría imposible diagnosticarla. Reporta el fallo y espera instrucciones.

- [ ] **Step 2: Write the failing test**

```python
# tests/ml/test_stack_b_removed.py
"""El stack de cascada XGBoost (trainer.py + evaluator.py) se retiró.

Se borró sólo después de que el test de paridad demostrara que el candidato
xgboost del banco reproduce sus métricas en los tres niveles, en LOSO y en
GroupKFold5, con rtol=1e-6.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name", ("swmm_resilience.ml.trainer", "swmm_resilience.ml.evaluator")
)
def test_stack_b_module_is_gone(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_persist_training_run_is_gone():
    """Su trabajo lo hace ahora bench/persist/, con la cadena completa."""
    from swmm_resilience.database import csv_backfill

    assert not hasattr(csv_backfill, "persist_training_run")


def test_the_data_backfill_survives():
    """--persist-sql conserva el volcado de datos; pierde el entrenamiento."""
    from swmm_resilience.database import csv_backfill

    assert hasattr(csv_backfill, "backfill_networks_and_runs")


def test_feature_analysis_uses_the_bench_registry():
    import inspect

    from swmm_resilience.ml import feature_analysis

    source = inspect.getsource(feature_analysis)
    assert "trainer" not in source
    assert "bench" in source


def test_the_bench_is_the_only_training_path():
    from swmm_resilience.ml.bench.train import train_candidate

    assert callable(train_candidate)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/test_stack_b_removed.py -q`
Expected: FAIL.

- [ ] **Step 4: Migrar los tres consumidores y borrar**

1. **`ml/feature_analysis.py:12`** — sustituir `from .trainer import FEATURE_COLS, make_classifier, make_regressor` por:
   ```python
   from .bench.registry import get_family
   from .contracts import FEATURE_COLUMNS_V17

   FEATURE_COLS = list(FEATURE_COLUMNS_V17)
   _FAMILY = get_family("xgboost")
   ```
   y reemplazar cada `make_classifier(config, spw)` por `_FAMILY.build_classifier(params, spw)` y `make_regressor(config)` por `_FAMILY.build_regressor(params)`, donde `params` sale de `config.bench.families["xgboost"]`. Lee el archivo entero (304 líneas) antes de editar: la ablación de features itera sobre subconjuntos y hay que respetar cómo pasa las columnas.

2. **`database/csv_backfill.py`** — eliminar `persist_training_run` (líneas ~218-450) y su import de `trainer`. Conservar `backfill_networks_and_runs`.

3. **`main.py`** — quitar `from swmm_resilience.ml.evaluator import evaluate_models` y `from swmm_resilience.ml.trainer import train_models`; en la rama de `--only-ml`/`--skip-extraction`, sustituir las llamadas por `_run_bench`; en `--persist-sql`, quitar la llamada a `persist_training_run` y ajustar el mensaje que hoy explica que se persisten modelos.

4. Borrar:
   ```bash
   git rm swmm_resilience/ml/trainer.py swmm_resilience/ml/evaluator.py tests/test_evaluator.py tests/ml/bench/test_parity_with_legacy_evaluator.py
   ```
   El test de paridad se va **en este mismo commit**: era desechable por diseño, su trabajo era autorizar este borrado.

5. Vaciar `ALLOWED` en `tests/ml/test_no_trainer_imports.py`.

- [ ] **Step 5: Run test to verify it passes**

```bash
./venv/Scripts/python.exe -m pytest -q
./venv/Scripts/python.exe main.py --bench
./venv/Scripts/python.exe main.py --predict --factor 2.0
./venv/Scripts/python.exe main.py --only-maps
./venv/Scripts/python.exe main.py --analyze-features
```
Expected: suite verde y los cuatro comandos funcionando. **`--predict` es el que demuestra que el contrato de artefactos se respetó.**

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: borrar trainer.py y evaluator.py; el banco es el unico camino"
```

---

## Task 6: Documentación

**Files:**
- Create: `docs/ML_BENCH.md`
- Modify: `docs/FLUJO_ACTUAL.md` (§6, §7, §9, §10, §11, tabla de §12.4)
- Modify: `COMANDOS.md`
- Modify: `MODEL_COMPARISON_AND_PREPROCESSING_HISTORY.md`, `XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md` (nota de histórico)
- Modify: `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md` (apuntar a la spec nueva)

**Interfaces:**
- Produces: documentación que describe el estado real del repo.

- [ ] **Step 1: Write the failing test**

```python
# tests/docs/test_documentation_is_current.py
"""La documentación no puede prometer módulos que ya no existen."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

RETIRED = ("ml/train.py", "ml/trainer.py", "ml/evaluator.py", "ml/preprocessing.py")
LIVE_DOCS = (ROOT / "docs" / "FLUJO_ACTUAL.md", ROOT / "COMANDOS.md")


@pytest.mark.parametrize("doc", LIVE_DOCS, ids=lambda p: p.name)
def test_live_docs_do_not_reference_retired_modules(doc):
    text = doc.read_text(encoding="utf-8")
    offenders = [name for name in RETIRED if name in text]
    assert not offenders, f"{doc.name} menciona módulos retirados: {offenders}"


def test_the_bench_guide_exists_and_covers_the_three_stages():
    guide = ROOT / "docs" / "ML_BENCH.md"
    assert guide.exists()

    text = guide.read_text(encoding="utf-8")
    for section in ("preprocesamiento", "entrenamiento", "evaluación"):
        assert section in text.lower(), f"falta la sección de {section}"


def test_the_bench_guide_explains_how_to_add_a_family():
    text = (ROOT / "docs" / "ML_BENCH.md").read_text(encoding="utf-8")
    assert "bench/models/" in text
    assert "build_classifier" in text


def test_the_bench_guide_documents_the_temporal_hook():
    """La puerta abierta del LSTM/CNN tiene que estar escrita, o no sirve."""
    text = (ROOT / "docs" / "ML_BENCH.md").read_text(encoding="utf-8")
    assert "score_predictions" in text
    assert "folds.parquet" in text


def test_the_bench_guide_documents_the_anti_leakage_guarantee():
    text = (ROOT / "docs" / "ML_BENCH.md").read_text(encoding="utf-8")
    assert "fuga" in text.lower()


def test_commands_doc_lists_the_bench_commands():
    text = (ROOT / "COMANDOS.md").read_text(encoding="utf-8")
    for flag in ("--bench-prepare", "--bench-train", "--bench-evaluate", "--bench-promote"):
        assert flag in text, f"falta {flag}"


@pytest.mark.parametrize(
    "doc", ("MODEL_COMPARISON_AND_PREPROCESSING_HISTORY.md", "XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md")
)
def test_historical_docs_are_marked_as_such(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "histórico" in text.lower() or "historico" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/docs/test_documentation_is_current.py -q`
Expected: FAIL — `docs/ML_BENCH.md` no existe.

- [ ] **Step 3: Escribir la documentación**

`docs/ML_BENCH.md` debe cubrir, con este orden:

1. **Qué es el banco y por qué existe** — un párrafo: comparar familias de modelos bajo condiciones idénticas, con evidencia trazable.
2. **Las tres etapas**, una sección cada una: qué hace, qué recibe, qué produce, dónde escribe.
3. **La garantía anti-fuga y dónde vive** — por qué la imputación y el escalado están dentro del `Pipeline` y no en la etapa 1; qué verifica `tests/ml/bench/test_stage1_imports.py`; por qué `folds.py` sí puede importar `sklearn.model_selection`.
4. **El informe de calidad** — qué campos trae `quality_report.json` y cuáles son los dos errores duros.
5. **Cómo añadir una familia** — el archivo de ~40 líneas en `bench/models/`, la interfaz que debe exponer, el registro, y la entrada en `config.yaml` **con su justificación escrita**.
6. **Cómo enganchar un modelo temporal (LSTM/CNN)** — los tres contratos: `keys` para unir por `(run_id, node_id)`, `folds.parquet` para entrenar sobre los mismos folds, y `score_predictions()` con el `prep_id` verificado. Deja claro que **no hay código temporal todavía** y que `node_timeseries` está vacía: llenarla es el bloqueador real y tendrá su propia spec.
7. **Cómo se conectará Optuna** — `evaluate_candidate` como función pura de `params`, y los folds internos para no sesgar la selección.
8. **La cadena de proveniencia en SQL** — el diagrama de tablas y qué garantiza cada finalización.
9. **Las métricas** — los tres niveles, y **explícitamente** la asimetría de agregación (clasificador promediado, regresor sobre el pool), porque cambia los números y hoy nadie lo sabe.

En `docs/FLUJO_ACTUAL.md`: reescribir §6 y §7 (el entrenamiento ya no es `trainer.py`), actualizar la tabla de §9 (desaparece la fila «pipeline legacy de 7 modelos → `model_artifacts/`, `swmm_resilience.db`, sólo GUI de escritorio»), añadir los comandos `--bench-*` a §10, redibujar §11, y actualizar la tabla de §12.4.

En `COMANDOS.md`: los cinco comandos del banco con un ejemplo de salida real.

En los dos documentos históricos, una nota al inicio:

```markdown
> **HISTÓRICO (2026-09-05).** Este documento describe el pipeline legacy de
> comparación de modelos (`ml/train.py`), retirado el 2026-09-05. Se conserva
> porque sus resultados están citados en la tesis. El pipeline vigente es el
> banco multi-modelo: ver `docs/ML_BENCH.md`.
```

En `2026-08-21-sqlite-v17-pipeline-consolidation-design.md`, cambiar la nota de *superseded* para que apunte a la spec del 2026-09-05 como su continuación aprobada, en vez de decir «no reanudar sin re-confirmar».

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/docs/ -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add docs/ COMANDOS.md MODEL_COMPARISON_AND_PREPROCESSING_HISTORY.md XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md tests/docs/
git commit -m "docs: guia del banco ML y actualizacion del flujo real"
```

---

## Verificación final del plan

- [ ] `./venv/Scripts/python.exe -m pytest -q` verde.
- [ ] `pytest -m scale` verde.
- [ ] Los cinco comandos del banco funcionan de punta a punta.
- [ ] `--predict`, `--simulate`, `--only-maps`, `--evaluate-hydrographs`, `--analyze-features`, `--evaluate-shapes`, `--evaluate-generalization` funcionan **sin haber sido modificados** (salvo el cambio de import de la Task 1).
- [ ] La GUI abre, simula y muestra resultados; no tiene pestaña ML.
- [ ] `grep -rn "ml.train\|ml.trainer\|ml.evaluator\|ml.preprocessing" --include=*.py swmm_resilience tests main.py` no devuelve nada.
- [ ] `swmm_resilience/ml/temporal/` intacto: `git log --oneline -- swmm_resilience/ml/temporal/` sin commits nuevos.
- [ ] `docs/ML_BENCH.md` escrito; `FLUJO_ACTUAL.md` y `COMANDOS.md` sin referencias a módulos retirados.

## Balance del retiro

| Archivo | Líneas | Destino |
|---|---|---|
| `ml/train.py` | 1116 | borrado |
| `ml/predict_tabular.py` | 226 | borrado |
| `ml/predict_from_inp.py` | 128 | borrado |
| `ml/preprocessing.py` | 110 | 2 funciones a `analysis/eda_columns.py`, resto borrado |
| `ml/trainer.py` | ~90 | borrado |
| `ml/evaluator.py` | 163 | borrado |
| `desktop/app.py` | 1314 | recortado (pestaña ML) |
| `database/csv_backfill.py` | 451 | recortado (`persist_training_run`) |
| `visualization/runner.py`, `loaders.py` | 488 | recortados (ramas ML) |

Aproximadamente **1800 líneas borradas** y unas 600 recortadas, a cambio de un solo camino de entrenamiento con evidencia trazable.
