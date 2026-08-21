# Búsqueda de hiperparámetros con Optuna (pipeline activa)

Fecha: 2026-08-04
Branch: `feature/optuna-hyperparam-search`

> Estado: **diseño pendiente; no implementado**. Este documento no implica que
> Optuna esté disponible en el código. Cualquier implementación debe usar el
> contrato activo de **17 features** definido por `ml/trainer.py`, incluidas
> `n_tuberias_in`, `n_tuberias_out`, `duracion_horas` y `tiempo_al_pico_h`.
> Debe revisarse nuevamente después de retirar la pipeline legacy.

## Objetivo

Reemplazar los hiperparámetros fijos de XGBoost en `config.yaml` (clasificador
y regresor) por valores encontrados con una búsqueda bayesiana (Optuna),
usando la misma validación agrupada que ya usa el proyecto (LOSO/GroupKFold5
por `factor_mult`). El resultado se aplica automáticamente a `config.yaml`
si mejora o iguala el desempeño actual.

## Contexto: qué pipeline se optimiza

El repositorio tiene dos rutas de entrenamiento ML paralelas. Según
`XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md` (fuente de verdad del propio
proyecto) y `AUDITORIA_CODIGO_BASURA_2026-06-15.md`:

- **Pipeline activa ("spec v4")**: `main.py` → `dataset/assembler.py` →
  `ml/trainer.py` → `ml/evaluator.py` → `ml/predict.py`, dirigida por
  `config.yaml` (`ml.classifier` / `ml.regressor`). Solo XGBoost o
  RandomForest. Escribe `outputs/models/classifier.joblib` /
  `regressor.joblib`. **Esta es la que se optimiza en este spec.**
- **Pipeline legacy ("Pipeline A")**: `ml/train.py`
  (`ML_MODEL_CONFIGS`, 7 modelos, `model_artifacts/`), `ml/predict_tabular.py`,
  `ml/predict_from_inp.py`, `ml/scenario_predict.py`. Sigue cableada a la GUI
  de escritorio (`desktop/app.py`) y a tests, por lo que no se borra aquí. Su
  retiro ya fue identificado por el proyecto como tarea aparte, controlada, y
  **queda explícitamente fuera de alcance de este trabajo**.

## Espacio de búsqueda

Los 4 hiperparámetros que ya existen en `ClassifierConfig` y `RegressorConfig`
(`swmm_resilience/config.py`), para el algoritmo ya configurado en
`config.yaml` (`xgboost`). No se agregan campos nuevos al esquema:

| Hiperparámetro | Rango | Escala |
|---|---|---|
| `n_estimators` | 100 – 800 | uniforme (int) |
| `max_depth` | 3 – 10 | uniforme (int) |
| `learning_rate` | 1e-3 – 0.3 | log-uniforme |
| `subsample` | 0.5 – 1.0 | uniforme |

`scale_pos_weight` del clasificador sigue calculándose dinámicamente por
fold (`n_neg / n_pos`), igual que hoy en `trainer.make_classifier`. No se
busca como hiperparámetro.

## Dos estudios independientes

Un estudio para el clasificador y otro para el regresor — no conjunto. Son
funciones independientes en `trainer.py` (`make_classifier` / `make_regressor`)
y acoplarlas duplicaría el espacio de búsqueda sin necesidad.

- **Estudio clasificador**: maximiza F1 medio.
- **Estudio regresor**: minimiza RMSE medio (nodos inundados, espacio
  `log1p`, igual que `trainer.train_models` ya entrena hoy).

## Validación en dos niveles

- **Búsqueda** (cada trial, dentro del estudio): `GroupKFold(5)` agrupado por
  `factor_mult` — mismo agrupamiento que ya usa `evaluator._run_cv`. Proxy
  rápido para explorar el espacio.
- **Reporte final** (una sola vez por estudio, con los mejores
  hiperparámetros encontrados): `LeaveOneGroupOut` agrupado por `factor_mult`
  — el mismo LOSO que ya calcula `evaluator.py` hoy. El resultado se escribe
  en el mismo formato que `metrics_classifier.json` / `metrics_regressor.json`
  para que sea directamente comparable con las corridas actuales.

Nota de alcance: agrupar por `factor_mult` prueba generalización a un
multiplicador de caudal no visto, no a una *forma* de hidrograma no vista —
hoy no existe una columna de identidad de hidrograma en la base de datos para
lo segundo. Si se agrega esa columna más adelante (trabajo de escalamiento de
base de datos, fuera de este spec), el agrupamiento del LOSO debería
revisarse.

## Arquitectura

Módulo nuevo `swmm_resilience/ml/hyperparam_search.py`.

```python
def build_objective(
    task: str,              # "classifier" | "regressor"
    df: pd.DataFrame,
    config: Config,
    pruner_report_fn,       # trial.report + trial.should_prune, inyectado para poder testear sin Optuna
) -> Callable[[optuna.Trial], float]:
    """Construye la función objetivo de un estudio.

    Reutiliza la lógica de fold de evaluator._run_cv (GroupKFold(5) por
    factor_mult), pero entrena solo el modelo de `task` en cada fold en vez
    de los dos, y reporta la métrica intermedia por fold para permitir
    pruning.
    """

def run_study(
    task: str,
    df: pd.DataFrame,
    config: Config,
    timeout_sec: int,
    seed: int = ML_RANDOM_STATE,
) -> optuna.Study:
    """Crea el estudio (TPESampler(seed=seed), MedianPruner) y llama
    study.optimize(objective, timeout=timeout_sec)."""

def evaluate_best_params_loso(
    task: str,
    best_params: dict,
    df: pd.DataFrame,
    config: Config,
) -> dict:
    """Refit con los mejores hiperparámetros, evalúa con LeaveOneGroupOut
    por factor_mult. Devuelve el mismo dict de métricas que ya produce
    evaluator._run_cv para ese nivel (classifier o regressor_oracle)."""

def apply_if_better(
    task: str,
    best_params: dict,
    new_score: dict,
    config_yaml_path: Path,
) -> bool:
    """Compara new_score contra el LOSO actual (recalculado con los
    hiperparámetros vigentes en config.yaml, mismos folds). Si new_score es
    igual o mejor: hace backup de config.yaml (timestamped) y sobrescribe la
    sección ml.classifier / ml.regressor correspondiente. Si es peor: no
    toca config.yaml. Devuelve True/False según si se aplicó."""

def run_hyperparam_search(
    config_path: str = "config.yaml",
    timeout_per_study: int = 600,
    tasks: list[str] | None = None,   # None = ambos
) -> dict:
    """Orquesta ambos estudios, escribe el reporte y aplica los resultados.
    Punto de entrada usado por main.py --tune-hyperparams."""
```

### Pruning

Loop manual por fold (no `cross_validate`, porque se necesita reportar
puntaje intermedio): por cada fold de `GroupKFold(5)`, `trial.report(score,
step=fold_index)` y `trial.should_prune()`. Pruner: `MedianPruner`. Esto
importa para cuando el dataset crezca — evita gastar el presupuesto de tiempo
en combinaciones de hiperparámetros claramente malas.

Sampler: `TPESampler(seed=ML_RANDOM_STATE)` para reproducibilidad entre
corridas con el mismo dataset.

## Aplicación automática y salvaguarda

`config.yaml` ya es un archivo externo editable — no se introduce un formato
nuevo. `apply_if_better`:

1. Antes de escribir, copia `config.yaml` a
   `config.yaml.bak.<YYYYMMDD-HHMMSS>` (mismo directorio).
2. Recalcula el LOSO **actual** (con los hiperparámetros vigentes) sobre los
   mismos folds, para comparar en igualdad de condiciones.
3. Compara según la dirección de la métrica de cada tarea: clasificador →
   `f1_nuevo >= f1_actual` es mejor-o-igual; regresor → `rmse_nuevo <=
   rmse_actual` es mejor-o-igual (RMSE: menor es mejor). Si se cumple:
   sobrescribe la sección `ml:` correspondiente en `config.yaml`.
4. Si no se cumple (empeora): dejar `config.yaml` intacto y marcar
   `"applied": false, "reason": "regression_detected"` en el reporte.

## Reporte

- Tabla en consola (mismo estilo que `print_results_table` de `train.py`):
  una fila por estudio con hiperparámetros encontrados, score de búsqueda
  (GroupKFold5), score de reporte (LOSO), score LOSO anterior, y si se
  aplicó.
- `outputs/metrics/hyperparam_search_report.json`:

```json
{
  "generated_at": "...",
  "dataset_csv": "data/training/dataset_final.csv",
  "dataset_rows": 4000,
  "classifier": {
    "params": {"n_estimators": ..., "max_depth": ..., "learning_rate": ..., "subsample": ...},
    "search_score_f1_groupkfold5": ...,
    "loso_f1_new": ...,
    "loso_f1_previous": ...,
    "n_trials": ...,
    "n_pruned": ...,
    "applied": true
  },
  "regressor": { "...": "mismo esquema, con rmse en vez de f1" }
}
```

## CLI

Nuevo flag `--tune-hyperparams` en el `main.py` raíz (consistente con cómo ya
se invocan las demás operaciones de la pipeline activa vía flags, a
diferencia de `train.py` que se corre aparte con `python -m`).

```
python main.py --tune-hyperparams [--timeout-per-study 600] [--only-classifier | --only-regressor]
```

Requiere `dataset_final.csv` ya generado (igual que `--only-ml`); si no
existe, error claro pidiendo correr el pipeline completo o `--skip-extraction`
primero.

Nueva dependencia: `optuna` en `requirements.txt`.

## Fuera de alcance (YAGNI)

- No se toca `ml/train.py` / Pipeline A ni sus 7 modelos — queda anotado como
  tarea de retiro controlado pendiente, no se ejecuta aquí.
- No se agregan hiperparámetros nuevos de XGBoost (`colsample_bytree`,
  `reg_alpha`, `reg_lambda`) — solo los 4 que ya existen en
  `ClassifierConfig`/`RegressorConfig`.
- No se optimiza `random_forest` — se mantiene el algoritmo ya configurado
  (`xgboost`) en `config.yaml`.
- No se implementa un estudio conjunto/end-to-end (clasificador + regresor
  optimizados juntos) — dos estudios independientes.
- No se agrega identidad de hidrograma a la base de datos ni se cambia el
  agrupamiento del LOSO a nivel de hidrograma — depende de un trabajo de
  escalamiento de base de datos que no forma parte de este spec.
- No se paraleliza la ejecución de trials (`n_jobs=1` en `study.optimize`) —
  mantiene reproducibilidad simple; paralelizar es una mejora futura posible.

## Pruebas

- `tests/ml/test_hyperparam_search.py`:
  - `build_objective`: con un DataFrame sintético pequeño y un `optuna.Trial`
    de prueba, verifica que devuelve un float y que usa `GroupKFold(5)` por
    `factor_mult` (no por `run_id`).
  - `evaluate_best_params_loso`: verifica que el dict de salida tiene las
    mismas claves que produce `evaluator._run_cv` para ese nivel.
  - `apply_if_better`: dos casos — score mejor (se sobrescribe `config.yaml`
    y se crea el backup) y score peor (no se toca `config.yaml`, se marca
    `applied: false`).
  - Smoke test de `run_hyperparam_search` con `timeout_per_study` muy bajo
    (2-3 segundos) para que corra rápido en CI y solo valide que no truena y
    que el JSON de reporte tiene el esquema esperado.

## Restricciones operativas

- Todo el trabajo en la branch `feature/optuna-hyperparam-search`.
- No borrar archivos, outputs ni branches existentes.
- No modificar `ml/train.py` ni el resto de Pipeline A.
