# Búsqueda automática de hiperparámetros (Optuna) — Plomería previa al benchmark

## 1. Objetivo y alcance

Construir el mecanismo de búsqueda automática de hiperparámetros con Optuna que exige el spec
`2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` §12, como capacidad disponible en el
banco — **no** como paso obligatorio antes del benchmark de comparación de familias. El benchmark que
sigue a este trabajo (con los datos actuales, protocolo `LeaveOneShapeOut` incluido) sigue usando los
hiperparámetros fijos de `config.yaml`, exactamente como manda §12: "Optuna se abordará solo después de
las fases anteriores y sobre la familia candidata elegida por el benchmark fijo."

Este documento no reabre fase A ni decide la familia ganadora; solo entrega la herramienta para usarla
después, manualmente, sobre la familia que el benchmark elija.

## 2. Evidencia comprobada

- `evaluate_candidate()` en `swmm_resilience/ml/bench/evaluate.py` ya está documentada como "función
  pura de los hiperparámetros: exactamente la forma que necesita un `objective` de Optuna", pero nada la
  invoca en un bucle de búsqueda hoy.
- `optuna` no está instalado (`ModuleNotFoundError`) ni declarado en `requirements.txt` ni
  `pyproject.toml`.
- `PROTOCOL_GROUP_COLUMNS` (fase A) solo cubre los tres protocolos persistidos del banco (`LOSO`,
  `GroupKFold5`, `LeaveOneShapeOut`); no existe hoy un generador de folds internos para tuning.
- `config.yaml` fija los hiperparámetros de las cinco familias sin mecanismo de búsqueda; varios
  comentarios dicen explícitamente "Pendiente de afinar con Optuna".
- `AUDITORIA_SURROGATE_HIDROGRAMAS.md` (línea 321) documenta la misma brecha de forma independiente.

## 3. Decisiones de alcance (confirmadas con el usuario)

- **Orden**: esta pieza es solo plomería. El benchmark que sigue a este trabajo usa hiperparámetros
  fijos de `config.yaml`; Optuna se ejecuta después, manualmente, sobre la familia ganadora.
- **Holdout de formas**: no se reserva ninguno en esta iteración. Todo resultado de un estudio Optuna se
  etiqueta explícitamente como `estimacion_exploratoria_condicionada`, tal como exige §12 cuando falta
  un holdout separado o selección de familia anidada.
- **Espacio de búsqueda**: declarado en código por familia (ver §6), no en YAML.
- **Presupuesto**: `n_trials = 50` por defecto, configurable en `config.yaml`; sin timeout de tiempo.

## 4. Arquitectura

Módulo nuevo `swmm_resilience/ml/bench/tune.py`, paralelo a `evaluate.py` y **fuera** del camino que
usan `--bench`, `--bench-train` y `--bench-evaluate`. Se invoca solo con el flag nuevo `--bench-tune`.

```
PreparedDataset (el mismo que usa el resto del banco)
        │
        ▼
build_tuning_folds(keys, n_splits=5, seed=ML_RANDOM_STATE)
        │  GroupKFold(5) agrupado por shape_id, sobre TODO el dataset preparado
        │  (no hay outer holdout separado en este modo)
        ▼
optuna.Study (TPESampler(seed=...), direction="minimize")
        │  cada trial: sample_params(family) → classifier_params, regressor_params
        │               → promedio de end_to_end.rmse_vol_todos_nodos sobre los 5 folds internos
        ▼
TuneResult → study.json bajo outputs/bench/tuning/<family>/<prep_id>_<timestamp>/
```

### 4.1 Helper compartido anti-fuga

Hoy `evaluate_candidate()` tiene inline la lógica de "ajustar clasificador y regresor solo con el train
del fold, predecir sobre el test del fold". Se extrae a un helper privado compartido:

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

    Devuelve (y_pred_clf, y_prob_clf, y_pred_reg). Única función que decide
    fit/predict de un fold; evaluate_candidate() y tune.py la comparten para
    que la garantía anti-fuga viva en un solo lugar.
    """
```

`evaluate_candidate()` pasa a llamarla por cada `(protocol, fold_id)` de `prepared.folds` (sin cambiar
su contrato externo ni el formato de `OOF_COLUMNS`). `tune.py` la llama por cada fold interno que genera
`build_tuning_folds()`, sin pasar por `score_predictions()` (que exige cobertura exacta contra
`prepared.folds` y no debe validar folds efímeros de un trial).

### 4.2 Folds internos de tuning

```python
def build_tuning_folds(
    keys: pd.DataFrame, n_splits: int = 5, seed: int = ML_RANDOM_STATE
) -> pd.DataFrame:
    """GroupKFold(n_splits) agrupado por shape_id sobre todo `keys`.

    Devuelve un frame con columnas ('fold_id', 'split', 'sample_idx'), sin
    'protocol' persistido: estos folds son efímeros, por-trial, y nunca se
    escriben en el manifiesto de PreparedDataset ni en PROTOCOL_GROUP_COLUMNS
    (que gobierna solo los tres protocolos de comparación de familias).
    """
```

No reutiliza `PROTOCOL_GROUP_COLUMNS`/`group_column_for()` de fase A a propósito: esa tabla es el
catálogo persistido de protocolos de comparación; los folds de tuning son un mecanismo interno de
optimización, nunca se declaran como protocolo del banco ni entran al `prep_id`.

### 4.3 Objetivo Optuna

```python
def _objective(trial, prepared, family_name, tuning_folds) -> float:
    classifier_params = _sample_params(trial, SEARCH_SPACES[family_name]["classifier"])
    regressor_params = _sample_params(trial, SEARCH_SPACES[family_name]["regressor"])
    fold_metrics = []
    for fold_id, chunk in tuning_folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk.split == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk.split == "test", "sample_idx"].to_numpy()
        yc_pred, yc_prob, yr_pred = _fit_predict_fold(
            prepared, family_name, classifier_params, regressor_params, train_idx, test_idx
        )
        routed = np.where(yc_pred == 1, yr_pred, 0.0)
        fold_metrics.append(
            end_to_end_metrics(
                prepared.y_reg.to_numpy()[test_idx], routed,
                prepared.y_clf.to_numpy()[test_idx], yc_pred,
            )["rmse_vol_todos_nodos"]
        )
    return float(np.mean(fold_metrics))
```

Optuna minimiza este valor. `trial.set_user_attr` guarda `classifier_params`/`regressor_params`
completos (con los defaults de la familia ya mezclados) para que cada fila de `trials` en `study.json`
sea autocontenida.

### 4.4 `run_study()` y `TuneResult`

```python
@dataclass(frozen=True)
class TuneResult:
    family: str
    prep_id: str
    inner_cv: dict            # {"method": "GroupKFold", "n_splits": 5, "group_column": "shape_id"}
    objective_metric: str     # "end_to_end.rmse_vol_todos_nodos"
    direction: str            # "minimize"
    n_trials: int
    seed: int
    optuna_version: str
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    trials: list[dict]        # [{number, classifier_params, regressor_params, value, duration_s}, ...]
    best_classifier_params: dict
    best_regressor_params: dict
    best_value: float
    status: str                # "estimacion_exploratoria_condicionada"
    status_reason: str

def run_study(
    prepared: PreparedDataset, family_name: str, n_trials: int = 50,
    n_splits: int = 5, seed: int = ML_RANDOM_STATE,
) -> TuneResult: ...

def save_tune_result(result: TuneResult, output_dir: Path) -> Path: ...
```

`status_reason` fijo: *"No se reservó un holdout de formas separado ni se aplicó selección de familia
anidada (spec §12, sección 'Optimización posterior'); este resultado no debe presentarse como
confirmatorio."*

## 5. Persistencia

```
outputs/bench/tuning/<family>/<prep_id>_<timestamp>/study.json
```

Ejemplo de forma (valores ilustrativos):

```json
{
  "family": "xgboost",
  "prep_id": "af3751551ca5df98",
  "inner_cv": {"method": "GroupKFold", "n_splits": 5, "group_column": "shape_id"},
  "objective_metric": "end_to_end.rmse_vol_todos_nodos",
  "direction": "minimize",
  "n_trials": 50,
  "seed": 42,
  "optuna_version": "...",
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "duration_s": 0.0,
  "trials": [
    {"number": 0, "classifier_params": {}, "regressor_params": {}, "value": 0.0, "duration_s": 0.0}
  ],
  "best_classifier_params": {},
  "best_regressor_params": {},
  "best_value": 0.0,
  "status": "estimacion_exploratoria_condicionada",
  "status_reason": "No se reservó un holdout de formas separado ni se aplicó selección de familia anidada (spec §12, sección 'Optimización posterior'); este resultado no debe presentarse como confirmatorio."
}
```

No se escribe ranking, no se promueve nada, no se toca `outputs/models/`. El resultado son
hiperparámetros que el usuario copia manualmente a `config.yaml` si decide adoptarlos.

## 6. Espacios de búsqueda por familia

Declarados en código (`SEARCH_SPACES` en `tune.py`), alrededor de los defaults actuales de
`config.yaml`. Notación `(tipo, low, high)`; `log` implica muestreo log-uniforme.

| Familia | Clasificador | Regresor |
|---|---|---|
| `xgboost` | `max_depth` int(3,10); `learning_rate` log(1e-3,0.3); `n_estimators` int(50,500); `subsample` float(0.5,1.0) | mismos cuatro parámetros, mismos rangos |
| `random_forest` | `max_depth` int(3,15); `n_estimators` int(50,500) | mismos dos parámetros, mismos rangos |
| `linear` | `C` log(1e-3,100) (`max_iter` fijo en 5000, no se busca) | `alpha` log(1e-3,100) |
| `svm` | `C` log(1e-2,100); `gamma` categórico {"scale","auto"} | `C` log(1e-2,100); `epsilon` float(0.01,1.0) |
| `mlp` | `hidden_sizes` categórico {(32,), (64,32), (128,64), (64,32,16)}; `learning_rate` log(1e-4,1e-2); `dropout` float(0.0,0.5) | mismos tres parámetros, mismos rangos |

`filter_estimator_params()` (ya existente en `param_filter.py`) sigue filtrando los parámetros que cada
estimador de sklearn/xgboost acepta, igual que hoy con los valores fijos de `config.yaml`; los valores
muestreados por Optuna pasan por el mismo filtro antes de construir el estimador.

## 7. CLI y configuración

`main.py`: flag nuevo `--bench-tune`. Exige `--models <una-sola-familia>` (rechaza más de una o
ninguna) y resuelve el `PreparedDataset` con `resolve_prepared()` igual que `--bench-train`, validando
protocolos y semántica de caché. No corre en el mismo `if` que `run_all`/`--bench`; es un modo aparte
que nunca se activa implícitamente.

`config.yaml`: bloque nuevo opcional `bench.tuning: {n_trials: 50, n_splits: 5}`. La semilla reutiliza
`ML_RANDOM_STATE` (ya existente en `swmm_resilience/config.py`); no se declara una semilla nueva en YAML
para evitar dos fuentes de verdad sobre el determinismo del banco.

`swmm_resilience/config.py`: `BenchTuningConfig` dataclass (`n_trials: int`, `n_splits: int`), con
defaults `50`/`5` si el bloque falta —`--bench-tune` sigue funcionando sin que el usuario toque
`config.yaml`.

## 8. Dependencia nueva

`optuna` no está instalado. Se añade a `requirements.txt` con versión fijada (siguiendo la convención
`paquete==version` del archivo). El import de `optuna` queda dentro de `tune.py`, que a su vez se
importa perezosamente dentro de `_run_bench()` en `main.py` (mismo patrón que las otras etapas del
banco), para que cargar `config.yaml` sin usar `--bench-tune` no pague el costo de importar Optuna.

## 9. Pruebas (TDD)

- `tests/ml/bench/test_tune.py` (nuevo):
  - `build_tuning_folds` produce una partición GroupKFold(5) completa por `shape_id`, sin fuga (mismo
    tipo de aserción que Task 1 de fase A, pero para 5 folds en vez de 40).
  - `_fit_predict_fold` da exactamente los mismos resultados que la ruta equivalente dentro de
    `evaluate_candidate()` para el mismo fold, antes/después del refactor (test de paridad, análogo al
    que ya existe entre `evaluate.py` y `ml/trainer.py`).
  - Un estudio con `n_trials` pequeño (2–3) sobre un fixture minúsculo corre sin error, produce
    `best_value` finito y `study.json` con todos los campos de §5, incluido `status` fijo.
  - `run_study` con una familia con hiperparámetros inválidos en un trial no aborta el estudio completo
    (Optuna debe poder descartar/podar ese trial sin tumbar los demás).
  - `test_bench_cli.py`: `--bench-tune` sin `--models` o con más de una familia falla con `SystemExit`
    explícito; `--bench-tune` con una familia y un `prep_id` de caché legacy (semántica incompatible) se
    rechaza igual que `--bench-train`; una corrida de `--bench-tune` no escribe en `outputs/models/` ni
    en `outputs/bench/reports/ranking.csv`.
  - `test_bench_config.py`: `bench.tuning` se parsea con defaults cuando falta el bloque; rechaza
    `n_trials`/`n_splits` no positivos.

## 10. Fuera de alcance (explícitamente pospuesto)

- Ejecutar el estudio contra la familia ganadora real: solo ocurre después del benchmark de fase D
  (fixed hyperparameters) que decide qué familia gana.
- Holdout de formas separado o selección de familia anidada para una afirmación confirmatoria: spec §12
  lo permite posponer, siempre que el resultado quede etiquetado como exploratorio (§5 de este
  documento ya lo hace).
- Nested CV por cada uno de los 40 folds exteriores de `LeaveOneShapeOut` (la lectura literal más
  costosa de "el inner GroupKFold verá exclusivamente el train del fold exterior"): se adopta en su
  lugar un único estudio con GroupKFold(5) interno sobre todo el dataset, más barato, y por eso mismo
  el resultado se etiqueta exploratorio en vez de confirmatorio.
- Cualquier cambio a `_run_bench()` que active tuning implícitamente desde `--bench`: no ocurre; el flag
  es siempre explícito.
