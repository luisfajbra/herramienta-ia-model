# Banco ML: preprocesamiento → entrenamiento → evaluación

**Fecha:** 2026-09-05
**Estado:** Aprobado en conversación; listo para plan de implementación
**Branch:** `feature/hydrograph-augmentation-sql-persistence`

**Relación con specs previas:**

- **Reanuda** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`,
  marcada *superseded* el 2026-09-03 con la nota "no reanudar este roadmap sin
  re-confirmar con el dueño del proyecto". Esta spec **es** esa re-confirmación,
  con un alcance redefinido: solo la capa ML (registro unificado de modelos +
  retiro del legacy). El `SimulationStore` y la persistencia por paso de tiempo
  siguen fuera de alcance.
- **Continúa** `docs/FLUJO_ACTUAL.md` §12: se apoya en el corte de lectura ya
  hecho (`load_training_frame`), no lo rehace.
- **Desbloquea** `docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`,
  que declara "debe revisarse nuevamente después de retirar la pipeline legacy".
  Esta spec deja el punto de enganche definido; la implementación de Optuna es
  trabajo posterior.

---

## 1. Propósito

Reemplazar los dos stacks de entrenamiento tabular que hoy coexisten por **una
sola capa ML**, alimentada desde SQLite v17, con tres etapas separadas —
preprocesamiento, entrenamiento y evaluación — ejecutables de forma
independiente, y capaz de comparar varias familias de modelos (incluidas redes
neuronales tabulares) bajo condiciones idénticas.

El resultado esperado es una tabla comparativa defendible: mismas entradas,
mismos folds, mismas métricas, para XGBoost, RandomForest, modelos lineales,
SVM y MLP, con la evidencia completa persistida en SQL.

## 2. No-objetivos

- **No** se implementan modelos temporales (LSTM/CNN). `swmm_resilience/ml/temporal/`
  no se toca. Esta spec deja definido el contrato de datos que un modelo
  temporal futuro deberá cumplir para ser comparable, y nada más.
- **No** se implementa búsqueda de hiperparámetros. Los valores quedan
  declarados en `config.yaml` con su justificación escrita. El punto de
  enganche para Optuna queda definido (§13.2).
- **No** se implementa `SimulationStore` ni persistencia por paso de tiempo
  (`node_timeseries` sigue vacía).
- **No** se cambia el contrato de 17 features (`TABULAR_V3_17`).
- **No** se cambian los protocolos de validación: siguen siendo LOSO y
  GroupKFold5, ambos agrupados por `factor_mult`.
- **No** se completa la Fase 2 del cutover de `FLUJO_ACTUAL.md` §12 (que
  `assembler.py` escriba a SQL). `--persist-sql` sigue siendo el camino de
  escritura de datos.

## 3. Decisiones tomadas

| # | Decisión | Valor |
|---|---|---|
| D1 | Alcance del retiro | Ambos stacks tabulares (`ml/train.py` y `ml/trainer.py`+`ml/evaluator.py`) se reemplazan por una capa nueva alimentada desde SQL |
| D2 | Modelos soportados | Banco multi-modelo: XGBoost, RandomForest, lineales, SVM, MLP — todos enchufables bajo la misma interfaz |
| D3 | Separación del preprocesamiento | Dos niveles: nivel dataset (determinista, materializado, sin fuga) y nivel fold (imputación/escalado ajustados solo con el train del fold) |
| D4 | Redes neuronales | MLP tabular sobre las mismas 17 features, como una familia más. LSTM/CNN: solo contrato de datos, sin código |
| D5 | GUI de escritorio | Se elimina toda su parte ML; la GUI queda para simulación SWMM y visor de resultados |
| D6 | Protocolos de validación | LOSO por factor + GroupKFold5 por factor, idénticos a hoy |
| D7 | Persistencia de resultados | SQL como fuente de verdad (cadena completa) + export a archivos |
| D8 | Unidad de comparación | Un candidato = cascada clasificador+regresor de la misma familia, medida en 3 niveles |
| D9 | Entorno para añadir modelos | Registro en código (`bench/models/*.py`) + hiperparámetros en `config.yaml` |
| D10 | Ejecución de etapas | Tres comandos independientes con artefacto intermedio persistido |
| D11 | Artefactos | Por candidato en `outputs/bench/candidates/<familia>/`, y el promovido también en `outputs/models/` con el formato actual |
| D12 | Orden de migración | Capa nueva en paralelo, test de paridad verde, y solo entonces se borra lo viejo |
| D13 | Criterio de ranking | `end_to_end.rmse_vol_todos_nodos`, dirección `minimize`, configurable |
| D14 | Cadena de proveniencia SQL | Completa: candidates → evaluations → oof → finalizations → trained_models → rankings → promotions → selections |
| D15 | `model_blob` | Migración 006: `model_blob` pasa a nullable, se añade `model_path`; los artefactos viven en disco con su sha256 en la base |
| D16 | Hiperparámetros | Fijos y justificados en `config.yaml`; búsqueda automática (Optuna) como etapa posterior con enganche ya definido |

## 4. Estado actual verificado (leído del código, 2026-09-05)

### 4.1 Stack A — comparación de 7 modelos (legacy, desconectado de `main.py`)

- `swmm_resilience/ml/train.py` (1116 líneas): `build_models()` (Ridge, Lasso,
  SVR-RBF, XGBoost, RandomForest), `build_classification_models()`,
  `evaluate_models()`, `evaluate_classification_models()`,
  `fit_and_save_inference_models()`, `load_saved_model_artifact()`, manifest en
  `model_artifacts/`. Usa `StandardScaler` + `PCA` (`ML_USE_PCA = True`,
  `ML_PCA_COMPONENTS = 5`) y `GroupShuffleSplit`/`GroupKFold` por `run_id`.
- `swmm_resilience/ml/preprocessing.py`: `select_features_for_model()` —
  selección por **lista negra** (`ML_DROP_COLUMNS`) + filtro de dtypes
  numéricos. Consumido únicamente por `ml/train.py`.
- Consumidores: `ml/predict_tabular.py` → `ml/predict_from_inp.py` →
  `visualization/loaders.py` (`load_from_ml`, `load_all_ml`) →
  `visualization/runner.py` → `desktop/app.py`.
- Fuente de datos: `data/training/dataset_ml.csv` + `swmm_resilience.db` +
  `model_artifacts/`. **`main.py` no importa ninguno de estos módulos.**

### 4.2 Stack B — cascada activa (CLI)

- `swmm_resilience/ml/trainer.py`: `make_classifier()` / `make_regressor()`
  (`SimpleImputer(median)` + XGBoost), `train_models()`. Re-exporta
  `FEATURE_COLS = list(FEATURE_COLUMNS_V17)` con el comentario
  `# removed in Plan D after consumers migrate`.
- `swmm_resilience/ml/evaluator.py`: `_run_cv()` con LOSO / GroupKFold5
  agrupados por `factor_mult`, métricas en 3 niveles, estratificación por factor.
- Fuente de datos: `load_training_frame(config.dataset.db_path)` sobre
  `outputs/training_v17.sqlite3`.
- Salida: `outputs/models/{classifier,regressor}.joblib` +
  `training_inp_hash.txt`; `outputs/metrics/metrics_*.json`.

### 4.3 Detalles del evaluador actual que hay que preservar

- Métricas del clasificador: **promediadas entre folds**.
- Métricas del regresor-oracle: **calculadas sobre el pool concatenado** de
  todos los folds, no promediadas.
- Nivel 2 (oracle) filtra las filas de test con las **etiquetas reales**; nivel
  3 (end-to-end) enruta con las **etiquetas predichas**.
- El regresor se entrena sobre `log1p(vol)` y se invierte con `expm1`, con
  `clip(min=0)`.
- XGBoost va **sin escalador**: solo `SimpleImputer(median)`.

Esta asimetría promedio/pool es hoy invisible y cambia los números. Se conserva
idéntica (para que la paridad sea comprobable) y se documenta explícitamente.

### 4.4 Esquema SQL disponible y sin usar

Migraciones 001–005 ya definen la cadena completa, con triggers de
inmutabilidad y coherencia:

```
training_runs (primary_metric, tie_breakers_json, grouping_strategy IN ('group_kfold','loso'))
 ├─ model_candidates (candidate_definition_sha256)
 │   ├─ model_candidate_evaluations → model_evaluations (UNIQUE por training_run+task+algorithm+fold)
 │   │                                  └─ oof_predictions (PK: evaluation_id, run_id, node_pk)
 │   └─ model_candidate_finalizations
 │       └─ model_artifact_candidates → trained_models
 ├─ model_metrics (owner_kind: training_run | evaluation | model)
 └─ model_rankings (primary_metric, primary_direction, metric_registry_sha256, invalid_score_policy)
     ├─ model_ranking_entries (classifier_candidate_id, regressor_candidate_id)
     ├─ model_ranking_scores (metric_ordinal, valid, invalid_reason)
     └─ model_ranking_finalizations
         └─ model_promotion_rankings → model_promotions → model_selections
```

Hoy solo `csv_backfill.py::persist_training_run` escribe una parte
(`training_runs → model_evaluations → oof_predictions → trained_models →
model_metrics`) para el par XGB. Las tablas de candidatos, ranking, promoción y
selección están vacías.

Triggers relevantes que condicionan el orden de escritura:

- `model_candidate_evaluations_matches_candidate`: la evaluación enlazada debe
  coincidir en `(task, algorithm, hyperparameters_json)` con su candidato.
- `model_candidate_evaluations_not_after_finalization`: no se añaden folds
  después de finalizar el candidato.
- `model_artifact_candidates_requires_finalized_matching_candidate`: el
  artefacto solo se enlaza a un candidato **finalizado** cuya receta completa
  (contrato, features ordenadas, preprocesamiento, transformación de target,
  hiperparámetros, algoritmo) sea idéntica.
- `model_ranking_entries_not_after_finalization` y su equivalente en scores.
- `model_promotions_immutable_update` / `_delete`.

### 4.5 Conflicto conocido

`trained_models.model_blob` es `BLOB NOT NULL` (migraciones 001 y 002), pero
`FLUJO_ACTUAL.md` §12.5 indica "no usar el blob; guardar el `.joblib` en disco
y registrar ruta + sha256". Con N familias × 2 modelos, el blob multiplica el
tamaño de la base. Resolución: **D15** (migración 006).

## 5. Arquitectura

Paquete nuevo `swmm_resilience/ml/bench/`, una etapa por módulo, dependencias en
una sola dirección:

```
outputs/training_v17.sqlite3
        │  load_training_frame()            ← ya existe, no se modifica
        ▼
  bench/preprocess.py      ETAPA 1 — determinista, sin fuga
        ▼  PreparedDataset (persistido, con prep_id)
  bench/models/*.py        REGISTRO de familias (construye Pipelines)
        ▼
  bench/train.py  ─────────  bench/evaluate.py
        │                          │
        └──────────┬───────────────┘
                   ▼
             bench/persist.py  →  SQL + outputs/bench/reports/
                   ▼
             bench/promote.py  →  outputs/models/ (formato actual)
```

### 5.1 Reglas de frontera

Estas reglas son verificables por test y son lo que hace la capa auditable:

1. **`preprocess.py` no importa `sklearn` ni `torch`.** Solo pandas, numpy y
   `ml/contracts.py`. Garantía estructural contra la fuga: no tiene con qué
   ajustar un transformador.
2. **`models/*.py` no leen SQL ni `config.yaml`.** Reciben un dict de
   hiperparámetros ya resuelto y devuelven un `Pipeline` de sklearn.
3. **`evaluate.py` no construye modelos.** Recibe una función constructora del
   registro, de modo que todos los candidatos ven exactamente los mismos folds.
4. **`train.py` y `evaluate.py` no se importan entre sí.** Comparten el
   `PreparedDataset` y el registro, nada más.
5. **`persist.py` es el único módulo que escribe SQL** en toda la capa.

## 6. Contrato de la etapa 1 — preprocesamiento

### 6.1 Firma

```python
prepare_dataset(
    db_path: Path,
    *,
    run_ids: list[int] | None = None,
    protocols: tuple[str, ...] = ("LOSO", "GroupKFold5"),
    flood_threshold_m3: float,
    output_dir: Path,
) -> PreparedDataset
```

### 6.2 Pasos, en orden

1. `load_training_frame(db_path)` — ya valida runs `COMPLETE`, cardinalidad,
   snapshot bajo `SAVEPOINT` y el contrato.
2. Selección por **lista blanca ordenada** (`FEATURE_COLUMNS_V17`).
3. Targets: se toman `inunda` y `vol_inundacion_m3` **tal como vienen de la
   vista `training_samples_v17`**; no se re-derivan. `inunda` ya está
   persistido y re-calcularlo contra `flood_threshold_m3` podría discrepar de
   lo que la base afirma. Lo que sí se hace es **verificar** la coherencia
   (`inunda == 1` ⟺ `vol_inundacion_m3 > flood_threshold_m3`) y reportar las
   filas discrepantes en el informe de calidad. `flood_threshold_m3` entra al
   `prep_id` como proveniencia, no como regla de derivación.
   La transformación `log1p` del regresor **no** se aplica aquí: se declara en
   `target_transform_json` y la aplica la familia.
4. **Orden canónico** por `(run_id, node_id)`. No es cosmético: sin él el
   `prep_id` no es reproducible entre corridas.
5. Construcción de grupos (`factor_mult`) y materialización de los folds de
   cada protocolo.
6. Informe de calidad (§6.4).
7. Cálculo del `prep_id` y escritura de artefactos.

### 6.3 Estructura de `PreparedDataset`

| Componente | Contenido | Archivo |
|---|---|---|
| `keys` | `run_id`, `node_pk`, `node_id`, `factor_mult`, `shape_id`, `scenario_id` | `keys.parquet` |
| `X` | las 17 features en el orden del contrato | `features.parquet` |
| `y_clf` / `y_reg` | `inunda`, `vol_inundacion_m3` | `targets.parquet` |
| `folds` | `protocol`, `fold_id`, `sample_idx`, `split` (`train`/`test`) | `folds.parquet` |
| manifest | proveniencia completa (§6.5) | `manifest.json` |
| calidad | informe de datos (§6.4) | `quality_report.json` |

`node_pk` es obligatorio en `keys` porque `oof_predictions` tiene clave
`(evaluation_id, run_id, node_pk)`.

Parquet (no CSV) porque `pyarrow` ya es dependencia, preserva dtypes exactos —
necesario para que el hash sea estable — y no re-interpreta nulos.

Ruta: `outputs/bench/prepared/<prep_id>/`.

### 6.4 Informe de calidad (`quality_report.json`)

Nuevo; hoy no existe nada equivalente dentro del pipeline (solo la EDA opcional
de `analysis/eda.py`, aparte). Contiene:

- nulos por columna, contrastados con lo que el contrato declara `nullable`;
- balance de clases global y por factor;
- número de filas por factor y por forma de hidrograma;
- min/max/percentiles (p1, p25, p50, p75, p99) por feature;
- columnas constantes o de varianza cero;
- duplicados por `(run_id, node_id)`;
- filas donde `inunda` discrepa del umbral (§6.2 paso 3);
- número de runs incluidos y su rango de fechas.

No bloquea la ejecución salvo en dos casos, que sí son errores duros: duplicados
en `(run_id, node_id)`, y cero filas con `inunda == 1` (el regresor no se puede
entrenar). Las discrepancias de umbral se reportan pero no bloquean: el umbral
pudo cambiar legítimamente entre la corrida que pobló la base y esta.

### 6.5 `prep_id` y manifest

`prep_id` = primeros 16 hex de `sha256` sobre un JSON canónico con:
`contract_id`, `feature_contract_sha256`, `run_ids` ordenados, orden de
columnas, `flood_threshold_m3`, definición de protocolos, y la clave de orden
canónico. El manifest guarda ese JSON completo, más `python_version`,
`library_versions`, `db_path`, y timestamp UTC.

Dos corridas sobre los mismos datos producen el mismo `prep_id`; cambiar el
conjunto de runs o el umbral lo cambia.

### 6.6 No-objetivos de la etapa 1

Documentados en el docstring del módulo. La etapa **no**: imputa, escala,
aplica PCA, elimina filas con nulos permitidos por el contrato, ni transforma
el target. Todo eso pertenece al nivel fold.

## 7. Registro de familias

Un archivo por familia en `bench/models/`. Cada uno expone:

```python
FAMILY = "xgboost"
SCALE_FEATURES = False          # propiedad de la familia, no del banco
def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline: ...
def build_regressor(params: dict) -> Pipeline: ...
def preprocessing_descriptor(params: dict) -> dict:  # → preprocessing_json en SQL
```

El `Pipeline` devuelto incorpora el **nivel fold** del preprocesamiento:

| Familia | Pipeline |
|---|---|
| `xgboost` | `SimpleImputer(median)` → `XGB{Classifier,Regressor}` |
| `random_forest` | `SimpleImputer(median)` → `RandomForest{Classifier,Regressor}` |
| `linear` | `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression` / `Ridge` |
| `svm` | `SimpleImputer(median)` → `StandardScaler` → `SVC` / `SVR` |
| `mlp` | `SimpleImputer(median)` → `StandardScaler` → MLP (PyTorch, envuelto con API sklearn) |

`xgboost` y `random_forest` van sin escalador deliberadamente: es lo que permite
que el candidato `xgboost` reproduzca exactamente los números de `trainer.py` en
el test de paridad (§14.2).

**PCA no vuelve.** Era del stack A, degradaba la interpretabilidad de la
importancia de variables y no hay evidencia registrada de que mejorara ninguna
métrica.

El MLP se envuelve para exponer `fit`/`predict`/`predict_proba` compatibles con
sklearn, de modo que caiga en el mismo `Pipeline` y el mismo evaluador que el
resto. Se usa un **wrapper propio** (~60 líneas sobre `torch`, que ya es
dependencia) en vez de `skorch`: evita añadir una dependencia para envolver un
modelo de tres capas, y mantiene explícito el control de semilla y de época,
que es lo que hace comparable al MLP con el resto del banco.

`scale_pos_weight` se calcula con el **train de cada fold**, nunca con el
dataset completo.

## 8. Contrato de la etapa 2 — entrenamiento

```python
train_candidate(prepared: PreparedDataset, family: str, params: dict,
                output_dir: Path) -> CandidateArtifacts
```

Ajusta la cascada sobre **todo** el dataset: clasificador sobre todas las filas,
regresor sobre `inunda == 1` con objetivo `log1p`. Es intencionalmente idéntico
a `trainer.train_models()`.

Escribe en `outputs/bench/candidates/<familia>/`:

- `classifier.joblib`, `regressor.joblib`
- `metadata.json`: familia, hiperparámetros, `prep_id`, orden de features,
  `feature_contract_sha256`, `target_transform`, `preprocessing_descriptor`,
  sha256 de cada artefacto, `python_version`, `library_versions`, semilla.

Un artefacto sin `prep_id` válido no se puede promover.

## 9. Contrato de la etapa 3 — evaluación

Dos capas. La separación **es** la puerta abierta para modelos futuros.

### 9.1 Capa núcleo — no sabe de modelos

```python
score_predictions(
    prepared: PreparedDataset,
    oof: pd.DataFrame,   # sample_idx, protocol, fold_id, y_pred_clf, y_prob_clf, y_pred_reg
    provenance: dict,    # debe incluir prep_id
) -> dict
```

Valida que `provenance["prep_id"]` coincida con el del `PreparedDataset` y
**rechaza** las predicciones si no. Verifica que la cobertura de `sample_idx`
corresponda exactamente a las filas de test de los folds declarados. Calcula las
métricas de los tres niveles y la estratificación por factor.

Cualquier modelo — tabular, temporal, escrito en otro script o dentro de seis
meses — puede pasar por aquí. Es el mecanismo que garantiza "mismas entradas,
mismos benchmarks".

### 9.2 Capa de conveniencia — la que usa el banco tabular

```python
evaluate_candidate(prepared, family, params) -> tuple[pd.DataFrame, dict]
```

Recorre los folds del protocolo, construye el `Pipeline` de la familia, lo
ajusta **solo con el train del fold**, predice el test, acumula OOF, y delega en
`score_predictions`. Es una **función pura de `params`**, que es exactamente la
forma que necesita un `objective` de Optuna (§13.2).

### 9.3 Métricas

Idénticas a `evaluator.py` de hoy, incluida la asimetría de agregación:

| Nivel | Métricas | Agregación |
|---|---|---|
| 1 — clasificador | `precision`, `recall`, `f1`, `auc_roc` | **promedio entre folds** |
| 2 — regresor oracle | `nse`, `log_nse`, `rmse`, `mae`, `r2` | **pool concatenado** |
| 3 — end-to-end | `pct_nodos_correctos`, `rmse_vol_todos_nodos`, `vol_total_pred_m3`, `vol_total_real_m3` | **promedio entre folds** |
| por factor | `f1`, `rmse_vol` | promedio entre folds, por valor de factor |

`auc_roc` es `NaN` cuando un fold no tiene ambas clases; esos casos se registran
con `valid=0` y su `invalid_reason` en `model_ranking_scores`.

### 9.4 Ranking

- Métrica primaria: `end_to_end.rmse_vol_todos_nodos`, dirección `minimize`.
- Desempates configurables; por defecto `classifier.f1` (`maximize`), luego
  `regressor_oracle.nse` (`maximize`).
- Configurable en `config.yaml` bajo `bench.ranking`.
- Un candidato con la métrica primaria inválida no se promueve; queda en el
  ranking con su motivo registrado.

## 10. Persistencia SQL

`bench/persist.py` es el único escritor. Orden de escritura, impuesto por los
triggers de §4.4:

1. `training_runs` (`status=PENDING` → `RUNNING`), con `primary_metric`,
   `tie_breakers_json`, `grouping_strategy`, `included_run_ids_json`,
   `query_sql`, `library_versions_json`.
2. Un `model_candidates` por (familia, tarea), con `candidate_definition_sha256`
   sobre la receta completa.
3. Por cada fold y candidato: `model_evaluations` → `oof_predictions` →
   `model_candidate_evaluations`.
4. `model_candidate_finalizations` cuando todos los folds del candidato están
   `COMPLETE`.
5. `trained_models` (artefacto ajustado sobre todo el dataset) →
   `model_artifact_candidates`. La receta debe coincidir exactamente con el
   candidato finalizado, o el trigger aborta.
6. `model_metrics` por evaluación y por modelo.
7. `model_rankings` → `model_ranking_entries` → `model_ranking_scores` →
   `model_ranking_finalizations`.
8. `model_promotion_rankings` → `model_promotions` → `model_selections`.
9. `training_runs.status = COMPLETE`.

Todo bajo una transacción por etapa, con `executemany` para OOF.

### 10.1 Migración 006

- `trained_models.model_blob` → nullable.
- Nueva columna `trained_models.model_path TEXT`.
- `CHECK(model_blob IS NOT NULL OR model_path IS NOT NULL)`.
- `model_sha256` sigue siendo obligatorio y ahora corresponde al archivo en
  disco.
- Recrear la tabla siguiendo el patrón de la migración 002, preservando filas
  existentes (su `model_path` queda `NULL`, su blob intacto) y re-creando los
  triggers e índices asociados.
- `csv_backfill.py::persist_training_run` desaparece en el bloque 2 del retiro
  (§12.3), así que no hay que adaptarlo.

### 10.2 `--persist-sql` queda partido

Hoy hace dos cosas: volcar el CSV a SQL **y** entrenar un par de modelos para
dejar evidencia. La segunda es exactamente lo que hace el banco. Se retira la
mitad de entrenamiento (`persist_training_run`); `--persist-sql` conserva solo
el volcado de datos. Dos escritores distintos sobre `training_runs` con reglas
distintas es incompatible con una cadena inmutable.

## 11. CLI y configuración

```bash
python main.py --bench-prepare                       # etapa 1
python main.py --bench-train [--models xgboost,mlp]  # etapa 2
python main.py --bench-evaluate                      # etapa 3
python main.py --bench-promote [--model xgboost]     # copia el ganador
python main.py --bench                               # las cuatro en secuencia
```

Cada subcomando resuelve el `PreparedDataset` más reciente por defecto, o uno
concreto con `--prep-id`. `--bench-train` y `--bench-evaluate` fallan con un
mensaje claro si no existe un `PreparedDataset` compatible.

`config.yaml`, bloque nuevo:

```yaml
bench:
  db_path: "outputs/training_v17.sqlite3"
  protocols: ["LOSO", "GroupKFold5"]
  ranking:
    primary_metric: "end_to_end.rmse_vol_todos_nodos"
    primary_direction: "minimize"
    tie_breakers:
      - {metric: "classifier.f1", direction: "maximize"}
      - {metric: "regressor_oracle.nse", direction: "maximize"}
  promote: "auto"        # "auto" = el primero del ranking; o el nombre de una familia
  families:
    xgboost:
      enabled: true
      # Justificación: valores heredados del pipeline validado en 2026-06;
      # profundidad 6 y lr 0.05 con 200 árboles evitan sobreajuste sobre
      # 175 escenarios sin perder capacidad en los factores extremos.
      classifier: {n_estimators: 200, max_depth: 6, learning_rate: 0.05, subsample: 0.8, scale_pos_weight: "auto"}
      regressor:  {n_estimators: 200, max_depth: 6, learning_rate: 0.05, subsample: 0.8}
    # ... una entrada por familia, cada una con su justificación escrita
```

Los bloques `ml:` y `evaluation:` actuales se pliegan dentro de `bench:` y
desaparecen al completar el retiro. `config_7shapes.yaml` se actualiza igual.

## 12. Retiro del legacy

### 12.1 Paso previo: desacoplar `FEATURE_COLS`

Seis módulos importan `FEATURE_COLS` de `ml/trainer.py`, cuando la constante
real vive en `ml/contracts.py` y `trainer.py` solo la re-exporta (con un
comentario que ya anticipa esta migración). Primer commit del trabajo: que
`ml/predict.py`, `ml/feature_importance.py`, `ml/feature_analysis.py`,
`ml/scenario_predict.py`, `database/csv_backfill.py` y los tests
(`tests/ml/test_feature_importance_labels.py`, `tests/ml/test_feature_analysis.py`,
`tests/test_evaluator.py`, `tests/test_scenario_predict.py`) importen
`FEATURE_COLUMNS_V17` del contrato.

Es mecánico y sin cambio de comportamiento. Después de eso, borrar `trainer.py`
no afecta a nadie más.

### 12.2 Bloque 1 — stack A (se puede borrar temprano; no lo usa `main.py`)

| Archivo | Acción |
|---|---|
| `ml/train.py` | borrar |
| `ml/predict_tabular.py` | borrar |
| `ml/predict_from_inp.py` | borrar |
| `ml/preprocessing.py` | borrar |
| `visualization/loaders.py` | quitar `load_from_ml`, `load_all_ml` |
| `visualization/runner.py` | quitar las ramas `--source ml` |
| `desktop/app.py` | quitar pestaña ML: entrenamiento, predicción y mapas ML |
| `config.py` | quitar `ML_MODEL_CONFIGS`, `ML_DROP_COLUMNS`, `ML_USE_PCA`, `ML_PCA_COMPONENTS`, `ML_PCA_SVD_SOLVER`, `ML_TEST_SIZE`, `ML_SPLIT_STRATEGY`, `DEFAULT_MODEL_ARTIFACTS_DIR` |
| `tests/ml/test_prediction_volume_output_schema.py` | borrar (cubre solo módulos retirados) |
| `tests/ml/test_preprocessing_feature_contract.py` | borrar |
| `tests/desktop/test_results_tab.py` | recortar lo que cubra la pestaña ML |

Antes de borrar cada constante de `config.py`, verificar consumidor por
consumidor. `DEFAULT_OUTPUT_CSV` y `DEFAULT_DB_FILE` **no** se tocan: los usa
`ml/temporal/`, que queda intacto.

### 12.3 Bloque 2 — stack B (solo con la paridad verde)

`ml/trainer.py`, `ml/evaluator.py`, `csv_backfill.py::persist_training_run` y
sus imports en `main.py`.

### 12.4 Consumidores que sobreviven y deben seguir funcionando

`main.py --predict`, `--simulate`, `--only-maps`, `--evaluate-hydrographs`,
`--analyze-features`, `--evaluate-shapes`, `--evaluate-generalization`,
`ml/predict.py`, `ml/scenario_predict.py`, `ml/feature_analysis.py`,
`ml/feature_importance.py`, `validation/hydrograph_batch.py`. Todos leen
`outputs/models/{classifier,regressor}.joblib`, que `--bench-promote` sigue
produciendo en el mismo formato. **Ninguno se modifica**, salvo el cambio de
import de §12.1.

## 13. Puertas abiertas

Ninguna de las dos añade código especulativo. Ambas son contratos que el banco
necesita de todos modos.

### 13.1 Modelos temporales (LSTM/CNN)

Lo que un modelo temporal futuro necesitará, y que esta spec deja listo:

1. `keys` con `(run_id, node_pk, node_id, factor_mult, shape_id)` — para unir
   sus secuencias a las mismas muestras.
2. `folds.parquet` — asignación literal de folds, para entrenar sobre
   exactamente el mismo train de cada fold.
3. `score_predictions(...)` — para que sus métricas caigan en las mismas tablas,
   con el mismo cálculo, y con el `prep_id` verificado.

Lo que **no** se hace: ningún módulo temporal, ningún adaptador vacío, ninguna
clase base abstracta, ningún cambio a `ml/temporal/`. La tabla
`node_timeseries` sigue sin poblarse; llenarla es el bloqueador real de esa
fase y tendrá su propia spec.

### 13.2 Búsqueda de hiperparámetros (Optuna)

`evaluate_candidate(prepared, family, params) -> metrics` es una función pura de
`params`: es literalmente la firma de un `objective` de Optuna.

Para evitar el sesgo de selección, el constructor de folds de `preprocess.py`
queda parametrizado de forma que pueda emitir **folds internos** sobre el train
de cada fold externo. La búsqueda se hará sobre los internos y el número
reportado saldrá de los externos.

No se añade `optuna` a `requirements.txt` ni se escribe `bench/tune.py` en esta
fase. Ver `docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`.

## 14. Estrategia de pruebas

Todo por TDD: test primero, en cada etapa.

### 14.1 Tests estructurales (los que protegen el diseño)

1. **`preprocess.py` no importa `sklearn` ni `torch`** — inspección del AST del
   módulo. Es el guardián contra la fuga de datos.
2. **Determinismo del `prep_id`** — dos llamadas sobre los mismos datos dan el
   mismo hash; cambiar `run_ids` o el umbral lo cambia.
3. **Rechazo por proveniencia** — `score_predictions` rechaza predicciones cuyo
   `prep_id` no coincida.
4. **Cobertura de folds** — la unión de las filas de test de un protocolo es
   exactamente el dataset, sin solapamientos; ningún `factor_mult` aparece a la
   vez en train y test del mismo fold.
5. **Ajuste dentro del fold** — el imputador/escalador de un fold se ajusta con
   estadísticos del train de ese fold, no del dataset completo (verificado sobre
   un dataset sintético con distribuciones deliberadamente distintas).

### 14.2 Test de paridad (desechable por diseño)

Corre el candidato `xgboost` del banco y `evaluator.evaluate_models()` sobre el
**mismo** frame, y exige coincidencia de las métricas de los tres niveles con
`rtol=1e-6`, en LOSO y en GroupKFold5.

Debe coincidir salvo ruido de punto flotante: mismos folds, misma semilla, mismo
`SimpleImputer(median)` + XGBoost sin escalador. Si no coincide, hay una
diferencia real no identificada y **no se borra nada** hasta entenderla.

Se elimina junto al stack B, en el mismo commit. Su función es justificar el
borrado, no vivir para siempre.

### 14.3 Tests de persistencia

- La cadena completa se escribe en el orden correcto y `training_runs` termina
  en `COMPLETE`.
- Cada trigger de coherencia dispara cuando debe: evaluación que no coincide con
  su candidato, fold añadido tras finalizar, artefacto con receta distinta.
- Migración 006 aplicada sobre una base con filas existentes las preserva.
- Un candidato con métrica primaria inválida no se promueve y queda registrado
  con su motivo.

### 14.4 Regresión

La suite completa debe seguir verde antes y después de cada bloque de borrado.
Última línea base registrada: `531 passed, 3 deselected` en HEAD `7a5ce36`, vía
`./venv/Scripts/python.exe -m pytest -q`. **El HEAD actual es `2d0cb5c`**, tres
commits más adelante, así que el primer paso del plan de implementación es
**re-medir la línea base** y anotarla; sin ese número no se puede afirmar que un
borrado no rompió nada. Ver `COMANDOS.md` para el entorno y un flake
intermitente preexistente en `tests/desktop/test_results_tab.py`.

## 15. Documentación

- **Nuevo** `docs/ML_BENCH.md`: qué hace cada etapa, contratos exactos, la
  garantía anti-fuga y por qué está donde está, el informe de calidad, cómo
  añadir una familia nueva, cómo enganchar un modelo temporal (§13.1), y cómo
  se conectará Optuna (§13.2).
- **Actualizar** `docs/FLUJO_ACTUAL.md` §6, §7, §9, §10, §11 y la tabla de §12.4.
- **Actualizar** `COMANDOS.md` con los subcomandos del banco.
- **Marcar como histórico**, sin borrar (probablemente citados en la tesis):
  `MODEL_COMPARISON_AND_PREPROCESSING_HISTORY.md` y
  `XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md`, con una nota al inicio: "describe
  código retirado el 2026-09-05; conservado como referencia histórica".
- **Actualizar el encabezado** de
  `2026-08-21-sqlite-v17-pipeline-consolidation-design.md` para apuntar a esta
  spec como su continuación aprobada.

## 16. Riesgos

| Riesgo | Mitigación |
|---|---|
| La cadena de proveniencia completa es el mayor bloque de trabajo y sus triggers son estrictos | Los triggers fallan ruidosamente, no en silencio; se implementa con tests por trigger antes que el banco completo |
| La paridad no da exacta por una diferencia no identificada en el evaluador actual | Es la puerta que bloquea el borrado; si no pasa, se investiga en vez de ajustar la tolerancia |
| El MLP requiere una dependencia o un wrapper que no encaja limpiamente en `Pipeline` | Preferencia por wrapper propio sin dependencia nueva; el MLP es una familia más, y si se retrasa no bloquea al resto del banco |
| Borrar la pestaña ML de la GUI rompe tests de escritorio no relacionados | El bloque 1 se hace en commits pequeños con la suite verde entre cada uno |
| Volumen de `oof_predictions` con N familias × 2 targets × folds de LOSO | `executemany` en una transacción; medir con `pytest -m scale` antes de cerrar |

## 17. Listo cuando

- [ ] `preprocess.py` produce un `PreparedDataset` determinista con informe de
      calidad, y el test estructural confirma que no importa `sklearn` ni `torch`.
- [ ] Las cinco familias corren de punta a punta y producen un ranking.
- [ ] El test de paridad está verde en LOSO y GroupKFold5.
- [ ] La cadena de proveniencia se escribe completa y `training_runs` termina en
      `COMPLETE`; la migración 006 aplicada y probada.
- [ ] `--bench-promote` deja `outputs/models/{classifier,regressor}.joblib` con
      el formato actual, y `--predict` / `--only-maps` / `--evaluate-hydrographs`
      funcionan sin cambios.
- [ ] `ml/train.py`, `ml/preprocessing.py`, `ml/predict_tabular.py`,
      `ml/predict_from_inp.py`, `ml/trainer.py`, `ml/evaluator.py` y la parte ML
      de la GUI están borrados; ningún import roto.
- [ ] `pytest -q` verde (línea base 531 passed) y `pytest -m scale` verde.
- [ ] `docs/ML_BENCH.md` escrito; `FLUJO_ACTUAL.md` y `COMANDOS.md` actualizados.
