# Anexo 02 — ML y protocolo de evaluación

**Auditoría:** proyecto de tesis `herramienta-ia-model` (modelo sustituto ML de inundación nodal sobre SWMM)
**Rama / commit base:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b`, con cambios sin commitear de la "Fase A" de otra tarea (no se tocaron)
**Fecha:** 2026-10-10
**Área:** fugas de datos, protocolos de validación (LOSO / GroupKFold5 / LeaveOneShapeOut), cascada clasificador→regresor, métricas y su agregación, umbral de decisión, incertidumbre, reproducibilidad, promoción de modelos, contratos train/predict y cobertura de tests de `tests/ml`.

> Convención de severidad: 🔴 CRÍTICO (compromete la validez científica) · 🟠 MAYOR (sesga resultados) · 🟡 MODERADO (limitación o riesgo) · 🔵 MENOR (calidad).
> Todo número de este anexo sale de un script ejecutado en esta auditoría (sección 5) o de un archivo leído. Lo no comprobado se marca **no verificado**.

---

## 1. Alcance y método

### 1.1 Código revisado

| Módulo | Revisión |
|---|---|
| `swmm_resilience/ml/bench/` (schemas, folds, preprocess, quality, train, evaluate, metrics, ranking, reports, promote, registry, param_filter, `models/*`) | Línea a línea, en el árbol de trabajo actual. Es decir, **con** los cambios de Fase A sin commitear en `folds.py`, `preprocess.py`, `quality.py` y `schemas.py`. |
| `swmm_resilience/ml/` (trainer, evaluator, preprocessing, contracts, predict, scenario_predict, feature_analysis, feature_importance, predict_tabular) | Línea a línea. |
| `swmm_resilience/ml/train.py` (stack *legacy* de escritorio, 1.116 líneas) | Solo las partes de protocolo: features, split, PCA, CV. |
| `swmm_resilience/ml/temporal/` (dataset, train_surrogate, train_cnn, predict, compare_surrogate) | Protocolo, features, escaladores y selección de modelo. No hay datos ni artefactos temporales en el árbol (se verificó que `data/training/swmm_resilience.db` no existe), así que no se pudo contrastar ningún resultado. |
| `main.py` (`_run_bench`, `--only-ml`, `--evaluate-*`, `--analyze-features`), `swmm_resilience/config.py`, bloque `bench:` de `config.yaml` | Cableado del CLI, promoción y escritura en `outputs/models`. |
| `tests/ml/**` y `tests/conftest.py` (fixtures `csv_shaped_dataset*`) | Se juzgó si cada test fallaría de verdad si se rompe el comportamiento. |
| `swmm_resilience/database/csv_backfill.py::persist_training_run` | Solo como fuente de las OOF que ya existen en SQL (sección 4). |

### 1.2 Diagnósticos empíricos (todos de solo lectura)

- Datos: `outputs/bench/prepared/af3751551ca5df98` es el PreparedDataset actual: 160.000 filas, 1.000 corridas, 160 nodos, 40 formas, 25 factores y los 3 protocolos. `4a48359e7e5b53c6` es el anterior (76.000 filas, 19 formas) y es el que produjo los reportes de `outputs/bench/reports/`. Ambos se leyeron directamente de los parquet.
- SQLite `outputs/training_v17.sqlite3`: se abrió solo con `file:…?mode=ro` (tabla `oof_predictions`).
- Se **regeneraron las predicciones OOF de XGBoost** con la función del proyecto `evaluate_candidate()` y los hiperparámetros de `config.yaml`, sobre `af3751551ca5df98` y los 3 protocolos (70 folds, 181,7 s). Es una sola familia, no el benchmark completo. Las OOF quedaron en el scratchpad y no en `outputs/`.
- Controles pequeños: dos baselines sin aprendizaje (tabla de consulta), un predictor perfecto puntuado con `score_predictions()`, coherencia `predict` frente a `predict_proba` por familia (submuestras de 4.000/3.000 filas) y un control de capacidad del regresor (solo filas inundadas, 5 folds).
- Tests: `pytest tests/ml/bench/{test_metrics,test_score_predictions,test_evaluate_candidate,test_parity_with_legacy_evaluator,test_folds}.py --basetemp=<scratchpad>` dio **51 passed**. No se ejecutó la suite completa.
- No se modificó código, `config.yaml`, `outputs/` ni git. Los scripts están en el scratchpad con prefijo `m_` y se reproducen en la sección 5.

### 1.3 Cómo leer los protocolos (verificado)

| Protocolo | Columna de grupo (`schemas.py:56-60`) | Folds reales | Qué mide realmente |
|---|---|---|---|
| `LOSO` | `factor_mult` | 25 × 6.400 filas. 5 folds (factores 0,2–1,0) **sin ninguna fila inundada** | Interpolación en el factor: el mismo nodo y la misma forma están en train a ±0,2 |
| `GroupKFold5` | `factor_mult` | 5 × 32.000, factores intercalados (p. ej. {1,0, 2,0, 3,0, 4,0, 5,0}) | Lo mismo que LOSO, con 5 factores retenidos por fold |
| `LeaveOneShapeOut` (solo Fase A, sin commitear) | `shape_id` | 40 × 4.000 | Generalización a una forma nueva, con todos los factores y nodos vistos |

En ningún protocolo se retienen nodos ni la combinación forma nueva × factor nuevo. Toda afirmación de generalización queda acotada a **esta red y estos 160 nodos**.

---

## 2. Hallazgos

### 2.1 Tabla resumen

| ID | Sev. | Archivo:línea | Hallazgo | Impacto cuantificado |
|---|---|---|---|---|
| M-01 | 🔴 | `ml/bench/schemas.py:56-58`, `ml/evaluator.py:58,147`, `config.yaml:48` | LOSO/GroupKFold5 por factor solo miden interpolación en una malla densa, y el banco no tiene baselines. Una tabla de consulta sin aprendizaje supera a XGBoost | LOSO: el baseline obtiene RMSE e2e pooled **9,0 m³** frente a **68,4 m³** de XGBoost, y F1 0,972 frente a 0,966 |
| M-02 | 🔴 | `ml/contracts.py:46-49` | Las features de forma (solo `duracion_horas`, `tiempo_al_pico_h`) no representan el volumen del hidrograma. Con formas no vistas, el volumen falla sistemáticamente | LeaveOneShapeOut: NSE oracle 0,814 (frente a 0,953 en LOSO); error de volumen total por forma de −63 % a +216 %; una sola forma concentra el 61 % del SSE |
| M-03 | 🟠 | `ml/bench/metrics.py:41-43`, `evaluate.py:94,119`, `evaluator.py:88,127` | F1/precisión/recall se promedian por fold con `zero_division=0`, de modo que los folds sin positivos cuentan como 0 | En LOSO se reporta F1 0,7408 cuando el pooled es 0,9656. Un predictor **perfecto** obtiene F1 = 0,80 |
| M-04 | 🟠 | `evaluate.py:121`, `ranking.py:97`, `main.py:207` | La métrica de ranking es la media no ponderada del RMSE por fold, sin IC ni comparación pareada | LeaveOneShapeOut: media 79,7 frente a pooled 136,1 m³; IC95 % por grupo del RMSE pooled [66,7; 212,2] |
| M-05 | 🟠 | `ml/bench/train.py:85`, `evaluate.py:170`, `config.yaml:62-63` | El regresor está subajustado y además tiene el sesgo de retransformación `expm1(log1p)` | RMSE in-sample 118 m³ ≈ OOF 127 m³; volumen total sistemáticamente −6 %; volúmenes pequeños +35 %/+49 % |
| M-06 | 🟠 | `config.yaml:56-90`, `models/random_forest_family.py:15` | Las familias se comparan con hiperparámetros fijos y heredados del ganador previo, sin afinar | RF `max_depth=6`: NSE oracle 0,45 (19 formas). El ranking compara configuraciones, no métodos |
| M-07 | 🟠 | `main.py:1082`, `ml/trainer.py:78-80`, `bench/promote.py:39-41` | `outputs/models` se sobrescribe en silencio con el pipeline por defecto, y `promotion.json` queda falso | El SHA del `classifier.joblib` real (`660be918…`) ≠ el de `promotion.json` (`e2e7925e…`); el modelo en uso se entrenó con 40 formas y el promovido era de 19 |
| M-08 | 🟡 | `main.py:198`, `bench/reports.py:54-75` | El banco no persiste las OOF. Los reportes no traen proveniencia, y el ranking se sobrescribe parcialmente | El `ranking.csv` actual tiene solo `svm` (rank 1 de 1); los `metrics_*.json` vienen de corridas distintas y no llevan `prep_id` |
| M-09 | 🟡 | `evaluate.py:163`, `models/svm_family.py:30-32`, `bench/train.py:92-110` | El umbral 0,5 es implícito, las probabilidades no están calibradas por la reponderación y en SVM `predict` ≠ `predict_proba` | SVM: 2,5 % de desacuerdo; en el bin 0,4–0,5 la tasa real es 0,18–0,27 |
| M-10 | 🟡 | `ml/bench/metrics.py:36-70` | Faltan métricas: matriz de confusión, especificidad, MCC/BA, PR-AUC, MAE e2e, error relativo de volumen y desglose por forma | La AUC de LOSO promedia 20 folds mientras que F1 promedia 25 |
| M-11 | 🟡 | `evaluate.py:82-123` | Pocas unidades independientes (25 factores / 40 formas) y nodos correlacionados, sin incertidumbre | IC95 % por grupo del F1 pooled en LOSO: [0,951; 0,975] |
| M-12 | 🟡 | `ml/feature_analysis.py:109-111,227-228,270-271`, `feature_importance.py:13` | La ablación de las features de forma se hace con LOSO por factor. SHAP e importancias suponen un árbol sin escalador | La ablación no puede medir el aporte de forma; si se promueve SVM/MLP/lineal, `--analyze-features` falla o informa mal |
| M-13 | 🟡 | `main.py:480-555, 557-640, 664-726` | `--evaluate-shapes` evalúa sobre los propios datos de entrenamiento sin rotularlo; `--evaluate-generalization` es interpolación | Las F1/NSE impresas son de entrenamiento (25 factores × formas entrenadas) |
| M-14 | 🟡 | `models/*_family.py` (`SimpleImputer(strategy="median")`) | La imputación por mediana de faltantes estructurales asigna capacidad aguas arriba a nodos de cabecera | 57.000 filas (35,6 %): `upstream_capacity_lps` pasa de NaN (=0 físico) a 110,6 L/s |
| M-15 | 🟡 | `docs/superpowers/specs/2026-09-29-optuna-…-design.md` §3-4 | El tuning planificado no está anidado: GroupKFold por forma sobre todo el dataset y el reporte después se hace sobre los mismos datos | Riesgo de sobreajuste al CV (no implementado en esta rama) |
| M-16 | 🟡 | `ml/temporal/dataset.py:34`, `temporal/predict.py:493,546`, `train_cnn.py:215`, `train_surrogate.py:165,291` | Surrogate temporal: usa como entrada una salida de SWMM, hay desajuste train/inferencia, selecciona el modelo con el fold de validación y la pérdida final ≠ la de CV | No cuantificable (no hay datos ni artefactos en el árbol) |
| M-17 | 🔵 | `tests/ml/bench/test_evaluate_candidate.py:57,79`, `test_score_predictions.py:87`, `test_parity_with_legacy_evaluator.py` | Tests que no fallarían si se rompe el comportamiento crítico | Ver detalle |
| M-18 | 🔵 | `XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md:569`, `config.yaml:61`, `evaluate.py:106` | El nombre "LOSO" es ambiguo ("Scenario" ≠ factor); hay un comentario obsoleto y una clave `f"{factor:.2f}"` | Riesgo de mala lectura en la tesis |
| M-19 | 🔵 | `extraction/labels.py:85`, `quality.py` (commit) | El volumen está cuantizado a 1 m³ y el umbral 1,0 m³ coincide con la resolución | 1.902 filas valen exactamente 1 m³; la mediana del volumen de los FN es 1 m³ |
| M-20 | 🔵 | `ml/predict.py:72`, `ml/scenario_predict.py:136` | En inferencia no se valida el contrato de features, y la bandera de extrapolación ignora la forma | Un NaN en una feature obligatoria se imputaría en silencio |
| M-21 | 🔵 | `database/csv_backfill.py:248,295`; `ml/train.py` | La evidencia SQL guarda `hyperparameters_json='{}'`; el stack legacy de escritorio usa otro protocolo (PCA, `run_id`) | Proveniencia incompleta |
| M-22 | 🔵 | `models/mlp_family.py:68` | `np.random.seed` global como efecto lateral; el determinismo de torch no está forzado | No verificado empíricamente |

**Conteo:** 🔴 2 · 🟠 5 · 🟡 9 · 🔵 6 (22 hallazgos).

---

### 2.2 Detalle de hallazgos

#### M-01 🔴 — Los protocolos por factor miden interpolación trivial y no hay baselines

- **Dónde:** `PROTOCOL_GROUP_COLUMNS` (`ml/bench/schemas.py:56-58`) agrupa `LOSO` y `GroupKFold5` por `factor_mult`, igual que el legacy (`ml/evaluator.py:58,147`). Las features dinámicas son exactamente `base × factor`: `max|q_pico_nodo − base_inflow_lps·factor| = 1,4e-14` (m_01). Al retener un factor, el mismo nodo con la misma forma está en train a ±0,2.
- **Evidencia (m_06, sobre el PreparedDataset actual):** el baseline toma, para cada (nodo, forma), el volumen interpolado linealmente entre los factores vecinos. No aprende nada.

  | LOSO (factor) | F1 pooled | Recall | RMSE e2e pooled (m³) | Error vol. total |
  |---|---|---|---|---|
  | Baseline de interpolación | **0,9718** | 0,9909 | **9,00** | −0,39 % |
  | XGBoost (`config.yaml`) | 0,9656 | 0,9808 | 68,43 | −6,75 % |

  En los factores interiores, el RMSE es 4,88 (baseline) frente a 64,99 (XGBoost). En LeaveOneShapeOut, el baseline "forma más cercana en (dur, t_pico), mismo nodo y factor" obtiene F1 0,9758 frente a 0,9789 y RMSE pooled 170,2 frente a 136,1. En la media por forma es 98,2 frente a 79,7, y el baseline gana en **20 de 40 formas**.
- **Impacto:** LOSO y GroupKFold5 no aportan evidencia de generalización. Un buen número en LOSO es compatible con un modelo peor que una tabla. En formas no vistas, el ML aporta una mejora modesta (≈20 % de RMSE medio) sobre el vecino más cercano. La tesis no puede afirmar el valor del sustituto sin estos baselines.
- **Recomendación:** (1) añadir al banco una "familia" baseline que no aprenda (interpolación por factor, vecino de forma más cercana y "siempre 0") y que pase por `score_predictions()`; (2) declarar LOSO/GroupKFold5 como **interpolación** y usar LeaveOneShapeOut, junto con un GroupKFold por forma más exigente, como protocolo principal; (3) añadir un protocolo combinado (forma × factor retenidos, o bloques de factores extremos) para medir extrapolación de verdad.

#### M-02 🔴 — Generalización a formas nuevas: la representación de la forma es insuficiente

- **Dónde:** `ml/contracts.py:46-49`. La forma entra solo por `duracion_horas` y `tiempo_al_pico_h`. El pico ya está normalizado, así que `q_pico_*` no cambia entre formas. Ninguna feature lleva el **volumen o área** del hidrograma.
- **Evidencia (OOF regenerados, m_02/m_03/m_04/m_05/m_07):**
  - Regresor oracle pooled: NSE **0,814** en LeaveOneShapeOut frente a 0,953 en LOSO; RMSE 253 frente a 127 m³; MAE 100,8 frente a 49,6 m³.
  - Error de volumen total por forma: de **−63,3 %** (`shape_quasi_uniform_block`) a **+216,3 %** (`common_bogota`). `shape_quasi_uniform_block` aporta por sí sola el **61,35 %** del SSE end-to-end.
  - El volumen total real por forma correlaciona con el área del hidrograma normalizado (Spearman **0,976**). El error por forma correlaciona con el factor de llenado `área/duración` (Spearman **−0,577**): las formas "en bloque" se subestiman y las "puntiagudas" se sobreestiman.
  - Control de capacidad (solo filas inundadas, GroupKFold5): un regresor de mayor capacidad baja el RMSE de 127,5 a 57,5 m³ cuando los grupos son factores, pero de 327,7 a 317,0 m³ cuando son formas (NSE 0,69 → 0,71). **Más capacidad no resuelve el problema con formas no vistas**: el cuello de botella es la representación.
  - Con 8 formas retenidas por fold (GroupKFold por forma), el NSE oracle cae a 0,69. LeaveOneShapeOut, con 39 formas en train, es el extremo optimista.
- **Impacto:** cualquier afirmación sobre hidrogramas nuevos (el objetivo de la validación de hidrogramas) queda sin respaldo en volumen. La clasificación sí generaliza bien (F1 pooled 0,979).
- **Recomendación:** implementar y evaluar con ablación **bajo LeaveOneShapeOut** las variables tributarias de volumen e integral ya propuestas en `docs/justificacion-features-tributarias-tabular-v4-21.md` (sin implementar). Reportar los errores por forma y tratar las formas fuera del rango (dur, t_pico, área) como extrapolación.

#### M-03 🟠 — Agregación de F1/precisión/recall con `zero_division=0` por fold

- **Dónde:** `ml/bench/metrics.py:41-43` (`zero_division=0`), `evaluate.py:94,119` (media por fold) y la misma lógica en `ml/evaluator.py:88,127`. Afecta también a `by_factor` (`evaluate.py:104-116`, `evaluator.py:122`).
- **Evidencia:**
  - m_08: un **predictor perfecto** puntuado con `score_predictions()` sobre el PreparedDataset real obtiene, en LOSO, `precision = recall = f1 = 0,80` (AUC 1,0, RMSE 0). Además, `by_factor["0.20"].f1 = 0,0` en **los tres protocolos**.
  - m_03 (XGBoost): en LOSO, F1 reportada 0,7408, F1 media en los 20 folds con positivos 0,9260 y F1 **pooled 0,9656**. La precisión reportada es 0,7226 y la pooled 0,9508. Los 5 folds sin positivos tienen 0 predicciones positivas: allí el clasificador acierta y aun así puntúa 0.
  - La "brecha" LOSO frente a GroupKFold5 de los reportes (0,727 frente a 0,963 en `outputs/bench/reports/metrics_xgboost.json`) es un **artefacto**: pooled da 0,9656 frente a 0,9659. El mismo valor 0,7408 aparece en `outputs/metrics/metrics_classifier.json` (legacy, 2026-10-03).
- **Impacto:** subestima F1, precisión y recall de LOSO en ≈0,22 y sugiere una diferencia entre protocolos que no existe. Afecta a todas las familias por igual (el orden del ranking cambia poco), pero los valores absolutos que se reporten son erróneos.
- **Recomendación:** calcular la matriz de confusión pooled (suma de TN/FP/FN/TP por protocolo) como métrica principal. Por fold, usar `zero_division=np.nan` y reportar cuántos folds tienen la métrica definida. En `by_factor`, devolver NaN donde no hay positivos.

#### M-04 🟠 — Métrica de ranking: media no ponderada por fold, sin incertidumbre ni prueba pareada

- **Dónde:** `evaluate.py:121` (media de `rmse_vol_todos_nodos` por fold), `ranking.py:97` (ordena por valor puntual), `main.py:207` (protocolo = `protocols[0]`, hoy `LeaveOneShapeOut` en Fase A).
- **Evidencia (m_03):** en LeaveOneShapeOut, la media por fold es 79,7 m³ y el pooled 136,1 m³ (−41 %). En LOSO, la media es 55,3 (con 5 folds de RMSE 0) y el pooled 68,4. El SSE e2e sale en un ~100 % de filas TP (los errores del clasificador no pesan en volumen) y lo domina una forma (61 %). El IC95 % bootstrap por grupo del RMSE pooled en LeaveOneShapeOut es **[66,7; 212,2]**. En los reportes antiguos, `svm` gana a `xgboost` por 37,70 frente a 39,97 (≈6 %), una diferencia muy dentro de esa dispersión.
- **Impacto:** el ganador depende de cómo se agregue y de una o dos formas, y el ranking no sabe distinguir un empate de una diferencia real.
- **Recomendación:** declarar la agregación (media por grupo para "forma nueva típica" o pooled para "volumen total"), reportar ambas con IC bootstrap por grupo y decidir entre familias con diferencias **pareadas por fold** (bootstrap de la diferencia o Wilcoxon). Añadir MAE y error relativo de volumen.

#### M-05 🟠 — Regresor subajustado y sesgo de retransformación log

- **Dónde:** `ml/bench/train.py:85` y `evaluate.py:169-170` (MSE en `log1p` y vuelta con `expm1` sin corrección). Hiperparámetros en `config.yaml:62-63`.
- **Evidencia:** el RMSE **in-sample** (entrenamiento) del regresor de `config.yaml` es 118,0 m³ con NSE 0,960, prácticamente igual al OOF de LOSO (127,2 m³): el error lo domina el sesgo y no la brecha de generalización. Un regresor de mayor capacidad baja a 16,7 m³ in-sample y a 57,5 m³ fuera de muestra por factor (m_07). `Σpred/Σreal` = 0,956 in-sample y 0,93–0,94 en los tres protocolos, de modo que el **volumen total se subestima en ≈6 %** de forma sistemática. Por cuantiles: los volúmenes ≤24 m³ se sobreestiman en +35 % (LOSO) y +49 % (LeaveOneShapeOut), y los >822 m³ se subestiman entre −8 % y −16 %.
- **Impacto:** las métricas de volumen y la curva de volumen total heredan un sesgo conocido y corregible.
- **Recomendación:** aplicar una corrección de retransformación (smearing de Duan, estimada **dentro del fold**) o un objetivo en escala original (Tweedie/Poisson). Afinar la capacidad con CV anidado (ver M-15) y reportar Σpred/Σreal.

#### M-06 🟠 — Comparación de familias con hiperparámetros fijos heredados

- **Dónde:** `config.yaml:56-90`. Los de XGBoost son "heredados del pipeline validado"; RF usa `max_depth=6` "para aislar el método" (`random_forest_family.py:15`). Los comentarios de MLP dicen "Pendiente de afinar".
- **Evidencia:** con los reportes de 19 formas, RF obtiene NSE oracle 0,446/0,449 y RMSE e2e 155–201, frente a 0,965–0,977 y 37–47 de XGB/SVM. Un RF con profundidad 6 está fuertemente restringido, a diferencia de un boosting de profundidad 6, así que igualar la profundidad **no** aísla el método.
- **Impacto:** el ranking compara configuraciones arbitrarias. Elegir la familia antes de afinarla (el orden que fija el spec Optuna) favorece a la familia cuya configuración se heredó.
- **Recomendación:** dar a cada familia un presupuesto de tuning equivalente dentro de un CV anidado antes de comparar, o declarar el benchmark fijo como "comparación de configuraciones por defecto".

#### M-07 🟠 — `outputs/models` se sobrescribe en silencio y la proveniencia de la promoción queda falsa

- **Dónde:** el pipeline por defecto (`python main.py`, `--only-ml`, `--skip-extraction`) llama a `train_models(df, config, MODELS_DIR)` (`main.py:1082`), que escribe `classifier.joblib`, `regressor.joblib` y `training_inp_hash.txt` (`ml/trainer.py:78-80`) sin tocar `promotion.json`. Por su parte, `promote_candidate` escribe `training_inp_hash.txt` con el MD5 del `.inp` **actual** de `config.yaml`, no con el del entrenamiento (`bench/promote.py:41`). El manifiesto del PreparedDataset tampoco guarda el hash del `.inp`.
- **Evidencia (leída en disco):** `outputs/models/promotion.json` declara `family: xgboost`, `prep_id: 4a48359e7e5b53c6` (19 formas) y `classifier_sha256: e2e7925e…` (2026-09-07). El `classifier.joblib` presente tiene SHA-256 `660be918a9681677…` (2026-10-03) y `scale_pos_weight = 2,4600579559707625 = 113.758/46.242`, que es el balance del dataset de 40 formas. Es decir, lo reentrenó el pipeline legacy.
- **Impacto:** mapas, `--evaluate-*`, SHAP y curvas usan un modelo que no es el candidato del banco al que se atribuyen las métricas. `promotion.json` afirma lo contrario. La protección por hash de `predict_network` certifica el `.inp` presente al promover, no el de entrenamiento.
- **Recomendación:** que `train_models` se niegue a escribir en `outputs/models` cuando exista `promotion.json`, o que lo invalide. Verificar al cargar los SHA de `promotion.json` contra los binarios. Guardar en la metadata del candidato el hash del `.inp` con que se generaron los datos.

#### M-08 🟡 — El banco no persiste OOF y los reportes no traen proveniencia

- **Dónde:** `main.py:198` descarta el frame OOF (`_, metrics = evaluate_candidate(...)`). `reports.py:54-75` escribe `metrics_<family>.json` sin `prep_id`, hiperparámetros, fecha ni commit, y sobrescribe `ranking.*` solo con las familias de la corrida en curso.
- **Evidencia:** el `outputs/bench/reports/ranking.csv` actual contiene **una sola fila** (`svm`, rank 1, 2026-09-07 02:16), mientras que `metrics_xgboost/random_forest/mlp.json` son de otras corridas (20:28 y 23:03). Todos corresponden al PreparedDataset antiguo de 19 formas y no hay reporte para el actual (40 formas, con LeaveOneShapeOut).
- **Recomendación:** persistir `oof_<family>_<prep_id>.parquet` e incluir en cada JSON `prep_id`, hiperparámetros, versión del código y fecha. Que el ranking lea todas las métricas del mismo `prep_id` y nunca mezcle corridas.

#### M-09 🟡 — Umbral de decisión implícito y probabilidades sin calibrar

- **Dónde:** `evaluate.py:163` usa `classifier.predict` (umbral 0,5 fijo de cada estimador) y la inferencia hace lo mismo (`predict.py:73`, `scenario_predict.py:140`). La metadata del candidato no guarda el umbral (`bench/train.py:92-110`). SVC usa `probability=True` (`svm_family.py:32`).
- **Evidencia:** en SVM, `predict` ≠ `predict_proba ≥ 0,5` en **75 de 3.000** filas (2,5 %); en XGB/RF/linear/MLP hay 0 desacuerdos (control con 4.000 filas de entrenamiento). La reponderación (`scale_pos_weight` ≈ 2,46) descalibra: en XGBoost LOSO, el bin de probabilidad 0,4–0,5 tiene una tasa real de 0,27 y en LeaveOneShapeOut de 0,18. En el linear, la probabilidad media es 0,356 frente a una tasa real de 0,299. El barrido de umbral en LeaveOneShapeOut da un F1 pooled máximo ≈0,982 en 0,6 (0,979 en 0,5) y un recall de 0,996 en 0,3.
- **Impacto:** la F1 de SVM y su AUC corresponden a reglas de decisión distintas. El umbral es una decisión de diseño (costo FN frente a FP) que hoy no se toma explícitamente.
- **Recomendación:** derivar `y_pred = y_prob ≥ t` en todas las familias, guardar `t` en la metadata y elegirlo en un CV interno (nunca sobre las OOF de reporte). Si las probabilidades se van a interpretar, calibrarlas dentro del fold.

#### M-10 🟡 — Métricas incompletas para un problema desbalanceado con cascada

- **Dónde:** `ml/bench/metrics.py:36-70`.
- **Evidencia:** solo se reportan precisión, recall, F1 y AUC de la clase positiva. Falta la matriz de confusión, la especificidad, MCC o accuracy balanceada, PR-AUC, el MAE end-to-end, el error relativo del volumen total y el desglose por forma (solo existe `by_factor`). `pct_nodos_correctos` (0,98) lo dominan los negativos (71 %). En LOSO, la AUC media excluye los 5 folds sin positivos (NaN), mientras que F1 los incluye como 0: las dos métricas se promedian sobre poblaciones distintas.
- **Recomendación:** ver sección 4. Añadir `by_shape` y MAE/sesgo relativo de volumen.

#### M-11 🟡 — Pocas unidades independientes y nada de incertidumbre

- **Evidencia:** las unidades independientes son 25 factores y 40 formas. Las 160 filas de cada corrida son nodos de la misma red, hidráulicamente acoplados. Los IC95 % bootstrap por grupo (m_03) son: F1 pooled en LOSO [0,951; 0,975], en GroupKFold5 [0,956; 0,974] y en LeaveOneShapeOut [0,970; 0,986]; RMSE pooled en LOSO [52,2; 82,0] y en LeaveOneShapeOut [66,7; 212,2]. La F1 por fold va de 0,466 (factor 1,2) a 0,994.
- **Recomendación:** reportar siempre los IC por grupo (no por fila) y el número de grupos. No interpretar 160.000 filas como 160.000 observaciones independientes.

#### M-12 🟡 — La ablación de forma usa el protocolo equivocado; SHAP e importancias suponen un árbol

- **Dónde:** `ml/feature_analysis.py:109-111` hace la ablación de `duracion_horas`/`tiempo_al_pico_h` con `LeaveOneGroupOut` por `factor_mult` y `np.nanmean` de F1 con `zero_division=0`, y usa `make_classifier` legacy en lugar de la familia del banco. En `:227-228,270-271`, SHAP aplica `named_steps["imputer"]` + `TreeExplainer(named_steps["model"])`. `feature_importance.py:13` usa `model.feature_importances_`.
- **Impacto:** con la forma siempre vista en train, la ablación no puede medir lo que aportan las features de forma a formas nuevas. Si se promueve una familia no arbórea, `--analyze-features` falla (`TreeExplainer` sobre SVC) o calcula SHAP sobre X sin escalar. Esto último es inferencia, **no verificado** en ejecución. Además, el reporte oculta `n_tuberias_in` aunque el modelo lo usa.
- **Recomendación:** hacer la ablación con LeaveOneShapeOut y métricas pooled. Elegir el explainer según la familia, o usar permutation importance sobre las OOF.

#### M-13 🟡 — Modos de "evaluación" que no son fuera de muestra

- **Dónde:** `main.py:480-555` (`--evaluate-shapes`) evalúa el modelo final sobre **las formas de entrenamiento × los 25 factores de entrenamiento** (`factors_to_eval = config.factors()`, `:514`) e imprime F1/NSE sin decir que son métricas de entrenamiento. `main.py:557-640` (`--evaluate-generalization`) usa formas entrenadas en puntos medios (±0,1 de factores vistos), es decir, interpolación (ver M-01). `main.py:664-726` (`--evaluate-validation-shapes`) protege la condición de fuera de muestra solo **por nombre** de forma y rotula "INTERPOLACIÓN/EXTRAPOLACIÓN" solo por la duración.
- **Recomendación:** rotular la salida (in-sample, interpolación o fuera de muestra). En validación, comparar el contenido de la forma (no el nombre) y calcular el régimen con (dur, t_pico, área).

#### M-14 🟡 — Imputación por mediana de faltantes estructurales

- **Evidencia:** `diam_max_in`, `pendiente_max_in` y `upstream_capacity_lps` son NaN **si y solo si** `n_tuberias_in == 0` (57.000 filas, 35,6 %; verificado). `SimpleImputer(median)` les asigna 0,2 m de diámetro y 110,6 L/s de capacidad aguas arriba a nodos de cabecera, cuyo valor físico es 0. Los árboles pueden compensarlo a través de `n_tuberias_in`, pero linear, SVM y MLP reciben un valor físicamente falso. Además, el imputador anula el manejo nativo de NaN de XGBoost.
- **Recomendación:** imputar con constante 0 en capacidad y diámetro, con `add_indicator=True`, o documentar la decisión. Verificar el efecto con ablación.

#### M-15 🟡 — El tuning planificado (Optuna) no está anidado

- **Dónde:** el spec `2026-09-29-optuna-tuning-familia-ganadora-design.md` §3-4 plantea GroupKFold(5) por `shape_id` sobre **todo** el PreparedDataset, sin holdout, después de elegir familia con el benchmark fijo. No está implementado en esta rama (solo existe en un worktree aparte). El propio spec etiqueta el resultado como "exploratorio".
- **Riesgo:** si los hiperparámetros afinados se reportan después con LeaveOneShapeOut sobre los mismos datos, la estimación es optimista (sobreajuste al CV). Con 40 formas y un IC tan ancho como el de M-04, el riesgo es real.
- **Recomendación:** CV anidado, con un bucle externo LeaveOneShapeOut o GroupKFold por forma y un bucle interno por forma sobre el train externo. Como alternativa, reservar formas de holdout que no se toquen hasta el final.

#### M-16 🟡 — Surrogate temporal (línea secundaria, sin datos en el árbol)

- `temporal/dataset.py:34-37` declara `total_inflow_lps` como "PRE_SWMM", pero viene de `node.total_inflow` de pyswmm (`simulation/runner.py:485`), que incluye el caudal ruteado desde aguas arriba: **es una salida de SWMM**. En inferencia se escala linealmente la serie de la corrida de menor multiplicador (`temporal/predict.py:493,546`), algo que no reproduce el ruteo no lineal con sobrecarga e inundación. Esto es un desajuste entre train e inferencia.
- El CV es GroupKFold por `run_id` (`dataset.py:199,326`), con los vecinos de forma y factor en train.
- `train_cnn.py:215` persiste el modelo del **mejor fold** según la pérdida de validación: hay sesgo de selección y el modelo se entrena con menos datos.
- `train_surrogate.py` evalúa en CV con `MSELoss` (`:291`) y entrena el modelo final con `HuberLoss` y sin scheduler (`:165`): el CV no evalúa el modelo desplegado.
- El docstring de `compare_surrogate.py:149` ("no SWMM outputs") es falso por lo dicho del primer punto.
- **Impacto:** no cuantificable (no existen `data/training/swmm_resilience.db` ni artefactos). **No verificado** con datos.

#### M-17 🔵 — Cobertura de tests: lo crítico no fallaría

- `test_evaluate_candidate.py:79` (`…a_fold_without_flooded_training_rows_still_produces_predictions`) **no ejercita** la rama `else` de `evaluate.py:171-180`. Con el fixture (`tests/conftest.py:57-90`: factores 1, 2 y 3, con inundación desde factor ≥2), todo fold LOSO tiene positivos en train. Verificado por lectura.
- `test_score_predictions.py:87` ("perfect predictions score perfectly") usa un fixture en el que todos los folds tienen positivos. Con los datos reales, el mismo predictor perfecto obtiene 0,80 (m_08), y ningún test lo detecta.
- `test_parity_with_legacy_evaluator.py` **consagra** el sesgo de M-03: los dos lados calculan igual la media con ceros. Su fixture LOSO tiene un fold sin positivos (factor 1,0).
- `test_evaluate_candidate.py:57` comprueba solo que el escalador se ajusta con menos filas que el total. No verifica que use exactamente las de train del fold, así que un ajuste sobre el test pasaría.
- No hay tests de umbral ni de calibración, de persistencia de OOF, de coherencia `predict`/`predict_proba`, ni de que `train_models` no pise una promoción.
- **Recomendación:** añadir tests con un fold sin positivos que exijan NaN (no 0) y F1 pooled = 1 para el predictor perfecto, más un espía que compare el conjunto exacto de índices ajustados.

#### M-18 🔵 — Nomenclatura y residuos

- "LOSO" agrupa por factor, pero la documentación dice "Leave One Scenario Out" (`XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md:569`), mientras que en la BD un `scenario` es (forma, factor). Conviene renombrarlo a "LeaveOneFactorOut" en la tesis.
- `config.yaml:61` menciona "175 escenarios", cuando hoy son 1.000 corridas.
- `by_factor` usa `f"{factor:.2f}"` (`evaluate.py:106`), que colisionaría con una malla más fina que 0,01.

#### M-19 🔵 — Resolución del volumen e inundación

- El `.rpt` reporta el volumen en 10⁶ L con 3 decimales (×1000 en `labels.py:85`), así que todos los volúmenes son **enteros en m³** (verificado: 0 valores no enteros). 1.902 filas valen exactamente 1,0 m³, que es el umbral, de modo que `inunda` equivale a "SWMM reportó alguna inundación". La mediana del volumen de los FN es 1 m³ (LOSO): parte del error de clasificación está en el límite de resolución.
- El `quality.py` commiteado usaba `>` y reportaba 1.072 (19 formas) o 1.902 (40 formas) "desacuerdos" falsos. La Fase A (sin commitear) lo corrige con `flood_label` (`>=`). En los datos, `inunda == (vol ≥ 1)` tiene 0 discrepancias.
- **Recomendación:** documentarlo en la tesis y analizar por separado los FN con volumen ≤2 m³.

#### M-20 🔵 — Contrato de features en inferencia

- `predict.py:72` y `scenario_predict.py:136` construyen `X = merged[FEATURE_COLS]` sin pasar por `TABULAR_V3_17.validate_frame` (entrenamiento sí lo hace: `preprocess.py:185`, `trainer.py:64`). Un NaN en una feature obligatoria se imputaría con la mediana en silencio. Las columnas y su orden sí coinciden (misma tupla `FEATURE_COLUMNS_V17`).
- `scenario_predict._extrapolation_flags` marca la extrapolación solo por la razón `q_pico/base`, no por (dur, t_pico) fuera del rango entrenado.

#### M-21 🔵 — Proveniencia en SQL y stack legacy

- `persist_training_run` inserta `hyperparameters_json='{}'` en `training_runs` y `model_evaluations` (`csv_backfill.py:248,295`): la evidencia SQL no dice con qué hiperparámetros se entrenó.
- `ml/train.py` (desktop) usa otro protocolo: `GroupShuffleSplit` por `run_id`, PCA(5) y regresión sobre todas las filas, con las features que queden tras `ML_DROP_COLUMNS` (lista negra, no contrato). Está inactivo porque no hay datos (`dataset_ml.csv` no existe), pero conviene aislarlo o retirarlo para que no se confunda con el banco.

#### M-22 🔵 — Determinismo de la familia MLP

- `mlp_family.py:68` llama a `np.random.seed(...)`, que altera el RNG global como efecto lateral. No se fuerza `torch.use_deterministic_algorithms` ni el número de hilos. **No verificado** si da resultados bit a bit iguales entre máquinas o hilos.

---

## 3. Lo que está bien (verificado)

1. **No hay fuga del target en `tabular_v3_17`.** `q_pico_nodo` y `q_pico_acum_escalado` son exactamente `base × factor` (error máximo 1,4e-14 y 4,5e-13). `duracion_horas` y `tiempo_al_pico_h` salen de la forma, y ninguna feature usa salidas de SWMM (m_01, `extraction/dynamic_features.py:20-25`).
2. **El preprocesamiento se ajusta dentro del fold.** Imputador y escalador viven en el `Pipeline` de cada familia y se ajustan con `X_train` (`evaluate.py:154-169`). `scale_pos_weight` se calcula por fold (`evaluate.py:158-159`). La etapa 1 no importa transformadores (guarda `test_stage1_imports.py`).
3. **Integridad de grupos en los folds.** En los folds reales: LOSO 25×6.400, GroupKFold5 5×32.000 y LeaveOneShapeOut 40×4.000. Cada muestra está en test una sola vez por protocolo, y ningún grupo aparece en train y test a la vez. La validación de Fase A lo exige al cargar (`schemas.py:170-362`). `build_folds` rechaza grupos de la columna equivocada (`folds.py:62-67`).
4. **Datos sin duplicados ni conflictos.** Hay 160.000 vectores X distintos de 160.000, 0 vectores con etiquetas en conflicto, 0 duplicados (run, node) y 0 discrepancias `inunda` frente a `vol ≥ 1`. No existen 2 formas con el mismo par (dur, t_pico).
5. **Paridad y determinismo.** Las métricas LOSO regeneradas con el banco coinciden **dígito a dígito** con `outputs/metrics/*.json` del pipeline legacy del 2026-10-03 (F1 0,74076924794143; RMSE e2e 55,304099293627814; NSE oracle 0,9530489319999711). Se comprueba así la equivalencia banco ≡ legacy sobre datos reales y la reproducibilidad de XGBoost con semilla 42.
6. **La cascada está bien separada.** El regresor se entrena solo con `inunda==1` del train del fold. El nivel oracle usa etiquetas reales y el end-to-end usa las predichas (`evaluate.py:96-102`), y la documentación dice explícitamente que el oracle es una cota optimista. Las predicciones negativas se recortan a 0 y el caso sin positivos en train predice 0 en lugar de fallar.
7. **Proveniencia de datos.** `prep_id` hashea el descriptor completo, incluido el SHA-256 canónico del frame. La captura se hace por `backup` SQLite de solo lectura, y `score_predictions` rechaza OOF de otro `prep_id`.
8. **Promoción (Fase A, sin commitear).** `--bench` ya no promueve y `--bench-promote` exige nombrar la familia (`main.py:213-219`, `config.yaml:55`). Esto cierra la autopromoción del commit base, donde `promote: "auto"` copiaba al ganador del ranking a `outputs/models`.
9. Los reportes JSON se sanean (NaN→null) y el ranking conserva a los candidatos inválidos con su motivo.

---

## 4. Matriz de confusión: estado actual y propuesta

### 4.1 Qué existe hoy

| Fuente | ¿Predicción por fila? | Contenido | Protocolos/familias | ¿Sirve para la matriz? |
|---|---|---|---|---|
| `evaluate_candidate()` → frame OOF (`schemas.py:44-51`: `sample_idx, protocol, fold_id, y_pred_clf, y_prob_clf, y_pred_reg`) | Sí, en memoria | `y_true` se obtiene con `prepared.y_clf[sample_idx]` | Todas las familias y protocolos | **Sí, pero se descarta**: `main.py:198` (`_`). No queda nada en disco |
| `outputs/bench/reports/metrics_*.json` | No | Promedios por fold de P/R/F1/AUC y `by_factor` (F1, RMSE) | 4 familias × {LOSO, GroupKFold5} sobre 19 formas | **No**: no hay conteos TN/FP/FN/TP ni forma de reconstruirlos |
| `outputs/metrics/metrics_*.json` (legacy) | No | Igual, solo LOSO principal | XGBoost | No |
| SQLite `oof_predictions` (`--persist-sql`, `csv_backfill.py:284-370`) | **Sí** | `run_id, node_pk, observed, predicted, probability, fold_id` (160.000 filas de `inunda`; las de volumen sin probabilidad) | **Solo** XGBoost × GroupKFold5 por factor (`training_runs.grouping_strategy='group_kfold'`, 2026-09-15) | Sí, solo para ese protocolo |

Con la tabla SQL (m_09), la matriz pooled de GroupKFold5 es **TN=111.468 · FP=2.290 · FN=920 · TP=45.322** (recall 0,9801, especificidad 0,9799, F1 0,9658). La de las OOF regeneradas (m_03) es TN=111.431 · FP=2.327 · FN=877 · TP=45.365. La pequeña diferencia viene, presumiblemente, del orden de filas frente al `subsample` de XGBoost; esto **no se verificó**.

Matrices pooled de las OOF regeneradas de XGBoost (clase positiva = inunda; m_03):

| Protocolo | TN | FP | FN | TP | Recall | Especif. | MCC | Vol. real en FN |
|---|---|---|---|---|---|---|---|---|
| GroupKFold5 | 111.431 | 2.327 | 877 | 45.365 | 0,9810 | 0,9795 | 0,9520 | 7.368 m³ (0,05 %) |
| LOSO | 111.412 | 2.346 | 890 | 45.352 | 0,9808 | 0,9794 | 0,9515 | 6.211 m³ (0,04 %) |
| LeaveOneShapeOut | 112.153 | 1.605 | 373 | 45.869 | 0,9919 | 0,9859 | 0,9703 | 728 m³ (0,00 %) |

Los FN son inundaciones pequeñas (mediana de 1–2 m³; máximo 327 m³). El peso real de los errores del clasificador sobre el volumen es despreciable frente al del regresor (M-04).

### 4.2 Qué falta

1. Persistir el frame OOF por familia y `prep_id`, añadiendo `y_true_clf`, `y_true_reg`, `factor_mult` y `shape_id` para no depender de un join.
2. Calcular en `score_predictions` los conteos TN/FP/FN/TP por fold y pooled por protocolo, y los desgloses `by_factor` y `by_shape`.
3. Fijar el umbral explícitamente y dejarlo registrado: hoy depende del `predict` de cada estimador (M-09).
4. Exportar la matriz (JSON/CSV) y, opcionalmente, una figura.

### 4.3 Dónde se enchufa

| Paso | Archivo | Cambio propuesto |
|---|---|---|
| Conteos | `ml/bench/metrics.py` | `confusion_counts(y_true, y_pred) -> {tn, fp, fn, tp}`, junto con métricas derivadas (recall, especificidad, precisión, F1, MCC, BA) calculadas **a partir de conteos sumados** |
| Agregación | `ml/bench/evaluate.py::score_predictions` (bucle `:87-116`) | Acumular conteos por fold, sumarlos por protocolo (pooled) y por factor y forma. Mantener las métricas por fold con `NaN` donde no estén definidas |
| Umbral | `ml/bench/evaluate.py:163-164`, `bench/train.py` (metadata) | `y_pred = (y_prob >= t)` con `t` en la config o la metadata. Para SVM elimina la incoherencia `predict`/`predict_proba` |
| Persistencia | `main.py:198`, `ml/bench/reports.py` | Guardar `oof` en `reports/oof_<family>_<prep_id>.parquet` y `confusion_<family>.json/csv` (pooled, por fold, por factor y por forma) |
| Esquema | `ml/bench/schemas.py:44-51` | Conservar `OOF_COLUMNS` como contrato mínimo y añadir columnas opcionales de trazabilidad (`threshold`) sin romper `_validate_oof` |
| Tests | `tests/ml/bench/test_score_predictions.py` | Predictor perfecto con un fold sin positivos → matriz diagonal, F1 pooled 1 y recall por fold NaN (no 0) |

### 4.4 Decisiones pendientes (para el autor)

1. **Clase positiva:** `inunda = 1` (vol ≥ 1 m³, es decir, cualquier inundación reportada; ver M-19). Recall de inundación y especificidad son las métricas a nombrar en la tesis.
2. **Umbral:** mantener 0,5 sobre probabilidades reponderadas o elegir `t` en un CV **interno** según un costo explícito (p. ej. maximizar F1 o fijar recall ≥ 0,99). No elegirlo nunca sobre las OOF de reporte. Para las OOF actuales, el umbral cambia F1 en <0,005 entre 0,4 y 0,7.
3. **Agregación:** pooled (suma de conteos) como cifra principal y dispersión por fold como incertidumbre (IC bootstrap por grupo). Los folds sin positivos aportan TN/FP válidos, pero su recall es indefinido (NaN), no 0.
4. **Normalización:** por fila real (recall/especificidad, la lectura natural de "qué fracción de inundaciones detecto"), además de los conteos crudos. La normalización por columna (precisión) va como complemento.
5. **Estratificación:** una matriz por protocolo y familia como mínimo, más una por factor (los factores ≤1,0 son todo TN/FP) y una por forma en LeaveOneShapeOut (las peores son `shape_ultra_short_impulse`, con FP=288, y `common_bogota`, con FP=178).
6. **Comparabilidad:** antes de comparar familias, unificar la regla `y_prob ≥ t` en todas (SVM).

---

## 5. Scripts reproducibles

Todos se ejecutan desde la raíz del repo con el Python del proyecto. Solo leen `outputs/` y escriben en el scratchpad. `S` es el scratchpad de la sesión (`…\288ddaf2-…\scratchpad`). Los que importan `swmm_resilience` necesitan `PYTHONPATH=.`.

| Script | Qué hace | Hallazgos | Tiempo |
|---|---|---|---|
| `m_01_datos.py <prepared>` | Estructura, consistencia `q = base×factor`, duplicados, umbral, folds sin positivos | M-01, M-03, M-14, M-19, §3 | <1 min |
| `m_02_oof_xgboost.py <prepared> <S> xgboost` | Regenera las OOF con `evaluate_candidate()` y `config.yaml` (3 protocolos) | Base de M-02…M-05, §4 | 182 s |
| `m_03_analisis_oof.py <prepared> <oof>` | Media frente a pooled, matriz de confusión, umbral, calibración, oracle por cuantil, descomposición del SSE, bootstrap por grupo | M-02…M-05, M-09…M-11 | ~1 min |
| `m_04_por_grupo.py <prepared> <oof>` | Matriz y error por forma y por factor | M-02, M-04 | <1 min |
| `m_05_area_forma.py <prepared> <oof>` | Correlación del error por forma con el área del hidrograma | M-02 | <1 min |
| `m_06_baselines.py <prepared> <oof>` | Baselines sin aprendizaje (interpolación de factor; forma más cercana) | M-01 | ~1 min |
| `m_07_capacidad.py <prepared>` | Control de capacidad del regresor por factor y por forma | M-02, M-05 | ~2 min |
| `m_08_oraculo_perfecto.py <prepared>` | Puntúa un predictor perfecto con `score_predictions()` | M-03, M-17 | ~1 min |
| `m_09_matriz_sql.py outputs/training_v17.sqlite3` | Matriz de confusión desde `oof_predictions` (`mode=ro`) | §4 | <10 s |

Comandos usados:

```bash
S="<scratchpad>"; P=outputs/bench/prepared/af3751551ca5df98
venv/Scripts/python.exe $S/m_01_datos.py $P
PYTHONPATH=. venv/Scripts/python.exe $S/m_02_oof_xgboost.py $P $S xgboost
venv/Scripts/python.exe $S/m_03_analisis_oof.py $P $S/m_oof_xgboost_af3751551ca5df98.parquet
venv/Scripts/python.exe $S/m_04_por_grupo.py   $P $S/m_oof_xgboost_af3751551ca5df98.parquet
PYTHONPATH=. venv/Scripts/python.exe $S/m_05_area_forma.py $P $S/m_oof_xgboost_af3751551ca5df98.parquet
venv/Scripts/python.exe $S/m_06_baselines.py   $P $S/m_oof_xgboost_af3751551ca5df98.parquet
PYTHONPATH=. venv/Scripts/python.exe $S/m_07_capacidad.py $P
PYTHONPATH=. venv/Scripts/python.exe $S/m_08_oraculo_perfecto.py $P
venv/Scripts/python.exe $S/m_09_matriz_sql.py outputs/training_v17.sqlite3
venv/Scripts/python.exe -m pytest tests/ml/bench/test_metrics.py tests/ml/bench/test_score_predictions.py \
  tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parity_with_legacy_evaluator.py \
  tests/ml/bench/test_folds.py -q -p no:cacheprovider --basetemp=$S/m_pytest   # 51 passed
```

Las salidas completas quedaron en `S/m_0*_out*.txt`. Abajo se reproducen los dos scripts que sostienen los hallazgos críticos, para que no dependan del scratchpad.

<details>
<summary><code>m_06_baselines.py</code> (M-01)</summary>

```python
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

d, oof_path = Path(sys.argv[1]), Path(sys.argv[2])
keys = pd.read_parquet(d / "keys.parquet"); X = pd.read_parquet(d / "features.parquet")
T = pd.read_parquet(d / "targets.parquet"); oof = pd.read_parquet(oof_path)
df = keys[["node_id", "shape_id", "factor_mult"]].copy()
df["f"] = df.factor_mult.round(4); df["v"] = T.vol_inundacion_m3.to_numpy(); df["y"] = T.inunda.to_numpy()
df["dur"] = X.duracion_horas.to_numpy(); df["tp"] = X.tiempo_al_pico_h.to_numpy()
piv = df.pivot_table(index=["node_id", "shape_id"], columns="f", values="v"); fs = np.array(sorted(piv.columns))
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
pred = np.empty(len(df)); lookup = {f: piv[f] for f in fs}
for i, (n, s, f) in enumerate(zip(df.node_id, df.shape_id, df.f)):          # LOSO: interpolación entre vecinos
    k = np.searchsorted(fs, f)
    if k == 0: pred[i] = lookup[fs[1]][(n, s)]
    elif k == len(fs) - 1: pred[i] = lookup[fs[-2]][(n, s)]
    else:
        lo, hi = fs[k - 1], fs[k + 1]; a, b = lookup[lo][(n, s)], lookup[hi][(n, s)]
        pred[i] = a + (b - a) * (f - lo) / (hi - lo)
o = oof[oof.protocol == "LOSO"].sort_values("sample_idx")
routed = np.where(o.y_pred_clf.to_numpy() == 1, o.y_pred_reg.to_numpy(), 0.0)
print("LOSO baseline F1=%.4f RMSE=%.2f | XGB F1=%.4f RMSE=%.2f" % (
    f1_score(df.y, pred >= 1), rmse(df.v, pred), f1_score(df.y, o.y_pred_clf), rmse(df.v, routed)))
sh = df[["shape_id", "dur", "tp"]].drop_duplicates().set_index("shape_id"); z = (sh - sh.mean()) / sh.std()
piv2 = df.pivot_table(index=["node_id", "f"], columns="shape_id", values="v")
nearest = {s: ((z.drop(index=s) - z.loc[s]) ** 2).sum(axis=1).idxmin() for s in sh.index}
pred2 = np.array([piv2.at[(n, f), nearest[s]] for n, s, f in zip(df.node_id, df.shape_id, df.f)])
o2 = oof[oof.protocol == "LeaveOneShapeOut"].sort_values("sample_idx")
routed2 = np.where(o2.y_pred_clf.to_numpy() == 1, o2.y_pred_reg.to_numpy(), 0.0)
print("LOSHO baseline F1=%.4f RMSE=%.2f | XGB F1=%.4f RMSE=%.2f" % (
    f1_score(df.y, pred2 >= 1), rmse(df.v, pred2), f1_score(df.y, o2.y_pred_clf), rmse(df.v, routed2)))
```
</details>

<details>
<summary><code>m_08_oraculo_perfecto.py</code> (M-03)</summary>

```python
import sys
import pandas as pd
from swmm_resilience.ml.bench.evaluate import score_predictions
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PreparedDataset

prepared = PreparedDataset.load(sys.argv[1])
tests = prepared.folds[prepared.folds["split"] == "test"]; idx = tests["sample_idx"].to_numpy()
oof = pd.DataFrame({"sample_idx": idx, "protocol": tests["protocol"].to_numpy(),
                    "fold_id": tests["fold_id"].to_numpy(),
                    "y_pred_clf": prepared.y_clf.to_numpy()[idx],
                    "y_prob_clf": prepared.y_clf.to_numpy()[idx].astype(float),
                    "y_pred_reg": prepared.y_reg.to_numpy()[idx]}, columns=list(OOF_COLUMNS))
for p, m in score_predictions(prepared, oof, {"prep_id": prepared.prep_id}).items():
    print(p, m["classifier"], m["by_factor"].get("0.20"))
# Salida: LOSO {'precision': 0.8, 'recall': 0.8, 'f1': 0.8, 'auc_roc': 1.0}; by_factor 0.20 f1 = 0.0 en los 3 protocolos
```
</details>

<details>
<summary>Núcleo de <code>m_03_analisis_oof.py</code> (media frente a pooled, matriz, bootstrap por grupo)</summary>

```python
# por protocolo: po = oof[oof.protocol == p]; idx = po.sample_idx
per = [{"f1": f1_score(yc[i], pc, zero_division=0), "rmse": rmse(yr[i], routed_i), "n_pos": yc[i].sum()}
       for each fold]                                   # métrica por fold como el banco
f1_banco  = mean(per.f1)                                # lo que reporta score_predictions
f1_pooled = f1_score(yc[idx], po.y_pred_clf)            # sobre la concatenación
tn, fp, fn, tp = confusion_matrix(yc[idx], po.y_pred_clf, labels=[0, 1]).ravel()
# bootstrap por grupo: remuestrear folds (grupos) con reemplazo 1000 veces y recalcular F1/RMSE pooled
```
Script completo: `S/m_03_analisis_oof.py`.
</details>
