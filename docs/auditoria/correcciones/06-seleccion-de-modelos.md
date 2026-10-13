# 06 — Selección de modelos: comparación justa XGBoost / MLP / SVM aproximado (parte 2)

**Fecha:** 2026-10-13 · **Corrida:** commit `8a06605` (`git_dirty: false`) · **Datos:** `datos_parte2/` · **Diseño:** `docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md`

Abreviaturas de fuentes usadas en todo el documento:

| Sigla | Archivo |
|---|---|
| CP | `datos_parte2/comparacion_protocolos.csv` |
| MC | `datos_parte2/matriz_confusion.csv` |
| BL | `datos_parte2/banda_limitrofe.csv` |
| PF | `datos_parte2/por_forma.csv` |
| RA | `datos_parte2/resumen_afinacion.csv` |
| CA | `datos_parte2/convergencia_afinacion.csv` |
| TP | `datos_parte2/tables_provenance.json` |
| ST | `outputs/bench/parte2/ca706e27654e0cf5/<familia>/<etapa>/stage.json` |
| BS | remuestreo bootstrap por forma de este documento (§5.3; script desechable, no versionado) |

Las cifras marcadas «derivado» se calculan aritméticamente a partir de la columna citada; las secciones marcadas **Lectura** son interpretación, no medición.

---

## 1. Resumen

1. Se compararon tres familias de modelos —XGBoost, una red neuronal densa (MLP) y un SVM aproximado (Nystroem + SVM lineal)— con los mismos datos, el mismo protocolo (LeaveOneShapeOut, 40 formas) y el mismo presupuesto de afinación (20 trials de Optuna por familia, mismo objetivo).
2. **XGBoost afinado** obtiene el menor RMSE agregado del volumen de extremo a extremo: **69,64 m³**, frente a 101,12 m³ del MLP afinado, 214,87 m³ del SVM aproximado afinado y 172,86 m³ de la referencia sin aprendizaje (CP `rmse_vol_agregado_m3`).
3. La afinación reduce el RMSE agregado un 38,3 % en XGBoost, un 11,6 % en el MLP y un 3,4 % en el SVM aproximado (derivado de CP).
4. Como clasificadores, XGBoost y el MLP son prácticamente indistinguibles (AUC 0,9993–0,9994, F1 0,978–0,981); el SVM aproximado queda por debajo (AUC 0,963–0,972) y su RMSE queda por encima de la referencia sin aprendizaje.
5. Las cifras son una **estimación exploratoria condicionada** (hiperparámetros elegidos con todas las formas), de una sola corrida y una semilla; un bootstrap por forma (§5.3) indica que la ventaja de XGBoost sobre el MLP es estable al remuestrear formas, pero los intervalos de cada modelo son anchos.

---

## 2. Diseño de la comparación

| Elemento | Valor | Fuente |
|---|---|---|
| Contrato de variables | `tabular_v4_21` | TP `feature_contract_id` |
| Conjunto preparado | `prep_id = ca706e27654e0cf5` | TP `prep_id` |
| Filas | 154 080 (nodo × corrida) | TP `n_rows` |
| Corridas | 963 de 1000 de la grilla; 37 celdas faltan tras el filtro de error de continuidad ≤ 5 % | TP `n_runs`, `n_runs_full_grid`, `n_missing_cells`, `max_continuity_error_pct` |
| Red | una sola red de drenaje | diseño |
| Umbral de inundación | 1,0 m³ | TP `flood_threshold_m3` |
| Protocolo de evaluación | LeaveOneShapeOut: 40 folds, en cada uno se retiene una forma de hidrograma completa | TP `evaluation_protocol`; PF (40 formas por modelo) |
| Métrica principal | RMSE agregado (*pooled*) del volumen de extremo a extremo sobre todos los nodos | CP `rmse_vol_agregado_m3` |
| Objetivo de afinación | el mismo RMSE de extremo a extremo, con GroupKFold(5) interno agrupado por forma | `scripts/ejecutar_parte2_v4_21.py` (docstring; `INNER_SPLITS = 5`) |
| Presupuesto | **20 trials por familia**, idéntico para las tres | RA `n_trials`; `N_TRIALS = 20` |
| Muestreador | TPE de Optuna (`TPESampler(seed=42)`), con sus 10 trials iniciales aleatorios por defecto: 10 aleatorios + 10 guiados | `swmm_resilience/ml/bench/tune.py` |
| Semilla | 42 | ST `seed` |

**Modelos.** Todos son de dos etapas: un clasificador decide si el nodo se inunda y un regresor estima el volumen; la predicción de extremo a extremo es el volumen del regresor donde el clasificador predice inundación y 0 en el resto (verificado: con esta regla los OOF reproducen exactamente CP, 69,642 y 101,117 m³; BS).

| Familia | Clasificador | Regresor | Hiperparámetros buscados (por etapa) | Parámetros fijos (no buscados) |
|---|---|---|---|---|
| XGBoost | árboles potenciados por gradiente | ídem | `max_depth` 3–10, `learning_rate` 1e-3–0,3 (log), `n_estimators` 50–500, `subsample` 0,5–1,0 | `n_jobs = 1` |
| MLP (red densa) | red densa | red densa | `hidden_sizes` ∈ {(32), (64, 32), (128, 64), (64, 32, 16)}, `learning_rate` 1e-4–1e-2 (log), `dropout` 0–0,5 | `epochs = 200`, `batch_size = 1024`, 1 hilo |
| SVM aproximado | Nystroem (núcleo RBF) + SVM lineal con pérdida *hinge* + calibración de Platt | Nystroem (RBF) + SVR lineal ε-insensible | `gamma` 1e-3–1 (log), `n_components` 100–400, `alpha` 1e-6–1e-2 (log); el regresor además `epsilon` 0,01–1,0 | ninguno |

Fuentes: `SEARCH_SPACES` en `swmm_resilience/ml/bench/tune.py`; RA `fixed_params`; `swmm_resilience/ml/bench/models/svm_family.py`.

El SVM aproximado **no es un SVM de núcleo exacto**: aproxima el núcleo RBF con a lo sumo 400 componentes de Nystroem y resuelve un problema lineal sobre ellas. Un SVM exacto no es viable con 154 080 filas × 40 folds × 20 trials en el tiempo disponible.

`n_jobs = 1` en XGBoost se fija porque, dentro de los procesos por fold (con `OMP_NUM_THREADS=1`), XGBoost ya corre con un hilo aunque se pida más; se midió que `n_jobs=4` en un proceso hijo da exactamente las predicciones de `n_jobs=1` y no las de `n_jobs=4` en el proceso principal. Fijarlo hace que afinación y evaluación coincidan y sean reproducibles (docstring de `scripts/ejecutar_parte2_v4_21.py`).

Para cada familia se evaluaron dos variantes en LeaveOneShapeOut: `default` (parámetros de `config.yaml` más los fijos de la tabla) y `tuned` (mejores parámetros del estudio). La referencia sin aprendizaje (`baseline_forma_mas_cercana`) copia la respuesta de la forma de entrenamiento más parecida.

---

## 3. Resultados

### 3.1 Tabla principal (LeaveOneShapeOut, 40 folds)

| Modelo | Variante | RMSE agregado (m³) | RMSE medio por fold (m³) | F1 | Recall | Precisión | AUC | NSE oráculo | Error de volumen total (%) |
|---|---|---|---|---|---|---|---|---|---|
| Referencia: forma más cercana | sin aprendizaje | 172,86 | 98,14 | 0,9764 | 0,9808 | 0,9721 | 0,9844 | 0,7102 | −2,98 |
| XGBoost | default | 112,80 | 58,65 | 0,9815 | 0,9919 | 0,9714 | 0,9993 | 0,8766 | −8,69 |
| XGBoost | **afinado** | **69,64** | **40,19** | 0,9812 | 0,9918 | 0,9707 | 0,9993 | **0,9530** | −2,95 |
| MLP | default | 114,38 | 64,08 | 0,9783 | 0,9889 | 0,9679 | 0,9992 | 0,8778 | −5,79 |
| MLP | afinado | 101,12 | 50,13 | 0,9783 | 0,9898 | 0,9672 | 0,9994 | 0,9018 | −5,67 |
| SVM aproximado | default | 222,46 | 142,64 | 0,8625 | 0,8544 | 0,8707 | 0,9719 | 0,5476 | −15,99 |
| SVM aproximado | afinado | 214,87 | 144,16 | 0,8302 | 0,8037 | 0,8586 | 0,9626 | 0,5831 | −15,81 |

Fuente: CP, columnas `rmse_vol_agregado_m3`, `rmse_vol_media_por_fold_m3`, `f1_agregado`, `recall_agregado`, `precision_agregada`, `auc_roc_agregado`, `nse_regresor_oracle`, `error_vol_total_pct`. En los siete modelos `folds_f1_indefinido = 0`.

Definiciones: el **RMSE agregado** junta los 154 080 residuos de los 40 folds y calcula un único RMSE (métrica principal); el **RMSE medio por fold** es la media de los 40 RMSE por forma (cada forma pesa igual, y como el error está concentrado en pocas formas, resulta menor). El **NSE oráculo** evalúa sólo el regresor sobre los nodos realmente inundados (sin errores del clasificador).

### 3.2 Efecto de la afinación (default → afinado)

| Familia | RMSE agregado (m³) | Cambio | RMSE medio por fold (m³) | Cambio | NSE oráculo | F1 |
|---|---|---|---|---|---|---|
| XGBoost | 112,80 → 69,64 | −38,3 % | 58,65 → 40,19 | −31,5 % | 0,8766 → 0,9530 | 0,9815 → 0,9812 |
| MLP | 114,38 → 101,12 | −11,6 % | 64,08 → 50,13 | −21,8 % | 0,8778 → 0,9018 | 0,9783 → 0,9783 |
| SVM aproximado | 222,46 → 214,87 | −3,4 % | 142,64 → 144,16 | +1,1 % | 0,5476 → 0,5831 | 0,8625 → 0,8302 |

Derivado de CP. La afinación mejora sobre todo el regresor (NSE oráculo) y apenas cambia el clasificador de XGBoost y del MLP. En el SVM aproximado el F1 y el AUC empeoran al afinar (0,9719 → 0,9626): el objetivo de la afinación es el RMSE de extremo a extremo, no el F1.

### 3.3 Mejores hiperparámetros por familia

| Familia | Mejor valor interno (RMSE, GroupKFold 5, m³) | Clasificador | Regresor |
|---|---|---|---|
| XGBoost | 65,70 | `max_depth` 7, `learning_rate` 0,0228, `n_estimators` 239, `subsample` 0,686 | `max_depth` 10, `learning_rate` 0,121, `n_estimators` 346, `subsample` 0,951 |
| MLP | 77,20 | capas (128, 64), `learning_rate` 1,79e-3, `dropout` 0,229 | capas (64, 32, 16), `learning_rate` 4,29e-3, `dropout` 0,013 |
| SVM aproximado | 202,40 | `gamma` 1,13e-3, `n_components` 100, `alpha` 4,55e-6 | `gamma` 0,0353, `n_components` 303, `alpha` 1,46e-4, `epsilon` 0,308 |

Fuente: RA `best_value`, `best_classifier_params`, `best_regressor_params` (se omiten los parámetros fijos de §2). Observación (derivada de RA y `SEARCH_SPACES`): el clasificador SVM elegido está en el borde inferior del espacio en `n_components` (100) y casi en el de `gamma` (1,13e-3 frente al mínimo 1e-3); el regresor XGBoost está en el borde superior de `max_depth` (10). Que el óptimo toque el borde sugiere que el espacio de búsqueda podría ser estrecho en esas dimensiones; no se probó ampliarlo.

**Convergencia (CA).** El mejor trial fue el 10 en XGBoost, el 11 en el MLP y el 17 en el SVM aproximado (columna `best_so_far`). En XGBoost los 5 primeros trials aleatorios dieron 311–331 m³ y del trial 5 en adelante 12 de los 15 trials quedaron por debajo de 120 m³ (columna `value`).

### 3.4 Duración

| Familia | default (s) | afinación (s) | afinado (s) | Procesos por fold |
|---|---|---|---|---|
| XGBoost | 227 | 761 | 146 | `--eval-jobs 16` |
| MLP | 805 | 3103 | 990 | por defecto del script (40) |
| SVM aproximado | 315 | 833 | 91 | `--eval-jobs 16` |

Fuente: ST `duration_s` y `command`; RA `duration_s` da para la afinación 756, 3101 y 829 s (tiempo del estudio sin la sobrecarga de la etapa). Las tres familias corrieron **a la vez** en la misma máquina de 64 CPU (ST `started_utc`: 02:55 UTC las tres), con distinto número de procesos; las duraciones **no son comparables entre familias** y sólo indican el orden de magnitud del costo.

### 3.5 Matrices de confusión y falsos negativos en la banda limítrofe

| Modelo | VN | FP | FN | VP | FN en banda 1–2 m³ | FN fuera de banda | FN con volumen ≥ 100 m³ |
|---|---|---|---|---|---|---|---|
| XGBoost afinado | 106 472 | 1383 | 378 | 45 847 | 333 | 45 | 0 |
| MLP afinado | 106 302 | 1553 | 473 | 45 752 | 354 | 119 | 4 |
| (XGBoost default) | 106 503 | 1352 | 373 | 45 852 | 325 | 48 | 0 |
| (MLP default) | 106 340 | 1515 | 514 | 45 711 | 368 | 146 | 19 |
| (SVM aproximado afinado) | 101 738 | 6117 | 9076 | 37 149 | 1217 | 7859 | 1443 |
| (Referencia) | 106 556 | 1299 | 889 | 45 336 | 611 | 278 | 0 |

Fuente: MC (`tn`, `fp`, `fn`, `tp`) y BL (`fn_en_banda_1_2m3`, `fn_fuera_de_banda`, `fn_con_vol_ge_100m3`). En XGBoost afinado el 88 % de los falsos negativos (333 de 378) son nodos con el volumen mínimo que la etiqueta considera inundación; en el MLP afinado, el 75 % (354 de 473), y 4 de sus falsos negativos superan los 100 m³ (derivado de BL).

---

## 4. Lectura (interpretación)

### 4.1 Orden de los candidatos

- **XGBoost afinado es el primero** en la métrica principal: 69,64 m³, 103,22 m³ (−59,7 %) por debajo de la referencia sin aprendizaje (derivado de CP).
- **El MLP afinado es el segundo**: 101,12 m³, 71,75 m³ (−41,5 %) por debajo de la referencia y 31,47 m³ (+45,2 %) por encima de XGBoost afinado (derivado de CP). Como clasificador es equivalente a XGBoost (AUC 0,9994 frente a 0,9993; F1 0,9783 frente a 0,9812); la diferencia está en el volumen (NSE oráculo 0,9018 frente a 0,9530).
- **El SVM aproximado es el tercero y queda por debajo de la referencia sin aprendizaje**: su RMSE agregado afinado (214,87 m³) supera al de la referencia en 42,00 m³ (+24,3 %); el default, en 49,59 m³ (+28,7 %) (derivado de CP). También es peor en el RMSE medio por fold (144,16 frente a 98,14 m³) y en el error de volumen total (−15,8 % frente a −3,0 %).

**Regla de presentación (decisión del autor, 2026-10-12, recogida en el diseño):** XGBoost y el MLP se presentan siempre; el SVM aproximado entra como tercer candidato con su resultado, aunque éste quede por debajo de la referencia.

### 4.2 Por qué el SVM aproximado podría rendir peor (hipótesis no probadas aquí)

Ninguna de estas causas se aisló experimentalmente en esta corrida; son explicaciones plausibles a verificar.

1. **Núcleo aproximado y de baja dimensión.** Nystroem con ≤ 400 componentes (el clasificador elegido usa 100) puede no representar bien la frontera ni la superficie de volumen, que en XGBoost necesitan árboles de profundidad 7–10.
2. **Espacio de búsqueda estrecho.** Sólo 3 hiperparámetros en el clasificador y 4 en el regresor, y el óptimo del clasificador toca el borde del espacio (§3.3); con 20 trials, el SVM sólo mejoró un 3,4 % al afinar.
3. **Calibración de Platt frente a `class_weight`.** El clasificador usa pérdida *hinge* con `class_weight`, pero la calibración de Platt se ajusta sobre folds internos sin ponderar y elimina casi todo el efecto de la ponderación sobre la probabilidad final. Su umbral 0,5 equivale aproximadamente a la decisión de Bayes sin ponderar, mientras que XGBoost y el MLP desplazan su umbral efectivo con `scale_pos_weight`. **Por eso el recall y el F1 del SVM no son directamente comparables** con los de las otras dos familias (docstring «Sesgo conocido» de `svm_family.py`). El AUC, que no depende del umbral, es la comparación de clasificadores más justa: 0,963–0,972 en el SVM frente a 0,999 en XGBoost y el MLP (CP `auc_roc_agregado`). Incluso con esa métrica, el SVM queda por debajo y también por debajo de la referencia (0,9844).
4. **Modelo lineal en el regresor.** El SVR ε-insensible es lineal sobre las componentes de Nystroem; con volúmenes muy asimétricos (0 en la mayoría de los nodos, cientos de m³ en pocos) la pérdida ε-insensible no penaliza los errores grandes cuadráticamente, al contrario que la métrica.

### 4.3 Dónde se concentra el error

| Modelo | Peor forma | Parte del error cuadrático de la peor forma | Error de volumen de esa forma | Top 3 formas | RMSE agregado sin la peor forma (m³) | Rango del error de volumen por forma |
|---|---|---|---|---|---|---|
| XGBoost afinado | `shape_quasi_uniform_block` | 61,0 % | −31,2 % | 75,4 % | 44,06 | −31,2 % a +33,0 % |
| MLP afinado | `shape_quasi_uniform_block` | 73,0 % | −26,3 % | 84,2 % | 53,28 | −26,3 % a +61,6 % |
| XGBoost default (esta corrida) | `shape_quasi_uniform_block` | 73,6 % | −55,9 % | 82,9 % | 58,74 | −55,9 % a +32,3 % |
| XGBoost (informe 04, `datos/`) | `shape_quasi_uniform_block` | 71,6 % | −52,1 % | — | 57,32 | −52,1 % a +33,3 % |

Derivado de PF: error cuadrático de una forma = `rmse_vol² × n_rows`; parte = ese valor sobre la suma de las 40 formas; rango de `error_vol_pct`. La última fila procede de `04-informe-consolidado.md` (`datos/por_forma.csv`).

- La peor forma es la misma en los dos modelos afinados (y en el default): `shape_quasi_uniform_block`, aporte casi uniforme en bloque, con el volumen subestimado. La afinación de XGBoost bajó su RMSE en esa forma de 600,6 a 337,6 m³ (PF `rmse_vol`) y su parte del error del 73,6 % al 61,0 %, pero sigue dominando.
- Las tres peores formas coinciden en ambos modelos afinados: `shape_quasi_uniform_block`, `shape_twin_plateau_15h` y `shape_sustained_plateau_14h` (en ese orden en los dos). Son formas de aporte sostenido o de meseta.
- **Lectura:** la debilidad está en extrapolar el volumen a formas de aporte prolongado no vistas, y es compartida por las dos familias, no un defecto de una sola. Sin la peor forma, la distancia entre XGBoost y el MLP se reduce a 9,2 m³ (44,06 frente a 53,28; derivado), frente a 31,5 m³ con ella; buena parte de la ventaja de XGBoost se juega en esa forma.

---

## 5. Límites y sesgos declarados

1. **Estimación exploratoria condicionada.** Los hiperparámetros se eligieron con GroupKFold(5) sobre **todas** las formas (RA `status = estimacion_exploratoria_condicionada`), así que la evaluación LeaveOneShapeOut posterior no es independiente: las cifras de los modelos afinados son optimistas, **por igual para las tres familias**. Sirven para comparar familias, no como estimación confirmatoria del error en formas nuevas. Una afinación anidada (una por forma retenida, ×40) queda como trabajo futuro.
2. **Presupuesto pequeño.** 20 trials por familia (10 aleatorios + 10 guiados). Con más trials o espacios más amplios el orden podría cambiar, en especial para el SVM (óptimos en el borde, §3.3) y el MLP (su mejor trial apareció en el 11 y el estudio fue el más costoso).
3. **Una corrida, una semilla.** No hay repeticiones con otras semillas, de modo que no hay intervalos de la variabilidad del entrenamiento. Para la variabilidad debida a qué formas hay en la muestra se calculó el bootstrap de §5.3.
4. **Defaults no idénticos a los de la parte 1.** `xgboost_default` usa `n_jobs = 1` (fijo en esta comparación), y `mlp_default` usa `batch_size = 1024` en lugar de 256 de `config.yaml`. Por eso `xgboost_default` (112,80 m³ agregado, 58,65 m³ medio por fold) **no coincide** con el XGBoost del informe 04 (106,21 y 56,81 m³; `datos/comparacion_protocolos.csv`), entrenado con los defaults de `config.yaml` y el `n_jobs` por defecto. Las dos cifras son correctas para su configuración; no deben mezclarse en una misma tabla sin esta nota.
5. **Duraciones no comparables entre familias** (§3.4): las tres corrieron a la vez en 64 CPU con distinto número de procesos.
6. **AUC agregado.** El AUC junta las probabilidades de los 40 folds, cada uno con su propia calibración (en el SVM, con su propio ajuste de Platt); las escalas de probabilidad no son estrictamente comparables entre folds.
7. **Dos definiciones de RMSE** (§3.1): el agregado pondera cada forma por su error y es el que domina la peor forma; el medio por fold pondera las formas por igual. El orden XGBoost < MLP < referencia < SVM se mantiene con ambas (CP).
8. **Una sola red de drenaje.** Nada de lo anterior dice cómo generalizan los modelos a otra red.
9. **Calibración interna del SVM** (`CalibratedClassifierCV(cv=3)`): sus folds internos no barajan y dependen del orden de las filas; si las filas no están ordenadas por forma, la calibración puede ser optimista (docstring de `svm_family.py`).

### 5.3 Bootstrap por forma (ayuda de interpretación)

**Método.** A partir de los OOF de `xgboost/tuned/oof.parquet` y `mlp/tuned/oof.parquet` y del volumen real del conjunto preparado, se calculó la suma de errores cuadráticos y el número de filas de cada una de las 40 formas (folds). Se remuestrearon con reemplazo las 40 formas 2000 veces (semilla 42), **emparejando** ambos modelos en cada remuestra, y en cada una se recalculó el RMSE agregado (√(Σ errores² / Σ filas)) de cada modelo y su diferencia. Intervalos de percentiles 2,5–97,5. El cálculo reproduce las cifras puntuales de CP (69,642 y 101,117 m³).

| Cantidad | Estimación puntual (m³) | Intervalo bootstrap 95 % (m³) |
|---|---|---|
| RMSE agregado, XGBoost afinado | 69,64 | 35,3 – 106,5 |
| RMSE agregado, MLP afinado | 101,12 | 40,8 – 161,6 |
| Diferencia MLP − XGBoost | 31,47 | 4,4 – 56,2 |

Fuente: BS. En el 0,15 % de las remuestras (3 de 2000) la diferencia fue ≤ 0.

**Lectura.** Los intervalos de cada modelo son muy anchos y se solapan, porque el RMSE agregado depende de si la forma `shape_quasi_uniform_block` entra o no en la remuestra. La diferencia emparejada, en cambio, queda por encima de 0: en casi todas las combinaciones de formas XGBoost afinado tiene menor RMSE agregado que el MLP afinado. Este intervalo sólo refleja la variabilidad por la elección de formas; no incluye la variabilidad por semilla ni el optimismo de §5.1, y las 40 formas de la grilla no son una muestra aleatoria de una población de hidrogramas, así que es una ayuda de interpretación, no una prueba formal.

---

## 6. Texto para la tesis

> **Selección de familias y protocolo de comparación.** Se compararon tres familias de modelos para el sustituto de dos etapas (clasificación de la inundación del nodo y regresión del volumen): árboles potenciados por gradiente (XGBoost), una red neuronal densa (MLP) y una máquina de vectores de soporte con núcleo RBF aproximado mediante Nystroem (a lo sumo 400 componentes), con un SVM lineal de pérdida *hinge* calibrado con Platt como clasificador y una SVR lineal ε-insensible como regresor; este último no es un SVM de núcleo exacto, inviable con 154 080 filas. Las tres familias usaron el mismo conjunto de datos (contrato `tabular_v4_21`, 963 de las 1000 simulaciones de la grilla tras descartar las de error de continuidad superior al 5 %, una sola red), el mismo protocolo de evaluación (dejar fuera una forma de hidrograma completa, 40 particiones) y el mismo presupuesto de afinación: 20 configuraciones por familia exploradas con el muestreador TPE de Optuna (10 aleatorias y 10 guiadas), minimizando el RMSE del volumen de extremo a extremo con validación cruzada interna agrupada por forma (cinco grupos). La métrica principal es el RMSE agregado del volumen sobre todos los nodos y particiones, y se incluye como referencia un predictor sin aprendizaje que copia la respuesta de la forma de entrenamiento más parecida.
>
> **Resultados y su alcance.** XGBoost afinado obtuvo el menor RMSE agregado (69,64 m³, con NSE del regresor de 0,953 en los nodos inundados), seguido del MLP afinado (101,12 m³; NSE 0,902); ambos mejoran claramente la referencia sin aprendizaje (172,86 m³) y son equivalentes como clasificadores (AUC 0,999; F1 0,978–0,981). El SVM aproximado quedó por debajo de la referencia (214,87 m³; AUC 0,963), posiblemente por la aproximación de baja dimensión del núcleo, un espacio de búsqueda reducido y una calibración que anula la ponderación de clases, causas que no se aislaron aquí. Un remuestreo bootstrap de las 40 formas sitúa la diferencia entre el MLP y XGBoost entre 4,4 y 56,2 m³ (95 %), lo que sugiere que la ventaja de XGBoost no depende de unas pocas formas, aunque el error de ambos modelos se concentra en las mismas formas de aporte sostenido: la peor de ellas reúne el 61 % (XGBoost) y el 73 % (MLP) del error cuadrático. Estas cifras deben leerse como una estimación exploratoria: los hiperparámetros se eligieron con todas las formas, lo que favorece por igual a las tres familias, proceden de una sola corrida y semilla, y se obtuvieron sobre una única red. Por ello se presentan XGBoost y el MLP como candidatos principales y el SVM aproximado como tercer candidato con su resultado; una afinación anidada por forma y la repetición con varias semillas quedan como trabajo futuro.

---

## 7. Procedencia

- **Commit de la corrida:** `8a0660589425e59849813b9ca59c468458f9d287` (`8a06605`), `git_dirty: false` (ST `git`, TP `git_commit`).
- **Versiones:** Python 3.12.10, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.6.0, optuna 5.0.0 (TP `versions`).
- **Comandos ejecutados** (ST `command`), desde la raíz del repositorio:
  - `python scripts/ejecutar_parte2_v4_21.py --family xgboost --eval-jobs 16`
  - `python scripts/ejecutar_parte2_v4_21.py --family mlp`
  - `python scripts/ejecutar_parte2_v4_21.py --family svm --eval-jobs 16`
  - `python scripts/ejecutar_parte2_v4_21.py --build-tables` (tablas generadas el 2026-10-13 17:19 UTC; TP `generated_at_utc`)
- **Artefactos por etapa** (solo lectura): `outputs/bench/parte2/ca706e27654e0cf5/{xgboost,mlp,svm}/{default,tune,tuned}/` con `stage.json`, `metrics.json` y `oof.parquet` (default, tuned) y el estudio `tune/<familia>/<prep_id>_<marca>/study.json`. Las huellas SHA-256 de los OOF, los estudios y las tablas están en TP `inputs_sha256` y `outputs_sha256`.
- **Tablas:** `docs/auditoria/correcciones/datos_parte2/{comparacion_protocolos,matriz_confusion,banda_limitrofe,por_forma,resumen_afinacion,convergencia_afinacion}.csv` y `tables_provenance.json`.
- **Referencia sin aprendizaje:** `outputs/bench/oof/ca706e27654e0cf5/baseline_forma_mas_cercana.parquet`.
- **Registros de la ejecución:** `svm.log`, `xgboost.log` y `mlp.log` de la sesión (temporales, fuera del repositorio); sus cifras finales coinciden con CP. El registro del MLP contiene sólo avisos de Optuna sobre opciones categóricas con tuplas (`hidden_sizes`), que no afectan a la búsqueda.
- **Bootstrap (§5.3):** script desechable de la sesión, no versionado; método descrito en §5.3 y reproducible con los OOF citados y `load_prepared()` de `scripts/ejecutar_parte2_v4_21.py`.
- **Reproducir:** `venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family <xgboost|mlp|svm>` (las etapas con marca `done` se saltan; borrar o usar otra raíz de salida para recalcular) y después `venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --build-tables`.
