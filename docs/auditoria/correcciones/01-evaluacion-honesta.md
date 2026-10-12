# Corrección 01 — Evaluación honesta del modelo sustituto (M-01, M-02, M-03, M-08)

- **Fecha:** 2026-10-11
- **Rama:** `worktree-optuna-tuning-familia-ganadora` (commits `b831f0f` … `284927d`; lista completa en [`04-informe-consolidado.md`](04-informe-consolidado.md)).
- **Origen de los hallazgos:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) y [`../anexos/02-ml-protocolo-evaluacion.md`](../anexos/02-ml-protocolo-evaluacion.md).
- **Fuente de todas las cifras nuevas:** [`datos/`](datos/README.md). Cada número lleva su archivo y su columna.

## 0. Alcance de las mediciones (leer antes de citar cualquier cifra)

| Aspecto | Valor | Fuente |
|---|---|---|
| Modelo evaluado | XGBoost con los hiperparámetros **por defecto de `config.yaml`** (200 árboles, profundidad 6, tasa 0,05, submuestreo 0,8, en clasificador y regresor). **No** es el modelo ajustado con Optuna ni el modelo promovido | `datos/runs/provenance.json`, `xgboost_params` y `xgboost_params_note` |
| Contrato de features | `tabular_v4_21` (21 features) | `datos/tables_provenance.json`, `feature_contract_id`; `datos/runs/provenance.json`, `n_features` |
| Conjunto preparado | `prep_id` `ca706e27654e0cf5` | `datos/tables_provenance.json`, `prep_id` |
| Filas | 154 080 = 963 corridas (forma, factor) × 160 nodos | `datos/tables_provenance.json`, `n_rows`, `n_runs` |
| Corridas excluidas | 37 de 1000 (filtro de continuidad, \|error\| ≤ 5 %) | `datos/tables_provenance.json`, `n_runs_full_grid`, `n_missing_cells`, `max_continuity_error_pct` |
| Red | Una sola red (`chico_hydro-qx1`), 160 nodos | `datos/README.md`; anexo 02 §1.3 |
| Umbral de la etiqueta | `inunda = 1` si el volumen del `.rpt` es ≥ 1 m³ (equivale a ≈ 0,5 m³ de volumen continuo en la muestra re-simulada; ver [`03`](03-resolucion-de-la-etiqueta.md)) | `datos/tables_provenance.json`, `flood_threshold_m3` |
| Referencias sin aprendizaje | `baseline_interpolacion_factor` (LOSO, GroupKFold5) y `baseline_forma_mas_cercana` (LeaveOneShapeOut) | `datos/runs/provenance.json`, `baselines` |
| Filas de respaldo de las referencias | Interpolación: 3840 de 308 160 filas (1,25 %), sumando LOSO y GroupKFold5. Forma más cercana: 640 de 154 080 (0,42 %) | `datos/runs/provenance.json`, `baseline_fallback` |

**Cifras anteriores.** Las cifras "antes" de este documento vienen de la auditoría (anexo 02), que se midió sobre **otro conjunto**: contrato `tabular_v3_17` (17 features), 160 000 filas sin filtro de continuidad, PreparedDataset `af3751551ca5df98`. El contrato y el conjunto de filas cambian a la vez, así que la comparación antes/después **no es un experimento controlado**: describe dos estados del proyecto, no el efecto aislado de un cambio. La única comparación controlada de contratos disponible es la de [`../../resultados/2026-10-06-explicacion-completa.md`](../../resultados/2026-10-06-explicacion-completa.md) §11.1 (RMSE medio por forma 78,0368 → 56,8088 m³), que no se volvió a ejecutar aquí; la corrida de esta serie reproduce el valor 56,8088 (`datos/comparacion_protocolos.csv`, `rmse_vol_media_por_fold_m3`, XGBoost, LeaveOneShapeOut).

**Qué mide cada protocolo** (columna `que_mide` de `datos/comparacion_protocolos.csv`):

| Protocolo | Grupo retenido | Qué mide |
|---|---|---|
| LOSO | Un factor de intensidad completo | Interpolación entre factores: todas las formas siguen en el entrenamiento |
| GroupKFold5 | Cinco factores por pliegue | Lo mismo que LOSO, con cinco factores retenidos por pliegue |
| LeaveOneShapeOut | Una forma de hidrograma completa | Generalización a formas de hidrograma no vistas. **Protocolo principal** |

Ningún protocolo retiene nodos ni la combinación "forma nueva × factor nuevo". Toda afirmación de generalización queda acotada a esta red y estos 160 nodos (anexo 02 §1.3).

**Vocabulario.** "Agregado" = métrica calculada sobre todas las predicciones fuera de pliegue del protocolo, concatenadas. "Media por pliegue" = promedio de la métrica de cada pliegue (la definición anterior del banco). "Oráculo" = regresor evaluado solo en los nodos realmente inundados. "Extremo a extremo" = volumen predicho por el regresor solo donde el clasificador predice inundación, evaluado en todos los nodos.

---

## 1. M-01 — LOSO y GroupKFold5 miden interpolación; faltaban referencias sin aprendizaje

### Problema y evidencia

La auditoría mostró que, en LOSO, una tabla de consulta que interpola el volumen del mismo nodo y la misma forma entre los factores vecinos supera a XGBoost (anexo 02, M-01, conjunto `tabular_v3_17` de 160 000 filas): RMSE extremo a extremo agregado de 9,00 m³ frente a 68,43 m³, y F1 agregado de 0,9718 frente a 0,9656. El banco no tenía ninguna referencia sin aprendizaje con la que comparar, así que un buen número en LOSO no permitía saber si el modelo aportaba algo.

### Causa

LOSO y GroupKFold5 agrupan por `factor_mult` (anexo 02 §1.3). Las features dinámicas son exactamente `base × factor`, y al retener un factor el mismo nodo con la misma forma está en el entrenamiento a ±0,2. Con una malla densa de 25 factores, predecir el factor retenido es una interpolación casi lineal que no exige aprender hidráulica.

### Qué cambió

| Commit | Archivo | Cambio |
|---|---|---|
| `9c8db1b` | `swmm_resilience/ml/bench/baselines.py` (nuevo), `swmm_resilience/ml/bench/schemas.py` | Dos referencias sin aprendizaje que producen predicciones fuera de pliegue con el mismo formato que un modelo y se puntúan con la misma función (`score_predictions`). `PROTOCOL_INTERPRETATION` describe qué mide cada protocolo y se escribe en los reportes |
| `02cdaab`, `6c48c46` | `baselines.py` | La referencia de forma más cercana acepta una tabla de descriptores de forma explícita (el contrato `tabular_v4_21` no contiene `duracion_horas` ni `tiempo_al_pico_h`) y rechaza tablas con NaN, formas duplicadas o sin columnas |
| `833760d`, `5f0586d` | `baselines.py` | Las referencias funcionan sobre la malla incompleta que deja el filtro de continuidad (37 celdas forma–factor ausentes) y cuentan las filas que usan un respaldo |
| `a26155e`, `4a2a87c`, `4717885`, `e26fc5a` | `scripts/ejecutar_evaluacion_v4_21.py`, `scripts/informe_evaluacion.py` | Corrida de XGBoost y de las dos referencias con los tres protocolos sobre `tabular_v4_21`, y tablas del informe con su procedencia |
| `395a2ee`, `284927d` | `docs/auditoria/correcciones/datos/` | Tablas, métricas completas, procedencia y README |

Definición de las referencias (docstring de `baselines.py`; `datos/README.md`):

- **Interpolación de factor** (LOSO, GroupKFold5): para cada fila de test (forma, nodo, factor), interpola linealmente el volumen del mismo nodo y forma entre los factores de entrenamiento vecinos disponibles en esa serie; en los extremos usa el vecino disponible. La etiqueta predicha es `volumen ≥ 1 m³`.
- **Forma más cercana** (LeaveOneShapeOut): copia el volumen del mismo nodo y factor de la forma de entrenamiento más parecida en (`duracion_horas`, `tiempo_al_pico_h`) estandarizados con las formas de entrenamiento; si esa forma no tiene la celda, baja a la siguiente del orden. Los descriptores se toman del conjunto preparado `tabular_v3_17` `2db0578f875526cb`, que tiene las mismas 40 formas (`datos/runs/provenance.json`, `shape_descriptors_source`).
- Las referencias no producen probabilidades: su "probabilidad" es la propia etiqueta (0/1), así que su AUC-ROC no se interpreta (spec §2.2).

### Verificación

**Tabla 1.1 — Antes (anexo 02, M-01; `tabular_v3_17`, 160 000 filas).**

| Protocolo | Modelo | F1 agregado | RMSE ext. a ext. agregado (m³) | RMSE medio por forma (m³) |
|---|---|---|---|---|
| LOSO | Interpolación de factor | 0,9718 | 9,00 | — |
| LOSO | XGBoost | 0,9656 | 68,43 | — |
| LeaveOneShapeOut | Forma más cercana | 0,9758 | 170,2 | 98,2 |
| LeaveOneShapeOut | XGBoost | 0,9789 | 136,1 | 79,7 |

**Tabla 1.2 — Después (`tabular_v4_21`, 154 080 filas).** Fuente: `datos/comparacion_protocolos.csv`, columnas `f1_agregado`, `recall_agregado`, `precision_agregada`, `rmse_vol_agregado_m3`, `rmse_vol_media_por_fold_m3`, `error_vol_total_pct`, `nse_regresor_oracle`.

| Protocolo | Modelo | F1 agregado | Recall | Precisión | RMSE agregado (m³) | RMSE medio por pliegue (m³) | Error vol. total (%) | NSE oráculo |
|---|---|---|---|---|---|---|---|---|
| LOSO | Interpolación de factor | 0,9718 | 0,9910 | 0,9534 | **9,13** | 4,77 | −0,39 | 0,9992 |
| LOSO | XGBoost | 0,9667 | 0,9810 | 0,9528 | **57,89** | 46,28 | −5,03 | 0,9675 |
| GroupKFold5 | Interpolación de factor | 0,9718 | 0,9910 | 0,9534 | 9,13 | 7,47 | −0,39 | 0,9992 |
| GroupKFold5 | XGBoost | 0,9681 | 0,9817 | 0,9549 | 56,19 | 55,44 | −4,64 | 0,9694 |
| LeaveOneShapeOut | Forma más cercana | 0,9764 | 0,9808 | 0,9721 | **172,86** | 98,14 | −2,98 | 0,7102 |
| LeaveOneShapeOut | XGBoost | **0,9817** | 0,9918 | 0,9718 | **106,21** | 56,81 | −7,98 | 0,8906 |

**Lectura.**

1. **En LOSO, la referencia sin aprendizaje gana a XGBoost en volumen** (RMSE agregado 9,13 frente a 57,89 m³) y en F1 (0,9718 frente a 0,9667). El hallazgo M-01 se mantiene con el contrato `tabular_v4_21`: LOSO y GroupKFold5 miden interpolación entre factores, y en esa tarea una tabla de consulta es mejor que el modelo.
2. **GroupKFold5 no es más exigente que LOSO para la interpolación.** Las predicciones de la referencia de interpolación son idénticas en los dos protocolos (diferencia absoluta máxima 0,0 en las 154 080 filas): con 25 factores repartidos de forma alterna en 5 pliegues, ningún pliegue retiene dos factores contiguos (pliegue 0: {0,6; 2,0; 3,0; 4,0; 5,0}; pliegue 1: {0,4; 1,8; 2,8; 3,8; 4,8}; …), de modo que cada factor retenido conserva sus dos vecinos en el entrenamiento. Fuente: comprobación del implementador en el informe de la tarea 8 (notas de trabajo no versionadas), repetida para este documento con el comando de la §5. Por la misma razón, en la Tabla 1.2 las filas de la interpolación coinciden entre LOSO y GroupKFold5 en todas las métricas agregadas; solo cambia la media por pliegue, porque los pliegues agrupan filas distintas.
3. **LeaveOneShapeOut es el protocolo que mide algo que la interpolación no resuelve.** Allí XGBoost supera a la referencia de forma más cercana en volumen (RMSE agregado 106,21 frente a 172,86 m³; medio por pliegue 56,81 frente a 98,14 m³; NSE oráculo 0,8906 frente a 0,7102) y, con margen pequeño, en F1 (0,9817 frente a 0,9764). La referencia tiene, sin embargo, menor error de volumen total (−2,98 % frente a −7,98 %). Por forma (derivado de `datos/por_forma.csv`, comparando `rmse_vol` y `error_vol_pct` de ambos modelos en LeaveOneShapeOut), XGBoost tiene menor RMSE en 31 de las 40 formas, menor error absoluto de volumen total en 33 y mayor F1 en 25.
4. **Las filas de respaldo son pocas.** La interpolación usa un vecino distinto del inmediato de la malla del pliegue en 1,25 % de sus filas, y la forma más cercana baja a la segunda o siguiente forma del orden en 0,42 % (`datos/runs/provenance.json`, `baseline_fallback`). El contador de la interpolación se define frente a la malla de entrenamiento del pliegue, no frente a la malla completa, para no contar como respaldo los factores que GroupKFold5 retiene a propósito (docstring de `baselines.py`).

**Pruebas que sostienen el cambio** (`tests/ml/bench/test_baselines.py`, 30 pruebas): entre otras, `test_interpolation_reproduces_a_linear_function_in_the_interior`, `test_nearest_shape_borrows_the_volume_of_the_closest_training_shape`, `test_interpolation_uses_the_series_own_neighbours_when_a_middle_factor_is_missing` (serie no lineal, valores calculados a mano), `test_nearest_shape_walks_down_the_ranking_when_the_nearest_lacks_the_cell`, `test_fallback_counters_for_the_interpolation_baseline` y `test_every_protocol_has_an_interpretation`.

### Límites que quedan

- **XGBoost sin ajustar.** Las cifras de XGBoost son las de los hiperparámetros por defecto de `config.yaml`; el modelo ajustado con Optuna no se evaluó aquí contra las referencias.
- **Una sola red y 963 de 1000 corridas.** Ningún protocolo retiene nodos ni redes; la generalización a otra red no está medida.
- **La referencia de forma más cercana depende de cómo se define "cercana".** Usa solo duración y tiempo al pico. Con otra definición (por ejemplo, incluyendo el volumen del hidrograma) su número cambiaría. Un ejemplo de su fragilidad (`datos/por_forma.csv`): `shape_quasi_uniform_block` y `shape_slow_broad_rise` se eligen mutuamente como vecinas y la referencia intercambia sus volúmenes (1 767 048 y 684 492 m³), con error de volumen total de −61,3 % y +158,2 %.
- **Filas de respaldo.** 1,25 % y 0,42 % de las filas de las referencias no usan su fuente primaria porque el filtro de continuidad dejó huecos en la malla.
- **No hay protocolo de extrapolación.** El anexo 02 recomendaba un protocolo combinado (forma y factor nuevos a la vez, o bloques de factores extremos). No se implementó.
- **El ranking del banco no cambió de métrica principal.** Sigue ordenando por la media por pliegue del RMSE extremo a extremo, sin intervalos de confianza (M-04, abierto).

### Texto para la tesis

> Para interpretar las métricas de validación cruzada se compararon los modelos con dos referencias que no aprenden. La primera estima el volumen de un nodo interpolando linealmente, para la misma forma de hidrograma, los volúmenes simulados en los factores de intensidad vecinos; la segunda copia el volumen del mismo nodo y factor en la forma de entrenamiento más parecida por duración y tiempo al pico. En los protocolos que retienen factores de intensidad (LOSO y GroupKFold5), la interpolación obtiene un RMSE de volumen de 9,13 m³ frente a 57,89 m³ de XGBoost en LOSO, y un F1 de 0,9718 frente a 0,9667. Esos protocolos miden, por tanto, la capacidad de interpolar entre factores ya vistos con la misma forma, y no aportan evidencia de generalización; con la malla de 25 factores usada, GroupKFold5 no es más exigente que LOSO, porque ningún pliegue retiene dos factores contiguos. El protocolo principal de este trabajo es la validación dejando fuera una forma de hidrograma completa (LeaveOneShapeOut). En él, XGBoost supera a la referencia de forma más cercana en volumen (RMSE agregado de 106,21 frente a 172,86 m³) y, por un margen pequeño, en clasificación (F1 de 0,9817 frente a 0,9764). Estas cifras corresponden a XGBoost con hiperparámetros por defecto, a una sola red de drenaje y a 963 de las 1000 simulaciones, las que cumplen un error de continuidad de a lo sumo 5 %.

---

## 2. M-02 — El volumen generaliza peor con formas no vistas

### Problema y evidencia

Con el contrato anterior (`tabular_v3_17`), la auditoría midió en LeaveOneShapeOut un NSE del regresor oráculo de 0,8141, frente a 0,9530 en LOSO, y un error de volumen total por forma de −63,3 % (`shape_quasi_uniform_block`) a +216,3 % (`common_bogota`). Una sola forma, `shape_quasi_uniform_block`, concentraba el 61,35 % del error cuadrático extremo a extremo (anexo 02, M-02; auditoría §3.1). El banco no tenía desglose por forma, así que estas cifras no salían de ningún reporte del proyecto.

### Causa

En `tabular_v3_17` la forma del hidrograma entra al modelo solo como `duracion_horas` y `tiempo_al_pico_h`, y los caudales están normalizados por el pico, de modo que ninguna feature describe el volumen del evento (anexo 02, M-02; anexo 01, H-02). El contrato `tabular_v4_21`, anterior a esta serie de correcciones, añade variables integradas de volumen (`volumen_local_integrado_m3`, `volumen_tributario_integrado_m3`, `integral_exceso_capacidad_m3`, entre otras; `datos/runs/provenance.json`, `feature_columns`). Faltaba medir con él el NSE y el error por forma (spec §1).

### Qué cambió

Esta serie no cambia el modelo ni las features. Añade la **medición**:

| Commit | Archivo | Cambio |
|---|---|---|
| `b831f0f` | `swmm_resilience/ml/bench/metrics.py` | `shape_breakdown_entry` (F1, RMSE y error de volumen total por forma) y `end_to_end_pooled_metrics` (RMSE, MAE y error de volumen total agregados) |
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py` | `score_predictions` devuelve `by_shape` y `end_to_end_pooled` para cada protocolo |
| `a26155e` … `284927d` | `scripts/`, `docs/auditoria/correcciones/datos/` | Corrida sobre `tabular_v4_21` y tabla `por_forma.csv` (6 combinaciones modelo–protocolo × 40 formas) |

### Verificación

**Tabla 2.1 — Antes y después.** "Antes": anexo 02 (`tabular_v3_17`, 160 000 filas). "Después": `tabular_v4_21`, 154 080 filas. No es una comparación controlada (§0).

| Métrica (XGBoost, LeaveOneShapeOut salvo que se indique) | Antes | Fuente "antes" | Después | Fuente "después" |
|---|---|---|---|---|
| NSE del regresor oráculo | 0,8141 | Auditoría §3.1 | 0,8906 | `comparacion_protocolos.csv`, `nse_regresor_oracle` |
| NSE oráculo en LOSO | 0,9530 | Auditoría §3.1 | 0,9675 | ídem |
| NSE oráculo en GroupKFold5 | 0,9553 | Auditoría §3.1 | 0,9694 | ídem |
| RMSE del regresor oráculo (m³) | 253 | Anexo 02, M-02 | 193,91 | `runs/metrics_xgboost.json`, `LeaveOneShapeOut.regressor_oracle.rmse` |
| RMSE extremo a extremo agregado (m³) | 136,1 | Anexo 02, M-04 | 106,21 | `comparacion_protocolos.csv`, `rmse_vol_agregado_m3` |
| RMSE extremo a extremo medio por forma (m³) | 79,7 | Anexo 02, M-04 | 56,81 | `comparacion_protocolos.csv`, `rmse_vol_media_por_fold_m3` |
| Error de volumen total por forma, mínimo | −63,3 % (`shape_quasi_uniform_block`) | Anexo 02, M-02 | −52,1 % (`shape_quasi_uniform_block`) | `por_forma.csv`, `error_vol_pct` |
| Error de volumen total por forma, máximo | +216,3 % (`common_bogota`) | Anexo 02, M-02 | +33,3 % (`shape_four_pulse_cluster`) | `por_forma.csv`, `error_vol_pct` |
| Parte del error cuadrático de la peor forma | 61,35 % (`shape_quasi_uniform_block`) | Anexo 02, M-02 | 71,6 % (`shape_quasi_uniform_block`) | Derivado de `por_forma.csv`: `rmse_vol² × n_rows` de la forma / suma de las 40 formas |
| Error de volumen total, todas las formas | — | — | −7,98 % | `comparacion_protocolos.csv`, `error_vol_total_pct` |

Cifras derivadas de `datos/por_forma.csv` (XGBoost, LeaveOneShapeOut), calculadas para este documento con el comando de la §5:

- Formas con \|error de volumen total\| > 20 %: 12 de 40. Mediana del \|error\| por forma: 18,2 %.
- Las dos formas de mayor error cuadrático suman el 77,6 % (`shape_quasi_uniform_block` 71,6 % y `shape_sustained_plateau_14h` 5,9 %); las tres primeras, el 81,5 %.
- Sin `shape_quasi_uniform_block`, el RMSE agregado de las otras 39 formas sería 57,32 m³ (frente a 106,21 m³ con las 40).
- La forma peor clasificada es `shape_ultra_short_impulse` (F1 0,8798), seguida de `common_bogota` (0,9188) y `rare_extreme` (0,9316); estas tres son también las que más celdas perdieron con el filtro de continuidad (`n_rows` 2400, 3200 y 2880 de 4000).

La tabla completa por forma está en la §6.

**Lectura.**

- Con `tabular_v4_21`, el NSE oráculo en LeaveOneShapeOut sube de 0,8141 a 0,8906 y el rango del error por forma se estrecha de [−63,3 %; +216,3 %] a [−52,1 %; +33,3 %]. Atribuir esa mejora a las variables de volumen es una **interpretación**: las dos mediciones difieren también en el conjunto de filas. La comparación controlada de contratos (mismo conjunto de filas) existe solo para el RMSE medio por forma: 78,0368 → 56,8088 m³ (`docs/resultados/2026-10-06-explicacion-completa.md` §11.1, no re-ejecutada aquí).
- **La concentración del error no bajó: subió.** La misma forma (`shape_quasi_uniform_block`, subestimada en un 52,1 %) pasa del 61,35 % al 71,6 % del error cuadrático. Que esto ocurra porque las demás formas mejoraron más es una **interpretación** compatible con los datos (las otras 39 formas dan un RMSE agregado de 57,32 m³), no una medición del mecanismo.
- **El RMSE agregado y el medio por forma cuentan historias distintas.** En LeaveOneShapeOut, 106,21 frente a 56,81 m³. La media por pliegue diluye las formas de cola pesada y el agregado las pondera por su error cuadrático. Las dos deben reportarse, declarando cuál responde a qué pregunta (M-04).

### Límites que quedan

- **M-02 sigue abierto para las formas en bloque.** El volumen de `shape_quasi_uniform_block` se subestima en un 52,1 % y esa forma domina el error agregado.
- **No se midió por qué.** La correlación del error por forma con el factor de llenado (Spearman −0,577 en `tabular_v3_17`, anexo 02) no se recalculó con `tabular_v4_21`.
- **No se hizo una ablación de las features de volumen bajo LeaveOneShapeOut** (recomendación del anexo 02). La mejora atribuible a cada variable no está medida.
- **Rediseñar las features de volumen más allá de `tabular_v4_21`** quedó fuera de alcance (spec §7).
- **LeaveOneShapeOut es el extremo optimista de la generalización por forma**: retiene una forma y deja 39 en el entrenamiento. Con 8 formas retenidas por pliegue, la auditoría midió en `tabular_v3_17` un NSE oráculo de 0,69 (anexo 02, M-02); no se midió con `tabular_v4_21`.
- **XGBoost sin ajustar, una red, 963 corridas** (§0).

### Texto para la tesis

> La generalización a formas de hidrograma no vistas se evaluó dejando fuera, en cada pliegue, una de las 40 formas (LeaveOneShapeOut). Con el contrato de 21 variables, que incluye variables integradas del volumen del evento, el regresor evaluado sobre los nodos realmente inundados alcanza un NSE de 0,8906, frente a 0,9675 cuando se retienen factores de intensidad. El error del volumen total predicho por forma va de −52,1 % a +33,3 %, y una sola forma, de aporte casi uniforme en bloque, concentra el 71,6 % del error cuadrático: su volumen se subestima en un 52,1 %. Sin ella, el RMSE agregado de las otras 39 formas sería de 57,32 m³, frente a 106,21 m³ con todas. La clasificación generaliza mejor que la magnitud del volumen: el F1 agregado es de 0,9817, con el peor valor por forma (0,8798) en el impulso ultracorto, que es además la forma con más simulaciones excluidas por el criterio de continuidad. Con el contrato anterior de 17 variables, la auditoría del proyecto había medido un NSE de 0,8141 y un rango de error por forma de −63,3 % a +216,3 %; esa medición usó un conjunto de filas distinto, por lo que la diferencia no se atribuye por completo al cambio de variables. Estas cifras corresponden a XGBoost con hiperparámetros por defecto y a una sola red.

---

## 3. M-03 — F1, precisión y recall promediados por pliegue con ceros

### Problema y evidencia

El banco promediaba F1, precisión y recall por pliegue con `zero_division=0`, de modo que un pliegue sin ninguna inundación real puntuaba 0 aunque el clasificador acertara todo. En LOSO hay 5 pliegues así (factores bajos sin nodos inundados). La auditoría midió, con `tabular_v3_17` y 160 000 filas, un F1 publicado de 0,7408 en LOSO frente a un F1 agregado de 0,9656, y una precisión de 0,7226 frente a 0,9508. Un predictor **perfecto** obtenía F1 = 0,80 en LOSO (anexo 02, M-03; auditoría §3.1). La "brecha" entre LOSO y GroupKFold5 de los reportes era un artefacto de esa agregación.

### Causa

`ml/bench/metrics.py:41-43` (`zero_division=0`) y la media por pliegue en `evaluate.py` y en el evaluador legado `ml/evaluator.py` (anexo 02, M-03).

### Qué cambió

| Commit | Archivo | Cambio |
|---|---|---|
| `b831f0f` | `swmm_resilience/ml/bench/metrics.py` | Funciones puras: `confusion_counts`, `pooled_classifier_metrics` (sobre todas las filas de test concatenadas, con especificidad), `fold_classifier_metrics` (NaN donde la métrica no está definida) y `summarize_fold_metrics` (mínimo, mediana, máximo, media y número de pliegues indefinidos) |
| `2871917` | ídem | Las métricas agregadas también son NaN, no 0, cuando no están definidas |
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py`, `tests/ml/bench/test_parity_with_legacy_evaluator.py` | `classifier` pasa a ser el valor agregado; `classifier_per_fold` da la dispersión entre pliegues; el valor anterior se conserva como `classifier_legacy_mean_zero_fill`. El test de paridad con el evaluador legado compara ahora contra ese campo |
| `5528479` | `tests/ml/bench/test_score_predictions.py`, spec | El parámetro `protocols` se prueba con un conjunto de dos protocolos; la spec se alinea con los nombres del código (`n_undefined`) |

Como el criterio de desempate del ranking es `classifier.f1` (`config.yaml`, `bench.ranking.tie_breakers`), ese desempate lee ahora el valor agregado.

### Verificación

**Tabla 3.1 — F1 en LOSO, definición anterior frente a la nueva.**

| Caso | Media por pliegue con ceros (anterior) | Agregado (nuevo) | Pliegues con F1 indefinido | Fuente |
|---|---|---|---|---|
| XGBoost, `tabular_v3_17`, 160 000 filas (antes) | 0,7408 | 0,9656 | 5 | Anexo 02, M-03 |
| XGBoost, `tabular_v4_21` (después) | 0,7379 | 0,9667 | 5 | `comparacion_protocolos.csv`, `f1_media_por_fold_con_ceros`, `f1_agregado`, `folds_f1_indefinido` |
| Interpolación de factor, `tabular_v4_21` | 0,7425 | 0,9718 | 5 | ídem |
| Predictor perfecto, `tabular_v3_17` (antes) | 0,80 | — | 5 | Anexo 02, M-03 (script `m_08`) |
| Predictor perfecto, `tabular_v4_21` (después) | 0,8000 | 1,0000 | 5 | Comando de la §5 (ejecutado para este documento) |

**Tabla 3.2 — XGBoost, `tabular_v4_21`, todas las métricas del clasificador en LOSO.** Fuente: `datos/runs/metrics_xgboost.json`, bloque `LOSO`.

| Métrica | Media por pliegue con ceros (`classifier_legacy_mean_zero_fill`) | Agregado (`classifier`) | Por pliegue definido: mínimo / mediana / máximo (`classifier_per_fold`) |
|---|---|---|---|
| Precisión | 0,7215 | 0,9528 | 0,2541 / 0,9488 / 0,9959 |
| Recall | 0,7601 | 0,9810 | 0,5679 / 0,9807 / 0,9972 |
| F1 | 0,7379 | 0,9667 | 0,3511 / 0,9648 / 0,9931 |
| Especificidad | — | 0,9792 | — |

**Tabla 3.3 — La "brecha" entre protocolos desaparece.** Fuente: `datos/comparacion_protocolos.csv` (XGBoost).

| Protocolo | F1 media por pliegue con ceros | F1 agregado |
|---|---|---|
| LOSO | 0,7379 | 0,9667 |
| GroupKFold5 | 0,9672 | 0,9681 |
| LeaveOneShapeOut | 0,9809 | 0,9817 |

**Lectura.** Con la definición anterior, LOSO parecía 0,23 puntos de F1 peor que GroupKFold5 (0,7379 frente a 0,9672); con la agregada, la diferencia es de 0,0014 (0,9667 frente a 0,9681). En GroupKFold5 y LeaveOneShapeOut no hay pliegues indefinidos y las dos definiciones casi coinciden. La dispersión por pliegue en LOSO es grande (F1 de 0,3511 a 0,9931 entre los pliegues con inundación): el mínimo corresponde al factor 1,2 (`runs/metrics_xgboost.json`, `LOSO.by_factor["1.20"].f1` = 0,3511), cerca del inicio de la inundación.

**Pruebas** (`tests/ml/bench/test_score_predictions.py`, `test_metrics.py`): `test_perfect_predictions_score_f1_one_even_with_a_fold_without_flooding`, `test_per_fold_dispersion_reports_the_undefined_fold`, `test_the_legacy_zero_fill_mean_is_kept_under_an_explicit_name` y, en el test de paridad, `test_the_new_classifier_value_is_pooled_and_differs_from_the_legacy_mean`.

### Límites que quedan

- **`by_factor` no cambió.** Sigue promediando con ceros: en XGBoost LOSO, los factores 0,20 y 1,00 dan F1 = 0,0 (`runs/metrics_xgboost.json`, `LOSO.by_factor`). No debe citarse para factores sin inundación.
- **El evaluador legado no cambió.** `ml/evaluator.py` y los JSON de `outputs/metrics/` (por ejemplo `metrics_classifier.json`, con F1 0,7408) conservan la definición anterior. Esas cifras no deben citarse.
- **Sin intervalos de confianza.** La dispersión por pliegue se reporta, pero no un IC por remuestreo de grupos (M-04, M-11, abiertos).
- **Umbral 0,5 implícito** en el clasificador (M-09, abierto).

### Texto para la tesis

> Las métricas de clasificación se calculan sobre la matriz de confusión agregada de todas las predicciones fuera de pliegue de cada protocolo, y la variación entre pliegues se informa aparte (mínimo, mediana y máximo), excluyendo los pliegues donde la métrica no está definida. Esta elección importa en la validación que retiene factores de intensidad: cinco de los 25 factores no producen ningún nodo inundado y, si en esos pliegues se asigna F1 = 0 y luego se promedia, XGBoost obtiene un F1 de 0,7379 en lugar del valor agregado de 0,9667, y un predictor perfecto obtendría 0,80. La diferencia aparente entre LOSO y GroupKFold5 en versiones anteriores de este trabajo era un efecto de esa agregación: con la matriz agregada, los F1 son 0,9667 y 0,9681.

---

## 4. M-08, M-10 y T-19 — Predicciones fuera de pliegue persistidas y matriz de confusión

### Problema y evidencia

`evaluate_candidate()` produce una predicción por fila, pero `main.py` la descartaba (`_, metrics = evaluate_candidate(...)`), y los reportes guardaban solo promedios por pliegue, sin `prep_id` ni hiperparámetros. No se podía reconstruir una matriz de confusión ni saber de qué conjunto venía cada reporte (auditoría §6.1; anexo 02, M-08 y M-10). La única matriz persistida era la de la tabla SQL `oof_predictions`, solo para XGBoost con GroupKFold5.

### Causa

El banco se diseñó para rankear con promedios por pliegue; la persistencia por fila y la matriz de confusión no formaban parte del reporte (anexo 02 §4.1).

### Qué cambió

| Commit | Archivo | Cambio |
|---|---|---|
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py` | Clave `confusion_matrix` = {tn, fp, fn, tp} agregada por protocolo; especificidad en `classifier`; `by_shape` y `end_to_end_pooled` (RMSE, MAE y error de volumen total agregados) |
| `b95a480` | `main.py`, `swmm_resilience/ml/bench/reports.py` | `--bench-evaluate` guarda las predicciones fuera de pliegue de cada familia y de cada referencia en `outputs/bench/oof/<prep_id>/<nombre>.parquet`, y un `provenance.json` con `prep_id`, hiperparámetros, referencias, fecha y qué mide cada protocolo |
| `4a2a87c` | `scripts/ejecutar_evaluacion_v4_21.py` | La corrida `tabular_v4_21` persiste predicciones y métricas al terminar cada etapa |
| `e26fc5a`, `284927d` | `scripts/informe_evaluacion.py`, `datos/tables_provenance.json`, `datos/README.md` | Las tablas registran el comando, el commit, el SHA-256 de cada archivo de predicciones leído y de cada CSV escrito |

### Verificación

**Tabla 4.1 — Matrices de confusión agregadas, `tabular_v4_21`.** Fuente: `datos/matriz_confusion.csv` (TN, FP, FN, TP); especificidad de `datos/runs/metrics_*.json`, `classifier.specificity`. En cada fila TN + FP + FN + TP = 154 080.

| Modelo | Protocolo | TN | FP | FN | TP | Especificidad |
|---|---|---|---|---|---|---|
| XGBoost | LeaveOneShapeOut | 106 526 | 1329 | 380 | 45 845 | 0,9877 |
| Forma más cercana | LeaveOneShapeOut | 106 556 | 1299 | 889 | 45 336 | 0,9880 |
| XGBoost | LOSO | 105 610 | 2245 | 879 | 45 346 | 0,9792 |
| Interpolación de factor | LOSO | 105 617 | 2238 | 416 | 45 809 | 0,9792 |
| XGBoost | GroupKFold5 | 105 711 | 2144 | 844 | 45 381 | 0,9801 |
| Interpolación de factor | GroupKFold5 | 105 617 | 2238 | 416 | 45 809 | 0,9792 |

**Antes.** La auditoría regeneró las predicciones en una carpeta temporal (no persistidas) y obtuvo, para XGBoost en LeaveOneShapeOut con `tabular_v3_17` y 160 000 filas: TN 112 153, FP 1605, FN 373, TP 45 869 (auditoría §6.2). Los totales difieren porque el conjunto de filas es otro; las dos matrices no se comparan celda a celda.

**Tabla 4.2 — Falsos negativos por volumen real.** Fuente: `datos/banda_limitrofe.csv`. La banda [1, 2) m³ contiene solo el entero 1 (volumen continuo de ≈ 0,5 a 1,5 m³; ver [`03`](03-resolucion-de-la-etiqueta.md) §6.1).

| Modelo | Protocolo | FN total | FN con volumen real en [1, 2) m³ | FN fuera de la banda | FN con volumen real ≥ 100 m³ |
|---|---|---|---|---|---|
| XGBoost | LeaveOneShapeOut | 380 | 333 | 47 | 0 |
| Forma más cercana | LeaveOneShapeOut | 889 | 611 | 278 | 0 |
| XGBoost | LOSO | 879 | 443 | 436 | 4 |
| Interpolación de factor | LOSO | 416 | 393 | 23 | 0 |
| XGBoost | GroupKFold5 | 844 | 435 | 409 | 5 |
| Interpolación de factor | GroupKFold5 | 416 | 393 | 23 | 0 |

En LeaveOneShapeOut, 333 de los 380 falsos negativos de XGBoost están en la banda de 1 m³ y ninguno tiene volumen real ≥ 100 m³. La matriz completa por clase de volumen y las métricas con y sin la banda están en [`03`](03-resolucion-de-la-etiqueta.md) §6.1–6.2 (F1 0,9817 con todas las filas; 0,9847 sin la banda, como análisis de sensibilidad).

**Trazabilidad.** `datos/tables_provenance.json` registra el SHA-256 de los tres archivos de predicciones (`inputs_sha256`) y de los cuatro CSV (`outputs_sha256`), el comando y el commit `e26fc5a` con árbol versionado limpio (`git_dirty: false`). Pruebas: `test_confusion_matrix_counts_every_test_row`, `test_a_false_negative_and_a_false_positive_show_up_in_the_confusion_matrix`, `test_bench_evaluate_persists_oof_baselines_and_provenance`, `test_write_provenance_records_prep_id_params_baselines_and_protocol_meaning`.

### Límites que quedan

- **Los archivos de predicciones no están versionados** (`outputs/` está en `.gitignore`). Se reproducen con los comandos de `datos/README.md` (≈ 6 min) y se verifican con los SHA-256 de `tables_provenance.json`.
- **Los JSON `metrics_<familia>.json` de `--bench-evaluate` no llevan el `prep_id` en el nombre**: un archivo de una corrida anterior puede quedar junto a un `provenance.json` más nuevo (deferido; ver `04`).
- **Métricas que siguen faltando** (M-10 parcial): MCC, exactitud balanceada y PR-AUC.
- **Umbral de decisión** de 0,5 implícito y probabilidades sin calibrar (M-09, abierto).
- **La lectura de los FN de la banda** como dificultad propia de la frontera es una interpretación (ver `03` §6.3).

### Texto para la tesis

> Las predicciones fuera de pliegue de cada modelo y de cada referencia se guardan por fila, junto con el identificador del conjunto de datos preparado, los hiperparámetros y la huella SHA-256 de cada archivo, de modo que todas las tablas de evaluación pueden reconstruirse sin reentrenar. Con ellas se obtiene la matriz de confusión agregada. Para XGBoost en la validación por forma retenida, sobre 154 080 casos nodo-escenario, hay 45 845 verdaderos positivos, 106 526 verdaderos negativos, 1329 falsos positivos y 380 falsos negativos (recall de 0,9918 y especificidad de 0,9877). De esos 380 falsos negativos, 333 corresponden a nodos cuyo volumen reportado es de 1 m³, el mínimo que la etiqueta considera inundación, y ninguno a nodos con 100 m³ o más.

---

## 5. Comandos usados para las cifras derivadas de este documento

Todos leen archivos existentes y no escriben en el repositorio. Se ejecutan desde la raíz del worktree con `../../../venv/Scripts/python.exe`.

**Tabla por forma, participación en el error cuadrático y comparación con la referencia** (lee `datos/por_forma.csv`):

```python
import pandas as pd, numpy as np
d = pd.read_csv("docs/auditoria/correcciones/datos/por_forma.csv")
x = d[(d.modelo == "xgboost") & (d.protocolo == "LeaveOneShapeOut")].set_index("forma")
b = d[d.modelo == "baseline_forma_mas_cercana"].set_index("forma")
x["sse"] = x.rmse_vol ** 2 * x.n_rows
share = 100 * x.sse / x.sse.sum()                      # % del error cuadrático por forma
w = x.sse.idxmax()                                     # shape_quasi_uniform_block, 71,6 %
rmse_sin_peor = np.sqrt((x.sse.sum() - x.sse[w]) / (x.n_rows.sum() - x.n_rows[w]))   # 57,32
print(w, share[w], rmse_sin_peor, share.sort_values(ascending=False).head(3).cumsum().tolist())
print((x.error_vol_pct.abs() > 20).sum(), x.error_vol_pct.abs().median())              # 12; 18,2
print((x.rmse_vol < b.rmse_vol.reindex(x.index)).sum(),                                 # 31
      (x.error_vol_pct.abs() < b.error_vol_pct.reindex(x.index).abs()).sum(),          # 33
      (x.f1 > b.f1.reindex(x.index)).sum())                                             # 25
```

**Predicciones de la interpolación idénticas en LOSO y GroupKFold5, y composición de los pliegues** (lee los archivos de predicciones y el conjunto preparado, ignorados por git):

```python
import pandas as pd, numpy as np
o = pd.read_parquet("outputs/bench/oof/ca706e27654e0cf5/baseline_interpolacion_factor.parquet")
a = o[o.protocol == "LOSO"].sort_values("sample_idx"); b = o[o.protocol == "GroupKFold5"].sort_values("sample_idx")
print(np.abs(a.y_pred_reg.to_numpy() - b.y_pred_reg.to_numpy()).max())                # 0.0
P = "outputs/bench/prepared_v4_21/ca706e27654e0cf5/"
f = pd.read_parquet(P + "folds.parquet"); k = pd.read_parquet(P + "keys.parquet")
t = f[(f.protocol == "GroupKFold5") & (f.split == "test")].merge(
    k.reset_index(drop=True).rename_axis("sample_idx").reset_index(), on="sample_idx")
print(t.groupby("fold_id").factor_mult.unique().apply(lambda v: sorted(np.round(v, 1))))
```

**Predictor perfecto sobre los datos reales** (`tabular_v4_21`):

```python
import pandas as pd
from swmm_resilience.ml.bench.evaluate import score_predictions
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PreparedDataset
prepared = PreparedDataset.load("outputs/bench/prepared_v4_21/ca706e27654e0cf5")
t = prepared.folds[prepared.folds["split"] == "test"]; idx = t["sample_idx"].to_numpy()
y = prepared.y_clf.to_numpy()[idx]
oof = pd.DataFrame({"sample_idx": idx, "protocol": t["protocol"].to_numpy(), "fold_id": t["fold_id"].to_numpy(),
                    "y_pred_clf": y, "y_prob_clf": y.astype(float), "y_pred_reg": prepared.y_reg.to_numpy()[idx]},
                   columns=list(OOF_COLUMNS))
for p, m in score_predictions(prepared, oof, {"prep_id": prepared.prep_id}).items():
    print(p, m["classifier"]["f1"], m["classifier_legacy_mean_zero_fill"]["f1"], m["classifier_per_fold"]["f1"]["n_undefined"])
# GroupKFold5 1.0 1.0 0 · LOSO 1.0 0.8 5 · LeaveOneShapeOut 1.0 1.0 0
```

---

## 6. Tabla completa por forma — XGBoost y forma más cercana, LeaveOneShapeOut

Fuente: `datos/por_forma.csv` (columnas `n_rows`, `n_positives`, `f1`, `rmse_vol`, `error_vol_pct`). La columna "% del error cuadrático" es derivada: `rmse_vol² × n_rows` de la forma dividido por la suma de las 40 formas (comando de la §5). Ordenada por esa columna. Las formas con menos de 4000 filas perdieron celdas por el filtro de continuidad.

| # | Forma | Filas | Positivos | F1 XGBoost | RMSE XGBoost (m³) | Error vol. total XGBoost (%) | % del error cuadrático (XGBoost) | RMSE forma más cercana (m³) | Error vol. total forma más cercana (%) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `shape_quasi_uniform_block` | 4 000 | 1 206 | 0,9925 | 557,92 | −52,1 | 71,63 | 647,68 | −61,3 |
| 2 | `shape_sustained_plateau_14h` | 4 000 | 1 201 | 0,9950 | 160,34 | −18,5 | 5,92 | 211,84 | −28,0 |
| 3 | `shape_twin_plateau_15h` | 4 000 | 1 186 | 0,9945 | 130,26 | 11,8 | 3,90 | 148,29 | 19,3 |
| 4 | `long_moderate` | 4 000 | 1 184 | 0,9950 | 95,39 | 7,8 | 2,09 | 141,93 | −15,4 |
| 5 | `shape_triple_peak_long_decreasing` | 4 000 | 1 163 | 0,9881 | 88,85 | 14,6 | 1,82 | 221,31 | −48,8 |
| 6 | `shape_slow_broad_rise` | 4 000 | 1 186 | 0,9962 | 85,35 | −15,7 | 1,68 | 647,68 | 158,2 |
| 7 | `shape_gentle_broad_long` | 4 000 | 1 192 | 0,9971 | 83,67 | −11,2 | 1,61 | 141,93 | 18,2 |
| 8 | `shape_double_peak_equal_separated` | 4 000 | 1 183 | 0,9911 | 82,94 | −25,0 | 1,58 | 181,06 | −53,4 |
| 9 | `shape_central_flat_plateau` | 4 000 | 1 209 | 0,9883 | 71,01 | −24,3 | 1,16 | 78,33 | −29,0 |
| 10 | `shape_double_peak_intensifying` | 4 000 | 1 174 | 0,9949 | 68,76 | −22,0 | 1,09 | 50,49 | −10,3 |
| 11 | `shape_five_pulse_8h5` | 4 000 | 1 140 | 0,9828 | 60,98 | 20,5 | 0,86 | 103,75 | 48,2 |
| 12 | `common_storm` | 4 000 | 1 172 | 0,9932 | 55,45 | 19,4 | 0,71 | 78,33 | 40,8 |
| 13 | `shape_5h` | 4 000 | 1 190 | 0,9945 | 53,77 | 18,3 | 0,67 | 18,84 | −10,1 |
| 14 | `shape_recession_rebound` | 4 000 | 1 150 | 0,9820 | 48,82 | 15,5 | 0,55 | 160,98 | −57,5 |
| 15 | `new_hydrograph_2` | 4 000 | 1 195 | 0,9933 | 47,93 | −23,1 | 0,53 | 63,59 | −31,2 |
| 16 | `shape_back_loaded_late_peak` | 4 000 | 1 172 | 0,9974 | 47,05 | 29,9 | 0,51 | 78,55 | 62,2 |
| 17 | `shape_early_peak_secondary_shoulder` | 4 000 | 1 148 | 0,9828 | 44,58 | 18,4 | 0,46 | 99,27 | 61,0 |
| 18 | `shape_early_broad_6h45` | 4 000 | 1 177 | 0,9962 | 42,52 | −12,6 | 0,42 | 99,27 | −37,9 |
| 19 | `shape_four_pulse_cluster` | 4 000 | 1 126 | 0,9770 | 40,06 | 33,3 | 0,37 | 24,64 | 14,9 |
| 20 | `shape_triple_peak_short` | 4 000 | 1 154 | 0,9764 | 38,22 | 23,2 | 0,34 | 115,40 | 75,6 |
| 21 | `shape_terminal_peak_7h` | 4 000 | 1 172 | 0,9953 | 37,84 | −12,1 | 0,33 | 41,36 | −17,9 |
| 22 | `shape_delayed_abrupt_peak` | 4 000 | 1 159 | 0,9906 | 37,83 | 16,7 | 0,33 | 41,36 | 21,7 |
| 23 | `shape_double_peak_close` | 4 000 | 1 151 | 0,9807 | 34,76 | −15,5 | 0,28 | 15,12 | −2,4 |
| 24 | `shape_symmetric_moderate` | 4 000 | 1 160 | 0,9923 | 31,64 | −21,6 | 0,23 | 42,17 | −32,0 |
| 25 | `shape_late_peak_4h` | 4 000 | 1 170 | 0,9966 | 28,51 | −18,5 | 0,19 | 24,64 | −13,0 |
| 26 | `shape_long_recession_tail` | 4 000 | 1 158 | 0,9746 | 26,97 | 23,4 | 0,17 | 79,29 | 84,4 |
| 27 | `shape_front_loaded_broad` | 4 000 | 1 177 | 0,9970 | 21,94 | −6,7 | 0,11 | 15,12 | 2,5 |
| 28 | `shape_flashy_long_tail_9h5` | 4 000 | 1 162 | 0,9906 | 20,57 | −6,8 | 0,10 | 79,29 | −45,8 |
| 29 | `shape_short_flat_top` | 3 840 | 1 210 | 0,9838 | 18,76 | −19,8 | 0,08 | 7,15 | 2,4 |
| 30 | `base` | 3 840 | 1 187 | 0,9879 | 18,37 | −4,3 | 0,07 | 28,99 | −27,5 |
| 31 | `new_hydrograph` | 3 840 | 1 225 | 0,9484 | 18,00 | 19,6 | 0,07 | 28,99 | 37,8 |
| 32 | `shape_2h` | 3 840 | 1 196 | 0,9900 | 14,10 | 2,1 | 0,04 | 7,15 | −2,3 |
| 33 | `shape_offset_double_peak_2h5` | 3 840 | 1 129 | 0,9813 | 13,95 | 6,7 | 0,04 | 43,04 | 47,0 |
| 34 | `shape_sharp_peak_2h15` | 3 840 | 1 138 | 0,9844 | 12,15 | 2,6 | 0,03 | 13,15 | −16,3 |
| 35 | `shape_stepped_plateau_1h45` | 3 840 | 1 128 | 0,9809 | 11,33 | 10,5 | 0,03 | 13,15 | 19,5 |
| 36 | `shape_micro_burst_45min` | 3 360 | 1 139 | 0,9608 | 6,39 | −13,7 | 0,01 | 5,18 | −16,0 |
| 37 | `common_bogota` | 3 200 | 1 056 | 0,9188 | 5,32 | 31,6 | 0,01 | 70,55 | 436,9 |
| 38 | `shape_flash_flood_urban` | 3 360 | 1 137 | 0,9642 | 4,40 | −5,4 | 0,00 | 13,65 | −53,8 |
| 39 | `rare_extreme` | 2 880 | 1 005 | 0,9316 | 3,25 | −18,1 | 0,00 | 14,74 | 116,8 |
| 40 | `shape_ultra_short_impulse` | 2 400 | 858 | 0,8798 | 2,38 | 19,5 | 0,00 | 28,25 | 341,4 |
