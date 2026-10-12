# Auditoría completa del proyecto — parte técnica e hidráulica

**Fecha:** 2026-10-10
**Rama:** `fix/cierre-auditoria-hidrogramas-20260927`, commit base `6e8579b`, **más** los 16 archivos con cambios sin commitear de la Fase A (`LeaveOneShapeOut`, regla inclusiva de inundación, promoción desactivada).
**Modo:** solo lectura. No se modificó código, `config.yaml`, `outputs/` ni se hizo ningún commit. Todo el material nuevo está en `docs/auditoria/`.
**Entregable:** este informe (consolidado y verificado) + 5 anexos con la evidencia, los scripts y el detalle de cada hallazgo.

| Anexo | Área | Hallazgos |
|---|---|---|
| [01-hidraulica-swmm](anexos/01-hidraulica-swmm.md) | Hidráulica / SWMM | H-01…H-14 |
| [02-ml-protocolo-evaluacion](anexos/02-ml-protocolo-evaluacion.md) | ML, protocolo de evaluación, matriz de confusión | M-01…M-22 |
| [03-datos-persistencia-sql](anexos/03-datos-persistencia-sql.md) | Datos y SQLite v17 | D-01…D-22 |
| [04-cierre-auditorias-previas-y-documentacion](anexos/04-cierre-auditorias-previas-y-documentacion.md) | Trazabilidad de auditorías previas, huérfanos, documentación | T-01…T-19 + 117 ítems trazados |
| [05-analisis-visualizacion-soporte](anexos/05-analisis-visualizacion-soporte.md) | Análisis, visualización, GUI, empaquetado | V-01…V-34 |

---

## 1. Resumen ejecutivo

**Veredicto.** La cadena hidráulica de generación de datos es sólida: las etiquetas no se invalidan. Re-simulaciones independientes con SWMM reproducen al m³ los totales de la BD v17 en 11 combinaciones forma/factor, el reescalado de hidrogramas conserva el volumen (error ≤ 0,08 %) y a factor 1,0 el pipeline reproduce el `.inp` original. Los datos están íntegros (160.000 filas = 40 formas × 25 factores × 160 nodos, sin duplicados ni huérfanos, `inunda` coincide con `vol ≥ 1` en todas las filas).

**El problema no está en los datos sino en lo que se concluye de ellos.** Hay dos hallazgos críticos, ambos del protocolo de evaluación, que cambian cómo deben leerse los resultados de la tesis:

1. **M-01 — Los protocolos LOSO y GroupKFold5 solo miden interpolación.** Agrupan por `factor_mult`, y con una malla densa de 25 factores una tabla de consulta sin aprendizaje (interpolar entre los factores vecinos del mismo nodo y forma) supera a XGBoost: RMSE de volumen **9,0 m³ frente a 68,4 m³**, F1 **0,972 frente a 0,966**. Esas cifras no demuestran que el modelo haya aprendido hidráulica.
2. **M-02 — Con formas de hidrograma no vistas, el volumen generaliza mal.** NSE del regresor **0,814** (frente a 0,953 en LOSO) y error de volumen total por forma de **−63 % a +216 %**. Una sola forma concentra el 61 % del error cuadrático. La causa más probable (H-02/M-02) es que la forma entra al modelo solo como `duracion_horas` y `tiempo_al_pico_h`, sin ninguna variable de **volumen del evento**.

Lo que sí generaliza bien es la **clasificación** inunda / no inunda con formas nuevas (F1 agregado 0,979, recall 0,992). La debilidad está en la magnitud del volumen.

**Otros siete hallazgos mayores relevantes para la tesis** (tabla completa en la sección 4):

| ID | Qué | Por qué importa |
|---|---|---|
| M-03 | F1/precisión/recall se promedian por fold con `zero_division=0` | En LOSO se reporta F1 = 0,74 cuando el valor agregado es 0,966. Un predictor perfecto obtendría 0,80. La "brecha" LOSO vs GroupKFold5 es un artefacto |
| M-07 / V-04 | El pipeline por defecto sobrescribe `outputs/models` y `promotion.json` queda falso | El modelo en disco (`660be918…`) no es el promovido (`e2e7925e…`). Mapas, SHAP y validaciones usan un modelo distinto del que se cita |
| V-01 / M-13 | Curvas de resiliencia, volumen total y `--evaluate-shapes` usan datos de entrenamiento sin rotularlo | Pueden leerse como validación y son ajuste en muestra |
| H-01 | Todos los conductos arrancan con `InitFlow` = suma de picos a factor 1 | 855 m³ almacenados en t=0; error de continuidad hasta 12,4 %; en eventos cortos a f=5 el volumen inundado sube +0,7 % a +10,5 % |
| H-02 | Las formas se normalizan por pico y ninguna feature describe su volumen | A igual duración y factor el volumen inundado difiere hasta 13,3 veces entre formas |
| D-04 / H-06 | El umbral de inundación (1,0 m³) coincide con la resolución del `.rpt` (1 m³) | Entre 27 % y 57 % de los nodos que desbordan quedan etiquetados 0; 1.902 filas valen exactamente 1,0 |
| D-02 / H-04 | La BD no contiene evidencia SWMM (error de continuidad NULL en 1000/1000 runs) | El balance de masa de las simulaciones no se puede demostrar desde los datos guardados |

**Estado de la suite de tests:** 881 tests, 880 pasan, 1 falla por entorno (tkinter/Tcl, ver 3.2).

---

## 2. Alcance, método y límites

**Alcance cubierto:** `swmm_resilience/` completo (~17.400 líneas), `main.py`, `config*.yaml`, `scripts/`, `tests/`, documentación principal y de `docs/`, y los datos reales en `outputs/` y `data/`.

**Método.**
- Cinco auditores en paralelo, uno por área: hidráulica (Fable), ML y datos y trazabilidad (Opus), análisis/visualización (Sonnet). Cada uno escribió su anexo con evidencia reproducible.
- Comprobaciones empíricas de solo lectura: ~150 simulaciones SWMM de control sobre copias en el scratchpad, bases SQLite abiertas con `mode=ro`, regeneración de predicciones fuera de fold de XGBoost (182 s, una familia), 26 pruebas de mutación sobre una copia del repo.
- Verificación independiente por parte del consolidador de los hallazgos críticos y de varios mayores (sección 3).

**Límites declarados.**
- Se audita **esta rama**. El worktree `.claude/worktrees/optuna-tuning-familia-ganadora` tiene 69 commits no integrados (fases B1, B2, C1, C2, Optuna, contrato `tabular_v4_21`); no se verificó su implementación (T-01, T-02).
- Se regeneró **solo XGBoost**. Los resultados de las otras cuatro familias (random forest, lineal, SVM, MLP) no se re-ejecutaron; los argumentos sobre ellas son de código.
- **Surrogate temporal LSTM/CNN:** no hay base ni artefactos en el árbol; M-16 es solo lectura de código.
- **No verificado:** GUI en ejecución, pipeline legacy con BD poblada, test de empaquetado en entorno limpio, unidad de las coordenadas del `.inp`, errores de continuidad reales de las 1000 simulaciones de entrenamiento (no están guardados), existencia de los `.rpt` de entrenamiento, pruebas `-m scale`.
- Un incidente menor: un auditor ejecutó `python -m swmm_resilience.ml.train --help`, que no tiene argparse e intentó entrenar. Abortó por falta de `dataset_ml.csv` sin escribir nada; se comprobó con `git status` y las fechas de `outputs/models`. Se registró como hallazgo T-04.

---

## 3. Verificación independiente de los hallazgos clave

Para no trasladar al informe afirmaciones sin contrastar, los hallazgos más importantes se re-verificaron por una vía distinta a la del auditor.

### 3.1 Resultado

| Hallazgo | Cómo se verificó | Resultado |
|---|---|---|
| M-01 baseline supera a XGBoost | Script propio: interpolación lineal leave-one-factor-out por (forma, nodo) sobre `dataset_final.csv` | **Confirmado.** RMSE 9,0 m³, F1 0,972 (idénticos a los del auditor) |
| M-02 volumen con formas nuevas | Recalculado desde el parquet de predicciones fuera de fold y los targets del PreparedDataset | **Confirmado.** NSE oracle: GroupKFold5 0,9553 · LOSO 0,9530 · LeaveOneShapeOut 0,8141. Error por forma −63,3 % a +216,3 %. `shape_quasi_uniform_block` = 61 % del SSE |
| M-03 promedio por fold con `zero_division=0` | Lectura de `ml/bench/metrics.py:41-43` y `_average`; salida de `outputs/metrics/metrics_classifier.json` | **Confirmado.** F1 publicado 0,74077 = media de 25 folds, 5 de ellos sin positivos (factores 0,2–1,0) con F1 fijado a 0. F1 agregado 0,9656 |
| M-07 / V-04 modelo ≠ promoción | `sha256sum` de `outputs/models/*.joblib` frente a `promotion.json` | **Confirmado.** `classifier.joblib` = `660be918…`; `promotion.json` declara `e2e7925e…` |
| M-08 / T-19 banco descarta las predicciones | `main.py:198`: `_, metrics = evaluate_candidate(...)` | **Confirmado** |
| D-04 volúmenes cuantizados | Conteo sobre `dataset_final.csv` | **Confirmado con corrección.** 46.236 de 46.242 positivos son enteros (el auditor decía 100 %). 1.902 valen exactamente 1,0 |
| H-01 `InitFlow` fijo | Lectura de `[CONDUITS]` del `.inp` base | **Parcialmente confirmado.** Los conductos tienen InitFlow ≠ 0 (5,5; 22; 22; 38; 50; 59 …) con `DYNWAVE`. El impacto cuantificado (855 m³, hasta 12,4 %) es del auditor y no se re-simuló |
| V-07 `halve_timeseries.py` | Lectura | **Confirmado.** Sin guarda `__main__`; escribe `DST` (el `.inp` base Qx1) al importarse (línea 35) |
| T-04 sin argparse | Lectura de `ml/train.py` | **Confirmado** (solo `if __name__ == "__main__"`) |
| Matrices de confusión | Recalculadas desde el parquet | **Confirmado** (sección 6) |
| Integridad del repo tras la auditoría | `git status`, fechas de `outputs/models` | Árbol idéntico al inicial salvo `docs/auditoria/` |

### 3.2 Tests

`pytest` completo sobre el árbol actual: **881 tests, 880 pasan, 1 falla.** El fallo es de entorno: `TclError: couldn't read file ".../pythoncore-3.12-64/tcl/tcl8.6/auto.tcl"`. Es intermitente: en una primera corrida (con `-x`) falló `test_display_image_shows_selected_map` y en la segunda `test_refresh_tree_populates_children`, ambos en `tests/desktop/test_results_tab.py`. Ninguno es un fallo de lógica del proyecto (V-34).
Adicionalmente: `tests/database` 208 passed (auditor 3) y `tests/ml/bench` 51 passed (auditor 2).

### 3.3 No re-verificado por el consolidador

Se aceptan con la evidencia del anexo, pero no se reprodujeron por separado: H-02 a H-14 (salvo lo indicado), D-01, D-02, D-03, D-05…D-22, M-04 a M-22, V-01 a V-06 y V-08 a V-34, T-01 a T-19 y el resultado de las 26 mutaciones. Cada anexo incluye los scripts para reproducirlos.

---

## 4. Matriz consolidada de hallazgos

**Escala.** 🔴 CRÍTICO: compromete la validez científica · 🟠 MAYOR: sesga resultados o produce comparaciones incorrectas · 🟡 MODERADO: limitación o riesgo latente · 🔵 MENOR: calidad.

| Área | 🔴 | 🟠 | 🟡 | 🔵 | Total |
|---|---|---|---|---|---|
| Hidráulica (H) | 0 | 2 | 6 | 6 | 14 |
| ML y protocolo (M) | 2 | 5 | 9 | 6 | 22 |
| Datos y SQL (D) | 0 | 4 | 9 | 9 | 22 |
| Visualización y soporte (V) | 0 | 8 | 18 | 8 | 34 |
| Gobierno y documentación (T) | 0 | 6 | 6 | 7 | 19 |
| **Total** | **2** | **25** | **48** | **36** | **111** |

*(El auditor de datos reportó 21 en su mensaje; el anexo lista 22: 4 mayores, 9 moderados y 9 menores.)*

### 4.1 Críticos y mayores

| ID | Sev. | Hallazgo | Impacto | Anexo |
|---|---|---|---|---|
| M-01 | 🔴 | LOSO/GroupKFold5 agrupan por factor: solo miden interpolación. No hay baselines | Baseline sin aprendizaje: RMSE 9,0 vs 68,4 m³ | 02 |
| M-02 | 🔴 | Features de forma sin volumen; el volumen falla con formas no vistas | NSE 0,814; error por forma −63 % a +216 % | 02 |
| M-03 | 🟠 | F1/precisión/recall promediados por fold con `zero_division=0` | F1 0,74 reportado vs 0,966 real | 02 |
| M-04 | 🟠 | Ranking por media no ponderada del RMSE por fold, sin IC | LeaveOneShapeOut: media 79,7 vs agregado 136,1 m³; IC95 % [66,7; 212,2]. Diferencias de ≈6 % entre familias quedan dentro del ruido | 02 |
| M-05 | 🟠 | Regresor subajustado y sesgo de retransformación `expm1(log1p)` | RMSE train 118 ≈ OOF 127 m³; volumen total −6 %; volúmenes pequeños +35 %/+49 % | 02 |
| M-06 | 🟠 | Familias comparadas con hiperparámetros fijos heredados del ganador previo | RF `max_depth=6`: NSE 0,45. El ranking compara configuraciones, no métodos | 02 |
| M-07 | 🟠 | Pipeline por defecto sobrescribe `outputs/models`; `promotion.json` falso | SHA del modelo en disco ≠ SHA promovido | 02 |
| H-01 | 🟠 | `InitFlow` fijo (suma de picos a factor 1) en las 1000 corridas | 855 m³ en t=0; continuidad hasta 12,4 %; +0,7 a +10,5 % de volumen inundado en eventos cortos a f=5 | 01 |
| H-02 | 🟠 | Formas normalizadas por pico; ninguna feature de volumen del evento | ∫q_norm varía 72×; volumen inundado hasta 13,3× a igual duración y factor | 01 |
| D-01 | 🟠 | Trigger de proveniencia solo bloquea `status`/`network_id` de un run usado | Intercambiar `scenario_id` reetiquetó 6 de 96 volúmenes sin aviso (BD sintética) | 03 |
| D-02 | 🟠 | Runs marcados `COMPLETE` desde el CSV, sin evidencia SWMM | `continuity_error_pct` NULL en 1000/1000; 988 con timestamps inicio=fin | 03 |
| D-03 | 🟠 | Proveniencia de modelos vacía (`{}`, `fit_seconds=0`); hash del contrato sobre texto literal | Los modelos se entrenan con el CSV, no con la vista SQL declarada | 03 |
| D-04 | 🟠 | Umbral 1,0 m³ = resolución del `.rpt` | 72 casos de "deja de inundar al subir el factor", todos con 1–2 m³: ruido de redondeo | 03 |
| V-01 | 🟠 | Figuras de resiliencia, volumen y factor evalúan en muestra sin rotular | 25 factores × 160 nodos de entrenamiento | 05 |
| V-02 | 🟠 | `format_node_label` no es inyectiva (`2C` y `2I` → "2") | 101 de 161 nodos comparten etiqueta | 05 |
| V-03 | 🟠 | Herramientas legacy usan el esquema pre-v17 sobre la BD v17 | `create_schema` añadió 6 tablas a una copia y luego falló | 05 |
| V-04 | 🟠 | `predict_network` carga modelos sin verificar procedencia | Figuras de sep. generadas con modelos distintos de los actuales | 05 |
| V-05 | 🟠 | `eda.py` usa targets legacy; sobre el dataset vigente los targets entran como features | Correlación y PCA contaminadas; tabla vs target vacía | 05 |
| V-06 | 🟠 | En la GUI, tabla y mapa del mismo botón pueden usar modelos distintos | | 05 |
| V-07 | 🟠 | `halve_timeseries.py` sobrescribe el `.inp` base al importarse | Invalidaría el hash de red de los modelos | 05 |
| V-08 | 🟠 | Scripts de reproducción exigen Python 3.13; el entorno es 3.12.10; `requirements.txt` sin `build` | | 05 |
| T-01 | 🟠 | La Fase A existe solo como cambios sin commitear | Riesgo de pérdida y divergencia | 04 |
| T-02 | 🟠 | 69 commits en un worktree no integrado; un documento sin rastrear describe un contrato inexistente en esta rama | | 04 |
| T-03 | 🟠 | 8 de 26 mutaciones no son detectadas por los tests | `tiempo_al_pico_h` en inferencia, nodos con base 0, `drain_down_hours`, cobertura de junctions, conversión ×1000 del `.rpt`, entre otras | 04 |
| T-04 | 🟠 | `python -m swmm_resilience.ml.train --help` ejecuta el entrenamiento | El README lo recomienda | 04 |
| T-05 | 🟠 | Si falta `FLOW_UNITS` en el `.inp`, la guarda pasa en silencio (SWMM asume CFS) | Error de unidades de 28,3× | 04 |
| T-06 | 🟠 | La inferencia no valida el contrato de features que sí usa el entrenamiento | Un NaN obligatorio se imputaría sin aviso | 04 |

> **Nota de corrección (2026-10-11, spike D-04):** en la fila D-04, atribuir los 72 casos a "ruido de redondeo" es solo parcialmente correcto. Se re-simularon 12: en los 12 el volumen continuo baja, en 7 con bajadas ≥ 0,3 m³ y en 2 con bajadas < 0,1 m³ (casos de borde). Las otras 60 no se verificaron. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §4.1.

### 4.2 Moderados (resumen; detalle en los anexos)

| Área | IDs y tema |
|---|---|
| Hidráulica | H-03 entrenamiento uniforme y sincrónico vs inferencia con CSV arbitrarios · H-04 error de continuidad no guardado · H-05 "drenaje" de 6 h es una rampa (+24,5 % de volumen de entrada en `shape_2h`) · H-06 etiqueta con resolución 1 m³ · H-07 parser de respaldo lee "Ponded Depth" (todas las etiquetas → 0) · H-08 capacidad solo aguas arriba, no del conducto de salida |
| ML | M-08 el banco no persiste OOF ni proveniencia · M-09 umbral 0,5 implícito, probabilidades sin calibrar, SVM `predict` ≠ `predict_proba` en 2,5 % · M-10 faltan matriz de confusión, MCC, PR-AUC, desglose por forma · M-11 pocas unidades independientes, sin incertidumbre · M-12 ablación de forma con LOSO por factor; SHAP asume árbol · M-13 `--evaluate-shapes` evalúa en entrenamiento · M-14 imputación por mediana de `upstream_capacity_lps` en 57.000 filas (35,6 %) · M-15 tuning Optuna no anidado · M-16 surrogate temporal con desajuste train/inferencia |
| Datos | D-05 una lectura aplica migraciones sin respaldo · D-06 `--persist-sql` no atómico ni idempotente · D-07 `WorkflowLock` casi sin uso · D-08 `reset_db` puede borrar tablas v17 homónimas · D-09 dos esquemas incompatibles con tablas homónimas · D-10 hash de red depende de `autocrlf` · D-11 datos canónicos sin versionar ni manifiesto · D-12 hash de escenario no cubre el hidrograma · D-13 fase 2 del cutover sin implementar |
| Visualización | V-09…V-26: etiquetas "XGBoost" fijas, visor de BD inexistente, filtros ausentes, métricas no condicionadas, "resiliencia" binaria, 49 posiciones de nodos apilados, ejes log con ceros, excepciones tragadas, plan EDA con pasos "completed" fijos, seguridad de hilos de la GUI, `parse_numeric_values` con coma decimal, calidad de tests, dependencias, grafo Obsidian desactualizado, GUI sin el banco de 5 familias, título "(Qx1)" fijo |
| Gobierno | T-07 ningún documento menciona `--bench*` ni LeaveOneShapeOut · T-09 `config_7shapes.yaml` revertiría la Fase A · T-10 deriva de entorno (Python 3.12/3.13/3.11; `optuna` no en requirements) · T-11 `scenario_predict.py` mal clasificado como pipeline A · T-12 spec Optuna contradice la spec de la Fase A · T-13 ~69 MB de `.pptx` versionados |

Los 36 hallazgos menores están en los anexos.

---

## 5. Lo que está bien y se verificó

- **Etiquetas:** re-simulaciones independientes reproducen al m³ la BD v17 (11 combinaciones forma/factor).
- **Hidrogramas:** el reescalado conserva el volumen (≤ 0,08 %); a f=1 el pipeline reproduce el `.inp` original; Qx2/Qx3 equivalen a Qx1×2 y ×3.
- **Topología y capacidad:** Manning dentro de ±3,7 % de la capacidad de SWMM; pendientes y offsets correctos; la red es un árbol acíclico (el cálculo de `q_pico_acum` es válido aquí).
- **Train/serve skew:** las features dinámicas se calculan igual en entrenamiento e inferencia (con la salvedad H-03).
- **Datos:** `integrity_check` limpio en las dos BD, sin violaciones de FK; cobertura 40×25×160 completa; BD idéntica a `dataset_final.csv`; la BD de 19 formas es subconjunto exacto de la de 40; hashes de red, modelos y contrato cuadran; el manifiesto del banco apunta a la BD actual.
- **Reproducibilidad de XGBoost:** el banco y el legacy dan métricas LOSO idénticas dígito a dígito (semilla 42).
- **Métricas de comparación:** `analysis/model_comparison.py` coincide con sklearn; `torch.load` usa `weights_only=True`; `factor_comparison.py` falla con error explícito ante nodos inconsistentes.
- **Tests de BD y banco:** 208 y 51 pasan.

---

## 6. Próximo paso: matriz de confusión del/los modelo(s) elegido(s)

### 6.1 Estado actual (hallazgos M-08, M-10, T-19)

- `evaluate_candidate()` ya produce una predicción por fila (etiqueta, probabilidad, volumen), pero `main.py:198` la descarta y los reportes solo guardan promedios por fold, de los que **no se puede reconstruir** una matriz de confusión.
- Lo único persistido por fila es la tabla `oof_predictions` en SQLite, solo para XGBoost con GroupKFold5 por factor (TN=111.468, FP=2.290, FN=920, TP=45.322). Difiere levemente de la matriz regenerada en 6.2 (TN=111.431, FP=2.327, FN=877, TP=45.365); no se investigó el origen de esa diferencia, que no cambia ninguna conclusión.
- Faltan en `ml/bench/metrics.py`: matriz de confusión, especificidad, MCC/balanced accuracy y PR-AUC.

### 6.2 Cifras de referencia (XGBoost, configuración actual, 40 formas, 160.000 filas)

Predicciones fuera de fold regeneradas por el auditor con `evaluate_candidate()` y recalculadas aquí. Clase positiva = `inunda` = 1, umbral 0,5, matriz agregada (pooled). **Son cifras de referencia de una sola familia, no el resultado final de la tesis.**

| Protocolo | TN | FP | FN | TP | Recall | Precisión | Especificidad | FN con vol ≤ 2 m³ |
|---|---|---|---|---|---|---|---|---|
| GroupKFold5 (por factor) | 111.431 | 2.327 | 877 | 45.365 | 0,9810 | 0,9512 | 0,9795 | 60,5 % |
| LOSO (por factor) | 111.412 | 2.346 | 890 | 45.352 | 0,9808 | 0,9508 | 0,9794 | 62,7 % |
| **LeaveOneShapeOut (por forma)** | 112.153 | 1.605 | 373 | 45.869 | 0,9919 | 0,9662 | 0,9859 | 88,2 % |

Lectura: en LeaveOneShapeOut ningún falso negativo supera 67 m³ y el 88 % tiene 2 m³ o menos. Esos casos son los de la banda 1–2 m³ de la etiqueta, que por D-04 son inestables por construcción: no deben leerse como un fallo del modelo.

> **Nota de corrección (2026-10-11, spike D-04):** "inestables por construcción" exagera. En la muestra re-simulada, la etiqueta es una función determinista del volumen continuo, sin ruido, con un umbral efectivo en (0,4988; 0,5002] m³. Lo que sí se midió es que la banda `V_rpt = 1` (volumen continuo de ≈ 0,5 a 1,5 m³) es la más próxima al umbral y concentra 333 de los 380 FN. Que esa cercanía explique los FN es una interpretación. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §6.

### 6.3 Propuesta

1. **Modelos:** una matriz por cada familia elegida (tras el tuning) y por protocolo, con **LeaveOneShapeOut como cifra principal** y LOSO/GroupKFold5 solo como referencia de interpolación (M-01).
2. **Persistir las predicciones fuera de fold** del banco (parquet por candidato, junto al `prep_id`). Es el requisito previo: `main.py:198` debe conservar el primer elemento de la tupla de `evaluate_candidate`. Resuelve a la vez M-08 y T-19.
3. **Añadir a `metrics.py`:** `confusion_matrix`, especificidad, MCC, PR-AUC y desglose por forma.
4. **Reportar:**
   - la matriz agregada (conteos y normalizada por fila real) como cifra principal;
   - la dispersión por fold (mínimo, mediana, máximo) y un IC95 % por bootstrap *por grupo* como incertidumbre;
   - recall y precisión por fold con **NaN** (no 0) cuando el fold no tiene positivos (corrige M-03);
   - una segunda matriz que separe la banda limítrofe (volumen real de 1 a 2 m³) del resto, por D-04/H-06;
   - la matriz por forma para las formas fuera de rango (`shape_ultra_short_impulse`, F1 0,844; `common_bogota`, 0,906).

   > **Nota de corrección (2026-10-11, spike D-04):** con la resolución de 1 m³, el único volumen reportado en [1, 2) m³ es el entero 1, que corresponde a un volumen continuo de ≈ 0,5 a 1,5 m³. La matriz por banda está en `correcciones/03-resolucion-de-la-etiqueta.md` §6.1.

5. **Barrido de umbral** de 0,2 a 0,8 como análisis de sensibilidad, pero con el umbral de operación elegido **en un CV interno**, nunca sobre las predicciones que se reportan.

### 6.4 Decisiones que tienes que tomar

| # | Decisión | Recomendación |
|---|---|---|
| 1 | ¿Qué modelos entran? (solo el ganador tras Optuna o las 5 familias) | Las 5 familias en el anexo; el ganador en el cuerpo de la tesis |
| 2 | ¿Protocolo principal? | LeaveOneShapeOut |
| 3 | ¿Umbral de decisión? | 0,5 como línea base y umbral de operación elegido por CV interno |
| 4 | ¿Se trata aparte la banda 1–2 m³? | Sí, en una matriz complementaria |
| 5 | ¿Matriz agregada o por fold como cifra principal? | Agregada, con la dispersión por fold como incertidumbre |

---

## 7. Plan de acción priorizado

Solo es una propuesta de orden; no se ha modificado nada.

**P0 — antes de citar cualquier resultado en la tesis**
1. Declarar que LOSO/GroupKFold5 miden interpolación, añadir baselines y pasar a LeaveOneShapeOut como protocolo principal (M-01).
2. Corregir la agregación de F1/precisión/recall (M-03) y reportar el agregado con su dispersión (M-04, M-11).
3. Persistir las predicciones fuera de fold y construir la matriz de confusión (sección 6).
4. Reconciliar el modelo en disco con `promotion.json`, bloquear la sobrescritura silenciosa de `outputs/models` y regenerar las figuras con el modelo que se declare (M-07, V-04).
5. Rotular como "en muestra" las figuras y salidas que lo son (V-01, M-13).
6. Neutralizar `halve_timeseries.py` y añadir argparse a `ml/train.py` (V-07, T-04).
7. Confirmar el commit de la Fase A y decidir la integración del worktree de 69 commits (T-01, T-02).

**P1 — validez científica y trazabilidad**
8. Decidir cómo representar el volumen del evento en las features (M-02, H-02). Es una decisión de investigación, no un parche.
9. Medir la sensibilidad a `InitFlow = 0` y al "drenaje" de 6 h con un subconjunto de re-simulaciones (H-01, H-05).
10. Leer el volumen inundado con mayor resolución que el `.rpt` y revisar el umbral (H-06, D-04).

    > **Nota de corrección (2026-10-11, spike D-04):** con el `REPORT_STEP` actual (3 min), la integral de la serie `flooding` del `.out` **no** es una fuente de mayor resolución: se aleja del volumen continuo hasta 9,5 m³ por nodo, y hasta 0,89 m³ incluso con 5 s. La única fuente medida que da el volumen continuo es `node.statistics['flooding_volume']` de pyswmm. El spike recomienda la ruta A (declarar, sin re-simular). Ver `correcciones/03-resolucion-de-la-etiqueta.md`.

11. Guardar el error de continuidad y la evidencia SWMM en la BD (H-04, D-02, D-03).
12. Reforzar los triggers de proveniencia y la atomicidad de `--persist-sql` (D-01, D-06).
13. Cerrar las 8 mutaciones sin test (T-03).

**P2 — mantenimiento**
14. Actualizar la documentación (34 discrepancias, T-07), sanear huérfanos y `.gitignore`, fijar el entorno de Python (T-08…T-17, V-08).
15. Migrar o retirar las herramientas legacy que usan el esquema pre-v17 (V-03, D-09, D-08).

---

## 8. Trazabilidad con las auditorías previas

El anexo 04 traza **117 ítems** de las auditorías previas y de los documentos de seguimiento contra el código actual.

| Estado | Ítems |
|---|---|
| Cerrado | 32 |
| Parcial | 30 |
| Abierto | 40 |
| No aplica | 13 |
| No verificado | 2 |

- De los 18 hallazgos técnicos de `AUDITORIA_SURROGATE_HIDROGRAMAS.md`: **4 cerrados** (M5, MO5, m2, m5), **11 parciales** y **3 abiertos** (M3 drenaje, MO1, MO4).
- Los que figuran como cerrados pero no tienen test que detecte su regresión están en T-03.
- Auditoría de código basura: se borraron `pruebas_locales.py`, `verificar.py` y `outputs/factor_comparisons/`. `halve_timeseries.py` sigue versionado y es peligroso (V-07). Siguen sin existir módulos del paquete inalcanzables. La auditoría clasificó mal `scenario_predict.py` como pipeline A (T-11).
- Hay **34 discrepancias documentación ↔ código** (por ejemplo: README y DICCIONARIO dicen 15 features y son 17; varios documentos dicen que `--predict` no corre SWMM y sí lo corre; `COMANDOS.md` propone un comando de validación que falla por formato).

---

## 9. Supuestos hidráulicos a declarar en la tesis

1. La red se simula con **onda dinámica** (`DYNWAVE`), paso de rutina de 5 s y hidrograma de entrada espacialmente **uniforme y sincrónico** en todos los nodos (H-03).
2. Las condiciones iniciales de los conductos (`InitFlow`) son las del `.inp` base y no cambian con el factor ni la forma (H-01).
3. La inundación es el volumen que reporta SWMM por nodo, con **resolución de 1 m³**; la etiqueta es `volumen ≥ 1 m³` (H-06, D-04).

   > **Nota de corrección (2026-10-11, spike D-04):** en la muestra re-simulada (una red, 13 escenarios con desborde), "volumen reportado ≥ 1 m³" es consistente con un umbral efectivo de ≈ 0,5 m³ sobre el volumen continuo, acotado a (0,4988; 0,5002] m³. Los desbordes menores cuentan como "no inunda".

4. El tramo final del hidrograma se extiende con un punto de caudal cero 6 h después del último dato, lo que puede crear una rampa (H-05).
5. Las formas se normalizan por pico; el modelo no recibe el volumen del evento (H-02).
6. La capacidad usada es la de los conductos **aguas arriba** (H-08).
7. La red es un árbol sin ciclos ni divisores, de modo que la suma de picos aguas arriba es válida (H-12).
8. Las etiquetas del entrenamiento no tienen guardado su error de continuidad (H-04, D-02).

---

## 10. Reproducibilidad de esta auditoría

- Cada anexo incluye sus scripts de diagnóstico (todos de solo lectura). Se ejecutan con `venv\Scripts\python.exe` desde la raíz del repo; las BD se abren con `mode=ro`.
- Las predicciones fuera de fold usadas en la sección 6 se regeneraron con `evaluate_candidate()` y la configuración de `config.yaml` sobre el PreparedDataset `af3751551ca5df98` (160.000 filas, 40 formas). El script está en el anexo 02.
- Estado del repo al cierre: idéntico al inicial, más la carpeta `docs/auditoria/`. Nada de esto está commiteado.
