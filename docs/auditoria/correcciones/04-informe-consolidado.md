# Corrección 04 — Informe consolidado de las correcciones de la auditoría (pasos 1 a 6)

- **Fecha:** 2026-10-11
- **Rama:** `worktree-optuna-tuning-familia-ganadora`, commits `b831f0f` … `867ce6e` más el commit de esta documentación. Base de la serie: `9ad4df0`. Sin push (pendiente de autorización).
- **Spec y plan:** [`../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md`](../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md), [`../../superpowers/plans/2026-10-11-correcciones-auditoria.md`](../../superpowers/plans/2026-10-11-correcciones-auditoria.md).
- **Auditoría de origen:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) y anexos 01–05.
- **Documentos de detalle:** [`01-evaluacion-honesta.md`](01-evaluacion-honesta.md), [`02-trazabilidad-y-seguridad.md`](02-trazabilidad-y-seguridad.md), [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md). Datos: [`datos/`](datos/README.md).

**Advertencia de alcance para todas las cifras de evaluación.** XGBoost con los hiperparámetros por defecto de `config.yaml` (no el modelo ajustado con Optuna); una sola red; contrato `tabular_v4_21`; 154 080 filas = 963 de las 1000 corridas (forma, factor) × 160 nodos, tras el filtro de continuidad de 5 % (`datos/tables_provenance.json`; `datos/runs/provenance.json`).

---

## 1. Estado por hallazgo

| Hallazgo | Sev. (auditoría) | Estado | Commits | Documento |
|---|---|---|---|---|
| M-01 — LOSO/GroupKFold5 miden interpolación; sin referencias | 🔴 | **Corregido en la evaluación.** Dos referencias sin aprendizaje en el banco; qué mide cada protocolo queda escrito en los reportes; LeaveOneShapeOut declarado como protocolo principal; medido con `tabular_v4_21` | `9c8db1b`, `02cdaab`, `6c48c46`, `833760d`, `5f0586d`; medición `a26155e`, `4a2a87c`, `4717885`, `395a2ee`, `e26fc5a`, `284927d` | [01 §1](01-evaluacion-honesta.md) |
| M-02 — el volumen generaliza mal con formas nuevas | 🔴 | **Medido; mitigado en parte; abierto.** El contrato `tabular_v4_21` (anterior a esta serie) mejora el NSE oráculo, pero una forma concentra el 71,6 % del error cuadrático | `b831f0f`, `bd0f23f`; medición (ídem) | [01 §2](01-evaluacion-honesta.md) |
| M-03 — F1 promediado por pliegue con ceros | 🟠 | **Corregido en el banco.** `classifier` es el valor agregado; el anterior se conserva con nombre explícito. `by_factor` y el evaluador legado no cambian | `b831f0f`, `2871917`, `bd0f23f`, `5528479` | [01 §3](01-evaluacion-honesta.md) |
| M-08 / T-19 — el banco no persiste predicciones fuera de pliegue | 🟡 | **Corregido.** `--bench-evaluate` y el script `tabular_v4_21` guardan predicciones por fila y procedencia | `b95a480`, `4a2a87c`, `e26fc5a` | [01 §4](01-evaluacion-honesta.md) |
| M-10 — faltan matriz de confusión y otras métricas | 🟡 | **Parcial.** Matriz de confusión, especificidad, desglose por forma, MAE y error de volumen total agregados. Faltan MCC, exactitud balanceada y PR-AUC | `b831f0f`, `bd0f23f` | [01 §4](01-evaluacion-honesta.md) |
| V-01 (y M-13 en parte) — figuras en muestra sin rotular | 🟠 | **Corregido para las figuras nuevas** de curvas y comparación por factor y para `--evaluate-shapes`. Figuras existentes no regeneradas; `--evaluate-generalization` y `--evaluate-validation-shapes` sin rótulo | `93e34b4` | [02 §1](02-trazabilidad-y-seguridad.md) |
| V-07 — `halve_timeseries.py` sobrescribe la red base | 🟠 | **Corregido** | `57dc170` | [02 §2](02-trazabilidad-y-seguridad.md) |
| M-07 / V-04 — modelo ≠ `promotion.json`; figuras sin procedencia | 🟠 | **Corregido en `predict_network` y en las figuras de `main.py` y `runner.py`.** Otros cargadores siguen sin verificación | `8f17147`, `0c8a3f4` | [02 §3](02-trazabilidad-y-seguridad.md) |
| Defecto nuevo: pruebas que escribían en `outputs/metrics` real | — | **Corregido** (pruebas aisladas y guardia automático) | `398d6bc` | [02 §4](02-trazabilidad-y-seguridad.md) |
| D-04 / H-06 — resolución de la etiqueta | 🟠 / 🟡 | **Investigado (spike). Decisión A/B pendiente del autor.** Recomendación: ruta A | `2035a51`, `867ce6e` y el commit de esta documentación (dos correcciones menores) | [03](03-resolucion-de-la-etiqueta.md) |

---

## 2. Antes y después de las métricas clave

"Antes" = cifras de la auditoría, medidas con el contrato `tabular_v3_17` (17 features) sobre 160 000 filas sin filtro de continuidad (PreparedDataset `af3751551ca5df98`; anexo 02). "Después" = esta serie, contrato `tabular_v4_21` sobre 154 080 filas. **Cambian a la vez el contrato y el conjunto de filas: no es un experimento controlado.** La tabla describe dos estados del proyecto; las diferencias no se atribuyen a un cambio concreto. Siglas de fuentes: A02 = `../anexos/02-ml-protocolo-evaluacion.md`; AC = `../2026-10-10-auditoria-completa.md`; CP = `datos/comparacion_protocolos.csv`; MC = `datos/matriz_confusion.csv`; PF = `datos/por_forma.csv`.

| Métrica | Modelo | Protocolo | Antes | Fuente antes | Después | Fuente después (columna) |
|---|---|---|---|---|---|---|
| F1 agregado | XGBoost | LOSO | 0,9656 | A02 M-03 | 0,9667 | CP `f1_agregado` |
| F1 media por pliegue con ceros (definición anterior) | XGBoost | LOSO | 0,7408 | A02 M-03 | 0,7379 | CP `f1_media_por_fold_con_ceros` |
| F1 agregado | XGBoost | LeaveOneShapeOut | 0,9789 | A02 M-01 | 0,9817 | CP `f1_agregado` |
| RMSE ext. a ext. agregado (m³) | XGBoost | LOSO | 68,43 | A02 M-01 | 57,89 | CP `rmse_vol_agregado_m3` |
| RMSE ext. a ext. medio por pliegue (m³) | XGBoost | LOSO | 55,3 | A02 M-04 | 46,28 | CP `rmse_vol_media_por_fold_m3` |
| RMSE ext. a ext. agregado (m³) | XGBoost | LeaveOneShapeOut | 136,1 | A02 M-04 | 106,21 | CP `rmse_vol_agregado_m3` |
| RMSE ext. a ext. medio por forma (m³) | XGBoost | LeaveOneShapeOut | 79,7 | A02 M-04 | 56,81 | CP `rmse_vol_media_por_fold_m3` |
| NSE del regresor oráculo | XGBoost | LeaveOneShapeOut | 0,8141 | AC §3.1 | 0,8906 | CP `nse_regresor_oracle` |
| NSE del regresor oráculo | XGBoost | LOSO | 0,9530 | AC §3.1 | 0,9675 | CP `nse_regresor_oracle` |
| NSE del regresor oráculo | XGBoost | GroupKFold5 | 0,9553 | AC §3.1 | 0,9694 | CP `nse_regresor_oracle` |
| Error de vol. total por forma (mín.; máx.) | XGBoost | LeaveOneShapeOut | −63,3 %; +216,3 % | A02 M-02 | −52,1 %; +33,3 % | PF `error_vol_pct` |
| Parte del error cuadrático de la peor forma (`shape_quasi_uniform_block`) | XGBoost | LeaveOneShapeOut | 61,35 % | A02 M-02 | 71,6 % | Derivado de PF (`rmse_vol² × n_rows`) |
| RMSE ext. a ext. agregado (m³) | Interpolación de factor | LOSO | 9,00 | A02 M-01 | 9,13 | CP `rmse_vol_agregado_m3` |
| F1 agregado | Interpolación de factor | LOSO | 0,9718 | A02 M-01 | 0,9718 | CP `f1_agregado` |
| RMSE ext. a ext. agregado (m³) | Forma más cercana | LeaveOneShapeOut | 170,2 | A02 M-01 | 172,86 | CP `rmse_vol_agregado_m3` |
| F1 agregado | Forma más cercana | LeaveOneShapeOut | 0,9758 | A02 M-01 | 0,9764 | CP `f1_agregado` |
| Predictor perfecto, F1 media por pliegue con ceros / agregado | — | LOSO | 0,80 / — | A02 M-03 | 0,8000 / 1,0000 | Comando de [01 §5](01-evaluacion-honesta.md) |

**Matrices de confusión de XGBoost (TN / FP / FN / TP).**

| Protocolo | Antes (160 000 filas; AC §6.2) | Después (154 080 filas; MC) |
|---|---|---|
| GroupKFold5 | 111 431 / 2327 / 877 / 45 365 | 105 711 / 2144 / 844 / 45 381 |
| LOSO | 111 412 / 2346 / 890 / 45 352 | 105 610 / 2245 / 879 / 45 346 |
| LeaveOneShapeOut | 112 153 / 1605 / 373 / 45 869 | 106 526 / 1329 / 380 / 45 845 |

Las matrices "antes" vienen de predicciones regeneradas por la auditoría en una carpeta temporal; no se persistieron. Las "después" se reconstruyen desde archivos con SHA-256 registrado (`datos/tables_provenance.json`).

La única comparación **controlada** entre contratos (mismas filas) es la del RMSE medio por forma en LeaveOneShapeOut: 78,0368 m³ (`tabular_v3_17`) frente a 56,8088 m³ (`tabular_v4_21`), en `../../resultados/2026-10-06-explicacion-completa.md` §11.1. No se re-ejecutó aquí; la corrida de esta serie reproduce el 56,8088 (CP `rmse_vol_media_por_fold_m3`).

---

## 3. Resultados vigentes: XGBoost frente a las referencias en los tres protocolos

Fuente: CP (métricas) y MC (matriz). Todas sobre 154 080 filas.

| Protocolo | Qué mide | Modelo | F1 agregado | RMSE agregado (m³) | RMSE medio por pliegue (m³) | Error vol. total (%) | NSE oráculo | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOSO | Interpolación entre factores | Interpolación de factor | 0,9718 | 9,13 | 4,77 | −0,39 | 0,9992 | 105 617 | 2238 | 416 | 45 809 |
| LOSO | ídem | XGBoost | 0,9667 | 57,89 | 46,28 | −5,03 | 0,9675 | 105 610 | 2245 | 879 | 45 346 |
| GroupKFold5 | ídem, 5 factores por pliegue | Interpolación de factor | 0,9718 | 9,13 | 7,47 | −0,39 | 0,9992 | 105 617 | 2238 | 416 | 45 809 |
| GroupKFold5 | ídem | XGBoost | 0,9681 | 56,19 | 55,44 | −4,64 | 0,9694 | 105 711 | 2144 | 844 | 45 381 |
| **LeaveOneShapeOut** | **Formas de hidrograma no vistas (principal)** | Forma más cercana | 0,9764 | 172,86 | 98,14 | −2,98 | 0,7102 | 106 556 | 1299 | 889 | 45 336 |
| **LeaveOneShapeOut** | ídem | **XGBoost** | **0,9817** | **106,21** | **56,81** | −7,98 | **0,8906** | 106 526 | 1329 | 380 | 45 845 |

**Cómo leerla.**

- En LOSO y GroupKFold5 la referencia sin aprendizaje gana: esos protocolos miden interpolación entre factores. Con esta malla, GroupKFold5 da a la referencia de interpolación exactamente las mismas predicciones que LOSO (diferencia máxima 0,0), porque ningún pliegue retiene dos factores contiguos; comprobado por el implementador (informe de la tarea 8) y repetido para [01 §5](01-evaluacion-honesta.md).
- En LeaveOneShapeOut, XGBoost supera a la forma más cercana en volumen y, por poco, en F1; la referencia tiene menor error de volumen total.
- La interpolación usa respaldo en 1,25 % de sus filas y la forma más cercana en 0,42 % (`datos/runs/provenance.json`, `baseline_fallback`).
- La AUC-ROC de las referencias no se interpreta (no producen probabilidades).

---

## 4. Resolución de la etiqueta (D-04 / H-06): decisión pendiente

Resumen de [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md):

- En la muestra re-simulada (17 escenarios, 15 con desborde; registro por nodo en 15 escenarios, 13 con desborde), la etiqueta `V_rpt ≥ 1 m³` equivale a `V_continuo ≥ t` con t en (0,4988; 0,5002] m³, sin discrepancias en 2400 casos nodo-escenario: es una cuantización determinista, no ruido (03 §4).
- La integral del `.out` con el `REPORT_STEP` actual (3 min) no es una fuente mejor: se aleja del volumen continuo hasta 9,5 m³ por nodo (03 §3.2).
- En LeaveOneShapeOut, 333 de los 380 falsos negativos de XGBoost tienen `V_rpt = 1` (`datos/banda_limitrofe.csv`); sin esa banda, el F1 pasa de 0,9817 a 0,9847 (03 §6.2, solo como sensibilidad, con exclusión asimétrica).
- Re-simular las 1000 corridas costaría del orden de 17–20 min de SWMM, pero obligaría a rehacer el dataset, la BD, el `prep_id`, el banco, Optuna, la validación y todas las cifras (03 §5).

**Recomendación del spike: ruta A** (mantener la resolución y el umbral, declararlos como supuesto y reportar la matriz con la banda separada). **La decisión es del autor y está pendiente.** El texto de la tesis sobre la etiqueta (03 §8.2) y la presentación de la Tabla 4.2 de `01` dependen de esa decisión: con la ruta B cambiarían las etiquetas, el `prep_id` y todas las cifras de este informe.

---

## 5. Limitaciones de las cifras de este informe

1. **XGBoost sin ajustar.** Hiperparámetros por defecto de `config.yaml` (200 árboles, profundidad 6, tasa 0,05, submuestreo 0,8). El modelo ajustado con Optuna no se evaluó contra las referencias. La corrida usa un script propio (`scripts/ejecutar_evaluacion_v4_21.py`) y no el CLI del banco, para no sobrescribir los reportes históricos ni `outputs/models` (registro de decisiones R1).
2. **Una sola red, 160 nodos.** Ningún protocolo retiene nodos ni redes.
3. **963 de 1000 corridas.** El filtro de continuidad deja fuera 37 celdas (forma, factor), concentradas en las formas cortas o extremas (`shape_ultra_short_impulse` 10, `rare_extreme` 7, `common_bogota` 5; informe de la tarea 8). Esas formas son también las peor clasificadas (`datos/por_forma.csv`).
4. **Referencias con respaldo.** 1,25 % (interpolación) y 0,42 % (forma más cercana) de sus filas no usan su fuente primaria.
5. **La referencia de forma más cercana** define "cercana" solo por duración y tiempo al pico, con descriptores tomados del conjunto `tabular_v3_17` `2db0578f875526cb` (mismas 40 formas; R7).
6. **Comparaciones antes/después no controladas** (§2).
7. **Sin intervalos de confianza.** Se reporta la dispersión por pliegue, pero no IC por remuestreo de grupos (M-04, M-11). Las unidades independientes son 25 factores y 40 formas, no 154 080 filas.
8. **Concentración del error.** El RMSE agregado de LeaveOneShapeOut depende en un 71,6 % de una forma; sin ella sería 57,32 m³ (derivado de `datos/por_forma.csv`, [01 §2](01-evaluacion-honesta.md)).
9. **Etiqueta con resolución de 1 m³** (decisión pendiente, §4).
10. **Predicciones por fila no versionadas** en git (`outputs/` ignorado); se reproducen y se verifican con los SHA-256 de `datos/tables_provenance.json`.

---

## 6. Lo que no se hizo y por qué

| Tema | Hallazgo | Por qué no se hizo | Dónde queda |
|---|---|---|---|
| `InitFlow` fijo en todas las corridas | H-01 🟠 | Exige re-simular un subconjunto para medir sensibilidad; fuera del alcance de los pasos 1–6 (spec). Su impacto es la cifra del auditor (anexo 01) y no se re-simuló | Plan de la auditoría, P1 punto 9 |
| Variables de volumen del evento más allá de `tabular_v4_21` | H-02 / M-02 | Fuera de alcance por diseño (spec §7): se decide después de medir, y la medición está en `01` §2 | Decisión de investigación del autor |
| Ablación de las features de volumen bajo LeaveOneShapeOut | M-02, M-12 | No estaba en la spec | — |
| Error de continuidad y evidencia SWMM en la BD; proveniencia de modelos en SQL | D-02, D-03, H-04 | Fuera del alcance de los pasos 1–6 | Plan de la auditoría, P1 punto 11 |
| Verificación de `promotion.json` en otros cargadores (`ScenarioPredictor`, SHAP, `predict_from_inp` y escritorio) | M-07 / V-04 | La spec §2.8 acota la verificación a `ml/predict.py` (R8) | `02` §3 |
| Regenerar las figuras existentes con el modelo declarado | V-01, V-04 | El modelo final de la tesis no está fijado (Parte 2) | — |
| Rótulo en `--evaluate-generalization` y `--evaluate-validation-shapes` | M-13 | La spec §2.6 solo cubría `--evaluate-shapes` | — |
| Curvas fuera de muestra a partir de las predicciones fuera de pliegue | V-01 | Mejora opcional (spec §7) | — |
| Métrica principal del ranking (media por pliegue) e IC por grupo | M-04, M-11 | No estaba en la spec; `end_to_end_pooled` ya da el agregado | — |
| Sesgo de retransformación `expm1(log1p)`; regresor subajustado | M-05 | No estaba en la spec | — |
| Familias con hiperparámetros heredados; SVM justo; selección final XGBoost + MLP | M-06, M-09 (SVM) | Es la **Parte 2**, subproyecto aparte con su propia spec | Parte 2 |
| Umbral de decisión explícito y calibración | M-09 | No estaba en la spec | — |
| Tuning con Optuna no anidado | M-15 | No estaba en la spec | Parte 2 |
| `python -m swmm_resilience.ml.train --help` entrena | T-04 | No estaba en la spec, aunque la auditoría lo emparejaba con V-07 | — |
| Parser de respaldo que lee "Ponded Depth" | H-07 | Fuera del alcance del spike (sin cambios de código); recomendado en `03` §8 | Independiente de la ruta A/B |
| GUI: tabla y mapa con modelos distintos | V-06 | No estaba en la spec | — |
| Push a `origin` | — | Se pide al autor al final (R3) | — |

---

## 7. Pruebas

| Momento | Resultado | Fuente |
|---|---|---|
| Línea base (commit `1ecca50`) | 1049 pasan, 0 fallan | Spec §1; commit `3f7a628` corrige una cifra inicial de 1058 que contaba puntos de líneas de advertencia |
| Tras cada tarea | 1065 (T1) · 1073 (T2) · 1093 (T3) · 1098 (T4) · 1103 (T5) · 1107 (T6) · 1128 (T7) · 1145 (R10) · 1154 (R9) · 1159 · 1165 (T8) | Registro de ejecución (notas de trabajo no versionadas) |
| **Cierre de la serie (esta documentación)** | **1164 pasan, 1 falla, 3 deseleccionadas (`scale`), 187 s.** La falla es `tests/desktop/test_results_tab.py::test_refresh_tree_populates_children` (`TclError: invalid command name "tcl_findLibrary"`), un fallo de entorno de Tcl/Tk conocido desde la auditoría (AC §3.2). Re-ejecutado ese archivo: **7 pasan**. Total en verde: **1165** (116 más que la línea base) | `../../../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=<temporal corto> -o addopts="-m 'not scale'"` |

---

## 8. Minucias diferidas (registro de revisiones)

No cambian ninguna cifra; se registran para no perderlas.

| Tarea | Minucia |
|---|---|
| T1 (métricas) | El RMSE se calcula de dos formas (`np.sqrt(np.mean(...))` y `mean_squared_error`); `shape_breakdown_entry` con arreglos vacíos emite un aviso |
| T2 (`score_predictions`) | La prueba de enrutamiento no lo ejercita (todas las clasificaciones son correctas); el nombre de una prueba de paridad promete más de lo que comprueba; `score_predictions` y `pooled_classifier_metrics` duplican la concatenación y los conteos; `protocols` no se valida contra el dataset (un nombre mal escrito o una cadena suelta); se perdió el razonamiento de un docstring de la prueba de paridad |
| T3 (referencias) | Las pruebas de interpolación usan sobre todo puntos medios (sin malla irregular ni prueba específica de GroupKFold5); ninguna prueba detecta estandarizar con estadísticos de la forma de test; aserción débil `match="C"`; no se comprueba `inf` en los descriptores; los mensajes de duplicados/NaN de la forma más cercana no tienen prueba; los contadores de respaldo son totales sobre protocolos (documentado); uso de `assert` (se omite con `-O`) |
| T4 (persistencia) | La prueba del CLI no relee el contenido del parquet/JSON de las referencias; `metrics_<familia>.json` no lleva `prep_id` y puede quedar junto a un `provenance.json` más nuevo; importaciones a mitad de `test_reports.py`; sin prueba de NaN→null en la procedencia; sin prueba de que otro `ValueError` se propague |
| T5 (en muestra) | La prueba comprueba el texto de la nota, no su posición; la comprobación de código fuente se ancla al primer `if args.evaluate_shapes:`; con más de 30 nodos la nota queda cerca del rótulo "Node ID" |
| T6 (`halve_timeseries`) | Aserciones laxas (`in`); `main()` sin prueba; ventana de carrera entre comprobar y escribir; sin aviso si no hay filas que dividir |
| T7 (promoción) | `promotion.json`/`manifest.json` malformados dan `JSONDecodeError` sin mensaje de dominio; granularidad de procedencia por carpeta; el guardia de `outputs/metrics` no tiene prueba propia; queda una carpeta vacía `outputs/metrics/factor_comparison/` |
| T8 (corrida y tablas) | Las funciones internas de `main()` del script de corrida no tienen prueba; el docstring sobre `ensure_allowed_output` promete más de lo que hace; `informe_evaluacion.py --out` no tiene guardia; `runs/*.json` no tienen los campos `n_runs`/`git_dirty` (declarado en `datos/README.md`); `git_dirty` ignora archivos sin seguimiento (definición escrita en el JSON) |
| T9 (spike) | Afirmaciones prudentes pero no sostenidas por medición: el sesgo del regresor como regresión a la media (03 §6.4), la cercanía de los FP a la frontera (03 §6.2), la conclusión de diseño de 03 §7.1 y la redacción de 03 §3; el commit del spike lleva la firma de otro modelo distinto del previsto (solo nota) |

---

## 9. Seguimiento: correcciones pendientes en otros documentos

Este commit corrige en `03` el conteo de escenarios (17 simulados, 15 con desborde; 15 con registro por nodo, 13 con desborde) y la cota de cuantización de §7.6 (≤ 0,51 m³, por el nodo `8C`, que difiere en 0,508 m³). Las mismas cifras aparecen en notas fechadas de otros documentos, que esta ronda **no** toca:

| Documento y línea | Texto actual | Corrección sugerida |
|---|---|---|
| `../2026-10-10-auditoria-completa.md`, l. 275 (nota del supuesto 3) | "en la muestra re-simulada (una red, 13 escenarios con desborde)" | "17 escenarios simulados (15 con desborde); la cota del umbral se calculó en los 15 con registro por nodo (13 con desborde)" |
| `../anexos/03-datos-persistencia-sql.md`, l. 76 (nota sobre D-04, punto 1) | ídem | ídem |
| `datos/README.md`, l. 22 | ídem | ídem |
| `../anexos/01-hidraulica-swmm.md`, l. 113 (nota de H-06) | "(≤ 0,5 m³ por debajo de 100 m³)" | "(≤ 0,51 m³)" |
| `../2026-10-10-auditoria-completa.md`, l. 215 (decisión 4, "banda 1–2 m³") | Sin precisar que es solo el entero 1 | Ya listado como no aplicado en 03 §8.1 |
| `03-resolucion-de-la-etiqueta.md`, Apéndice, `s09_notas.py` | Contiene el texto original de las notas | Se deja igual: es el registro del script que se ejecutó |

---

## 10. Decisiones tomadas durante la ejecución (resumen)

Del registro de ejecución (no versionado), las que afectan a la lectura de las cifras:

| Id | Decisión | Consecuencia |
|---|---|---|
| R1 | La medición sobre datos reales usa `scripts/ejecutar_evaluacion_v4_21.py`, no el CLI, con los hiperparámetros por defecto de `config.yaml` | Las cifras describen XGBoost sin ajustar; `outputs/models`, `outputs/bench/reports` y `ranking.csv` no se tocaron |
| R7 | La referencia de forma más cercana recibe descriptores de forma explícitos, tomados del conjunto `tabular_v3_17` | "Cercana" = por duración y tiempo al pico |
| R9 | Las referencias funcionan sobre la malla incompleta (963 de 1000 corridas) en lugar de re-ejecutar sobre las 160 000 filas sin filtrar | Referencias y XGBoost se puntúan sobre exactamente las mismas filas; 1,25 % y 0,42 % de filas de respaldo |
| R10 | Aislar las pruebas que escribían en `outputs/metrics` y añadir un guardia | Ver `02` §4 |
| R11 | El contador de respaldo de la interpolación se define frente a la malla de entrenamiento del pliegue | Los factores retenidos a propósito por GroupKFold5 no cuentan como respaldo |
| R13 | El spike añadió notas fechadas a los documentos anteriores sin reescribir el texto original | Ver `03` §8.1 y §9 de este informe |

---

## 11. Resumen para el informe de la investigación

> **Evaluación del modelo sustituto tras la auditoría del protocolo**
>
> Una auditoría interna del protocolo de evaluación identificó tres problemas que afectaban la lectura de los resultados: dos de los protocolos de validación cruzada medían interpolación y no generalización, faltaban referencias sin aprendizaje con las que comparar, y las métricas de clasificación se promediaban por pliegue de un modo que las subestimaba. Se corrigieron la evaluación y su trazabilidad, y se repitió la medición con el contrato de 21 variables, que incluye variables integradas del volumen del evento. Las cifras siguientes corresponden a XGBoost con hiperparámetros por defecto (no al modelo final ajustado), a una sola red de drenaje de 160 nodos y a 963 de las 1000 simulaciones (forma de hidrograma, factor de intensidad), las que cumplen un error de continuidad de a lo sumo 5 %, es decir, 154 080 casos nodo-escenario.
>
> **Qué mide cada protocolo.** Cuando se retiene un factor de intensidad completo (LOSO) o cinco factores por pliegue (GroupKFold5), una referencia que solo interpola el volumen entre los factores vecinos de la misma forma y nodo obtiene un RMSE de volumen de 9,13 m³, frente a 57,89 m³ de XGBoost en LOSO, y un F1 de 0,9718 frente a 0,9667. Esos protocolos miden interpolación entre factores ya vistos y no se usan como evidencia de generalización. Con la malla de 25 factores, GroupKFold5 no es más exigente que LOSO, porque ningún pliegue retiene dos factores contiguos.
>
> **Protocolo principal: forma de hidrograma no vista (LeaveOneShapeOut).** XGBoost supera a la referencia que copia el volumen de la forma de entrenamiento más parecida: RMSE agregado de 106,21 frente a 172,86 m³, NSE del volumen estimado en los nodos realmente inundados de 0,8906 frente a 0,7102 y F1 de 0,9817 frente a 0,9764. La matriz de confusión agregada de XGBoost tiene 45 845 verdaderos positivos, 106 526 verdaderos negativos, 1329 falsos positivos y 380 falsos negativos; 333 de esos falsos negativos son nodos con el volumen mínimo que la etiqueta considera inundación (1 m³ reportado) y ninguno supera los 100 m³.
>
> **La magnitud del volumen es la parte débil.** El error del volumen total por forma va de −52,1 % a +33,3 %, y una sola forma, de aporte casi uniforme en bloque, concentra el 71,6 % del error cuadrático, con su volumen subestimado en un 52,1 %. Sin esa forma, el RMSE agregado de las otras 39 sería de 57,32 m³. Con el contrato anterior de 17 variables, la auditoría había medido un NSE de 0,8141 y un error por forma de −63,3 % a +216,3 %, sobre un conjunto de filas distinto; la comparación no aísla el efecto de las nuevas variables.
>
> **Agregación de métricas.** Las métricas de clasificación se calculan sobre la matriz de confusión agregada. Promediar por pliegue asignando F1 = 0 a los pliegues sin inundaciones reducía el F1 de LOSO de 0,9667 a 0,7379, y daba 0,80 a un predictor perfecto; la aparente diferencia entre protocolos era un efecto de esa agregación.
>
> **Trazabilidad.** Las predicciones fuera de pliegue se guardan por fila con la huella SHA-256 de cada archivo, de modo que todas las tablas pueden reconstruirse sin reentrenar; las figuras evaluadas sobre datos de entrenamiento se rotulan como ajuste en muestra, y la predicción con el modelo promovido se detiene si sus archivos no coinciden con el registro de promoción.
>
> **Pendiente.** La etiqueta de inundación usa el volumen del reporte de SWMM, con resolución de 1 m³; en re-simulaciones de control equivale a un umbral de ≈ 0,5 m³ sobre el volumen continuo. Si se mantiene esa definición y se declara como supuesto (opción recomendada) o se re-simulan las 1000 corridas es una decisión aún abierta; la segunda cambiaría todas las cifras anteriores. Tampoco se han evaluado aún el modelo ajustado ni las demás familias con este protocolo.

**Fuentes de las cifras del resumen:** `datos/comparacion_protocolos.csv` (F1, RMSE, NSE), `datos/matriz_confusion.csv` (matriz), `datos/banda_limitrofe.csv` (333 FN, 0 FN ≥ 100 m³), `datos/por_forma.csv` (rango por forma, 71,6 %, 57,32 m³; derivados en [01 §5](01-evaluacion-honesta.md)), `datos/tables_provenance.json` (154 080 filas, 963 de 1000), anexo 02 (0,8141; −63,3 % a +216,3 %), [01 §5](01-evaluacion-honesta.md) (predictor perfecto 0,80), [03](03-resolucion-de-la-etiqueta.md) (≈ 0,5 m³).
