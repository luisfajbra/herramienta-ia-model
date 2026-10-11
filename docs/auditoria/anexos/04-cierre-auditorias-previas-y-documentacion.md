# Anexo 04 — Cierre de auditorías previas y coherencia de la documentación

**Fecha:** 2026-10-10
**Rama auditada:** `fix/cierre-auditoria-hidrogramas-20260927`, commit base `6e8579b` (2026-09-27), **más** los cambios sin commitear del árbol de trabajo (fase A: 14 archivos `M`). El código evaluado es el del árbol de trabajo actual.
**Modo:** solo lectura. No se modificó código, documentación existente, `config.yaml` ni `outputs/`. No se hicieron commits.

---

## 1. Método

1. **Inventario de hallazgos previos.** Se leyeron íntegramente: `AUDITORIA_SURROGATE_HIDROGRAMAS.md` (incluida la sección "Seguimiento 2026-09-27"), `AUDITORIA_CODIGO_BASURA_2026-06-15.md`, `AVANCES_RECIENTES_PROYECTO.md`, `docs/superpowers/specs/2026-06-12-validation-harness-inference-fixes-design.md`, `docs/revision_academica_redes_neuronales_2026-09-22.md`, `docs/proximos_estudios.md`, `docs/superpowers/plans/2026-09-27-auditoria-fase-a-integridad-protocolo.md` (y su spec `2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md`), `docs/superpowers/plans/2026-09-29-optuna-tuning-familia-ganadora.md` y su spec. Cada hallazgo, recomendación, criterio de aceptación o tarea pendiente recibió un ID (sección 2).
2. **Contraste con el código actual.** Se leyeron los módulos implicados con número de línea (`scenario_predict.py`, `dynamic_features.py`, `labels.py`, `hydrograph_batch.py`, `timeseries_scenario.py`, `static_features.py`, `topology.py`, `contracts.py`, `swmm_api_io.py`, `ml/bench/*`, `config.py`, `config.yaml`, `main.py`). Se usaron `git log`, `git show`, `git diff` y `git ls-tree`.
3. **Calidad de los tests (lectura crítica + mutación).** Para cada comportamiento clave se aplicó una **mutación** sobre una copia del repositorio en el scratchpad (nunca sobre el repo) y se ejecutaron solo los tests focalizados. "DETECTADA" = algún test falla con la mutación; "SOBREVIVE" = todos pasan, es decir, el test no protege ese comportamiento. Línea base de la copia: 160 y 147 tests en verde (dos lotes). 26 mutaciones: **18 detectadas, 8 sobreviven**. Los resultados están en la sección 2 y el script en la sección 5.
4. **Alcanzabilidad (código basura).** Se rehízo el grafo de imports por AST (sección 5, `reachability.py`) desde las entradas de producción (`main.py`, `swmm_resilience/main.py`, `swmm_resilience/desktop/app.py`), los tests y los scripts de la raíz. Se excluyeron `venv/`, `build/`, `.claude/` (worktree), `data/`, `outputs/` y `docs/`.
5. **Documentación.** Se compararon `README.md`, `QUICKSTART.md`, `COMANDOS.md`, `comandos.txt`, `docs/FLUJO_ACTUAL.md`, `DOCUMENTACION_COMPLETA_PROYECTO.md`, `spec_tecnico_desarrollo.md` y `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md` con el código, el `--help` real y los datos (`data/`, `outputs/`).
6. **Conteo de tests sin ejecutar la suite:** `pytest --collect-only` da **881/884 recolectados (3 deselected)** y **258** en `tests/ml/bench`. La suite completa no se ejecutó.
7. **CLI ejecutados:** `python main.py --help` y `python -m swmm_resilience.ml.temporal.train_cnn --help` (usan argparse; solo lectura). **Incidente que hay que reportar:** también se ejecutó `python -m swmm_resilience.ml.train --help`. Ese módulo **no tiene argparse**, así que ignoró `--help` y empezó a entrenar. Abortó en `load_dataset` con `FileNotFoundError` porque `data/networks/chico_hydro-qx1/results/dataset_ml.csv` no existe. Se verificó después con `git status` y listando `data/networks/chico_hydro-qx1/` que no escribió ningún archivo. El incidente se convirtió en el hallazgo T-04.

**Contexto que condiciona los estados.** Existe un worktree git en `.claude/worktrees/optuna-tuning-familia-ganadora` (rama `worktree-optuna-tuning-familia-ganadora`, HEAD `70e3717`, 2026-10-06) con **69 commits no integrados** en esta rama: fases B1 (cierre de aporte), B2 (features tributarias `tabular_v4_21`), C1 (estado de drenaje), C2 (filtro de continuidad), Optuna (`tune.py`, `--bench-tune`) y promoción v4_21. **Los estados de este anexo se refieren a la rama auditada.** Cuando un pendiente aparece resuelto en esa rama se indica como "en rama no integrada (commit X)". Esa implementación **no se verificó** aquí: solo se constató que los archivos existen (`git ls-tree`) y se leyeron los mensajes de commit.

**Criterio de estado.**
- **CERRADO:** código presente + test que **detecta** la regresión (o evidencia de ejecución cuando el criterio la exige).
- **PARCIAL:** falta alguna de esas piezas.
- **ABIERTO:** sin implementar en esta rama.
- **NO APLICA:** fuera de alcance, opcional o superado.
- **NO VERIFICADO:** no se pudo comprobar sin ejecutar algo prohibido.

---

## 2. Tabla de trazabilidad

Abreviaturas: `hb` = `swmm_resilience/validation/hydrograph_batch.py`; `sp` = `swmm_resilience/ml/scenario_predict.py`; `ts` = `swmm_resilience/simulation/timeseries_scenario.py`; `wt` = cambio sin commitear del árbol de trabajo; "mut:" = resultado de la mutación.

### 2.1 `AUDITORIA_SURROGATE_HIDROGRAMAS.md` — hallazgos

| ID | Resumen | Estado | Evidencia (código/commit/test) | Qué falta |
|---|---|---|---|---|
| C1 | Features ciegas a la forma (solo pico) | PARCIAL | `ml/contracts.py:32-50`: 17 features; la forma solo entra vía `duracion_horas`/`tiempo_al_pico_h` (escalares por forma). No existe `extraction/event_features.py`. El contrato `tabular_v4_21` (6 features tributarias) solo está en la rama no integrada (`587b704`, `2236c9a`, `976f45f`) | Integrar el contrato con volumen/excedencia, ablación 17 vs nuevo y sensibilidad por forma (spec 09-27 §13) |
| C2 | Un grado de libertad; validación optimista por factor | PARCIAL | wt: `ml/bench/schemas.py:56-60` (`LeaveOneShapeOut→shape_id`), `folds.py:54-98`, `config.yaml:48`, `config.py:247`. Test `tests/ml/bench/test_folds.py:37`; mut `loso_shape_groups_by_factor` DETECTADA. Existe el PreparedDataset `outputs/bench/prepared/af3751551ca5df98` (160 000 filas, `protocol_definitions`). Dataset: 39 CSV en `data/hydrograph_shapes` + forma base = 40 formas | Sin commit. **Ningún OOF/ranking por forma ejecutado** en esta rama (`outputs/bench/reports/ranking.csv` = solo SVM, LOSO por factor, 2026-09-07, 76k filas). El evaluador legacy (`config.yaml:35-39`) y `--persist-sql` (`main.py:899`) siguen agrupando por factor |
| M1 | Features de inferencia inconsistentes con entrenamiento | PARCIAL | Picos reales + acumulado por ancestros: `extraction/dynamic_features.py:32-64`; test `tests/test_scenario_predict.py:127`; mut `accum_without_ancestors` DETECTADA. Factor promedio eliminado | `tiempo_al_pico_h` se toma de la **primera serie no vacía** (`sp:118-119`); mut `tpeak_from_last_series` **SOBREVIVE**. `duracion_horas = last_time_hours` del CSV vs la duración de la forma normalizada en entrenamiento (`simulation/hydrograph_shapes.py:35-36`). La inferencia **no** pasa por `TABULAR_V3_17.validate_frame` (`sp:136`, a diferencia de `ml/trainer.py:64`). Sin test de paridad entrenamiento↔inferencia |
| M2 | Extrapolación silenciosa | PARCIAL | `sp:75-95` (`extrapolated`), test `tests/test_scenario_predict.py:178`; mut `extrapolation_never_flags` DETECTADA | Rama de nodos con base 0 sin cobertura (mut `extrap_base0_never_flags` **SOBREVIVE**). No hay bandera de forma fuera de entrenamiento en el predictor; solo la etiqueta INTERPOLACIÓN/EXTRAPOLACIÓN por duración en `--evaluate-validation-shapes` (`main.py:~700-710`) |
| M3 | Simulación truncada (sin drenaje) | ABIERTO | `ts:120-123` añade **un único** punto cero en `last+6h`, lo que interpola una cola lineal de aporte artificial. Las formas de entrenamiento usan el mismo writer (`simulation/batch.py:60-62`). Formas que no terminan en cero: `long_moderate` (q final 0.35), `shape_2h` (0.0747), `shape_5h` (0.0744). mut `drain_zero_point_off`/`drain_end_time_off` DETECTADAS, pero mut `batch_ignores_drain_down_param` (no propagar `drain_down_hours` desde el batch) **SOBREVIVE** | Política de cierre del aporte, verificación de drenaje desde `.out` y regeneración de la verdad SWMM afectada. En rama no integrada: `92d2c23`/`53de401` (B1), `6c0ac1a`/`95d888a` (C1) |
| M4 | Solo nodos con aporte directo | PARCIAL | Lógica correcta: `hb:283-284` (`all_node_ids = predictor.node_ids`), `hb:196-209`; test del helper `tests/test_hydrograph_batch.py:394` | El cableado no está protegido: mut `truth_only_inflow_nodes_M4` (usar `expected_nodes`) **SOBREVIVE** porque el predictor falso tiene los mismos nodos que el CSV (`tests/test_hydrograph_batch.py:19,51`). La auditoría pedía "mantener prueba automatizada" |
| M5 | `>` vs `>=` y valores distintos de umbral | CERRADO | `extraction/labels.py:19-23` (`>=`), `config.yaml:14` (1.0), fallback CLI 1.0 (`main.py:771-778`). wt `ml/bench/quality.py:95` usa `flood_label`. Tests `tests/test_labels.py:33`, `tests/ml/bench/test_quality.py:106`; mut `labels_ge_to_gt`, `quality_revert_gt`, `label_zero_threshold_counts_dry` DETECTADAS | La conciliación clasificador/regresor se trata en m1 |
| MO1 | `q_pico_acum_base` asume picos simultáneos; capacidad solo inmediata | ABIERTO | Sin cambios: `extraction/topology.py:84-91`. La spec 09-27 §10 acepta como cierre **documentar** la limitación, y no está documentada (ni en `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md` ni en `README.md`) | Documentar el proxy y su limitación; la capacidad aguas abajo queda pospuesta |
| MO2 | Métricas que maquillan errores | PARCIAL | Harness de validación: CSI, PR-AUC, MAE/RMSE condicionales y métricas por escenario (`hb:360-385, 418-428`); test `tests/test_hydrograph_batch.py:297,312` | El banco (`ml/bench/metrics.py:36-70`, `evaluate.py:85-122`) no tiene CSI, PR-AUC, desglose por forma ni por escenario (solo `by_factor`); falta el error relativo por nodo inundado |
| MO3 | Sin validación de unidades | PARCIAL | `hb:102-107` aborta si `FLOW_UNITS≠LPS`; test `tests/test_hydrograph_batch.py:339`; mut `flow_units_check_off` DETECTADA. El INP base declara `FLOW_UNITS LPS` (línea 6) | Si `FLOW_UNITS` **falta**, pasa en silencio (el default de SWMM es CFS). El chequeo solo existe en el harness, no en la extracción de entrenamiento. La conversión ×1000 de la ruta `swmm_api` (`simulation/swmm_api_io.py:303-307`) no está testeada (mut `swmm_api_rpt_scale_wrong` **SOBREVIVE**). Las unidades no se registran en ningún manifiesto |
| MO4 | `base_inflow_lps` es un pico | ABIERTO | `extraction/static_features.py:7-26,110` toma el `max()`; `contracts.py:41` lo describe como "Persisted baseline node inflow"; `DICCIONARIO…:71` dice "baseline por nodo" | Documentarlo como pico (o renombrarlo) |
| MO5 | Volumen total por escenario no reportado | CERRADO | `hb:366-385, 459-470` (`scenario_totals.csv`, `totals_comparison.png`); consola `main.py:827-837`; test `tests/test_hydrograph_batch.py:253`; mut `error_pct_sign` DETECTADA; salidas reales en `validation_output/*/scenario_totals.csv` (2026-09-15) | — (el PNG se escribe en la raíz de `out_dir`, no en `plots/`: ver D-04) |
| MO6 | Sin tiempos SWMM vs ML | PARCIAL | `hb:298-312, 387-401`; predictor creado una sola vez (`hb:283`); test `tests/test_scenario_predict.py:188` (a nivel de predictor) | mut `batch_reloads_predictor_per_scenario` **SOBREVIVE** (nada detecta que el batch recargue modelos por escenario). `device` fijo en `"cpu"` (`hb:399`). Falta repetir la medición en la máquina final |
| m1 | `inunda_pred=1` con `vol_pred<umbral` (conciliación) | PARCIAL | Regla documentada (spec 06-12 §4e): el clasificador decide (`sp:140-146, 176-178`; `README.md:86-87`) | Regla versionada de la spec 09-27 §10 (`inunda_pred = vol_final>=1`) sin implementar. mut `regressor_on_all_nodes` **SOBREVIVE** (no se testea `vol=0` cuando la clase es 0) |
| m2 | Parsing duplicado del Node Flooding Summary | CERRADO | Única fuente: `extraction/labels.py:93-139`. Usada por `hb:206` y `main.py:298,331,1019,1045`. Residuo legacy: `simulation/runner.py:606` (pipeline A/GUI) | Retirar el residuo junto con el pipeline A |
| m3 | Parser toma el último token (ponding) | PARCIAL | Texto: `labels.py:79` (último token). swmm_api: `swmm_api_io.py:293-296` mapea cualquier columna con "volume" (ambiguo con ponding). Aviso de ponding solo en el harness (`hb:109-116`, mut DETECTADA). INP base con `ALLOW_PONDING NO` | Parser guiado por encabezados que aborte ante ambigüedad (spec 09-27 §10) |
| m4 | Errores de continuidad no registrados | PARCIAL | `hb:175-193, 314-321, 383` (columna `continuity_error_pct`); test de parseo `tests/test_hydrograph_batch.py:378` | El warning no está testeado (mut `continuity_warning_off` **SOBREVIVE**). La extracción de entrenamiento no mide ni excluye por continuidad (en rama no integrada: `30b1110`, `2bb77fc`). Evidencia: 8/24 escenarios ultrafast >5 % en `validation_output/val_extrap_ultrafast_20min/metrics_per_scenario.csv` |
| m5 | No se verifica el hash del INP | CERRADO | `hb:118-137` + `--allow-inp-mismatch`; test `tests/test_hydrograph_batch.py:350`; mut `inp_hash_check_off` DETECTADA; `outputs/models/training_inp_hash.txt` = `19a8768c…` = MD5 del INP de `config.yaml:2` | Matiz: MD5 (no SHA-256) y, si falta el archivo de hash, solo emite un warning (`hb:119-125`) |

### 2.2 `AUDITORIA_SURROGATE…` — fortalezas, recomendaciones y seguimiento

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| F1 | El INP base nunca se modifica | CERRADO | `ts:89,134-135`; test `tests/test_timeseries_scenario.py:141` | — |
| F2 | Validación estricta del CSV de hidrogramas | CERRADO | `validation/hydrograph_csv.py:42-80` (columnas exactas, numéricos, finitos, no negativos) | La cifra de "9 reglas" no se recontó (no verificado) |
| F3 | Regresor `log1p`/`expm1` + clip ≥ 0 | CERRADO | `sp:145-146`; `ml/bench/evaluate.py:169`; test `tests/test_scenario_predict.py:159` | — |
| F4 | Evaluación en 3 niveles + `log_nse` OOF | CERRADO | `ml/bench/metrics.py:36-70`; `evaluate.py:118-123` | — |
| F5 | Conversión 10⁶ L → m³ documentada | CERRADO | `labels.py:30-31,86`; `swmm_api_io.py:304` | Ver MO3: la ruta swmm_api no tiene test |
| R1 | (C1/C2) Descriptores de evento | PARCIAL | = C1 | = C1 |
| R2 | (C2) Diversidad de eventos + LOSO por forma | PARCIAL | 40 formas × 25 factores × 160 nodos = 160 000 filas (`data/training/dataset_final.csv`); LeaveOneShapeOut en wt | Ejecutar y reportar el protocolo por forma |
| R3 | (M1) Corregir la inferencia | PARCIAL | = M1 | = M1 |
| R4 | (M3) Colchón de drenaje + regla "termina en ~0" | ABIERTO | = M3 (el aviso sí existe: `ts:94-109`, mut DETECTADA) | = M3 |
| R5 | (M5/MO3) Umbral único y `FLOW_UNITS` | PARCIAL | M5 CERRADO; MO3 PARCIAL | = MO3 |
| R6 | (M2) Reportar extrapolación | PARCIAL | = M2 | = M2 |
| R7 | (MO2) CSI, PR-AUC, condicionales, por escenario | PARCIAL | = MO2 | = MO2 |
| R8 | (MO5) Totales por escenario | CERRADO | = MO5 | — |
| R9 | (MO6) Tiempos por escenario, carga única | PARCIAL | = MO6 | = MO6 |
| R10 | (Estratégico) Scaffold temporal CNN/LSTM | NO APLICA | La spec 09-27 §4/§12 lo condiciona a una brecha residual medida | — |
| S1 | Implementar Leave-One-Shape-Out (prio 1) | PARCIAL | = C2 | Tabla por forma (media, dispersión, resultado por forma retenida) |
| S2 | Features de evento (prio 2) | ABIERTO | = C1 en esta rama (v4_21 en rama no integrada) | Integrar y reentrenar bajo LeaveOneShapeOut |
| S3 | Drenaje y trazabilidad (prio 3) | ABIERTO | `outputs/models/promotion.json`: `classifier_sha256 = e2e7925e…`, prep `4a48359e7e5b53c6`, 2026-09-07. SHA-256 real de `outputs/models/classifier.joblib` = `660be918…` (coincide con el árbol de `HEAD`, commit `6e8579b`). Las salidas de `validation_output/` (2026-09-15) no registran los hashes de modelo, dataset ni protocolo | Manifiesto de promoción coherente; hashes en cada resultado de validación |
| S4 | Decidir red neuronal (prio 4) | ABIERTO | Depende de S1/S2/S5 | — |
| S5 | Reevaluar las 5 familias con folds por forma | ABIERTO | Sin ejecución en esta rama (ver C2). La rama no integrada declara un "benchmark de 5 familias" (`51c923d`), no verificado | Ejecutar con folds idénticos |
| OPT-1 | Optuna no implementado | ABIERTO | No hay `ml/bench/tune.py`; `requirements.txt` sin `optuna`; `--help` sin `--bench-tune` | Ver OP-1…OP-9 |
| OPT-2 | Naive Bayes no integrado | NO APLICA | Opcional y de baja prioridad; el registro solo tiene 5 familias (`ml/bench/registry.py:36-37`) | — |
| OPT-3 | Tuning solo dentro del train de cada fold por forma | ABIERTO | No implementado; además la spec 09-29 §10 se desvía de este criterio (T-12) | Selección anidada o holdout de formas |
| FP-1 | Seleccionar la familia candidata con OOF por forma | ABIERTO | Sin OOF por forma | — |
| FP-2 | Ejecutar Optuna sobre la ganadora | ABIERTO | = OPT-1 | — |
| FP-3 | Reentrenar y comparar con los mismos folds | ABIERTO | — | — |
| FP-4 | Naive Bayes como baseline opcional | NO APLICA | = OPT-2 | — |

### 2.3 `AUDITORIA_CODIGO_BASURA_2026-06-15.md`

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| B0 | "0 módulos del paquete huérfanos" | CERRADO (sigue vigente) | `reachability.py`: 99 módulos del paquete; 92 alcanzables desde producción; **0 inalcanzables** desde cualquier entrada. Solo vía tests: `database.recovery`, `database.upgrade`, `database.workflow_lock` (API documentada en `COMANDOS.md:284-288`), `ml.temporal.compare_surrogate`, `train_cnn`, `train_surrogate` (tienen `__main__` propio) | — |
| B1 | Borrar `pruebas_locales.py` | CERRADO | Eliminado en `8195315` (2026-06-24) | — |
| B2 | Borrar `verificar.py` | CERRADO | Eliminado en `8195315` | — |
| B3 | Borrar `halve_timeseries.py` (sobrescribe el INP de producción) | ABIERTO | Sigue versionado (último commit `46c5152`); `DST` = el mismo INP de `config.yaml:2`; nadie lo importa (`reachability`: importado por []) | `git rm` (el riesgo de sobrescritura sigue vigente) |
| B4 | Borrar `outputs/factor_comparisons/` | CERRADO | No existe | — |
| B5 | Cachés `__pycache__/`, `.pytest_cache/` | NO APLICA | Se regeneraron (132 `__pycache__` fuera de `venv`, `.pytest_cache/`); están ignorados | — |
| BR1 | Retiro del pipeline A (decisión aparte) | ABIERTO | `ml/train.py`, `predict_from_inp.py`, `predict_tabular.py`, `preprocessing.py` y `analysis/eda.py` solo son alcanzables desde `desktop/app.py` y `swmm_resilience/main.py`. No funciona sin regenerar datos: faltan `data/networks/chico_hydro-qx1/results/dataset_ml.csv` (`config.py:43`) y `data/training/swmm_resilience.db` (`config.py:42`). **Corrección:** `ml/scenario_predict.py` **no** es pipeline A; lo usa la validación activa (`main.py → hb:34`), ver T-11 | Ejecutar el plan `2026-08-21-sqlite-v17-legacy-retirement` excluyendo `scenario_predict.py` |
| BR2 | `validation_output_smoke/` (¿fixtures?) | ABIERTO | `validation_output_smoke/inp/hydrograph_{2h,5h}.inp` son byte a byte idénticos a `validation_output/inp/…` (`git diff --no-index` vacío). Ningún test ni módulo los referencia. Sus CSV fuente (`data/hydrographs/`) se borraron en `3e17363` | Se pueden retirar; ver T-08 |
| BR3 | Contenido de `outputs/` | NO APLICA | Artefactos locales ignorados; `outputs/new_csv_comparison/` ya no existe | — |
| BR4 | Conservar `comandos.txt` | NO APLICA | Se conserva, pero está desactualizado (D-15) | Actualizar o fusionar con `COMANDOS.md` |
| BR5 | Docs `.md` fuera de alcance | NO APLICA | Se cubren en la sección 4 de este anexo | — |
| BR6 | Código muerto intra-archivo no auditado | ABIERTO | No auditado en este anexo (no verificado) | Revisión dedicada con cobertura |

### 2.4 `AVANCES_RECIENTES_PROYECTO.md` (pendientes §17)

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| AV-1 | Validar la Fase 1 temporal (Parquet por corrida) | NO VERIFICADO | No hay `data/networks/*/results/temporal/` en disco | — |
| AV-2 | Tabla `temporal_artifacts` | CERRADO | `database/schema.py:113`; `tests/database/test_temporal_artifacts.py` (esquema legacy, no v17) | — |
| AV-3 | Ventanas temporales | CERRADO | `ml/temporal/dataset.py:103` (`build_temporal_windows`); `tests/ml/temporal/test_window_builder.py` | — |
| AV-4 | Revisar el trade-off precisión/recall de `svc_rbf` | NO APLICA | Métricas del pipeline legacy, ya obsoletas | — |

### 2.5 Spec `2026-06-12-validation-harness-inference-fixes-design.md`

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| SP-1 | Regla de etiquetado única ≥ 1 m³ | CERRADO | = M5, m2 | — |
| SP-2 | Contrato v2 (15 features, sin `factor_mult`) | CERRADO (superado) | `factor_mult` no está en `contracts.py:32-50`; el contrato vigente es `tabular_v3_17` (17 features, spec 06-26) | Actualizar las docs que dicen 15 (D-01, D-22) |
| SP-3 | `ScenarioPredictor` (carga única, todas las junctions, prob, extrapolated) | CERRADO | `sp:41-163`; tests `tests/test_scenario_predict.py:144-205` | Ver M2/MO6 |
| SP-4a | Drenaje post-evento | PARCIAL | Implementado tal como se especificó (`ts:120-131`), pero el diseño mismo genera una cola artificial (M3) | = M3 |
| SP-4b | Guardas (unidades, ponding, hash, continuidad) | PARCIAL | `hb:84-137, 314-321` | `FLOW_UNITS` ausente; warning de continuidad sin test |
| SP-4c | Cobertura de todas las junctions | PARCIAL | = M4 | Test de cableado |
| SP-4d | Salidas nuevas (totals, timings, metrics_per_scenario, PNG, consola) | CERRADO | `hb:456-477`; `main.py:804-861` | Ruta del PNG ≠ spec (D-04) |
| SP-4e | Regla de conciliación documentada y testeada | PARCIAL | Documentada (`README.md:86-87`) | No testeada (mut `regressor_on_all_nodes` sobrevive) |
| SP-5 | Re-etiquetar, reentrenar y actualizar README/DICCIONARIO | PARCIAL | Modelos activos con 17 features (`feature_names_in_` verificado), hash INP coincide | `DICCIONARIO…` sin actualizar (D-22…D-24); `README.md` dice 15 features |
| AC-1 | `pytest tests` en verde | NO VERIFICADO | 881 recolectados; `.superpowers/sdd/…/task-5-report.md` informa 881 passed; aquí solo se ejecutaron 160+147 tests focalizados (verde) | — |
| AC-2 | Corrida `--evaluate-hydrographs` con los 3 CSV + PNG + speed-up | CERRADO | `validation_output/val_*/{scenario_totals,timings,metrics_per_scenario}.csv` (2026-09-15). Cifras del seguimiento verificadas: marathon f2.5 CSI 0.941, 28 227 vs 23 801 m³ (−15.7 %); ultrafast f3.7 CSI 0.889, 829 vs 623 m³ (−24.9 %); speed-up 29.3–50.8× y 8.7–11.2× | — |
| AC-3 | Carga de modelo y features estáticas una vez por batch | CERRADO | `hb:283`, `t_model_load_s` constante en `timings.csv` | Test a nivel de batch (MO6) |
| AC-4 | Misma función de etiquetado en entrenamiento y validación | CERRADO | = m2 | — |
| AC-5 | Dataset regenerado, modelos reentrenados, hash verificado | CERRADO | `training_inp_hash.txt` = MD5 del INP; modelos con 17 features | — |

### 2.6 Spec `2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` (fases y criterios no cubiertos arriba)

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| SC-A | Fase A: regla inclusiva, LeaveOneShapeOut, procedencia SHA-256, rechazo de caché | PARCIAL | wt: `schemas.py:53-170` (validación semántica, `prep_id`), `preprocess.py:137-230` (backup SQLite de solo lectura, `training_frame_sha256`); mut `prepared_semantics_check_off` y `held_out_groups_empty` DETECTADAS | §5 exige commit + snapshot del código + hash del diff + hashes de config/semilla: el manifiesto `af37…` **no** los tiene. Registro de exclusiones inexistente. **Sin commit** en esta rama |
| SC-B | Fase B: política de cierre del aporte y features tributarias | ABIERTO | Ausente aquí; en rama no integrada (B1/B2) | Integrar |
| SC-C | Fase C: estado `drained/not_drained/unverified`; continuidad > 5 % excluida | ABIERTO | Ausente aquí; en rama no integrada (C1/C2) | Integrar |
| SC-D | Fase D: benchmark justo (CSI, PR-AUC, macro por forma, OOF persistido, tiempos) | ABIERTO | `main.py:194-211` descarta el OOF (`_, metrics = …`); `reports.py` solo escribe métricas y ranking; el ranking usa f1/nse como desempate (`config.yaml:49-54`) | Implementar según §8 |
| SC-9 | Promoción explícita, transaccional, con guardrails | PARCIAL | No implícita: `main.py:213-223`; mut `bench_promotes_implicitly`/`promote_accepts_none` DETECTADAS | `ml/bench/promote.py:29-50` copia sin rollback, sin verificar hashes ni guardrails |
| SC-10 | Reconciliación versionada; `temporal_descriptor_out_of_range`; `shape_status` | PARCIAL | Solo la bandera de magnitud (= M2) | El resto ausente |
| SC-11 | Ejecución por etapas (piloto, checkpoints, timeout de SVM) | ABIERTO | Nada en `ml/bench/` | — |
| SC-12 | Optuna anidado sin outer test | ABIERTO | = OPT-3 | — |
| SC-2 | Registrar la inconsistencia hash/fechas sin reescribirla | CERRADO | Registrada en `AUDITORIA_SURROGATE_HIDROGRAMAS.md:315`; `promotion.json` intacto | Reconciliar (S3) |
| SC-3a | No ejecutar la spec Optuna 08-04 tal cual | CERRADO | No hay rastro de ejecución; la sustituye la spec 09-29 | — |
| SC-3b | Tiempos no medidos → nulo / `not_measured` | ABIERTO | `database/csv_backfill.py:291-295` inserta `fit_seconds=0`, `predict_seconds=0` y `hyperparameters_json='{}'`; el esquema exige `NOT NULL` (`sql/001_v17_initial.sql:208-209`) | Migración + valores reales |

### 2.7 Plan `2026-09-27-auditoria-fase-a-integridad-protocolo.md`

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| FA-1 | Catálogo de protocolos y folds por forma | CERRADO (sin commit) | wt `schemas.py:53-60,137-143`, `folds.py:54-98`; `test_folds.py:37,136` | Commit |
| FA-2 | Preparación reproducible y rechazo de caché incompatible | CERRADO (sin commit) | wt `preprocess.py:162-282`; `resolve_prepared('…/4a48359e7e5b53c6')` es rechazado (informe task-5); mut DETECTADA | Commit |
| FA-3 | Auditoría de umbral inclusiva en `quality.py` | CERRADO (sin commit) | wt `quality.py:18,95`; `test_quality.py:106`; mut DETECTADA | Commit |
| FA-4 | Promoción desactivada; LeaveOneShapeOut principal | CERRADO (sin commit) | wt `config.yaml:48,55`, `config.py:247,343`, `main.py:151,213-223` | Commit. Help/docs desactualizados (D-08, D-09) |
| FA-5 | Verificación de la fase | CERRADO | 258 tests en bench y 881 en total recolectados (coinciden con el informe); `af3751551ca5df98` existe con `protocol_definitions` | El comando del plan `--config config.yaml` (línea 297) no existe en el CLI (T-15) |

### 2.8 Plan y spec `2026-09-29-optuna-tuning-familia-ganadora*`

| ID | Resumen | Estado | Evidencia (esta rama) | En rama no integrada |
|---|---|---|---|---|
| OP-1 | Dependencia `optuna` fijada | ABIERTO | `requirements.txt` sin optuna (aunque `optuna` **está instalado** en `venv`: T-10) | `618342b` (`optuna==5.0.0`) |
| OP-2 | Fixture `sql_training_db_six_shapes` | ABIERTO | `grep six_shapes tests/` vacío | `da016a5` |
| OP-3 | Extraer `_fit_predict_fold()` | ABIERTO | Ausente en `ml/bench/evaluate.py` | `ee8248a`, `d2ec61a` |
| OP-4 | `build_tuning_folds()` | ABIERTO | No hay `tune.py` | `5b56a3b` |
| OP-5 | `SEARCH_SPACES`/`_sample_params` | ABIERTO | — | `e82ca82` |
| OP-6 | `run_study`/`TuneResult`/`save_tune_result` | ABIERTO | — | `ae37188`, `11ffd13` |
| OP-7 | `BenchTuningConfig` | ABIERTO | Ausente en `config.py` | `08b452a` |
| OP-8 | CLI `--bench-tune` | ABIERTO | `main.py --help` no lo lista | `f4b2fde` |
| OP-9 | Verificación de cierre | ABIERTO | — | no verificado |

### 2.9 `docs/revision_academica_redes_neuronales_2026-09-22.md` y `docs/proximos_estudios.md`

| ID | Resumen | Estado | Evidencia | Qué falta |
|---|---|---|---|---|
| RA-1 | MLP: épocas fijas, sin selección de pesos/early stopping | ABIERTO | `ml/bench/models/mlp_family.py:81` (`for _ in range(self.epochs)`) | Validación interna + early stopping |
| RA-2 | MLP solo en CPU (hay GPU disponible) | ABIERTO | Sin `device`/`cuda` en `mlp_family.py` | — |
| RA-3 | Particiones por factor ≠ generalización de forma | PARCIAL | = C2 | — |
| RA-4 | Espacio de búsqueda MLP (1–3 capas, weight decay, lote, Huber) | ABIERTO | Sin tuning; la spec 09-29 §6 no incluye weight decay, lote, pérdida ni early stopping | — |
| RA-5 | Ajuste (incluido el umbral del clasificador) con validación interna agrupada | ABIERTO | El umbral es el 0.5 implícito de `predict` | — |
| RA-6 | Ampliar features de forma | ABIERTO | = C1/S2 | — |
| RA-7 | Regresor condicional sobre inundadas de train; E2E sobre todos los nodos | CERRADO | `ml/bench/evaluate.py:166-169`; `evaluate.py:101-102` | — |
| RA-8 | Métricas: RMSE en m³, CSI, PR-AUC, MAE, sesgo por evento, por familia | PARCIAL | El banco tiene P/R/F1/AUC-ROC/RMSE E2E | CSI, PR-AUC, MAE E2E, sesgo, por forma |
| RA-9 | Repetir con varias semillas | ABIERTO | Semilla única (`mlp_family.py:67-68`) | — |
| RA-10 | Calibración si se ponderan clases | ABIERTO | Se usa `scale_pos_weight` (`evaluate.py:159`), sin calibración | — |
| RA-11 | Curvas de aprendizaje por eventos/familias | ABIERTO | — | — |
| RA-12 | Persistir modelo + transformadores + unidades + contrato; tiempos | PARCIAL | Pipelines joblib con imputador/escalador; `contract_id` en el manifiesto del banco; tiempos en el harness | Unidades no persistidas |
| RA-13 | Transferencia multi-red | NO APLICA | Etapa posterior; hay una sola red | — |
| RA-14 | GNN como segunda línea | NO APLICA | Futuro | — |
| RA-15 | Verificar el texto completo de la ASCE y el preprint de Zhang | NO APLICA | Pendiente bibliográfico, no de código | — |
| PE-1 | GLUE (incertidumbre paramétrica SWMM) | NO APLICA | El propio documento dice "no implica compromiso" | — |
| PE-2 | Incertidumbre del surrogate (bootstrap, conformal, cuantílica) | NO APLICA | Futuro; nada implementado | — |

**Conteo (117 ítems):** CERRADO **32** · PARCIAL **30** · ABIERTO **40** · NO APLICA **13** · NO VERIFICADO **2**.
Solo los 18 hallazgos técnicos de la auditoría surrogate (C1…m5): CERRADO 4 · PARCIAL 11 · ABIERTO 3.

**Resumen de mutaciones** (las que sobreviven = brechas de test): `tpeak_from_last_series` (M1), `extrap_base0_never_flags` (M2), `batch_ignores_drain_down_param` (M3), `truth_only_inflow_nodes_M4` (M4), `batch_reloads_predictor_per_scenario` (MO6), `regressor_on_all_nodes` (m1), `continuity_warning_off` (m4), `swmm_api_rpt_scale_wrong` (MO3/F5).

---

## 3. Hallazgos nuevos (documentación, código huérfano y gobierno)

| ID | Sev. | Hallazgo | Ubicación | Recomendación |
|---|---|---|---|---|
| T-01 | 🟠 | La fase A existe **solo como cambios sin commitear** en esta rama (14 archivos `M`, +1148/−119). En la rama no integrada sí está commiteada (`2de534d`). Hay riesgo de pérdida y de divergencia | `git status`; `swmm_resilience/ml/bench/{schemas,folds,preprocess,quality}.py`, `config.yaml`, `main.py` | Commitear o integrar la rama antes de seguir; después, reejecutar este anexo |
| T-02 | 🟠 | Hay un worktree con **69 commits no integrados** (B1, B2, C1, C2, Optuna, v4_21, promoción v4_21). La documentación y el cierre de auditoría de esta rama no lo reflejan. El documento sin rastrear `docs/justificacion-features-tributarias-tabular-v4-21.md` describe un contrato (`tabular_v4_21`) que **no existe** en esta rama | `.claude/worktrees/optuna-tuning-familia-ganadora` (HEAD `70e3717`) | Decidir el merge; mientras tanto, marcar el documento v4_21 como "pertenece a la rama X" |
| T-03 | 🟠 | Ocho comportamientos cerrados según la auditoría **no tienen test que detecte su regresión** (ver resumen de mutaciones) | `tests/test_hydrograph_batch.py:19-51` (el predictor falso tiene los mismos nodos que el CSV y `write_scenario_inp` falso con default 6.0), `tests/test_scenario_predict.py` | Añadir: paridad de `tiempo_al_pico_h`, nodo sin aporte en el batch, conteo de construcciones del predictor, propagación de `drain_down_hours`, base 0 con pico>0, `vol=0` cuando la clase es 0, warning de continuidad, escala ×1000 de la ruta swmm_api |
| T-04 | 🟠 | `python -m swmm_resilience.ml.train --help` **ejecuta el entrenamiento** (no tiene argparse) y escribiría artefactos legacy si existiera el CSV. `README.md:184` y `DOCUMENTACION_COMPLETA_PROYECTO.md:833` lo recomiendan. `swmm_resilience/main.py` (`:318`) tampoco tiene argparse | `swmm_resilience/ml/train.py:1062-1116` | Añadir argparse o retirarlo con el pipeline A; quitarlo de la documentación |
| T-05 | 🟠 | Si `FLOW_UNITS` falta en el INP, la guarda pasa en silencio (el default de SWMM es CFS) | `hb:102-107` | Tratar `None` como error o asumir CFS y abortar |
| T-06 | 🟠 | La inferencia no valida el contrato de features: entrenamiento usa `TABULAR_V3_17.validate_frame` y la inferencia indexa columnas directamente | `sp:136`; `ml/predict.py` (sin validador); compárese `ml/trainer.py:64` | Pasar `X` por el mismo validador (la rama no integrada lo hace en `70e3717`) |
| T-07 | 🟡 | **Ningún documento principal menciona `--bench*` ni LeaveOneShapeOut** (0 coincidencias en README, COMANDOS, FLUJO_ACTUAL, QUICKSTART, DOCUMENTACION_COMPLETA, spec_tecnico, comandos.txt) | ver la sección 4 | Añadir una sección "Banco de modelos" a `COMANDOS.md` y `README.md` |
| T-08 | 🔵 | `validation_output/inp/*.inp` y `validation_output_smoke/inp/*.inp`: versionados, idénticos entre sí, sin referencias y con fuentes borradas. `validation_output/` está en `.gitignore` pero contiene archivos versionados | `.gitignore` (regla "validation_output/"); commits `c703383`, `939c1e9`, `3e17363` | `git rm` de ambos pares |
| T-09 | 🟡 | `config_7shapes.yaml` es una configuración huérfana y obsoleta: `promote: "auto"`, protocolos sin LeaveOneShapeOut, sin `db_path`. Su cabecera indica "swap this file for config.yaml", lo que revertiría la fase A. La rama no integrada lo eliminó (`6f4b202`) | `config_7shapes.yaml:1-4` | Eliminarlo |
| T-10 | 🟡 | Deriva del entorno. venv = Python **3.12.10**; `scripts/reproduce_results.{sh,ps1}` exigen **3.13** y `.venv`; `pyproject.toml` ≥3.11; spec_tecnico 3.10+; plan fase A "3.11". `optuna` está instalado en `venv` pero no figura en `requirements.txt`; `build` (lo necesitan los tests de packaging) tampoco | `scripts/reproduce_results.sh:7-17`; `requirements.txt` | Unificar la versión y fijar las dependencias reales |
| T-11 | 🟡 | La auditoría de basura clasificó mal `ml/scenario_predict.py` como "Pipeline A". Lo usa la validación activa (`main.py → hb:34`); retirarlo junto con el pipeline A rompería `--evaluate-hydrographs/--evaluate-shapes/--evaluate-validation-shapes` | `AUDITORIA_CODIGO_BASURA_2026-06-15.md:52` | Corregir el alcance del retiro del pipeline A |
| T-12 | 🟡 | La spec Optuna 09-29 §10 adopta **un único estudio GroupKFold(5) sobre todo el dataset**. Eso contradice la spec 09-27 §12 ("el inner GroupKFold verá exclusivamente el train del fold exterior"). Si después se compara la configuración ajustada contra la fija sobre los mismos 40 folds por forma (spec 09-27 §12), la comparación queda contaminada. Además, la evidencia de la spec 09-29 §2 ("optuna no instalado") ya no es cierta | `docs/superpowers/specs/2026-09-29-optuna-tuning-familia-ganadora-design.md:15-27,266-269` | Prohibir explícitamente esa comparación o anidar el tuning por fold externo |
| T-13 | 🟡 | 6 presentaciones `.pptx` (~69 MB) versionadas en la raíz | `Presentacion_MLSMFP_15-09*.pptx` | Moverlas fuera del repo o a Git LFS |
| T-14 | 🔵 | `build/` contiene una copia obsoleta del paquete (`build/lib/swmm_resilience`, con `ml/bench/preprocess.py`) y temporales de pytest de 2026-09-08 (`build/pytest_hydro_*`); está ignorado | `build/` | Limpiar |
| T-15 | 🔵 | Restos de ejecuciones: `.pytest-tmp-phase-a/` (9 dirs de `test_bench_config`, 2026-09-27, sin rastrear y **sin regla de ignore**) y `tmp20qwka3w/` (vacío; el handoff de fase A lo describe como inaccesible). El plan de fase A (línea 297) documenta un comando inválido: `main.py --bench-prepare --config config.yaml` | raíz; `.superpowers/sdd/…/handoff.md:9` | Borrar; añadir `.pytest-tmp-*/` y `tmp*/` a `.gitignore`; corregir el plan |
| T-16 | 🔵 | `.gitignore` incoherente: `.claude/` está ignorado pero `.claude/settings.json` está versionado; lista blanca `!data/hydrographs/*.csv` para un directorio inexistente; regla `validation_output/` con archivos versionados | `.gitignore` | Sanear |
| T-17 | 🔵 | `diagramas_pipeline.html` está versionado, nadie lo referencia y afirma "19 formas" (hoy hay 40) | `diagramas_pipeline.html` | Actualizarlo o retirarlo |
| T-18 | 🔵 | `_run_bench` solo convierte `ValueError` en `SystemExit`. Si no hay ningún PreparedDataset, `resolve_prepared` lanza `FileNotFoundError` y sale un traceback | `main.py:175-181`; `ml/bench/preprocess.py:264-280` | Capturar también `FileNotFoundError` |
| T-19 | 🔵 | El banco no persiste las predicciones OOF ni los tiempos de inferencia (requisito de reproducibilidad de la spec 09-27 §8) | `main.py:198` | Persistir OOF por candidato (parquet) junto al `prep_id` |

---

## 4. Discrepancias documentación ↔ código

| ID | Documento:línea | Dice | Realidad (código/datos) |
|---|---|---|---|
| D-01 | `README.md:77` | "los modelos usan 15 features" | 17 (`ml/contracts.py:32-50`; `feature_names_in_` del modelo activo = 17) |
| D-02 | `README.md:67-69` | "no usa … SQLite … pipeline basado en CSV" | `--only-ml/--skip-extraction/--only-maps` y 6 flags más leen `outputs/training_v17.sqlite3` (`main.py:376,394,414,441,482,560,921,1072`); el banco lee SQL |
| D-03 | `README.md:4-5`; `QUICKSTART.md:25`; `docs/FLUJO_ACTUAL.md:145`; `spec_tecnico_desarrollo.md:427,455`; `main.py:9,72` | `--predict` = "inferencia sin SWMM" | También corre SWMM para verificar (`main.py:294-307`) |
| D-04 | `README.md:134`; spec 06-12 §4d | `plots/totals_comparison.png` | `out_dir/totals_comparison.png` (`hb:470`) |
| D-05 | `README.md:111-112` | "aborta si `FLOW_UNITS != LPS`" | No aborta si la opción falta (T-05) |
| D-06 | `README.md:178`, `:389`, `:412`; `DOCUMENTACION_COMPLETA_PROYECTO.md:106,108`; `AVANCES…:406` | `app.py`, `view_db.py`, `REVISION_DETALLADA_DELTA_Y_ESCENARIOS_PARCIALES.md` | No existen |
| D-07 | `README.md:288-291, 362-384` | regresión sobre `peak_flooding_lps`, "reemplaza a flooding_volume_m3" | El target activo es `vol_inundacion_m3` (`contracts.py:57-60`); el legacy usa `total_flood_volume_m3` (`config.py:78`). El README se contradice con su línea 73 |
| D-08 | `main.py:125-126` (help) | `--bench`: "Corre las cuatro etapas" | Corre 3; la promoción ya no forma parte (`main.py:151,213`) |
| D-09 | `main.py:123-124` (help) | `--bench-promote`: "Copia el candidato ganador" | Exige una familia explícita en `config.bench.promote` (≠ none/auto); no usa el ranking (`main.py:214-219`) |
| D-10 | `main.py:107-110` (help) | `--clf-path/--reg-path` "(.pkl)" | Los artefactos son `.joblib` |
| D-11 | `main.py:6,8`; help `:64-67`; `QUICKSTART.md:22-24` | `--skip-extraction/--only-ml/--only-maps` leen "CSV existente" | Leen SQLite (requieren `--persist-sql` previo) |
| D-12 | `COMANDOS.md:30`; `docs/FLUJO_ACTUAL.md:302,373,388` | `pd.read_csv` en `main.py:573` | `main.py:874` |
| D-13 | `COMANDOS.md:187` | guard de formas en `main.py:681-687` | `main.py:663-675` |
| D-14 | `COMANDOS.md:86-89`; `docs/FLUJO_ACTUAL.md:152,257,403` | "531 passed" | 881 recolectados (informe fase A: 881 passed) |
| D-15 | `comandos.txt:8-9` | `--skip-extraction` "re-entrena desde el CSV existente" | Lee SQLite |
| D-16 | `COMANDOS.md:237` | `--evaluate-hydrographs data/hydrograph_shapes …` | Esos CSV son `time_h,q_norm`; el loader exige `node_id,time,value_lps` (`validation/hydrograph_csv.py:43`): el comando falla |
| D-17 | `COMANDOS.md:333-334` | Optuna "solo diseñada" (spec 08-04) | La spec 08-04 quedó sustituida por la 09-29 (spec 09-27 §3) |
| D-18 | `docs/FLUJO_ACTUAL.md:4` | rama `feature/hydrograph-augmentation-sql-persistence` | Rama actual `fix/cierre-auditoria-hidrogramas-20260927` |
| D-19 | `docs/FLUJO_ACTUAL.md:12,45,164`; `config.yaml:60-61` | "≈2 docenas" de formas, "~12 MB", "~25 formas"; "175 escenarios" | 39 CSV + base = 40 formas; `dataset_final.csv` = 27.4 MB, 160 000 filas, 24 columnas; 1 000 corridas |
| D-20 | `docs/FLUJO_ACTUAL.md:300` | `load_training_frame` en `training_queries.py:334` | `:368` |
| D-21 | `docs/FLUJO_ACTUAL.md:95`; `spec_tecnico_desarrollo.md:150` | Evaluación LOSO principal, por factor | El banco usa LeaveOneShapeOut como protocolo principal (`config.yaml:48`, wt) |
| D-22 | `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md:3-7` | "15 features", entrena desde `dataset_final.csv` | 17 features; lee SQL v17 |
| D-23 | `DICCIONARIO…:13-16, 165-226` | Base `data/training/swmm_resilience.db`, `dataset_ml.csv`; lista de features legacy (`inflow_multiplier`, `invert_elev_m`, …) | No existen en disco; las features activas son las de `contracts.py:32-50` |
| D-24 | `DICCIONARIO…:71` | `base_inflow_lps` = "baseline por nodo" | Es el **pico** de la serie base (`static_features.py:7-26`) |
| D-25 | `DOCUMENTACION_COMPLETA_PROYECTO.md:648` | `ML_TARGET_REGRESSION = "flooding_volume_m3"` | `config.py:78` = `"total_flood_volume_m3"` (legacy); target activo `vol_inundacion_m3` |
| D-26 | `DOCUMENTACION_COMPLETA_PROYECTO.md:1-9, 694-740` | Se presenta como "documento maestro"; describe PCA como base | Describe el pipeline A legacy; PCA solo existe en `ml/train.py:243` (GUI) |
| D-27 | `spec_tecnico_desarrollo.md:124, 202, 213` | `flood_threshold_m3: 0.0`; `factor_mult` como feature; `inunda = vol > threshold` | 1.0; `factor_mult` es metadato; `>=` |
| D-28 | `spec_tecnico_desarrollo.md:34` | Python 3.10+ | `pyproject.toml` ≥3.11; venv 3.12.10 |
| D-29 | `spec_tecnico_desarrollo.md:451` | `--skip-simulation` solo "ya tienes los .rpt" | Rechazado sin `--skip-extraction/--only-ml` (`main.py:265-266`) |
| D-30 | `spec_tecnico_desarrollo.md:207` | `vol_hidrograma_nodo` eliminado por "forma temporal idéntica" | Hoy hay 40 formas distintas: la justificación ya no vale (relacionado con C1) |
| D-31 | `README.md:147-158` | `reproduce_results` con Python 3.13 y `.venv` | venv local `venv/` con 3.12.10 (T-10) |
| D-32 | `README.md:41-65` | Arquitectura: `extraction/assembler.py` | Está en `swmm_resilience/dataset/assembler.py`; faltan `ml/bench/`, `database/`, `validation/` |
| D-33 | `AVANCES_RECIENTES_PROYECTO.md:217-305` | Métricas, PCA, `GroupShuffleSplit` por `run_id` como estado actual | Corresponden al pipeline legacy; el pipeline activo usa XGBoost sin PCA y folds por factor/forma |
| D-34 | `diagramas_pipeline.html` | "19 formas" | 40 formas (T-17) |

---

## 5. Scripts reproducibles

Todos se ejecutan con `venv\Scripts\python.exe -I` desde un directorio fuera del repo (scratchpad). Ninguno escribe en el repositorio.

### 5.1 Alcanzabilidad por AST — `reachability.py <repo_root>`

```python
import ast, sys
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve()
EXCLUDE = {"venv", ".git", "build", "__pycache__", ".pytest_cache", "swmm_resilience.egg-info",
           ".pytest-tmp-phase-a", "tmp20qwka3w", "data", "outputs", "docs", "obsidian",
           ".superpowers", ".claude"}
def iter_py():
    for p in ROOT.rglob("*.py"):
        if not any(part in EXCLUDE for part in p.relative_to(ROOT).parts):
            yield p
def modname(p):
    parts = list(p.relative_to(ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
files = {modname(p): p for p in iter_py()}
is_pkg = {m: files[m].name == "__init__.py" for m in files}
def resolve(cur, node):
    if isinstance(node, ast.Import):
        return {a.name for a in node.names}
    if node.level:
        base = cur.split(".") if is_pkg.get(cur) else cur.split(".")[:-1]
        if node.level > 1: base = base[: len(base) - (node.level - 1)]
        prefix = ".".join(base); mod = f"{prefix}.{node.module}" if node.module else prefix
    else:
        mod = node.module or ""
    return {mod} | {f"{mod}.{a.name}" for a in node.names}
edges = {}
for m, p in files.items():
    tgt = set()
    for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for cand in resolve(m, n):
                parts = cand.split(".")
                tgt |= {".".join(parts[:i]) for i in range(1, len(parts)+1) if ".".join(parts[:i]) in files}
    edges[m] = tgt
def closure(starts):
    seen, stack = set(), list(starts)
    while stack:
        m = stack.pop()
        if m in seen or m not in files: continue
        seen.add(m); stack.extend(edges.get(m, ()))
    return seen
pkg = sorted(m for m in files if m.startswith("swmm_resilience"))
prod = closure([m for m in ("main", "swmm_resilience.main", "swmm_resilience.desktop.app") if m in files])
tests = closure(list(prod) + [m for m in files if m.startswith("tests") or m == "conftest"])
allr = closure(list(tests) + [m for m in files if not m.startswith(("swmm_resilience", "tests"))])
print("modulos:", len(pkg), "prod:", len([m for m in pkg if m in prod]))
print("solo via tests:", [m for m in pkg if m not in prod and m in tests])
print("inalcanzables:", [m for m in pkg if m not in allr])
```

Resultado (2026-10-10): 99 módulos; 92 alcanzables desde producción; solo vía tests: `database.recovery`, `database.upgrade`, `database.workflow_lock`, `desktop` (artefacto: el `__init__` del paquete que contiene la entrada `desktop.app`), `ml.temporal.compare_surrogate`, `ml.temporal.train_cnn`, `ml.temporal.train_surrogate`; inalcanzables: ninguno. Scripts de raíz: `generate_obsidian_graph` (lo importa `tests/test_obsidian_graph.py`) y `halve_timeseries` (no lo importa nadie).

### 5.2 Mutaciones sobre una copia — `mutation_check.py <repo_root> <work_dir>`

```python
import shutil, subprocess, sys
from pathlib import Path
SRC, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
REPO, PY = WORK / "repo", str(Path(sys.argv[1]) / "venv" / "Scripts" / "python.exe")
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
for d in ("swmm_resilience", "tests"):
    shutil.copytree(SRC / d, REPO / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for f in ("main.py", "conftest.py", "pytest.ini", "config.yaml"):
    shutil.copy2(SRC / f, REPO / f)
MUTATIONS = [  # (id, archivo, original, mutado, tests)
    ("labels_ge_to_gt", "swmm_resilience/extraction/labels.py",
     "return (volumes >= threshold_m3).astype(int)", "return (volumes > threshold_m3).astype(int)",
     ["tests/test_labels.py", "tests/ml/bench/test_quality.py"]),
    # ... (lista completa en la tabla de abajo)
]
def run(tests):
    p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                        "--basetemp", str(WORK / "bt"), *tests], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, ([l for l in p.stdout.splitlines() if l.strip()] or ["-"])[-1]
print("BASELINE", run(sorted({t for m in MUTATIONS for t in m[4]})))
for mid, rel, old, new, tests in MUTATIONS:
    path = REPO / rel; orig = path.read_text(encoding="utf-8")
    assert orig.count(old) == 1, mid
    path.write_text(orig.replace(old, new), encoding="utf-8")
    try: rc, tail = run(tests)
    finally: path.write_text(orig, encoding="utf-8")
    print(mid, "DETECTADA" if rc else "SOBREVIVE", tail)
```

Mutaciones aplicadas (original → mutado) y resultado:

| Mutación | Archivo | Cambio | Tests | Resultado |
|---|---|---|---|---|
| labels_ge_to_gt | `extraction/labels.py` | `>=` → `>` | test_labels, test_quality | DETECTADA |
| quality_revert_gt | `ml/bench/quality.py` | `flood_label(...)` → `(y_reg > t)` | test_quality | DETECTADA |
| loso_shape_groups_by_factor | `ml/bench/schemas.py` | `LeaveOneShapeOut: shape_id` → `factor_mult` | test_folds/schemas/preprocess | DETECTADA |
| accum_without_ancestors | `extraction/dynamic_features.py` | `ancestors \| {nid}` → `{nid}` | test_scenario_predict | DETECTADA |
| extrapolation_never_flags | `ml/scenario_predict.py` | `ratio<fmin or ratio>fmax` → `False` | test_scenario_predict | DETECTADA |
| tpeak_from_last_series | `ml/scenario_predict.py` | primera serie → última serie | test_scenario_predict, test_predict_shape_features | **SOBREVIVE** |
| flow_units_check_off | `hb` | `if flow_units…` → `if False` | test_hydrograph_batch | DETECTADA |
| inp_hash_check_off | `hb` | `if stored and current != stored` → `if False` | idem | DETECTADA |
| continuity_warning_off | `hb` | `if continuity_pct…` → `if False` | idem | **SOBREVIVE** |
| truth_only_inflow_nodes_M4 | `hb` | `list(predictor.node_ids)` → `sorted(expected_nodes)` | idem | **SOBREVIVE** |
| drain_zero_point_off | `ts` | `if drain_down_hours > 0` → `if False` | test_timeseries_scenario | DETECTADA |
| drain_end_time_off | `ts` | `+ max(drain_down_hours,0)` → eliminado | idem | DETECTADA |
| bench_promotes_implicitly | `main.py` | `if args.bench_promote` → `… or run_all` | test_bench_cli | DETECTADA |
| promote_accepts_none | `main.py` | `if chosen in {"none","auto"}` → `if False` | test_bench_cli | DETECTADA |
| batch_reloads_predictor_per_scenario | `hb` | recrear el predictor dentro del bucle | test_hydrograph_batch | **SOBREVIVE** |
| batch_ignores_drain_down_param | `hb` | no pasar `drain_down_hours` | idem | **SOBREVIVE** |
| ponding_warning_off | `hb` | `if ponding…` → `if False` | idem | DETECTADA |
| unfinished_hydrograph_warning_off | `ts` | `if peak > 0 and last > 0.01*peak` → `if False` | test_timeseries_scenario | DETECTADA |
| extrap_base0_never_flags | `ml/scenario_predict.py` | `flags.append(peak > 0.0)` → `False` | test_scenario_predict | **SOBREVIVE** |
| regressor_on_all_nodes | `ml/scenario_predict.py` | regresor aplicado a todos los nodos | idem | **SOBREVIVE** |
| prepared_semantics_check_off | `ml/bench/schemas.py` | `if definitions != expected` → `if False` | test_schemas/preprocess/bench_cli | DETECTADA |
| held_out_groups_empty | `ml/bench/folds.py` | `held_out_groups` → `[]` | test_folds | DETECTADA |
| label_zero_threshold_counts_dry | `extraction/labels.py` | `volumes > 0` → `>= 0` | test_labels | DETECTADA |
| text_parser_scale_wrong | `extraction/labels.py` | sin `×1000` (parser de texto) | test_labels | DETECTADA |
| swmm_api_rpt_scale_wrong | `simulation/swmm_api_io.py` | `×1000` → `×1` | test_labels, test_flood_volume_extraction, test_hydrograph_batch | **SOBREVIVE** |
| error_pct_sign | `hb` | signo invertido de `error_pct` | test_hydrograph_batch | DETECTADA |

### 5.3 Comprobaciones puntuales (comandos de solo lectura)

```bash
# Ítems "seguro de borrar" de la auditoría de basura
git log --oneline --diff-filter=D -- pruebas_locales.py verificar.py   # 8195315
git ls-files halve_timeseries.py validation_output/inp validation_output_smoke/inp
git diff --no-index --stat validation_output/inp/hydrograph_2h.inp validation_output_smoke/inp/hydrograph_2h.inp  # vacío = idénticos
# Rama no integrada
git worktree list; git log --oneline 6e8579b..worktree-optuna-tuning-familia-ganadora | wc -l   # 69
# Trazabilidad de modelos (PowerShell)
Get-FileHash outputs\models\classifier.joblib -Algorithm SHA256   # 660BE918… vs promotion.json e2e7925e…
# Formas que no terminan en cero (>1 % del pico)
for f in data/hydrograph_shapes/*.csv; do tail -1 "$f"; done      # long_moderate 0.350, shape_2h 0.0747, shape_5h 0.0744
# Conteo de tests sin ejecutarlos
venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider tests          # 881/884
venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider tests/ml/bench # 258
```
