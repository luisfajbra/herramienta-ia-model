# Corrección 03 — Resolución del volumen de inundación y semántica de la etiqueta (D-04 / H-06)

- **Fecha:** 2026-10-11
- **Tipo:** *spike* acotado. Es una investigación cuyo producto es este documento con cifras y una recomendación. No se modificó código del proyecto, ni `config.yaml`, ni `data/`, ni `outputs/`. Ningún artefacto construido se conserva.
- **Rama:** `worktree-optuna-tuning-familia-ganadora` (base `6e8579b`, con cambios sin commitear de otra fase que no se tocaron). Los números de línea de `main.py` se refieren al árbol de trabajo.
- **Motor y librerías (venv del proyecto):** pyswmm 2.1.0, swmm-toolkit 0.17.0, swmm-api 0.4.73.
- **Red:** `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (`FLOW_UNITS LPS`, `DYNWAVE`, `ROUTING_STEP` 5 s, `REPORT_STEP` 3 min, `ALLOW_PONDING NO`).
- **Datos de evaluación:** PreparedDataset `ca706e27654e0cf5` (`outputs/bench/prepared_v4_21/`) y OOF de `outputs/bench/oof/ca706e27654e0cf5/` (154 080 filas por protocolo), leídos solo en lectura.
- **Simulaciones SWMM:** 9 en total (6 de control en `s01`, 1 con `REPORT_STEP` de 5 s en `s02` y 2 de verificación en `s06`). Todas sobre copias en el scratchpad, armadas con las mismas funciones del pipeline.
- **Regla de citas:** cada cifra lleva entre corchetes el script que la produjo (§9). Lo que no se midió se marca como **no verificado**.

---

## 1. Resumen

1. La etiqueta actual es `inunda = (V_rpt ≥ 1 m³)`, donde `V_rpt` es el *Total Flood Volume* del `.rpt`. Ese volumen se imprime en 10⁶ L con 3 decimales, es decir, con resolución de 1 m³.
2. Para volúmenes menores de 100 m³, `V_rpt` es **exactamente** el redondeo al entero del volumen continuo que SWMM acumula en cada paso de cálculo (852 de 852 nodos). La etiqueta actual coincide en 960 de 960 casos nodo-escenario con `V_continuo ≥ 0,5 m³` [s05]. No hay ruido de etiqueta: el umbral es nítido, pero está en 0,5 m³ continuos y no en 1 m³.
3. La fuente correcta del volumen continuo es `pyswmm` (`node.statistics['flooding_volume']`), que no añade costo. La integral de la serie `flooding` del `.out` **no sirve** con el `REPORT_STEP` actual de 3 min: el error llega a 9,3 m³ por nodo. Con un `REPORT_STEP` de 5 s el error baja a 0,89 m³, pero sigue siendo peor que el del propio `.rpt` [s01, s02].
4. En los 6 escenarios de control, pasar a `V_continuo ≥ 1 m³` cambia 21 etiquetas, todas de 1 a 0. Pasar a `V_continuo > 0` cambia 106, todas de 0 a 1 [s01].
5. Re-simular las 1000 corridas cuesta unos **17 min de SWMM** (unos 20 min con la extracción) [s03]. El costo real está en la cadena posterior (dataset, SQL, prep_id, banco, Optuna, tablas y validación) y en volver a trazar todas las cifras de la tesis.
6. En LeaveOneShapeOut, 333 de los 380 falsos negativos de XGBoost tienen `V_rpt = 1` (volumen continuo entre 0,5 y 1,5 m³). Si se excluyen esas 1894 filas, el F1 agregado pasa de 0,9817 a 0,9847 y el recall de 0,9918 a 0,9989 [s04].
7. **Recomendación: ruta A.** Se mantiene la resolución y el umbral actuales, se declaran como supuesto (umbral efectivo de 0,5 m³ continuos) y se reporta la matriz de confusión con la banda separada. La ruta B solo conviene si se re-simula por otra causa (H-01 InitFlow, H-05). En ese caso, capturar el volumen continuo tiene costo marginal casi nulo.

---

## 2. Cómo se obtiene hoy el volumen y la etiqueta

### 2.1 Cadena de resolución

| Paso | Dónde | Qué ocurre | Unidad y resolución |
|---|---|---|---|
| 1. Cálculo SWMM | motor SWMM 5.2 (swmm-toolkit 0.17.0) | El desborde de cada nodo se acumula en cada paso de routing (dt real de 1 a 5 s, anexo 01, V-11) | Volumen continuo en coma flotante |
| 2. Reporte | `.rpt`, sección *Node Flooding Summary*, columna *Total Flood Volume* | Se imprime con 3 decimales (formato observado: `1.286`) | 10⁶ L; resolución 0,001·10⁶ L = **1 m³** |
| 3. Lectura | `swmm_api_io.py:271-320` (`read_node_flooding_summary`) | `swmm_api.read_rpt_file(...).node_flooding_summary`. Toma la columna cuyo nombre contiene "volume" y "10^6" (l. 296) y multiplica por 1000 (l. 305-308) | m³, enteros salvo excepciones |
| 3b. Respaldo | `labels.py:26-90` (`_parse_node_flooding_text`) | Solo si swmm_api falla. Toma `parts[-1]` (l. 79), que en SWMM 5.2 es *Maximum Ponded Depth* (H-07) | Da **0** en todos los nodos (confirmado de nuevo aquí: Σ = 0 en los 4 escenarios con desborde [s01]) |
| 4. Etiquetado | `labels.py:122-139` (`extract_labels`) + `labels.py:19-23` (`flood_label`) | Los nodos ausentes del resumen reciben 0. `inunda = vol ≥ threshold` | Umbral `config.yaml:14` `dataset.flood_threshold_m3: 1.0` |
| 5. Generación | `main.py:1213` (forma `base`, vía `batch.run_batch` → `runner.run_simulation_simple`) y `main.py:1280` (39 formas, vía `batch.run_batch_shapes`) | Simula con `pyswmm.Simulation` (`for _ in sim: pass`) y lee el `.rpt` | Ídem |
| 6. Dataset | `dataset/assembler.py` → `data/training/dataset_final.csv` → `scripts/migrar_dataset_v4_21.py` → `dataset_final_v4_21.csv` | Columnas `vol_inundacion_m3` e `inunda` | Ídem |
| 7. Persistencia | `database/csv_backfill.py:207` (`--persist-sql`) → `outputs/training_v17.sqlite3`, tabla `node_results` | Se copian las dos columnas | Ídem |
| 8. Banco | PreparedDataset `ca706e27654e0cf5`, `targets.parquet` | `inunda` y `vol_inundacion_m3` | 154 080 filas |

**Fórmula vigente:**

`vol_inundacion_m3 = 1000 · TFV_rpt`, con `TFV_rpt` = *Total Flood Volume* [10⁶ L] con 3 decimales; `inunda = 1 ⇔ vol_inundacion_m3 ≥ 1,0`.

### 2.2 Lo que dicen los datos persistidos [s04]

| Conteo en el PreparedDataset (154 080 filas) | Valor |
|---|---|
| Filas positivas (`inunda = 1`) | 46 225 |
| `inunda == (vol ≥ 1)` en todas las filas | Sí |
| Filas con volumen en (0, 1) m³ | 0 |
| Filas con volumen exactamente 1,0 m³ | 1894 |
| Filas con volumen en [0,5, 2) m³ | 1894 (son las mismas) |
| Positivos no enteros | 6 (todos ≥ 2 m³) |

Las cifras de la auditoría (46 242 positivos, 1902 filas con 1,0) corresponden al dataset completo de 160 000 filas. El PreparedDataset excluye 37 corridas por continuidad.

---

## 3. Fuentes de volumen con mayor resolución: comparación en escenarios de control

### 3.1 Método

- **Escenarios:** se combinaron la forma `base`, dos formas cortas (`shape_ultra_short_impulse`, 0,5 h; `shape_micro_burst_45min`) y una larga (`long_moderate`, 12 h) con los factores 0,2, 1,0, 3,0 y 5,0.
- **Construcción de los `.inp`:** `write_scaled_inp` para `base`; `apply_shape` + `apply_closure_policy` (tolerancia 0,01 L/s) + `write_scenario_inp` (drenaje de 6 h) para las formas, como en `batch._prepare_closed_scenario_inp`.
- **Fuentes comparadas nodo a nodo:**
  - **rpt:** `extract_labels` (la ruta del pipeline).
  - **out:** integral trapezoidal de la variable `flooding` (L/s, porque `FLOW_UNITS = LPS`) del `.out`, leída con `swmm_api.SwmmOutput`; L/s·s/1000 = m³. También se calcularon rectángulos por izquierda y por derecha. En `s01` la integral empieza en el primer registro del `.out` (t = 3 min); en `s02` se antepone q(0) = 0.
  - **API:** `pyswmm` `node.statistics['flooding_volume']` en m³, leído al terminar el bucle de simulación.
- **Reproducción:** en los 4 escenarios que están en la BD, el volumen del `.rpt` replicado es **idéntico** al de `targets.parquet` en 160 de 160 nodos de cada escenario (640 de 640). Los totales también coinciden: `base` f = 3 da 7669 m³ y 61 nodos inundados, `shape_ultra_short_impulse` f = 5 da 1320 / 96, `shape_micro_burst_45min` f = 3 da 1704 / 60 y `long_moderate` f = 5 da 123 097 / 91 [s01, s04]. `base` f = 0,2 reproduce el error de continuidad de 5,168 % de `outputs/continuity_excluded.csv`, que es justamente la razón por la que esa corrida está excluida.

### 3.2 Volumen por fuente y diferencias por nodo [s01]

Diferencias absolutas en m³, calculadas sobre los nodos con desborde según alguna de las fuentes.

| Escenario | CE (%) | En BD | Nodos con desborde | Σ rpt | Σ API | Σ out (3 min) | *Flooding Loss* (.rpt) | máx \|rpt−API\| | mediana \|rpt−API\| | máx \|rpt−out\| | mediana \|rpt−out\| | máx \|out−API\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` f = 0,2 | 5,168 | no (excluida) | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `base` f = 3,0 | 0,459 | sí | 88 | 7669 | 7673,19 | 7672,18 | 7673 | 0,515 | 0,210 | 3,695 | 0,236 | 3,618 |
| `shape_ultra_short_impulse` f = 5,0 | 1,978 | sí | 128 | 1320 | 1327,26 | 1303,11 | 1328 | 0,494 | 0,270 | 9,305 | 0,788 | 9,535 |
| `shape_micro_burst_45min` f = 3,0 | 1,587 | sí | 101 | 1704 | 1713,09 | 1705,42 | 1713 | 0,500 | 0,183 | 5,492 | 0,293 | 5,339 |
| `long_moderate` f = 1,0 | 0,286 | sí | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `long_moderate` f = 5,0 | 0,055 | sí | 97 | 123 097 | 123 111,53 | 123 100,71 | 123 099 | 0,857 | 0,280 | 1,343 | 0,240 | 1,790 |

| Global (414 nodos-escenario con desborde) | máximo | mediana | p95 |
|---|---|---|---|
| \|rpt − API\| | 0,857 | 0,241 | 0,482 |
| \|rpt − out (3 min, trapecio)\| | 9,305 | 0,313 | 3,014 |
| \|out (3 min, trapecio) − API\| | 9,535 | 0,301 | 2,820 |
| \|out (3 min, rect. derecha) − API\| | 10,234 | 0,303 | 3,009 |
| \|out (3 min, rect. izquierda) − API\| | 9,535 | 0,303 | 3,009 |

**Lectura:**

- **rpt frente a API.** Para volúmenes menores de 100 m³, `V_rpt` es exactamente `round(V_API)` en 852 de 852 nodos. Solo 12 nodos, todos con más de 283 m³, difieren en más de 0,5 m³ (máximo 0,857 m³, relativo ≤ 0,18 %) [s05]. El origen de esas 12 diferencias en volúmenes grandes es **no verificado**. No afectan la etiqueta.
- **El `.out` no es una fuente de mayor resolución con la configuración actual.** SWMM escribe allí valores instantáneos cada `REPORT_STEP` = 180 s (60 a 380 registros por corrida; el primero a los 3 min [s01]). Integrar esos valores pierde los desbordes cortos: en `shape_ultra_short_impulse` el error llega a 9,5 m³ por nodo (344 % relativo en un nodo con V ≥ 1 m³). Usar `out ≥ 1` en vez del `.rpt` cambiaría 30 etiquetas en los 6 escenarios [s01].
- **Prueba con `REPORT_STEP` = 5 s** (`shape_ultra_short_impulse` f = 5, 1 corrida) [s02]:

| Fuente frente a API | máx (m³) | mediana (m³) |
|---|---|---|
| out con `REPORT_STEP` 5 s | 0,891 | 0,039 |
| out con `REPORT_STEP` 180 s | 9,535 | 0,575 |
| rpt | 0,494 | 0,270 |

  Con 5 s el `.out` pasa de 0,89 a 31,8 MB (4680 registros) y aun así cambia 1 etiqueta frente a la API. Cambiar `REPORT_STEP` no altera el cálculo: la API y el `.rpt` quedan idénticos a los de `s01` (diferencia ≤ 3,6·10⁻¹⁵ m³).
- **`.rpt` con más decimales:** SWMM no ofrece ninguna opción para cambiar el formato del *Node Flooding Summary*. **No verificado** en el código fuente C. En los `.rpt` generados la columna tiene siempre 3 decimales.
- **Conclusión de fuente:** si se quiere el volumen continuo, la fuente es `pyswmm` `node.statistics['flooding_volume']`. Ya se usa en el venv (anexo 01, V-04 y d11) y leerla al final del bucle cuesta ≤ 0,023 s por corrida [s03].

---

## 4. Cambios de etiqueta según la regla [s01, s05]

Las reglas son: (i) actual, `V_rpt ≥ 1`; (ii) `V_continuo ≥ 1`; (iii) `V_continuo > 0`. `V_continuo` = API.

| Escenario | Desbordan (V_cont > 0) | (i) positivos | (ii) positivos | (iii) positivos | Cambios (i)→(ii) | Cambios (i)→(iii) | V_cont en (0; 0,5) | V_cont en [0,5; 1) | Desbordan con etiqueta 0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| `base` f = 0,2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| `base` f = 3,0 | 88 | 61 | 60 | 88 | 1 | 27 | 27 | 1 | 30,7 |
| `shape_ultra_short_impulse` f = 5,0 | 128 | 96 | 83 | 128 | 13 | 32 | 32 | 13 | 25,0 |
| `shape_micro_burst_45min` f = 3,0 | 101 | 60 | 53 | 101 | 7 | 41 | 41 | 7 | 40,6 |
| `long_moderate` f = 1,0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| `long_moderate` f = 5,0 | 97 | 91 | 91 | 97 | 0 | 6 | 6 | 0 | 6,2 |
| **Total** | **414** | **308** | **287** | **414** | **21** | **106** | **106** | **21** | — |

**Lectura:**

- Todos los cambios (i)→(ii) son de 1 a 0 y coinciden exactamente con los nodos con `V_cont` en [0,5; 1). Todos los cambios (i)→(iii) son de 0 a 1 y coinciden con los nodos con `V_cont` en (0; 0,5). La regla actual equivale, en 960 de 960 casos, a `V_cont ≥ 0,5 m³` [s05].
- Los desbordes que hoy quedan con etiqueta 0 son pequeños: máximo 0,499 m³, mediana 0,153 m³, suma de 20,3 m³ en los 4 escenarios con desborde [s05]. En estos controles representan entre el 6 % y el 41 % de los nodos que desbordan. La auditoría informó entre 27 % y 57 % en otros escenarios (f = 1,6 y 2,0, más cerca del inicio de la inundación). Esa cifra viene del anexo 01 (`d11`) y no se volvió a verificar aquí.
- El volumen positivo mínimo que ve la API es 0,0027 m³. La regla (iii) es sensible a esas trazas: con `V_cont > 0,01` habría 412 positivos en vez de 414.
- **Proyección al dataset (estimación, no verificada):** en los controles, 21 de los 28 nodos con `V_rpt = 1` tienen `V_cont < 1` (rango de 0,514 a 1,455 m³), y 0 de los 7 nodos con `V_rpt = 2`. Si esa proporción (75 %, con n = 28) se mantuviera, la regla (ii) pasaría a 0 unas 1400 de las 1894 filas con `V_rpt = 1` (≈ 3 % de los positivos). La regla (iii) pasaría a 1 alrededor del 32 % de los nodos secos de los escenarios con desborde (106 de 332 en los controles). Ninguna de las dos proyecciones se midió sobre el dataset completo.

### 4.1 La no monotonía con el factor no es ruido de redondeo [s06]

El anexo 03 (D-04) reporta 72 casos de "deja de inundar al subir el factor", todos con 1–2 m³, y los atribuye a ruido de redondeo. Se re-simularon `common_bogota` f = 3,4 y 3,6 (BD = rpt en 160 de 160 nodos en ambas):

| Nodo | V_rpt f = 3,4 | V_cont f = 3,4 | V_rpt f = 3,6 | V_cont f = 3,6 |
|---|---|---|---|---|
| `107C` | 1 | 0,8623 | 0 | 0,0000 |
| `52C` | 1 | 0,8675 | 0 | 0,0197 |
| `66I` | 1 | 0,5002 | 0 | 0,4663 |

En 11 de los 117 nodos que desbordan a f = 3,4, el volumen **continuo** baja al pasar a f = 3,6. La no monotonía es hidráulica: el desborde se redistribuye entre nodos al aumentar la carga. El redondeo no puede crearla, porque redondear es una función monótona. Lo único que hace el umbral es dejarla a la vista. Con un umbral continuo de 1 m³, `107C` y `52C` dejarían de ser casos no monótonos, pero aparecerían otros alrededor de 1 m³. Esto corrige la interpretación de D-04 del anexo 03.

---

## 5. Costo de re-simular (ruta B)

### 5.1 Tiempos medidos [s01, s03]

| Escenario | Horas simuladas | Construcción del `.inp` (s) | SWMM (s) | Lectura de etiquetas (s) | Corrida completa* (s) | `.out` (MB) |
|---|---|---|---|---|---|---|
| `base` f = 0,2 | 3,00 | 0,118 | 0,371 | 0,044 | 0,533 | 0,415 |
| `base` f = 3,0 | 3,00 | 0,158 | 0,380 | 0,060 | 0,598 | 0,415 |
| `shape_ultra_short_impulse` f = 5,0 | 6,50 | 0,103 | 0,611 | 0,069 | 0,868 | 0,892 |
| `shape_micro_burst_45min` f = 3,0 | 6,75 | 0,131 | 0,595 | 0,049 | 0,859 | 0,926 |
| `long_moderate` f = 1,0 | 19,00 | 0,085 | 1,656 | 0,043 | 1,867 | 2,592 |
| `long_moderate` f = 5,0 | 19,00 | 0,058 | 1,398 | 0,051 | 1,591 | 2,592 |

\* Construcción + SWMM + etiquetas + lectura de drenaje del `.out` (0,084 s de media, solo en las formas CSV, como hace `run_batch_shapes`).

**Extrapolación** (mínimos cuadrados `t = a + b·H`, residuo máximo de 0,14 s; las 40 formas suman 478,2 h simuladas por factor, contando el cierre del aporte y el drenaje de 6 h):

| Concepto | Modelo | 1000 corridas |
|---|---|---|
| Solo SWMM | 0,142 s + 0,0726 s/h | 1010 s ≈ **16,8 min** |
| Corrida completa | 0,366 s + 0,0720 s/h | 1226 s ≈ **20,4 min** |

- No incluye los reintentos por drenaje de `run_batch_shapes` (hasta 2 extensiones de +2 h). En el peor caso el tiempo se triplicaría (≈ 1 h). Cuántas corridas reintentarían: **no verificado**.
- Son tiempos de pared en este equipo, en serie y con un solo proceso.

### 5.2 Qué habría que rehacer después de re-simular (no se ejecutó nada)

| Paso | Qué | Costo conocido |
|---|---|---|
| 1 | Cambiar el código para leer `node.statistics['flooding_volume']` en `runner.run_simulation_simple`, `batch.run_batch_shapes` **y** en la ruta de validación (`validation/hydrograph_batch.py`), para mantener la regla única de `labels.py`. Guardar el volumen continuo junto al del `.rpt`. Corregir H-07 de paso. Tests nuevos | Desarrollo y tests; no medido |
| 2 | Regenerar las 1000 corridas y `data/training/dataset_final.csv` (`python main.py`) | ≈ 17–20 min de SWMM (§5.1) más ensamblado; no medido |
| 3 | `scripts/migrar_dataset_v4_21.py` → `dataset_final_v4_21.csv` (features tributarias) | No medido |
| 4 | `python main.py --persist-sql` → `outputs/training_v17.sqlite3` (D-06: no es atómico ni idempotente, así que conviene usar una BD nueva) | No medido |
| 5 | Nuevo PreparedDataset (nuevo `prep_id`) y re-evaluación `scripts/ejecutar_evaluacion_v4_21.py` + `scripts/informe_evaluacion.py` | 330 s según `runs/provenance.json` (88 preparación + 194 XGBoost + 28 referencias + 20 tablas) |
| 6 | Volver a correr el tuning con Optuna de la familia ganadora (rama actual) y la promoción | No medido |
| 7 | Re-ejecutar la validación fuera de muestra (`--evaluate-validation-shapes`), que etiqueta con la misma regla | No medido |
| 8 | Actualizar todas las tablas y cifras de `docs/auditoria/correcciones/datos/` y de la tesis | Trabajo editorial; no medido |

El cuello de botella no es SWMM, sino los pasos 1 y 4–8 y el riesgo de desalinear la trazabilidad ya construida (D-01, D-06).

---

## 6. Efecto en la evaluación (ruta A, solo con los OOF guardados) [s04]

La matriz reconstruida desde los OOF coincide con `matriz_confusion.csv` en los 6 pares (modelo, protocolo).

### 6.1 Matriz de confusión con la banda separada — XGBoost, LeaveOneShapeOut

| Volumen real (`.rpt`) | Etiqueta real | n | Predicho 0 | Predicho 1 |
|---|---|---|---|---|
| 0 (V_cont < 0,5) | 0 | 107 855 | 106 526 (TN) | 1329 (FP) |
| [1, 2) = exactamente 1 (V_cont en [0,5; 1,5)) | 1 | 1894 | 333 (FN) | 1561 (TP) |
| [2, 5) | 1 | 2416 | 34 (FN) | 2382 (TP) |
| [5, 100) | 1 | 18 035 | 13 (FN) | 18 022 (TP) |
| ≥ 100 | 1 | 23 880 | 0 (FN) | 23 880 (TP) |

En los otros protocolos de XGBoost, los FN en la banda son 443 de 879 (LOSO) y 435 de 844 (GroupKFold5).

### 6.2 Métricas agregadas con y sin la banda ambigua [0,5; 2) m³

| Modelo | Protocolo | Filas | F1 | Recall | Precisión | FN | FP |
|---|---|---|---|---|---|---|---|
| xgboost | LeaveOneShapeOut (todas) | 154 080 | 0,9817 | 0,9918 | 0,9718 | 380 | 1329 |
| xgboost | LeaveOneShapeOut (sin banda) | 152 186 | 0,9847 | 0,9989 | 0,9709 | 47 | 1329 |
| xgboost | LOSO (todas) | 154 080 | 0,9667 | 0,9810 | 0,9528 | 879 | 2245 |
| xgboost | LOSO (sin banda) | 152 186 | 0,9704 | 0,9902 | 0,9513 | 436 | 2245 |
| xgboost | GroupKFold5 (todas) | 154 080 | 0,9681 | 0,9817 | 0,9549 | 844 | 2144 |
| xgboost | GroupKFold5 (sin banda) | 152 186 | 0,9718 | 0,9908 | 0,9535 | 409 | 2144 |
| baseline_forma_mas_cercana | LeaveOneShapeOut (todas) | 154 080 | 0,9764 | 0,9808 | 0,9721 | 889 | 1299 |
| baseline_forma_mas_cercana | LeaveOneShapeOut (sin banda) | 152 186 | 0,9824 | 0,9937 | 0,9714 | 278 | 1299 |
| baseline_interpolacion_factor | LOSO / GroupKFold5 (todas) | 154 080 | 0,9718 | 0,9910 | 0,9534 | 416 | 2238 |
| baseline_interpolacion_factor | LOSO / GroupKFold5 (sin banda) | 152 186 | 0,9751 | 0,9995 | 0,9519 | 23 | 2238 |

**Matices obligatorios al reportar la exclusión:**

- **La exclusión es asimétrica.** Solo puede quitar el lado positivo de la frontera (`V_rpt = 1`). El lado negativo (desbordes de 0 a 0,5 m³) queda mezclado con los nodos secos en `V_rpt = 0` y no se puede separar sin el volumen continuo. Por eso la exclusión infla el recall y deja la precisión casi igual (0,9718 → 0,9709). Debe presentarse como análisis de sensibilidad, no como la métrica principal.
- **Los FP también son de frontera.** De los 1329 FP de LeaveOneShapeOut, el regresor predice menos de 2 m³ en 578 y menos de 5 m³ en 1127. La probabilidad media del clasificador en los FN de la banda es 0,241.
- Los FN de la banda se concentran en formas fuera de rango o de transición: `new_hydrograph` 77, `rare_extreme` 27, `shape_ultra_short_impulse` 27, `common_bogota` 26 y `shape_flash_flood_urban` 20.

### 6.3 Lo que sugiere la ruta B para las métricas (no verificado)

Con etiquetas continuas `V_cont ≥ 1`, unas 1400 filas de la banda pasarían a negativas (proyección de §4). Eso movería la frontera, pero no la eliminaría: seguiría habiendo una banda de nodos con V_cont cerca de 1 m³ que el modelo confunde, ahora sin ninguna cuantización que la explique. Sin reentrenar no se puede saber si el F1 sube o baja. Una evaluación de los OOF actuales contra etiquetas reetiquetadas no tendría sentido, porque el modelo se entrenó con la otra frontera. La concentración de FN en la banda se explica por la cercanía al umbral, no por la resolución: según §4 la etiqueta actual es una función exacta del volumen continuo.

---

## 7. Recomendación

**Ruta A: mantener la resolución y el umbral, declararlos y reportar la matriz con la banda separada.**

**Razones:**

1. **La etiqueta actual no tiene ruido.** Es `V_cont ≥ 0,5 m³` exacta en 960 de 960 casos [s05]. El problema de D-04/H-06 es de **semántica** (el umbral nominal de 1 m³ es en realidad de 0,5 m³ continuos), y se resuelve declarándolo, sin tocar los datos.
2. **La ruta B no elimina el fenómeno que motivó el hallazgo.** Los FN en la banda de 1–2 m³ son el efecto natural de cualquier frontera. Con un umbral continuo de 1 m³ la frontera se desplaza unos 0,5 m³ y cambia la etiqueta de aproximadamente el 3 % de los positivos (proyección). La no monotonía con el factor también es hidráulica, no de redondeo (§4.1).
3. **Relación costo/beneficio.** SWMM es barato (unos 17 min), pero la ruta B obliga a rehacer el dataset, la BD, el `prep_id`, el banco, Optuna, la validación y todas las cifras ya trazadas de la tesis (§5.2). Todo eso para un cambio de definición que la ruta A expresa sin costo.
4. **Condición que invierte la recomendación.** Si el autor decide re-simular por H-01 (InitFlow fijo a factor 1, el hallazgo 🟠 con mayor efecto en volúmenes) o por H-05, entonces **sí** conviene incorporar la ruta B en esa misma re-simulación. Capturar `node.statistics['flooding_volume']` cuesta ≤ 0,023 s por corrida. Habría que guardar las dos columnas (`vol_rpt_m3` y `vol_continuo_m3`) y definir la etiqueta sobre el volumen continuo, con el umbral elegido de forma explícita.
5. **Fuente descartada.** En ningún caso debe usarse la integral del `.out` con `REPORT_STEP` = 3 min: su error (hasta 9,5 m³) es mayor que la cuantización que se quiere corregir [s01, s02].

**Costo de cada ruta:**

| Ruta | Simulaciones | Código | Re-evaluación | Cifras de la tesis |
|---|---|---|---|---|
| A | 0 | Ninguno obligatorio. Opcional: añadir la tabla de §6.1 a `scripts/informe_evaluacion.py` (ya existe `banda_limitrofe.csv`). Recomendable aparte: corregir H-07 (parser de respaldo), que no cambia datos | Ninguna: §6 sale de los OOF existentes | Solo agregar la declaración y la matriz por banda |
| B | 1000 (≈ 17–20 min, hasta ≈ 1 h con reintentos) | Captura del volumen continuo en 3 rutas, más tests | Dataset, BD, `prep_id`, banco (≈ 5,5 min), Optuna, validación | Todas las de evaluación |

---

## 8. Decisión pendiente del autor

1. **Elegir la ruta A o la B.** La recomendación es A, salvo que se re-simule por H-01/H-05; en ese caso, B dentro de esa misma re-simulación.
2. **Redacción del umbral (ruta A).** Mantener "≥ 1 m³ del `.rpt`" y declarar que equivale a "≥ 0,5 m³ de volumen continuo", o enunciar directamente el umbral como 0,5 m³ continuos. Los datos son los mismos en ambos casos.
3. **Métrica principal frente a sensibilidad.** Confirmar que la métrica principal sigue siendo la matriz completa (F1 0,9817 en LeaveOneShapeOut), que la matriz por banda (§6.1) se reporta siempre y que la exclusión de la banda (§6.2) va solo como sensibilidad, con la advertencia de asimetría.
4. **Ruta B: umbral continuo.** Si se elige B, decidir el umbral (0,5, 1 o 5 m³, o un barrido de sensibilidad como propone H-06) **antes** de mirar las métricas.
5. **Corrección de D-04 en el anexo 03.** Decidir si se corrige la atribución de los 72 casos no monótonos a "ruido de redondeo" (§4.1).
6. **H-07.** Corregir el parser de respaldo (lectura por encabezado y fallo explícito). Es independiente de la ruta elegida.

### 8.1 Texto para la tesis (ruta A)

> La variable objetivo de clasificación se define a partir del volumen desbordado por nodo que reporta SWMM 5.2 en la tabla *Node Flooding Summary* del archivo de reporte (`.rpt`). Esa tabla expresa el volumen en 10⁶ L con tres decimales, por lo que la resolución del volumen disponible es de 1 m³. Un nodo se etiqueta como inundado (`inunda = 1`) si su volumen reportado es de al menos 1 m³. Se comprobó, mediante re-simulaciones de control en las que se comparó el volumen reportado con el volumen que SWMM acumula en cada paso de cálculo, que el valor del reporte es el redondeo al entero de ese volumen continuo. En consecuencia, la regla equivale exactamente a un umbral de 0,5 m³ sobre el volumen continuo, y los desbordes menores de 0,5 m³ (cuya mediana fue 0,15 m³ en los escenarios de control) se consideran no inundación. La etiqueta es, por tanto, una función determinista del volumen simulado, sin ruido de redondeo. Sin embargo, como en todo problema con umbral, los casos con volumen cercano al umbral son los más difíciles de clasificar: en la validación por forma retenida (LeaveOneShapeOut), 333 de los 380 falsos negativos de XGBoost corresponden a nodos con volumen reportado de 1 m³ (volumen continuo entre 0,5 y 1,5 m³). Por ello, además de la matriz de confusión agregada, se informa la matriz separada por clase de volumen real y, como análisis de sensibilidad, las métricas sin esa banda (F1 de 0,9817 con todas las filas y de 0,9847 sin la banda). Esta sensibilidad excluye solo el lado positivo de la frontera, porque los desbordes de 0 a 0,5 m³ no se distinguen de los nodos secos en los datos disponibles.

---

## 9. Scripts usados y comandos exactos

Todos están en el scratchpad de la sesión (`<scratch>` = `C:/Users/LF409F~1.FAJ/AppData/Local/Temp/claude/c--Users-lf-fajardo-Downloads-Tesis-herramienta-ia-model/288ddaf2-bd7c-4ac1-bbe9-c4293e922cdc/scratchpad/spike_d04`). Las copias de `.inp`, `.rpt` y `.out` quedaron en `<scratch>/runs/`. Son de solo lectura sobre el repositorio y su código completo está en el Apéndice. Se ejecutan desde `<scratch>`, con `<repo>` = `C:/Users/lf.fajardo/Downloads/Tesis/herramienta-ia-model/.claude/worktrees/optuna-tuning-familia-ganadora` y `<py>` = `C:/Users/lf.fajardo/Downloads/Tesis/herramienta-ia-model/venv/Scripts/python.exe`.

| Script | Comando | Simulaciones | Qué produce |
|---|---|---|---|
| `s01_control.py` | `<py> s01_control.py <repo>` | 6 | rpt, out y API por nodo; reproducción frente a la BD; tiempos (`s01_nodos.csv`, `s01_resumen.csv`, `s01_tiempos.csv`) |
| `s02_reportstep.py` | `<py> s02_reportstep.py <repo>` | 1 | Convergencia del `.out` con `REPORT_STEP` de 5 s (`s02_nodos.csv`) |
| `s03_costo.py` | `<py> s03_costo.py <repo>` | 0 | Extrapolación del costo a 1000 corridas |
| `s04_oof_banda.py` | `<py> s04_oof_banda.py <repo>` | 0 | Matriz por banda y métricas con y sin banda desde los OOF (`s04_metricas.csv`, `s04_matriz_por_banda.csv`) |
| `s05_equivalencia.py` | `<py> s05_equivalencia.py` | 0 | Equivalencia `rpt ≥ 1` ⇔ `V_cont ≥ 0,5`; `rpt = round(V_cont)` |
| `s06_no_monotonia.py` | `<py> s06_no_monotonia.py <repo>` | 2 | Casos no monótonos de `common_bogota` f = 3,4 → 3,6 con volumen continuo |

Orden de ejecución: `s01` → `s02` → `s03` → `s04` → `s05` → `s06`. `s02`, `s03`, `s04` y `s05` leen salidas de `s01`. Total: 9 simulaciones SWMM, cada una de menos de 2 s.

---

## Apéndice — Código de los scripts

### `s01_control.py`

```python
"""Spike D-04/H-06: volumen de inundacion por nodo segun tres fuentes en escenarios de control.

Fuentes comparadas, nodo a nodo:
  rpt  : Node Flooding Summary del .rpt leido por el pipeline (extract_labels -> swmm_api), m3
  out  : integral temporal de la serie 'flooding' (L/s) de los nodos en el .out binario
         (swmm_api.SwmmOutput), con paso REPORT_STEP; trapecio, rectangulo izq. y der.
  api  : pyswmm node.statistics['flooding_volume'] (acumulado interno de SWMM en cada paso de routing)

Construye los .inp con las MISMAS funciones del pipeline (write_scaled_inp para 'base';
apply_shape + apply_closure_policy + write_scenario_inp(drain_down 6 h) para las formas CSV),
sobre COPIAS en ./runs del directorio de trabajo (scratchpad). Solo lectura sobre el repo.

Uso: python s01_control.py <repo>   (ejecutar desde spike_d04)
Salida: s01_nodos.csv (una fila por escenario y nodo), s01_resumen.csv, s01_tiempos.csv
"""
import sys, time, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from pyswmm import Simulation, Nodes
from swmm_api import SwmmOutput, read_inp_file
from swmm_resilience.simulation.swmm_api_io import write_scaled_inp, read_continuity_error
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels, _parse_node_flooding_text

INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
SHAPES = repo / "data/hydrograph_shapes"
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
OUT = Path("runs"); OUT.mkdir(exist_ok=True)

opts = read_inp_file(str(INP), encoding="utf-8")["OPTIONS"]
print("FLOW_UNITS:", opts["FLOW_UNITS"], " REPORT_STEP:", opts["REPORT_STEP"], " ROUTING_STEP:", opts["ROUTING_STEP"], " ALLOW_PONDING:", opts["ALLOW_PONDING"])
assert str(opts["FLOW_UNITS"]).upper() == "LPS"

static = extract_static_features(INP)
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
nodes = static.node_id.astype(str).tolist()

keys = pd.read_parquet(PREP / "keys.parquet"); tg = pd.read_parquet(PREP / "targets.parquet")
db = pd.concat([keys[["shape_id", "factor_mult", "node_id"]], tg], axis=1)
db["factor_mult"] = db.factor_mult.round(3)

CASES = [("base", 0.2), ("base", 3.0), ("shape_ultra_short_impulse", 5.0),
         ("shape_micro_burst_45min", 3.0), ("long_moderate", 1.0), ("long_moderate", 5.0)]


def build(sid, f):
    if sid == "base":
        p = OUT / f"base_f{f:.3f}.inp"
        write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
        return p
    shape = load_shape(SHAPES / f"{sid}.csv")
    sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t, _ in shape], shape[-1][0])
    sc, _ = apply_closure_policy(sc, 0.01)  # config.validation.inflow_closure_tolerance_lps
    return write_scenario_inp(INP, sc, OUT, drain_down_hours=6.0)


def out_integrals(out_path):
    with SwmmOutput(str(out_path)) as o:
        fl = o.get_part(kind="node", variable="flooding")  # L/s (FLOW_UNITS LPS)
        idx = o.index
    t = np.array([(x - idx[0]).total_seconds() for x in idx])
    step = float(np.median(np.diff(t)))
    # SWMM escribe el primer registro en START + REPORT_STEP? se reporta t0 para documentarlo
    q = fl[[c for c in fl.columns if c in nodes]].to_numpy(dtype=float)
    # se antepone q(t0 - step)=0 solo si el primer registro no es t=0 de la simulacion: se documenta abajo
    trap = np.trapz(q, t, axis=0) / 1000.0
    rect_r = (q[1:] * np.diff(t)[:, None]).sum(0) / 1000.0
    rect_l = (q[:-1] * np.diff(t)[:, None]).sum(0) / 1000.0
    cols = [c for c in fl.columns if c in nodes]
    return (pd.DataFrame({"vol_out_trap": trap, "vol_out_rect_der": rect_r, "vol_out_rect_izq": rect_l}, index=cols),
            dict(n_registros=len(t), report_step_s=step, primer_registro=str(idx[0]), ultimo_registro=str(idx[-1])))


def flooding_loss(rpt):
    t = Path(rpt).read_text(errors="replace")
    m = re.search(r"Flooding Loss \.+\s+(-?[\d.]+)\s+(-?[\d.]+)", t)
    return float(m.group(2)) * 1000 if m else float("nan")


allrows, summ, times = [], [], []
for sid, f in CASES:
    t0 = time.perf_counter(); p = build(sid, f); t_build = time.perf_counter() - t0
    api = {}
    t1 = time.perf_counter()
    with Simulation(str(p)) as sim:
        for _ in sim:
            pass
        t_loop = time.perf_counter() - t1
        for n in Nodes(sim):
            if n.is_junction():
                api[n.nodeid] = n.statistics["flooding_volume"]
    t_sim = time.perf_counter() - t1
    rpt = p.with_suffix(".rpt"); outp = p.with_suffix(".out")
    t2 = time.perf_counter(); lab = extract_labels(rpt, nodes, 1.0).set_index("node_id"); t_lab = time.perf_counter() - t2
    txt = _parse_node_flooding_text(rpt).set_index("node_id")["flooding_volume_m3"]
    oi, meta = out_integrals(outp)
    d = lab.rename(columns={"vol_inundacion_m3": "vol_rpt", "inunda": "inunda_actual"})
    d["vol_api"] = pd.Series(api)
    d = d.join(oi)
    d["vol_parser_texto"] = txt.reindex(d.index).fillna(0.0)
    d = d.fillna({"vol_out_trap": 0.0, "vol_out_rect_der": 0.0, "vol_out_rect_izq": 0.0})
    dbs = db[(db.shape_id == sid) & (db.factor_mult == round(f, 3))].set_index("node_id")
    d["vol_bd"] = dbs.vol_inundacion_m3.reindex(d.index)
    d.insert(0, "shape_id", sid); d.insert(1, "factor", f)
    allrows.append(d.reset_index().rename(columns={"index": "node_id"}))
    ce = read_continuity_error(rpt)
    times.append(dict(shape_id=sid, factor=f, t_build_s=t_build, t_sim_loop_s=t_loop, t_sim_total_s=t_sim,
                      t_labels_s=t_lab, inp_kb=p.stat().st_size / 1024, out_mb=outp.stat().st_size / 2**20,
                      rpt_kb=rpt.stat().st_size / 1024, **meta))
    print(f"{sid} f={f}: CE={ce}  sim={t_sim:.2f}s  build={t_build:.2f}s  out={meta}", flush=True)
    cont = d.vol_api
    lab_i = (d.vol_rpt >= 1.0); lab_ii = (cont >= 1.0); lab_iii = (cont > 0)
    any_fl = (d.vol_rpt > 0) | (cont > 0) | (d.vol_out_trap > 0)
    e_ro = (d.vol_rpt - d.vol_out_trap)[any_fl].abs(); e_ra = (d.vol_rpt - d.vol_api)[any_fl].abs()
    e_oa = (d.vol_out_trap - d.vol_api)[any_fl].abs()
    summ.append(dict(
        shape_id=sid, factor=f, continuity_error_pct=ce, en_bd=bool(len(dbs)),
        bd_igual_rpt=(int((d.vol_bd == d.vol_rpt).sum()) if len(dbs) else None),
        n_nodos_con_desborde=int(any_fl.sum()),
        sum_rpt=d.vol_rpt.sum(), sum_api=cont.sum(), sum_out_trap=d.vol_out_trap.sum(), flooding_loss_rpt=flooding_loss(rpt),
        sum_parser_texto=d.vol_parser_texto.sum(),
        max_abs_rpt_out=e_ro.max() if len(e_ro) else 0, med_abs_rpt_out=e_ro.median() if len(e_ro) else 0,
        max_abs_rpt_api=e_ra.max() if len(e_ra) else 0, med_abs_rpt_api=e_ra.median() if len(e_ra) else 0,
        max_abs_out_api=e_oa.max() if len(e_oa) else 0, med_abs_out_api=e_oa.median() if len(e_oa) else 0,
        max_rel_out_api_v_ge1=((d.vol_out_trap - cont).abs() / cont)[cont >= 1].max() if (cont >= 1).any() else 0,
        n_api_0_05=int(((cont > 0) & (cont < 0.5)).sum()), n_api_05_1=int(((cont >= 0.5) & (cont < 1)).sum()),
        n_api_1_2=int(((cont >= 1) & (cont < 2)).sum()),
        n_lab_i=int(lab_i.sum()), n_lab_ii=int(lab_ii.sum()), n_lab_iii=int(lab_iii.sum()),
        n_lab_iii_gt_0p01=int((cont > 0.01).sum()),
        cambia_i_vs_ii=int((lab_i != lab_ii).sum()), i1_ii0=int((lab_i & ~lab_ii).sum()), i0_ii1=int((~lab_i & lab_ii).sum()),
        cambia_i_vs_iii=int((lab_i != lab_iii).sum()),
        cambia_i_vs_out_ge1=int((lab_i != (d.vol_out_trap >= 1.0)).sum()),
        min_api_positivo=cont[cont > 0].min() if (cont > 0).any() else None,
    ))

nd = pd.concat(allrows, ignore_index=True); nd.to_csv("s01_nodos.csv", index=False)
sm = pd.DataFrame(summ); sm.to_csv("s01_resumen.csv", index=False)
tm = pd.DataFrame(times); tm.to_csv("s01_tiempos.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(sm.T); print(tm.T)
fl = nd[(nd.vol_rpt > 0) | (nd.vol_api > 0) | (nd.vol_out_trap > 0)]
print("GLOBAL nodos con desborde:", len(fl))
for a, b in [("vol_rpt", "vol_api"), ("vol_rpt", "vol_out_trap"), ("vol_out_trap", "vol_api"), ("vol_out_rect_der", "vol_api"), ("vol_out_rect_izq", "vol_api")]:
    e = (fl[a] - fl[b]).abs(); print(f"  |{a}-{b}|: max={e.max():.4f} mediana={e.median():.4f} p95={e.quantile(.95):.4f}")
print("  |rpt-api| > 0.5:", int(((fl.vol_rpt - fl.vol_api).abs() > 0.5).sum()))
print("  rpt - round(api) distinto de 0 (nodos):", int((fl.vol_rpt != fl.vol_api.round(0)).sum()))
```

### `s02_reportstep.py`

```python
"""Spike D-04: la integral del .out converge al volumen interno de SWMM si REPORT_STEP baja a 5 s?
Toma la COPIA runs/shape_ultra_short_impulse_f5.000.inp creada por s01, cambia solo REPORT_STEP a 0:00:05
(copia nueva runs/rs5_ultra_f5.inp), simula una vez y compara rpt / out(5 s) / api nodo a nodo.
Uso: python s02_reportstep.py <repo>   (despues de s01, desde spike_d04)"""
import sys, time, warnings, datetime
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from pyswmm import Simulation, Nodes
from swmm_api import SwmmOutput, read_inp_file
from swmm_resilience.extraction.labels import extract_labels

src = Path("runs/shape_ultra_short_impulse_f5.000.inp"); dst = Path("runs/rs5_ultra_f5.inp")
inp = read_inp_file(str(src), encoding="utf-8")
inp["OPTIONS"]["REPORT_STEP"] = datetime.time(0, 0, 5)
inp.write_file(str(dst))
nodes = [str(n) for n in inp["JUNCTIONS"].keys()]
api = {}
t1 = time.perf_counter()
with Simulation(str(dst)) as sim:
    for _ in sim:
        pass
    for n in Nodes(sim):
        if n.is_junction():
            api[n.nodeid] = n.statistics["flooding_volume"]
t_sim = time.perf_counter() - t1
with SwmmOutput(str(dst.with_suffix(".out"))) as o:
    fl = o.get_part(kind="node", variable="flooding"); idx = o.index
t = np.array([(x - idx[0]).total_seconds() for x in idx]) + (idx[0] - datetime.datetime.combine(idx[0].date(), datetime.time(0))).total_seconds()
q = fl[nodes].to_numpy(float)
# se antepone t=0 con q=0 (SWMM no escribe el instante inicial)
t = np.r_[0.0, t]; q = np.vstack([np.zeros(q.shape[1]), q])
vout = pd.Series(np.trapz(q, t, axis=0) / 1000.0, index=nodes)
lab = extract_labels(dst.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
d = pd.DataFrame({"vol_rpt": lab.vol_inundacion_m3, "vol_api": pd.Series(api), "vol_out5s": vout})
prev = pd.read_csv("s01_nodos.csv"); prev = prev[prev.shape_id == "shape_ultra_short_impulse"].set_index("node_id")
d["vol_out180s"] = prev.vol_out_trap
d["vol_api_s01"] = prev.vol_api
m = (d[["vol_rpt", "vol_api", "vol_out5s"]] > 0).any(axis=1)
print(f"REPORT_STEP=5 s: t_sim={t_sim:.2f}s  registros={len(idx)}  out_MB={dst.with_suffix('.out').stat().st_size/2**20:.1f}")
print("api identico a s01 (REPORT_STEP no altera el calculo):", float((d.vol_api - d.vol_api_s01).abs().max()))
print("rpt identico a s01:", float((d.vol_rpt - prev.vol_rpt).abs().max()))
for a in ["vol_out5s", "vol_out180s", "vol_rpt"]:
    e = (d[a] - d.vol_api)[m].abs(); print(f"|{a}-api|: max={e.max():.4f} mediana={e.median():.4f}")
print("cambios de etiqueta (>=1) out5s vs api:", int(((d.vol_out5s >= 1) != (d.vol_api >= 1)).sum()))
d.to_csv("s02_nodos.csv")
```

### `s03_costo.py`

```python
"""Spike D-04: extrapola el costo de re-simular las 1000 corridas (40 formas x 25 factores) a partir
de los tiempos medidos en s01 (s01_tiempos.csv). No simula nada.
Modelo: t_corrida = a + b * horas_simuladas (minimos cuadrados sobre las 6 corridas de control).
Ademas mide (sin simular) el costo de la lectura de drenaje del .out que hace run_batch_shapes.
Uso: python s03_costo.py <repo>   (desde spike_d04, despues de s01)"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
from swmm_resilience.validation.drainage_state import read_tail_window_observables

tm = pd.read_csv("s01_tiempos.csv")
H = {"base": 3.0}
shapes = load_all_shapes(repo / "data/hydrograph_shapes")
from swmm_resilience.simulation.inflow_closure import close_series_to_zero
for sid, sh in shapes.items():
    # la politica de cierre (inflow_closure) agrega un paso de malla si la forma no termina en 0
    closed, _ = close_series_to_zero(sh, 1e-9)
    H[sid] = closed[-1][0] + 6.0
tm["horas_sim"] = tm.shape_id.map(H)
# costo de lectura de drenaje del .out (lo hace run_batch_shapes en cada corrida de forma CSV)
t_dr = []
for _, r in tm.iterrows():
    if r.shape_id == "base":
        continue
    p = Path("runs") / f"{r.shape_id}_f{r.factor:.3f}.out"
    t0 = time.perf_counter(); read_tail_window_observables(p, 6.0, 1.0); t_dr.append(time.perf_counter() - t0)
t_drain = float(np.mean(t_dr))
tm["t_corrida_s"] = tm.t_build_s + tm.t_sim_total_s + tm.t_labels_s + np.where(tm.shape_id == "base", 0.0, t_drain)
X = np.c_[np.ones(len(tm)), tm.horas_sim]
for col in ("t_sim_total_s", "t_corrida_s"):
    a, b = np.linalg.lstsq(X, tm[col], rcond=None)[0]
    pred = a + b * np.array([H[s] for s in H])
    tot = 25 * pred.sum()
    print(f"{col}: a={a:.3f}s b={b:.4f}s/h  residuo max={np.abs(X@[a,b]-tm[col]).max():.3f}s  -> 1000 corridas: {tot:.0f} s = {tot/60:.1f} min")
print("horas simuladas por forma: min %.2f med %.2f max %.2f; suma 40 formas = %.1f h" % (min(H.values()), np.median(list(H.values())), max(H.values()), sum(H.values())))
print(f"lectura de drenaje del .out por corrida: {t_drain:.3f}s (media de {len(t_dr)})")
print("tiempo medio de simulacion de las 6 corridas: %.2f s (min %.2f, max %.2f)" % (tm.t_sim_total_s.mean(), tm.t_sim_total_s.min(), tm.t_sim_total_s.max()))
print("overhead de leer node.statistics tras el bucle: max %.3f s" % (tm.t_sim_total_s - tm.t_sim_loop_s).max())
print(tm[["shape_id", "factor", "horas_sim", "t_build_s", "t_sim_total_s", "t_labels_s", "t_corrida_s", "out_mb"]].round(3).to_string(index=False))
```

### `s04_oof_banda.py`

```python
"""Spike D-04, ruta A: efecto de la banda ambigua en la evaluacion, usando SOLO los OOF persistidos
(sin reentrenar). Solo lectura sobre outputs/.
- Verifica que la matriz de confusion reconstruida coincide con docs/.../matriz_confusion.csv.
- Matriz de confusion separada por clase de volumen real (.rpt): 0 | [1,2) | [2,5) | [5,100) | >=100.
- Metricas agregadas excluyendo las filas con volumen real en [0.5, 2) m3 (banda ambigua).
- Ruta B (proyeccion): con s01_nodos.csv, fraccion de nodos con vol .rpt = 1 cuyo volumen continuo es < 1.
Uso: python s04_oof_banda.py <repo>   (desde spike_d04)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]).resolve()
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
OOF = repo / "outputs/bench/oof/ca706e27654e0cf5"
CM = pd.read_csv(repo / "docs/auditoria/correcciones/datos/matriz_confusion.csv")
keys = pd.read_parquet(PREP / "keys.parquet"); tg = pd.read_parquet(PREP / "targets.parquet")
base = pd.concat([keys[["shape_id", "factor_mult", "node_id"]], tg], axis=1)
base["sample_idx"] = np.arange(len(base))
v = base.vol_inundacion_m3
print("filas:", len(base), " positivas:", int(tg.inunda.sum()), " vol==1.0:", int((v == 1.0).sum()),
      " vol en [0.5,2):", int(((v >= 0.5) & (v < 2)).sum()), " vol en (0,1):", int(((v > 0) & (v < 1)).sum()),
      " no enteros >0:", int(((v > 0) & (v != v.round())).sum()))
print("inunda == (vol>=1):", bool((tg.inunda == (v >= 1.0)).all()))
# comprobacion de reproduccion de las 4 corridas de control presentes en la BD
for sid, f in [("base", 3.0), ("shape_ultra_short_impulse", 5.0), ("shape_micro_burst_45min", 3.0), ("long_moderate", 5.0)]:
    s = base[(base.shape_id == sid) & (base.factor_mult.round(3) == f)]
    print(f"  BD {sid} f={f}: suma vol={s.vol_inundacion_m3.sum():.0f} m3, n_inunda={int(s.inunda.sum())}")

bins = [-np.inf, 0.5, 2.0, 5.0, 100.0, np.inf]
labs = ["0", "[1,2)", "[2,5)", "[5,100)", ">=100"]
base["clase_vol"] = pd.cut(v, bins, right=False, labels=labs)


def metr(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, f1=f1, recall=rc, precision=pr)


rows, bandrows = [], []
for model in ["xgboost", "baseline_interpolacion_factor", "baseline_forma_mas_cercana"]:
    o = pd.read_parquet(OOF / f"{model}.parquet")
    for prot, g in o.groupby("protocol"):
        d = g.merge(base, on="sample_idx", how="left")
        assert len(d) == len(base) and d.sample_idx.is_unique
        y = d.inunda.to_numpy(); p = d.y_pred_clf.to_numpy()
        full = metr(y, p)
        ref = CM[(CM.modelo == model) & (CM.protocolo == prot)]
        ok = (len(ref) == 1) and all(int(ref[k].iloc[0]) == full[k] for k in ("tn", "fp", "fn", "tp"))
        amb = ((d.vol_inundacion_m3 >= 0.5) & (d.vol_inundacion_m3 < 2.0)).to_numpy()
        excl = metr(y[~amb], p[~amb])
        rows.append(dict(modelo=model, protocolo=prot, variante="todas", coincide_csv=ok, n=len(y), **full))
        rows.append(dict(modelo=model, protocolo=prot, variante="sin_banda_[0.5,2)", coincide_csv=None, n=int((~amb).sum()), **excl))
        for c in labs:
            m = (d.clase_vol == c).to_numpy()
            bandrows.append(dict(modelo=model, protocolo=prot, clase_vol_real=c, n=int(m.sum()),
                                 pred_0=int((p[m] == 0).sum()), pred_1=int((p[m] == 1).sum())))
        if model == "xgboost":
            fpm = (y == 0) & (p == 1); fnm = (y == 1) & (p == 0)
            pr = d.y_pred_reg.to_numpy()
            print(f"{model} {prot}: FP={int(fpm.sum())}, de ellos con vol predicho <2 m3: {int((fpm & (pr < 2)).sum())}, <5 m3: {int((fpm & (pr < 5)).sum())}; "
                  f"FN={int(fnm.sum())}, en [1,2): {int((fnm & amb).sum())}; prob media de FN en banda={d.y_prob_clf[fnm & amb].mean():.3f}")
            # por forma: cuantos FN de banda por forma (top 5) en LeaveOneShapeOut
            if prot == "LeaveOneShapeOut":
                print("   FN en banda por forma (top 5):", d[fnm & amb].shape_id.value_counts().head(5).to_dict())
                print("   FP por forma (top 5):", d[fpm].shape_id.value_counts().head(5).to_dict())

res = pd.DataFrame(rows); bm = pd.DataFrame(bandrows)
res.to_csv("s04_metricas.csv", index=False); bm.to_csv("s04_matriz_por_banda.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(4).to_string(index=False)); print(bm.to_string(index=False))

# --- Ruta B: proyeccion con los escenarios de control (s01) ---
nd = pd.read_csv("s01_nodos.csv")
r1 = nd[nd.vol_rpt == 1.0]; r0 = nd[nd.vol_rpt == 0.0]; r2 = nd[nd.vol_rpt == 2.0]
print(f"Control: nodos con rpt=1: {len(r1)}, de ellos continuo<1: {int((r1.vol_api < 1).sum())} "
      f"(rango continuo {r1.vol_api.min():.3f}-{r1.vol_api.max():.3f}); rpt=2: {len(r2)}, continuo<1: {int((r2.vol_api < 1).sum())}; "
      f"rpt=0: {len(r0)}, continuo>0: {int((r0.vol_api > 0).sum())}, continuo en [0.5,1): {int(((r0.vol_api >= .5) & (r0.vol_api < 1)).sum())}")
print("Control: nodos con rpt=0 y continuo>0, por escenario:", r0[r0.vol_api > 0].groupby(["shape_id", "factor"]).size().to_dict())
print("Control: nodos con rpt=0 (en escenarios con algun desborde):", r0[r0.groupby(["shape_id", "factor"]).vol_api.transform("max") > 0].groupby(["shape_id", "factor"]).size().to_dict())
```

### `s05_equivalencia.py`

```python
"""Spike D-04: la etiqueta actual (rpt >= 1) es una funcion determinista del volumen continuo?
Comprueba sobre s01_nodos.csv (6 escenarios x 160 nodos) que (rpt>=1) == (continuo>=0.5) y que rpt == round(continuo).
Uso: python s05_equivalencia.py   (desde spike_d04, despues de s01)"""
import pandas as pd, numpy as np
d = pd.read_csv("s01_nodos.csv")
print("filas nodo-escenario:", len(d))
print("discrepancias (rpt>=1) vs (api>=0.5):", int(((d.vol_rpt >= 1) != (d.vol_api >= 0.5)).sum()))
for lim in (10, 100, 1e9):
    m = d.vol_api < lim
    print(f"api<{lim:g}: n={int(m.sum())}, rpt != round(api): {int((d.vol_rpt[m] != np.round(d.vol_api[m])).sum())}")
big = d[d.vol_api >= 100]
print("api>=100: |rpt-api|/api max = %.2e" % ((big.vol_rpt - big.vol_api).abs() / big.vol_api).max())
g = d.groupby(["shape_id", "factor"])
t = pd.DataFrame({"desbordan_api_gt0": g.apply(lambda x: int((x.vol_api > 0).sum())),
                  "etiqueta_actual_1": g.apply(lambda x: int((x.vol_rpt >= 1).sum())),
                  "desbordan_con_etiqueta_0": g.apply(lambda x: int(((x.vol_api > 0) & (x.vol_rpt < 1)).sum()))})
t["pct_ocultos"] = (100 * t.desbordan_con_etiqueta_0 / t.desbordan_api_gt0).round(1)
print(t.to_string())
oc = d[(d.vol_api > 0) & (d.vol_rpt < 1)]
print("volumen continuo de los ocultos: max %.3f, suma %.1f m3, mediana %.3f" % (oc.vol_api.max(), oc.vol_api.sum(), oc.vol_api.median()))
```

### `s06_no_monotonia.py`

```python
"""Spike D-04: los 72 casos 'deja de inundar al subir el factor' (D-04, anexo 03) son ruido de redondeo
o el volumen continuo tambien baja? Simula common_bogota f=3.4 y f=3.6 (2 corridas, copias en ./runs)
y compara nodo a nodo rpt vs volumen continuo (pyswmm node.statistics['flooding_volume']).
Uso: python s06_no_monotonia.py <repo>   (desde spike_d04)"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import pandas as pd
from pyswmm import Simulation, Nodes
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
static = extract_static_features(INP); nodes = static.node_id.astype(str).tolist()
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
db = pd.concat([pd.read_parquet(PREP / "keys.parquet")[["shape_id", "factor_mult", "node_id"]], pd.read_parquet(PREP / "targets.parquet")], axis=1)
sid = "common_bogota"; res = {}
for f in (3.4, 3.6):
    shape = load_shape(repo / "data/hydrograph_shapes" / f"{sid}.csv")
    sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t, _ in shape], shape[-1][0])
    sc, _ = apply_closure_policy(sc, 0.01)
    p = write_scenario_inp(INP, sc, Path("runs"), drain_down_hours=6.0)
    api = {}
    with Simulation(str(p)) as sim:
        for _ in sim:
            pass
        for n in Nodes(sim):
            if n.is_junction():
                api[n.nodeid] = n.statistics["flooding_volume"]
    lab = extract_labels(p.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    bd = db[(db.shape_id == sid) & (db.factor_mult.round(3) == f)].set_index("node_id").vol_inundacion_m3
    res[f] = pd.DataFrame({"rpt": lab.vol_inundacion_m3, "api": pd.Series(api), "bd": bd})
    print(f"f={f}: BD==rpt en {int((res[f].bd == res[f].rpt).sum())}/160 nodos; suma rpt={res[f].rpt.sum():.0f}")
a, b = res[3.4], res[3.6]
cand = a[(a.rpt >= 1) & (b.rpt < 1)].index
print("nodos con etiqueta 1 a f=3.4 y 0 a f=3.6:", list(cand))
print(pd.DataFrame({"rpt_3.4": a.rpt[cand], "cont_3.4": a.api[cand].round(4), "rpt_3.6": b.rpt[cand], "cont_3.6": b.api[cand].round(4)}).to_string())
dec = (b.api < a.api) & (a.api > 0)
print("nodos cuyo volumen continuo BAJA de f=3.4 a f=3.6:", int(dec.sum()), "de", int((a.api > 0).sum()), "con desborde a f=3.4")
```
