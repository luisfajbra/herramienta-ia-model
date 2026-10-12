# Corrección 03 — Resolución del volumen de inundación y semántica de la etiqueta (D-04 / H-06)

- **Fecha:** 2026-10-11
- **Tipo:** *spike* acotado. Es una investigación cuyo producto es este documento con cifras y una recomendación. No se modificó código del proyecto, ni `config.yaml`, ni `data/`, ni `outputs/`. En el repositorio solo entran este documento y las notas de corrección fechadas de §8.1. Los scripts y las copias de `.inp`, `.rpt` y `.out` quedan solo en el scratchpad de la sesión (§9), que es temporal. Por eso el código completo de los scripts va en el Apéndice.
- **Rama:** `worktree-optuna-tuning-familia-ganadora` (base `6e8579b`, con cambios sin commitear de otra fase que no se tocaron). Los números de línea de `main.py` se refieren al árbol de trabajo.
- **Motor y librerías (venv del proyecto):** pyswmm 2.1.0, swmm-toolkit 0.17.0, swmm-api 0.4.73.
- **Red:** `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (`FLOW_UNITS LPS`, `DYNWAVE`, `ROUTING_STEP` 5 s, `REPORT_STEP` 3 min, `ALLOW_PONDING NO`).
- **Datos de evaluación:** PreparedDataset `ca706e27654e0cf5` (`outputs/bench/prepared_v4_21/`) y OOF de `outputs/bench/oof/ca706e27654e0cf5/` (154 080 filas por protocolo), leídos solo en lectura.
- **Simulaciones SWMM:** 18 en total, todas sobre copias en el scratchpad y armadas con las mismas funciones del pipeline:
  - 6 de control (`s01`);
  - 1 con `REPORT_STEP` de 5 s (`s02`);
  - 2 de verificación (`s06`);
  - 9 añadidas en la ronda de correcciones (`s07`).
- **Alcance de toda afirmación cuantitativa:** una sola red, con 17 escenarios distintos simulados (forma, factor), de los que 15 tienen desborde: 6 de `s01`, 2 de `s06` (`common_bogota` f = 3,4 y 3,6) y 9 de `s07`; la corrida de `s02` repite un escenario de `s01` con otro `REPORT_STEP`. Las comparaciones nodo a nodo y la cota del umbral efectivo usan los 15 escenarios con registro por nodo (`s01` + `s07`, 13 con desborde), porque `s06` no guardó el volumen de cada nodo. Cualquier generalización más allá de eso es inferencia y se señala como tal.
- **Regla de citas:** cada cifra lleva entre corchetes el script que la produjo (§9). Lo que no se midió se marca como **no verificado**.

---

## 1. Resumen

1. La etiqueta actual es `inunda = (V_rpt ≥ 1 m³)`, donde `V_rpt` es el *Total Flood Volume* del `.rpt`. Ese volumen se imprime en 10⁶ L con 3 decimales, es decir, con resolución de 1 m³.
2. En los 15 escenarios con registro por nodo (`s01` + `s07`, 13 con desborde; de 17 escenarios simulados en total), `V_rpt` coincide con el redondeo al entero del volumen continuo que SWMM acumula en cada paso de cálculo en 1407 de los 1408 casos con 0 < V < 100 m³. Hay 13 casos (8C con 83,5 m³ y 12 nodos con más de 283 m³) en los que `V_rpt` queda **por debajo** del valor de la API, entre 0,5 y 0,86 m³; su origen está **no verificado**. La etiqueta actual coincide con `V_cont ≥ t` en los 2400 casos nodo-escenario (874 son nodos secos triviales) para cualquier umbral t en el intervalo (0,4988; 0,5002] m³ [s05, s08]. Es una **cuantización determinista, no ruido**: el umbral nominal de 1 m³ funciona como un umbral de ≈ 0,5 m³ de volumen continuo.
3. Si se quiere el volumen continuo, la fuente es `pyswmm` (`node.statistics['flooding_volume']`), que no añade costo apreciable. La integral de la serie `flooding` del `.out` **no sirve** con el `REPORT_STEP` actual de 3 min: se aleja del volumen continuo hasta 9,5 m³ por nodo (\|out − API\|) y del `.rpt` hasta 9,3 m³. Con un `REPORT_STEP` de 5 s, la diferencia máxima frente a la API baja a 0,89 m³ (mediana 0,039). Eso es peor que el `.rpt` en el máximo (0,49) pero mejor en la mediana (0,27) [s01, s02].
4. En los 6 escenarios de control, pasar a `V_cont ≥ 1 m³` cambia 21 etiquetas, todas de 1 a 0. Pasar a `V_cont > 0` cambia 106, todas de 0 a 1 [s01].
5. Re-simular las 1000 corridas cuesta unos **17 min de SWMM**. Sumando la construcción del `.inp`, la lectura de etiquetas y la del drenaje, sin ensamblar features, son unos 20 min. Es una extrapolación con incertidumbre (§5.1) [s03]. El costo real está en la cadena posterior (dataset, SQL, prep_id, banco, Optuna, tablas y validación) y en volver a trazar todas las cifras de la tesis.
6. En LeaveOneShapeOut, 333 de los 380 falsos negativos de XGBoost tienen `V_rpt = 1`, es decir, volumen continuo de unos 0,5 a 1,5 m³. Si se excluyen esas 1894 filas, el F1 agregado pasa de 0,9817 a 0,9847 y el recall de 0,9918 a 0,9989 [s04]. En esas filas, el error absoluto mediano del regresor (1,18 m³) duplica el error máximo de cuantización (0,5 m³) [s08].
7. Se re-simularon 12 de los 72 casos de "deja de inundar al subir el factor". En los 12 el volumen continuo baja de verdad, pero en 2 baja menos de 0,1 m³ y en 3 entre 0,1 y 0,3 m³ [s06, s07] (§4.1).
8. **Recomendación: ruta A.** Se mantiene la resolución y el umbral actuales, se declaran como supuesto (umbral efectivo de ≈ 0,5 m³ continuos) y se reporta la matriz de confusión con la banda separada. La ruta B solo conviene si el autor decide regenerar el dataset completo por otra causa. En ese caso, capturar el volumen continuo tiene costo marginal casi nulo (§7).

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

- **rpt frente a API.** En los 6 escenarios de `s01` (960 casos nodo-escenario: 546 nodos secos, 306 con 0 < V < 100 m³ y 108 con V ≥ 100 m³), `V_rpt = round(V_API)` en los 306 casos con 0 < V < 100 m³ [s05, s08]. En 12 de los 108 casos con V ≥ 100 m³ (todos de más de 283 m³), `V_rpt` queda **por debajo** del valor de la API, entre 0,51 y 0,86 m³ (relativo ≤ 0,18 %). En esos 12 casos `V_rpt = floor(V_API)`, lo que parece un truncamiento o una diferencia entre acumuladores; el origen está **no verificado**. En los 9 escenarios de `s07` aparece un caso más del mismo tipo con un volumen menor: `8C`, `rare_extreme` f = 4,6, con V_API = 83,508 y V_rpt = 83 [s08]. Ninguna de estas diferencias afecta la etiqueta.
- **El `.out` no es una fuente de mayor resolución con la configuración actual.** SWMM escribe allí valores instantáneos cada `REPORT_STEP` = 180 s (60 a 380 registros por corrida; el primero a los 3 min [s01]). Integrar esos valores pierde los desbordes cortos. En `shape_ultra_short_impulse`, la integral del `.out` se aleja del volumen continuo (API) hasta 9,5 m³ por nodo, un error relativo de 344 % frente a V_API en un nodo con V ≥ 1 m³. Frente al `.rpt`, la diferencia llega a 9,3 m³. Usar `out ≥ 1` en vez del `.rpt` cambiaría 30 etiquetas en los 6 escenarios [s01].
- **Prueba con `REPORT_STEP` = 5 s** (`shape_ultra_short_impulse` f = 5, 1 corrida) [s02]:

| Fuente frente a API | máx (m³) | mediana (m³) |
|---|---|---|
| out con `REPORT_STEP` 5 s | 0,891 | 0,039 |
| out con `REPORT_STEP` 180 s | 9,535 | 0,575 |
| rpt | 0,494 | 0,270 |

  Con 5 s el `.out` pasa de 0,89 a 31,8 MB (4680 registros) y aun así cambia 1 etiqueta frente a la API. En el máximo sigue peor que el `.rpt` (0,891 frente a 0,494 m³), pero en la mediana es mejor (0,039 frente a 0,270 m³). Cambiar `REPORT_STEP` no altera el cálculo: la API y el `.rpt` quedan idénticos a los de `s01` (diferencia ≤ 3,6·10⁻¹⁵ m³).
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

- Todos los cambios (i)→(ii) son de 1 a 0 y coinciden con los nodos con `V_cont` en [0,5; 1). Todos los cambios (i)→(iii) son de 0 a 1 y coinciden con los nodos con `V_cont` en (0; 0,5) [s01].
- **Alcance y cota del umbral efectivo [s05, s08].** Se busca el intervalo de umbrales t para los que `(V_rpt ≥ 1) ⇔ (V_cont ≥ t)` se cumple en todos los casos simulados. La cota inferior es el mayor V_cont con etiqueta 0 y la superior, el menor V_cont con etiqueta 1. Los casos con 0 < V < 100 m³ son los únicos informativos para el umbral; los nodos secos y los grandes lo cumplen de forma trivial.

| Conjunto | Escenarios (con desborde) | Casos nodo-escenario | Secos | 0 < V < 100 m³ | V ≥ 100 m³ | Discrepancias de etiqueta | Umbral efectivo t |
|---|---|---|---|---|---|---|---|
| `s01` | 6 (4) | 960 | 546 | 306 | 108 | 0 | (0,4988; 0,5142] |
| `s07` | 9 (9) | 1440 | 328 | 1102 | 10 | 0 | (0,4987; 0,5002] |
| `s01` + `s07` | 15 (13) | 2400 | 874 | 1408 | 118 | 0 | **(0,4988; 0,5002]** |

  `s06` lo confirma con un dato que no quedó guardado por nodo: `66I` vale 0,5002 m³ con etiqueta 1. La repetición independiente del revisor, sobre otros 3 escenarios, dio una cota de (0,4643; 0,5262] (no re-ejecutada aquí). Es más ancha, pero compatible: su intersección con la nuestra sigue siendo (0,4988; 0,5002]. Todo esto vale para **una red y los 13 escenarios con desborde que tienen registro por nodo** (más el dato aislado de `66I` en `s06`). Que el umbral efectivo sea ≈ 0,5 m³ en el resto del dataset es una inferencia: el mecanismo es el redondeo a 3 decimales del `.rpt`, pero se midió solo en esa muestra.
- Los desbordes que hoy quedan con etiqueta 0 son pequeños: máximo 0,499 m³, mediana 0,153 m³, suma de 20,3 m³ en los 4 escenarios con desborde [s05]. En estos controles representan entre el 6 % y el 41 % de los nodos que desbordan. La auditoría informó entre 27 % y 57 % en otros escenarios (f = 1,6 y 2,0, más cerca del inicio de la inundación). Esa cifra viene del anexo 01 (`d11`) y no se volvió a verificar aquí.
- El volumen positivo mínimo que ve la API es 0,0027 m³. La regla (iii) es sensible a esas trazas: con `V_cont > 0,01` habría 412 positivos en vez de 414.
- **Proyección al dataset (estimación, no verificada):** en los controles hay 28 nodos con `V_rpt = 1`, con V_cont entre 0,514 y 1,455 m³ en el conjunto de los 28. De ellos, 21 tienen `V_cont < 1`. De los 7 nodos con `V_rpt = 2`, ninguno. Si esa proporción (21/28 = 75 %, IC95 % de Wilson [56,6 %; 87,3 %]) se mantuviera, la regla (ii) pasaría a 0 unas 1420 de las 1894 filas con `V_rpt = 1` (intervalo de 1073 a 1654 filas), es decir, el 3,1 % de los 46 225 positivos (de 2,3 % a 3,6 %) [s08]. El intervalo supone que los 28 nodos son una muestra representativa, cosa que no son (4 escenarios). La regla (iii) pasaría a 1 alrededor del 32 % de los nodos secos de los escenarios con desborde (106 de 332 en los controles). Ninguna de las dos proyecciones se midió sobre el dataset completo.

### 4.1 Los 72 casos "deja de inundar al subir el factor" [s06, s07]

El anexo 03 (D-04) reporta 72 transiciones de la BD (misma forma y nodo, `inunda = 1` a f y `inunda = 0` a f + 0,2): 68 con `V_rpt = 1` y 4 con `V_rpt = 2`. Las atribuye a "ruido de redondeo".

**Lo deducido (sin simular).** En los casos medidos, `V_rpt` es una función monótona no decreciente de `V_cont`: el redondeo, o el truncamiento en los casos grandes. Por lo tanto, toda transición 1 → 0 de la etiqueta exige que el volumen **continuo** baje entre f y f + 0,2. El redondeo puede exponer una bajada, pero no crearla. Lo que el redondeo sí determina es si una bajada pequeña cruza el umbral: una bajada de 0,03 m³ alrededor de 0,5 m³ cambia la etiqueta, y la misma bajada alrededor de 10 m³ no.

**Lo medido.** Se re-simularon 12 de las 72 transiciones: 3 en `s06` y 9 en `s07`. Esta ronda pedía hasta 8 transiciones nuevas; `rare_extreme` 4,0 → 4,2 entró sin simulación adicional, porque comparte corridas con los pares vecinos. Se cubren 3 formas (`common_bogota`, `rare_extreme` y `shape_ultra_short_impulse`) y 7 pares de factores, incluidos los 4 casos con `V_rpt = 2`. Las 60 restantes **no se re-simularon (no verificado)**. En todas las corridas, la BD coincide con el `.rpt` replicado en 160 de 160 nodos.

| Forma | f → f + 0,2 | Nodo | V_rpt antes | V_cont antes | V_rpt después | V_cont después | Bajada de V_cont (m³) |
|---|---|---|---|---|---|---|---|
| `common_bogota` | 3,4 → 3,6 | `107C` | 1 | 0,8623 | 0 | 0,0000 | 0,862 |
| `common_bogota` | 3,4 → 3,6 | `52C` | 1 | 0,8675 | 0 | 0,0197 | 0,848 |
| `common_bogota` | 3,4 → 3,6 | `66I` | 1 | 0,5002 | 0 | 0,4663 | 0,034 |
| `common_bogota` | 4,4 → 4,6 | `51C` | 1 | 0,8617 | 0 | 0,3936 | 0,468 |
| `rare_extreme` | 3,8 → 4,0 | `16I` | 1 | 0,5516 | 0 | 0,4738 | 0,078 |
| `rare_extreme` | 3,8 → 4,0 | `40I` | 1 | 0,5221 | 0 | 0,4202 | 0,102 |
| `rare_extreme` | 3,8 → 4,0 | `51C` | 2 | 2,1310 | 0 | 0,2586 | 1,872 |
| `rare_extreme` | 4,0 → 4,2 | `66I` | 1 | 0,6192 | 0 | 0,4561 | 0,163 |
| `rare_extreme` | 4,2 → 4,4 | `40C` | 2 | 2,4388 | 0 | 0,1498 | 2,289 |
| `rare_extreme` | 4,4 → 4,6 | `16I` | 1 | 0,6038 | 0 | 0,4372 | 0,167 |
| `rare_extreme` | 4,4 → 4,6 | `51C` | 2 | 2,3164 | 0 | 0,2789 | 2,037 |
| `shape_ultra_short_impulse` | 4,4 → 4,6 | `40C` | 2 | 1,7461 | 0 | 0,0000 | 1,746 |

**Distribución medida (n = 12):**

| Bajada de V_cont | Casos |
|---|---|
| ≥ 0,3 m³ | 7 (los 4 con `V_rpt = 2` y 3 con `V_rpt = 1`) |
| Entre 0,1 y 0,3 m³ | 3 |
| < 0,1 m³ | 2 (`66I` 0,034 y `16I` 0,078) |

En los 12 casos el volumen continuo baja de verdad, pero solo en 7 la bajada es grande frente al umbral.

**Lectura:**

- La explicación del anexo 03 es **parcialmente correcta**. En 5 de los 12 casos (los de bajada < 0,3 m³) el volumen continuo pasa de poco más a poco menos de 0,5 m³. Esos casos son de borde: una variación pequeña que el umbral convierte en un cambio de etiqueta. En los otros 7, la bajada es grande y no tiene relación con la resolución.
- La causa física de las bajadas no se investigó. Que el desborde se redistribuya entre nodos al aumentar la carga es una **interpretación no verificada**.
- **Con un umbral continuo de 1 m³ [s07].** En los 6 pares de `s07`, la regla actual produce 9 transiciones no monótonas y la regla `V_cont ≥ 1` produce 5. De esas 5, 4 son los mismos nodos con `V_rpt = 2` (`51C` dos veces y `40C` dos veces) y 1 es nueva: `12C` en `rare_extreme` 4,0 → 4,2, que baja de ≥ 1 a < 1 m³ sin cambiar su `V_rpt`. En el par `common_bogota` 3,4 → 3,6 de `s06`, la regla `V_cont ≥ 1` no produce ningún caso (0 transiciones). Es decir: cambiar el umbral reduce estos casos en la muestra, pero no los elimina, y aparecen otros nuevos alrededor del nuevo umbral.

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

- **Alcance de "corrida completa":** construcción del `.inp`, SWMM, lectura de etiquetas y lectura del drenaje. No incluye el ensamblado de features, el dataset ni la escritura en la BD.
- **Incertidumbre de la extrapolación:** el ajuste usa 6 corridas con solo 4 horizontes distintos (3; 6,5; 6,75 y 19 h). Las dos corridas de 19 h difieren en un 18 % (1,656 frente a 1,398 s). Las cifras de 16,8 y 20,4 min deben leerse como un orden de magnitud (decenas de minutos), no como estimaciones precisas.
- No incluye los reintentos por drenaje de `run_batch_shapes` (hasta 2 extensiones de +2 h cada una). Cuántas corridas reintentarían está **no verificado**. Una cota superior gruesa, sin medir, es que cada corrida se simule 3 veces con horizontes más largos, lo que llevaría el total a algo más del triple (del orden de 1 h).
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
| 8 | Regenerar todas las tablas de `docs/auditoria/correcciones/datos/` (`comparacion_protocolos.csv`, `matriz_confusion.csv`, `banda_limitrofe.csv`, `por_forma.csv`, `runs/*.json`). Quedarían invalidados `datos/tables_provenance.json` (hashes de OOF y CSV, prep_id) y `datos/README.md` (prep_id `ca706e27654e0cf5`, definición de la etiqueta, comandos). También habría que actualizar este documento y las cifras de la tesis | Trabajo editorial; no medido |

El cuello de botella no es SWMM, sino los pasos 1 y 4–8 y el riesgo de desalinear la trazabilidad ya construida (D-01, D-06).

---

## 6. Efecto en la evaluación (ruta A, solo con los OOF guardados) [s04]

La matriz reconstruida desde los OOF coincide con `matriz_confusion.csv` en los 6 pares (modelo, protocolo).

### 6.1 Matriz de confusión con la banda separada — XGBoost, LeaveOneShapeOut

| Volumen real (`.rpt`) | Etiqueta real | n | Predicho 0 | Predicho 1 |
|---|---|---|---|---|
| 0 (V_cont < ≈ 0,5) | 0 | 107 855 | 106 526 (TN) | 1329 (FP) |
| [1, 2) = solo el entero 1 (V_cont ≈ [0,5; 1,5)) | 1 | 1894 | 333 (FN) | 1561 (TP) |
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
- **Muchos FP también están cerca de la frontera.** De los 1329 FP de LeaveOneShapeOut, el regresor predice menos de 2 m³ en 578 y menos de 5 m³ en 1127. `y_pred_reg` es la predicción del regresor sobre todas las filas de test, no la enrutada por el clasificador (`evaluate._fit_predict_fold`). La probabilidad media del clasificador en los FN de la banda es 0,241.
- Los FN de la banda se concentran en formas fuera de rango o de transición: `new_hydrograph` 77, `rare_extreme` 27, `shape_ultra_short_impulse` 27, `common_bogota` 26 y `shape_flash_flood_urban` 20.

### 6.3 Lo que sugiere la ruta B para las métricas (no verificado)

Con etiquetas continuas `V_cont ≥ 1`, unas 1070 a 1650 filas de la banda pasarían a negativas (proyección de §4). Eso movería la frontera, pero no la eliminaría. Es una **interpretación** esperar que siga habiendo una banda de nodos con V_cont cerca de 1 m³ difícil de clasificar. Sin reentrenar no se puede saber si el F1 sube o baja. Evaluar los OOF actuales contra etiquetas reetiquetadas no tendría sentido, porque el modelo se entrenó con la otra frontera.

Que los FN se concentren en la banda es compatible con que la dificultad venga de la cercanía al umbral. En la muestra medida, la etiqueta actual es una función determinista del volumen continuo (§4), así que la cuantización no añade ruido a la etiqueta. No se demostró que esa sea la causa: es una **interpretación**.

### 6.4 Efecto sobre el regresor [s08]

El objetivo de regresión (`vol_inundacion_m3`) también está cuantizado a enteros. El error de cuantización es como máximo ±0,5 m³, es decir, ±50 % a 1 m³ y ±10 % a 5 m³. Para ver si ese error pesa frente al error del modelo, se midió el error de XGBoost en LeaveOneShapeOut sobre las filas inundadas con volumen real pequeño, usando los OOF guardados (`y_pred_reg`, predicción del regresor en todas las filas de test).

| Volumen real (m³) | n | \|error\| mediana (m³) | \|error\| media (m³) | p90 (m³) | Error relativo mediano | Sesgo medio (m³) | Filas con \|error\| ≤ 0,5 m³ | Error máximo de cuantización relativo |
|---|---|---|---|---|---|---|---|---|
| = 1 | 1894 | 1,181 | 2,002 | 4,702 | 1,18 | +1,94 | 30,6 % | 0,50 |
| = 2 | 969 | 1,821 | 2,839 | 6,955 | 0,91 | +2,73 | 20,0 % | 0,25 |
| [1, 5] | 4925 | 1,626 | 2,828 | 6,882 | 0,77 | +2,57 | 22,0 % | 0,50 |
| [3, 5] | 2062 | 2,035 | 3,581 | 9,045 | 0,52 | +3,07 | 15,0 % | 0,17 |
| [5, 20] | 6372 | 3,866 | 6,356 | 15,162 | 0,36 | +3,33 | 7,1 % | 0,10 |
| [20, 100] | 12 059 | 13,853 | 19,909 | 44,067 | 0,31 | +3,74 | 2,0 % | 0,03 |

**Lectura (interpretación):**

- En el rango de 1 a 5 m³, el error absoluto mediano del modelo (1,6 m³) es más de 3 veces el error máximo de cuantización (0,5 m³). Solo el 22 % de las filas cae dentro de ±0,5 m³.
- Además, el modelo **sobreestima** de forma sistemática (sesgo de +2 a +3 m³). Eso no se explica por una cuantización simétrica de ±0,5 m³.
- En este rango, el error del regresor está dominado por el modelo y no por la resolución de la etiqueta. Un objetivo continuo reduciría como mucho un componente de ≤ 0,5 m³ por fila, y no se puede saber cuánto de ese componente aprovecharía el modelo sin reentrenar.

---

## 7. Recomendación

**Ruta A: mantener la resolución y el umbral, declararlos y reportar la matriz con la banda separada.**

**Razones:**

1. **La etiqueta actual es una cuantización determinista, no ruido.** En la muestra con registro por nodo (una red, 13 escenarios con desborde, 2400 casos nodo-escenario de los que 1408 son informativos) es `V_cont ≥ t` con t en (0,4988; 0,5002] m³, sin ninguna discrepancia [s05, s08]. El problema de D-04/H-06 es de **semántica**: el umbral nominal de 1 m³ funciona como ≈ 0,5 m³ continuos. Se resuelve declarándolo, sin tocar los datos.
2. **La ruta B no elimina el fenómeno que motivó el hallazgo.** Con un umbral continuo de 1 m³, la frontera se desplaza unos 0,5 m³ y cambia la etiqueta de entre el 2,3 % y el 3,6 % de los positivos (proyección, §4). Que los FN de la banda reflejen la dificultad propia de cualquier frontera es una **interpretación**: ningún reentrenamiento lo comprobó. Las transiciones no monótonas se redujeron, pero no desaparecieron, con `V_cont ≥ 1` en los pares re-simulados: 5 en lugar de 9, una de ellas nueva (§4.1).
3. **El regresor tampoco lo justifica.** En las filas de 1 a 5 m³, el error mediano del modelo (1,6 m³) y su sesgo (+2,6 m³) superan claramente el error máximo de cuantización (0,5 m³) (§6.4).
4. **Relación costo/beneficio.** SWMM es barato (del orden de 17 a 20 min), pero la ruta B obliga a rehacer el dataset, la BD, el `prep_id`, el banco, Optuna, la validación y todas las cifras ya trazadas de la tesis (§5.2). Todo eso para un cambio de definición que la ruta A expresa sin costo.
5. **Condición que invierte la recomendación (hipótesis condicional).** Solo si el autor decide **regenerar el dataset completo** por otra causa, conviene incorporar la ruta B en esa misma regeneración. Capturar `node.statistics['flooding_volume']` cuesta ≤ 0,023 s por corrida. Habría que guardar las dos columnas (`vol_rpt_m3` y `vol_continuo_m3`) y definir la etiqueta con un umbral elegido de forma explícita.
   - Una de esas causas posibles es H-01 (InitFlow fijo a factor 1). Su impacto es la cifra del auditor (anexo 01) y **no se re-simuló aquí**. Además, H-01 comparte la severidad 🟠 con H-02, que no exige re-simular.
   - El informe consolidado propone para H-01 y H-05 una re-simulación de un **subconjunto**, para medir sensibilidad, y no la regeneración de las 1000 corridas.
   - Una re-simulación de un subconjunto **no** activa la ruta B: no cambia las etiquetas del dataset. Sí permitiría, sin costo adicional, registrar el volumen continuo en ese subconjunto y ampliar la muestra de §4.
6. **Fuente descartada.** No debe usarse la integral del `.out` con `REPORT_STEP` = 3 min. Se aleja del volumen continuo hasta 9,5 m³ por nodo, mucho más que la cuantización que se quiere corregir (≤ 0,51 m³; el caso mayor por debajo de 100 m³ es `8C`, que difiere en 0,508 m³) [s01, s02, s08]. La misma corrección está pendiente en la nota de `anexos/01-hidraulica-swmm.md` (ver `04-informe-consolidado.md`).

**Costo de cada ruta:**

| Ruta | Simulaciones | Código | Re-evaluación | Cifras de la tesis |
|---|---|---|---|---|
| A | 0 | Ninguno obligatorio. Opcional: añadir la tabla de §6.1 a `scripts/informe_evaluacion.py` (ya existe `banda_limitrofe.csv`). Recomendable aparte: corregir H-07 (parser de respaldo), que no cambia datos | Ninguna: §6 sale de los OOF existentes | Solo agregar la declaración y la matriz por banda |
| B | 1000 (del orden de 17–20 min; con reintentos por drenaje, no verificado) | Captura del volumen continuo en 3 rutas, más tests | Dataset, BD, `prep_id`, banco (≈ 5,5 min), Optuna, validación | Todas las de evaluación |

---

## 8. Decisión pendiente del autor

1. **Elegir la ruta A o la B.** La recomendación es A, salvo que se decida regenerar el dataset completo por otra causa (§7.5); en ese caso, B dentro de esa misma regeneración.
2. **Redacción del umbral (ruta A).** Mantener "≥ 1 m³ del `.rpt`" y declarar que es consistente con "≈ 0,5 m³ de volumen continuo", o enunciar el umbral directamente como ≈ 0,5 m³ continuos. Los datos son los mismos en ambos casos.
3. **Métrica principal frente a sensibilidad.** Confirmar tres cosas:
   - la métrica principal sigue siendo la matriz completa (F1 0,9817 en LeaveOneShapeOut);
   - la matriz por banda (§6.1) se reporta siempre;
   - la exclusión de la banda (§6.2) va solo como sensibilidad, con la advertencia de asimetría.
4. **Ruta B: umbral continuo.** Si se elige B, decidir el umbral (0,5, 1 o 5 m³, o un barrido de sensibilidad como propone H-06) **antes** de mirar las métricas.
5. **H-07.** Corregir el parser de respaldo (lectura por encabezado y fallo explícito). Es independiente de la ruta elegida.
6. **Revisar las notas de corrección aplicadas** a los documentos anteriores (§8.1) y decidir si su texto se integra en la versión final de la auditoría.

### 8.1 Correcciones a documentos anteriores

Se aplicaron en el mismo commit que esta revisión, como notas fechadas añadidas sin reescribir el texto original (script `s09_notas.py`).

| Documento y lugar | Afirmación original | Corrección | Estado |
|---|---|---|---|
| `anexos/03-datos-persistencia-sql.md`, fila D-04 (l. 54; nota tras la tabla) | "Son ruido de redondeo y de cálculo en el borde, no física" | 12 de las 72 transiciones re-simuladas. En las 12 baja el volumen continuo: 7 con bajadas ≥ 0,3 m³ y 2 con bajadas < 0,1 m³ (borde). Las otras 60 están sin verificar | Aplicada |
| Ídem | "el umbral no filtra nada distinto del redondeo del informe" | Cuantización determinista equivalente a `V_cont ≥ t`, t ∈ (0,4988; 0,5002]. Filtra desbordes reales < ≈ 0,5 m³ | Aplicada |
| Ídem | "leer el volumen desde el `.out` binario, que no está redondeado" | Descartado con `REPORT_STEP` de 3 min (hasta 9,5 m³ de error). La fuente es la estadística de pyswmm | Aplicada |
| Ídem | "umbral con margen sobre la resolución (p. ej. 5-10 m³)" | Desplaza la banda, no la elimina. El umbral es una decisión semántica | Aplicada |
| `2026-10-10-auditoria-completa.md` l. 125, fila D-04 (nota tras la tabla 4.1) | "72 casos … ruido de redondeo" | Ídem: parcialmente correcto, 12 re-simulados | Aplicada |
| Ídem l. 186 | "inestables por construcción" | Etiqueta determinista. La relación de los FN con la cercanía al umbral es una interpretación | Aplicada |
| Ídem l. 197 | "banda limítrofe (volumen real de 1 a 2 m³)" | El único entero en [1, 2) es 1, que corresponde a V_cont ≈ [0,5; 1,5) | Aplicada |
| Ídem l. 229, ítem 10 | "Leer el volumen inundado con mayor resolución que el `.rpt`" | La integral del `.out` no es mejor fuente con el `REPORT_STEP` actual; solo la estadística de la API | Aplicada |
| Ídem l. 263, supuesto 3 | "la etiqueta es `volumen ≥ 1 m³`" | Se añade el umbral efectivo de ≈ 0,5 m³ continuos, acotado | Aplicada |
| `anexos/01-hidraulica-swmm.md` l. 111, recomendación de H-06 | "o la integración del `.out`" | Descartada con el `REPORT_STEP` actual | Aplicada |
| `correcciones/datos/README.md` l. 20 | "Una fila está inundada si su volumen es ≥ 1 m³" | Se añade el umbral efectivo de ≈ 0,5 m³ continuos | Aplicada |
| `2026-10-10-auditoria-completa.md` l. 215 (decisión 4: "¿Se trata aparte la banda 1–2 m³?") | Usa "banda 1–2 m³" sin precisar que es solo el entero 1 | Misma aclaración que en l. 197 | **No aplicada**: fuera del alcance autorizado en esta ronda. La nota de l. 197 la cubre |
| `anexos/01-hidraulica-swmm.md` H-06 "Qué pasa" y tabla de resumen; `anexos/03` l. 195 | "≥ 1 m³ equivale a ≥ 0,5 m³ continuo"; conteo de 72 transiciones | Revisadas: son coherentes con lo medido. No requieren corrección | — |

### 8.2 Texto para la tesis (ruta A)

> La variable objetivo de clasificación se define a partir del volumen desbordado por nodo que reporta SWMM 5.2 en la tabla *Node Flooding Summary* del archivo de reporte (`.rpt`). Esa tabla expresa el volumen en 10⁶ L con tres decimales, por lo que la resolución del volumen disponible es de 1 m³. Un nodo se etiqueta como inundado (`inunda = 1`) si su volumen reportado es de al menos 1 m³. En re-simulaciones de control sobre la misma red (17 escenarios, 15 de ellos con desborde) se comparó el volumen reportado con el volumen que SWMM acumula en cada paso de cálculo; la comparación nodo a nodo se registró en 15 de esos escenarios (13 con desborde). En los nodos con desbordes menores de 100 m³, el valor del reporte coincidió con el redondeo al entero de ese volumen continuo, salvo un caso de 83,5 m³. En algunos nodos con volúmenes grandes el reporte quedó hasta 0,86 m³ por debajo, por una causa no identificada que no afecta la etiqueta. En consecuencia, en esa muestra la regla es consistente con un umbral de ≈ 0,5 m³ sobre el volumen continuo, acotado al intervalo (0,4988; 0,5002] m³. Los desbordes menores de ese umbral (mediana de 0,15 m³ en los escenarios de control) se consideran no inundación. La etiqueta es, por tanto, una cuantización determinista del volumen simulado, no una fuente de ruido. Que ese comportamiento se mantenga en todo el dataset es una inferencia basada en el mecanismo de redondeo del reporte y no se verificó caso por caso. Los nodos con volumen cercano al umbral son, además, los más difíciles de clasificar: en la validación por forma retenida (LeaveOneShapeOut), 333 de los 380 falsos negativos de XGBoost corresponden a nodos con volumen reportado de 1 m³ (volumen continuo de aproximadamente 0,5 a 1,5 m³). Por ello, además de la matriz de confusión agregada, se informa la matriz separada por clase de volumen real y, como análisis de sensibilidad, las métricas sin esa banda (F1 de 0,9817 con todas las filas y de 0,9847 sin la banda). Esta sensibilidad excluye solo el lado positivo de la frontera, porque los desbordes de 0 a 0,5 m³ no se distinguen de los nodos secos en los datos disponibles.

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
| `s07_no_monotonia_ampliada.py` | `<py> s07_no_monotonia_ampliada.py <repo>` | 9 | 9 transiciones más (incluidos los 4 casos con 2 m³); transiciones con `V_cont ≥ 1` (`s07_nodos.csv`, `s07_transiciones.csv`) |
| `s08_cota_y_regresor.py` | `<py> s08_cota_y_regresor.py <repo>` | 0 | Desglose de casos y cota del umbral efectivo; error del regresor por banda desde los OOF; IC de Wilson |
| `s09_notas.py` | `<py> s09_notas.py <repo>` | 0 | Inserta las notas de corrección de §8.1 en los documentos anteriores (escritura solo en `docs/auditoria/`) |

Orden de ejecución: `s01` → `s02` → `s03` → `s04` → `s05` → `s06` → `s07` → `s08` → `s09`. `s02`, `s03`, `s04`, `s05` y `s08` leen salidas de `s01` (y `s08` también de `s07`). Total: 18 simulaciones SWMM, cada una de menos de 2 s.

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

### `s07_no_monotonia_ampliada.py`

```python
"""Spike D-04, ronda de correcciones: re-simula mas transiciones 'deja de inundar al subir el factor'
(las 72 de la BD: inunda=1 a f y vol<1 a f+0.2, misma forma y nodo) y mide el volumen continuo
(pyswmm node.statistics['flooding_volume']) antes y despues. 9 corridas SWMM sobre copias en ./runs:
  rare_extreme f = 3.8, 4.0, 4.2, 4.4, 4.6  (transiciones 3.8->4.0, 4.0->4.2, 4.2->4.4, 4.4->4.6)
  shape_ultra_short_impulse f = 4.4, 4.6     (4.4->4.6)
  common_bogota f = 4.4, 4.6                 (4.4->4.6)
Incluye los 4 casos con vol .rpt = 2.0. Tambien guarda todos los nodos (s07_nodos.csv) para acotar el
umbral efectivo y contar transiciones no monotonas con el umbral continuo de 1 m3.
Uso: python s07_no_monotonia_ampliada.py <repo>   (desde spike_d04)"""
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
db = pd.concat([pd.read_parquet(PREP / "keys.parquet")[["shape_id", "factor_mult", "node_id"]],
                pd.read_parquet(PREP / "targets.parquet")], axis=1)
db["f5"] = (db.factor_mult * 5).round().astype(int)

PLAN = {"rare_extreme": [3.8, 4.0, 4.2, 4.4, 4.6], "shape_ultra_short_impulse": [4.4, 4.6], "common_bogota": [4.4, 4.6]}
res = {}
for sid, fs in PLAN.items():
    shape = load_shape(repo / "data/hydrograph_shapes" / f"{sid}.csv")
    for f in fs:
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
        bd = db[(db.shape_id == sid) & (db.f5 == round(f * 5))].set_index("node_id").vol_inundacion_m3
        d = pd.DataFrame({"vol_rpt": lab.vol_inundacion_m3, "vol_api": pd.Series(api), "vol_bd": bd.reindex(lab.index)})
        res[(sid, f)] = d
        print(f"{sid} f={f}: BD==rpt en {int((d.vol_bd == d.vol_rpt).sum())}/160; suma rpt={d.vol_rpt.sum():.0f}", flush=True)

allnodes = pd.concat({k: v for k, v in res.items()}, names=["shape_id", "factor", "node_id"]).reset_index()
allnodes.to_csv("s07_nodos.csv", index=False)

rows, nm_cont = [], []
for sid, fs in PLAN.items():
    for f1, f2 in zip(fs[:-1], fs[1:]):
        a, b = res[(sid, f1)], res[(sid, f2)]
        for n in a.index[(a.vol_bd >= 1) & (b.vol_bd < 1)]:  # transiciones de la BD
            rows.append(dict(shape_id=sid, f1=f1, f2=f2, node_id=n, rpt_f1=a.vol_rpt[n], cont_f1=a.vol_api[n],
                             rpt_f2=b.vol_rpt[n], cont_f2=b.vol_api[n], caida=a.vol_api[n] - b.vol_api[n]))
        m = (a.vol_api >= 1) & (b.vol_api < 1)
        nm_cont.append(dict(shape_id=sid, f1=f1, f2=f2, transiciones_rpt=int(((a.vol_rpt >= 1) & (b.vol_rpt < 1)).sum()),
                            transiciones_cont_ge1=int(m.sum()), nodos=list(a.index[m]),
                            nodos_cont_baja=int(((b.vol_api < a.vol_api) & (a.vol_api > 0)).sum()),
                            nodos_desbordan_f1=int((a.vol_api > 0).sum())))
t = pd.DataFrame(rows); t.to_csv("s07_transiciones.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
print("n transiciones re-simuladas:", len(t), " caida>=0.3:", int((t.caida >= 0.3).sum()), " caida<0.1:", int((t.caida < 0.1).sum()),
      " 0.1<=caida<0.3:", int(((t.caida >= 0.1) & (t.caida < 0.3)).sum()))
print(pd.DataFrame(nm_cont).to_string(index=False))
```

### `s08_cota_y_regresor.py`

```python
"""Spike D-04, ronda de correcciones (sin simular):
(1) Alcance de la equivalencia rpt = round(V_cont): desglose de los casos nodo-escenario (secos, 0<V<100, >=100)
    y cota del umbral efectivo t tal que (V_rpt >= 1) <=> (V_cont >= t), con s01 (6 escenarios) y s07 (9 escenarios).
(2) Efecto de la cuantizacion sobre el regresor: OOF de XGBoost en LeaveOneShapeOut (solo lectura), filas con
    volumen real en [1, 5] m3.
(3) Intervalo de Wilson al 95 % para la proporcion 21/28 de s01.
Uso: python s08_cota_y_regresor.py <repo>   (desde spike_d04, despues de s01 y s07)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]).resolve()

# (1) alcance y cota
s01 = pd.read_csv("s01_nodos.csv")[["shape_id", "factor", "node_id", "vol_rpt", "vol_api"]]
s07 = pd.read_csv("s07_nodos.csv")[["shape_id", "factor", "node_id", "vol_rpt", "vol_api"]]
for name, d in [("s01", s01), ("s07", s07), ("s01+s07", pd.concat([s01, s07]))]:
    seco = d.vol_api == 0; mid = (d.vol_api > 0) & (d.vol_api < 100); big = d.vol_api >= 100
    neg = d[d.vol_rpt < 1]; pos = d[d.vol_rpt >= 1]
    lo = neg.vol_api.max(); hi = pos.vol_api.min()
    print(f"[{name}] casos={len(d)} escenarios={d.groupby(['shape_id','factor']).ngroups} con desborde={int((d.groupby(['shape_id','factor']).vol_api.max()>0).sum())}; "
          f"secos={int(seco.sum())} 0<V<100={int(mid.sum())} V>=100={int(big.sum())}; "
          f"rpt!=round(V) en 0<V<100: {int((d.vol_rpt[mid] != np.round(d.vol_api[mid])).sum())}; "
          f"en V>=100: {int((d.vol_rpt[big] != np.round(d.vol_api[big])).sum())} de {int(big.sum())}; "
          f"cota umbral: ({lo:.4f}, {hi:.4f}]  [max V_cont con etiqueta 0, min V_cont con etiqueta 1]")
    b = d[big]; e = b.vol_rpt - b.vol_api
    print(f"    V>=100: rpt-api min={e.min():.3f} max={e.max():.3f}; nodos con rpt-api < -0.5: {int((e < -0.5).sum())}; "
          f"con rpt-api > +0.5: {int((e > 0.5).sum())}; floor(api)==rpt en esos: "
          f"{int((np.floor(b.vol_api[e < -0.5]) == b.vol_rpt[e < -0.5]).sum())}")

# (2) regresor
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
tg = pd.read_parquet(PREP / "targets.parquet"); tg["sample_idx"] = np.arange(len(tg))
o = pd.read_parquet(repo / "outputs/bench/oof/ca706e27654e0cf5/xgboost.parquet")
o = o[o.protocol == "LeaveOneShapeOut"].merge(tg, on="sample_idx")
print("\n[regresor] y_pred_reg = expm1 de la prediccion del regresor sobre TODAS las filas de test (evaluate._fit_predict_fold)")
for lo_, hi_ in [(1, 1), (2, 2), (1, 5), (3, 5), (5, 20), (20, 100)]:
    m = (o.vol_inundacion_m3 >= lo_) & (o.vol_inundacion_m3 <= hi_) & (o.inunda == 1)
    x = o[m]; ae = (x.y_pred_reg - x.vol_inundacion_m3).abs(); re_ = ae / x.vol_inundacion_m3
    sesgo = (x.y_pred_reg - x.vol_inundacion_m3).mean()
    print(f"  V real en [{lo_},{hi_}] (n={len(x)}): |err| mediana={ae.median():.3f} media={ae.mean():.3f} p90={ae.quantile(.9):.3f}; "
          f"rel mediana={re_.median():.3f} media={re_.mean():.3f}; sesgo medio={sesgo:+.3f}; "
          f"frac |err|<=0.5: {(ae <= 0.5).mean():.3f}; cuantizacion max relativa = {0.5/lo_:.2f}")

# (3) Wilson
k, n, z = 21, 28, 1.959964
p = k / n; den = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
lo_w, hi_w = c - h, c + h
print(f"\n[Wilson] {k}/{n}={p:.3f}; IC95 % = [{lo_w:.3f}, {hi_w:.3f}] -> filas de 1894: [{lo_w*1894:.0f}, {hi_w*1894:.0f}] "
      f"(punto {p*1894:.0f}); % de 46 225 positivos: [{100*lo_w*1894/46225:.1f}, {100*hi_w*1894/46225:.1f}] (punto {100*p*1894/46225:.1f})")
```

### `s09_notas.py`

```python
"""Inserta (sin reescribir texto existente) las notas de correccion fechadas del spike D-04 en los documentos
de auditoria anteriores. Cada insercion se ancla al texto exacto de la linea previa y conserva el fin de linea.
Uso: python s09_notas.py <repo>"""
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
P = "> **Nota de corrección (2026-10-11, spike D-04):** "

NOTES = [
 ("docs/auditoria/anexos/03-datos-persistencia-sql.md", "| **D-22** |", "", [
  P + "ver `../correcciones/03-resolucion-de-la-etiqueta.md`. Sobre **D-04**:",
  ">",
  "> 1. \"Son ruido de redondeo\" no se sostiene en general. En la muestra re-simulada (una red, 13 escenarios con desborde), el volumen del `.rpt` es una función monótona del volumen continuo, así que toda transición 1→0 exige que el volumen continuo baje. De las 72 transiciones se re-simularon 12, y en las 12 el volumen continuo baja: 7 con bajadas ≥ 0,3 m³ (entre ellas las 4 con 2 m³), 3 con bajadas de 0,1 a 0,3 m³ y 2 con bajadas < 0,1 m³ (casos de borde alrededor de 0,5 m³). Las otras 60 no se verificaron.",
  "> 2. \"El umbral no filtra nada distinto del redondeo\": en la muestra, la etiqueta es determinista y equivale a un volumen continuo ≥ t, con t en (0,4988; 0,5002] m³. Sí filtra desbordes reales menores de ≈ 0,5 m³: 106 de los 414 nodos con desborde en 6 escenarios de control, con un máximo de 0,499 m³. Es una cuantización determinista, no ruido.",
  "> 3. \"Leer el volumen desde el `.out` binario\" queda descartado con el `REPORT_STEP` actual de 3 min: la integral del `.out` se aleja del volumen continuo hasta 9,5 m³ por nodo. La fuente adecuada es la estadística `node.statistics['flooding_volume']` de pyswmm.",
  "> 4. Un \"umbral con margen (5-10 m³)\" no elimina la banda limítrofe, solo la desplaza. Elegir el umbral es una decisión semántica pendiente del autor y no corrige la resolución.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "| T-06 | 🟠 |", "", [
  P + "en la fila D-04, atribuir los 72 casos a \"ruido de redondeo\" es solo parcialmente correcto. Se re-simularon 12: en los 12 el volumen continuo baja, en 7 con bajadas ≥ 0,3 m³ y en 2 con bajadas < 0,1 m³ (casos de borde). Las otras 60 no se verificaron. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §4.1.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "Lectura: en LeaveOneShapeOut ningún falso negativo supera 67 m³", "", [
  P + "\"inestables por construcción\" exagera. En la muestra re-simulada, la etiqueta es una función determinista del volumen continuo, sin ruido, con un umbral efectivo en (0,4988; 0,5002] m³. Lo que sí se midió es que la banda `V_rpt = 1` (volumen continuo de ≈ 0,5 a 1,5 m³) es la más próxima al umbral y concentra 333 de los 380 FN. Que esa cercanía explique los FN es una interpretación. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §6.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "   - la matriz por forma para las formas fuera de rango", "   ", [
  P + "con la resolución de 1 m³, el único volumen reportado en [1, 2) m³ es el entero 1, que corresponde a un volumen continuo de ≈ 0,5 a 1,5 m³. La matriz por banda está en `correcciones/03-resolucion-de-la-etiqueta.md` §6.1.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "10. Leer el volumen inundado con mayor resolución que el `.rpt`", "    ", [
  P + "con el `REPORT_STEP` actual (3 min), la integral de la serie `flooding` del `.out` **no** es una fuente de mayor resolución: se aleja del volumen continuo hasta 9,5 m³ por nodo, y hasta 0,89 m³ incluso con 5 s. La única fuente medida que da el volumen continuo es `node.statistics['flooding_volume']` de pyswmm. El spike recomienda la ruta A (declarar, sin re-simular). Ver `correcciones/03-resolucion-de-la-etiqueta.md`.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "3. La inundación es el volumen que reporta SWMM por nodo, con **resolución de 1 m³**", "   ", [
  P + "en la muestra re-simulada (una red, 13 escenarios con desborde), \"volumen reportado ≥ 1 m³\" es consistente con un umbral efectivo de ≈ 0,5 m³ sobre el volumen continuo, acotado a (0,4988; 0,5002] m³. Los desbordes menores cuentan como \"no inunda\".",
 ]),
 ("docs/auditoria/anexos/01-hidraulica-swmm.md", "- **Recomendación:** tomar el volumen continuo (`flooding_volume` de PySWMM o la integración del `.out`)", "  ", [
  P + "queda descartada la \"integración del `.out`\" con el `REPORT_STEP` actual de 3 min. Se aleja del volumen continuo hasta 9,5 m³ por nodo, más que la cuantización del `.rpt` (≤ 0,5 m³ por debajo de 100 m³). Si se quiere el volumen continuo, la fuente es `flooding_volume` de pyswmm. Ver `../correcciones/03-resolucion-de-la-etiqueta.md`.",
 ]),
 ("docs/auditoria/correcciones/datos/README.md", "  37 celdas (forma, factor). Una fila está inundada si su volumen es ≥ 1 m³.", "  ", [
  P + "el volumen es el del `.rpt`, con resolución de 1 m³. En la muestra re-simulada (una red, 13 escenarios con desborde), \"≥ 1 m³\" es consistente con un umbral efectivo de ≈ 0,5 m³ de volumen continuo, acotado a (0,4988; 0,5002] m³. Ver `../03-resolucion-de-la-etiqueta.md`.",
 ]),
]

for rel, anchor, indent, lines in NOTES:
    path = repo / rel
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    L = raw.split(eol)
    idx = [i for i, l in enumerate(L) if l.startswith(anchor)]
    assert len(idx) == 1, (rel, anchor, idx)
    i = idx[0]
    if any("spike D-04" in l for l in L[i + 1:i + 4]):
        print("ya existe:", rel, anchor[:40]); continue
    block = [""] + [indent + l if l else indent.rstrip() for l in lines]
    if i + 1 < len(L) and L[i + 1].strip() != "":
        block = block + [""]
    L[i + 1:i + 1] = block
    path.write_bytes(eol.join(L).encode("utf-8"))
    print(f"insertada: {rel} tras la linea {i + 1} (eol={'CRLF' if eol == chr(13)+chr(10) else 'LF'})")
```
