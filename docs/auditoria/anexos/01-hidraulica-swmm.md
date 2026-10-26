# Anexo 01 — Auditoría hidráulica / SWMM

- **Fecha:** 2026-10-10
- **Rama / commit base:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b`. El árbol tiene cambios sin commitear de otra fase (bench/Optuna); no se tocaron. Los números de línea de `main.py` se refieren al **árbol de trabajo**: en `6e8579b` son 12 líneas más (el diff sin commitear elimina 12 líneas por encima de la línea 225). En los demás archivos citados no hay cambios sin commitear.
- **Motor y librerías (venv del proyecto):** pyswmm 2.1.0, swmm-toolkit 0.17.0, swmm-api 0.4.73.
- **Red auditada:** `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (la que usa `config.yaml`): 160 junctions, 1 outfall (`109C`, FREE), 160 conductos circulares de PVC (n = 0.0082), `FLOW_UNITS LPS`, `DYNWAVE`, `ALLOW_PONDING NO`.
- **Dataset auditado:** `outputs/training_v17.sqlite3` (abierta solo con `mode=ro`): 1000 corridas = 40 formas (`base` + 39 CSV) × 25 factores (0.2–5.0), con 160 000 filas nodo-escenario.

---

## 1. Alcance y método

**Alcance (área hidráulica/SWMM):** `swmm_resilience/simulation/` (runner, swmm_api_io, hydrograph_shapes, timeseries_scenario, batch), `swmm_resilience/extraction/` (static_features, topology, dynamic_features, labels), `swmm_resilience/validation/` (hydrograph_csv, hydrograph_batch), la construcción de features e inferencia en `swmm_resilience/ml/scenario_predict.py`, `predict_from_inp.py` y `predict_tabular.py`, `swmm_resilience/analysis/flood_volume.py` y `resilience.py`, `swmm_resilience/dataset/`, `config.yaml` (parámetros hidráulicos) y los `.inp` de `data/networks/` (solo lectura).

**Método:**
1. Lectura línea a línea del código citado y de las secciones `[OPTIONS]`, `[JUNCTIONS]`, `[CONDUITS]`, `[XSECTIONS]`, `[INFLOWS]` y `[TIMESERIES]` del `.inp`.
2. Consultas de solo lectura a la BD v17 (`file:...?mode=ro`).
3. Unas 150 simulaciones SWMM de control (1 a 3 s cada una), todas sobre **copias** en el scratchpad y armadas con **las mismas funciones del pipeline** (`write_scaled_inp`, `apply_shape` + `write_scenario_inp`, `extract_labels`). Sobre esas copias se aplicaron variantes controladas: InitFlow = 0, END_TIME extendido y corte a cero en vez de rampa. Primero se comprobó que las réplicas reproducen **exactamente** las etiquetas guardadas en la BD (§3, V-03).
4. Scripts `d01`–`d17`, incluidos completos en §5. No se modificó código del proyecto, ni `config.yaml`, ni `outputs/`, ni ningún `.inp`.

**Escala de severidad:** 🔴 CRÍTICO (compromete la validez científica) · 🟠 MAYOR (sesga resultados) · 🟡 MODERADO (limitación o riesgo latente) · 🔵 MENOR (calidad).

**Resultado global:** 0 🔴 · 2 🟠 · 6 🟡 · 6 🔵 (14 hallazgos). No encontré ningún defecto que invalide las etiquetas SWMM del dataset: las réplicas independientes las reproducen al m³. Los dos hallazgos 🟠 son la condición inicial del `.inp` (InitFlow) y la ausencia de un descriptor de volumen del hidrograma entre las features.

---

## 2. Hallazgos

### 2.1 Tabla resumen

| ID | Sev. | Archivo:línea | Descripción | Impacto hidráulico cuantificado |
|---|---|---|---|---|
| H-01 | 🟠 | `.inp` Qx1 `[CONDUITS]` col. InitFlow (l. 220–379); `swmm_api_io.py:229-263`; `timeseries_scenario.py:47-136` | Todos los conductos arrancan con InitFlow = suma de picos aguas arriba **a factor 1**, en las 1000 corridas, sin importar el factor ni la forma | 855 m³ almacenados en t = 0 (hasta 5.9 veces el volumen de entrada del evento). Error de continuidad de hasta 12.4 % (15 de 80 corridas de control > 5 %; con InitFlow = 0 todas quedan en \|CE\| < 1 %). En eventos cortos a f = 5: volumen inundado total +0.7 % a +10.5 % y 2–5 nodos inundados menos |
| H-02 | 🟠 | `hydrograph_shapes.py:31-37`; `dynamic_features.py:5-29`; `contracts.py:46-49` | Las formas se normalizan **por pico** y las únicas features de forma son `duracion_horas` y `tiempo_al_pico_h`. El volumen del evento no aparece en ninguna feature | ∫q_norm·dt varía 72 veces (0.133–9.57 h). Con igual duración y factor, el volumen inundado difiere hasta 13.3 veces (`shape_2h` frente a `common_bogota`, f = 2: 1593 frente a 120 m³) |
| H-03 | 🟡 | `scenario_predict.py:113-124`, `75-95`; `dynamic_features.py:48-54`; `hydrograph_batch.py:54-61`; `main.py:704-726` | El entrenamiento es solo espacialmente uniforme y sincrónico, pero la inferencia acepta CSV arbitrarios por nodo. `tiempo_al_pico_h` se toma de la primera serie del CSV (puede ser nula y dar 0 h). La bandera de extrapolación solo mira pico/base | En entrenamiento, q_pico_nodo/base = factor con desviación de 8.9·10⁻¹⁶ en los 147 nodos (uniformidad total). Con un CSV cuyo primer nodo es nulo, t_pico = 0.0 h aunque el real sea 1.0 h |
| H-04 | 🟡 | `batch.py:7-73`; `csv_backfill.py:147-156` | El entrenamiento no lee ni guarda el error de continuidad de SWMM (solo lo hace la validación) | `runs.continuity_error_pct` es NULL en 1000 de 1000 corridas. Control: 15 de 80 corridas (19 %) con \|CE\| > 5 %, que es el umbral que el propio proyecto usa en validación |
| H-05 | 🟡 | `timeseries_scenario.py:120-123` (docstring 58-61) | El "drenaje" de 6 h agrega el punto (t_fin + 6 h, 0). Si la forma no termina en 0, SWMM interpola una **rampa lineal** y no un corte a cero | Volumen de entrada +24.5 % (`shape_2h`), +13.8 % (`long_moderate`) y +9.8 % (`shape_5h`) frente al hidrograma declarado. Volumen inundado +2.2 % (`long_moderate`, f = 5) y +0.7 % (f = 4); sin cambio en los otros casos probados |
| H-06 | 🟡 | `labels.py:19-23`, `122-139`; `swmm_api_io.py:304-307`; `config.yaml:14` | La etiqueta sale del `.rpt`, con 3 decimales en 10⁶ L, así que la resolución es 1 m³. El umbral "≥ 1 m³" equivale a "≥ 0.5 m³ continuo" | Ningún volumen de la BD está en (0, 1); 1902 filas (4.1 % de los positivos) valen exactamente 1.0, así que cambiar `>=` por `>` las convierte en negativas. En 6 corridas de control, entre 10 y 28 nodos por escenario desbordan (27–57 % de los que desbordan) pero quedan etiquetados como 0 |
| H-07 | 🟡 | `labels.py:75-86` (`parts[-1]`, l. 79); `swmm_api_io.py:293-298` | El parser de texto de respaldo lee la última columna, que en SWMM 5.2 es "Maximum Ponded Depth" y no el volumen | Si se activa, **todas** las etiquetas pasan a 0 sin aviso. Verificado: 0.0 m³ frente a 7669 m³ por swmm_api en `base` f = 3 |
| H-08 | 🟡 | `topology.py:90-118` | La única feature de capacidad es la de los conductos **aguas arriba**. La que gobierna la sobrecarga del nodo es la del conducto de **salida** | `upstream_capacity_lps` es NaN en 57 nodos de cabecera (35.6 % de las filas, imputadas). AUC de q_acum/cap_aguas_arriba = 0.732 frente a 0.884 para q_acum/cap_salida |
| H-09 | 🔵 | `static_features.py:82-87`, `109` | `compute_slope` solo usa cotas de JUNCTIONS, así que `pendiente_out` sale NaN en el nodo que descarga al outfall | 1 nodo (`108C`, 0.6 % de las filas) con pendiente de salida imputada |
| H-10 | 🔵 | `runner.py:646-667`; `batch.py:7-16`; `main.py:992-1020` | La forma `base` se simula con END = 3 h, sin drenaje y con las series originales; las otras 39 formas llevan +6 h | Comprobado: no cambia ninguna etiqueta (END = 9 h da el mismo resultado a f = 0.2, 1, 3 y 5). Quedan sin simular 147–449 m³ almacenados y alrededor del 7.4 % de la recesión |
| H-11 | 🔵 | `labels.py:122-139`; `static_features.py:7-26`; `hydrograph_shapes.py:40-53`; `csv_backfill.py:88` | La ruta de entrenamiento no comprueba `FLOW_UNITS`, ignora Sfactor y Baseline de `[INFLOWS]` y fija `flow_units = 'LPS'` en la BD | Sin impacto actual (LPS, Sfactor = 1 y Baseline = 0 verificados en los 160 inflows). Con unidades US el volumen saldría errado por un factor de 3.785 |
| H-12 | 🔵 | `topology.py:14-32`, `83-88`; `dynamic_features.py:48-54` | `q_pico_acum` suma picos sin desfase ni atenuación, solo sobre CONDUITS. Con divisores o ciclos contaría el caudal dos veces | Sin impacto actual: la red es un árbol acíclico con out-degree máximo 1. Riesgo al transferir a otras redes |
| H-13 | 🔵 | `runner.py:427-429`, `456-462`, `515-530`; `swmm_api_io.py:131-134` | En la ruta legacy (`swmm_resilience/main.py` y app de escritorio) se leen claves de PySWMM que no existen (`time_flooded`, `time_max_depth`), se asume un dt constante y `base_inflow_lps` es el Baseline (homónimo de la feature v17, que es el pico) | Duración de inundación 0 y `time_to_peak` None salvo por el overlay del `.rpt`. El dt real varía entre 1 y 5 s (5 valores distintos). No afecta al pipeline v17 de la tesis |
| H-14 | 🔵 | `hydrograph_csv.py:127-129`, `153-156` | `write_shape_validation_csv` redondea los tiempos a minutos enteros (H:MM) | Desplazamiento ≤ 0.5 min y volumen ≤ 1.22 % (`shape_ultra_short_impulse`). En las formas de validación ≤ 0.01 %. t_pico de entrenamiento e inferencia difieren ≤ 3.3·10⁻⁴ h |

### 2.2 Fichas detalladas

#### H-01 🟠 — Condición inicial (InitFlow) fija a factor 1 en todos los escenarios

- **Dónde:** `.inp` Qx1, `[CONDUITS]`, columna `InitFlow` (líneas 220–379). Las funciones `write_scaled_inp` (`swmm_api_io.py:229-263`) y `write_scenario_inp` (`timeseries_scenario.py:47-136`) solo reescriben `[TIMESERIES]` y `[OPTIONS]`, así que el InitFlow se hereda sin cambios. Qx2 y Qx3 tienen el mismo InitFlow que Qx1 (cociente 1.000).
- **Qué pasa:** los 160 conductos tienen InitFlow > 0, y el valor es **exactamente** la suma de los picos de las series aguas arriba del nodo de entrada (diferencia máxima 0.0 L/s; 1525 L/s en `T160`). Como los nodos tienen InitDepth = 0, SWMM inicializa sus profundidades a partir de las de los conductos. La red arranca "llena" con el caudal pico de factor 1 en todas las corridas, aunque el escenario sea f = 0.2 o una forma que empieza en q = 0 (33 de las 39 formas tienen q0 = 0).
- **Impacto cuantificado (d05, d07, d09):**
  - Initial Stored Volume = 855 m³ en todas las corridas. Es 5.9 veces el volumen de entrada de `shape_ultra_short_impulse` a f = 0.2 (146 m³).
  - Error de continuidad del routing con la configuración del pipeline: hasta **12.38 %** (ultra_short, f = 0.2). A f = 0.2, 12 de 40 formas superan 5 %; a f = 1.0, 3 de 40. La correlación entre CE y (almacenamiento inicial / entrada) es 0.857. Con InitFlow = 0, el CE queda entre −0.92 % y +0.03 % en todos los casos probados.
  - Etiquetas (pipeline frente a InitFlow = 0):

    | Escenario | Vol. inundado pipeline | Vol. con InitFlow = 0 | Δ | Nodos inundados (pipeline / InitFlow = 0) |
    |---|---|---|---|---|
    | `shape_ultra_short_impulse` f = 5 | 1320 m³ | 1195 m³ | +10.5 % | 96 / 101 |
    | `rare_extreme` f = 5 | 2099 | 2054 | +2.2 % | 98 / 100 |
    | `common_bogota` f = 5 | 2864 | 2825 | +1.4 % | 98 / 100 |
    | `shape_flash_flood_urban` f = 5 | 4162 | 4121 | +1.0 % | 95 / 99 |
    | `shape_micro_burst_45min` f = 5 | 4908 | 4875 | +0.7 % | 95 / 97 |
    | `shape_2h` f = 3 | 5033 | 5038 | −0.1 % | 60 / 62 |
    | `base` f = 3 / f = 5 | 7669 / 19940 | 7668 / 19941 | ≈ 0 | 61 / 61 · 93 / 93 |

  - Conclusión: el sesgo se concentra en los eventos cortos e intensos (≤ 1 h), justo en el régimen "flash" que la tesis quiere cubrir. En los eventos largos es despreciable.
- **Evidencia:** `d03_base_inp.py` (InitFlow = suma de picos, 160/160), `d05_experiments.py`, `d07_more.py`, `d09_continuity_sweep.py`. Corroboración externa: `validation_output/val_extrap_ultrafast_20min/metrics_per_scenario.csv` tiene 8 de 24 escenarios con CE > 5 % (máximo 11.37 %), y `AUDITORIA_SURROGATE_HIDROGRAMAS.md:309` ya lo observaba sin identificar la causa.
- **Recomendación:** generar los escenarios con InitFlow = 0, o con InitFlow = factor × q(t = 0) acumulado y coherente con la forma. Otra opción es un *spin-up* o hotstart en tiempo seco común a todas las corridas. Después hay que regenerar el dataset, o al menos las formas ≤ 1.5 h, y declarar el criterio en la tesis.

#### H-02 🟠 — Ninguna feature describe el volumen del hidrograma

- **Dónde:** `hydrograph_shapes.py:31-37` (`get_shape_stats` devuelve solo duración y t_pico), `dynamic_features.py:5-29`, `contracts.py:46-49`.
- **Qué pasa:** las 39 formas de entrenamiento y las 7 de validación están normalizadas a pico = 1 (`d02`). El factor escala el **pico**, y el volumen del evento por nodo (base·f·∫q_norm) depende de la forma: ∫q_norm·dt va de 0.133 h (`shape_ultra_short_impulse`) a 9.57 h (`shape_quasi_uniform_block`), es decir, 72 veces. En hidráulica de drenaje, el volumen desbordado depende sobre todo del volumen de entrada que supera la capacidad, pero el modelo solo ve (duración, t_pico).
- **Impacto cuantificado (`d12_db_shapes.py`, BD v17):**
  - Con duración 10 h: `shape_slow_broad_rise` (t_pico 6.0) da 9716 m³ y `shape_long_recession_tail` (t_pico 0.5) da 1366 m³ a f = 2, una relación de 7.1. A f = 5 la relación es 77 404 frente a 15 668 (4.9).
  - Con duración 2 h: `shape_2h` da 1593 m³ y `common_bogota` 120 m³ a f = 2 (13.3 veces).
  - Con 39 formas, el par (duración, t_pico) puede servir de "identificador" de la forma en entrenamiento, pero no determina el volumen de una forma nueva. Esto explica físicamente la subestimación de volumen en formas no vistas que reporta `AUDITORIA_SURROGATE_HIDROGRAMAS.md:309`.
- **Recomendación:** agregar features de volumen: `vol_evento_nodo = q_pico·∫q_norm` (m³), su acumulado aguas arriba, la duración sobre un umbral (por ejemplo q > 0.5·q_pico) y el centroide temporal. Deben calcularse igual en `compute_dynamic_features` y en `compute_scenario_dynamic_features`.

#### H-03 🟡 — Entrenamiento uniforme y sincrónico, inferencia sobre hidrogramas arbitrarios

- **Dónde:** `scenario_predict.py:113-124` (`peak_map` = max por nodo; `tiempo_al_pico_h` de la **primera** serie, l. 118-119); `dynamic_features.py:48-54` (suma de picos aguas arriba); `hydrograph_batch.py:54-61` (`expected_nodes` incluye los 13 nodos con serie nula); `scenario_predict.py:75-95` (bandera de extrapolación solo por pico/base); `main.py:704-726` (régimen interp/extrap solo por duración).
- **Qué pasa:** en las 1000 corridas, los 147 nodos con aporte reciben la **misma** forma y el **mismo** factor, con picos sincrónicos (`d01`: |q_pico_nodo/base − factor| ≤ 8.9·10⁻¹⁶). El modo `--evaluate-hydrographs` acepta CSV con series distintas por nodo, y ahí: (a) la suma de picos no coincidentes no es comparable con la del entrenamiento; (b) `tiempo_al_pico_h` depende del orden de los nodos en el CSV, y si el primero es uno de los 13 nodos nulos (`3C`, `4C`, `20C`, …) da 0 h (`d17`: 0.0 h frente a 1.0 h reales); (c) nada avisa que el escenario está fuera de la distribución de entrenamiento.
- **Impacto:** no sesga los resultados ya reportados, porque la validación fuera de muestra (`--evaluate-validation-shapes`) también usa formas uniformes. Sí limita lo que puede afirmarse: el surrogate no está validado para lluvia con variabilidad espacial.
- **Recomendación:** calcular t_pico y duración sobre el hidrograma **agregado** (suma de los nodos) tanto en entrenamiento como en inferencia, marcar como extrapolación cualquier CSV con formas heterogéneas o picos desfasados, y declarar la limitación en la tesis.

#### H-04 🟡 — Sin control de continuidad de masa en la generación del dataset

- **Dónde:** `batch.py:7-73` (run_batch y run_batch_shapes no leen el `.rpt` más allá de las etiquetas); `csv_backfill.py:147-156` (inserta `runs` sin `continuity_error_pct`). La validación sí lo hace (`hydrograph_batch.py:175-193`, `314-321`).
- **Evidencia:** `d01_db.py`: `runs` tiene 1000 corridas COMPLETE y `continuity_error_pct IS NULL` = 1000. `d09_continuity_sweep.py` (configuración exacta del pipeline, 80 corridas): a f = 0.2, 12 de 40 con |CE| > 5 %, 40 de 40 con |CE| > 1 % y mediana 3.67 %; a f = 1.0, 3 de 40 con > 5 % y mediana 0.93 %. Todas las corridas con CE > 5 % tienen 0 nodos inundados, así que **no contaminan positivos**, pero sí forman parte del 20 % de filas negativas (factor ≤ 1.0).
- **Recomendación:** leer y guardar el CE de routing (y el *Highest Continuity Errors* por nodo) para cada corrida, fijar un criterio de aceptación (por ejemplo |CE| ≤ 1 %) y reportarlo. La causa principal se corrige con H-01.

#### H-05 🟡 — El "drain-down" es una rampa lineal, no un corte a cero

- **Dónde:** `timeseries_scenario.py:120-123`. Se agrega el punto `(last_time + drain_down_hours, 0.0)`. SWMM interpola linealmente entre puntos, así que si q(t_fin) ≠ 0 el caudal baja **linealmente durante 6 h**. La docstring (l. 58-61) lo presenta como un punto de caudal cero "para que la red drene", y además el caudal no llega a 0 hasta exactamente END_TIME, de modo que esas formas no tienen ningún período real de drenaje.
- **Formas afectadas:** `shape_2h` (q_fin = 0.075), `shape_5h` (0.074) y `long_moderate` (0.35). Son 3 de 39 formas de entrenamiento y 0 de 7 formas de validación. También afecta a la forma `base` cuando se re-simula en `--evaluate-shapes` (`main.py:495-509`, q_fin/q_pico = 0.074).
- **Impacto cuantificado (d05, d07):** entrada externa de SWMM 6259 m³ frente a 5032 m³ con corte (`shape_2h`, f = 1, +24.4 %, igual a la predicción analítica de 0.5·q_fin·6 h / ∫ = 24.5 %); `long_moderate` +13.8 %; `shape_5h` +9.8 %. Volumen inundado: `long_moderate` f = 5 da 125 350 frente a 122 624 m³ (+2.2 %), f = 4 da 83 237 frente a 82 654 (+0.7 %), f = 3 y f = 1 sin cambio; `shape_2h` y `shape_5h` sin cambio a f = 1, 3, 4 y 5 (la cola queda por debajo de la capacidad). La feature `duracion_horas` (12 h en `long_moderate`) no incluye las 6 h adicionales de aporte.
- **Recomendación:** escribir `(t_fin + 1 min, 0)` y `(t_fin + dd, 0)`, o exigir q_fin = 0 en las formas. Si la cola es intencional, incluirla en `duracion_horas` y en el volumen del evento (H-02).

#### H-06 🟡 — Etiqueta cuantizada a 1 m³ y semántica real del umbral

- **Dónde:** `labels.py:19-23` (`vol >= threshold`), `122-139`; `swmm_api_io.py:304-307` (×1000 desde 10⁶ L); `config.yaml:14` (`flood_threshold_m3: 1.0`).
- **Qué pasa:** el *Node Flooding Summary* imprime el volumen en 10⁶ L con 3 decimales, así que la resolución es 0.001·10⁶ L = **1 m³**. Un desborde real menor de 0.5 m³ se reporta como 0.000 y uno de 0.5 a 1.5 m³ como 1 m³. El umbral nominal "≥ 1 m³" equivale entonces a "≥ 0.5 m³ de volumen continuo".
- **Evidencia:**
  - `d01`/`d12`: en la BD ningún volumen cae en (0, 1); el mínimo positivo es 1.0; 1902 filas valen exactamente 1.0 (4.1 % de los 46 242 positivos); solo 6 filas no son enteras.
  - `d11_quantization.py` (volumen continuo de PySWMM, `node.statistics['flooding_volume']` en m³, comparado con el `.rpt`): `base` f = 1.6 tiene 45 nodos que desbordan y 27 etiquetados (18 ocultos, máximo 0.347 m³); `base` f = 2.0, 63 frente a 39; `shape_2h` f = 2.0, 67 frente a 39; `rare_extreme` f = 2.0, 37 frente a 16. El error de cuantización llega a ±0.498 m³ (±50 % para un volumen de 1 m³).
  - El nodo `2I` en `base` f = 3: "Hours Flooded 0.01, Max Rate 11.67 LPS, Volume 0.000", es decir, desborda pero queda etiquetado como 0.
- **Recomendación:** tomar el volumen continuo (`flooding_volume` de PySWMM o la integración del `.out`), documentar el umbral como volumen continuo y hacer un análisis de sensibilidad del umbral (0, 0.5, 1, 5 m³) en las métricas. El regresor en log1p es especialmente sensible en el rango de 1–5 m³.

#### H-07 🟡 — Parser de texto de respaldo lee la columna equivocada

- **Dónde:** `labels.py:75-86`. La línea 79 toma `float(parts[-1])` como volumen. En SWMM 5.2 la tabla tiene siempre la columna final **"Maximum Ponded Depth (Meters)"**, también con `ALLOW_PONDING NO` (ver el encabezado en `d04`). La docstring supone que esa columna solo aparece con ponding.
- **Evidencia:** `d04_base_run.py` sobre `base` f = 3 da 7669 m³ (88 nodos) con swmm_api y 0.0 m³ con el parser de texto (diferencia máxima por nodo 602 m³).
- **Impacto:** hoy es latente, porque la ruta swmm_api funciona y las etiquetas de la BD coinciden con mis réplicas. Si swmm_api falla (cambio de versión o encoding), `read_flooding_volumes` cae al fallback y genera un dataset con todas las etiquetas en cero, sin error. La heurística de `swmm_api_io.py:293-298` (`"volume" in cl and (... "vol" in cl ...)`) también es frágil: cualquier columna que contenga "volume" se mapea al volumen.
- **Recomendación:** parsear por encabezado (columna "Total Flood Volume"), fallar en voz alta en vez de degradar, y cruzar Σ vol_nodos con "Flooding Loss" de la continuidad (7669 frente a 7673 m³ en el control; la diferencia viene del redondeo y de los nodos con menos de 0.5 m³).

#### H-08 🟡 — La capacidad hidráulica usada es la de aguas arriba, no la de salida

- **Dónde:** `topology.py:90-118` (`upstream_capacity_lps` = Σ de la capacidad Manning a tubo lleno de los conductos que **entran** al nodo).
- **Qué pasa:** la sobrecarga y el desborde de un junction dependen de que el caudal acumulado supere la capacidad del conducto de **salida**, además del remanso aguas abajo. La capacidad de entrada es un proxy débil y no existe en los nodos de cabecera.
- **Evidencia (`d16_capacity_index.py`, BD v17):** `upstream_capacity_lps` es NaN en 57 nodos (57 000 filas, 35.6 %), que luego se imputan con la mediana. AUC univariada frente a `inunda`: q_pico_acum_escalado da 0.753; q_pico_acum_escalado/upstream_capacity da 0.732 (solo en 103 000 filas); q_pico_acum_escalado/cap_salida da **0.884** (en 160 000 filas). La fórmula de Manning está bien implementada (§3, V-05).
- **Recomendación:** agregar `cap_salida_lps` y el índice `q_pico_acum_escalado / cap_salida`, calculados igual en entrenamiento e inferencia.

#### H-09 🔵 — `pendiente_out` NaN en el nodo que descarga al outfall

- **Dónde:** `static_features.py:82-87`. `compute_slope` busca las cotas solo en `junctions`; `109C` es un OUTFALL, así que la pendiente de `T160` queda `None` y `pendiente_out` de `108C` sale NaN aunque `n_tuberias_out = 1`.
- **Evidencia:** `d13_manning.py` muestra que el único nodo con `pendiente_out` NaN y salida es `['108C']`. Es 1 nodo (1000 filas). El mismo patrón está en `topology.py:99-108`, pero ahí no afecta porque un outfall nunca es predecesor.
- **Recomendación:** incluir `[OUTFALLS]` en el diccionario de cotas.

#### H-10 🔵 — Protocolo distinto para la forma `base`

- **Dónde:** `batch.py:7-16` y `runner.py:646-667` (END_TIME original de 3 h, sin drenaje y con las 160 series originales); las 39 formas restantes pasan por `write_scenario_inp` con +6 h. En `main.py:1002`, `duracion_horas` de `base` sale de la **primera** serie del `.inp` (`1C`).
- **Evidencia:** `d03`: todas las series terminan en 3 h con q/q_pico = 0.074–0.075, y la recesión no simulada (extrapolación exponencial) es el 6.8–8.3 % del volumen. `d05`: END = 9 h deja Flooding Loss y etiquetas **idénticos** a END = 3 h para f = 0.2, 1, 3 y 5; solo cambian el almacenamiento final (147–449 m³ frente a 0) y la salida. Las formas normalizadas de los 147 nodos coinciden con desviación ≤ 0.00075, así que tomar `1C` como representante es correcto.
- **Recomendación:** unificar el protocolo de simulación de las 40 formas por coherencia metodológica. No hace falta regenerar las etiquetas por esta causa.

#### H-11 🔵 — Supuestos de unidades y de `[INFLOWS]` sin comprobar en la ruta de entrenamiento

- **Dónde:** `labels.py:122-139` y `swmm_api_io.py:304-307` asumen 10⁶ **litros**, pero solo la validación comprueba `FLOW_UNITS = LPS` (`hydrograph_batch.py:102-107`). `static_features.py:7-26` toma el máximo de la serie sin aplicar Sfactor ni Baseline; `hydrograph_shapes.apply_shape` (40-53) reemplaza la serie pero deja Sfactor y Baseline en `[INFLOWS]`, con lo que se aplicarían dos veces. `csv_backfill.py:88` graba `flow_units='LPS'` fijo, sin leerlo del `.inp`.
- **Evidencia:** `d03` confirma `FLOW_UNITS LPS` y 0 inflows con Sfactor ≠ 1 o Baseline ≠ 0. No hay impacto actual.
- **Recomendación:** comprobar unidades e `[INFLOWS]` al inicio del pipeline de entrenamiento con las mismas reglas que la validación, y abortar si no se cumplen.

#### H-12 🔵 — Acumulación topológica de picos (riesgo al transferir a otras redes)

- **Dónde:** `topology.py:14-32` (grafo solo con CONDUITS), `83-88` y `dynamic_features.py:48-54` (Σ de picos de los ancestros).
- **Evidencia:** `d03` muestra un grafo acíclico con out-degree máximo 1, un único sumidero (`109C`) y ningún nodo aislado. En esta red, la suma es exacta como índice de aporte sincrónico (en entrenamiento todos los nodos tienen su pico a la misma hora) e ignora la atenuación y el desfase del tránsito.
- **Recomendación:** documentar que `q_pico_acum` es una cota superior sin tránsito. Si se transfiere a redes con divisores, bombas o vertederos, incluir todos los tipos de enlace y repartir el caudal en las bifurcaciones.

#### H-13 🔵 — Errores en la ruta legacy (`runner.run_simulation`)

- **Dónde:** `runner.py:515-517` (`stats.get("time_flooded")`) y `521-530` (`stats.get("time_max_depth")`): **ninguna** de las dos claves existe en PySWMM 2.1 (`d08` lista las claves reales: `flooding_duration`, `max_depth_date`, …). `runner.py:427-429` y `456-462` calculan la tasa de profundidad con el dt del primer paso, cuando el dt real varía (`d08`: 1–5 s, 5 valores distintos, 2205 pasos). `swmm_api_io.py:131-134` y `runner.py:325`: `base_inflow_lps` = Baseline de `[INFLOWS]` (0 en esta red), mientras que en v17 la columna homónima es el pico de la serie.
- **Impacto:** solo en `swmm_resilience/main.py` y en la app de escritorio. La duración de inundación queda en 0 para los nodos que no aparecen en el `.rpt` y `time_to_peak_min` siempre en None. Ninguna de estas columnas entra al modelo v17 (`ML_DROP_COLUMNS`). No verificado empíricamente: que el modo por defecto `steady` (`config.py:69`) aplicado a un `.inp` hidrográfico con Baseline = 0 aborte en `write_scaled_inp`; se deduce de leer `swmm_api_io.py:217-226` y `256-260`.
- **Recomendación:** corregir las claves (`flooding_duration`, `max_depth_date`), usar el dt real de cada paso y renombrar `base_inflow_lps` del legacy, o retirar esa ruta.

#### H-14 🔵 — Redondeo a minutos al convertir forma → CSV de validación

- **Dónde:** `hydrograph_csv.py:127-129` y `153-156`.
- **Evidencia (`d14_parity.py`):** el desplazamiento máximo es 0.5 min; ΔVol = −1.22 % (`shape_ultra_short_impulse`), −0.26 % (`shape_short_flat_top`) y ≤ 0.05 % en el resto. En las 7 formas de validación, ≤ 0.01 %.
- **Recomendación:** permitir H:MM:SS en el contrato del CSV, o validar que la forma ya esté en minutos enteros.

**Nota fuera de área:** `--evaluate-shapes` (`main.py:480-555`) re-simula las **mismas** formas y factores del entrenamiento con el modelo final, así que no es una validación fuera de muestra. Lo remito al anexo de ML; aquí solo constato el hecho.

---

## 3. Lo que sí está bien hecho (verificado)

| ID | Verificación | Evidencia |
|---|---|---|
| V-01 | **El escalado conserva el volumen en proporción y SWMM integra el hidrograma declarado.** La entrada externa del `.rpt` coincide con la integral trapezoidal: `base` f = 1 da 7508 m³ frente a 7510 analíticos (−0.03 %); `shape_2h` con corte, 5032 frente a 5027.8 (+0.08 %); ultra_short f = 1, 731 frente a 730.8. f = 3 da 22 525 = 3 × 7508.3. Esto confirma la interpolación lineal entre puntos y el caudal 0 después del último punto. | `d05` |
| V-02 | **Identidad del escalado.** Pipeline f = 1.0 ≡ `.inp` original (misma entrada, CE de 1.439 % en ambos, etiquetas idénticas). Los `.inp` Qx2 y Qx3 originales ≡ Qx1 × 2 y × 3 por el pipeline: 0 etiquetas distintas y \|ΔVol\| máximo por nodo de 1 m³. La geometría de Qx1, Qx2 y Qx3 es idéntica (ΔElev = ΔMaxDepth = ΔDiam = 0). | `d06`, `d15` |
| V-03 | **Reproducibilidad de las etiquetas de la BD.** Las réplicas independientes reproducen exactamente vol/n_inunda: `base` f = 3 (7669/61) y f = 5 (19 940/93), `long_moderate` f = 3 (44 540/61) y f = 5 (125 348/91), `shape_2h` f = 3 (5033/60), `shape_ultra_short_impulse` f = 5 (1320/96), `shape_micro_burst_45min` f = 5 (4908/95), `rare_extreme` f = 2 y 5, `common_bogota` f = 5 y `shape_flash_flood_urban` f = 5. | `d05`, `d07`, `d12`, `d18` |
| V-04 | **Unidades del volumen.** `.rpt` en 10⁶ L × 1000 = m³ es correcto en SI. La ruta swmm_api elige bien la columna (Σ = 7669 m³ frente a "Flooding Loss" 7673 m³; la diferencia es redondeo). PySWMM `flooding_volume` (95.67 m³) coincide con el `.rpt` (0.096·10⁶ L) para `1C`. | `d04`, `d08`, `d11` |
| V-05 | **Capacidad Manning a tubo lleno** (`utils.py:66-76`): Q = (1/n)·A·(D/4)^(2/3)·S^(1/2)·1000 [L/s] es correcta. Comparada con la capacidad implícita de SWMM (Max Flow / Max-Full Flow) en los 160 conductos, el cociente va de 0.972 a 1.037 (mediana 0.9999); la dispersión se debe al cociente de 2 decimales del `.rpt`. | `d13` |
| V-06 | **Pendientes y offsets.** No hay pendientes adversas ni nulas (0.00123–0.0923 m/m) ni longitudes cero (58.9–218.3 m). Los offsets (34 conductos) se suman correctamente a la cota de fondo con `LINK_OFFSETS DEPTH`. Δz/L frente al Δz/√(L²−Δz²) de SWMM difiere < 0.43 % para S ≤ 0.092. | `d03`, `d13` |
| V-07 | **Topología.** Árbol dirigido acíclico, 1 outfall, ningún nodo aislado, `dist_outfall_m` sin NaN; q_pico_acum_base del outfall = 1525 L/s = InitFlow de `T160`. | `d03`, `d13` |
| V-08 | **Paridad entre entrenamiento e inferencia de las features dinámicas.** BD frente a la ruta de inferencia (`write_shape_validation_csv` → `load_scenario` → `compute_scenario_dynamic_features`) en las 39 formas a f = 2: q_pico_nodo, q_pico_acum_escalado y duracion_horas son **idénticas**; tiempo_al_pico_h difiere ≤ 3.3·10⁻⁴ h. | `d14` |
| V-09 | **El truncamiento de la recesión en `base` (3 h) no altera las etiquetas** (END = 9 h da el mismo resultado a f = 0.2, 1, 3 y 5). | `d05` |
| V-10 | **Formas.** Las 39 de entrenamiento y las 7 de validación empiezan en t = 0 (lo impone `load_shape`) y tienen pico 1.000. Las 7 de validación terminan en q = 0, sin rampa (H-05 no las afecta). Existe la guarda de solapamiento entre entrenamiento y validación (`main.py:664-675`). | `d02` |
| V-11 | **Paso de cálculo.** DYNWAVE con ROUTING_STEP de 5 s, VARIABLE_STEP 0.75 y MINIMUM_STEP 0.5; el dt real es de 1–5 s. El *Node Flooding Summary* integra en cada paso de routing, no en el REPORT_STEP de 3 min. Los volúmenes de PySWMM (paso a paso) y del `.rpt` coinciden dentro de ±0.5 m³ (pura cuantización). | `d08`, `d11` |
| V-12 | **Umbral físico de la red.** Ninguna forma inunda a f ≤ 1.0 (200 corridas, 20 % del dataset); el inicio de la inundación está entre f = 1.2 (`base`, `common_storm`, `long_moderate`) y f = 1.6 (`shape_ultra_short_impulse`). Es coherente con una red dimensionada para el caudal de diseño Qx1. | `d18` |
| V-13 | **Validación de los CSV de escenario** (`hydrograph_csv.py:33-124`): exige columnas exactas, nodos exactos, formato H:MM, ausencia de duplicados, valores finitos ≥ 0, malla común y t0 = 0. Además, la correspondencia nodo↔serie es 1:1 (`timeseries_scenario.py:19-44`). Son reglas sólidas. | lectura |

---

## 4. Supuestos y limitaciones hidráulicas que la tesis debe declarar

1. **Modelo 1D de onda dinámica sin drenaje dual.** SWMM 5.2 (swmm-toolkit 0.17.0) con `DYNWAVE`, `ALLOW_PONDING NO`, SurDepth = 0 y Aponded = 0. El "volumen de inundación" es el **volumen desbordado por la tapa** del pozo (MaxDepth de 1.4–3.3 m). Ese volumen sale del sistema: no se acumula ni vuelve a entrar aguas abajo, y no hay escurrimiento superficial 2D. No representa lámina, extensión ni daño.
2. **Condición de borde aguas abajo:** un único outfall `109C` de tipo FREE, sin remanso del cuerpo receptor.
3. **Forzante:** no hay subcuencas ni lluvia. Los aportes son hidrogramas impuestos como inflow directo en 147 nodos (13 nodos sin aporte). El "factor" escala **caudales**, no precipitación, así que deja fuera la no linealidad lluvia-escorrentía.
4. **Escenarios uniformes y sincrónicos:** la misma forma normalizada y el mismo factor en todos los nodos (H-03). El modelo no está validado para tormentas con variabilidad espacial ni con desfase entre subcuencas.
5. **Normalización por pico:** el factor fija el pico, y el volumen del evento depende de la forma (H-02).
6. **Condición inicial:** si se mantiene el InitFlow actual, hay que declararlo como "red inicialmente en régimen de caudal de diseño (factor 1)" e informar su efecto en la continuidad (H-01).
7. **Definición de la etiqueta:** `inunda = 1` si el volumen del `.rpt`, cuantizado a 1 m³, es ≥ 1 m³, lo que equivale a ≥ 0.5 m³ continuo. Los desbordes menores cuentan como "no inunda" (H-06).
8. **Duración simulada:** la duración de la forma más 6 h para las 39 formas CSV, y 3 h exactas para `base`. Las formas que no terminan en cero reciben una rampa (H-05).
9. **Una sola red:** Chico Sur, PVC con n = 0.0082, 160 conductos circulares en un árbol sin estructuras especiales (sin bombas, vertederos, orificios, almacenamientos ni divisores). No se ha demostrado que el modelo sea transferible a otras redes (H-12).
10. **Rango de factores:** 0.2–5.0 con paso 0.2. El 20 % de las corridas (f ≤ 1.0) no inunda en ningún nodo.
11. **Índice de "resiliencia"** (`analysis/resilience.py:32`): 1 − fracción de nodos inundados. Es un conteo binario que no pondera magnitud, duración ni recuperación, así que no corresponde a los índices de resiliencia hidráulica de la literatura (severidad × duración). Debe nombrarse con precisión.
12. **Capacidad de los conductos:** se usa la capacidad nominal de Manning a tubo lleno, que no considera presurización, remanso ni pérdidas locales.
13. **Calidad numérica:** con la configuración actual, el error de continuidad llega a 12.4 % en eventos cortos y de baja intensidad (H-01, H-04). Hay que declarar el criterio de aceptación de las corridas.

---

## 5. Scripts de diagnóstico reproducibles

Todos los scripts son de **solo lectura** sobre el repositorio y la BD (`mode=ro`). Las simulaciones se hacen sobre copias en `./runs/` del directorio de trabajo. Se ejecutan desde un mismo directorio con el Python del proyecto:

```
<repo>\venv\Scripts\python.exe dNN_*.py "<ruta-al-repo>" [args]
```

`d07`, `d09`, `d11` y `d15` reutilizan las funciones auxiliares de `d05_experiments.py`, que debe estar en el mismo directorio. Hay que ejecutarlo antes, porque crea `runs/exp`. El tiempo total es de unos 6 minutos (alrededor de 150 corridas SWMM de 1–3 s).

| Script | Argumentos | Qué demuestra |
|---|---|---|
| `d01_db.py` | ruta a la BD | Conteos, NULL de continuidad, cuantización de etiquetas, uniformidad q_pico/base |
| `d02_shapes.py` | carpeta de formas | Normalización por pico, q_fin, ∫q_norm, rampa de drenaje |
| `d03_base_inp.py` | `.inp` | OPTIONS, InitFlow = Σ picos, recesión truncada, pendientes, topología |
| `d04_base_run.py` | repo, factor | Formato del *Node Flooding Summary*, continuidad, parser de texto frente a swmm_api |
| `d05_experiments.py` | repo | InitFlow = 0, END extendido, rampa frente a corte |
| `d06_qx_compare.py` | repo | Qx2/Qx3 con la misma geometría e InitFlow que Qx1 (termina con KeyError al llegar a Steady, que es otra red; los resultados de Qx2/Qx3 se imprimen antes) |
| `d07_more.py` | repo | Rampa a f = 4–5 e InitFlow en formas cortas |
| `d08_pyswmm_keys.py` | `.inp` de control | Claves reales de PySWMM y dt variable |
| `d09_continuity_sweep.py` | repo | Barrido de CE en 80 corridas |
| `d10_validation_csvs.py` | repo | Formato de las formas de validación |
| `d11_quantization.py` | repo | Volumen del `.rpt` frente al de PySWMM (cuantización) |
| `d12_db_shapes.py` | ruta a la BD | Volumen por forma con igual duración; filas con vol = 1.0 |
| `d13_manning.py` | repo, `.rpt` | Capacidad Manning frente a SWMM; pendientes; NaN estáticos |
| `d14_parity.py` | repo | Paridad de features entre entrenamiento e inferencia; redondeo a minutos |
| `d15_identity.py` | repo | Pipeline f = 1 ≡ original; Qx2/Qx3 ≡ Qx1 × 2/× 3 |
| `d16_capacity_index.py` | repo | AUC de capacidad aguas arriba frente a capacidad de salida |
| `d17_tpico_inferencia.py` | repo | t_pico de inferencia tomado de la primera serie |
| `d18_db_checks.py` | ruta a la BD | Totales de la BD que se cruzan con las re-simulaciones (V-03); umbral físico (V-12) |

Código completo:


### `d01_db.py`

```python
import sqlite3, sys
db = sys.argv[1]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
q = lambda s: c.execute(s).fetchall()
print("networks:", q("select network_id,name,source_filename,flow_units,length(inp_bytes) from networks"))
print("n scenarios:", q("select count(*) from scenarios"), "runs:", q("select status,count(*),sum(continuity_error_pct is null) from runs group by status"))
print("shapes:", q("select count(distinct shape_id) from scenarios"), "factors:", q("select count(distinct factor_mult), min(factor_mult), max(factor_mult) from scenarios"))
print("dur/tpico por shape (base):", q("select shape_id, duracion_horas, tiempo_al_pico_h from scenarios where shape_id='base' limit 3"))
print("dur/tpico distintos:", q("select shape_id, min(duracion_horas),min(tiempo_al_pico_h) from scenarios group by shape_id order by 2"))
print("nodos:", q("select count(*), sum(base_inflow_lps>0) from nodes"))
print("filas training:", q("select count(*), sum(inunda), avg(inunda) from training_samples_v17"))
print("vol en (0,1):", q("select count(*) from training_samples_v17 where vol_inundacion_m3>0 and vol_inundacion_m3<1"))
print("vol en [1,2):", q("select count(*) from training_samples_v17 where vol_inundacion_m3>=1 and vol_inundacion_m3<2"))
print("vol==0 & inunda=1:", q("select count(*) from training_samples_v17 where vol_inundacion_m3<1 and inunda=1"))
print("min vol positivo:", q("select min(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0"))
print("cuantiles vol>0 (n, sum):", q("select count(*), sum(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0"))
print("vol perdida por umbral (suma vol 0<v<1):", q("select sum(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0 and vol_inundacion_m3<1"))
print("ratio q_pico_nodo/base vs factor (max abs dev):", q("select max(abs(q_pico_nodo/base_inflow_lps - factor_mult)) from training_samples_v17 where base_inflow_lps>0"))
print("node_timeseries rows:", q("select count(*) from node_timeseries"), "scenario_inflows:", q("select count(*) from scenario_inflows"))
```

### `d02_shapes.py`

```python
import csv, glob, os, sys
import numpy as np
root = sys.argv[1]
print(f"{'shape':40s} {'n':>4s} {'dt_min':>12s} {'dur_h':>6s} {'max':>6s} {'q0':>6s} {'qlast':>6s} {'int_qn_h':>8s} {'dd_ramp_h':>9s} {'sub_min':>7s}")
for p in sorted(glob.glob(os.path.join(root, "*.csv"))):
    rows = list(csv.DictReader(open(p)))
    t = np.array([float(r["time_h"]) for r in rows]); q = np.array([float(r["q_norm"]) for r in rows])
    dt = np.diff(t)*60
    integ = np.trapz(q, t)
    # extra volume from drain-down ramp (last -> 0 over 6 h), in units of peak*hour
    ramp = 0.5*q[-1]*6.0
    submin = np.any(np.abs(t*60 - np.round(t*60)) > 1e-6)
    print(f"{os.path.basename(p)[:-4]:40s} {len(t):4d} {dt.min():5.2f}-{dt.max():5.2f} {t[-1]:6.2f} {q.max():6.3f} {q[0]:6.3f} {q[-1]:6.3f} {integ:8.3f} {ramp:9.3f} {str(submin):>7s}")
```

### `d03_base_inp.py`

```python
"""Analiza las series embebidas y la geometria del .inp base (forma 'base'). Solo lectura."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from swmm_api import read_inp_file
inp = read_inp_file(sys.argv[1], encoding="utf-8")
o = inp["OPTIONS"]
print("OPTIONS:", {k: o.get(k) for k in ["FLOW_UNITS","FLOW_ROUTING","ALLOW_PONDING","START_TIME","END_TIME","REPORT_STEP","ROUTING_STEP","VARIABLE_STEP","MIN_SLOPE","LINK_OFFSETS","SKIP_STEADY_STATE","NORMAL_FLOW_LIMITED","MINIMUM_STEP"]})
print("n junctions", len(inp["JUNCTIONS"]), "outfalls", list(inp["OUTFALLS"].keys()), "conduits", len(inp["CONDUITS"]))
print("tipo de tiempo en TIMESERIES:", type(next(iter(inp["TIMESERIES"].values())).data[0][0]))
mf = {k: (v.scale_factor, v.base_value) for k, v in inp["INFLOWS"].items()}
print("INFLOWS con Mfactor!=1 o Baseline!=0:", [k for k,(m,b) in mf.items() if (m not in (1,1.0,None)) or (b not in (0,0.0,None))])
P={}; R=[]; TP=[]; VP=[]; TR=[]; SH=[]
for name, ts in inp["TIMESERIES"].items():
    t = np.array([float(a) for a,_ in ts.data]); q = np.array([float(b) for _,b in ts.data])
    P[name]=q.max()
    if q.max() <= 0: continue
    R.append(q[-1]/q.max()); TP.append(t[q.argmax()]); VP.append(np.trapz(q,t)/q.max()); SH.append(q/q.max())
    k = np.log(q[-2]/q[-1])/(t[-1]-t[-2]); TR.append((q[-1]/k)/np.trapz(q,t))
R,TP,VP,TR=map(np.array,(R,TP,VP,TR))
pk=np.array(list(P.values()))
print("n series", len(pk), "con pico 0:", int((pk<=0).sum()), " pico>0 [min,med,max] L/s:", pk[pk>0].min(), np.median(pk[pk>0]), pk.max(), " suma picos:", pk.sum())
print("q_final(3h)/q_pico: min %.3f med %.3f max %.3f" % (R.min(), np.median(R), R.max()))
print("t_pico (h) unique:", np.unique(np.round(TP,3)))
print("integral/pico (h): min %.3f med %.3f max %.3f" % (VP.min(), np.median(VP), VP.max()))
S=np.vstack(SH); print("max desviacion entre formas normalizadas de distintos nodos:", float(np.abs(S-S[0]).max()))
print("recesion no simulada tras 3 h (extrap. exponencial) / volumen simulado: min %.3f med %.3f max %.3f" % (TR.min(), np.median(TR), TR.max()))
init = {lid: float(c.flow_initial or 0) for lid,c in inp["CONDUITS"].items()}
print("conduits con InitFlow>0:", sum(v>0 for v in init.values()), "de", len(init), " max InitFlow:", max(init.values()))
lens = [float(c.length) for c in inp["CONDUITS"].values()]
print("longitud min/max:", min(lens), max(lens))
print("conduits con offsets != 0:", sum(1 for c in inp["CONDUITS"].values() if (c.offset_upstream or 0) or (c.offset_downstream or 0)))
print("formas XSECTIONS:", {x.shape for x in inp["XSECTIONS"].values()}, " barrels!=1:", sum(1 for x in inp["XSECTIONS"].values() if (x.n_barrels or 1)!=1))
print("rugosidades:", {float(c.roughness) for c in inp["CONDUITS"].values()})
J = inp["JUNCTIONS"]
print("SurDepth>0:", sum(1 for j in J.values() if (j.depth_surcharge or 0)>0), " Aponded>0:", sum(1 for j in J.values() if (j.area_ponded or 0)>0), " InitDepth>0:", sum(1 for j in J.values() if (j.depth_init or 0)>0))
print("MaxDepth range:", min(j.depth_max for j in J.values()), max(j.depth_max for j in J.values()))
for sec in ["STORAGE","DIVIDERS","PUMPS","ORIFICES","WEIRS","OUTLETS","SUBCATCHMENTS","DWF","RDII","LOSSES","CONTROLS","CURVES"]:
    if sec in inp: print("seccion presente:", sec, len(inp[sec]))
# InitFlow vs suma de picos aguas arriba
import networkx as nx
G=nx.DiGraph()
for lid,c in inp["CONDUITS"].items(): G.add_edge(str(c.from_node),str(c.to_node),lid=lid)
dev=[]
for lid,c in inp["CONDUITS"].items():
    n=str(c.from_node); acc=sum(P.get(a,0) for a in nx.ancestors(G,n)|{n})
    dev.append(abs(acc-init[lid]))
print("max |InitFlow - suma picos aguas arriba del nodo de entrada| (L/s):", max(dev))
print("grafo: aciclico:", nx.is_directed_acyclic_graph(G), " max out_degree:", max(d for _,d in G.out_degree()), " nodos con out_degree 0:", [n for n,d in G.out_degree() if d==0])
print("nodos sin conductos:", [n for n in J if str(n) not in G])
```

### `d04_base_run.py`

```python
"""E1: corre la forma 'base' exactamente como run_batch (run_simulation_simple) sobre una COPIA
en el scratchpad, y compara parsers de volumen de inundacion. Uso: python d04_base_run.py <repo> <factor>"""
import sys, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]); factor = float(sys.argv[2])
sys.path.insert(0, str(repo))
from swmm_resilience.simulation.runner import run_simulation_simple
from swmm_resilience.extraction.labels import _parse_node_flooding_text, read_flooding_volumes
inp = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
rd = Path("runs") / f"base_f{factor:.2f}"
rpt = run_simulation_simple(inp, factor, rd)
txt = rpt.read_text(errors="replace")
i = txt.find("Node Flooding Summary"); print(txt[i-10:i+900] if i>=0 else "SIN SECCION DE INUNDACION")
for key in ["Flow Routing Continuity", "Continuity Error"]:
    j = txt.find(key); print(txt[j:j+1100] if key=="Flow Routing Continuity" else "")
a = read_flooding_volumes(rpt); b = _parse_node_flooding_text(rpt)
m = a.merge(b, on="node_id", suffixes=("_swmmapi","_texto"), how="outer")
print("nodos inundados:", len(a), " suma vol swmm_api:", a.flooding_volume_m3.sum(), " suma vol parser texto:", b.flooding_volume_m3.sum())
print("max |dif| parsers:", (m.flooding_volume_m3_swmmapi - m.flooding_volume_m3_texto).abs().max())
```

### `d05_experiments.py`

```python
"""Experimentos de control (copias en scratchpad, nunca toca el repo).
Construye los .inp con las MISMAS funciones del pipeline (write_scaled_inp / apply_shape +
write_scenario_inp) y luego aplica variantes (InitFlow=0, END_TIME extendido, rampa vs corte a 0).
Uso: python d05_experiments.py <repo>"""
import sys, re, warnings, datetime, shutil
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
import pandas as pd
from pyswmm import Simulation
from swmm_api import read_inp_file
from swmm_resilience.simulation.swmm_api_io import write_scaled_inp
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels

INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
SHAPES = repo / "data/hydrograph_shapes"
OUT = Path("runs/exp"); OUT.mkdir(parents=True, exist_ok=True)
static = extract_static_features(INP)
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
nodes = static.node_id.astype(str).tolist()

def rpt_metrics(rpt):
    t = Path(rpt).read_text(errors="replace")
    g = lambda lab: float(re.search(lab + r" \.+\s+(-?[\d.]+)\s+(-?[\d.]+)", t).group(2)) * 1000  # 10^6 L -> m3
    ce = re.findall(r"Continuity Error \(%\) \.+\s+(-?[\d.]+)", t)
    lab = extract_labels(Path(rpt), nodes, 1.0)
    return dict(ext_in_m3=g("External Inflow"), out_m3=g("External Outflow"), flood_loss_m3=g("Flooding Loss"),
                init_stor_m3=g("Initial Stored Volume"), final_stor_m3=g("Final Stored Volume"),
                cont_err_pct=float(ce[-1]), n_inunda=int(lab.inunda.sum()), vol_labels_m3=float(lab.vol_inundacion_m3.sum()))

def run(inp_path):
    with Simulation(str(inp_path)) as sim:
        for _ in sim: pass
    return rpt_metrics(Path(inp_path).with_suffix(".rpt"))

def variant(src, dst, init_zero=False, end_hours=None, cut_after=None):
    inp = read_inp_file(str(src), encoding="utf-8")
    if init_zero:
        for c in inp["CONDUITS"].values(): c.flow_initial = 0.0
    if cut_after is not None:  # reemplaza el punto de drenaje (rampa) por un corte a 0 un minuto despues
        for ts in inp["TIMESERIES"].values():
            d = list(ts.data)
            if len(d) >= 2 and d[-1][1] == 0.0 and d[-2][0] == cut_after:
                d[-1] = (cut_after + 1/60, 0.0); ts.data = d
    if end_hours is not None:
        o = inp["OPTIONS"]; st = datetime.datetime.combine(o["START_DATE"], o["START_TIME"]); e = st + datetime.timedelta(hours=end_hours)
        o["END_DATE"] = e.date(); o["END_TIME"] = e.strftime("%H:%M:%S")
    inp.write_file(str(dst)); return dst

rows = []
def rec(name, inp_path): 
    m = run(inp_path); m["caso"] = name; rows.append(m); print(name, m, flush=True)

# --- Forma base (run_batch): InitFlow y truncamiento a 3 h
for f in (0.2, 1.0, 3.0, 5.0):
    p = OUT / f"base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    rec(f"base f={f} (pipeline)", p)
    rec(f"base f={f} InitFlow=0", variant(p, OUT / f"base_f{f}_i0.inp", init_zero=True))
    rec(f"base f={f} END=9h", variant(p, OUT / f"base_f{f}_e9.inp", end_hours=9.0))
    rec(f"base f={f} InitFlow=0 END=9h", variant(p, OUT / f"base_f{f}_i0e9.inp", init_zero=True, end_hours=9.0))

# --- Formas con drenaje: rampa lineal del ultimo valor a 0 en 6 h vs corte a 0
for shape_id, f in (("shape_2h", 1.0), ("shape_2h", 3.0), ("long_moderate", 1.0), ("long_moderate", 3.0), ("shape_ultra_short_impulse", 1.0), ("shape_ultra_short_impulse", 5.0)):
    shape = load_shape(SHAPES / f"{shape_id}.csv")
    sc = HydrographScenario(f"{shape_id}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
    p = write_scenario_inp(INP, sc, OUT / "shapes", drain_down_hours=6.0)
    rec(f"{shape_id} f={f} (pipeline)", p)
    rec(f"{shape_id} f={f} InitFlow=0", variant(p, OUT / f"{shape_id}_f{f}_i0.inp", init_zero=True))
    if shape[-1][1] > 0:
        rec(f"{shape_id} f={f} corte a 0 (sin rampa)", variant(p, OUT / f"{shape_id}_f{f}_cut.inp", cut_after=shape[-1][0]))
    # volumen nominal del hidrograma (trapecio) sin y con rampa
    import numpy as np
    vn = sum(np.trapz([v for _,v in s], [t for t,_ in s]) for s in sc.node_series.values()) * 3.6  # L/s*h -> m3
    vr = sum(0.5 * s[-1][1] * 6.0 for s in sc.node_series.values()) * 3.6
    print(f"   volumen nominal {vn:.1f} m3; volumen anadido por la rampa de drenaje {vr:.1f} m3 ({100*vr/vn:.1f} %)")

pd.DataFrame(rows).to_csv("d05_results.csv", index=False)
```

### `d06_qx_compare.py`

```python
"""Compara Qx2/Qx3/Steady contra Qx1 (geometria, InitFlow, series) y corre Qx2 vs Qx1*2 (pipeline)."""
import sys, warnings, re
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file
N = repo / "data/networks"
q1 = read_inp_file(str(N/"chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"), encoding="utf-8")
for name, rel, k in [("Qx2","chico_hydro-qx2/SWMM - Chico (PVC) Prueba 5 - Qx2.00.inp",2),("Qx3","chico_hydro-qx3/SWMM - Chico (PVC) Prueba 5 - Qx3.00.inp",3),("Steady","chico_steady/SWMM - Chico (PVC) Prueba 1 - Steady.inp",None)]:
    q = read_inp_file(str(N/rel), encoding="utf-8")
    dz = max(abs(q["JUNCTIONS"][j].elevation - q1["JUNCTIONS"][j].elevation) for j in q1["JUNCTIONS"])
    dd = max(abs(q["JUNCTIONS"][j].depth_max - q1["JUNCTIONS"][j].depth_max) for j in q1["JUNCTIONS"])
    dg = max(abs(q["XSECTIONS"][l].height - q1["XSECTIONS"][l].height) for l in q1["XSECTIONS"])
    ri = [q["CONDUITS"][l].flow_initial / q1["CONDUITS"][l].flow_initial for l in q1["CONDUITS"] if q1["CONDUITS"][l].flow_initial]
    print(f"{name}: max|dElev|={dz} max|dMaxDepth|={dd} max|dDiam|={dg} InitFlow ratio vs Qx1: {min(ri):.3f}-{max(ri):.3f} END={q['OPTIONS'].get('END_TIME')}")
    if "TIMESERIES" in q and k:
        r = []
        for ts in q1["TIMESERIES"]:
            a = np.array([v for _,v in q1["TIMESERIES"][ts].data]); b = np.array([v for _,v in q["TIMESERIES"][ts].data])
            m = a>0; r += list(b[m]/a[m])
        print(f"   serie Qx/Qx1: min {min(r):.4f} max {max(r):.4f}")
    if "INFLOWS" in q and not k:
        print("   Steady INFLOWS baseline sum:", sum((v.base_value or 0) for v in q["INFLOWS"].values()), " timeseries:", "TIMESERIES" in q)
```

### `d07_more.py`

```python
"""E-rampa a factor alto y E-InitFlow en formas cortas (copias, scratchpad). Reusa helpers de d05."""
import sys, time
sys.argv = [sys.argv[0], sys.argv[1]]
src = open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)
rows = []
def rec(name, p):
    t0 = time.perf_counter(); m = run(p); m["caso"] = name; m["t_s"] = round(time.perf_counter()-t0, 1); rows.append(m)
    print(name, {k: m[k] for k in ("ext_in_m3","flood_loss_m3","vol_labels_m3","n_inunda","cont_err_pct","t_s")}, flush=True)
def build(shape_id, f):
    shape = load_shape(SHAPES / f"{shape_id}.csv")
    sc = HydrographScenario(f"{shape_id}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
    return shape, write_scenario_inp(INP, sc, OUT / "shapes2", drain_down_hours=6.0)
for shape_id in ("long_moderate", "shape_2h", "shape_5h"):
    for f in (4.0, 5.0):
        shape, p = build(shape_id, f)
        rec(f"{shape_id} f={f} rampa (pipeline)", p)
        rec(f"{shape_id} f={f} corte a 0", variant(p, OUT / f"{shape_id}_f{f}_cut.inp", cut_after=shape[-1][0]))
for shape_id in ("shape_micro_burst_45min", "rare_extreme", "shape_flash_flood_urban", "common_bogota"):
    for f in (2.0, 5.0):
        shape, p = build(shape_id, f)
        rec(f"{shape_id} f={f} (pipeline)", p)
        rec(f"{shape_id} f={f} InitFlow=0", variant(p, OUT / f"{shape_id}_f{f}_i0.inp", init_zero=True))
pd.DataFrame(rows).to_csv("d07_results.csv", index=False)
```

### `d08_pyswmm_keys.py`

```python
"""Verifica claves de estadisticas PySWMM usadas por runner.run_simulation (ruta legacy) y el paso de tiempo."""
import sys, warnings
warnings.filterwarnings("ignore")
from pyswmm import Simulation, Nodes, Links
p = sys.argv[1]
dts = []; prev = None
with Simulation(p) as sim:
    st = sim.start_time
    for _ in sim:
        t = (sim.current_time - st).total_seconds()
        if prev is not None: dts.append(t - prev)
        prev = t
    n = Nodes(sim)["1C"]; l = list(Links(sim))[0]
    print("node.statistics keys:", sorted(n.statistics.keys()))
    print("  'time_flooded' presente:", "time_flooded" in n.statistics, " 'time_max_depth' presente:", "time_max_depth" in n.statistics)
    print("  flooding_volume (pyswmm) 1C:", n.statistics.get("flooding_volume"), " flooding_duration:", n.statistics.get("flooding_duration"))
    print("conduit_statistics keys:", sorted(l.conduit_statistics.keys()))
import numpy as np
dts = np.array(dts); print("pasos de rutina: n=%d dt min=%.2f s, mediana=%.2f s, max=%.2f s, distintos=%d" % (len(dts), dts.min(), np.median(dts), dts.max(), len(np.unique(np.round(dts,3)))))
```

### `d09_continuity_sweep.py`

```python
"""Barrido de control del error de continuidad (configuracion EXACTA del pipeline) para las 41 formas
(base + 40 CSV) a factores 0.2 y 1.0 (82 corridas, ~1.5 s c/u). Copias en scratchpad."""
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
import glob, os
rows = []
for f in (0.2, 1.0):
    p = OUT / f"sweep_base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    m = run(p); m.update(shape="base", factor=f); rows.append(m)
    for csvp in sorted(glob.glob(str(SHAPES / "*.csv"))):
        sid = os.path.basename(csvp)[:-4]; shape = load_shape(Path(csvp))
        sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
        p = write_scenario_inp(INP, sc, OUT / "sweep", drain_down_hours=6.0)
        m = run(p); m.update(shape=sid, factor=f); rows.append(m)
df = pd.DataFrame(rows); df.to_csv("d09_continuity.csv", index=False)
df["ratio_init_ext"] = df.init_stor_m3 / df.ext_in_m3
for f, g in df.groupby("factor"):
    print(f"factor {f}: n={len(g)}  |CE|>5%: {(g.cont_err_pct.abs()>5).sum()}  |CE|>1%: {(g.cont_err_pct.abs()>1).sum()}  CE max={g.cont_err_pct.max():.2f}%  mediana={g.cont_err_pct.median():.2f}%")
print(df.sort_values("cont_err_pct", ascending=False)[["shape","factor","ext_in_m3","init_stor_m3","ratio_init_ext","cont_err_pct","n_inunda"]].head(12).to_string(index=False))
print("corr(CE, init/ext):", df[["cont_err_pct","ratio_init_ext"]].corr().iloc[0,1].round(3))
```

### `d10_validation_csvs.py`

```python
"""Caracteriza los CSV de validacion out-of-sample: forma por nodo, pico/base, tiempo al pico por nodo, fin != 0."""
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.validation.hydrograph_csv import load_scenario
inp = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
st = extract_static_features(inp); base = dict(zip(st.node_id.astype(str), st.base_inflow_lps))
exp = {n for n,v in base.items() if v > 0}
for p in sorted(glob.glob(str(repo / "data/hydrograph_validation/*.csv"))):
    try:
        sc = load_scenario(Path(p), exp)
    except Exception as e:
        df = pd.read_csv(p); print(os.path.basename(p), "NO CARGA con expected_nodes(base>0):", str(e)[:120], "cols:", list(df.columns)[:5]); continue
    ratios = []; tps = []; ends = []; forms = []; vols = []
    for n, s in sc.node_series.items():
        t = np.array([a for a,_ in s]); q = np.array([b for _,b in s])
        ratios.append(q.max()/base[n]); tps.append(t[q.argmax()]); ends.append(q[-1]/q.max()); forms.append(q/q.max()); vols.append(np.trapz(q,t)/q.max())
    F = np.vstack(forms); first = next(iter(sc.node_series))
    print(f"{sc.scenario_id:28s} dur={sc.last_time_hours:5.2f}h dt={60*np.diff(sc.time_grid_hours).min():4.1f}min pico/base=[{min(ratios):.2f},{max(ratios):.2f}] "
          f"t_pico nodos=[{min(tps):.2f},{max(tps):.2f}] (feature usa nodo '{first}': {tps[0]:.2f}) q_fin/pico max={max(ends):.3f} "
          f"desv. forma entre nodos={np.abs(F-F[0]).max():.3f} integral/pico=[{min(vols):.2f},{max(vols):.2f}]")
```

### `d11_quantization.py`

```python
"""Compara el volumen de inundacion del .rpt (3 decimales en 10^6 L => resolucion 1 m3) con el
volumen continuo de PySWMM (node.statistics['flooding_volume'], m3) en corridas de control."""
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
import numpy as np
from pyswmm import Nodes
def both(inp_path):
    prec = {}
    with Simulation(str(inp_path)) as sim:
        for _ in sim: pass
        for n in Nodes(sim):
            if n.is_junction(): prec[n.nodeid] = n.statistics["flooding_volume"]
    lab = extract_labels(Path(inp_path).with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    d = lab.join(pd.Series(prec, name="vol_pyswmm"))
    return d
tot = []
cases = [("base", 1.6), ("base", 2.0), ("shape_2h", 1.6), ("shape_2h", 2.0), ("rare_extreme", 2.0), ("common_storm", 1.6)]
for sid, f in cases:
    if sid == "base":
        p = OUT / f"q_base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    else:
        shape = load_shape(SHAPES / f"{sid}.csv")
        sc = HydrographScenario(f"q_{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
        p = write_scenario_inp(INP, sc, OUT / "quant", drain_down_hours=6.0)
    d = both(p)
    flooded_hidden = d[(d.vol_pyswmm > 0) & (d.inunda == 0)]
    err = (d.vol_inundacion_m3 - d.vol_pyswmm)[d.vol_pyswmm > 0]
    small = d[(d.vol_pyswmm > 0) & (d.vol_pyswmm < 5)]
    print(f"{sid} f={f}: nodos con inundacion>0 (pyswmm)={int((d.vol_pyswmm>0).sum())}, inunda=1 (rpt>=1 m3)={int(d.inunda.sum())}, "
          f"inundan pero etiqueta 0={len(flooded_hidden)} (vol max oculto {flooded_hidden.vol_pyswmm.max() if len(flooded_hidden) else 0:.3f} m3); "
          f"|error cuantizacion| max={err.abs().max():.3f} m3; error relativo max en nodos 0<v<5 m3: {((small.vol_inundacion_m3-small.vol_pyswmm).abs()/small.vol_pyswmm).max() if len(small) else float('nan'):.2f}")
```

### `d12_db_shapes.py`

```python
"""Lectura (mode=ro) de training_v17: volumen total por forma/factor; formas con igual duracion."""
import sqlite3, sys
import pandas as pd
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
df = pd.read_sql("select shape_id, factor_mult, duracion_horas d, tiempo_al_pico_h tp, sum(vol_inundacion_m3) vol, sum(inunda) n from training_samples_v17 group by shape_id, factor_mult", c)
print("filas con vol exactamente 1.0:", c.execute("select count(*) from training_samples_v17 where vol_inundacion_m3=1.0").fetchone()[0])
print("vol no entero:", c.execute("select count(*) from training_samples_v17 where vol_inundacion_m3 != cast(vol_inundacion_m3 as integer)").fetchone()[0])
for f in (2.0, 3.0, 5.0):
    sub = df[(df.factor_mult - f).abs() < 1e-6].set_index("shape_id")
    for a, b in (("shape_slow_broad_rise", "shape_long_recession_tail"), ("new_hydrograph", "shape_symmetric_moderate"), ("rare_extreme", "shape_flash_flood_urban"), ("shape_2h","common_bogota")):
        print(f"f={f}: {a} (d={sub.loc[a,'d']}, tp={sub.loc[a,'tp']}) vol={sub.loc[a,'vol']:.0f} n={sub.loc[a,'n']} | {b} (d={sub.loc[b,'d']}, tp={sub.loc[b,'tp']}) vol={sub.loc[b,'vol']:.0f} n={sub.loc[b,'n']}")
# mismo factor 1.0: hay inundacion?
print(df[(df.factor_mult-1.0).abs()<1e-6].query("n>0")[["shape_id","vol","n"]].to_string(index=False))
```

### `d13_manning.py`

```python
"""Compara capacidad Manning a tubo lleno de utils.circular_full_flow_lps (via topology/static) con la
capacidad implicita de SWMM (Max Flow / (Max/Full Flow)) del Link Flow Summary; revisa pendientes."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file, read_rpt_file
from swmm_resilience.utils import circular_full_flow_lps
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
inp = read_inp_file(str(INP), encoding="utf-8")
rpt = read_rpt_file(sys.argv[2])
lfs = rpt.link_flow_summary
print("columnas link_flow_summary:", list(lfs.columns))
J = inp["JUNCTIONS"]; O = inp["OUTFALLS"]
def elev(n): return (J[n].elevation if n in J else O[n].elevation)
rows = []
for lid, c in inp["CONDUITS"].items():
    z1 = elev(c.from_node) + (c.offset_upstream or 0); z2 = elev(c.to_node) + (c.offset_downstream or 0)
    s = (z1 - z2) / c.length
    cap = circular_full_flow_lps(inp["XSECTIONS"][lid].height, abs(s) if s else None, c.roughness)
    r = lfs.loc[lid]
    qmax = float(r.iloc[[i for i,col in enumerate(lfs.columns) if "Maximum" in col and "Flow" in col][0]]) if False else None
    rows.append(dict(link=lid, slope=s, cap_utils=cap, to_outfall=c.to_node in O))
d = pd.DataFrame(rows).set_index("link")
colmax = [c for c in lfs.columns if "Max" in c and "Full" in c and "Flow" in c][0]
colq = [c for c in lfs.columns if ("Maximum" in c or "Max" in c) and ("|Flow|" in c or "Flow" in c) and "Full" not in c and "Time" not in c][0]
d["qmax"] = lfs[colq].astype(float); d["ratio"] = lfs[colmax].astype(float)
d["cap_swmm"] = d.qmax / d.ratio
m = d[(d.ratio > 0.05) & d.cap_utils.notna()]
print(f"conductos comparados: {len(m)}; cap_utils/cap_swmm: min {np.min(m.cap_utils/m.cap_swmm):.4f} mediana {np.median(m.cap_utils/m.cap_swmm):.4f} max {np.max(m.cap_utils/m.cap_swmm):.4f}")
print("pendientes: n<0 (adversas):", int((d.slope<0).sum()), " n==0:", int((d.slope==0).sum()), " min %.5f max %.5f" % (d.slope.min(), d.slope.max()))
print("conductos hacia outfall:", d[d.to_outfall].index.tolist())
st = compute_topology_features(extract_static_features(INP), INP)
print("NaN en features estaticas (n nodos):", st[["diam_max_in","pendiente_max_in","pendiente_out","dist_outfall_m","upstream_capacity_lps"]].isna().sum().to_dict())
print("nodo(s) con pendiente_out NaN y n_tuberias_out>0:", st[(st.pendiente_out.isna()) & (st.n_tuberias_out>0)].node_id.tolist())
```

### `d14_parity.py`

```python
"""Paridad train/serve de features dinamicas: DB (entrenamiento) vs ruta de inferencia
(write_shape_validation_csv -> load_scenario -> mismas lineas que ScenarioPredictor.predict_timed).
Tambien mide el efecto del redondeo a minutos (H:MM) sobre volumen y tiempo al pico. Sin simular."""
import sys, glob, os, sqlite3, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features, build_network_graph
from swmm_resilience.extraction.dynamic_features import compute_scenario_dynamic_features
from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
from swmm_resilience.simulation.swmm_api_io import load_inp
from swmm_resilience.validation.hydrograph_csv import write_shape_validation_csv, load_scenario
from swmm_resilience.ml.scenario_predict import _peak_inflow_lps, _time_to_peak_h
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
full = compute_topology_features(extract_static_features(INP), INP); G, _ = build_network_graph(load_inp(INP))
base = dict(zip(full.node_id.astype(str), full.base_inflow_lps)); exp = {n for n,v in base.items() if v > 0}
c = sqlite3.connect(f"file:{repo/'outputs/training_v17.sqlite3'}?mode=ro", uri=True)
tmp = Path(tempfile.mkdtemp(dir="."))
f = 2.0; worst = {}
for root in ("data/hydrograph_shapes", "data/hydrograph_validation"):
    for sid, shape in load_all_shapes(repo / root).items():
        sc = load_scenario(write_shape_validation_csv(sid, shape, base, f, tmp / sid), exp)
        pk = {n: _peak_inflow_lps(s) for n, s in sc.node_series.items()}
        rep = next((s for s in sc.node_series.values() if s), [])
        dyn = compute_scenario_dynamic_features(full, pk, G, sc.last_time_hours, _time_to_peak_h(rep)).set_index("node_id")
        # volumen con tiempos redondeados vs originales
        t0 = np.array([t for t,_ in shape]); q0 = np.array([q for _,q in shape])
        t1 = np.array(sc.time_grid_hours)
        dv = np.trapz(q0, t1)/np.trapz(q0, t0) - 1
        msg = f"{sid:38s} redondeo H:MM: max|dt|={60*np.abs(t1-t0).max():.2f} min, dVol={100*dv:+.2f}%"
        if root.endswith("shapes"):
            db = pd.read_sql("select node_id, q_pico_nodo, q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h from training_samples_v17 where shape_id=? and abs(factor_mult-?)<1e-9", c, params=(sid, f)).set_index("node_id")
            j = db.join(dyn, rsuffix="_inf")
            d = {k: float((j[k]-j[k+"_inf"]).abs().max()) for k in ("q_pico_nodo","q_pico_acum_escalado","duracion_horas","tiempo_al_pico_h")}
            msg += "  | max|train-inferencia|: " + ", ".join(f"{k}={v:.4g}" for k,v in d.items())
        print(msg)
```

### `d15_identity.py`

```python
"""Control de identidad: .inp original (copia) vs pipeline factor 1.0; Qx2/Qx3 originales vs Qx1 x2/x3 por pipeline."""
import sys, shutil
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
N = repo / "data/networks"
cases = [("Qx1 original (copia)", N/"chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp", None),
         ("Qx1 pipeline f=1.0", None, 1.0),
         ("Qx2 original (copia)", N/"chico_hydro-qx2/SWMM - Chico (PVC) Prueba 5 - Qx2.00.inp", None),
         ("Qx1 pipeline f=2.0", None, 2.0),
         ("Qx3 original (copia)", N/"chico_hydro-qx3/SWMM - Chico (PVC) Prueba 5 - Qx3.00.inp", None),
         ("Qx1 pipeline f=3.0", None, 3.0)]
labs = {}
for name, src, f in cases:
    if src is not None:
        p = OUT / ("id_" + name.split()[0] + ".inp"); shutil.copy(src, p)
    else:
        p = OUT / f"id_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    m = run(p); labs[name] = extract_labels(p.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    print(name, {k: m[k] for k in ("ext_in_m3","flood_loss_m3","vol_labels_m3","n_inunda","cont_err_pct")})
for a, b in (("Qx2 original (copia)","Qx1 pipeline f=2.0"), ("Qx3 original (copia)","Qx1 pipeline f=3.0"), ("Qx1 original (copia)","Qx1 pipeline f=1.0")):
    d = labs[a].join(labs[b], rsuffix="_b")
    print(f"{a} vs {b}: etiquetas distintas={int((d.inunda!=d.inunda_b).sum())}, max|dVol| nodo={float((d.vol_inundacion_m3-d.vol_inundacion_m3_b).abs().max()):.0f} m3")
```

### `d16_capacity_index.py`

```python
"""Poder discriminante (AUC) de un indice hidraulico q_pico_acum_escalado / capacidad Manning del conducto
de SALIDA vs. features existentes. Lectura mode=ro de la BD; capacidad con utils.circular_full_flow_lps."""
import sys, sqlite3, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file
from swmm_resilience.utils import circular_full_flow_lps
inp = read_inp_file(str(repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"), encoding="utf-8")
J, O = inp["JUNCTIONS"], inp["OUTFALLS"]; el = lambda n: J[n].elevation if n in J else O[n].elevation
cap_out = {}
for lid, c in inp["CONDUITS"].items():
    s = (el(c.from_node) + (c.offset_upstream or 0) - el(c.to_node) - (c.offset_downstream or 0)) / c.length
    cap_out[str(c.from_node)] = circular_full_flow_lps(inp["XSECTIONS"][lid].height, abs(s), c.roughness)
db = sqlite3.connect(f"file:{repo/'outputs/training_v17.sqlite3'}?mode=ro", uri=True)
df = pd.read_sql("select node_id, q_pico_acum_escalado, q_pico_nodo, upstream_capacity_lps, inunda from training_samples_v17", db)
df["cap_out_lps"] = df.node_id.map(cap_out)
df["indice_q_cap_out"] = df.q_pico_acum_escalado / df.cap_out_lps
for col in ("q_pico_acum_escalado", "q_pico_nodo", "indice_q_cap_out"):
    m = df[col].notna(); print(f"AUC({col}) = {roc_auc_score(df.inunda[m], df[col][m]):.3f}  (n={m.sum()})")
m = df.upstream_capacity_lps.notna(); print(f"AUC(q_pico_acum_escalado/upstream_capacity_lps) = {roc_auc_score(df.inunda[m], (df.q_pico_acum_escalado/df.upstream_capacity_lps)[m]):.3f} (n={m.sum()}, excluye {(~m).sum()} filas de cabecera)")
```

### `d17_tpico_inferencia.py`

```python
"""Muestra que ScenarioPredictor toma tiempo_al_pico_h de la PRIMERA serie del CSV (aunque sea nula)."""
import sys; sys.path.insert(0, sys.argv[1])
from swmm_resilience.ml.scenario_predict import _time_to_peak_h
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
sc = HydrographScenario('x', {'3C': [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], '1C': [(0.0, 0.0), (1.0, 12.0), (2.0, 0.0)]}, [0, 1, 2], 2.0)
rep = next((s for s in sc.node_series.values() if s), [])   # misma linea que scenario_predict.py:118
print('t_pico usado por ScenarioPredictor:', _time_to_peak_h(rep), '(real del evento: 1.0 h)')
```

### `d18_db_checks.py`

```python
"""Consultas (mode=ro): totales por escenario para cruzar con las re-simulaciones (V-03) y umbral fisico (V-12)."""
import sqlite3, sys
import pandas as pd
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for s, f in [('base',3.0),('base',5.0),('long_moderate',3.0),('long_moderate',5.0),('shape_ultra_short_impulse',5.0),('shape_2h',3.0),('shape_micro_burst_45min',5.0)]:
    print(s, f, c.execute('select sum(vol_inundacion_m3), sum(inunda) from training_samples_v17 where shape_id=? and abs(factor_mult-?)<1e-9', (s, f)).fetchone())
d = pd.read_sql('select shape_id, factor_mult, sum(inunda) n from training_samples_v17 group by 1,2', c)
print('corridas sin ningun nodo inundado:', int((d.n == 0).sum()), 'de', len(d), '; corridas con factor<=1.0:', int((d.factor_mult <= 1.0 + 1e-9).sum()))
first = d[d.n > 0].groupby('shape_id').factor_mult.min()
print('factor minimo con inundacion: min', first.min(), 'max', first.max())
```
