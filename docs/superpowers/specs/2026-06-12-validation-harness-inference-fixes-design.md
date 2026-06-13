# Spec — Corrección del harness de validación e inferencia del surrogate

**Fecha:** 2026-06-12
**Origen:** hallazgos M1–M5, MO2, MO3, MO5, MO6 y menores #1/2/4/5 de
`AUDITORIA_SURROGATE_HIDROGRAMAS.md`.
**Fuera de alcance:** C1/C2 (features de forma de evento + dataset diverso — próximo
spec), app desktop, modelos temporales LSTM/CNN.

## Decisiones ya tomadas con el usuario

| Decisión | Elección |
|---|---|
| Drenaje post-evento (M3) | Auto-extensión con caudal cero (buffer configurable) |
| Umbral de inundación (M5) | `vol >= 1.0 m³` en todas partes → exige re-etiquetar y reentrenar |
| Cobertura de nodos (M4) | Todas las junctions, no solo nodos con inflow |
| `factor_mult` (M1) | Se elimina de `FEATURE_COLS` (queda como metadato del dataset) |

---

## 1. Regla de etiquetado unificada (M5, menor #2)

**Cambio.** Una única función en `swmm_resilience/extraction/labels.py` será el solo
punto que convierte un `.rpt` en `(node_id, vol_m3, inunda)`, con la regla
`inunda = vol >= threshold` y `flood_threshold_m3: 1.0` como nuevo default en
`config.yaml`. El parser de texto fallback de
`hydrograph_batch._parse_node_flooding_text` se muda a este módulo;
`hydrograph_batch` deja de tener lógica propia de parsing/etiquetado.

**Por qué importa.** Hoy entrenamiento usa `vol > 0.0` y validación `vol >= 0.0`
(con fallback CLI 0.1): el mismo nodo con el mismo .rpt puede ser "no inundado" para
entrenar y "inundado" para validar. Con el redondeo del .rpt (resolución 1 m³),
los nodos marginales generan falsos errores de clasificación que **no son del modelo
sino del harness**. Umbral ≥ 1 m³ alinea la definición con la resolución física del
reporte y elimina esa fuente de ruido. Centralizar en una función única hace
estructuralmente imposible que las definiciones vuelvan a divergir.

## 2. Contrato de features v2 — sin `factor_mult` (M1 raíz)

**Cambio.**

- `FEATURE_COLS` pasa de 16 a 15 columnas (sale `factor_mult`). El dataset CSV
  **conserva** la columna `factor_mult` como metadato (LOSO agrupa por escenario y la
  evaluación estratifica por factor), pero ya no se alimenta a los modelos.
- Nueva función `compute_scenario_dynamic_features(scenario, topo_df, inp)`:
  - `q_pico_nodo` = pico real de la serie del escenario por nodo; 0.0 para junctions
    sin inflow directo (consistente con entrenamiento, donde `base_inflow=0 × factor = 0`).
  - `q_pico_acum_escalado` = **suma de los picos reales del escenario** sobre
    ancestros topológicos ∪ el propio nodo, reutilizando el grafo de
    `extraction/topology.py`.
- Se elimina el hack del "factor promedio" de `scenario_predict.py`.
- El pipeline de entrenamiento (`compute_dynamic_features`) sigue funcionando por
  factor, solo que `factor_mult` queda fuera del vector de entrada.

**Por qué importa.** `factor_mult` es un atributo global del escenario que no tiene
definición válida para un hidrograma arbitrario — el "factor promedio" existía solo
para poder llenarlo, y al combinarse con picos reales por nodo rompía las invariantes
con que se entrenaron los árboles (`q_pico_nodo = base × factor`), entregando al
modelo puntos fuera de la distribución de entrenamiento. Además, para escenarios
espacialmente no uniformes, `q_pico_acum_base × factor_medio` es directamente un valor
incorrecto del caudal acumulado. Este cambio hace que **el mismo vector de features
signifique lo mismo en entrenamiento y en inferencia**, condición mínima para que la
validación con hidrogramas distintos mida al modelo y no al pegamento.

## 3. Refactor del predictor: `ScenarioPredictor` (MO6-prerrequisito, M2, M4)

**Cambio.** `predict_scenario` se convierte en una clase:

```python
predictor = ScenarioPredictor(clf_path, reg_path, inp_path, factor_range=(0.2, 5.0))
# constructor: joblib.load ×2 + features estáticas/topológicas, UNA vez
result_df = predictor.predict(scenario)
# → node_id, inunda_pred, prob_inunda, vol_pred_m3, extrapolated (bool)
```

- Predice sobre **todas las junctions**, no solo los nodos del CSV.
- Devuelve `prob_inunda` (`predict_proba`) además de la clase.
- Marca `extrapolated=True` cuando la razón pico-escenario/pico-base del nodo cae
  fuera de `[factor_min, factor_max]` del config (nodos con base 0 nunca se marcan).

**Por qué importa.**

- *Carga única:* hoy cada escenario recarga ambos joblib y re-parsea el .inp; con N
  escenarios se paga N veces. Sin separarlo, cualquier medición de tiempo ML (MO6)
  quedaría inflada y el speed-up reportado sería falso. Con una red neuronal futura,
  la carga de pesos puede dominar el tiempo total — debe pagarse una sola vez.
- *Probabilidad:* habilita PR-AUC/calibración; con clases tan desbalanceadas la
  exactitud sola es engañosa.
- *Extrapolación:* XGBoost satura fuera del rango de entrenamiento sin avisar; un
  hidrograma con pico 7× el base devolvería hoy la predicción de ~5× en silencio. La
  bandera convierte un error invisible en un dato auditable por nodo.

## 4. Harness de validación (M3, M4, MO2, MO3, MO5, MO6, menores #4/5)

### 4a. Drenaje post-evento (M3)

`write_scenario_inp` agrega un punto de caudal 0 al final de cada serie y extiende
`END_TIME` en `validation.drain_down_hours` (nuevo parámetro de config, default 6.0).
Si el último valor del CSV de algún nodo supera una tolerancia (1 % de su pico), se
emite warning con la lista de nodos.

**Por qué importa.** Si la simulación corta cuando la red sigue en carga, el *Node
Flooding Summary* reporta volúmenes truncados: la "verdad" SWMM queda sesgada a la
baja y el surrogate parecería **sobre-predecir** sin estarlo. El buffer garantiza que
la referencia contra la que se mide el modelo sea el volumen de inundación completo
del evento.

### 4b. Guardas de validez (MO3, menores #4/5, parte de menor #3)

| Guarda | Acción |
|---|---|
| `FLOW_UNITS ≠ LPS` en el .inp base | **Abortar** con mensaje claro |
| `ALLOW_PONDING` activado | **Warning** (cambia la semántica de columnas del .rpt y del volumen perdido) |
| `md5(base_inp) ≠ training_inp_hash.txt` | **Abortar**, con flag `--allow-inp-mismatch` para continuar bajo responsabilidad del usuario |
| Error de continuidad de routing del .rpt > 5 % | **Warning** + columna `continuity_error_pct` por escenario |

**Por qué importa.** Cada una de estas condiciones invalida la comparación sin
producir ningún error visible: unidades US convertirían 10⁶ gal como si fueran 10⁶ L
(factor ~3.8 de error silencioso); validar contra una red distinta a la de
entrenamiento mide ruido; un error de continuidad alto significa que ni siquiera SWMM
se cree su propio balance de masa. Son cheques baratos que protegen todas las
conclusiones aguas abajo.

### 4c. Cobertura de todos los nodos (M4)

El zero-fill de la verdad SWMM se construye sobre **todas las junctions** del .inp
(desde las features estáticas), no sobre las llaves del CSV. `predict()` ya cubre
todas las junctions (sección 3); `build_comparison_df` no cambia (sigue exigiendo
conjuntos idénticos).

**Por qué importa.** La inundación por sobrecarga aguas arriba ocurre típicamente en
junctions intermedias **sin** aporte directo. Hoy esas junctions están excluidas de la
comparación: un falso negativo del modelo en un punto crítico de la red sería
invisible para las métricas. Con cobertura total, la validación responde la pregunta
operativa real: *¿dónde se inunda la red?*, no *¿se inundan los puntos de inyección?*

### 4d. Salidas nuevas (MO5, MO6, MO2)

Todas en `out_dir`:

| Archivo | Contenido |
|---|---|
| `scenario_totals.csv` | `scenario_id, vol_total_swmm_m3, vol_total_pred_m3, error_m3, error_pct, n_extrapolated` |
| `timings.csv` | `scenario_id, t_write_inp_s, t_swmm_s, t_parse_rpt_s, t_features_s, t_inference_s, speedup` (+ fila/header con `t_model_load_s`, `t_static_features_s` una vez, y columna `device`) |
| `metrics_per_scenario.csv` | métricas de clasificación + **CSI** + MAE/RMSE condicionales a nodos inundados, por escenario |
| `plots/totals_comparison.png` | barras pareadas SWMM vs ML por escenario (volumen total de red) |
| consola | totales por escenario, totales agregados, speed-up medio, warnings |

El dict retornado por `run_batch_validation` agrega `per_scenario`, `timings` y
`pr_auc` (sobre `prob_inunda` agrupado de todos los nodos×escenarios).

**Por qué importa.**

- *Totales por escenario (MO5):* el volumen total inundado de la red es la cantidad
  que conecta con decisiones de ingeniería (dimensionamiento, riesgo agregado), y por
  escenario revela si una **forma** de hidrograma específica falla sistemáticamente —
  el dato central de la pregunta de investigación. Hoy solo existe en pool, donde
  sobre- y sub-predicciones entre escenarios se cancelan.
- *Tiempos (MO6):* el speed-up t_SWMM/t_ML **es** la justificación de existir de un
  surrogate; sin medirlo no hay argumento cuantitativo para la tesis. La separación
  por componente (escritura .inp, SWMM, parseo, features, inferencia) permite saber
  qué escala y qué no cuando se cambie de XGBoost a una red neuronal — la interfaz es
  agnóstica al modelo y la columna `device` deja listo el reporte CPU/GPU.
- *CSI y métricas condicionales (MO2):* con mayoría de pares (0, 0), accuracy y MAE
  globales se inflan solos. El CSI (estándar en validación de inundaciones) ignora los
  verdaderos negativos triviales; los errores condicionales a nodos inundados miden el
  caso que importa.

### 4e. Regla de conciliación clasificador/regresor (menor #1)

Se documenta y testea la regla: `inunda_pred` la decide el clasificador;
`vol_pred_m3` se reporta tal cual (sin forzar coherencia con el umbral).

**Por qué importa.** La ambigüedad actual (¿qué significa `inunda_pred=1` con
`vol_pred=0.4 m³`?) genera dudas al interpretar resultados; fijar la regla evita
discusiones posteriores y hace los números reproducibles.

## 5. Re-etiquetado y reentrenamiento

Con umbral 1.0 m³ y el contrato de 15 features:

1. `python main.py --skip-simulation --skip-extraction` no sirve aquí: hay que
   regenerar etiquetas desde los .rpt existentes (o re-simular si no se conservan) y
   reensamblar `dataset_final.csv`.
2. Reentrenar clasificador y regresor; regenerar métricas y feature importance.
3. Actualizar `README.md` y `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md` (contrato de
   features, umbral, nuevas salidas de validación).

**Por qué importa.** Sin reentrenar, los modelos guardados esperan 16 features y
etiquetas `> 0`: el harness corregido sería incompatible con ellos. Este paso cierra
el ciclo para que entrenamiento, inferencia y validación cuenten la misma historia.

## Flujo de datos resultante

```text
CSVs ─► load_scenario ─► write_scenario_inp (+drain-down) ─► SWMM ─► labels.py (umbral ≥1 m³, todas las junctions)
                 │                                                            │
                 └─► ScenarioPredictor.predict (features v2, todas las junctions, prob, extrapolated)
                                                  │
                     build_comparison_df ◄────────┘
                     ├─ comparison_summary.csv (por nodo)
                     ├─ scenario_totals.csv / metrics_per_scenario.csv / timings.csv
                     └─ plots (existentes + totals_comparison.png)
```

## Estrategia de pruebas

TDD por componente. Tests existentes que cambian de contrato:
`test_scenario_predict`, `test_hydrograph_batch`, `test_timeseries_scenario`,
`test_ml_trainer_predict`, `tests/ml/test_preprocessing_feature_contract`. Tests
nuevos: regla de etiquetado unificada (≥, umbral 1.0, redondeo), acumulación de picos
reales sobre ancestros (red sintética de 4–5 nodos), drain-down (END_TIME y punto
cero), guardas (FLOW_UNITS, hash, continuidad), `extrapolated`, esquemas de
`scenario_totals.csv` / `timings.csv` / `metrics_per_scenario.csv`, CSI con casos
borde (división por cero → `None`).

## Criterios de aceptación

1. `pytest tests` en verde con los contratos nuevos.
2. Una corrida `--evaluate-hydrographs` sobre los CSVs de prueba produce los tres CSV
   nuevos, el gráfico de totales y el resumen en consola con speed-up.
3. Cargas de modelo y features estáticas ocurren exactamente una vez por batch
   (verificable en `timings.csv`).
4. La misma función de etiquetado es importada por entrenamiento y validación (no hay
   segunda implementación viva).
5. Dataset regenerado y modelos reentrenados con el contrato v2; hash del .inp de
   entrenamiento verificado por el harness.
