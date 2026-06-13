# Auditoría técnica — Surrogate ML de inundación nodal y validación con hidrogramas

**Fecha:** 2026-06-12
**Alcance:** pipeline tabular spec v4 (XGBoost clasificador + regresor) y el flujo de
validación `--evaluate-hydrographs` (CSV → .inp → SWMM vs. predicción ML).
**Perspectiva:** ingeniería de recursos hídricos / hidroinformática / modelos surrogate
en redes de drenaje urbano.

Clasificación de hallazgos:

| Nivel | Significado |
|---|---|
| 🔴 CRÍTICO | Compromete la validez científica del experimento de generalización a hidrogramas distintos |
| 🟠 MAYOR | Sesga resultados o produce comparaciones incorrectas en casos realistas |
| 🟡 MODERADO | Limitación metodológica o riesgo latente; documentar y mitigar |
| 🔵 MENOR | Calidad de código / consistencia; bajo impacto en conclusiones |

---

## 🔴 CRÍTICOS

### C1. El espacio de características es "ciego a la forma" del hidrograma

Las únicas variables dinámicas son `factor_mult`, `q_pico_nodo` y
`q_pico_acum_escalado` ([trainer.py:13-19](swmm_resilience/ml/trainer.py#L13-L19),
[dynamic_features.py](swmm_resilience/extraction/dynamic_features.py)). Todas derivan
exclusivamente del **caudal pico**. El modelo no recibe información de:

- volumen total del evento (∫Q dt)
- duración del hidrograma
- tiempo al pico / forma (adelantado, centrado, retrasado)
- tiempo o volumen por encima de la capacidad de conducción

Consecuencia hidráulica: dos hidrogramas con el mismo pico pero, p.ej., 2 h vs. 12 h de
duración producen **features idénticas → predicciones idénticas**, mientras que el
volumen de inundación en SWMM difiere drásticamente (la inundación nodal es
fundamentalmente un fenómeno de *volumen excedente sobre capacidad*, no solo de pico).
El experimento actual de "¿generaliza a otros hidrogramas?" solo puede salir bien para
hidrogramas que sean aproximadamente re-escalados del hidrograma base; para formas
distintas el modelo fallará **por construcción**, no por falta de entrenamiento.

### C2. El dataset de entrenamiento tiene un solo grado de libertad dinámico

Los escenarios sintéticos son barridos de un factor multiplicativo uniforme
(0.2–5.0, paso 0.2; [config.yaml:5-8](config.yaml#L5-L8), [batch.py](swmm_resilience/simulation/batch.py))
aplicados al mismo hidrograma base. Dentro del dataset, para cada nodo las tres
features dinámicas son colineales perfectas (lineales en `factor`). El modelo entrenado
es, en la práctica, un **interpolador de superficie de respuesta nodo×factor**, no un
surrogate consciente del evento. Implicaciones:

- LOSO y GroupKFold5 sobre estos escenarios miden **interpolación en factor**, no
  generalización a eventos nuevos. Las métricas reportadas son optimistas respecto a
  la pregunta de investigación actual.
- La identidad de cada nodo está memorizada vía su huella de features estáticas
  (válido para una red fija, pero debe declararse explícitamente como surrogate
  *red-específico*).

---

## 🟠 MAYORES

### M1. Construcción de features en inferencia inconsistente con el entrenamiento

En [scenario_predict.py:88-126](swmm_resilience/ml/scenario_predict.py#L88-L126):

1. `factor_mult` y `q_pico_acum_escalado` se calculan con el **factor promedio** de
   todos los nodos, pero `q_pico_nodo` se sobreescribe con el pico real por nodo.
   En entrenamiento siempre se cumplía la invariante
   `q_pico_nodo = base_inflow_lps × factor_mult` y
   `q_pico_acum_escalado = q_pico_acum_base × factor_mult`. Con hidrogramas no
   uniformes esa invariante se rompe → el modelo recibe **combinaciones de features
   fuera de la distribución de entrenamiento** (los árboles nunca vieron esos puntos).
2. Para escenarios espacialmente heterogéneos, `q_pico_acum_escalado` es directamente
   incorrecto: debería recomputarse como la suma de los picos *reales* del escenario
   sobre los ancestros topológicos (la lógica ya existe en
   [topology.py:83](swmm_resilience/extraction/topology.py#L83)), no como
   `q_pico_acum_base × factor_medio`.
3. Nodos con `base_inflow_lps = 0` aportan factor 1.0 al promedio
   ([scenario_predict.py:36-38](swmm_resilience/ml/scenario_predict.py#L36-L38)),
   diluyendo/distorsionando el factor efectivo.

### M2. Extrapolación silenciosa fuera del rango de entrenamiento

XGBoost (árboles) **no extrapola**: para factores efectivos > 5.0 o < 0.2 las
predicciones se saturan en el valor de la hoja extrema. Un hidrograma de validación con
pico 7× el base devolverá la predicción de ~5× sin ninguna advertencia. Recomendado:
calcular el factor efectivo por nodo y emitir warning/columna `extrapolated=True`
cuando salga de `[factor_min, factor_max]`.

### M3. La simulación de validación termina en el último punto del CSV (sin drenaje)

[timeseries_scenario.py:95-100](swmm_resilience/simulation/timeseries_scenario.py#L95-L100)
fija `END_TIME` = último timestamp del CSV. Si en ese instante la red sigue en carga,
el *Node Flooding Summary* reporta volúmenes **truncados** → la "verdad" SWMM queda
sesgada a la baja y contamina las métricas de validación. Recomendado: exigir que el
CSV termine en caudal ~0 (regla de validación adicional) **y** añadir un colchón de
drenaje (p.ej. +2–6 h con caudal cero) antes de `END_TIME`.

### M4. La comparación solo cubre nodos con aporte directo

`expected_nodes` se deriva de `[INFLOWS]`
([hydrograph_batch.py:45-52](swmm_resilience/validation/hydrograph_batch.py#L45-L52)) y
tanto `_build_swmm_df` como `predict_scenario` se restringen a esos nodos. Cualquier
junction **sin** inflow directo que se inunde en SWMM (sobrecarga por aportes aguas
arriba, caso hidráulicamente común) queda **fuera de las métricas**, y el
entrenamiento sí incluyó todas las junctions. Verificar si en la red Chico Sur todas
las junctions tienen inflow; si no, la validación subestima los falsos negativos.

### M5. Semántica inconsistente del umbral de inundación

- Entrenamiento: `inunda = vol > threshold` (estricto, [labels.py:24](swmm_resilience/extraction/labels.py#L24)).
- Validación: `inunda_swmm = vol >= threshold` (no estricto, [hydrograph_batch.py:103](swmm_resilience/validation/hydrograph_batch.py#L103)).
- Con `flood_threshold_m3 = 0.0` ([config.yaml:12](config.yaml#L12)) y el redondeo del
  .rpt (3 decimales de 10⁶ L ⇒ resolución 1 m³; volúmenes < ~500 L se imprimen como
  `0.000`): un nodo listado con volumen redondeado a 0 es **no-inundado en
  entrenamiento** pero **inundado en validación**. Además el fallback CLI usa 0.1 m³
  ([main.py:164-171](main.py#L164-L171)), distinto del config (0.0). Unificar
  operador y valor (sugerencia: ≥ 1 m³, coherente con la resolución del .rpt).

---

## 🟡 MODERADOS

### MO1. `q_pico_acum_base` asume simultaneidad de picos y omite tránsito

La suma de picos aguas arriba ([topology.py:83](swmm_resilience/extraction/topology.py#L83))
ignora atenuación y desfase por tránsito hidráulico (el pico de la suma ≠ suma de los
picos). Es un proxy estático razonable, pero su error crece con el tamaño de la red y
con hidrogramas de duración corta. `upstream_capacity_lps` solo considera los conductos
inmediatos (Manning a tubo lleno), no la capacidad limitante del camino aguas abajo,
que suele controlar la inundación. Documentar como limitación; considerar feature
"capacidad mínima del camino al outfall".

### MO2. Métricas de validación: agregación que maquilla errores

- MAE/RMSE se calculan sobre **todos** los nodos×escenarios, incluidos los muchos pares
  (0, 0): deflacta el error. Reportar también métricas condicionales a nodos inundados
  (en SWMM o en predicción).
- El clasificador usa `predict` (umbral 0.5) sin calibración; reportar PR-AUC/Brier y
  el **CSI (Critical Success Index)**, métrica estándar en validación de inundaciones.
- Las métricas se agregan en pool sobre escenarios; añadir desglose por escenario para
  detectar si una forma de hidrograma particular falla sistemáticamente (clave para la
  pregunta de investigación).
- `error_pct_total` compara volúmenes totales que pueden compensarse entre nodos
  (sobre-predicción en unos cancela sub-predicción en otros); complementarlo con el
  error absoluto medio relativo por nodo inundado.

### MO5. El volumen total inundado de la red (SWMM vs ML) no se reporta de forma utilizable

Requerimiento del usuario: ver el volumen total inundado en la red para ambos modelos.
Estado actual:

- `vol_total_swmm_m3` y `vol_total_pred_m3` existen, pero solo **agregados en pool
  sobre todos los escenarios** dentro del dict que retorna `run_batch_validation`
  ([model_comparison.py:99-101](swmm_resilience/analysis/model_comparison.py#L99-L101));
  no se persisten por escenario.
- El gráfico "aggregated parity" dibuja **un único punto por escenario** en una figura
  separada ([model_comparison.py:100-122](swmm_resilience/visualization/model_comparison.py#L100-L122)),
  lo que impide comparar escenarios entre sí de un vistazo.
- `comparison_summary.csv` es por nodo; obtener el total por escenario exige
  post-procesar el CSV a mano.

Recomendado: generar un `scenario_totals.csv` (columnas: `scenario_id`,
`vol_total_swmm_m3`, `vol_total_pred_m3`, `error_m3`, `error_pct`) y un gráfico único
de barras pareadas (o parity con un punto por escenario) con todos los escenarios,
además de imprimir ambos totales en la salida de consola del CLI.

### MO6. No se mide el tiempo de cómputo SWMM vs ML (la justificación central de un surrogate)

Requerimiento del usuario: conocer el tiempo que tarda cada simulación SWMM y el
tiempo de la inferencia ML. El *speed-up* (t_SWMM / t_ML) es el argumento principal
para usar un surrogate y un eje clave de escalabilidad si a futuro se migra a una red
neuronal; hoy no se registra ningún tiempo en
[hydrograph_batch.py](swmm_resilience/validation/hydrograph_batch.py).

Problemas asociados que sesgarían la medición si se añadiera de forma ingenua:

- `predict_scenario` hace `joblib.load` del clasificador y el regresor **en cada
  llamada** ([scenario_predict.py:69-70](swmm_resilience/ml/scenario_predict.py#L69-L70)),
  y dentro del loop de escenarios eso recarga los modelos N veces. El tiempo de carga
  del modelo debe separarse del tiempo de inferencia (con una red neuronal la carga de
  pesos puede dominar y se paga una sola vez en producción).
- `predict_scenario` también re-parsea el .inp y recalcula features estáticas y
  topológicas por escenario; en un benchmark honesto conviene separar
  *features estáticas (una vez)* vs *features dinámicas + inferencia (por escenario)*.
- El tiempo SWMM debe medirse alrededor de `_run_swmm` (simulación pura), excluyendo
  la escritura del .inp y el parseo del .rpt, y reportando también esos componentes
  por separado para diagnóstico.

Recomendado: instrumentar con `time.perf_counter()` y persistir por escenario en un
`timings.csv` (columnas sugeridas: `scenario_id`, `t_write_inp_s`, `t_swmm_s`,
`t_parse_rpt_s`, `t_features_s`, `t_inference_s`, `speedup`), imprimir el speed-up
medio en consola y graficarlo. Diseñar la interfaz de medición de forma agnóstica al
modelo (la misma columna `t_inference_s` debe servir para XGBoost hoy y para una
LSTM/CNN mañana, incluyendo una columna opcional `device` CPU/GPU).

### MO3. Sin validación de unidades

Se asume `FLOW_UNITS = LPS` (CSV `value_lps`) y .rpt en SI (volumen en 10⁶ L)
([swmm_api_io.py:303-307](swmm_resilience/simulation/swmm_api_io.py#L303-L307),
[hydrograph_batch.py:178](swmm_resilience/validation/hydrograph_batch.py#L178)). Un
.inp en unidades US (CFS/GPM, volumen en 10⁶ gal) se convertiría mal **en silencio**.
Añadir un check de `OPTIONS → FLOW_UNITS` al cargar el .inp base y abortar si ≠ LPS.

### MO4. Nomenclatura engañosa: `base_inflow_lps` es el **pico** de la serie base

[static_features.py:7-26](swmm_resilience/extraction/static_features.py#L7-L26) toma el
`max()` de la serie; el fallback usa `base_value` (caudal constante), mezclando
semánticas. Consistente internamente, pero propenso a errores de interpretación en la
tesis y en mantenimiento. Renombrar a `q_pico_base_lps` (o documentar en el
diccionario de datos).

---

## 🔵 MENORES

1. `predict_scenario` recibe `flood_threshold_m3` pero no lo aplica: puede emitir
   `inunda_pred=1` con `vol_pred_m3 < threshold` (incoherencia interna del par
   clasificador/regresor). Decidir y documentar la regla de conciliación.
2. Parsing duplicado del *Node Flooding Summary* (vía `swmm_api` en
   [labels.py](swmm_resilience/extraction/labels.py) y vía swmm_api + parser de texto en
   [hydrograph_batch.py](swmm_resilience/validation/hydrograph_batch.py)); consolidar en
   un único módulo para garantizar etiquetas idénticas entrenamiento/validación.
3. El parser de texto toma "el último token numérico" como volumen; en .rpt con
   ponding habilitado la última columna es *Maximum Ponded Volume*, no *Total Flood
   Volume* — verificar `ALLOW_PONDING` en el .inp base.
4. `_run_swmm` no captura errores/warnings de continuidad de SWMM; un error de
   continuidad de routing > 5 % invalidaría la "verdad" de referencia. Loggear el
   *continuity error* del .rpt por escenario.
5. `training_inp_hash.txt` se escribe pero el flujo de validación no verifica que el
   `--base-inp` coincida con el hash de entrenamiento (riesgo de validar contra una
   red distinta a la del modelo).

---

## ✅ Fortalezas (mantener)

- Pipeline modular y testeado; el .inp base nunca se modifica en disco.
- Validación estricta del CSV de hidrogramas (9 reglas, malla temporal compartida,
  mapeo 1-a-1 nodo↔serie con detección de series compartidas).
- Regresor en espacio `log1p` con `expm1` + clip ≥ 0: correcto para volúmenes con
  distribución sesgada.
- Evaluación honesta en entrenamiento: clasificador aislado / regresor oracle /
  end-to-end, y `log_nse` sobre predicciones out-of-fold apiladas.
- Conversión de unidades del .rpt documentada (10⁶ L → m³).

---

## Recomendaciones priorizadas

1. **(C1/C2) Añadir descriptores de evento como features dinámicas** — por nodo:
   volumen total del hidrograma (m³), duración efectiva, tiempo al pico,
   volumen/tiempo por encima de `upstream_capacity_lps`. Esto convierte el modelo de
   interpolador-en-factor a surrogate sensible a la forma.
2. **(C2) Regenerar el dataset con diversidad de eventos**: tormentas de diseño de
   varias duraciones y períodos de retorno (bloques alternos / Chicago / SCS),
   estiramientos temporales del hidrograma base a igual pico, e hidrogramas
   espacialmente no uniformes. Re-evaluar con LOSO *por forma de evento* (dejar fuera
   una familia de hidrogramas completa), que es la métrica honesta para la pregunta
   actual.
3. **(M1) Corregir la inferencia**: recomputar `q_pico_acum_escalado` con los picos
   reales del escenario sobre los ancestros del grafo; eliminar el factor promedio o
   definir `factor_mult` por nodo; excluir nodos con base 0 del promedio.
4. **(M3) Colchón de drenaje** en `write_scenario_inp` + regla de CSV "termina en ~0".
5. **(M5/MO3) Unificar umbral (`>` vs `>=`, valor único ≥ 1 m³) y validar `FLOW_UNITS`.**
6. **(M2) Detectar y reportar extrapolación** (factor efectivo fuera de [0.2, 5.0]).
7. **(MO2) Ampliar métricas**: CSI, PR-AUC, métricas condicionales a inundados,
   desglose por escenario.
8. **(MO5) Reportar el volumen total inundado de la red por escenario** (SWMM vs ML):
   `scenario_totals.csv` + gráfico comparativo único con todos los escenarios + totales
   en consola.
9. **(MO6) Instrumentar tiempos SWMM vs ML por escenario** (`timings.csv` + speed-up),
   separando carga de modelo, features e inferencia; cargar los modelos una sola vez
   fuera del loop de escenarios. Diseño agnóstico al modelo para soportar una futura
   red neuronal (CPU/GPU).
8. **(Estratégico)** El repo ya contiene un scaffold temporal (LSTM/CNN en
   `swmm_resilience/ml/temporal/`): es el camino natural para capturar la forma del
   hidrograma si las features de evento del punto 1 resultan insuficientes.
