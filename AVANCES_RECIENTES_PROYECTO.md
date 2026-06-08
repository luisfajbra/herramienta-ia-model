# Avances recientes del proyecto SWMM Resilience

Este documento resume los cambios realizados durante las ultimas semanas para
presentar el avance tecnico del proyecto, explicar que problemas se corrigieron
y mostrar hacia donde queda orientada la siguiente fase.

## 1. Resumen ejecutivo

Durante este periodo el proyecto avanzo desde un pipeline tabular funcional,
pero con varias ambiguedades semanticas, hacia una arquitectura mas clara y
preparada para entrenamiento temporal.

Los avances principales fueron:

- correccion de la semantica de `delta_inflow_lps` e `inflow_multiplier`
- mejora de metricas temporales calculadas durante la simulacion
- integracion mas clara entre `swmm-api` y `PySWMM`
- persistencia de artefactos de modelos ML
- consolidacion de la inferencia desde `.inp`
- implementacion de la Fase 1 del dataset temporal para CNN/LSTM
- documentacion tecnica completa del proyecto
- actualizacion del README y documentos de soporte

El cambio mas importante a nivel estrategico es que el proyecto ya no depende
solo del resumen tabular por nodo. Ahora empieza a guardar series temporales por
nodo y timestep, que son la base necesaria para entrenar CNN 1D y LSTM.

## 2. Problema inicial

Antes de estos cambios, el proyecto ya podia:

- correr simulaciones SWMM
- guardar resultados en SQLite
- exportar `dataset_ml.csv`
- entrenar modelos tabulares

Sin embargo, existian problemas importantes:

- `delta_inflow_lps` podia confundirse con `inflow_multiplier`
- algunos valores de delta podian representar un factor y no un caudal
- ciertas metricas temporales asumian pasos de tiempo de forma insegura
- `predict_tabular.py` podia reentrenar modelos durante la prediccion
- el flujo de inferencia dependia demasiado de CSV auxiliares
- no existia una capa de series temporales para CNN/LSTM
- la documentacion estaba fragmentada o desactualizada

## 3. Correccion de `delta_inflow_lps` e `inflow_multiplier`

### 3.1. Decision tomada

Se separaron dos conceptos que antes podian mezclarse:

- `inflow_multiplier`: factor global de la corrida.
- `delta_inflow_lps`: diferencia fisica de caudal en `L/s`.

Ejemplo:

```text
inflow_multiplier = 2.0
```

significa duplicar el caudal o hidrograma.

En cambio:

```text
delta_inflow_lps = caudal_nuevo - caudal_base
```

representa una magnitud hidraulica.

### 3.2. Impacto

Esta correccion mejora:

- trazabilidad de escenarios
- coherencia fisica del dataset
- interpretabilidad de las variables
- seguridad del entrenamiento ML

Tambien evita que nodos con `base_inflow_lps = 0` reciban artificialmente un
delta positivo solo porque la corrida tiene un multiplicador global.

## 4. Ajustes en resultados por links

Se aclaro que `delta_inflow_lps` no tiene sentido fisico directo en
`link_results`, porque los inflows se aplican a nodos, no a enlaces.

Decision:

- los links conservan `inflow_multiplier` como metadata de corrida
- `delta_inflow_lps` en links queda como `NULL` o dato legacy
- el analisis fisico de caudal adicional se mantiene a nivel de nodo

## 5. Correccion de metricas temporales

### 5.1. `time_to_first_flood_min`

Antes podia calcularse usando una combinacion de `step_count` y timestep.

Ahora se guarda directamente:

```text
elapsed_min del primer timestep con node.flooding > 0
```

Esto evita errores cuando el timestep no es constante.

### 5.2. `depth_rate_m_per_min`

Se corrigio el calculo de tasa de aumento de profundidad.

Decision:

- el primer timestep solo inicializa el valor previo
- la tasa se calcula a partir del segundo valor real

Motivo:

- no comparar el primer valor real contra un cero artificial

### 5.3. `time_to_peak_min`

Se reviso la forma de obtener el tiempo al pico de profundidad para que sea
mas consistente con el tiempo real de simulacion.

## 6. Integracion de `swmm-api` y `PySWMM`

Se consolido la division de responsabilidades:

| Herramienta | Responsabilidad |
|---|---|
| `swmm-api` | Leer y modificar `.inp`, escribir `.inp` temporal, leer `.rpt` |
| `PySWMM` | Ejecutar la simulacion y exponer variables hidraulicas durante la corrida |

Archivo clave:

```text
swmm_resilience/simulation/swmm_api_io.py
```

El uso de `swmm-api` queda encapsulado en esa capa para reducir riesgo ante
cambios futuros de la libreria.

## 7. Uso del reporte `.rpt`

Se decidio preferir el reporte de SWMM para:

- `flooding_volume_m3`
- `flooding_duration_min`

Motivo:

- son metricas agregadas que SWMM reporta oficialmente
- se reduce la posibilidad de inconsistencias al calcularlas manualmente

PySWMM sigue usandose para:

- ejecucion de la simulacion
- profundidad
- caudales de entrada y salida
- flooding por timestep
- estadisticas de links

## 8. Cambios en ML tabular

### 8.1. Modelos ya no se reentrenan al predecir

Se corrigio el flujo de inferencia.

Antes:

```text
predecir -> entrenar otra vez -> predecir
```

Ahora:

```text
entrenar -> guardar artefactos -> cargar artefactos -> predecir
```

Esto es fundamental para que la herramienta sea usable y escalable.

### 8.2. Artefactos persistidos

El entrenamiento guarda:

- modelo de regresion
- modelo de clasificacion
- pipeline completo
- columnas usadas
- metadata del entrenamiento
- `manifest.json`

Ruta:

```text
data/networks/<red>/results/model_artifacts/
```

### 8.3. Inferencia desde `.inp`

Se reforzo la ruta recomendada:

```text
archivo .inp + artefactos entrenados -> prediccion ML
```

Motivo:

- el `.inp` es la fuente original de la red
- no es escalable exigir CSV auxiliares para cada red nueva

La inferencia desde CSV se conserva como ruta legacy o de depuracion.

## 9. Resultados actuales de modelos ML

Con los resultados disponibles para `chico_hydro-qx1`, el mejor desempeno lo
presentan modelos basados en XGBoost.

### 9.1. Regresion

Target:

```text
flooding_volume_m3
```

Mejor modelo:

```text
xgboost
```

Metricas principales:

| Modelo | MAE | RMSE | R2 | CV R2 mean |
|---|---:|---:|---:|---:|
| `xgboost` | 7.739 | 19.528 | 0.958 | 0.980 |
| `svr_rbf` | 26.080 | 72.093 | 0.423 | 0.356 |
| `ridge` | 51.465 | 85.686 | 0.184 | 0.251 |
| `lasso` | 51.469 | 85.688 | 0.184 | 0.251 |

Conclusion:

```text
XGBoost es claramente el mejor regresor actual.
```

### 9.2. Clasificacion

Target:

```text
flooded
```

Mejor modelo global:

```text
xgboost_classifier
```

Metricas principales:

| Modelo | Accuracy | Precision | Recall | F1 | CV F1 mean |
|---|---:|---:|---:|---:|---:|
| `xgboost_classifier` | 0.925 | 0.991 | 0.795 | 0.882 | 0.970 |
| `svc_rbf` | 0.893 | 0.846 | 0.848 | 0.847 | 0.889 |
| `logistic_regression` | 0.812 | 0.802 | 0.615 | 0.696 | 0.819 |

Conclusion:

```text
XGBoost Classifier es el mejor clasificador global actual.
```

Nota:

- `svc_rbf` tiene mayor recall en el split de test actual.
- Si el objetivo operativo es detectar la mayor cantidad posible de fallas,
  conviene revisar el trade-off entre precision y recall.

## 10. PCA y reduccion de variables

La configuracion actual usa PCA:

```python
ML_USE_PCA = True
ML_PCA_COMPONENTS = 5
```

Motivos:

- reducir redundancia entre variables hidraulicas
- estabilizar modelos con datasets todavia pequenos
- comparar modelos en un espacio compacto
- reducir ruido de variables correlacionadas

Decision:

- PCA se conserva como linea base actual
- si se necesita interpretabilidad directa de variables, se puede entrenar sin
  PCA cambiando `ML_USE_PCA = False`

## 11. Validacion agrupada por corrida

Se usa:

```text
GroupShuffleSplit / GroupKFold
```

agrupando por:

```text
run_id
```

Motivo:

- una corrida produce muchas filas, una por nodo
- si se mezclan filas al azar, nodos de la misma corrida pueden quedar en train
  y test
- eso produciria una evaluacion artificialmente optimista

## 12. Fase 0 del pipeline temporal

Se definieron las decisiones base para el futuro dataset CNN/LSTM:

- falla temporal: `failed_now = 1` si `flooding_lps > 0`
- frecuencia objetivo para ML: `5 minutos`
- ventana historica inicial: `20 minutos`
- horizonte inicial de prediccion: `5 minutos`
- avance entre ventanas: `5 minutos`
- todos los escenarios generan `node_timeseries`
- no guardar `head_m`
- no guardar `link_timeseries` en el MVP
- usar PyTorch cuando existan ventanas listas

## 13. Fase 1 temporal implementada

Se implemento la persistencia de series temporales por nodo.

Cada corrida genera:

```text
data/networks/<red>/results/temporal/node_timeseries/run_<run_id>.parquet
```

Columnas:

```text
run_id
network_hash
node_id
step_index
time_sec
time_min
total_inflow_lps
lateral_inflow_lps
depth_m
depth_ratio
flooding_lps
total_outflow_lps
failed_now
```

Impacto:

- el proyecto ya tiene la base cruda para entrenar modelos temporales
- no depende solo del resumen tabular
- permite construir ventanas para CNN/LSTM en fases posteriores

## 14. Ruta acordada para CNN/LSTM

Orden de trabajo:

1. Validar Parquet reales de `node_timeseries`.
2. Crear tabla `temporal_artifacts`.
3. Registrar cada Parquet temporal por `run_id`.
4. Construir ventanas temporales.
5. Entrenar CNN 1D baseline con PyTorch.
6. Comparar contra LSTM.
7. Crear predictor temporal operativo.

Decision sobre modelos:

- empezar con CNN 1D
- usar LSTM como benchmark posterior

Motivo:

- CNN 1D suele ser mas estable y eficiente para empezar
- LSTM puede servir para comparar dependencias temporales mas largas

## 15. Documentacion creada o actualizada

Se actualizaron o crearon:

- `README.md`
- `DOCUMENTACION_COMPLETA_PROYECTO.md`
- `PLAN_TEMPORAL_LSTM_CNN.md`
- `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md`
- `REVISION_DETALLADA_DELTA_Y_ESCENARIOS_PARCIALES.md`

Tambien se reorganizo la documentacion para separar:

- guia rapida
- documentacion completa
- plan temporal
- diccionario de datos
- revision tecnica de deltas

## 16. Estado Git y sincronizacion

Se resolvieron conflictos con `origin/main`.

Se conservo la red Qx1 con el nombre largo:

```text
data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp
```

Se subio el commit:

```text
af8f89b Implementa persistencia temporal por nodo
```

Ese commit contiene la Fase 1 temporal implementada.

## 17. Siguientes pasos recomendados

### 17.1. Validar Fase 1

Ejecutar una corrida controlada y revisar:

- que se genera un Parquet por corrida
- que las columnas son correctas
- que `nodos * timesteps` produce un numero coherente de filas
- que `failed_now` coincide con `flooding_lps > 0`
- que no aparece `head_m`
- que no se genera `link_timeseries`

### 17.2. Implementar Fase 2

Crear:

```text
temporal_artifacts
```

Con columnas:

```text
artifact_id
run_id
artifact_type
path
rows_count
created_at
```

Objetivo:

- enlazar cada corrida SQLite con su Parquet temporal

### 17.3. Preparar Fase 3

Construir ventanas temporales:

```text
node_timeseries -> resampleo 5 min -> ventanas -> targets
```

Salida esperada:

```text
windows.npz
window_metadata.parquet
```

## 18. Conclusion

El proyecto avanzo de forma importante en tres frentes:

1. **Calidad semantica de datos**
   Se corrigieron ambiguedades entre factor, caudal y resultados hidraulicos.

2. **Madurez del pipeline ML**
   Los modelos ahora se entrenan, evaluan y guardan como artefactos reutilizables.

3. **Preparacion para aprendizaje temporal**
   Ya existe la primera capa de series por nodo y timestep, necesaria para
   construir CNN/LSTM.

Con esto, el proyecto queda mejor documentado, mas trazable y con una ruta
clara hacia modelos temporales y escalabilidad multired.
