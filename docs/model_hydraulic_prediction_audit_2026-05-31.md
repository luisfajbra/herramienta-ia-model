# Auditoria de riesgos para redes neuronales, modelos ML y predicciones hidraulicas

Fecha: 2026-05-31

## Alcance

Revise el flujo completo del proyecto: simulacion SWMM, generacion de SQLite/CSV/Parquet, modelos tabulares, CNN/LSTM temporales, surrogates, inferencia y artefactos existentes. El foco fue encontrar bugs, fugas de informacion, inconsistencias de datos y practicas que puedan dañar las predicciones hidraulicas.

Evidencia local usada:

- `data/training/swmm_resilience.db`: 24 corridas `completed`, una red, multipliers Qx1.00 a Qx6.75.
- `data/networks/chico_hydro-qx1/results/dataset_ml.csv`: 3,864 filas + header.
- `data/networks/chico_hydro-qx1/results/model_artifacts/manifest.json`: artefactos tabulares entrenados con 3,864 filas.
- `data/networks/chico_hydro-qx1/results/temporal/model_artifacts/*.csv`: metricas CNN/LSTM/surrogate.

## Resumen ejecutivo

Los mayores riesgos no estan en la arquitectura de PyTorch en si, sino en los contratos de datos alrededor del modelo. Hay tres problemas prioritarios:

1. La ruta temporal `build_temporal_windows()` usa variables hidraulicas de salida de SWMM como features (`depth`, `flooding`, `outflow`) para predecir flooding futuro. Eso puede inflar metricas y no sirve para inferencia sin correr SWMM.
2. Los modelos surrogate guardan el "mejor fold" de validacion, no un modelo final reentrenado con todos los datos; ademas eligen el mejor fold solo por BCE de clasificacion aunque tambien predicen `peak_flooding_lps`.
3. Hay inconsistencias de artefactos y rutas: el `DEFAULT_INP_FILE` apunta a un `.inp` que no existe, y hay 40 Parquets en disco pero solo 24 registrados en la DB. Algunos CSV de metricas parecen pertenecer a corridas viejas.

## Hallazgos criticos

### 1. Fuga de informacion hidraulica en el dataset temporal de ventanas

Referencia: `swmm_resilience/ml/temporal/dataset.py:34-41`, `swmm_resilience/ml/temporal/dataset.py:178-185`

`TEMPORAL_COLS` incluye:

- `depth_m`
- `depth_ratio`
- `flooding_lps`
- `total_outflow_lps`

Luego esas columnas entran en `X_seq`, mientras el target se calcula desde el horizonte futuro. Para una tarea de alerta temprana o predictor sin ejecutar SWMM, esas variables son salidas del simulador, no entradas disponibles. El modelo puede aprender senales directas de flooding actual o casi actual, y las metricas se vuelven optimistas.

Impacto:

- Falsas metricas altas para CNN temporal.
- Modelo no desplegable para prediccion previa al evento.
- Riesgo de que el sistema "prediga" flooding usando flooding ya observado.

Mejor practica:

- Separar dos tareas:
  - Predictor operacional sin SWMM: solo lluvia/hidrograma/inflow planificado + features estaticas.
  - Post-procesador de simulacion: puede usar depth/outflow, pero no debe venderse como prediccion sin SWMM.
- Crear listas explicitas: `AVAILABLE_BEFORE_SWMM_COLS`, `SWMM_OUTPUT_COLS`, `TARGET_COLS`.
- Agregar test que falle si `X_seq` contiene `flooding_lps`, `depth_m`, `depth_ratio` o `total_outflow_lps` en el modo predictor.

### 2. Escalado por `target_nodes` puede modificar nodos no seleccionados si comparten timeseries

Referencia: `swmm_resilience/simulation/swmm_api_io.py:150-181`

`_scale_target_timeseries()` construye un set de nombres de series usadas por los nodos seleccionados y escala la serie completa en `[TIMESERIES]`. Si dos nodos comparten la misma serie temporal y solo uno esta en `target_nodes`, ambos quedan escalados en el `.inp` temporal.

Impacto:

- Escenarios parciales pueden no ser parciales.
- Etiquetas hidraulicas y targets quedan contaminados.
- El modelo puede aprender respuestas de una perturbacion espacial distinta a la solicitada.

Mejor practica:

- Para `target_nodes`, duplicar la timeseries por nodo seleccionado y reasignar el inflow de ese nodo a la copia escalada.
- Agregar test con dos nodos compartiendo la misma timeseries, escalar solo uno, y verificar que el otro conserve el hidrograma original.

### 3. `delta_inflows_lps` se interpreta como multiplier, no como delta en L/s

Referencia: `swmm_resilience/main.py:167-172`, `swmm_resilience/main.py:222-239`

La API acepta `delta_inflows_lps`, pero lo normaliza con `normalize_inflow_multipliers()` y lo guarda tanto como `delta_inflow_lps` como `inflow_multiplier`. Si un usuario pasa `10` esperando sumar 10 L/s, el sistema correra Qx10.

Impacto:

- Corridas hidraulicas drasticamente equivocadas.
- Dataset con metadata enganosa: `delta_inflow_lps` a nivel `runs` no representa un delta fisico.
- Riesgo alto en calibracion y entrenamiento si se mezclan conceptos.

Mejor practica:

- Deprecar `delta_inflows_lps` o implementar un modo aditivo real separado.
- Renombrar UI/API a `inflow_multipliers` en todos los lugares.
- Mantener `delta_inflow_lps` solo como valor calculado por nodo despues de la simulacion.

## Hallazgos altos

### 4. Los surrogates guardan el mejor fold, no un modelo final entrenado con todos los datos

Referencia: `swmm_resilience/ml/temporal/train_surrogate.py:99-118`, `swmm_resilience/ml/temporal/train_surrogate.py:207-219`

Cada fold entrena con una parte de las corridas y valida con otra. Al final se guarda el `state_dict` del fold con menor `val_bce`. Eso deja fuera del entrenamiento final varios multipliers completos. En cambio, el tabular hace evaluacion y despues reentrena para inferencia con todo el dataset (`swmm_resilience/ml/train.py:416-450`).

Impacto:

- El artefacto de inferencia no usa todos los datos disponibles.
- Cambiar `n_cv_folds` cambia tambien el modelo final, no solo la evaluacion.
- Predicciones de Qx pueden depender de que multipliers quedaron fuera del fold ganador.

Mejor practica:

- Usar CV solo para medir.
- Luego reentrenar el modelo final con todos los grupos usando hiperparametros elegidos.
- Guardar manifiesto con `trained_run_ids`, rango de multipliers, feature schema y hash del dataset.

### 5. Seleccion del mejor surrogate ignora la regresion hidraulica

Referencia: `swmm_resilience/ml/temporal/train_surrogate.py:174-179`, `swmm_resilience/ml/temporal/train_surrogate.py:185-203`, `swmm_resilience/ml/temporal/train_surrogate.py:207-212`

El modelo dual predice clasificacion y `peak_flooding_lps`, pero el mejor fold se elige solo con `val_bce`. Un fold puede clasificar bien y estimar mal los picos de flooding, que son el valor hidraulico mas importante para mapas y decisiones.

Impacto:

- Mapas de volumen/caudal de flooding pueden ser malos aunque F1/AUC se vean bien.
- El modelo favorece "inunda/no inunda" sobre magnitud.

Mejor practica:

- Elegir por metrica compuesta normalizada, por ejemplo `BCE + lambda * log1p_RMSE`.
- Guardar tambien el mejor modelo por regresion si se usan cabezas separadas.
- Reportar MAE/RMSE por rangos de multiplier y por nodos criticos.

### 6. La regresion neural usa MSE en L/s sin transformar ni escalar el target

Referencia: `swmm_resilience/ml/temporal/train_surrogate.py:133-156`, `swmm_resilience/ml/temporal/compare_surrogate.py:89-130`

`peak_flooding_lps` llega hasta ~300 L/s en la DB actual. Aunque `beta=0.01`, un error de 100 L/s produce MSE=10,000 y contribucion 100, mucho mayor que una BCE cercana a 1. La ponderacion real depende de la escala fisica del target.

Impacto:

- Entrenamiento inestable entre tareas.
- Cabeza de clasificacion y regresion compiten de forma no controlada.
- Un cambio de unidades o red cambia el balance de perdidas.

Mejor practica:

- Entrenar regresion con `log1p(peak_flooding_lps)` o target estandarizado.
- Evaluar en unidades originales despues de invertir la transformacion.
- Considerar `HuberLoss` para robustez a picos.

### 7. Artefactos temporales inconsistentes con la DB actual

Evidencia:

- DB: `temporal_artifacts` tiene 24 Parquets registrados.
- Disco: `data/networks/chico_hydro-qx1/results/temporal/node_timeseries/` contiene 40 Parquets.
- `surrogate_cnn_metrics.csv` referencia run IDs como `03460a9d...` que ya no existen en `runs`.
- `surrogate_lstm_metrics.csv` si referencia run IDs que existen en la DB actual.

Impacto:

- Es facil creer que un modelo corresponde al dataset actual cuando fue entrenado con otro.
- Comparaciones CNN/LSTM pueden mezclar generaciones de datos.
- Reproducibilidad debil.

Mejor practica:

- Agregar un `manifest.json` para artefactos temporales con:
  - `db_path`, hash DB o timestamp de export.
  - `run_ids`, `network_hashes`, rango Qx.
  - columnas temporales y estaticas.
  - commit git y parametros de entrenamiento.
- Limpiar o mover Parquets no registrados.
- Antes de inferir, validar que los run IDs del artefacto coinciden con la DB.

### 8. El `DEFAULT_INP_FILE` apunta a un archivo inexistente

Referencia: `swmm_resilience/config.py:29-34`, `swmm_resilience/main.py:48-54`

`DEFAULT_NETWORK_KEY = "chico_hydro-qx1"`, pero `DEFAULT_INP_FILE` apunta a:

`data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Steady.inp`

Ese archivo no existe. Tampoco existe el fallback `LEGACY_INP_FILE` en la raiz. El steady real esta en `data/networks/chico_steady/`.

Impacto:

- `python main.py` o predicciones por defecto pueden fallar.
- Usuarios pueden terminar entrenando/prediciendo con rutas manuales distintas a las constantes usadas por artefactos.

Mejor practica:

- Corregir el default a un `.inp` existente.
- Agregar test de configuracion: todas las rutas default criticas existen o tienen mensaje claro.

### 9. Validacion solo por `run_id` no prueba generalizacion a nuevas redes ni a nodos nuevos

Referencia: `swmm_resilience/ml/train.py:293-310`, `swmm_resilience/ml/train.py:731-805`, `swmm_resilience/ml/temporal/train_surrogate.py:79-99`

Agrupar por `run_id` evita que una misma corrida caiga en train/test, lo cual es bueno. Pero la DB actual tiene una sola red y los mismos nodos se repiten en todos los multipliers. El modelo puede memorizar caracteristicas por nodo/topologia y no se esta midiendo transferencia a otra red o geometria.

Impacto:

- Buenas metricas dentro de Chico no garantizan predicciones fiables en otro `.inp`.
- Riesgo alto para uso general de "nueva red".

Mejor practica:

- Cuando haya varias redes, agregar `GroupKFold` por `network_hash` o leave-one-network-out.
- Reportar dos niveles: interpolacion de multipliers en la misma red y generalizacion a red no vista.
- Evitar publicar metricas globales sin aclarar el tipo de generalizacion.

### 10. Padding con ceros en secuencias temporales sin mascara

Referencia: `swmm_resilience/ml/temporal/dataset.py:318-322`, `swmm_resilience/ml/temporal/dataset.py:460-464`

Las secuencias se rellenan con ceros hasta `T_max`. Luego `StandardScaler` se entrena sobre el arreglo completo, incluyendo los ceros de padding. El modelo no recibe mascara de longitud.

Impacto:

- El modelo puede aprender duracion/padding como senal artificial.
- Los ceros pueden parecer caudal real bajo despues del escalado.
- Redes con duraciones distintas pueden degradar la inferencia.

Mejor practica:

- Usar longitudes + mascara.
- Para CNN, aplicar masked pooling.
- Alternativa simple: construir una grilla temporal fija comun y truncar/llenar con valores fisicamente definidos antes del split.

## Hallazgos medios

### 11. `predict_from_parquet()` trata regresion temporal como probabilidad

Referencia: `swmm_resilience/ml/temporal/predict.py:63-90`, `swmm_resilience/ml/temporal/predict.py:170-188`

Si `task="regression"`, se carga `cnn_regressor`, pero el output se mete en columnas llamadas `max_flood_prob`, `mean_flood_prob` y se umbraliza con `>= 0.5`.

Impacto:

- Resultados absurdos si se usa la tarea de regresion.
- Mapas y conteos pueden interpretar L/s como probabilidad.

Mejor practica:

- Separar funciones de inferencia: `predict_temporal_classification()` y `predict_temporal_regression()`.
- Cambiar nombres de columnas segun tarea y bloquear `plot_prediction_map()` si no recibe probabilidades.

### 12. El mapa de prediccion usa umbral por cuantiles para confusion matrix

Referencia: `swmm_resilience/ml/temporal/predict.py:241-249`

La funcion define `predicted_at_risk` como el top 25% de riesgo predicho por defecto, no como `prob >= 0.5` ni umbral calibrado. Esto puede hacer que siempre aparezcan falsos positivos/verdaderos positivos incluso si todas las probabilidades son bajas.

Impacto:

- Estadisticas visuales no representan el comportamiento real del clasificador.
- Puede ocultar modelos subentrenados.

Mejor practica:

- Mostrar dos cosas separadas: ranking top-k y clasificacion por umbral calibrado.
- Guardar el umbral seleccionado en el artefacto.

### 13. No hay semillas deterministicas para PyTorch/DataLoader

Referencia: `swmm_resilience/ml/temporal/train_surrogate.py:131-142`, `swmm_resilience/ml/temporal/train_cnn.py:130-144`, `swmm_resilience/ml/temporal/compare_surrogate.py:87-101`

Los modelos tabulares tienen `ML_RANDOM_STATE`; los modelos PyTorch no fijan `torch.manual_seed`, `np.random.seed`, generador del DataLoader ni flags deterministas.

Impacto:

- Pesos y metricas cambian entre ejecuciones.
- Dificulta saber si una mejora viene del codigo o del azar.

Mejor practica:

- Centralizar `ML_RANDOM_STATE`.
- Fijar semillas de NumPy/Torch/DataLoader.
- Guardar seed en manifest.

### 14. Falta calibracion de probabilidades y umbral por costo hidraulico

Referencia: `swmm_resilience/ml/temporal/train_surrogate.py:170-173`, `swmm_resilience/ml/predict_tabular.py:192-196`

Se usa umbral fijo 0.5. En inundaciones, falsos negativos suelen costar mas que falsos positivos, y la clase positiva cambia mucho por Qx.

Impacto:

- Recall puede ser insuficiente en rangos peligrosos.
- Probabilidades de XGBoost/SVC/CNN pueden no estar calibradas.

Mejor practica:

- Reportar precision-recall y elegir umbral por objetivo: recall minimo, costo esperado o F-beta.
- Calibrar probabilidades (`CalibratedClassifierCV` para tabular; temperature scaling/isotonic para neural).

### 15. Features categoricas importantes se pierden en la ruta tabular

Referencia: `swmm_resilience/ml/preprocessing.py:53-56`, `swmm_resilience/analysis/dataset.py:22-41`

La ruta tabular toma solo columnas numericas, por lo que `node_type`, `network_file`, `scenario_type`, `spatial_pattern` desaparecen. Algunas se deben excluir, pero `node_type` puede ser fisicamente relevante.

Impacto:

- El modelo no distingue junction/outfall/storage salvo indirectamente por otros features.
- En nuevas redes, perder tipo de nodo puede reducir robustez.

Mejor practica:

- Usar `ColumnTransformer` con one-hot para `node_type`.
- Mantener metadata no predictiva separada, no depender de `select_dtypes`.

### 16. PCA esta activado por defecto para arboles tambien

Referencia: `swmm_resilience/config.py:76-80`, `swmm_resilience/ml/train.py:269-289`

`ML_USE_PCA=True` agrega `StandardScaler` + `PCA` a todos los modelos, incluido XGBoost. Esto reduce interpretabilidad y puede quitar estructura no lineal que los arboles aprovechan. No es bug matematico, pero es una decision riesgosa para features hidraulicas tabulares.

Impacto:

- Menor interpretabilidad de features fisicas.
- Posible perdida de desempeno en XGBoost.

Mejor practica:

- Comparar XGBoost con features crudas vs PCA.
- Dejar PCA para modelos lineales/SVR si realmente mejora CV.
- Guardar feature importance/permutation importance en espacio original.

### 17. Falta control de extrapolacion por rango Qx

Evidencia DB: multipliers entrenados Qx1.00-Qx6.75.

`predict_steady_flows_from_inp()` y `predict_surrogate_from_multiplier()` aceptan cualquier multiplier >= 1.0. No advierten si el usuario pide Qx fuera del rango entrenado.

Impacto:

- Predicciones fuera de dominio pueden parecer validas.
- Muy critico en eventos extremos.

Mejor practica:

- Guardar `min_multiplier`/`max_multiplier` en manifest.
- Emitir warning fuerte o requerir confirmacion al extrapolar.
- Reportar incertidumbre o "out of training domain".

## Hallazgos bajos / deuda tecnica que puede confundir modelos

### 18. Documentacion principal desactualizada

Referencia: `README.md` y `swmm_resilience/ml/temporal/README.md`

El README dice que `temporal_artifacts`, ventanas temporales, entrenamiento CNN/LSTM y predictor temporal no estan implementados, pero el codigo y artefactos ya existen.

Impacto:

- Usuarios pueden ejecutar flujos incorrectos.
- Mayor riesgo de entrenar con artefactos viejos o rutas equivocadas.

Mejor practica:

- Actualizar estado del proyecto y marcar claramente que partes son experimentales.

### 19. Nombres historicos de archivos mezclan volumen y caudal pico

Evidencia: `regression_comparison_flooding_volume_m3.csv` coexiste con `regression_comparison_peak_flooding_lps.csv`.

El codigo actual guarda `peak_flooding_lps`; el `.rpt` aun parsea volumen pero ya no se usa como target.

Impacto:

- Se pueden comparar metricas de targets distintos.
- Riesgo de usar un CSV viejo por error.

Mejor practica:

- Archivar artefactos legacy en carpeta `deprecated/`.
- Agregar version de target en manifest.

### 20. Valores faltantes estaticos se convierten a cero sin indicador

Referencia: `swmm_resilience/ml/temporal/dataset.py:139-141`, `swmm_resilience/ml/temporal/dataset.py:256-258`, `swmm_resilience/ml/temporal/dataset.py:381-382`

La DB actual tiene 58 nodos con algun valor faltante en features estaticas criticas (`full_depth`, capacidades o diametros agregados). El pipeline temporal convierte NaN a 0.0.

Impacto:

- Cero puede significar "no existe", "desconocido" o valor fisico real.
- El modelo puede aprender reglas incorrectas para fuentes/outfalls.

Mejor practica:

- Agregar indicadores `is_missing_*`.
- Imputar por tipo de nodo o categoria hidraulica.
- Revisar si los NaN representan ausencia fisica valida.

## Observaciones positivas

- La ruta tabular ya usa `GroupShuffleSplit` y `GroupKFold` por `run_id`, lo cual evita la fuga mas obvia entre filas de una misma corrida.
- Los pipelines tabulares encapsulan imputer/scaler/PCA/modelo, reduciendo fuga train/test.
- La inferencia CSV ya evita mezclar silenciosamente varias redes cuando hay `network_hash`/`network_file`.
- La validacion de geometria en `runner.py` bloquea junction depths sospechosos, una defensa importante para calidad hidraulica.
- El surrogate full usa solo `[total_inflow_lps, lateral_inflow_lps]` como serie temporal, que es una mejor direccion para inferencia sin SWMM que la ventana temporal con outputs hidraulicos.

## Prioridad recomendada

1. Corregir la semantica de escenarios parciales y `delta_inflows_lps`.
2. Separar formalmente features disponibles antes de SWMM vs outputs hidraulicos.
3. Reentrenar surrogates finales con todos los datos tras CV y manifestar artefactos.
4. Escalar/transformar `peak_flooding_lps` en redes neuronales y elegir modelo por metrica compuesta.
5. Limpiar artefactos temporales viejos y corregir `DEFAULT_INP_FILE`.
6. Agregar pruebas de no leakage, consistencia de manifest, rango Qx y reproducibilidad.

## Checklist de tests utiles

- `test_temporal_predictor_features_exclude_swmm_outputs`
- `test_timeseries_partial_scaling_does_not_change_unselected_nodes`
- `test_delta_inflows_lps_is_rejected_or_additive_not_multiplier`
- `test_default_inp_file_exists`
- `test_temporal_artifact_manifest_matches_db_run_ids`
- `test_surrogate_final_model_trains_on_all_groups_after_cv`
- `test_prediction_warns_outside_training_multiplier_range`
- `test_temporal_regression_inference_does_not_emit_probability_columns`

## Safe stabilization implementation status

This audit is being addressed by `docs/superpowers/plans/2026-05-31-safe-stabilization-pass.md`.

Covered in the first pass:

- Findings 1 and 11: temporal feature contracts and task-specific temporal inference schemas.
- Finding 2: selected-node timeseries scaling isolation.
- Finding 3: rejection of ambiguous `delta_inflows_lps`.
- Findings 4, 5, 6, 7, 13, and 17: deterministic surrogate training, final full-data fit, manifests, target transform, and inference guardrails.
- Finding 8: default `.inp` path correction.

Deferred:

- Findings 9, 14, 15, 16, 18, 19, and 20 remain planned for later model-quality and documentation passes.
