# SWMM Resilience

Herramienta para ejecutar escenarios SWMM, guardar resultados hidraulicos en
SQLite, exportar datasets tabulares para ML y empezar la transicion hacia un
dataset temporal para CNN/LSTM.

El proyecto hoy tiene dos rutas:

- **Ruta tabular actual:** simulaciones SWMM -> SQLite -> `dataset_ml.csv` ->
  modelos tabulares.
- **Ruta temporal en construccion:** simulaciones SWMM -> Parquet por timestep
  y nodo -> futuras ventanas temporales -> CNN/LSTM.

## Estado Actual

Ya esta implementado:

- ejecucion de corridas con PySWMM
- manipulacion de `.inp` con `swmm-api`
- escalado de caudales/hidrogramas segun modo de escenario
- almacenamiento de resultados en SQLite
- exportacion de `dataset_ml.csv`
- entrenamiento tabular con artefactos persistidos
- inferencia ML desde `.inp` y desde CSV legacy
- persistencia temporal MVP: `node_timeseries` por corrida en Parquet

Todavia no esta implementado:

- registro de Parquet temporales en SQLite (`temporal_artifacts`)
- construccion de ventanas temporales
- entrenamiento CNN/LSTM real con PyTorch
- predictor temporal operativo

## Estructura Principal

```text
app.py
main.py
view_db.py
requirements.txt
swmm_resilience/
  config.py
  main.py
  simulation/
    runner.py
    swmm_api_io.py
  database/
    schema.py
    repository.py
  analysis/
    dataset.py
    eda.py
  ml/
    train.py
    predict_from_inp.py
    predict_tabular.py
    temporal/
      dataset.py
      schemas.py
      train_cnn.py
      predict.py
data/
  training/
    swmm_resilience.db
  networks/
    <red>/
      archivo.inp
      results/
        dataset_ml.csv
        temporal/
          node_timeseries/
            run_<run_id>.parquet
```

## Instalacion

```bash
pip install -r requirements.txt
```

Dependencias importantes:

- `pyswmm`: ejecuta la simulacion hidraulica
- `swmm-api`: lee/modifica archivos `.inp` y ayuda con reportes
- `pandas`, `numpy`, `scikit-learn`, `xgboost`: dataset y ML tabular
- `pyarrow`: escritura de Parquet temporal

## Ejecucion Rapida

Ejecutar el pipeline por consola:

```bash
python main.py
```

Abrir la aplicacion local:

```bash
python app.py
```

Entrenar y comparar modelos tabulares:

```bash
python -m swmm_resilience.ml.train
```

Ver el scaffold temporal:

```bash
python -m swmm_resilience.ml.temporal.train_cnn
```

## Flujo Hidraulico

Por cada corrida:

```text
.inp original
  -> swmm_api_io.write_scaled_inp()
  -> PySWMM Simulation(temp.inp)
  -> .rpt/.out generados por SWMM
  -> swmm_api_io.read_node_flooding_summary(.rpt)
  -> runner.py calcula metricas adicionales
  -> SQLite + dataset_ml.csv + node_timeseries Parquet
```

`swmm-api` queda encapsulado en:

```text
swmm_resilience/simulation/swmm_api_io.py
```

`PySWMM` se usa en:

```text
swmm_resilience/simulation/runner.py
```

## Modos De Escenario

Los modos principales estan en `swmm_resilience/config.py`:

```python
SCENARIO_MODE_TIMESERIES = "timeseries"
SCENARIO_MODE_STEADY = "steady"
```

- `timeseries`: escala hidrogramas internos definidos en `[TIMESERIES]`.
- `steady`: escala caudales base definidos en `[INFLOWS]`.

La lista default de factores esta en:

```python
DEFAULT_INFLOW_MULTIPLIERS = _decimal_range(1.0, 2.5, 0.5)
```

Ejemplo:

```text
1.0, 1.5, 2.0
```

significa usar el caudal base, luego 1.5 veces y luego 2 veces.

## Salidas

Base central:

```text
data/training/swmm_resilience.db
```

Dataset tabular por red:

```text
data/networks/<red>/results/dataset_ml.csv
```

Series temporales por corrida:

```text
data/networks/<red>/results/temporal/node_timeseries/run_<run_id>.parquet
```

Artefactos ML tabulares:

```text
data/networks/<red>/results/model_artifacts/
```

## Dataset Tabular

`dataset_ml.csv` es una vista plana para modelos tabulares. Tiene una fila por:

```text
run_id + node_id
```

Combina:

- metadata de corrida: `run_id`, `network_hash`, `network_file`,
  `inflow_multiplier`, `scenario_type`, `spatial_pattern`
- topologia estatica del nodo: profundidad, cotas, grado de conectividad,
  diametros, pendientes y capacidades agregadas
- resultados agregados por nodo: profundidad maxima, flooding, volumen,
  duracion y caudales de salida

El entrenamiento tabular usa como targets:

- clasificacion: `flooded`
- regresion: `flooding_volume_m3`

La validacion se agrupa por `run_id` para evitar que filas de una misma corrida
queden repartidas entre entrenamiento y prueba.

## Prediccion ML

La ruta recomendada para inferencia escalable es desde `.inp`:

```text
archivo .inp + artefactos entrenados -> predict_from_inp.py
```

La interfaz tambien conserva una ruta desde CSV para compatibilidad y
depuracion:

```text
dataset_ml.csv -> predict_tabular.py
```

Ese flujo CSV debe tratarse como **legacy**. La direccion objetivo del proyecto
es que redes nuevas se evaluen desde el archivo `.inp`, no construyendo CSV
auxiliares manuales.

## Fase Temporal Actual

La Fase 0 quedo definida asi:

- `failed_now = 1` cuando `flooding_lps > 0`
- todas las corridas generan `node_timeseries`, incluyendo steady
- frecuencia objetivo para ML: `5 minutos`
- ventana historica inicial: `20 minutos`
- horizonte inicial de prediccion: `5 minutos`
- avance entre ventanas: `5 minutos`
- `link_timeseries` queda fuera del MVP
- `head_m` no se guarda porque se deriva de `invert_elev_m + depth_m`
- PyTorch entra solo cuando existan ventanas temporales listas

La Fase 1 ya esta implementada: cada corrida genera un Parquet con columnas:

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

La siguiente fase es registrar esos archivos en SQLite con una tabla
`temporal_artifacts`.

## Base De Datos

Tablas principales:

- `runs`: una fila por corrida
- `run_inputs`: entradas aplicadas por nodo
- `network_nodes`: atributos estaticos por nodo
- `network_links`: atributos estaticos por enlace
- `node_results`: resultados agregados por nodo
- `link_results`: resultados agregados por enlace
- `run_summary`: resumen ejecutivo de cada corrida

El visor local se abre con:

```bash
python view_db.py
```

## Archivos Clave

- `swmm_resilience/config.py`: rutas, factores, modos de escenario y parametros ML.
- `swmm_resilience/main.py`: orquesta corridas, BD, CSV y Parquet temporal.
- `swmm_resilience/simulation/runner.py`: ejecuta PySWMM y calcula resultados.
- `swmm_resilience/simulation/swmm_api_io.py`: lectura/escritura estructurada de SWMM.
- `swmm_resilience/database/schema.py`: esquema SQLite y migraciones.
- `swmm_resilience/database/repository.py`: persistencia en SQLite.
- `swmm_resilience/analysis/dataset.py`: exporta `dataset_ml.csv`.
- `swmm_resilience/ml/train.py`: entrena modelos tabulares y guarda artefactos.
- `swmm_resilience/ml/predict_from_inp.py`: inferencia desde `.inp`.
- `swmm_resilience/ml/predict_tabular.py`: inferencia CSV legacy.
- `swmm_resilience/ml/temporal/dataset.py`: helpers del dataset temporal.

## Documentos De Apoyo

- `DOCUMENTACION_COMPLETA_PROYECTO.md`: explicacion integral del proyecto, datos, ML, PCA y ruta temporal.
- `QUICKSTART.md`: guia rapida.
- `PLAN_TEMPORAL_LSTM_CNN.md`: plan temporal por fases.
- `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md`: diccionario de datos para ML.
- `REVISION_DETALLADA_DELTA_Y_ESCENARIOS_PARCIALES.md`: revision de deltas y escenarios.

## Escalabilidad Multired

Para que el modelo generalice a varias redes:

- entrenar con muchas redes, no solo con Chico
- mantener features fisicas comparables entre redes
- separar entrenamiento y prueba por `network_hash`
- revisar metricas por red, no solo metricas globales
- evitar depender de IDs de nodos o nombres locales
- validar redes nuevas desde `.inp`

La meta final es que el modelo aprenda patrones hidraulicos transferibles y no
solo memorice una red especifica.
