# SWMM Resilience

Pipeline spec v4 para generar escenarios SWMM, construir un dataset tabular,
entrenar modelos de inundacion por nodo y producir metricas, mapas e
inferencia rapida sin volver a correr SWMM.

El flujo principal se configura desde `config.yaml`.

## Pipeline Spec V4

```bash
python main.py
python main.py --skip-extraction
python main.py --only-ml
python main.py --only-maps
python main.py --predict --factor 3.5
```

Salidas principales:

- `data/training/dataset_final.csv`
- `outputs/models/classifier.joblib`
- `outputs/models/regressor.joblib`
- `outputs/models/training_inp_hash.txt`
- `outputs/metrics/*.json`
- `outputs/maps/*.png`

## Modos De Uso

- `python main.py`: ejecuta el pipeline completo desde `config.yaml`.
- `python main.py --skip-extraction`: reutiliza `dataset.output_path` y salta
  extraccion/simulacion.
- `python main.py --skip-simulation --skip-extraction`: modo honesto para
  reutilizar CSV existente sin intentar reconstruir reportes `.rpt`.
- `python main.py --only-ml`: entrena, evalua y genera importancia de variables
  desde `data/training/dataset_final.csv`.
- `python main.py --only-maps`: regenera mapas desde el CSV y la red del config.
- `python main.py --predict --factor 3.5`: predice nodos inundados y volumen
  usando los modelos guardados en `outputs/models`.

## Arquitectura Actual

```text
main.py
config.yaml
swmm_resilience/
  config.py
  simulation/
    batch.py
    runner.py
  extraction/
    static_features.py
    dynamic_features.py
    labels.py
    assembler.py
  dataset/
    validator.py
  ml/
    trainer.py
    evaluator.py
    feature_importance.py
    predict.py
  visualization/
    flood_map.py
```

La version spec v4 no usa frontend desktop, SQLite ni los modulos temporales
legacy como flujo principal. Es un pipeline CLI basado en CSV, modelos joblib,
metricas JSON y mapas PNG.

## Modelos Y Metricas

El clasificador predice `inunda`. El regresor predice `vol_inundacion_m3` solo
para nodos inundados; se entrena en espacio `log1p` y las predicciones se
devuelven a m3 con `expm1`.

Contrato de features v2 (junio 2026): los modelos usan 15 features;
`factor_mult` ya NO es entrada del modelo (es un atributo global de escenario
sin definicion valida para hidrogramas arbitrarios) y queda en el dataset solo
como metadato para LOSO y estratificacion. La señal dinamica entra via
`q_pico_nodo` y `q_pico_acum_escalado`.

Regla de etiquetado unificada (`swmm_resilience/extraction/labels.py`):
`inunda = vol >= flood_threshold_m3`, con umbral default 1.0 m3 (resolucion
del .rpt). Entrenamiento y validacion importan la misma funcion. Regla de
conciliacion: `inunda_pred` la decide el clasificador; `vol_pred_m3` se
reporta tal cual.

La evaluacion reporta:

- clasificador aislado
- regresor oracle con etiquetas reales para filtrar inundados
- sistema end-to-end con etiquetas predichas
- estratificacion por `factor_mult`

`metrics_regressor.json` incluye `nse` y `log_nse`. El `log_nse` se calcula en
espacio logaritmico sobre predicciones out-of-fold apiladas para evitar que
folds LOSO con muy pocos nodos inundados dominen la metrica global.

## Validacion Con Hidrogramas (CSV)

```bash
python main.py --evaluate-hydrographs DIR --base-inp PATH \
  --clf-path outputs/models/classifier.joblib \
  --reg-path outputs/models/regressor.joblib \
  [--flood-threshold M3] [--allow-inp-mismatch] [--out-dir DIR]
```

Comportamiento del harness:

- Guardas de validez: aborta si `FLOW_UNITS != LPS` o si el hash MD5 del
  `.inp` base no coincide con `training_inp_hash.txt` (forzable con
  `--allow-inp-mismatch`); advierte con `ALLOW_PONDING` activo y con error de
  continuidad SWMM > 5% por escenario.
- Drenaje post-evento: cada serie se extiende con caudal 0 durante
  `validation.drain_down_hours` (config, default 6.0) para no truncar el
  volumen de inundacion; advierte si el CSV no termina cerca de cero.
- La comparacion cubre TODAS las junctions de la red (no solo los nodos con
  inflow del CSV); los modelos y las features estaticas se cargan una sola
  vez por batch (`ScenarioPredictor`).
- Marca `extrapolated` por nodo cuando el pico del escenario sale del rango
  de entrenamiento `base_inflow x [factor_min, factor_max]`.

Salidas en `--out-dir`:

- `comparison_summary.csv`: por nodo y escenario.
- `scenario_totals.csv`: volumen total de red SWMM vs ML por escenario, con
  error absoluto/porcentual y `n_extrapolated`.
- `timings.csv`: `t_write_inp_s, t_swmm_s, t_parse_rpt_s, t_features_s,
  t_inference_s, speedup` por escenario, mas `t_model_load_s` y
  `t_static_features_s` (costos unicos del batch) y `device`.
- `metrics_per_scenario.csv`: clasificacion + CSI + MAE/RMSE condicionales a
  nodos inundados + error de continuidad, por escenario.
- `plots/totals_comparison.png`: barras pareadas SWMM vs ML por escenario.
- La consola imprime totales por escenario, PR-AUC y speed-up medio.

## Instalacion

```bash
pip install -r requirements.txt
```

Para regenerar desde cero el dataset, los modelos, las metricas y los mapas
con las versiones validadas del proyecto:

```bash
./scripts/reproduce_results.sh
```

En Windows PowerShell:

```powershell
.\scripts\reproduce_results.ps1
```

El script requiere Python 3.13, crea un entorno virtual local en `.venv`,
instala las dependencias fijadas en `requirements.txt`, ejecuta las pruebas y
corre el pipeline completo definido por `config.yaml`. Los resultados se
generan localmente y no se versionan.

Dependencias principales:

- `pyswmm` y `swmm-api` para SWMM
- `pandas`, `numpy`, `scikit-learn` y `xgboost` para ML
- `matplotlib` y `networkx` para mapas
- `pytest` para verificacion

## Verificacion

```bash
python -m pytest tests -v
python -m compileall main.py swmm_resilience
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
- regresion: `peak_flooding_lps`

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

### Metrica de inundacion por nodo

`peak_flooding_lps` reemplaza al antiguo `flooding_volume_m3`. Es el **caudal
maximo instantaneo de desborde** (lps) observado en cada nodo a lo largo de
la simulacion, es decir, el peor valor puntual de `node.flooding` durante
todos los pasos de tiempo.

Por que el pico y no el volumen total:

- El volumen total acumula todo el desborde durante el evento; depende del
  paso de tiempo y de cuanto dura la lluvia, lo que introduce variabilidad
  no relacionada con la capacidad hidraulica del nodo.
- El caudal pico es mas interpretable como indicador de severidad: refleja
  directamente que tan saturado quedo el nodo en su momento critico.
- Para clasificacion (`flooded`), ambas metricas son equivalentes: si el
  pico > 0, el nodo desbordo.

`flooding_duration_min` sigue siendo la duracion total del desborde, leida
desde el `.rpt` generado por SWMM cuando `USE_SWMM_API_RPT_RESULTS = True`.

En `run_summary`, `total_peak_flooding_lps` es la suma de los picos de todos
los nodos (no es un pico de red, sino un indicador escalar del nivel de
saturacion global del sistema en esa corrida).

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
