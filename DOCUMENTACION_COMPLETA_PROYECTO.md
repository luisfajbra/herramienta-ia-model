# Documentacion completa del proyecto SWMM Resilience

Este documento explica de forma integral que hace el proyecto, cual es la
logica hidraulica y de datos, que informacion se guarda, que informacion se
usa para entrenar modelos, que columnas se excluyen y por que, como funciona la
reduccion de variables con PCA, que metricas salen de las pruebas, y cual es el
camino acordado para llegar a modelos temporales CNN/LSTM.

La intencion es que este archivo sirva como documento maestro del proyecto.

## 1. Objetivo general

El objetivo del proyecto es construir una herramienta escalable para evaluar
redes de alcantarillado modeladas en SWMM y usar los resultados hidraulicos
para entrenar modelos de machine learning.

En el corto plazo, el proyecto busca:

- ejecutar corridas SWMM automaticamente
- escalar caudales o hidrogramas dentro del archivo `.inp`
- guardar resultados hidraulicos en SQLite
- exportar un dataset tabular para ML
- entrenar modelos tabulares de clasificacion y regresion
- predecir resultados de nuevas redes desde un `.inp` usando modelos ya
  entrenados

En el largo plazo, el proyecto busca:

- entrenar modelos con muchas redes y muchos escenarios
- generalizar a redes de alcantarillado no vistas
- pasar de modelos tabulares a modelos temporales
- construir una capa de series por timestep
- entrenar CNN 1D y LSTM para prediccion temprana de falla
- usar el modelo como surrogate para evaluar alternativas sin correr SWMM en
  todos los casos

El objetivo final no es reemplazar SWMM. La idea es usar SWMM para generar datos
confiables, entrenar modelos que aprendan patrones hidraulicos, y luego usar
esos modelos para evaluacion rapida, exploracion de escenarios y apoyo a la
toma de decisiones.

## 2. Vision conceptual del proyecto

El proyecto tiene dos caminos de aprendizaje:

### 2.1. Camino tabular actual

Este es el flujo que ya funciona:

```text
.inp de SWMM
  -> simulacion con PySWMM
  -> resultados en SQLite
  -> exportacion a dataset_ml.csv
  -> entrenamiento de modelos tabulares
  -> artefactos guardados
  -> inferencia desde .inp o CSV legacy
```

Este camino trabaja con una fila por `run_id + node_id`. Es decir, cada fila
representa el resumen de lo que le paso a un nodo durante una corrida.

Sirve para:

- establecer una linea base de ML
- comparar modelos como Ridge, Lasso, SVR y XGBoost
- predecir si un nodo falla
- predecir volumen de flooding
- auditar datos hidraulicos agregados

### 2.2. Camino temporal CNN/LSTM

Este es el camino en construccion:

```text
.inp de SWMM
  -> simulacion con PySWMM
  -> series por timestep y nodo
  -> Parquet node_timeseries
  -> ventanas temporales
  -> Dataset/DataLoader de PyTorch
  -> CNN 1D / LSTM
  -> prediccion temprana de falla
```

Este camino no debe entrenar desde `dataset_ml.csv`, porque ese CSV es un
resumen agregado. CNN y LSTM necesitan secuencias temporales reales.

La regla central decidida es:

```text
CNN/LSTM se entrenan desde ventanas construidas a partir de node_timeseries,
no desde el CSV tabular resumen.
```

## 3. Arquitectura del programa

Nota de migracion: la version spec v4 usa un pipeline directo basado en
`config.yaml`, CSV, modelos joblib, metricas JSON y mapas PNG. La arquitectura
anterior con SQLite, app Tkinter y modulos temporales fue reemplazada en esta
rama de trabajo para alinear el codigo con `spec_tecnico_desarrollo.md`.

Los archivos principales son:

```text
app.py
main.py
view_db.py
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
    preprocessing.py
    train.py
    predict_from_inp.py
    predict_tabular.py
    temporal/
      dataset.py
      schemas.py
      train_cnn.py
      predict.py
```

### 3.1. Responsabilidad de cada capa

| Capa | Archivos | Responsabilidad |
|---|---|---|
| Configuracion | `config.py` | Rutas, modos de escenario, factores, parametros ML |
| Orquestacion | `swmm_resilience/main.py` | Ejecuta el flujo completo de simulacion a dataset |
| Simulacion | `simulation/runner.py` | Corre PySWMM y calcula resultados hidraulicos |
| SWMM I/O | `simulation/swmm_api_io.py` | Lee/modifica `.inp`, lee `.rpt`/`.out` con swmm-api |
| Base de datos | `database/schema.py`, `repository.py` | Crea tablas y guarda resultados |
| Dataset | `analysis/dataset.py` | Exporta `dataset_ml.csv` desde SQLite |
| ML tabular | `ml/preprocessing.py`, `ml/train.py` | Selecciona features, entrena, evalua y guarda modelos |
| Inferencia | `ml/predict_from_inp.py`, `ml/predict_tabular.py` | Predice desde `.inp` o CSV legacy |
| ML temporal | `ml/temporal/` | Guarda series, luego construira ventanas y modelos CNN/LSTM |

## 4. Flujo completo de una corrida

El flujo de una corrida es:

```text
1. El usuario selecciona un archivo .inp
2. Se calcula network_hash del archivo
3. Se extrae topologia estatica
4. Se crea una fila en runs
5. Si aplica, se escribe un .inp temporal escalado
6. PySWMM ejecuta la simulacion
7. Durante la simulacion se capturan metricas por timestep
8. Al final se guardan resultados agregados
9. Se lee el .rpt para corregir volumen y duracion de flooding
10. Se guardan tablas SQLite
11. Se exporta dataset_ml.csv
12. Se guarda node_timeseries en Parquet
```

## 5. Uso de swmm-api y PySWMM

El proyecto divide responsabilidades:

| Herramienta | Uso |
|---|---|
| `swmm-api` | Leer/modificar archivos `.inp`, escribir `.inp` temporal, leer reportes `.rpt` y preparar lectura `.out` |
| `PySWMM` | Ejecutar la simulacion y consultar objetos hidraulicos durante la corrida |

La decision importante es no mezclar `swmm-api` por todo el codigo. Todo su uso
debe estar encapsulado en:

```text
swmm_resilience/simulation/swmm_api_io.py
```

Esto reduce riesgo porque `swmm-api` no es una API completamente estable. Si en
el futuro cambia, el impacto deberia concentrarse en ese archivo.

## 6. Modos de escenario

El proyecto reconoce dos modos principales:

```python
SCENARIO_MODE_TIMESERIES = "timeseries"
SCENARIO_MODE_STEADY = "steady"
```

### 6.1. `timeseries`

Se usa cuando la red tiene hidrogramas internos definidos en el `.inp`, por
ejemplo mediante `[TIMESERIES]`.

La herramienta escala esos hidrogramas con un multiplicador.

Ejemplo:

```text
inflow_multiplier = 2.0
```

significa que el hidrograma embebido se multiplica por 2.

### 6.2. `steady`

Se usa cuando los caudales estan definidos como baseline en `[INFLOWS]`.

El multiplicador escala ese caudal base.

### 6.3. Factores de corrida

Los factores por defecto estan en `config.py`:

```python
DEFAULT_INFLOW_MULTIPLIERS = _decimal_range(1.0, 2.5, 0.5)
```

Esto genera:

```text
1.0, 1.5, 2.0
```

## 7. Datos que se guardan

El proyecto guarda informacion en tres niveles:

1. SQLite
2. CSV tabular
3. Parquet temporal

### 7.1. SQLite

La base principal es:

```text
data/training/swmm_resilience.db
```

Tablas actuales:

- `runs`
- `run_inputs`
- `network_nodes`
- `network_links`
- `node_results`
- `link_results`
- `run_summary`

### 7.2. CSV tabular

Se exporta por red:

```text
data/networks/<red>/results/dataset_ml.csv
```

Este CSV es la entrada actual para los modelos tabulares.

### 7.3. Parquet temporal

Fase 1 ya guarda:

```text
data/networks/<red>/results/temporal/node_timeseries/run_<run_id>.parquet
```

Este archivo contiene la serie cruda por nodo y timestep. Todavia no se usa
para entrenar modelos, pero es la base para CNN/LSTM.

## 8. Diccionario de tablas SQLite

### 8.1. `runs`

Una fila por corrida.

| Columna | Significado | Fuente | Uso |
|---|---|---|---|
| `run_id` | Identificador unico | Codigo | Join principal |
| `network_file` | Archivo `.inp` usado | Usuario/codigo | Trazabilidad |
| `network_hash` | Hash del `.inp` | Codigo | Identificar red |
| `scenario_type` | Tipo de escenario guardado | Config/UI | Metadata |
| `spatial_pattern` | Patron espacial | Config/UI | Metadata |
| `delta_inflow_lps` | Metadata historica/global | Codigo | Legacy/metadata |
| `inflow_multiplier` | Factor global de corrida | Config/UI | Feature tabular |
| `executed_at` | Fecha de ejecucion | SQLite | Auditoria |
| `status` | Estado de corrida | Codigo | Filtrado |

Decision importante:

- `inflow_multiplier` es el control global correcto del escenario.
- `delta_inflow_lps` no debe interpretarse automaticamente como caudal real por
  nodo.

### 8.2. `network_nodes`

Una fila por nodo y red.

| Columna | Significado | Uso potencial |
|---|---|---|
| `network_hash` | Red a la que pertenece | Join |
| `node_uid` | ID del nodo | Join |
| `invert_elev_m` | Cota de fondo | Feature |
| `full_depth_m` | Profundidad total | Feature |
| `base_inflow_lps` | Caudal base en `.inp` | Feature |
| `node_type` | junction/outfall/storage/etc. | Metadata |
| `in_degree` | Links que entran | Excluido hoy |
| `out_degree` | Links que salen | Excluido hoy |
| `upstream_pipes_count` | Numero de tuberias aguas arriba | Feature |
| `upstream_diam_max_m` | Diametro maximo aguas arriba | Feature |
| `upstream_diam_min_m` | Diametro minimo aguas arriba | Feature |
| `upstream_diam_avg_m` | Diametro promedio aguas arriba | Excluido hoy |
| `upstream_slope_avg` | Pendiente promedio aguas arriba | Feature |
| `upstream_slope_max` | Pendiente maxima aguas arriba | Feature |
| `upstream_capacity_lps` | Capacidad teorica aguas arriba | Excluido hoy |
| `downstream_pipes_count` | Numero de tuberias aguas abajo | Feature |
| `downstream_diam_max_m` | Diametro maximo aguas abajo | Feature |
| `downstream_diam_min_m` | Diametro minimo aguas abajo | Feature |
| `downstream_diam_avg_m` | Diametro promedio aguas abajo | Excluido hoy |
| `downstream_slope_avg` | Pendiente promedio aguas abajo | Feature |
| `downstream_slope_max` | Pendiente maxima aguas abajo | Feature |
| `downstream_capacity_lps` | Capacidad teorica aguas abajo | Excluido hoy |

Estas variables describen la geometria y conectividad local de cada nodo.

### 8.3. `network_links`

Una fila por link y red.

| Columna | Significado |
|---|---|
| `network_hash` | Red |
| `link_uid` | ID del link |
| `inlet_node` | Nodo de entrada |
| `outlet_node` | Nodo de salida |
| `link_type` | conduit/weir/orifice/pump |
| `diameter_m` | Diametro o geometria principal |
| `length_m` | Longitud |
| `roughness` | Rugosidad |
| `slope_m_per_m` | Pendiente calculada |
| `full_flow_capacity_lps` | Capacidad teorica a flujo lleno |

Esta tabla no entra directamente al modelo tabular actual, pero sus agregados
por nodo si entran mediante `network_nodes`.

### 8.4. `run_inputs`

Registra entradas aplicadas por nodo.

| Columna | Significado |
|---|---|
| `input_id` | ID del registro |
| `run_id` | Corrida |
| `delta_inflow_lps` | Delta aplicado por nodo |
| `inflow_multiplier` | Factor de la corrida |
| `node_uid` | Nodo |

Decision importante:

- Esta tabla no debe verse como descripcion completa del hidrograma real cuando
  el input vive dentro del `.inp`.
- En el futuro debe evolucionar a una tabla de perturbaciones o metadata de
  entrada, no a una reconstruccion completa del hidrograma.

### 8.5. `node_results`

Una fila por nodo y corrida.

| Columna | Significado | Fuente |
|---|---|---|
| `result_id` | ID del resultado | Codigo |
| `run_id` | Corrida | Codigo |
| `delta_inflow_lps` | Delta real por nodo | Codigo |
| `inflow_multiplier` | Factor de corrida | Config/UI |
| `node_id` | Nodo | PySWMM |
| `flooded` | Si el nodo tuvo flooding | PySWMM/.rpt |
| `flooding_volume_m3` | Volumen de flooding | `.rpt` preferido |
| `flooding_duration_min` | Duracion de flooding | `.rpt` preferido |
| `max_depth_m` | Profundidad maxima | PySWMM |
| `max_depth_ratio` | `max_depth_m / full_depth_m` | Codigo |
| `time_to_peak_min` | Minuto de profundidad maxima | PySWMM/codigo |
| `depth_rate_m_per_min` | Tasa maxima de crecimiento de profundidad | Codigo |
| `max_total_outflow_lps` | Pico de salida total | PySWMM timestep |
| `time_to_peak_outflow_min` | Minuto del pico de salida | Codigo |
| `downstream_link_peak_flows_lps_json` | Picos por link saliente | Codigo |

### 8.6. `link_results`

Una fila por link y corrida.

| Columna | Significado |
|---|---|
| `result_id` | ID del resultado |
| `run_id` | Corrida |
| `delta_inflow_lps` | `NULL` o metadata legacy para links |
| `inflow_multiplier` | Factor |
| `link_id` | Link |
| `max_flow_lps` | Caudal maximo |
| `max_velocity_mps` | Velocidad maxima |
| `max_depth_m` | Profundidad maxima |
| `max_capacity_ratio` | `max_flow_lps / full_flow_capacity_lps` |
| `surcharged` | Si estuvo a flujo lleno |
| `time_full_flow_hrs` | Tiempo a flujo lleno |

Decision importante:

- El caudal de entrada se aplica a nodos, no a links.
- Por eso `delta_inflow_lps` en links no es una variable fisica relevante.

### 8.7. `run_summary`

Una fila por corrida.

| Columna | Significado |
|---|---|
| `summary_id` | ID |
| `run_id` | Corrida |
| `inflow_multiplier` | Factor |
| `total_nodes` | Nodos evaluados |
| `failed_nodes_count` | Nodos con flooding |
| `total_flooding_volume_m3` | Volumen total |
| `pct_flooded_nodes` | Porcentaje de nodos fallidos |
| `time_to_first_flood_min` | Primer minuto con flooding |
| `resilience_index` | `1 - failed_nodes_count / total_nodes` |

## 9. Dataset de entrenamiento tabular

El modelo tabular no entrena directamente desde SQLite. Primero se exporta:

```text
dataset_ml.csv
```

El exportador esta en:

```text
swmm_resilience/analysis/dataset.py
```

Actualmente combina:

- `runs`
- `network_nodes`
- `node_results`

No usa directamente:

- `run_inputs`
- `network_links`
- `link_results`
- `run_summary`

Estas tablas se conservan por auditoria, analisis y futuras mejoras, pero no
alimentan directamente el entrenamiento tabular actual.

## 10. Columnas exportadas al dataset y significado

| Columna | Origen | Significado | Rol actual |
|---|---|---|---|
| `run_id` | `node_results` | Corrida | Grupo, no feature |
| `node_id` | `node_results` | Nodo | ID, no feature |
| `network_hash` | `runs` | Red | Trazabilidad |
| `network_file` | `runs` | Archivo `.inp` | Trazabilidad |
| `inflow_multiplier` | `runs` | Factor de corrida | Feature |
| `scenario_type` | `runs` | Tipo de escenario | Metadata |
| `spatial_pattern` | `runs` | Patron espacial | Metadata |
| `invert_elev_m` | `network_nodes` | Cota fondo | Feature |
| `full_depth_m` | `network_nodes` | Profundidad total | Feature |
| `base_inflow_lps` | `network_nodes` | Caudal base | Feature |
| `node_type` | `network_nodes` | Tipo de nodo | Exportada, no feature |
| `in_degree` | `network_nodes` | Links entrantes | Excluida hoy |
| `out_degree` | `network_nodes` | Links salientes | Excluida hoy |
| `upstream_pipes_count` | `network_nodes` | Numero de pipes arriba | Feature |
| `upstream_diam_max_m` | `network_nodes` | Diametro max arriba | Feature |
| `upstream_diam_min_m` | `network_nodes` | Diametro min arriba | Feature |
| `upstream_diam_avg_m` | `network_nodes` | Diametro promedio arriba | Excluida hoy |
| `upstream_slope_avg` | `network_nodes` | Pendiente promedio arriba | Feature |
| `upstream_slope_max` | `network_nodes` | Pendiente maxima arriba | Feature |
| `upstream_capacity_lps` | `network_nodes` | Capacidad total arriba | Excluida hoy |
| `downstream_pipes_count` | `network_nodes` | Numero de pipes abajo | Feature |
| `downstream_diam_max_m` | `network_nodes` | Diametro max abajo | Feature |
| `downstream_diam_min_m` | `network_nodes` | Diametro min abajo | Feature |
| `downstream_diam_avg_m` | `network_nodes` | Diametro promedio abajo | Excluida hoy |
| `downstream_slope_avg` | `network_nodes` | Pendiente promedio abajo | Feature |
| `downstream_slope_max` | `network_nodes` | Pendiente maxima abajo | Feature |
| `downstream_capacity_lps` | `network_nodes` | Capacidad total abajo | Excluida hoy |
| `max_depth_m` | `node_results` | Profundidad maxima | Excluida por leakage |
| `max_depth_ratio` | `node_results` | Profundidad relativa maxima | Excluida por leakage |
| `time_to_peak_min` | `node_results` | Momento de profundidad maxima | Excluida por leakage |
| `depth_rate_m_per_min` | `node_results` | Tasa maxima de profundidad | Excluida por leakage |
| `max_total_outflow_lps` | `node_results` | Salida total maxima | Excluida por leakage |
| `time_to_peak_outflow_min` | `node_results` | Momento de salida maxima | Excluida por leakage |
| `downstream_link_peak_flows_lps_json` | `node_results` | JSON de picos por link | No numerica/no feature |
| `flooded` | `node_results` | Nodo fallo o no | Target clasificacion |
| `flooding_volume_m3` | `node_results` | Volumen de flooding | Target regresion |
| `flooding_duration_min` | `node_results` | Duracion de flooding | Excluida por leakage |

## 11. Que datos se excluyen y por que

El archivo `config.py` define:

```python
ML_DROP_COLUMNS = [...]
```

Estas columnas no se borran de la base ni necesariamente del CSV. Se excluyen
del entrenamiento cuando se construye `X`.

### 11.1. Identificadores

Se excluyen:

- `run_id`
- `node_id`

Motivo:

- Son llaves, no propiedades fisicas.
- Si entraran al modelo, podria memorizar corridas o nodos.

### 11.2. Metadata no numerica o de escenario

Se excluyen:

- `scenario_type`
- `spatial_pattern`

Motivo:

- Hoy no se codifican categoricamente.
- Sirven para reportes y filtros, no para el modelo tabular actual.

### 11.3. `delta_inflow_lps`

Se excluye:

- `delta_inflow_lps`

Motivo:

- Su semantica ha sido delicada en el proyecto.
- En corridas con multiplicador, el valor canonico global es
  `inflow_multiplier`.
- En escenarios parciales, un delta por nodo puede ser util, pero requiere una
  politica clara para evitar mezclar unidades o significados.

Decision actual:

- usar `inflow_multiplier` como feature global
- conservar `delta_inflow_lps` como dato de auditoria/legacy
- no usarlo como feature tabular principal por ahora

### 11.4. Promedios y capacidades excluidas

Se excluyen:

- `upstream_diam_avg_m`
- `downstream_diam_avg_m`
- `in_degree`
- `out_degree`
- `upstream_capacity_lps`
- `downstream_capacity_lps`

Motivo:

- Reducir redundancia en el espacio de features.
- Evitar un espacio demasiado correlacionado en la fase actual.
- Mantener un conjunto inicial mas estable y controlado.

No significa que sean inutiles. Son candidatas para revision futura.

### 11.5. Resultados de simulacion excluidos por leakage

Se excluyen:

- `flooding_duration_min`
- `max_depth_m`
- `max_depth_ratio`
- `time_to_peak_min`
- `depth_rate_m_per_min`
- `max_total_outflow_lps`
- `time_to_peak_outflow_min`

Motivo:

- Son resultados observados despues de correr la simulacion.
- Si se usan como input para predecir flooding o volumen, el modelo tendria
  informacion del futuro.
- Eso produciria metricas artificialmente buenas y no serviria para inferencia
  real.

Regla:

```text
Una feature de inferencia debe estar disponible antes de conocer el resultado.
```

## 12. Features actuales de entrenamiento

Luego de aplicar `ML_DROP_COLUMNS` y seleccionar solo columnas numericas, las
features principales actuales son:

- `inflow_multiplier`
- `invert_elev_m`
- `full_depth_m`
- `base_inflow_lps`
- `upstream_pipes_count`
- `upstream_diam_max_m`
- `upstream_diam_min_m`
- `upstream_slope_avg`
- `upstream_slope_max`
- `downstream_pipes_count`
- `downstream_diam_max_m`
- `downstream_diam_min_m`
- `downstream_slope_avg`
- `downstream_slope_max`

Dependiendo de futuras columnas y del CSV exportado, el selector puede cambiar
automaticamente, pero siempre bajo estas reglas:

- se elimina lo listado en `ML_DROP_COLUMNS`
- se conserva solo lo numerico
- se separa el target
- los nulos quedan para el `SimpleImputer` del pipeline

## 13. Targets actuales

### 13.1. Clasificacion

```python
ML_TARGET_CLASSIFICATION = "flooded"
```

El modelo responde:

```text
El nodo se inunda o no se inunda en esta corrida?
```

### 13.2. Regresion

```python
ML_TARGET_REGRESSION = "flooding_volume_m3"
```

El modelo responde:

```text
Cuanto volumen de flooding se espera en el nodo?
```

## 14. Preprocesamiento tabular

El preprocesamiento esta separado entre:

- seleccion de columnas
- pipeline de entrenamiento

### 14.1. Seleccion de columnas

Archivo:

```text
swmm_resilience/ml/preprocessing.py
```

Proceso:

1. Validar que el target exista.
2. Copiar el dataframe.
3. Eliminar columnas de `ML_DROP_COLUMNS`, excepto si esa columna es el target.
4. Seleccionar solo columnas numericas.
5. Separar `X` e `y`.
6. Rellenar nulos del target numerico con `0.0`.

### 14.2. Pipeline de modelo

Archivo:

```text
swmm_resilience/ml/train.py
```

Cada modelo se entrena dentro de un `Pipeline` de sklearn:

```text
SimpleImputer(strategy="median")
-> StandardScaler() si aplica
-> PCA() si esta activo
-> modelo
```

`SimpleImputer` rellena nulos de las features usando la mediana calculada en
entrenamiento.

`StandardScaler` estandariza variables para modelos sensibles a escala y para
PCA.

`PCA` reduce dimensionalidad si `ML_USE_PCA = True`.

## 15. Por que se usa PCA

La configuracion actual es:

```python
ML_USE_PCA = True
ML_PCA_COMPONENTS = 5
ML_PCA_SVD_SOLVER = "full"
```

PCA se usa por estas razones:

1. **Reducir redundancia entre variables hidraulicas.**
   Diametros, pendientes, capacidades y conteos pueden estar correlacionados.

2. **Estabilizar modelos con pocos datos.**
   Si el dataset aun no es grande, demasiadas features pueden llevar a
   sobreajuste.

3. **Crear un espacio compacto para comparar modelos.**
   Todos los modelos reciben el mismo espacio reducido.

4. **Disminuir ruido de variables repetidas o parcialmente redundantes.**

5. **Facilitar una linea base.**
   PCA no es necesariamente la solucion final, pero ayuda a tener una base de
   comparacion controlada.

Importante:

- PCA no sabe de hidraulica.
- PCA combina variables originales en componentes.
- Los componentes no son tan interpretables como las variables originales.
- Si se quiere interpretar directamente el efecto de cada variable, se puede
  poner `ML_USE_PCA = False`.

## 16. Validacion y particion de datos

La configuracion actual es:

```python
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5
ML_GROUP_COLUMN = "run_id"
ML_SPLIT_STRATEGY = "grouped_by_run_id"
```

La separacion train/test se hace con:

```text
GroupShuffleSplit
```

agrupando por:

```text
run_id
```

Motivo:

- Una corrida produce muchas filas, una por nodo.
- Si se separaran filas aleatoriamente, nodos de una misma corrida quedarian en
  train y test.
- Eso haria la evaluacion demasiado optimista.

La validacion cruzada usa:

```text
GroupKFold
```

tambien agrupando por `run_id`.

## 17. Modelos actuales

### 17.1. Regresion

Modelos disponibles:

- `ridge`
- `lasso`
- `svr_rbf`
- `xgboost` si `xgboost` esta instalado

Target:

```text
flooding_volume_m3
```

Modelo preferido para persistencia:

```text
xgboost
```

si esta disponible.

### 17.2. Clasificacion

Modelos disponibles:

- `logistic_regression`
- `svc_rbf`
- `xgboost_classifier` si `xgboost` esta instalado

Target:

```text
flooded
```

Modelo preferido para persistencia:

```text
xgboost_classifier
```

si esta disponible.

## 18. Parametros y metricas que salen de las pruebas

Cuando se ejecuta:

```bash
python -m swmm_resilience.ml.train
```

el programa imprime y guarda resultados con:

### 18.1. Parametros generales

- `Dataset`
- `Directorio artefactos`
- `Test size`
- `Random state`
- `CV folds`
- `Split strategy`
- `Group column`
- `Espacio de features`
- `Componentes PCA`
- `Modelos de regresion`
- `Modelos de clasificacion`
- `Regresor para persistencia`
- `Clasificador para persistencia`

### 18.2. Salidas de regresion

Columnas principales:

- `model_name`
- `target`
- `feature_space`
- `pca_components`
- `split_strategy`
- `group_column`
- `train_group_count`
- `test_group_count`
- `train_size`
- `test_size`
- `random_state`
- `cv_folds`
- `mae`
- `rmse`
- `r2`
- `cv_mae_mean`
- `cv_mae_std`
- `cv_rmse_mean`
- `cv_rmse_std`
- `cv_r2_mean`
- `cv_r2_std`

Tambien se generan metricas por tipo de escenario cuando existe
`scenario_type`, por ejemplo:

- `scenario_rows_<escenario>`
- `mae_<escenario>`
- `rmse_<escenario>`
- `r2_<escenario>`

### 18.3. Salidas de clasificacion

Columnas principales:

- `model_name`
- `target`
- `feature_space`
- `pca_components`
- `split_strategy`
- `group_column`
- `train_group_count`
- `test_group_count`
- `train_size`
- `test_size`
- `random_state`
- `cv_folds`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `cv_accuracy_mean`
- `cv_accuracy_std`
- `cv_precision_mean`
- `cv_precision_std`
- `cv_recall_mean`
- `cv_recall_std`
- `cv_f1_mean`
- `cv_f1_std`

Tambien se generan metricas por tipo de escenario:

- `scenario_rows_<escenario>`
- `accuracy_<escenario>`
- `precision_<escenario>`
- `recall_<escenario>`
- `f1_<escenario>`

### 18.4. Archivos generados

Los resultados se guardan en la carpeta `results` de la red:

```text
regression_comparison_flooding_volume_m3.csv
classification_comparison_flooded.csv
```

Si hay motor Excel disponible (`openpyxl` o `xlsxwriter`), tambien se generan:

```text
regression_comparison_flooding_volume_m3.xlsx
classification_comparison_flooded.xlsx
```

Los modelos persistidos se guardan como `.joblib` y se registran en:

```text
model_artifacts/
  manifest.json
  regression_<modelo>.joblib
  classification_<modelo>.joblib
```

## 19. Artefactos de inferencia

Los artefactos guardan:

- tarea: regresion o clasificacion
- nombre del modelo
- target usado
- columnas de features usadas
- pipeline completo entrenado
- ruta del dataset usado
- numero de filas entrenadas
- fecha de entrenamiento
- espacio de features (`raw_features` o `pca_components`)
- numero de componentes PCA
- columna de agrupacion
- estrategia de split

Esto permite predecir despues sin reentrenar.

Decision importante:

```text
La prediccion no debe reentrenar modelos cada vez.
Debe cargar artefactos persistidos.
```

## 19.1. Resultados actuales de modelos ML

Esta seccion resume los resultados concretos disponibles actualmente en:

```text
data/networks/chico_hydro-qx1/results/
```

Archivos usados:

```text
regression_comparison_flooding_volume_m3.csv
classification_comparison_flooded.csv
```

Estos resultados corresponden al dataset de la red `chico_hydro-qx1`, con:

- `feature_space = pca_components`
- `pca_components = 5`
- `split_strategy = grouped_by_run_id`
- `group_column = run_id`
- `train_group_count = 24`
- `test_group_count = 7`
- `train_size = 3864`
- `test_size = 1127`
- `random_state = 42`
- `cv_folds = 5`
- escenario evaluado en test: `embedded_hydrograph`

### 19.1.1. Regresion: `flooding_volume_m3`

El objetivo de regresion es predecir el volumen de flooding por nodo.

| Modelo | MAE | RMSE | R2 | CV MAE mean | CV RMSE mean | CV R2 mean |
|---|---:|---:|---:|---:|---:|---:|
| `xgboost` | 7.739 | 19.528 | 0.958 | 8.757 | 17.078 | 0.980 |
| `svr_rbf` | 26.080 | 72.093 | 0.423 | 46.696 | 107.220 | 0.356 |
| `ridge` | 51.465 | 85.686 | 0.184 | 73.678 | 115.025 | 0.251 |
| `lasso` | 51.469 | 85.688 | 0.184 | 73.682 | 115.026 | 0.251 |

Interpretacion:

- El mejor modelo actual de regresion es `xgboost`.
- Tiene el menor `MAE` y `RMSE`.
- Tiene el mayor `R2` en test.
- Tambien tiene el mejor desempeno en validacion cruzada agrupada.

Conclusion:

```text
Para predecir flooding_volume_m3, XGBoost es claramente el mejor modelo actual.
```

### 19.1.2. Clasificacion: `flooded`

El objetivo de clasificacion es predecir si el nodo se inunda o no.

| Modelo | Accuracy | Precision | Recall | F1 | CV Accuracy mean | CV Precision mean | CV Recall mean | CV F1 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_classifier` | 0.925 | 0.991 | 0.795 | 0.882 | 0.973 | 0.990 | 0.951 | 0.970 |
| `svc_rbf` | 0.893 | 0.846 | 0.848 | 0.847 | 0.895 | 0.867 | 0.913 | 0.889 |
| `logistic_regression` | 0.812 | 0.802 | 0.615 | 0.696 | 0.829 | 0.810 | 0.829 | 0.819 |

Interpretacion:

- El mejor modelo global actual de clasificacion es `xgboost_classifier`.
- Tiene el mayor `accuracy`, `precision` y `F1` en test.
- En validacion cruzada agrupada tambien es el mejor por `CV F1 mean`.
- `svc_rbf` tiene mejor `recall` en test que `xgboost_classifier`
  (`0.848` frente a `0.795`), pero con menor precision y menor F1.

Conclusion:

```text
Para clasificacion general, XGBoost Classifier es el mejor modelo actual.
Si el objetivo operativo fuera priorizar no perder fallas, se debe revisar el
trade-off con SVC porque tiene mayor recall en el split de test actual.
```

### 19.1.3. Decision practica actual

Con los resultados disponibles, la configuracion recomendada para artefactos de
inferencia tabular es:

```text
Regresion:     xgboost
Clasificacion: xgboost_classifier
```

Esto coincide con la preferencia implementada en `train.py`, que selecciona
XGBoost como modelo preferido cuando esta disponible.

### 19.1.4. Advertencia sobre estos resultados

Estos resultados no deben interpretarse como desempeno final del proyecto.
Representan el estado actual con:

- una red principal evaluada
- un conjunto de escenarios disponible
- features tabulares con PCA
- particion agrupada por `run_id`

Para afirmar generalizacion real a otras redes, falta validar con:

- multiples redes
- split por `network_hash`
- escenarios hidraulicos mas diversos
- comparacion contra modelos entrenados sin PCA
- validacion temporal cuando exista el dataset CNN/LSTM

## 20. Inferencia

Existen dos rutas.

### 20.1. Inferencia recomendada desde `.inp`

Archivo:

```text
swmm_resilience/ml/predict_from_inp.py
```

Esta es la ruta objetivo para escalar el proyecto a varias redes.

Motivo:

- El `.inp` es la fuente original de la red.
- No tiene sentido exigir CSV auxiliares para cada red nueva.
- La herramienta debe poder leer topologia y features desde el archivo SWMM.

### 20.2. Inferencia legacy desde CSV

Archivo:

```text
swmm_resilience/ml/predict_tabular.py
```

Se conserva por compatibilidad y depuracion.

No es la ruta objetivo de produccion.

## 21. `delta_inflow_lps` e `inflow_multiplier`

Esta ha sido una decision importante del proyecto.

### 21.1. `inflow_multiplier`

Representa el factor global de la corrida.

Ejemplo:

```text
inflow_multiplier = 2.0
```

significa duplicar el caudal o hidrograma segun el modo.

### 21.2. `delta_inflow_lps`

Debe representar una diferencia de caudal en `L/s`, no un factor.

Para un nodo:

```text
delta_nodo = caudal_nuevo_nodo - caudal_base_nodo
```

En un esquema con multiplicador:

```text
delta_nodo = base_inflow_lps * (inflow_multiplier - 1)
```

Decision:

- `runs.inflow_multiplier` es el control global.
- `node_results.delta_inflow_lps` representa el delta por nodo.
- No se debe tratar `0` como faltante.
- No se debe copiar el delta global hacia nodos si eso cambia la semantica.

## 22. Correcciones y supuestos importantes

### 22.1. Volumen y duracion de flooding

Decision:

- Preferir el `.rpt` para `flooding_volume_m3` y
  `flooding_duration_min`.

Motivo:

- El reporte de SWMM es la salida oficial agregada para esas metricas.
- PySWMM puede entregar estadisticas, pero para esas dos variables se decidio
  tomar el `.rpt` cuando esta disponible.

### 22.2. `time_to_first_flood_min`

Decision:

- Guardar directamente el `elapsed_min` del primer timestep donde
  `node.flooding > 0`.

Motivo:

- Evita errores por asumir timestep fijo.

### 22.3. `depth_rate_m_per_min`

Decision:

- La tasa de crecimiento de profundidad se calcula a partir de diferencias
  entre timesteps reales.
- El primer timestep solo inicializa el valor previo.

Motivo:

- No se debe comparar el primer valor real contra un cero artificial.

### 22.4. `head_m`

Decision:

- No guardar `head_m` en el MVP temporal.

Motivo:

- Es derivable como:

```text
invert_elev_m + depth_m
```

### 22.5. `link_timeseries`

Decision:

- No guardar series temporales por link en el MVP.

Motivo:

- Aumenta el volumen y la complejidad.
- Primero se validara el pipeline temporal por nodo.

## 23. Fase temporal actual

### 23.1. Fase 0 definida

Decisiones:

- falla temporal: `failed_now = 1` si `flooding_lps > 0`
- frecuencia objetivo para ML: 5 minutos
- ventana historica: 20 minutos
- horizonte inicial: 5 minutos
- avance entre ventanas: 5 minutos
- todos los escenarios generan `node_timeseries`
- no se guarda `head_m`
- no se guarda `link_timeseries` en el MVP
- PyTorch entra despues de tener ventanas listas

### 23.2. Fase 1 implementada

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

Este Parquet contiene la serie cruda observada por PySWMM.

## 24. Preparacion futura del dataset CNN/LSTM

### 24.1. Capa cruda

La capa cruda es `node_timeseries`.

Unidad:

```text
run_id + node_id + step_index
```

Esta capa no se entrega directamente al modelo. Primero se transforma.

### 24.2. Resampleo

Decision:

```text
ML_TEMPORAL_RESAMPLE_MIN = 5
```

Motivo:

- La red actual puede tener pasos cercanos a 3 minutos.
- Usar 1 minuto generaria mas datos.
- 5 minutos es un MVP razonable y escalable.

Si en el futuro se necesita mas resolucion temporal, se puede bajar a 2 o 3
minutos.

### 24.3. Ventanas temporales

Configuracion:

```python
ML_TEMPORAL_WINDOW_MIN = 20
ML_TEMPORAL_HORIZON_MIN = 5
ML_TEMPORAL_STEP_MIN = 5
```

Interpretacion:

- El modelo mira los ultimos 20 minutos.
- Predice si habra falla en los proximos 5 minutos.
- La siguiente muestra avanza 5 minutos.

Ejemplo:

```text
Entrada:  0, 5, 10, 15, 20
Target:   falla entre 20 y 25

Entrada:  5, 10, 15, 20, 25
Target:   falla entre 25 y 30
```

### 24.4. Target temporal

Primer target:

```text
failure_within_horizon_5m
```

Definicion:

```text
1 si existe flooding_lps > 0 en el horizonte futuro de 5 minutos
0 si no existe flooding_lps > 0 en ese horizonte
```

### 24.5. Features temporales MVP

Se usaran inicialmente:

- `total_inflow_lps`
- `lateral_inflow_lps`
- `depth_m`
- `depth_ratio`
- `flooding_lps`
- `total_outflow_lps`

No se usaran inicialmente:

- `head_m`
- `upstream_flow_sum_lps`
- `downstream_flow_sum_lps`
- series por link

### 24.6. Features estaticas opcionales

El modelo temporal puede combinar:

- rama temporal
- rama estatica

Features estaticas candidatas:

- `invert_elev_m`
- `full_depth_m`
- `base_inflow_lps`
- `upstream_pipes_count`
- `upstream_diam_max_m`
- `upstream_diam_min_m`
- `upstream_slope_avg`
- `upstream_slope_max`
- `downstream_pipes_count`
- `downstream_diam_max_m`
- `downstream_diam_min_m`
- `downstream_slope_avg`
- `downstream_slope_max`

## 25. Arquitectura CNN/LSTM propuesta

### 25.1. Orden acordado

1. Validar `node_timeseries`.
2. Crear `temporal_artifacts`.
3. Construir ventanas temporales.
4. Entrenar CNN 1D baseline.
5. Comparar con LSTM.
6. Construir predictor temporal.
7. Usar modelo como surrogate.

### 25.2. Por que empezar con CNN 1D

Motivos:

- suele necesitar menos datos que LSTM
- entrena mas rapido
- es estable con series cortas
- detecta patrones locales de hidrogramas

### 25.3. Por que dejar LSTM como benchmark

Motivos:

- LSTM puede capturar dependencias temporales mas largas
- pero necesita mas datos y mas cuidado
- es mejor compararla despues de tener una CNN baseline

### 25.4. Uso de PyTorch

Decision:

- usar PyTorch para CNN/LSTM.

Motivos:

- soporte natural de GPU con CUDA
- control claro de `Dataset` y `DataLoader`
- facilidad para modelos con rama temporal y rama estatica
- guardado de `state_dict` y configuraciones

PyTorch no entra antes de tener ventanas temporales. La GPU acelera
entrenamiento, pero no corrige un dataset mal definido.

### 25.5. Formas de tensores

Guardar ventanas inicialmente como:

```text
[samples, timesteps, features]
```

Para CNN 1D en PyTorch:

```text
[batch, features, timesteps]
```

Se transforma con:

```python
X.permute(0, 2, 1)
```

Para LSTM:

```text
[batch, timesteps, features]
```

## 26. Riesgos principales

### 26.1. Leakage

Riesgo:

- usar variables de resultado como features.

Ejemplo:

- `max_depth_m`
- `flooding_duration_min`
- `flooding_volume_m3`

Estas variables se conocen despues de la simulacion. No deben usarse para
predecir resultados en inferencia.

### 26.2. Mezclar escenarios incompatibles

Riesgo:

- entrenar steady y timeseries como si fueran lo mismo.

Mitigacion:

- conservar `scenario_type`
- evaluar por tipo de escenario
- en el futuro agregar `input_source`

### 26.3. Confundir factor y caudal

Riesgo:

- tratar `inflow_multiplier = 1.5` como si fueran `1.5 L/s`.

Mitigacion:

- `inflow_multiplier` es factor
- `delta_inflow_lps` es caudal
- no mezclar semanticas

### 26.4. Generalizacion falsa

Riesgo:

- validar por filas cuando muchas filas vienen de la misma corrida.

Mitigacion:

- usar `run_id` como grupo.
- a futuro, validar tambien por `network_hash`.

## 27. Objetivo a largo plazo

El proyecto debe evolucionar hacia:

```text
muchas redes + muchos escenarios + series temporales
  -> modelos temporales entrenados con PyTorch
  -> prediccion temprana de falla
  -> evaluacion rapida de alternativas
  -> validacion final con SWMM
```

La meta no es que el modelo reemplace al simulador, sino que funcione como un
modelo sustituto rapido para priorizar decisiones.

## 28. Proximos pasos tecnicos

El siguiente paso recomendado es Fase 2 temporal:

1. Crear tabla `temporal_artifacts`.
2. Registrar cada Parquet temporal por `run_id`.
3. Guardar:
   - `artifact_id`
   - `run_id`
   - `artifact_type`
   - `path`
   - `rows_count`
   - `created_at`
4. Validar que la ruta existe.
5. Validar que el Parquet tiene las columnas esperadas.

Despues:

1. Construir ventanas temporales.
2. Guardar `windows.npz`.
3. Guardar `window_metadata.parquet`.
4. Entrenar CNN 1D.
5. Comparar LSTM.

## 29. Resumen ejecutivo

El proyecto ya tiene un pipeline tabular funcional y una primera capa temporal.

La parte tabular sirve para:

- generar datos
- entrenar modelos baseline
- comparar estrategias
- producir artefactos de inferencia

La parte temporal sirve para:

- preparar el camino a CNN/LSTM
- conservar la dinamica real por nodo
- evitar depender solo de resumenes agregados

Las decisiones mas importantes tomadas son:

- el `.inp` es la fuente operativa de redes nuevas
- el CSV es artefacto de entrenamiento y auditoria
- prediccion desde CSV queda como legacy
- `inflow_multiplier` es factor, no caudal
- `delta_inflow_lps` es caudal, no factor
- no usar variables de resultado como features
- usar PCA como linea base compacta
- validar por `run_id`
- preparar CNN/LSTM desde series temporales, no desde CSV tabular
- empezar con CNN 1D y usar LSTM como comparacion posterior
