# SWMM Resilience

Pipeline para ejecutar simulaciones SWMM, guardar resultados en SQLite y exportar un dataset plano para análisis y ML.

## Estructura actual

```text
main.py
view_db.py
app.py
swmm_resilience/
  __init__.py
  config.py
  main.py
  utils.py
  simulation/
    __init__.py
    runner.py
    swmm_api_io.py
  database/
    __init__.py
    schema.py
    repository.py
  analysis/
    __init__.py
    dataset.py
data/
  training/
    swmm_resilience.db
  networks/
    chico_steady/
      SWMM - Chico (PVC) Prueba 1 - Steady.inp
      results/
        dataset_ml.csv
```

## Cómo ejecutar

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ubica tu archivo `.inp` dentro de una carpeta de red en:

```text
data/networks/<nombre_de_red>/
```

3. Ejecuta el pipeline:

```bash
python main.py
```

4. Si prefieres usar la aplicacion local de escritorio:

```bash
python app.py
```

La app abre una ventana local para seleccionar el `.inp`, definir nodos, ejecutar corridas, entrenar modelos, predecir nuevos caudales steady con ML sin correr PySWMM y abrir el visor SQLite. En la pestaña `Predicción ML` puedes elegir el clasificador y el regresor tabular que quieres usar.

## Cómo evaluar los modelos

Despues de generar el dataset con `python main.py`, puedes ejecutar la comparacion de modelos de ML con:

```bash
python -m swmm_resilience.ml.train
```

Ese comando sirve para:

- entrenar y comparar modelos de regresion usando como target `flooding_volume_m3`
- entrenar y comparar modelos de clasificacion usando como target `flooded`
- imprimir las metricas en consola
- guardar los resultados comparativos en archivos CSV y, si es posible, tambien en XLSX dentro de la carpeta `results`

Este paso es importante porque `python main.py` genera los resultados hidraulicos y el dataset, pero la evaluacion de modelos se corre aparte con `python -m swmm_resilience.ml.train`.

## Arquitectura preparada para CNN temporal

El flujo ML actual sigue siendo tabular y se mantiene en `swmm_resilience/ml/train.py`. Esa ruta sirve como baseline fuerte para modelos como XGBoost.

Tambien queda preparada una carpeta para el trabajo futuro con CNN 1D sobre hidrogramas:

```text
swmm_resilience/ml/temporal/
  README.md
  schemas.py
  dataset.py
  train_cnn.py
  predict.py
```

Por ahora esa carpeta es solo scaffold: no entrena redes neuronales ni agrega dependencias como PyTorch o TensorFlow. La idea es usarla mas adelante cuando guardemos series temporales por nodo y podamos construir ventanas tipo `[samples, timesteps, features]`.

### Fase temporal actual

La Fase 0 queda definida asi:

- `failed_now = 1` cuando `flooding_lps > 0`
- todas las corridas generan `node_timeseries`, incluyendo escenarios steady
- frecuencia objetivo para ML: `5 minutos`
- ventana historica inicial: `20 minutos`
- horizonte inicial de prediccion: `5 minutos`
- avance entre ventanas: `5 minutos`
- `link_timeseries` queda fuera del MVP
- `head_m` no se guarda porque se puede derivar de `invert_elev_m + depth_m`

La Fase 1 ya persiste un Parquet temporal por corrida en:

```text
data/networks/<red>/results/temporal/node_timeseries/run_<run_id>.parquet
```

Columnas obligatorias de `node_timeseries`:

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

Este archivo guarda la serie cruda observada durante la simulacion. Todavia no
construye ventanas ni entrena CNN/LSTM; eso corresponde a fases posteriores.
Para escribir Parquet se requiere `pyarrow`.

Puedes ver el resumen del plan con:

```bash
python -m swmm_resilience.ml.temporal.train_cnn
```

## Dónde cambiar los caudales

Edita esta variable en [swmm_resilience/config.py](swmm_resilience/config.py):

```python
DEFAULT_DELTA_INFLOWS_LPS = list(range(2, 52, 2))
```

Los valores están en `L/s` por nodo.

## Arquitectura swmm_api + PySWMM

La herramienta divide responsabilidades entre dos librerías:

| Librería | Rol |
|---|---|
| `swmm-api` | Manipula archivos `.inp` y lee resultados de `.rpt`/`.out` |
| `PySWMM` | Ejecuta la simulación hidráulica |

El flujo por corrida es:

```text
.inp original
  -> swmm_api_io.write_scaled_inp()   # escala hidrogramas, escribe .inp temporal
  -> PySWMM Simulation(temp.inp)      # ejecuta la simulación
  -> .rpt/.out generados por SWMM
  -> swmm_api_io.read_node_flooding_summary(.rpt)   # flooding_volume_m3, flooding_duration_min
  -> PySWMM link.conduit_statistics                 # resultados de links
  -> guardar en SQLite
```

Todo el acceso a `swmm-api` está encapsulado en `swmm_resilience/simulation/swmm_api_io.py`. El resto del código no importa objetos de `swmm_api` directamente.

### Notas de riesgo

- `swmm-api` aún no es versión `1.0`; su API puede cambiar en versiones futuras.
- Si cambia la API de `swmm-api`, el único archivo que debe editarse es `swmm_api_io.py`.
- No mezclar objetos de `swmm_api` fuera de esa capa.
- Las unidades del `.rpt` son: volúmenes en `10^6 litros` (se convierten a m³ multiplicando por 1000), duraciones en horas (se convierten a minutos).

## Hidrogramas en archivos `.inp`

Los hidrogramas se definen dentro de la red SWMM, usando las secciones `[INFLOWS]` y `[TIMESERIES]` del archivo `.inp`. La herramienta no carga hidrogramas desde CSV externo.

Para comparar escenarios con hidrogramas distintos, selecciona el `.inp` correspondiente, por ejemplo redes `Qx1.00`, `Qx2.00` o `Qx3.00`, y ejecuta la corrida sobre ese archivo.

## Salidas

- Base de datos central de entrenamiento: `data/training/swmm_resilience.db`
- Dataset CSV por red: `data/networks/chico_steady/results/dataset_ml.csv`

## Diccionario de variables

Esta seccion resume las variables que aparecen en la base de datos y en el dataset exportado. Varias salen directamente de SWMM/PySWMM y otras son metricas derivadas que calcula este proyecto.

### Variables de corrida (`runs`)

- `run_id`
  Identificador unico de la corrida.

- `network_file`
  Ruta o nombre del archivo `.inp` usado en la corrida.

- `network_hash`
  Hash del archivo de red. Sirve para identificar de forma estable la red simulada.

- `scenario_type`
  Tipo de escenario ejecutado. Hoy el valor por defecto es `uniform_inflow_sweep`.

- `spatial_pattern`
  Patron espacial de aplicacion del caudal adicional. Hoy el valor por defecto es `uniform`.

- `delta_inflow_lps`
  Caudal adicional uniforme aplicado por nodo, en `L/s`.

- `executed_at`
  Fecha y hora de ejecucion de la corrida.

- `status`
  Estado de la corrida: por ejemplo `pending`, `completed` o `failed`.

### Variables de entrada por nodo (`run_inputs`)

- `input_id`
  Identificador unico del registro de entrada.

- `run_id`
  Corrida a la que pertenece ese registro de entrada.

- `node_uid`
  Nodo al que se le aplico la entrada.

- `delta_inflow_lps`
  Caudal de entrada asociado a ese nodo en esa corrida, en `L/s`. Se conserva una sola variable de entrada para evitar duplicidad con el dataset de entrenamiento.

### Variables estaticas de nodos (`network_nodes`)

- `node_uid`
  Identificador del nodo dentro de la red.

- `network_hash`
  Hash de la red a la que pertenece el nodo.

- `invert_elev_m`
  Cota de solera o elevacion de fondo del nodo, en metros.

- `full_depth_m`
  Profundidad total util del nodo, en metros.

- `node_type`
  Tipo de nodo segun SWMM/PySWMM, por ejemplo junction, outfall o storage.

- `in_degree`
  Numero de enlaces que llegan al nodo.

- `out_degree`
  Numero de enlaces que salen del nodo.

- `upstream_pipes_count`
  Numero de tuberias aguas arriba conectadas al nodo.

- `upstream_diam_max_m`
  Diametro maximo de las tuberias aguas arriba, en metros.

- `upstream_diam_min_m`
  Diametro minimo de las tuberias aguas arriba, en metros.

- `upstream_diam_avg_m`
  Diametro promedio de las tuberias aguas arriba, en metros.

- `upstream_slope_avg`
  Pendiente promedio absoluta de las tuberias aguas arriba, en `m/m`.

- `upstream_slope_max`
  Pendiente maxima absoluta de las tuberias aguas arriba, en `m/m`.

- `upstream_capacity_lps`
  Suma de las capacidades teoricas a flujo lleno de las tuberias aguas arriba, en `L/s`.

- `downstream_pipes_count`
  Numero de tuberias aguas abajo conectadas al nodo.

- `downstream_diam_max_m`
  Diametro maximo de las tuberias aguas abajo, en metros.

- `downstream_diam_min_m`
  Diametro minimo de las tuberias aguas abajo, en metros.

- `downstream_diam_avg_m`
  Diametro promedio de las tuberias aguas abajo, en metros.

- `downstream_slope_avg`
  Pendiente promedio absoluta de las tuberias aguas abajo, en `m/m`.

- `downstream_slope_max`
  Pendiente maxima absoluta de las tuberias aguas abajo, en `m/m`.

- `downstream_capacity_lps`
  Suma de las capacidades teoricas a flujo lleno de las tuberias aguas abajo, en `L/s`.

### Variables dinamicas por nodo (`node_results`)

- `result_id`
  Identificador unico del resultado por nodo.

- `run_id`
  Corrida a la que pertenece el resultado.

- `node_id`
  Nodo al que corresponde el resultado.

- `flooded`
  Indicador binario de inundacion del nodo. Vale `1` si hubo volumen de flooding mayor que cero y `0` si no.

- `flooding_volume_m3`
  Volumen total inundado o desbordado por el nodo durante la corrida, en `m3`.

- `flooding_duration_min`
  Tiempo total durante el cual el nodo permanecio inundado, en minutos.

- `max_depth_m`
  Profundidad maxima alcanzada en el nodo durante la simulacion, en metros.

- `max_depth_ratio`
  Relacion entre la profundidad maxima y la profundidad total del nodo: `max_depth_m / full_depth_m`.
  Sirve para medir que tan cerca estuvo el nodo de llenarse por completo.

- `time_to_peak_min`
  Tiempo desde el inicio de la simulacion hasta que el nodo alcanzo su profundidad maxima, en minutos.

- `depth_rate_m_per_min`
  Maxima tasa de aumento de profundidad observada en el nodo, en metros por minuto.
  Se calcula entre pasos de simulacion como cambio de profundidad dividido por el paso de tiempo.

- `max_total_outflow_lps`
  Caudal total maximo de salida observado en el nodo durante la corrida, en `L/s`.
  Corresponde a `node.total_outflow` en PySWMM.

- `time_to_peak_outflow_min`
  Tiempo desde el inicio de la simulacion hasta el instante en que el nodo alcanzo su mayor `total_outflow`, en minutos.

- `downstream_link_peak_flows_lps_json`
  Desglose en formato JSON con el caudal maximo positivo observado hacia cada link saliente del nodo.
  Ejemplo: `{"TUB_01": 152.4, "TUB_02": 87.1}`.

### Variables dinamicas por enlace (`link_results`)

- `result_id`
  Identificador unico del resultado por enlace.

- `run_id`
  Corrida a la que pertenece el resultado.

- `link_id`
  Enlace al que corresponde el resultado.

- `max_flow_lps`
  Caudal maximo alcanzado en el enlace, en `L/s`.

- `max_velocity_mps`
  Velocidad maxima alcanzada en el enlace, en `m/s`.

- `max_depth_m`
  Profundidad maxima alcanzada en el enlace, en metros.

- `max_capacity_ratio`
  Relacion entre el caudal maximo observado y la capacidad teorica a flujo lleno del enlace.

- `surcharged`
  Indicador binario de sobrecarga. Vale `1` si el enlace paso tiempo a flujo lleno y `0` si no.

- `time_full_flow_hrs`
  Tiempo total que el enlace permanecio a flujo lleno, en horas.

### Variables resumen de corrida (`run_summary`)

- `summary_id`
  Identificador unico del resumen de corrida.

- `run_id`
  Corrida a la que pertenece el resumen.

- `inflow_multiplier`
  Factor de incremento del escenario para esa corrida.
  En `run_summary` se conserva una sola columna de escenario para evitar duplicidad con otras tablas.

- `total_nodes`
  Numero total de nodos evaluados en la corrida.

- `failed_nodes_count`
  Numero total de nodos que fallaron en la corrida.
  En la implementacion actual, un nodo se considera fallado si presento inundacion.

- `total_flooding_volume_m3`
  Suma del volumen inundado en todos los nodos, en `m3`.

- `pct_flooded_nodes`
  Porcentaje de nodos inundados respecto al total de nodos.

- `time_to_first_flood_min`
  Tiempo desde el inicio de la simulacion hasta el primer instante en que algun nodo presento `flooding > 0`, en minutos.
  En este proyecto se calcula manualmente recorriendo los pasos de simulacion; no es una variable nativa ya lista en PySWMM.

- `resilience_index`
  Indice simple de resiliencia definido como `1 - failed_nodes_count / total_nodes`.
  Vale `1.0` si ningun nodo se inunda y disminuye a medida que aumenta la fraccion de nodos inundados.

### Variables presentes en el CSV exportado

El archivo `dataset_ml.csv` contiene una combinacion de:

- variables de corrida: `run_id`, `delta_inflow_lps`, `scenario_type`, `spatial_pattern`
- variables de entrada por nodo: `delta_inflow_lps`
- variables estaticas del nodo: columnas de `network_nodes`
- variables resultado por nodo: `max_depth_m`, `max_depth_ratio`, `time_to_peak_min`, `depth_rate_m_per_min`, `max_total_outflow_lps`, `time_to_peak_outflow_min`, `downstream_link_peak_flows_lps_json`, `flooded`, `flooding_volume_m3`, `flooding_duration_min`

## Organización por red

La carpeta `data/networks/` ya no se usa solo para "guardar .inp". Ahora es el contenedor principal de cada red.

La idea es que cada red tenga su propia carpeta, por ejemplo:

```text
data/networks/chico_steady/
```

Y dentro de esa carpeta queden juntos:

- el archivo `.inp`
- la base de datos generada para esa red
- el dataset exportado para esa red
- futuras comparaciones de modelos para esa red

Esto ayuda a:

- identificar fácilmente a qué red pertenece cada resultado
- evitar mezclar resultados de redes distintas
- preparar el proyecto para un enfoque multired más adelante

## Explicación de cada archivo `.py`

### Archivos que sí ejecutas directamente

- [main.py](main.py)
  Punto de entrada del proyecto. Es el archivo que debes correr para generar la base de datos y el dataset.

- [view_db.py](view_db.py)
  Visor rápido en `tkinter` para abrir la base SQLite, cambiar de tabla, filtrar por `run_id` y revisar filas sin escribir SQL.

### Módulos internos del paquete

- [swmm_resilience/__init__.py](swmm_resilience/__init__.py)
  Archivo de paquete. Solo marca `swmm_resilience/` como módulo importable. No contiene lógica de trabajo.

- [swmm_resilience/config.py](swmm_resilience/config.py)
  Configuración general del proyecto: rutas base, ubicación del `.inp`, carpeta de resultados, nombre de la BD, nombre del CSV y lista de caudales de inyección.

- [swmm_resilience/main.py](swmm_resilience/main.py)
  Orquestador principal. Crea carpetas, decide qué archivo `.inp` usar, reinicia la base de datos si existe, llama a la extracción de topología, ejecuta todas las simulaciones, guarda resultados y exporta el dataset.

- [swmm_resilience/utils.py](swmm_resilience/utils.py)
  Funciones auxiliares compartidas, como generación de IDs, hash de archivos, redondeo seguro y cálculo de capacidad hidráulica teórica en tuberías circulares.

### Simulación

- [swmm_resilience/simulation/__init__.py](swmm_resilience/simulation/__init__.py)
  Exporta las funciones principales del módulo de simulación.

- [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py)
  Núcleo hidráulico del proyecto. Aquí están:
  - la extracción de topología estática (`network_nodes`, `network_links`) usando `swmm_api_io`
  - el escalado de hidrogramas internos delegado a `swmm_api_io.write_scaled_inp`
  - la ejecución de la simulación con PySWMM
  - la lectura de resultados de inundación por nodo desde el `.rpt` vía `swmm_api_io`
  - la extracción de estadísticas de links con `link.conduit_statistics` de PySWMM

- [swmm_resilience/simulation/swmm_api_io.py](swmm_resilience/simulation/swmm_api_io.py)
  Capa de I/O estructurado sobre archivos SWMM. Centraliza todo el uso de `swmm-api`:
  - leer `.inp` con estructura de objetos (sin parseo manual de texto)
  - detectar nodos con inflows y obtener sus timeseries
  - escalar hidrogramas y escribir `.inp` temporales
  - leer el resumen de inundación por nodo desde el `.rpt`
  - leer series temporales desde el `.out` (scaffold para dataset temporal futuro)

### Base de datos

- [swmm_resilience/database/__init__.py](swmm_resilience/database/__init__.py)
  Exporta las funciones principales del módulo de base de datos.

- [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py)
  Define el esquema SQLite. Aquí se crean las tablas `runs`, `run_inputs`, `network_nodes`, `network_links`, `node_results`, `link_results` y `run_summary`.

- [swmm_resilience/database/repository.py](swmm_resilience/database/repository.py)
  Capa de persistencia. Se encarga de abrir la base, guardar topología estática, guardar resultados de simulación, actualizar estados de corrida y mostrar un resumen de corridas.

### Análisis y exportación

- [swmm_resilience/analysis/__init__.py](swmm_resilience/analysis/__init__.py)
  Exporta las funciones principales del módulo de análisis.

- [swmm_resilience/analysis/dataset.py](swmm_resilience/analysis/dataset.py)
  Construye el dataset plano a partir de la base de datos. Hace los `JOIN` entre tablas y exporta el CSV final.

## Qué archivo tocar según el cambio que quieras hacer

- Cambiar rutas, archivo `.inp` o lista de caudales:
  [swmm_resilience/config.py](swmm_resilience/config.py)

- Cambiar cómo se inyecta el caudal o cómo se calculan resultados:
  [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py)

- Cambiar estructura de tablas:
  [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py)

- Cambiar cómo se guarda o consulta la información:
  [swmm_resilience/database/repository.py](swmm_resilience/database/repository.py)

- Cambiar el dataset exportado:
  [swmm_resilience/analysis/dataset.py](swmm_resilience/analysis/dataset.py)

- Visualizar rápidamente la base:
  [view_db.py](view_db.py)

## Qué no debes ejecutar directamente

Estos archivos son módulos internos. Normalmente no se ejecutan solos:

- `swmm_resilience/main.py`
- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/database/schema.py`
- `swmm_resilience/database/repository.py`
- `swmm_resilience/analysis/dataset.py`

El flujo normal es:

```bash
python main.py
```

## Escalabilidad a otras redes

Hoy el pipeline está bien orientado a aprender sobre una red concreta, pero si en el futuro quieres que el modelo generalice a otros archivos `.inp`, conviene tener en cuenta estas reglas desde ya:

1. Entrenar con muchas redes, no con una sola.
   Si el modelo solo ve una red, puede aprender la "firma" particular de esa red y no principios generales.

2. Mantener features comparables entre redes.
   Las variables deben describir propiedades físicas y topológicas transferibles, no depender de identificadores o convenciones específicas de una sola red.

3. Validar por red, no solo por fila.
   Este es uno de los puntos más importantes. No basta con mezclar todas las filas y hacer un `train/test split` aleatorio. Si nodos de una misma red quedan repartidos entre entrenamiento y prueba, el resultado puede verse muy bueno pero ser engañoso.

4. Separar entrenamiento y prueba con redes no vistas.
   La validación correcta para generalización multired es: entrenar en un grupo de redes y probar en otras redes completamente nuevas.

5. Revisar rangos y distribución de variables físicas.
   Si otra red tiene diámetros, pendientes, profundidades o capacidades muy distintas, el modelo puede quedar fuera de distribución y degradarse rápido.

6. Incluir diversidad hidráulica.
   Para un modelo robusto no basta con varias copias parecidas de la misma red; conviene incluir redes con distintos tamaños, topologías, materiales, pendientes, estructuras especiales y escenarios hidráulicos.

7. Medir desempeño por red además de desempeño global.
   No solo importa el promedio general. Conviene revisar métricas red por red para detectar en cuáles el modelo transfiere bien y en cuáles falla.

## Estrategia para pasar a un enfoque multired

Una ruta práctica para volver el proyecto multired sería esta:

### Fase 1. Organizar los datos por red

- Guardar varios archivos `.inp` en `data/networks/`
- Ejecutar el pipeline para cada red
- Guardar en la BD y en el dataset un identificador de red claro, por ejemplo `network_hash` o `network_file`

Objetivo:
tener un dataset combinado donde cada fila siga sabiendo de qué red viene.

### Fase 2. Consolidar un dataset único multired

- Exportar un solo CSV con todas las redes
- Añadir explícitamente columnas de identificación de red:
  - `network_hash`
  - `network_file`

Objetivo:
poder agrupar, filtrar y separar por red en entrenamiento y evaluación.

### Fase 3. Cambiar la validación

En vez de hacer solo `train_test_split` aleatorio por filas:

- hacer separación por red completa
- por ejemplo:
  - entrenar con redes A, B y C
  - probar con red D

Opciones recomendables:
- `GroupKFold`
- `GroupShuffleSplit`

Usando como grupo:
- `network_hash`

Objetivo:
medir generalización real a redes no vistas.

### Fase 4. Comparar dos tipos de evaluación

Conviene tener ambas:

- Evaluación por fila:
  útil para ver si el modelo aprende patrones internos dentro del dataset

- Evaluación por red:
  útil para saber si realmente transfiere a otra red

Objetivo:
no confundir buen ajuste interno con verdadera capacidad de generalización.

### Fase 5. Revisar features

Conviene priorizar:
- diámetros
- pendientes
- capacidades
- grados de conectividad
- agregados upstream/downstream
- profundidades y relaciones geométricas

Conviene evitar depender demasiado de:
- IDs de nodos
- convenciones locales de nombrado
- variables que solo existan en una red concreta

Objetivo:
construir un espacio de entrada más estable entre redes.

### Fase 6. Probar modelos simples primero

Antes de complejizar:
- Ridge
- Lasso
- SVR
- XGBoost

Objetivo:
establecer una línea base multired seria antes de pasar a modelos más complejos.

## Features actuales del entrenamiento tabular

Segun la configuracion actual de `ML_DROP_COLUMNS`, estas son las columnas que hoy entran al preprocesamiento del pipeline. Como `ML_USE_PCA = True`, el modelo no aprende directamente sobre estas variables, sino sobre los componentes PCA derivados de ellas. Con la configuracion actual, el pipeline reduce este espacio a `5` componentes antes de entrenar. Si quieres volver al entrenamiento con variables originales, basta con cambiar `ML_USE_PCA = False` en `swmm_resilience/config.py`.

- `delta_inflow_lps`
- `invert_elev_m`
- `full_depth_m`
- `in_degree`
- `out_degree`
- `upstream_pipes_count`
- `upstream_diam_max_m`
- `upstream_diam_min_m`
- `upstream_slope_avg`
- `upstream_slope_max`
- `upstream_capacity_lps`
- `downstream_pipes_count`
- `downstream_diam_max_m`
- `downstream_diam_min_m`
- `downstream_slope_avg`
- `downstream_slope_max`
- `downstream_capacity_lps`

Los `targets` usados hoy son:

- clasificacion: `flooded`
- regresion: `flooding_volume_m3`

La evaluacion actual del modelo ya no separa filas aleatoriamente. El `train/test split` y la validacion cruzada se hacen agrupando por `run_id`, para que todas las filas de una misma corrida queden juntas en train o en test. Esto evita una validacion demasiado optimista cuando la misma corrida aporta muchos nodos.

Estas columnas no entran como `features`:

- identificadores: `run_id`, `node_id`
- columnas de escenario: `scenario_type`, `spatial_pattern`
- columna duplicada o legacy: `applied_inflow_lps`
- columna categorica no numerica: `node_type`
- promedios excluidos por configuracion: `upstream_diam_avg_m`, `downstream_diam_avg_m`
- resultados de simulacion excluidos para evitar fuga de informacion: `flooding_duration_min`, `max_depth_m`, `max_depth_ratio`, `time_to_peak_min`, `depth_rate_m_per_min`, `max_total_outflow_lps`, `time_to_peak_outflow_min`

### Fase 7. Reportar resultados de forma útil

Para cada modelo conviene guardar:

- métricas globales
- métricas por red
- red de entrenamiento y red de prueba
- tamaño de train/test
- configuración del split

Objetivo:
tener trazabilidad y entender en qué redes funciona y en cuáles no.
