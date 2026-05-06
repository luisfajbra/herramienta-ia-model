# Plan Temporal Para LSTM y CNN

## Objetivo

Construir una ruta de trabajo para pasar del pipeline tabular actual a un
pipeline temporal que permita:

- clasificacion temprana de falla por nodo
- prediccion de severidad de la falla
- estimacion del tiempo restante a la falla
- evaluacion rapida de escenarios
- busqueda futura de alternativas de solucion con un modelo sustituto

## Estado actual del proyecto

Hoy el proyecto ya tiene una base tabular funcional para escenarios agregados:

- `swmm_resilience/ml/train.py`
- `swmm_resilience/analysis/dataset.py`
- `swmm_resilience/ml/temporal/`

La carpeta temporal existe como scaffold, pero todavia no se guardan series de
tiempo por nodo. El cuello de botella real no es la red neuronal: es el dataset.

En este momento el sistema guarda:

- topologia estatica por nodo y link
- resultados resumidos por nodo
- resultados resumidos por corrida

Todavia no guarda, por paso de tiempo:

- caudal de entrada por nodo
- profundidad por nodo
- flooding por nodo
- caudal total de salida por nodo
- series temporales por link

## Arquitectura de datos recomendada

Para evitar mezclar problemas distintos, el proyecto deberia organizar sus
datos en cuatro capas:

### Capa 1. Metadata de corrida

- una fila por `run_id`
- describe el escenario ejecutado

### Capa 2. Series crudas por timestep

- una fila por `run_id + node_id + step_index`
- es la fuente real para `CNN` y `LSTM`

### Capa 3. Resumen tabular derivado

- una fila por `run_id + node_id`
- se deriva de las series por timestep
- sirve para modelos tabulares, analisis rapido y reportes

### Capa 4. Ventanas temporales de entrenamiento

- tensores derivados de la capa por timestep
- forma esperada: `[samples, timesteps, features]`
- es la entrada final para `CNN` y `LSTM`

Regla central:

- `CNN` y `LSTM` no deben entrenarse desde el CSV tabular resumen
- deben entrenarse desde ventanas construidas a partir de series crudas por
  timestep


### 1. Mantener dos pipelines

No conviene reemplazar el pipeline tabular actual.

La recomendacion es separar:

- pipeline tabular para escenarios steady y baselines rapidos
- pipeline temporal para escenarios con hidrograma y alerta anticipada

### 2. Empezar por CNN 1D

Para este problema recomiendo empezar por `CNN 1D` y dejar `LSTM` como modelo
comparativo.

Motivos:

- la CNN 1D suele necesitar menos datos que una LSTM
- entrena mas rapido
- es mas estable para series cortas o medianas
- detecta bien patrones locales de forma en hidrogramas

### 3. Probar LSTM como benchmark

La LSTM si vale la pena, pero como comparacion posterior. Si el dataset no es
grande, la CNN probablemente dara mejor relacion costo/beneficio.

## Preguntas que debe responder el modelo temporal

### Clasificacion

- `failure_within_horizon_5m`
- `failure_within_horizon_10m`
- `failure_within_horizon_15m`

Interpretacion:

- dado el historial reciente del nodo, predecir si va a fallar dentro del
  horizonte siguiente

### Regresion

- `time_to_failure_min`
- `max_depth_ratio_next_horizon`
- `max_flooding_lps_next_horizon`
- `flooding_volume_next_horizon`

### Salida operativa deseada

Para cada nodo y cada instante:

- probabilidad de falla
- tiempo estimado a la falla
- severidad esperada

## Diseño recomendado del dataset temporal

### Unidad de analisis

La unidad base debe ser:

- una corrida
- un nodo
- un instante de tiempo

### Tabla o archivo temporal base

Columnas recomendadas:

```text
run_id
network_hash
node_id
time_min
scenario_type
input_source
inflow_lps
total_inflow_lps
lateral_inflow_lps
depth_m
depth_ratio
head_m
flooding_lps
total_outflow_lps
failed_now
upstream_flow_sum_lps
downstream_flow_sum_lps
static node features...
```

### Esquema recomendado de la capa por timestep

La serie cruda por nodo deberia guardar al menos:

```text
run_id
network_hash
node_id
step_index
time_sec
time_min
sim_datetime
scenario_type
input_source
total_inflow_lps
lateral_inflow_lps
depth_m
depth_ratio
head_m
flooding_lps
total_outflow_lps
failed_now
upstream_flow_sum_lps
downstream_flow_sum_lps
```

Y para links:

```text
run_id
network_hash
link_id
step_index
time_sec
time_min
sim_datetime
flow_lps
depth_m
velocity_mps
capacity_ratio
```

Notas:

- `step_index` debe existir aunque tambien se guarde `time_sec`
- `time_sec` debe ser el tiempo real desde inicio, no un supuesto paso fijo
- `depth_ratio` puede derivarse usando `full_depth_m` del nodo
- `failed_now` debe ser binario y simple en la primera version

### Regla de entrenamiento temporal

La capa por timestep no entra al modelo directamente "como tabla".

El flujo correcto es:

1. guardar la serie por timestep
2. construir ventanas deslizantes
3. etiquetar cada ventana
4. entrenar la `CNN` o la `LSTM`

Por eso la capa por timestep es obligatoria para la parte temporal, no opcional.

### Formato de almacenamiento

Para series temporales grandes recomiendo:

- `SQLite` para metadatos y resultados resumidos
- `Parquet` para series temporales por corrida

Ruta sugerida:

```text
data/networks/<red>/results/temporal/
  runs_metadata.parquet
  node_timeseries/
    run_<run_id>.parquet
  link_timeseries/
    run_<run_id>.parquet
```

Motivo:

- SQLite sigue siendo muy util para explorar corridas
- Parquet escala mejor para ventanas temporales y entrenamiento

## Forma de entrada al modelo

### Rama temporal

Ventana temporal:

```text
X_temporal shape = [samples, timesteps, features]
```

Features temporales recomendadas:

- `inflow_lps`
- `total_inflow_lps`
- `lateral_inflow_lps`
- `depth_m`
- `depth_ratio`
- `flooding_lps`
- `total_outflow_lps`
- `upstream_flow_sum_lps`
- `downstream_flow_sum_lps`

### Rama estatica

Features estaticas recomendadas:

- `invert_elev_m`
- `full_depth_m`
- `node_type`
- `in_degree`
- `out_degree`
- `upstream_capacity_lps`
- `downstream_capacity_lps`
- diametros y pendientes agregadas

### Arquitectura recomendada

Primera version:

- rama temporal CNN 1D
- rama estatica dense
- fusion de ambas ramas
- cabeza de clasificacion
- cabeza de regresion

Esto deja un modelo multi-tarea que sirve tanto para alerta como para
estimacion de severidad.

### Papel de cada capa frente al modelo

- `runs` y metadata: contexto de escenario
- `node_timeseries` por timestep: fuente principal del aprendizaje temporal
- `dataset_ml.csv`: baseline tabular y comparacion
- `windows`: entrada final para `CNN/LSTM`

## Hidrogramas embebidos en el archivo `.inp`

### Respuesta corta

Si, es valido que la red ya tenga los inflows e hidrogramas metidos dentro del
`.inp`.

De hecho, para realismo hidraulico esa es la ruta esperada del proyecto.

### Pero hay una implicacion importante

Si dejas los hidrogramas embebidos en el `.inp`, el pipeline tabular actual deja
de representar bien el escenario usando solo estas columnas:

- `delta_inflow_lps`
- `inflow_multiplier`

Eso alcanza para escenarios simples, pero no describe bien:

- hidrogramas distintos por nodo
- desfases temporales
- formas distintas con el mismo pico
- duraciones distintas
- varios aportes temporales dentro de una misma corrida

### Conclusión técnica

Para simulacion hidraulica: si, los hidrogramas dentro del `.inp` estan bien.

Para machine learning temporal: si haces eso, necesitas cambiar como se guardan
los datos de entrada y no depender solo del esquema tabular actual.

## Qué habría que cambiar si los hidrogramas van dentro del `.inp`

### 1. No confiar solo en parsear `[INFLOWS]`

Hoy `swmm_resilience/simulation/runner.py` hace una lectura minima de
`[INFLOWS]` tomando el ultimo valor numerico. Eso no alcanza para reconstruir
hidrogramas complejos definidos en el `.inp`.

Por eso, para el flujo temporal, recomiendo:

- usar el `.inp` como fuente de simulacion
- usar la simulacion misma como fuente de datos
- guardar por tiempo lo que realmente vio el modelo

Es mejor persistir:

- `node.total_inflow`
- `node.lateral_inflow`
- `node.depth`
- `node.flooding`
- `node.total_outflow`

que intentar reconstruir toda la logica de inflows leyendo el texto del `.inp`.

### 2. Mantener el tabular actual como resumen

No hace falta botar la parte tabular.

La recomendacion es:

- dejar `runs`, `run_inputs`, `node_results`, `run_summary` como resumen
- agregar un almacenamiento temporal nuevo para entrenamiento profundo

### 3. Ajustar la semantica de `run_inputs`

Si la fuente principal del escenario es el `.inp`, entonces `run_inputs` ya no
deberia entenderse como "el caudal completo aplicado al nodo", sino como una de
estas dos opciones:

- opcion A: dejarla solo para escenarios sintéticos o perturbaciones adicionales
- opcion B: convertirla en metadata de escenario, no en descripcion completa del
  hidrograma

Mi recomendacion es la opcion A.

### 4. Agregar metadata de escenario

Si los hidrogramas van dentro del `.inp`, conviene guardar por corrida:

- `input_source = inp_embedded`
- `hydrograph_profile_id`
- `hydrograph_peak_lps`
- `hydrograph_duration_min`
- `hydrograph_volume_m3`

Estas columnas no reemplazan la serie temporal real, pero ayudan mucho a
organizar y filtrar corridas.

## Plan de implementacion del cambio en la estructura tabular

### Objetivo del cambio

Ajustar la estructura tabular para que siga siendo util cuando los hidrogramas
viven dentro del `.inp`, sin fingir que una sola columna como
`delta_inflow_lps` describe completamente el escenario.

La idea no es borrar el enfoque tabular, sino cambiar su rol:

- de "descripcion completa del input hidraulico"
- a "resumen estructurado del escenario y de sus resultados"

Y dejar explicito que:

- el tabular debe derivarse de la simulacion y, cuando exista, de la capa por
  timestep
- el tabular no es la fuente de entrenamiento de `CNN/LSTM`

### Principio de diseño

La tabla tabular debe guardar:

- metadata del escenario
- resumenes agregados del comportamiento hidraulico
- variables estaticas por nodo
- targets de entrenamiento tabular

La tabla tabular no debe intentar reconstruir:

- toda la forma temporal del hidrograma
- desfases entre nodos
- series temporales completas

Eso debe quedar en la capa temporal.

### Cambio conceptual recomendado

#### Estructura actual

Hoy el pipeline tabular mezcla dos ideas:

- el escenario de entrada
- el resumen de salida

Y la parte de entrada se representa con columnas como:

- `delta_inflow_lps`
- `inflow_multiplier`

Eso funciona para barridos simples, pero no para hidrogramas embebidos.

#### Estructura objetivo

Separar claramente:

- `metadata de corrida`
- `metadata de entrada por nodo`
- `resultados agregados por nodo`
- `resultados agregados por corrida`

Y aceptar que la serie temporal real vive fuera del dataset tabular.

### Diseño tabular objetivo

#### 1. Tabla `runs`

Debe quedar como descriptor de la corrida y del escenario.

Columnas recomendadas:

```text
run_id
network_file
network_hash
scenario_type
spatial_pattern
input_source
inflow_multiplier
hydrograph_profile_id
hydrograph_peak_lps
hydrograph_duration_min
hydrograph_volume_m3
executed_at
status
```

Cambios clave:

- mantener `inflow_multiplier` como campo general de escala
- deprecar el uso de `delta_inflow_lps` como descriptor principal del escenario
- agregar `input_source` para distinguir:
  - `generated_uniform`
  - `inp_embedded`
  - `inp_embedded_plus_generated`

#### 2. Tabla `run_inputs`

Debe dejar de interpretarse como "descripcion completa del hidrograma del nodo".

Nuevo rol:

- registrar perturbaciones adicionales aplicadas por el pipeline
- registrar parametros de entrada sinteticos por nodo

Columnas recomendadas:

```text
input_id
run_id
node_uid
input_mode
base_inflow_lps
added_inflow_lps
inflow_multiplier
hydrograph_profile_id
notes
```

Regla:

- si el escenario viene solo desde el `.inp`, `run_inputs` puede quedar vacia
  o guardar solo metadata minima
- si el pipeline agrega perturbaciones, ahi si se llena

#### 3. Tabla `node_results`

Debe seguir siendo el corazon del dataset tabular.

Debe guardar:

- features agregadas por nodo
- targets de clasificacion
- targets de regresion
- resumenes derivados de la serie temporal

Columnas agregadas recomendadas:

```text
max_total_inflow_lps
max_lateral_inflow_lps
avg_depth_m
avg_depth_ratio
peak_flooding_lps
time_to_first_flood_min
time_over_threshold_min
peak_outflow_to_peak_inflow_ratio
```

Idea:

- estas columnas permiten que el modelo tabular siga siendo competitivo
- no reemplazan la serie temporal, pero resumen mejor una corrida con hidrograma

#### 4. Tabla `run_summary`

Debe quedar como resumen ejecutivo de la corrida.

Columnas recomendadas:

```text
summary_id
run_id
inflow_multiplier
failed_nodes_count
total_nodes
pct_flooded_nodes
total_flooding_volume_m3
time_to_first_flood_min
resilience_index
peak_system_outflow_lps
peak_system_flooding_lps
```

#### 5. Dataset exportado `dataset_ml.csv`

Debe seguir siendo plano, una fila por:

- `run_id`
- `node_id`

Pero con una semantica distinta:

- el CSV ya no es la descripcion completa del hidrograma
- el CSV es una vista agregada por nodo lista para modelos tabulares

### Esquema tabular objetivo v2

Esta seccion define el esquema objetivo de forma mas precisa. No significa que
todo deba implementarse en un solo commit, pero si fija la estructura final a
la que deberiamos converger.

#### Reglas generales

- se mantiene `run_id` como llave principal de corrida
- se mantiene `node_id` o `node_uid` como identificador de nodo
- se preserva compatibilidad con `delta_inflow_lps` durante la migracion
- las columnas nuevas deben tener semantica estable y no depender de un tipo de
  escenario demasiado especifico

#### Tabla `runs` v2

Rol:

- describir de forma compacta el escenario ejecutado

SQL propuesto:

```sql
CREATE TABLE runs (
    run_id                      TEXT PRIMARY KEY,
    network_file                TEXT NOT NULL,
    network_hash                TEXT NOT NULL,
    scenario_type               TEXT NOT NULL,
    spatial_pattern             TEXT NOT NULL,
    input_source                TEXT NOT NULL,
    inflow_multiplier           REAL NOT NULL DEFAULT 1,
    delta_inflow_lps            REAL,
    hydrograph_profile_id       TEXT,
    hydrograph_peak_lps         REAL,
    hydrograph_duration_min     REAL,
    hydrograph_volume_m3        REAL,
    embedded_inflow_nodes_count INTEGER,
    generated_input_nodes_count INTEGER,
    executed_at                 TEXT NOT NULL,
    status                      TEXT NOT NULL DEFAULT 'pending'
);
```

Notas:

- `input_source` debe aceptar solo valores controlados:
  - `generated_uniform`
  - `inp_embedded`
  - `inp_embedded`
  - `inp_embedded_plus_generated`
- `delta_inflow_lps` queda como columna legacy o representativa
- `hydrograph_profile_id` referencia la forma principal del hidrograma si existe

#### Tabla `hydrograph_profiles` v2

Rol:

- catalogar hidrogramas para no duplicar metadata en cada corrida

SQL propuesto:

```sql
CREATE TABLE hydrograph_profiles (
    hydrograph_profile_id   TEXT PRIMARY KEY,
    profile_source          TEXT NOT NULL,
    profile_name            TEXT,
    profile_hash            TEXT,
    profile_scope           TEXT NOT NULL,
    peak_lps                REAL,
    duration_min            REAL,
    volume_m3               REAL,
    series_file             TEXT,
    created_at              TEXT NOT NULL
);
```

Notas:

- `profile_source`: `inp_embedded`, `generated`
- `profile_scope`: `global`, `node_specific`
- `series_file` debe apuntar al archivo donde esta la serie original cuando
  aplique

#### Tabla `run_inputs` v2

Rol:

- registrar entradas sinteticas o perturbaciones aplicadas por el pipeline
- no describir por si sola el hidrograma real completo del nodo

SQL propuesto:

```sql
CREATE TABLE run_inputs (
    input_id                 TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL REFERENCES runs(run_id),
    node_uid                 TEXT NOT NULL,
    input_mode               TEXT NOT NULL,
    base_inflow_lps          REAL,
    added_inflow_lps         REAL,
    inflow_multiplier        REAL NOT NULL DEFAULT 1,
    delta_inflow_lps         REAL,
    hydrograph_profile_id    TEXT,
    profile_peak_lps         REAL,
    profile_duration_min     REAL,
    profile_volume_m3        REAL,
    notes                    TEXT
);
```

Notas:

- `input_mode` sugerido:
  - `none`
  - `generated_uniform`
  - `inp_embedded`
  - `embedded_only`
  - `embedded_plus_generated`
- si el escenario proviene solo del `.inp`, esta tabla puede tener:
  - cero filas
  - o una fila minima por nodo afectado

#### Tabla `network_nodes` v2

Rol:

- mantener los atributos estaticos de la red

Estado:

- la tabla actual es suficiente para la primera etapa
- no requiere rediseño grande inmediato

Ampliaciones opcionales futuras:

- `storage_curve_type`
- `is_outfall`
- `is_divider`
- `base_dwf_lps`
- `base_rdii_lps`

#### Tabla `node_results` v2

Rol:

- ser la base principal del dataset tabular por nodo y corrida

SQL propuesto:

```sql
CREATE TABLE node_results (
    result_id                           TEXT PRIMARY KEY,
    run_id                              TEXT NOT NULL REFERENCES runs(run_id),
    node_id                             TEXT NOT NULL,
    inflow_multiplier                   REAL NOT NULL DEFAULT 1,
    delta_inflow_lps                    REAL,
    flooded                             INTEGER NOT NULL DEFAULT 0,
    flooding_volume_m3                  REAL,
    flooding_duration_min               REAL,
    max_depth_m                         REAL,
    max_depth_ratio                     REAL,
    time_to_peak_min                    REAL,
    depth_rate_m_per_min                REAL,
    max_total_inflow_lps                REAL,
    max_lateral_inflow_lps              REAL,
    avg_depth_m                         REAL,
    avg_depth_ratio                     REAL,
    peak_flooding_lps                   REAL,
    time_to_first_flood_min             REAL,
    time_over_threshold_min             REAL,
    max_total_outflow_lps               REAL,
    time_to_peak_outflow_min            REAL,
    peak_outflow_to_peak_inflow_ratio   REAL,
    downstream_link_peak_flows_lps_json TEXT
);
```

Notas:

- `avg_depth_ratio` = promedio temporal de `depth / full_depth`
- `time_over_threshold_min` requiere fijar un umbral de severidad
- `peak_outflow_to_peak_inflow_ratio` ayuda a resumir respuesta dinamica del nodo

#### Tabla `link_results` v2

Rol:

- mantener agregados hidraulicos por link y permitir analisis complementario

SQL propuesto:

```sql
CREATE TABLE link_results (
    result_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES runs(run_id),
    link_id                 TEXT NOT NULL,
    inflow_multiplier       REAL NOT NULL DEFAULT 1,
    delta_inflow_lps        REAL,
    max_flow_lps            REAL,
    max_velocity_mps        REAL,
    max_depth_m             REAL,
    max_capacity_ratio      REAL,
    max_reverse_flow_lps    REAL,
    surcharged              INTEGER NOT NULL DEFAULT 0,
    time_full_flow_hrs      REAL
);
```

Notas:

- `max_reverse_flow_lps` es util si el hidrograma genera retrocesos o inversión
  de flujo

#### Tabla `run_summary` v2

Rol:

- resumen ejecutivo de la corrida

SQL propuesto:

```sql
CREATE TABLE run_summary (
    summary_id                  TEXT PRIMARY KEY,
    run_id                      TEXT NOT NULL REFERENCES runs(run_id),
    inflow_multiplier           REAL NOT NULL DEFAULT 1,
    failed_nodes_count          INTEGER,
    total_nodes                 INTEGER,
    total_flooding_volume_m3    REAL,
    pct_flooded_nodes           REAL,
    time_to_first_flood_min     REAL,
    resilience_index            REAL,
    peak_system_outflow_lps     REAL,
    peak_system_flooding_lps    REAL
);
```

Notas:

- `peak_system_outflow_lps` y `peak_system_flooding_lps` se calculan a nivel de
  corrida
- `failed_nodes_count` reemplaza cualquier conteo heredado basado en nombres
  ambiguos

#### Vista exportada `dataset_ml.csv` v2

Rol:

- alimentar modelos tabulares con una fila por nodo y corrida

Claves:

- `run_id`
- `node_id`

Columnas de contexto de corrida:

- `network_hash`
- `scenario_type`
- `spatial_pattern`
- `input_source`
- `inflow_multiplier`
- `hydrograph_peak_lps`
- `hydrograph_duration_min`
- `hydrograph_volume_m3`

Columnas estaticas:

- columnas de `network_nodes`

Columnas agregadas por nodo:

- columnas de `node_results`, excepto aquellas que se decidan excluir por fuga
  de informacion en cada experimento

#### Companion temporal no tabular

Aunque no es parte del dataset tabular, el diseño objetivo necesita una
referencia clara a los artefactos temporales.

Tabla sugerida:

```sql
CREATE TABLE temporal_artifacts (
    run_id               TEXT PRIMARY KEY REFERENCES runs(run_id),
    node_series_file     TEXT,
    link_series_file     TEXT,
    file_format          TEXT NOT NULL DEFAULT 'parquet',
    timestep_sec         REAL,
    total_steps          INTEGER,
    created_at           TEXT NOT NULL
);
```

Motivo:

- evita que la capa temporal quede "invisible"
- permite enlazar una corrida tabular con sus series reales

#### Relacion entre capa temporal y capa tabular

La relacion correcta debe ser:

- `runs` describe el escenario
- `temporal_artifacts` apunta a las series crudas por timestep
- `node_results` resume por nodo lo que ocurrio en esas series
- `dataset_ml.csv` exporta una vista derivada de `runs + network_nodes + node_results`

En otras palabras:

- el resumen tabular deberia derivarse del comportamiento temporal observado
- no deberia inventarse como si fuera una descripcion completa del hidrograma

### Estrategia de compatibilidad

#### Mantener compatibilidad hacia atras

Durante una fase de transicion conviene:

- conservar `delta_inflow_lps` en `runs`, `run_inputs`, `node_results` y
  `link_results` si todavia lo usan partes del proyecto
- dejar de usarlo como columna principal en nuevas consultas y nuevos modelos
- marcarlo en documentacion como `legacy` o `representative only`

#### Regla de transicion

- lo nuevo debe leer `inflow_multiplier` y `input_source`
- lo viejo puede seguir leyendo `delta_inflow_lps` mientras se migra

### Backlog detallado de implementacion

El siguiente backlog esta ordenado por dependencias reales. La idea es que cada
tarea deje el sistema corriendo, sin exigir una migracion total en un solo paso.

#### Epic 0. Formalizar la capa por timestep

##### Tarea 0.1. Definir contrato de series por nodo

Archivos:

- `PLAN_TEMPORAL_LSTM_CNN.md`
- `swmm_resilience/ml/temporal/dataset.py`
- `README.md`

Cambios exactos:

- fijar columnas obligatorias de `node_timeseries`
- fijar unidades
- fijar llaves:
  - `run_id`
  - `node_id`
  - `step_index`

Resultado esperado:

- el proyecto tiene una definicion unica de la capa temporal cruda

Validacion:

- checklist documental completado

##### Tarea 0.2. Definir contrato de series por link

Archivos:

- `PLAN_TEMPORAL_LSTM_CNN.md`
- `swmm_resilience/ml/temporal/dataset.py`
- `README.md`

Cambios exactos:

- fijar columnas obligatorias de `link_timeseries`
- definir si se guarda siempre o solo cuando sea necesario

Resultado esperado:

- la capa temporal de links deja de ser ambigua

Validacion:

- checklist documental completado

##### Tarea 0.3. Definir politica de resampleo y ventanas

Archivos:

- `PLAN_TEMPORAL_LSTM_CNN.md`
- `swmm_resilience/ml/temporal/dataset.py`

Cambios exactos:

- fijar:
  - frecuencia objetivo de resampleo
  - `window_min`
  - `horizon_min`
  - `step_min`
- definir como manejar pasos irregulares

Resultado esperado:

- la construccion de ventanas ya no depende de supuestos implícitos

Validacion:

- ejemplo documentado de una ventana temporal

#### Epic 1. Definir el esquema v2 en código

##### Tarea 1.1. Extender `runs`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/database/repository.py`
- `README.md`
- `view_db.py`

Cambios exactos:

- agregar en `runs`:
  - `input_source`
  - `hydrograph_profile_id`
  - `hydrograph_peak_lps`
  - `hydrograph_duration_min`
  - `hydrograph_volume_m3`
  - `embedded_inflow_nodes_count`
  - `generated_input_nodes_count`
- mantener `delta_inflow_lps` como legacy

Reglas de migracion:

- si `scenario_type` es steady y no hay hidrograma, `input_source` debe quedar
  como `generated_uniform`
- si `scenario_type` es `hydrograph_inflow`, `input_source`
  debe quedar como `inp_embedded`
- las corridas antiguas sin esta metadata deben completarse con defaults
  razonables

Resultado esperado:

- la tabla `runs` puede describir bien el origen del input aun cuando el
  hidrograma este dentro del `.inp`

Validacion:

- abrir una BD vieja y comprobar que migra
- crear una BD nueva y comprobar que el esquema coincide con `runs v2`

##### Tarea 1.2. Crear `hydrograph_profiles`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/database/repository.py`
- `README.md`

Cambios exactos:

- crear tabla nueva `hydrograph_profiles`
- definir helpers para insertar o reutilizar perfiles por `profile_hash`

Resultado esperado:

- los hidrogramas dejan de representarse como texto suelto o metadata duplicada

Validacion:

- insertar un perfil de prueba
- reutilizarlo en dos corridas distintas sin duplicar metadata

##### Tarea 1.3. Redefinir `run_inputs`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/database/repository.py`
- `README.md`

Cambios exactos:

- agregar:
  - `input_mode`
  - `base_inflow_lps`
  - `added_inflow_lps`
  - `hydrograph_profile_id`
  - `profile_peak_lps`
  - `profile_duration_min`
  - `profile_volume_m3`
  - `notes`
- conservar `delta_inflow_lps` mientras exista compatibilidad legacy

Reglas de poblamiento:

- si el escenario es solo `.inp`, `run_inputs` puede ir vacia
- si hay inyeccion adicional del pipeline, registrar esa perturbacion

Resultado esperado:

- `run_inputs` deja de mentir sobre el input real del nodo

Validacion:

- caso `embedded_only`
- caso `generated_uniform`
- caso `inp_embedded_plus_generated`

##### Tarea 1.4. Extender `node_results`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/database/repository.py`
- `README.md`

Cambios exactos:

- agregar:
  - `max_total_inflow_lps`
  - `max_lateral_inflow_lps`
  - `avg_depth_m`
  - `avg_depth_ratio`
  - `peak_flooding_lps`
  - `time_to_first_flood_min`
  - `time_over_threshold_min`
  - `peak_outflow_to_peak_inflow_ratio`

Resultado esperado:

- el dataset tabular resume mejor la dinamica sin necesitar la serie completa

Validacion:

- corrida corta de prueba
- comprobar que las columnas nuevas tienen valores no nulos cuando corresponde

##### Tarea 1.5. Extender `link_results` y `run_summary`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/database/repository.py`

Cambios exactos:

- agregar `max_reverse_flow_lps` a `link_results`
- agregar `peak_system_outflow_lps` y `peak_system_flooding_lps` a
  `run_summary`

Resultado esperado:

- mejores variables agregadas de comportamiento global

Validacion:

- corrida con posible reversa o backwater
- resumen de corrida con nuevos picos del sistema

#### Epic 2. Actualizar el runner para poblar el esquema nuevo

##### Tarea 2.1. Detectar el origen del escenario

Archivos:

- `swmm_resilience/main.py`
- `swmm_resilience/simulation/runner.py`

Cambios exactos:

- definir una rutina que clasifique cada corrida en:
  - `generated_uniform`
  - `inp_embedded`
  - `inp_embedded`
  - `inp_embedded_plus_generated`

Resultado esperado:

- cada corrida queda etiquetada con una semantica clara

Validacion:

- prueba por cada tipo de escenario

##### Tarea 2.2. Mejorar el parseo minimo del `.inp`

Archivos:

- `swmm_resilience/simulation/runner.py`

Cambios exactos:

- dejar de depender solo del ultimo valor numerico en `[INFLOWS]`
- extraer al menos conteos y metadata minima del uso de inflows embebidos

Regla:

- no intentar reconstruir toda la serie temporal desde texto si eso complica de
  mas
- usar el parseo para metadata y la simulacion para resultados reales

Resultado esperado:

- poder identificar si una red trae inflows embebidos y cuantos nodos afecta

Validacion:

- `.inp` sin inflows
- `.inp` con inflows simples
- `.inp` con hidrograma o referencia temporal

##### Tarea 2.3. Calcular nuevos agregados por nodo

Archivos:

- `swmm_resilience/simulation/runner.py`

Cambios exactos:

- durante el loop temporal acumular:
  - maximos
  - promedios
  - primer tiempo de flooding
  - tiempo sobre umbral
  - relacion entre picos de entrada y salida

Resultado esperado:

- `node_results` sale listo para el nuevo exportador tabular

Validacion:

- revisar varias filas de `node_results`
- verificar que los promedios y maximos son coherentes

##### Tarea 2.4. Calcular metadata global del hidrograma

Archivos:

- `swmm_resilience/main.py`
- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/database/repository.py`

Cambios exactos:

- poblar en `runs` y, cuando aplique, en `hydrograph_profiles`:
  - pico
  - duracion
  - volumen

Resultado esperado:

- las corridas quedan filtrables por propiedades fisicas del hidrograma

Validacion:

- comparar contra el CSV o perfil de entrada original

#### Epic 3. Persistir series reales por timestep

##### Tarea 3.1. Crear `temporal_artifacts`

Archivos:

- `swmm_resilience/database/schema.py`
- `swmm_resilience/database/repository.py`

Cambios exactos:

- registrar por `run_id` donde estan las series temporales

Resultado esperado:

- cada corrida tabular puede enlazarse con sus datos temporales reales

Validacion:

- insertar artefacto temporal de prueba y leerlo desde la BD

##### Tarea 3.2. Persistir `node_timeseries`

Archivos:

- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/ml/temporal/dataset.py`

Cambios exactos:

- guardar por timestep:
  - `step_index`
  - `time_sec`
  - `time_min`
  - `total_inflow_lps`
  - `lateral_inflow_lps`
  - `depth_m`
  - `depth_ratio`
  - `head_m`
  - `flooding_lps`
  - `total_outflow_lps`
  - `failed_now`

Resultado esperado:

- existe la serie cruda necesaria para `CNN/LSTM`

Validacion:

- abrir un Parquet y revisar columnas, orden temporal y tipos

##### Tarea 3.3. Persistir `link_timeseries`

Archivos:

- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/ml/temporal/dataset.py`

Cambios exactos:

- guardar por timestep:
  - `flow_lps`
  - `depth_m`
  - `velocity_mps`
  - `capacity_ratio`

Resultado esperado:

- existe soporte temporal por link para analisis complementario

Validacion:

- abrir Parquet de links y revisar coherencia

#### Epic 4. Actualizar exportacion y ML tabular

##### Tarea 3.1. Rediseñar `dataset_ml.csv`

Archivos:

- `swmm_resilience/analysis/dataset.py`
- `README.md`

Cambios exactos:

- incluir metadata nueva de `runs`
- incluir agregados nuevos de `node_results`
- dejar `delta_inflow_lps` como campo legacy o excluirlo del dataset principal

Resultado esperado:

- el CSV exportado representa mejor corridas con hidrogramas embebidos

Validacion:

- exportar un CSV de prueba
- revisar columnas y nulos

##### Tarea 3.2. Redefinir espacio de features tabulares

Archivos:

- `swmm_resilience/config.py`
- `swmm_resilience/ml/preprocessing.py`
- `swmm_resilience/ml/train.py`

Cambios exactos:

- revisar `ML_DROP_COLUMNS`
- decidir que columnas nuevas son:
  - metadata
  - features
  - targets
  - leakage

Resultado esperado:

- el entrenamiento tabular sigue siendo valido y entendible

Validacion:

- correr entrenamiento baseline
- comparar contra baseline anterior

##### Tarea 3.3. Actualizar prediccion tabular

Archivos:

- `swmm_resilience/ml/predict_tabular.py`

Cambios exactos:

- dejar de asumir una unica columna equivalente a `delta_inflow_lps`
- preferir `inflow_multiplier`
- usar `input_source` para detectar como construir escenarios de inferencia

Resultado esperado:

- la inferencia no se rompe cuando cambia la semantica del input

Validacion:

- inferencia con dataset legacy
- inferencia con dataset v2

#### Epic 5. Construir ventanas y entrenar temporal

##### Tarea 5.1. Construir ventanas desde `node_timeseries`

Archivos:

- `swmm_resilience/ml/temporal/dataset.py`

Cambios exactos:

- construir tensores `[samples, timesteps, features]`
- usar `run_id` como grupo
- opcionalmente adjuntar features estaticas

Resultado esperado:

- dataset listo para `CNN/LSTM`

Validacion:

- revisar shapes
- revisar targets y balance de clases

##### Tarea 5.2. Entrenar baseline `CNN 1D`

Archivos:

- `swmm_resilience/ml/temporal/train_cnn.py`

Cambios exactos:

- entrenar la primera CNN baseline
- guardar artefactos y metricas

Resultado esperado:

- existe un baseline temporal real

Validacion:

- entrenamiento completo sobre una muestra de corridas

##### Tarea 5.3. Entrenar benchmark `LSTM`

Archivos:

- `swmm_resilience/ml/temporal/train_cnn.py`
- o nuevo archivo `train_lstm.py` si se decide separarlo

Cambios exactos:

- entrenar un modelo LSTM usando exactamente la misma entrada y targets

Resultado esperado:

- comparacion justa entre `CNN` y `LSTM`

Validacion:

- tabla comparativa de metricas

#### Epic 6. Convertir el backlog en entregables cortos

##### Sprint 1

- capa por timestep definida
- `runs v2`
- `run_inputs v2`
- migraciones
- documentación

##### Sprint 2

- persistencia `node_timeseries`
- persistencia `link_timeseries`
- `temporal_artifacts`
- `node_results v2`
- `link_results v2`
- `run_summary v2`
- exportador actualizado

##### Sprint 3

- actualización del entrenamiento tabular
- predicción tabular v2
- validación sobre escenarios legacy y nuevos

##### Sprint 4

- ventanas para CNN/LSTM
- baseline `CNN`
- benchmark `LSTM`

##### Sprint 5

- predictor temporal
- evaluación de series de tiempo
- preparación para búsqueda de alternativas

### Checklist de aceptación final

- una corrida con inflows embebidos en `.inp` puede guardarse sin ambiguedad
- existe una capa cruda por `timesteps`
- las series por timestep se pueden vincular con `run_id`
- el dataset tabular v2 sigue teniendo una fila por `run_id + node_id`
- `delta_inflow_lps` deja de ser la columna principal del escenario
- `input_source` y metadata del hidrograma quedan visibles
- el entrenamiento tabular sigue funcionando
- `CNN/LSTM` se entrenan desde ventanas derivadas de `timesteps`

### Riesgos a controlar

#### Riesgo 1. Fuga de informacion

Si agregas demasiados resumenes de salida como `features`, el modelo tabular
puede quedar artificialmente bueno.

Regla:

- cualquier variable claramente posterior a la falla debe evaluarse con cuidado

#### Riesgo 2. Mezclar escenarios incomparables

No conviene entrenar como si fueran iguales:

- corridas steady
- corridas con hidrograma embebido en `.inp`

Recomendacion:

- agregar `input_source`
- evaluar por subset

#### Riesgo 3. Crecimiento desordenado del esquema

No agregar muchas columnas sin semantica estable.

Primero definir:

- que columna es entrada
- que columna es metadata
- que columna es target
- que columna es solo analitica

### Criterio de exito del cambio tabular

El cambio se considera bien implementado si:

- una corrida con hidrogramas dentro del `.inp` puede guardarse sin ambiguedad
- el dataset tabular sigue sirviendo para baseline y analisis rapido
- la metadata del escenario queda clara sin depender de `delta_inflow_lps`
- el pipeline temporal puede convivir sin duplicar mal la informacion

### Recomendacion practica final

No intentaria hacer todo en un solo salto.

Orden practico:

1. definir formalmente la capa por `timesteps`
2. rediseñar `runs` y `run_inputs`
3. persistir series por timestep
4. derivar mejores agregados para `node_results`
5. actualizar `dataset_ml.csv`
6. entrenar `CNN/LSTM` sobre ventanas

Ese orden minimiza riesgo y te deja una base tabular mejor incluso antes de
entrenar la primera CNN o LSTM, pero ya sin perder de vista que la capa por
`timesteps` es la fuente primaria del flujo temporal.

## Estrategia de migracion recomendada

### Fase 0. Consolidar objetivos

Definir que es "falla".

Recomendacion inicial:

- `failed_now = 1` si `flooding > 0`
- luego se puede endurecer con umbrales de profundidad, volumen o duracion

### Fase 1. Persistir series temporales

Cambios principales:

- ampliar `swmm_resilience/simulation/runner.py`
- guardar series por nodo y opcionalmente por link
- exportarlas a Parquet por corrida

### Fase 2. Derivar agregados tabulares desde la simulacion

Cambios principales:

- mejorar `node_results`
- mejorar `run_summary`
- actualizar metadata en `runs`

### Fase 3. Construir ventanas

Implementar en:

- `swmm_resilience/ml/temporal/dataset.py`

Salida esperada:

- `X_temporal`
- `X_static`
- `y_classification`
- `y_regression`
- `groups`

Agrupacion:

- por `run_id`
- luego, cuando haya varias redes, tambien validar por `network_hash`

### Fase 4. Entrenar baseline temporal

Implementar primero:

- `CNN 1D`

Comparar despues con:

- `LSTM`

### Fase 5. Prediccion operativa

Implementar en:

- `swmm_resilience/ml/temporal/predict.py`

Salida deseada:

- timeline de riesgo por nodo
- probabilidad de falla
- tiempo a falla
- severidad esperada

### Fase 6. Busqueda de alternativas

Una vez exista un surrogate temporal razonable:

- generar alternativas de operacion o diseño
- evaluarlas primero con el surrogate
- validar solo las mejores con PySWMM

Objetivos posibles:

- minimizar `failed_nodes_count`
- minimizar `total_flooding_volume_m3`
- minimizar costo
- minimizar tiempo de recuperacion

## Estrategia de evaluacion

### Para clasificacion

- `F1`
- `precision`
- `recall`
- `PR-AUC`
- tiempo medio de anticipacion

### Para regresion

- `MAE`
- `RMSE`
- error en `time_to_failure_min`

### Para busqueda de alternativas

- reduccion de nodos fallidos
- reduccion de volumen inundado
- robustez frente a distintos hidrogramas
- costo computacional

## Recomendacion final

La mejor ruta para este repo es:

1. conservar el pipeline tabular actual
2. aceptar que los hidrogramas esten dentro del `.inp`
3. crear primero la capa cruda por `timesteps`
4. derivar desde ahi un tabular mejor y ventanas temporales
5. empezar con `CNN 1D`
6. usar `LSTM` como comparacion
7. despues construir el surrogate para exploracion de alternativas

## Siguiente paso concreto

El siguiente entregable tecnico deberia ser implementar la primera parte de ese
diseño:

- agregar el esquema `runs v2`
- agregar `run_inputs v2`
- crear `temporal_artifacts`
- persistir `node_timeseries` por corrida con `step_index`, `time_sec` y
  variables hidraulicas clave

Sin esa capa por `timesteps`, la implementacion de la `CNN` o la `LSTM`
seguiría prematura.
