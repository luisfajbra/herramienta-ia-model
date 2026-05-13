# Diccionario de datos de simulacion y entrenamiento ML

## Resumen rapido

- La simulacion se ejecuta con `PySWMM`.
- El `.inp` y el `.rpt` se leen con `swmm-api`.
- La base SQLite de entrenamiento es `data/training/swmm_resilience.db`.
- El modelo tabular actual **no entrena directo desde SQLite**.
- Antes de entrenar, el proyecto exporta `dataset_ml.csv` desde SQLite.
- El entrenamiento actual usa solo columnas del `dataset_ml.csv`.

## Flujo actual de extraccion

1. Se lee el `.inp` con `swmm-api`.
2. Se extrae topologia estatica de nodos y links.
3. Se escribe un `.inp` temporal escalado si aplica.
4. Se corre la simulacion con `PySWMM`.
5. Durante la corrida se capturan variables dinamicas por nodo y por link.
6. Al final se leen estadisticas finales desde `PySWMM`.
7. Si esta habilitado, se relee el resumen de inundacion desde el `.rpt` y se sobreescriben:
   - `flooding_volume_m3`
   - `flooding_duration_min`
8. Se guardan tablas en SQLite.
9. Se exporta `dataset_ml.csv` para entrenamiento.

## Que se guarda en SQLite pero no entra hoy al modelo

Estas tablas quedan en la base de entrenamiento, pero **no se usan directamente** para entrenar el modelo tabular actual:

- `run_inputs`
- `network_links`
- `link_results`
- `run_summary`

Las tablas que si alimentan el dataset de ML son:

- `runs`
- `network_nodes`
- `node_results`

## Diccionario por tabla SQLite

### `runs`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `run_id` | codigo | no | si | UUID de la corrida | si |
| `network_file` | `.inp` seleccionado | no | no | nombre del archivo de red | si |
| `network_hash` | archivo `.inp` | no | si | hash MD5 del `.inp` | si |
| `scenario_type` | config/UI | no | si | etiqueta del escenario | si |
| `spatial_pattern` | config/UI | no | si | patron espacial del experimento | si |
| `delta_inflow_lps` | metadata del run | no | si | hoy se usa como metadata global; no es una salida hidraulica de SWMM | no |
| `inflow_multiplier` | config/UI | no | si | factor global de la corrida | si |
| `executed_at` | SQLite | no | si | timestamp de ejecucion | no |
| `status` | codigo | no | si | `running`, `completed`, `failed` | filtrado indirecto |

### `network_nodes`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `network_hash` | codigo | no | si | join con la red | join |
| `node_uid` | PySWMM | si | no | id del nodo | si como `node_id` por join |
| `invert_elev_m` | PySWMM | si | no | cota de solera | si |
| `full_depth_m` | PySWMM | si | no | profundidad total | si |
| `base_inflow_lps` | `.inp` `[INFLOWS]` | no | no | baseline por nodo | si |
| `node_type` | PySWMM | si | si | se normaliza a `junction`, `outfall`, etc. | si |
| `in_degree` | topologia | no | si | links que llegan al nodo | si |
| `out_degree` | topologia | no | si | links que salen del nodo | si |
| `upstream_pipes_count` | topologia | no | si | numero de links aguas arriba | si |
| `upstream_diam_max_m` | `.inp` + topologia | no | si | diametro max aguas arriba | si |
| `upstream_diam_min_m` | `.inp` + topologia | no | si | diametro min aguas arriba | si |
| `upstream_diam_avg_m` | `.inp` + topologia | no | si | diametro promedio aguas arriba | si |
| `upstream_slope_avg` | `.inp` + topologia | no | si | pendiente promedio aguas arriba | si |
| `upstream_slope_max` | `.inp` + topologia | no | si | pendiente maxima aguas arriba | si |
| `upstream_capacity_lps` | codigo | no | si | suma de capacidades teoricas aguas arriba | si |
| `downstream_pipes_count` | topologia | no | si | numero de links aguas abajo | si |
| `downstream_diam_max_m` | `.inp` + topologia | no | si | diametro max aguas abajo | si |
| `downstream_diam_min_m` | `.inp` + topologia | no | si | diametro min aguas abajo | si |
| `downstream_diam_avg_m` | `.inp` + topologia | no | si | diametro promedio aguas abajo | si |
| `downstream_slope_avg` | `.inp` + topologia | no | si | pendiente promedio aguas abajo | si |
| `downstream_slope_max` | `.inp` + topologia | no | si | pendiente maxima aguas abajo | si |
| `downstream_capacity_lps` | codigo | no | si | suma de capacidades teoricas aguas abajo | si |

### `network_links`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `network_hash` | codigo | no | si | join de red | no |
| `link_uid` | PySWMM | si | no | id del link | no |
| `inlet_node` | `.inp` / PySWMM | si | no | nodo de entrada | no |
| `outlet_node` | `.inp` / PySWMM | si | no | nodo de salida | no |
| `link_type` | PySWMM | si | si | conduit, weir, orifice, pump | no |
| `diameter_m` | `.inp` / PySWMM | si | no | geom1 o altura segun xsection | no directo |
| `length_m` | `.inp` | si | no | longitud | no directo |
| `roughness` | `.inp` | si | no | rugosidad | no directo |
| `slope_m_per_m` | codigo | no | si | calculada con cotas y offsets | no directo |
| `full_flow_capacity_lps` | codigo | no | si | Manning teorico a flujo lleno | no directo |

### `run_inputs`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `input_id` | codigo | no | si | UUID | no |
| `run_id` | codigo | no | si | join | no |
| `delta_inflow_lps` | codigo | no | si | delta calculado por nodo | no directo |
| `inflow_multiplier` | config/UI | no | si | factor del run | no directo |
| `node_uid` | PySWMM / codigo | si | no | nodo afectado | no directo |

### `node_results`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `result_id` | codigo | no | si | UUID | no |
| `run_id` | codigo | no | si | join | si |
| `delta_inflow_lps` | codigo | no | si | delta real por nodo | no hoy |
| `inflow_multiplier` | config/UI | no | si | redundante con `runs` | no en export actual |
| `node_id` | PySWMM | si | no | nodo de resultado | si |
| `flooded` | PySWMM / `.rpt` | parcialmente | si | bandera binaria final | si |
| `flooding_volume_m3` | `.rpt` o PySWMM fallback | si | si | si viene del `.rpt`, se convierte a m3 | si |
| `flooding_duration_min` | `.rpt` o PySWMM fallback | si | si | si viene del `.rpt`, se convierte a minutos | si |
| `max_depth_m` | PySWMM | si | no | maxima profundidad | si |
| `max_depth_ratio` | codigo | no | si | `max_depth_m / full_depth_m` | si |
| `time_to_peak_min` | PySWMM | si | si | conversion a minutos | si |
| `depth_rate_m_per_min` | codigo | no | si | maxima tasa de cambio de profundidad | si |
| `max_total_outflow_lps` | PySWMM timestep | si | si | pico de `node.total_outflow` | si |
| `time_to_peak_outflow_min` | PySWMM timestep | si | si | minuto del pico de salida total | si |
| `downstream_link_peak_flows_lps_json` | PySWMM timestep | si | si | mapa JSON de picos por link saliente | si |

### `link_results`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `result_id` | codigo | no | si | UUID | no |
| `run_id` | codigo | no | si | join | no |
| `delta_inflow_lps` | metadata | no | no | `NULL` para links | no |
| `inflow_multiplier` | config/UI | no | si | factor del run | no |
| `link_id` | PySWMM | si | no | id del link | no |
| `max_flow_lps` | PySWMM | si | no | pico de flujo | no |
| `max_velocity_mps` | PySWMM | si | no | pico de velocidad | no |
| `max_depth_m` | PySWMM | si | no | pico de profundidad en el link | no |
| `max_capacity_ratio` | codigo | no | si | `peak_flow / full_flow_capacity_lps` | no |
| `surcharged` | codigo | no | si | `time_full_flow_hrs > 0` | no |
| `time_full_flow_hrs` | PySWMM | si | no | tiempo a flujo lleno | no |

### `run_summary`

| Columna | Fuente | Sale directo de SWMM | Calculada en codigo | Notas | Entra al dataset ML |
|---|---|---:|---:|---|---:|
| `summary_id` | codigo | no | si | UUID | no |
| `run_id` | codigo | no | si | join | no directo |
| `inflow_multiplier` | config/UI | no | si | factor del run | no directo |
| `total_nodes` | codigo | no | si | total de nodos evaluados | no |
| `failed_nodes_count` | codigo | no | si | nodos con flooding | no |
| `total_flooding_volume_m3` | codigo | no | si | suma del flooding de todos los nodos | no |
| `pct_flooded_nodes` | codigo | no | si | porcentaje de nodos inundados | no |
| `time_to_first_flood_min` | PySWMM timestep | si | si | primer instante con flooding > 0 | no |
| `resilience_index` | codigo | no | si | `1 - failed_nodes_count / total_nodes` | no |

## Columnas del `dataset_ml.csv` y rol en ML

Estas son las columnas que el proyecto exporta hoy al dataset de entrenamiento.

| Columna en CSV | Tabla origen | Fuente final | Rol en ML hoy |
|---|---|---|---|
| `run_id` | `node_results` | codigo | agrupacion, no feature |
| `node_id` | `node_results` | PySWMM | identificador, no feature |
| `network_hash` | `runs` | codigo | trazabilidad, no feature |
| `network_file` | `runs` | `.inp` | trazabilidad, no feature |
| `inflow_multiplier` | `runs` | config/UI | **feature** |
| `scenario_type` | `runs` | config/UI | metadata, no feature |
| `spatial_pattern` | `runs` | config/UI | metadata, no feature |
| `invert_elev_m` | `network_nodes` | PySWMM | **feature** |
| `full_depth_m` | `network_nodes` | PySWMM | **feature** |
| `base_inflow_lps` | `network_nodes` | `.inp` | **feature** |
| `node_type` | `network_nodes` | PySWMM normalizado | exportada, no feature tabular actual |
| `in_degree` | `network_nodes` | codigo | exportada, hoy se descarta |
| `out_degree` | `network_nodes` | codigo | exportada, hoy se descarta |
| `upstream_pipes_count` | `network_nodes` | codigo | **feature** |
| `upstream_diam_max_m` | `network_nodes` | codigo | **feature** |
| `upstream_diam_min_m` | `network_nodes` | codigo | **feature** |
| `upstream_diam_avg_m` | `network_nodes` | codigo | exportada, hoy se descarta |
| `upstream_slope_avg` | `network_nodes` | codigo | **feature** |
| `upstream_slope_max` | `network_nodes` | codigo | **feature** |
| `upstream_capacity_lps` | `network_nodes` | codigo | exportada, hoy se descarta |
| `downstream_pipes_count` | `network_nodes` | codigo | **feature** |
| `downstream_diam_max_m` | `network_nodes` | codigo | **feature** |
| `downstream_diam_min_m` | `network_nodes` | codigo | **feature** |
| `downstream_diam_avg_m` | `network_nodes` | codigo | exportada, hoy se descarta |
| `downstream_slope_avg` | `network_nodes` | codigo | **feature** |
| `downstream_slope_max` | `network_nodes` | codigo | **feature** |
| `downstream_capacity_lps` | `network_nodes` | codigo | exportada, hoy se descarta |
| `max_depth_m` | `node_results` | PySWMM | exportada, hoy se descarta |
| `max_depth_ratio` | `node_results` | codigo | exportada, hoy se descarta |
| `time_to_peak_min` | `node_results` | PySWMM + conversion | exportada, hoy se descarta |
| `depth_rate_m_per_min` | `node_results` | codigo | exportada, hoy se descarta |
| `max_total_outflow_lps` | `node_results` | PySWMM + seguimiento | exportada, hoy se descarta |
| `time_to_peak_outflow_min` | `node_results` | PySWMM + seguimiento | exportada, hoy se descarta |
| `downstream_link_peak_flows_lps_json` | `node_results` | PySWMM + codigo | exportada, no feature |
| `flooded` | `node_results` | PySWMM / `.rpt` | **target clasificacion** |
| `flooding_volume_m3` | `node_results` | `.rpt` o PySWMM fallback | **target regresion** |
| `flooding_duration_min` | `node_results` | `.rpt` o PySWMM fallback | exportada, hoy se descarta |

## Features que hoy usa el modelo tabular

El pipeline actual usa estas columnas numericas como features:

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

## Targets que hoy usa ML

- Clasificacion:
  - `flooded`

- Regresion:
  - `flooding_volume_m3`

## Columnas exportadas pero excluidas del entrenamiento actual

Estas columnas quedan en el CSV y en la SQLite, pero el preprocesamiento actual las excluye del entrenamiento tabular:

- `run_id`
- `node_id`
- `scenario_type`
- `spatial_pattern`
- `delta_inflow_lps`
- `upstream_diam_avg_m`
- `downstream_diam_avg_m`
- `flooded` cuando no es el target
- `flooding_volume_m3` cuando no es el target
- `flooding_duration_min`
- `max_depth_m`
- `max_depth_ratio`
- `time_to_peak_min`
- `depth_rate_m_per_min`
- `max_total_outflow_lps`
- `time_to_peak_outflow_min`
- `in_degree`
- `out_degree`
- `upstream_capacity_lps`
- `downstream_capacity_lps`

## Nota importante

Aunque la SQLite guarda mas informacion que el CSV de entrenamiento, el estado actual del proyecto es:

- SQLite = almacenamiento historico de corridas y resultados
- `dataset_ml.csv` = dataset plano para ML
- modelo tabular actual = usa solo un subconjunto de columnas numericas del CSV

