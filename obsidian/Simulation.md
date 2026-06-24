# Simulation

> Ejecuta simulaciones SWMM vía PySWMM y extrae resultados de nodos

**Ruta:** `swmm_resilience/simulation/`

## Archivos clave

- [[runner.py]] — Orquesta simulaciones y extrae topología estática y series temporales
- [[swmm_api_io.py]] — Adaptador de IO para la API nativa de SWMM

## Recibe datos de
[[Config]] · [[INP File]]

## Produce
[[SQLite DB]] · [[Parquets Timeseries]]

## Depende de
[[Config]]
