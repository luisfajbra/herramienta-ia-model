# Visualization

> Mapas de inundación, red hidráulica y parser de archivos .inp

**Ruta:** `swmm_resilience/visualization/`

## Archivos clave

- [[flood_map.py]] — Genera mapa de inundación coloreado por volumen o probabilidad
- [[network_map.py]] — Visualiza la red hidráulica con nodos y conductos
- [[_inp_parser.py]] — Parsea coordenadas y conductos de archivos .inp de SWMM
- [[runner.py]] — Orquesta generación de mapas ML y SWMM

## Recibe datos de
[[Surrogate Weights]] · [[Tabular Model Artifacts]] · [[SQLite DB]] · [[INP File]]

## Produce
[[Surrogate Maps]]

## Depende de
[[Config]] · [[ML Temporal]] · [[ML Tabular]]
