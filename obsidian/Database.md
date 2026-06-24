# Database

> Esquema SQLite, consultas y repositorio para artefactos de simulación

**Ruta:** `swmm_resilience/database/`

## Archivos clave

- [[schema.py]] — Define y migra el esquema SQLite (runs, network_nodes, temporal_artifacts)
- [[queries.py]] — Consultas SQL parametrizadas para lectura de datos
- [[repository.py]] — Capa de repositorio que abstrae acceso a la DB

## Recibe datos de
[[Config]]

## Produce
[[SQLite DB]]

## Depende de
[[Config]]
