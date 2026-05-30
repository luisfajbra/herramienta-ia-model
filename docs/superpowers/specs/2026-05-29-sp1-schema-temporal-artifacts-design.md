# SP1 — Schema v2 + Registro de Artefactos Temporales

## Objetivo

Agregar la tabla `temporal_artifacts` a SQLite y la columna `input_source` a `runs`, de modo que cada corrida con hidrograma quede vinculada a su archivo Parquet de series temporales. La persistencia Parquet ya existe; lo que falta es el registro en la base de datos.

---

## Contexto

El runner (`swmm_resilience/simulation/runner.py`) ya recolecta series temporales por nodo a cada timestep y las devuelve en `results["node_timeseries_records"]`. `main.py` ya las guarda en Parquet dentro de:

```
data/networks/<net>/results/temporal/node_timeseries/run_<run_id>.parquet
```

Lo que no existe todavía:
- Tabla `temporal_artifacts` en `swmm_resilience.db`
- Columna `input_source` en `runs`
- Llamada en `main.py` que registre la ruta del Parquet en `temporal_artifacts` después de guardarlo
- Limpieza de `temporal_artifacts` en `reset.py`

---

## Schema objetivo

### Columna nueva en `runs`

```sql
ALTER TABLE runs ADD COLUMN input_source TEXT NOT NULL DEFAULT 'steady';
-- Valores: 'steady' | 'hydrograph'
```

Motivo: distinguir corridas de escenario estacionario (pipeline tabular) de corridas con hidrograma (pipeline temporal). Esta columna no se usa aún para lógica de rama, solo para filtrado en el constructor de ventanas (SP2).

### Tabla nueva `temporal_artifacts`

```sql
CREATE TABLE IF NOT EXISTS temporal_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    network_hash    TEXT NOT NULL,
    parquet_path    TEXT NOT NULL,
    node_count      INTEGER NOT NULL,
    step_count      INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
```

- `artifact_id`: UUID4 generado al insertar.
- `parquet_path`: ruta absoluta al archivo Parquet (permite mover los datos y actualizar el registro).
- `node_count` / `step_count`: estadísticas de tamaño registradas en el momento del guardado.

---

## Cambios en el código

### `swmm_resilience/database/schema.py`

1. Agregar el bloque `temporal_artifacts` a `SCHEMA_SQL` (al final, después de `run_summary`).
2. Agregar una función de migración `_migrate_add_temporal_artifacts(conn)` que:
   - Verifica si la tabla ya existe (compatible con bases de datos ya creadas).
   - Si no existe, la crea.
3. Agregar `input_source TEXT NOT NULL DEFAULT 'steady'` a `REQUIRED_COLUMNS["runs"]` para que el mecanismo de migración existente la agregue automáticamente.
4. Llamar a `_migrate_add_temporal_artifacts(conn)` desde `create_schema(conn)`.

### `swmm_resilience/database/queries.py` (archivo nuevo)

Función pública:

```python
def register_temporal_artifact(
    conn: sqlite3.Connection,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    node_count: int,
    step_count: int,
) -> str:
    """Inserta una fila en temporal_artifacts. Devuelve artifact_id."""
```

Separa la lógica SQL de `main.py` y facilita las pruebas.

### `swmm_resilience/main.py`

Justo después de que se llama a `save_node_timeseries_parquet(...)` (líneas 255-272), agregar:

```python
if parquet_path and run_id:
    conn = sqlite3.connect(db_path)
    try:
        register_temporal_artifact(
            conn,
            run_id=run_id,
            network_hash=network_hash,
            parquet_path=parquet_path,
            node_count=node_count,
            step_count=step_count,
        )
    finally:
        conn.close()
```

`node_count` y `step_count` se derivan del DataFrame guardado (ya disponible en ese bloque).

### `swmm_resilience/reset.py`

Agregar `"temporal_artifacts"` al inicio de `_DB_TABLES_IN_ORDER`, antes de `"node_results"`, para que se borre antes que las corridas que referencia:

```python
_DB_TABLES_IN_ORDER = [
    "temporal_artifacts",
    "node_results",
    "link_results",
    ...
]
```

---

## Contrato de interfaz hacia SP2

SP2 (constructor de ventanas) consulta `temporal_artifacts` con:

```sql
SELECT ta.run_id, ta.parquet_path, r.network_hash
FROM temporal_artifacts ta
JOIN runs r ON r.run_id = ta.run_id
WHERE r.status = 'completed'
  AND r.input_source = 'hydrograph'
ORDER BY ta.created_at;
```

Si SP1 no está completo, SP2 no puede descubrir los Parquet automáticamente.

---

## Pruebas

- `tests/database/test_temporal_artifacts.py`
  - `test_create_schema_creates_temporal_artifacts_table`: verifica que `create_schema` crea la tabla en una BD vacía.
  - `test_migrate_adds_table_to_existing_db`: arranca con el schema anterior (sin `temporal_artifacts`) y verifica que `create_schema` la agrega sin perder filas existentes.
  - `test_register_temporal_artifact_inserts_row`: llama a `register_temporal_artifact` y verifica la fila con `SELECT *`.
  - `test_register_temporal_artifact_returns_uuid`: el `artifact_id` devuelto es un UUID4 válido.
  - `test_reset_db_clears_temporal_artifacts`: ejecuta `reset_db()` y verifica que `temporal_artifacts` queda vacía.

---

## Lo que este sub-proyecto NO hace

- No modifica el runner para cambiar cómo se generan los Parquet.
- No agrega `hydrograph_profile_id`, `hydrograph_peak_lps` ni otras columnas de hidrograma al schema (esas pertenecen a la futura extensión del runner).
- No implementa la construcción de ventanas (SP2).
