# Plan de migracion parcial a `swmm_api`

Este plan cubre una migracion acotada: usar `swmm_api` para manipular archivos
`.inp` y extraer resultados de `.rpt/.out`, manteniendo PySWMM como motor de
simulacion mientras siga siendo util.

## Objetivo

- Reemplazar el parseo manual de `[INFLOWS]` y `[TIMESERIES]`.
- Reemplazar la escritura manual de `.inp` temporales escalados.
- Leer resultados finales desde `.rpt` y, cuando haga falta, series desde `.out`.
- Mantener la estructura actual de base de datos y dataset ML.

## No objetivos

- No reescribir toda la herramienta.
- No eliminar PySWMM de inmediato.
- No cambiar el esquema SQLite en esta primera migracion.
- No migrar todavia el entrenamiento ML ni la app completa.

## Estado actual relevante

El flujo hidraulico vive principalmente en:

- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/main.py`
- `swmm_resilience/database/repository.py`

Actualmente `runner.py`:

- parsea manualmente secciones del `.inp`;
- crea `.inp` temporales multiplicando texto en `[TIMESERIES]`;
- corre PySWMM;
- extrae resultados con `node.statistics` y `link.conduit_statistics`.

## Arquitectura propuesta

Crear un modulo nuevo:

```text
swmm_resilience/simulation/swmm_api_io.py
```

Responsabilidades:

- leer `.inp` con `swmm_api`;
- detectar nodos con inflows;
- escalar hidrogramas internos;
- escribir `.inp` temporal;
- leer resumen de inundacion desde `.rpt`;
- leer series temporales desde `.out` cuando se necesiten.

PySWMM quedaria en `runner.py` para ejecutar la simulacion y como fallback de
extraccion mientras validamos equivalencias.

## Fase 0 - Preparacion

1. Agregar dependencia:

```text
swmm-api
```

en `requirements.txt`.

2. Verificar instalacion local:

```bash
python -c "import swmm_api; print(swmm_api.__version__)"
```

3. Crear `swmm_resilience/simulation/swmm_api_io.py`.

4. Crear pruebas manuales con estas redes:

- `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp`
- `data/networks/chico_hydro-qx3/SWMM - Chico (PVC) Prueba 5 - Qx3.00.inp`

## Fase 1 - Lectura estructurada de `.inp`

Accion 1.1. Implementar:

```python
def load_inp(inp_file: Path | str):
    ...
```

Debe devolver el objeto `SwmmInput` de `swmm_api`.

Accion 1.2. Implementar:

```python
def list_inflow_nodes(inp) -> set[str]:
    ...
```

Debe leer la seccion `[INFLOWS]` y devolver nodos con inflow tipo `FLOW`.

Accion 1.3. Implementar:

```python
def get_node_timeseries_map(inp) -> dict[str, str]:
    ...
```

Debe devolver:

```python
{
    "1C": "1C",
    "2C": "2C",
}
```

donde la llave es el nodo y el valor es la serie asociada.

Accion 1.4. Comparar contra el parser actual `_parse_inp_sections()`.

Criterio de aceptacion:

- mismo numero de nodos con inflow;
- mismas series asociadas por nodo;
- sin parseo manual de tokens para `[INFLOWS]`.

## Fase 2 - Escalado de hidrogramas internos

Accion 2.1. Implementar:

```python
def write_scaled_inp(
    inp_file: Path | str,
    multiplier: float,
    target_nodes: set[str] | None,
    output_file: Path | str,
) -> Path:
    ...
```

Debe:

- leer el `.inp`;
- ubicar las series usadas por los nodos objetivo;
- multiplicar sus valores por `multiplier`;
- escribir un `.inp` nuevo.

Accion 2.2. Mantener `multiplier = 1.0` como caso sin cambios.

Accion 2.3. Validar con Qx1:

- generar temporal Qx1 con factor `3`;
- comparar los primeros valores de `[TIMESERIES]` de `1C` contra Qx3;
- deben coincidir aproximadamente:

```text
Qx1 1C 0:00 = 0.600
Qx1 * 3     = 1.800
Qx3 1C 0:00 = 1.800
```

Accion 2.4. Sustituir en `runner.py` la funcion manual `_write_scaled_inp()`.

Criterio de aceptacion:

- `runner.py` deja de escribir lineas de `[TIMESERIES]` manualmente;
- el `.inp` temporal se genera por `swmm_api_io.write_scaled_inp()`;
- se siguen limpiando temporales `.inp/.rpt/.out`.

## Fase 3 - Lectura de `.rpt`

Accion 3.1. Implementar:

```python
def read_node_flooding_summary(rpt_file: Path | str):
    ...
```

Debe devolver un `DataFrame` con el resumen de inundacion por nodo desde el
`.rpt`.

Accion 3.2. Mapear columnas del `.rpt` a las columnas actuales:

```text
node_id
flooding_volume_m3
flooding_duration_min
```

Accion 3.3. Comparar contra `node.statistics` de PySWMM.

Caso de validacion:

- para Qx3, nodo `1C`, el volumen del `.rpt` debe coincidir con el valor
  esperado visible en el reporte.

Accion 3.4. Agregar fallback:

- si falla lectura del `.rpt`, usar `node.statistics`;
- registrar warning en consola.

Criterio de aceptacion:

- la herramienta puede poblar `node_results` desde `.rpt`;
- los nodos inundados coinciden con el reporte SWMM.

## Fase 4 - Lectura de `.out`

Accion 4.1. Implementar:

```python
def read_out_timeseries(out_file: Path | str):
    ...
```

Debe devolver datos tabulares desde `.out`.

Accion 4.2. Implementar extractores especificos:

```python
def get_node_series(out, node_id: str) -> DataFrame:
    ...

def get_link_series(out, link_id: str) -> DataFrame:
    ...
```

Accion 4.3. Definir columnas minimas para futuro dataset temporal:

```text
run_id
node_id
time_sec
total_inflow_lps
lateral_inflow_lps
depth_m
flooding_lps
total_outflow_lps
```

Accion 4.4. No activar persistencia temporal todavia.

Criterio de aceptacion:

- se puede leer `.out` de una corrida y obtener series por nodo/link;
- no cambia el dataset tabular actual.

## Fase 5 - Integracion en `runner.py`

Accion 5.1. Cambiar flujo de corrida:

```text
inp original
-> swmm_api_io.write_scaled_inp()
-> PySWMM Simulation(temp_inp)
-> .rpt/.out generados
-> swmm_api_io.read_node_flooding_summary()
-> construir node_records/link_records
```

Accion 5.2. Mantener PySWMM para:

- correr la simulacion;
- extraer topologia si todavia es mas simple;
- extraer `link.conduit_statistics` hasta migrar links.

Accion 5.3. Agregar flag interno:

```python
USE_SWMM_API_RESULTS = True
```

Puede vivir en `config.py` durante la transicion.

Accion 5.4. Validar que `save_results()` no requiere cambios.

Criterio de aceptacion:

- las tablas SQLite se llenan igual que antes;
- la fuente de inundacion por nodo viene del `.rpt`;
- el escalado de hidrograma viene de `swmm_api`.

## Fase 6 - Validacion numerica

Accion 6.1. Ejecutar Qx1 factor `1`.

Esperado:

- resultados equivalentes a Qx1 original.

Accion 6.2. Ejecutar Qx1 factor `3`.

Esperado:

- `[TIMESERIES]` temporal coincide con Qx3 para los hidrogramas.
- Si los resultados hidraulicos no coinciden con Qx3, documentar diferencias
  estructurales entre archivos `.inp`.

Accion 6.3. Ejecutar Qx3 factor `1`.

Esperado:

- resultados equivalentes al `.rpt` original de Qx3.

Accion 6.4. Comparar:

```text
failed_nodes_count
total_flooding_volume_m3
node_results para 1C
link_results principales
```

## Fase 7 - Limpieza

Accion 7.1. Eliminar parseos manuales ya reemplazados.

Accion 7.2. Actualizar README:

- explicar que `swmm_api` manipula `.inp`;
- explicar que PySWMM ejecuta;
- explicar que `.rpt/.out` alimentan resultados.

Accion 7.3. Agregar notas de riesgo:

- `swmm_api` aun no es version `1.0`;
- encapsular uso en `swmm_api_io.py`;
- no mezclar objetos `swmm_api` fuera de esa capa.

## Riesgos

- Diferencias entre archivos Qx1 y Qx3 que no son hidrogramas.
- Cambios futuros de API en `swmm_api`.
- Codificacion de archivos `.inp` con caracteres especiales.
- Unidades de reporte: confirmar siempre LPS, m3, minutos.

## Resultado esperado final

La herramienta seguira funcionando igual para el usuario, pero internamente:

- los hidrogramas internos se escalan de forma estructurada;
- los resultados finales se leen desde los archivos SWMM;
- PySWMM queda reservado para ejecutar y para extraccion dinamica cuando haga
  falta.
