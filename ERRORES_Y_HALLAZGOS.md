# Errores y Hallazgos del Proyecto

## Descripción del proyecto

`swmm_resilience` es una herramienta para evaluar la resiliencia de redes de
drenaje urbano usando SWMM. El pipeline completo tiene cuatro capas:

1. **Simulación hidráulica** — corre PySWMM con distintos escenarios de caudal
2. **Base de datos SQLite** — persiste topología, resultados por nodo/link y resúmenes
3. **Dataset ML** — exporta un CSV plano para entrenar modelos tabulares
4. **Modelos ML** — regresión (`flooding_volume_m3`) y clasificación (`flooded`)

Existe además una **interfaz Tkinter** para usuarios no técnicos y un **scaffold
temporal** para futuros modelos CNN/LSTM sobre hidrogramas (aún sin implementar).

---

## ERRORES CRÍTICOS

Estos errores producen datos incorrectos o crashes en escenarios reales.

---

### ERROR 1 — `runner.py:458` — `delta_inflow_lps` de links copia `inflow_multiplier`

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L456-L460)

**Código actual:**
```python
"delta_inflow_lps": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
"inflow_multiplier": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
```

**Problema:** Ambas columnas reciben el mismo valor (el multiplicador). Para nodos,
`delta_inflow_lps` usa `scenario_reference_inflow_lps(node_id)` que calcula el caudal
real. Para links el código simplemente copia `inflow_multiplier`, que es un factor
adimensional (ej. `1.3`), no un caudal en L/s. La tabla `link_results` queda con
`delta_inflow_lps` semánticamente erróneo.

**Corrección sugerida:** Para links no hay un "caudal delta" por nodo análogo.
Lo correcto es guardar el multiplicador de escenario como `inflow_multiplier`
(ya correcto) y dejar `delta_inflow_lps` en `NULL` o calcular el pico de flujo
real del link en su lugar.

---

### ERROR 2 — `runner.py:391-394` — Cálculo de `time_to_peak_min` potencialmente incorrecto

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L391-L394)

**Código actual:**
```python
raw_peak = stats.get("time_max_depth")
if raw_peak and sim_start:
    peak_dt = sim_start + timedelta(days=raw_peak - int(sim_start.toordinal()))
    time_to_peak = max(0.0, (peak_dt - sim_start).total_seconds() / 60.0)
```

**Problema:** `time_max_depth` en PySWMM puede devolver el número de día juliano
(ej. `738500.5`). `sim_start.toordinal()` también devuelve un ordinal gregoriano
(similar en magnitud, ej. `738500`). La resta `raw_peak - int(sim_start.toordinal())`
puede producir valores muy pequeños o negativos dependiendo de la fecha de inicio
de la simulación, generando tiempos de pico incorrectos. El `max(0.0, ...)` oculta
los valores negativos sin avisar.

**Corrección sugerida:**
```python
if raw_peak and sim_start:
    # time_max_depth es tiempo elapsed en días desde el inicio de la simulación
    time_to_peak = max(0.0, raw_peak * 24.0 * 60.0)
```
O si la API devuelve fecha absoluta:
```python
    time_to_peak = max(0.0, (peak_dt - sim_start).total_seconds() / 60.0)
    # donde peak_dt = datetime.fromordinal(int(raw_peak)) + timedelta(raw_peak % 1)
```
Verificar con la versión de PySWMM instalada qué unidad retorna `time_max_depth`.

---

### ERROR 3 — `runner.py` — Doble inyección de caudal cuando `.inp` ya tiene `[INFLOWS]`

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L340-L348)

**Código actual:**
```python
base_inflow_lps = base_node_inflows_lps.get(node_id, 0.0) or 0.0
inflow_lps = (
    _hydrograph_value_lps(hydrograph, elapsed_min) * hydrograph_multiplier
    if hydrograph is not None
    else base_inflow_lps * inflow_multiplier
)
node.generated_inflow(applied_lps)
```

**Problema:** `node.generated_inflow()` en PySWMM **agrega** caudal sobre lo que
SWMM ya calcula internamente desde la sección `[INFLOWS]` del `.inp`. Si el archivo
`.inp` ya tiene inflows definidos (ej. los casos `chico_hydro-qx2` y `chico_hydro-qx3`),
el caudal total es `inflow_del_inp + base_inflow_lps * inflow_multiplier`. Esto
doble-cuenta los caudales cuando `inflow_multiplier = 1.0`, produciendo resultados
hidráulicos erróneos.

El plan en `PLAN_TEMPORAL_LSTM_CNN.md` reconoce este problema pero no se ha resuelto
en el código.

**Corrección sugerida:** Para redes con inflows embebidos en el `.inp`, no llamar
`node.generated_inflow()` (o llamarla con 0.0). Detectar el origen del escenario
(`inp_embedded` vs `generated_uniform`) y aplicar la inyección solo cuando
corresponde. Ver Epic 2 del plan.

---

### ERROR 4 — `runner.py:52-56` — Parseo de sección `[INFLOWS]` toma el último token incorrectamente

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L52-L57)

**Código actual:**
```python
elif current_section == "[INFLOWS]" and len(parts) >= 2:
    try:
        inflow_rows[parts[0]] += float(parts[-1])
    except ValueError:
        continue
```

**Problema:** El formato real de la sección `[INFLOWS]` en SWMM es:
```
;;Nodo    Constituyente  TimeSeries  Tipo  Mfactor  Sfactor  Baseline  Patron
J1        FLOW           HYDRO1      FLOW  1.0      1.0      0.0       PAT1
```
El último token (`parts[-1]`) puede ser un nombre de patrón (`PAT1`), no un número.
El `float(parts[-1])` falla silenciosamente con `continue`, produciendo
`base_node_inflows_lps` incorrecto o vacío para muchas redes SWMM reales.
Cuando sí es numérico (el `Baseline`), suma el caudal base, no el caudal del
hidrograma. El resultado es `base_node_inflows_lps` no confiable.

**Corrección sugerida:** Parsear la columna `Baseline` por posición fija (columna 7
si existe), no el último token. O mejor aún: leer los inflows reales durante la
simulación con `node.total_inflow` en lugar de parsear el texto del `.inp`.

---

## ERRORES MEDIOS

Comportamiento incorrecto que afecta calidad de datos o rendimiento.

---

### ERROR 5 — `predict_tabular.py` — Reentrena el modelo en cada predicción

**Archivo:** [swmm_resilience/ml/predict_tabular.py](swmm_resilience/ml/predict_tabular.py#L129-L130)

**Código actual:**
```python
regressor.fit(X_reg, y_reg)
classifier.fit(X_cls, y_cls)
```

**Problema:** `predict_steady_flows()` carga el dataset completo y reentrena los
modelos desde cero **en cada llamada de predicción**. Esto es lento (especialmente
con XGBoost y datasets grandes) e inconsistente: el modelo que predice no es el
mismo que fue evaluado y guardado con `fit_and_save_inference_models()`.
`predict_from_inp.py` sí usa artefactos guardados correctamente.

**Corrección sugerida:** Cargar los artefactos persistidos con
`train.load_saved_model_artifact()` en lugar de reentrenar. Si no existen
artefactos, lanzar un error útil que indique que hay que entrenar primero.

---

### ERROR 6 — `temporal/dataset.py:20` — Columna `max_depth_ratio` debería ser `depth_ratio`

**Archivo:** [swmm_resilience/ml/temporal/dataset.py](swmm_resilience/ml/temporal/dataset.py#L14-L22)

**Código actual:**
```python
REQUIRED_TIMESERIES_COLUMNS = [
    ...
    "max_depth_ratio",  # incorrecto
    ...
]
```

**Problema:** `max_depth_ratio` es la columna agregada de `node_results` (máximo
de toda la corrida). La capa temporal por timestep debe usar `depth_ratio`
(profundidad relativa instante a instante). El plan en `PLAN_TEMPORAL_LSTM_CNN.md`
define explícitamente `depth_ratio` como columna de la serie temporal.

**Corrección sugerida:** Cambiar `"max_depth_ratio"` por `"depth_ratio"` y agregar
`"step_index"`, `"time_sec"`, `"total_inflow_lps"`, `"lateral_inflow_lps"`,
`"head_m"`, `"flooding_lps"`, `"total_outflow_lps"`, `"failed_now"` según el
contrato definido en el plan.

---

### ERROR 7 — `runner.py` — `depth_rate_m_per_min` inicializado en 0, no en la profundidad real del primer paso

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L322)

**Código actual:**
```python
depth_rate_tracker[node_id] = {"prev": 0.0, "max_rate": 0.0}
```

**Problema:** El tracker empieza con `prev=0.0`. En el primer paso de simulación,
si el nodo ya tiene profundidad no nula (por condiciones iniciales del `.inp`),
la tasa calculada es `(depth_actual - 0.0) / timestep`, que sobreestima la tasa
de llenado real. Esto infla `depth_rate_m_per_min` artificialmente en el primer paso.

**Corrección sugerida:** Inicializar `prev` con la profundidad real del primer paso
de simulación, o ignorar el primer paso en el cálculo de la tasa máxima.

---

### ERROR 8 — `runner.py` — `timestep_sec` de paso fijo asume timestep constante

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L333-L335)

**Código actual:**
```python
if timestep_sec is None:
    dt = (sim.current_time - sim_start).total_seconds()
    timestep_sec = dt if dt > 0 else 1.0
```

**Problema:** Se captura el timestep del **primer paso** y se usa para calcular
`time_to_first_flood_min = first_flood_step * timestep_sec / 60.0`. Para
simulaciones con timestep dinámico (variable) esto da tiempos incorrectos.
Aunque para steady-flow con reporting interval fijo es OK, la lógica es frágil.

**Corrección sugerida:** Acumular tiempo real con `elapsed_min` en lugar de
`step_count * timestep_sec`. Guardar `first_flood_elapsed_min` directamente:
```python
if first_flood_step is None and node.flooding > 0:
    first_flood_elapsed_min = elapsed_min
```

---

### ERROR 9 — `eda.py` — `KNOWN_RESULT_COLUMNS` incompleto: columnas sin clasificar

**Archivo:** [swmm_resilience/analysis/eda.py](swmm_resilience/analysis/eda.py#L38-L44)

**Código actual:**
```python
KNOWN_RESULT_COLUMNS = {
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "flooding_duration_min",
}
```

**Problema:** El dataset exportado incluye también `max_total_outflow_lps`,
`time_to_peak_outflow_min` y `downstream_link_peak_flows_lps_json`. Estas columnas
no están en `KNOWN_RESULT_COLUMNS`, por lo que caen en `unclassified_columns`
en el reporte de revisión, dando una clasificación incompleta de las columnas.

**Corrección sugerida:** Agregar al set:
```python
KNOWN_RESULT_COLUMNS = {
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "flooding_duration_min",
    "max_total_outflow_lps",        # agregar
    "time_to_peak_outflow_min",     # agregar
    "downstream_link_peak_flows_lps_json",  # agregar
}
```

---

### ERROR 10 — `predict_tabular.py:75` — `keep="last"` en `drop_duplicates` es arbitrario

**Archivo:** [swmm_resilience/ml/predict_tabular.py](swmm_resilience/ml/predict_tabular.py#L75)

**Código actual:**
```python
base_rows = base_df.drop_duplicates(subset=["node_id"], keep="last")
```

**Problema:** El CSV está ordenado por `inflow_multiplier` ascendente. `keep="last"`
toma la fila con el multiplicador más alto. Eso significa que las features estáticas
del nodo se toman desde el escenario más extremo, no desde el baseline. Para
features verdaderamente estáticas (topología) no importa, pero si hay alguna
feature de escenario mezclada, los resultados son incorrectos.

**Corrección sugerida:** Usar `keep="first"` para tomar la fila con el multiplicador
más bajo (más cercano al baseline), o documentar explícitamente por qué se elige
el último.

---

## PROBLEMAS DE DISEÑO / DEUDA TÉCNICA

Estos no crashean ni producen resultados dramáticamente erróneos, pero
representan inconsistencias o riesgos que escalan con el proyecto.

---

### DISEÑO 1 — `schema.py` — `delta_inflow_lps REAL NOT NULL` en tabla `runs` es demasiado restrictivo

**Archivo:** [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py#L53)

**Problema:** La columna `delta_inflow_lps NOT NULL` obliga a tener un valor en
todos los escenarios. Para redes con hidrogramas embebidos en el `.inp`, este valor
es solo una aproximación (`hydrograph_peak_lps * multiplier`), no una descripción
real del input. El plan v2 depreca esta columna como `legacy`.

**Corrección sugerida:** Cambiar a `delta_inflow_lps REAL` (nullable) e implementar
`input_source` y las columnas `hydrograph_*` según el plan v2 de la tabla `runs`.

---

### DISEÑO 2 — `requirements.txt` — Falta `joblib`, dependencia obligatoria

**Archivo:** [requirements.txt](requirements.txt)

**Código actual:**
```
pandas>=1.3.0
numpy>=1.21.0
pyswmm>=0.15.0
scikit-learn>=1.0.0
ydata-profiling>=4.17,<5
```

**Problema:** `joblib` es una dependencia **obligatoria** del módulo `train.py`
(se usa para guardar y cargar artefactos). No está en `requirements.txt`. En un
entorno limpio, la instalación funciona porque `scikit-learn` instala `joblib`
como dependencia transitiva, pero no se debe confiar en eso.

`xgboost` tampoco está listado aunque es el modelo recomendado para producción.

**Corrección sugerida:**
```
pandas>=1.3.0
numpy>=1.21.0
pyswmm>=0.15.0
scikit-learn>=1.0.0
joblib>=1.2.0
xgboost>=1.7.0
ydata-profiling>=4.17,<5
```

---

### DISEÑO 3 — `runner.py` — Funciones auxiliares redefinidas dentro del loop de nodos

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L216-L233)

**Código actual:**
```python
for node in nodes:
    ...
    def agg(link_ids: list, field: str): ...
    def safe_avg(values): ...
    def safe_max(values): ...
    def safe_min(values): ...
    def safe_sum(values): ...
```

**Problema:** Estas cuatro funciones se redefinen en cada iteración del loop de
nodos. Python no reutiliza los objetos de función, los crea nuevos cada vez. En
una red con 100 nodos, crea 400 funciones innecesarias. Es ineficiente y confunde
al lector que busca dónde están definidas las funciones.

**Corrección sugerida:** Mover `agg`, `safe_avg`, `safe_max`, `safe_min`, `safe_sum`
fuera del loop de nodos (antes del `for node in nodes:`), o convertirlas en
funciones de módulo en `utils.py`.

---

### DISEÑO 4 — `desktop/app.py:779` — Importación local frágil de `view_db`

**Archivo:** [swmm_resilience/desktop/app.py](swmm_resilience/desktop/app.py#L779)

**Código actual:**
```python
from view_db import SQLiteViewerApp
```

**Problema:** Esta importación depende de que el directorio de trabajo sea la raíz
del proyecto. Si la app se lanza desde otro directorio (ej. `python app.py` desde
cualquier otra carpeta), fallará con `ModuleNotFoundError`. Además, mezcla un módulo
de nivel raíz (`view_db.py`) con un submódulo del paquete.

**Corrección sugerida:** Mover `SQLiteViewerApp` a `swmm_resilience/desktop/viewer.py`
y actualizar el import a `from swmm_resilience.desktop.viewer import SQLiteViewerApp`.

---

### DISEÑO 5 — `config.py` — `ML_DROP_COLUMNS` contiene `applied_inflow_lps` que ya no existe

**Archivo:** [swmm_resilience/config.py](swmm_resilience/config.py#L73)

**Código actual:**
```python
ML_DROP_COLUMNS = [
    ...
    "applied_inflow_lps",
    ...
]
```

**Problema:** `applied_inflow_lps` fue la columna original antes de la migración.
Fue renombrada a `delta_inflow_lps`. El drop es un no-op inofensivo, pero es deuda
técnica: confunde al lector y hace pensar que esa columna aún puede aparecer.

**Corrección sugerida:** Reemplazar `"applied_inflow_lps"` por `"delta_inflow_lps"`
si se quiere excluir del dataset ML, o eliminarlo si no se quiere excluir.

---

### DISEÑO 6 — `main.py` — Variables `hydrograph_multiplier` e `inflow_multiplier` redundantes

**Archivo:** [swmm_resilience/main.py](swmm_resilience/main.py#L182-L183)

**Código actual:**
```python
hydrograph_multiplier = scenario_multiplier if hydrograph is not None else 1.0
inflow_multiplier = scenario_multiplier if hydrograph is None else hydrograph_multiplier
```

**Problema:** Cuando `hydrograph is not None`, ambas variables terminan con
`scenario_multiplier`. Cuando `hydrograph is None`, `inflow_multiplier = scenario_multiplier`
y `hydrograph_multiplier = 1.0`. El resultado: siempre es `inflow_multiplier = scenario_multiplier`.
La duplicación hace el código más difícil de mantener y puede llevar a errores
si alguien modifica una variable sin actualizar la otra.

**Corrección sugerida:**
```python
inflow_multiplier = scenario_multiplier
hydrograph_multiplier = scenario_multiplier if hydrograph is not None else 1.0
```

---

### DISEÑO 7 — `runner.py` — Segunda apertura del `.inp` en `extract_static_topology` ejecuta simulación completa

**Archivo:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py#L130)

**Problema:** `extract_static_topology` abre la simulación con `Simulation(inp_file)`
para leer topología. Dependiendo de la versión de PySWMM, esto puede ejecutar la
simulación completa internamente aunque no haya `for _ in sim`. Si PySWMM ejecuta
la simulación al entrar al context manager, duplica el tiempo de ejecución.

**Verificación sugerida:** Confirmar con la versión de PySWMM usada si el context
manager adelanta la simulación o solo inicializa. Si adelanta, extraer la topología
dentro de la misma corrida de `run_simulation`.

---

### DISEÑO 8 — `schema.py` — F-strings en SQL durante migración (patrón peligroso)

**Archivo:** [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py#L200-L241)

**Código actual:**
```python
conn.executescript(
    f"""
    ...
    {inflow_expr},
    {failed_expr},
    ...
    """
)
```

**Problema:** Aunque `inflow_expr` y `failed_expr` se construyen desde nombres de
columna verificados (no de input del usuario), el patrón de f-strings en SQL es
peligroso por convención. Si alguien extiende la lógica y agrega un valor de origen
externo sin darse cuenta, crea SQL injection.

**Corrección sugerida:** En lugar de f-strings, usar valores por posición o hacer
las dos actualizaciones en sentencias `UPDATE` separadas con parámetros.

---

## FUNCIONALIDAD FALTANTE (scaffold no implementado)

Estas son funciones que el plan define pero que actualmente están incompletas.
No son bugs hoy, pero bloquean el roadmap futuro.

---

### FALTANTE 1 — No existe persistencia de series temporales por timestep

**Archivos:** [swmm_resilience/simulation/runner.py](swmm_resilience/simulation/runner.py),
[swmm_resilience/ml/temporal/dataset.py](swmm_resilience/ml/temporal/dataset.py)

El `runner.py` no guarda series por `run_id + node_id + step_index`. Sin esto,
la CNN/LSTM no puede entrenarse. `build_temporal_windows()` lanza
`NotImplementedError` explícitamente.

**Siguiente paso:** Épica 3 del plan — implementar captura de `node_timeseries`
y exportación a Parquet.

---

### FALTANTE 2 — La tabla `runs` no tiene `input_source`

**Archivo:** [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py)

Sin `input_source`, no es posible distinguir corridas steady de corridas con
hidrograma externo o embebido. El entrenamiento ML mezcla escenarios incomparables.

**Siguiente paso:** Épica 1, Tarea 1.1 del plan.

---

### FALTANTE 3 — No existe `hydrograph_profiles` ni `temporal_artifacts`

**Archivo:** [swmm_resilience/database/schema.py](swmm_resilience/database/schema.py)

Tablas necesarias para catalogar hidrogramas y enlazar corridas tabulares con
sus series temporales reales.

**Siguiente paso:** Épicas 1 y 3 del plan.

---

### FALTANTE 4 — `node_results` no tiene las columnas de agregados dinámicos planeadas

Según el plan v2, `node_results` debería tener:
- `max_total_inflow_lps`
- `max_lateral_inflow_lps`
- `avg_depth_m`
- `avg_depth_ratio`
- `peak_flooding_lps`
- `time_to_first_flood_min`
- `time_over_threshold_min`
- `peak_outflow_to_peak_inflow_ratio`

Ninguna de estas columnas existe actualmente en el esquema ni se calcula en
`runner.py`. El `REQUIRED_COLUMNS` en `schema.py` tampoco las incluye.

**Siguiente paso:** Épica 1, Tarea 1.4 + Épica 2, Tarea 2.3 del plan.

---

## RESUMEN DE PRIORIDADES

| Prioridad | Error | Impacto |
|-----------|-------|---------|
| 🔴 CRÍTICO | ERROR 1 — `delta_inflow_lps` de links incorrecto | Datos incorrectos en DB |
| 🔴 CRÍTICO | ERROR 2 — `time_to_peak_min` potencialmente incorrecto | Métrica de timing falsa |
| 🔴 CRÍTICO | ERROR 3 — Doble inyección de caudal con `.inp` embebido | Simulación hidráulica incorrecta |
| 🔴 CRÍTICO | ERROR 4 — Parseo de `[INFLOWS]` por último token | `base_node_inflows_lps` no confiable |
| 🟠 MEDIO | ERROR 5 — Reentrenamiento en cada predicción | Lento e inconsistente |
| 🟠 MEDIO | ERROR 6 — `max_depth_ratio` en schema temporal | Schema temporal incorrecto |
| 🟠 MEDIO | ERROR 7 — `depth_rate` inicializado en 0 | Métrica de velocidad inflada |
| 🟠 MEDIO | ERROR 8 — `timestep_sec` fijo para tiempo a primer flood | Incorrecto con timestep variable |
| 🟠 MEDIO | ERROR 9 — `KNOWN_RESULT_COLUMNS` incompleto | Clasificación incorrecta en review |
| 🟡 BAJO | ERROR 10 — `keep="last"` en drop_duplicates | Fila de referencia arbitraria |
| 🟡 BAJO | DISEÑO 1-8 | Deuda técnica, robustez |
| ⚪ INFO | FALTANTE 1-4 | Roadmap temporal no implementado |
