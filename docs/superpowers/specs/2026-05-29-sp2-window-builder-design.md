# SP2 — Constructor de Ventanas Temporales

## Objetivo

Implementar `build_temporal_windows()` en `swmm_resilience/ml/temporal/dataset.py`. Toma todos los Parquet registrados en `temporal_artifacts`, construye tensores deslizantes `[samples, timesteps, features]` y devuelve splits de entrenamiento/validación/prueba agrupados por `run_id` para evitar data leakage.

---

## Contexto

La función `build_temporal_windows()` existe como placeholder que lanza `NotImplementedError`. Los parámetros de ventana ya están en `config.py`:

```python
ML_TEMPORAL_RESAMPLE_MIN = 5   # resolución temporal del resampleo
ML_TEMPORAL_WINDOW_MIN   = 20  # largo de la ventana de entrada
ML_TEMPORAL_HORIZON_MIN  = 5   # horizonte de predicción adelantado
ML_TEMPORAL_STEP_MIN     = 5   # paso entre ventanas consecutivas
```

Con resample=5 min y window=20 min → **4 timesteps por ventana**. Horizonte de 5 min = 1 step adelante.

---

## Features por timestep (rama temporal)

Estas columnas del Parquet componen la matriz temporal `X_seq`:

| columna | descripción |
|---|---|
| `total_inflow_lps` | caudal total de entrada |
| `lateral_inflow_lps` | caudal lateral directo |
| `depth_m` | profundidad absoluta |
| `depth_ratio` | profundidad / full_depth_m |
| `flooding_lps` | caudal de desbordamiento |
| `total_outflow_lps` | caudal total de salida |

→ **6 features temporales** → tensores de forma `[N, 4, 6]` (N muestras, 4 timesteps, 6 features).

## Features estáticas por nodo (rama estática)

Se unen desde `network_nodes` usando `(network_hash, node_id)`:

| columna | descripción |
|---|---|
| `full_depth_m` | profundidad máxima del pozo |
| `in_degree` | grado de entrada en el grafo |
| `out_degree` | grado de salida |
| `upstream_diam_avg_m` | diámetro promedio de tuberías upstream |
| `downstream_diam_avg_m` | diámetro promedio de tuberías downstream |
| `upstream_capacity_lps` | capacidad hidráulica upstream |
| `downstream_capacity_lps` | capacidad hidráulica downstream |

→ **7 features estáticas** → vector `[N, 7]` por ventana.

## Targets

### Clasificación

- `failed_now` (int 0/1): el nodo está inundado (`flooding_lps > 0`) en el último step de la ventana.
- `failure_within_horizon` (int 0/1): habrá inundación en algún step dentro del horizonte de predicción. Este es el target principal de clasificación.

### Regresión

- `peak_flooding_lps`: máximo `flooding_lps` dentro del horizonte.
- `time_to_failure_min`: minutos hasta el primer step con `flooding_lps > 0` dentro del horizonte. `NaN` si no hay falla.

---

## Estructura del output

`build_temporal_windows()` devuelve un `TemporalWindowDataset` (dataclass):

```python
@dataclass
class TemporalWindowDataset:
    X_seq:    np.ndarray      # [N, timesteps, temporal_features]
    X_static: np.ndarray      # [N, static_features]
    y_class:  np.ndarray      # [N] int8, failure_within_horizon
    y_reg:    np.ndarray      # [N] float32, peak_flooding_lps
    groups:   np.ndarray      # [N] str, run_id para GroupKFold
    meta:     pd.DataFrame    # run_id, node_id, window_start_min por fila
```

Los splits de entrenamiento/validación/prueba se generan externamente usando `sklearn.model_selection.GroupKFold` con `groups` como clave. Esto evita data leakage entre corridas.

---

## Algoritmo interno

```
Para cada Parquet en temporal_artifacts (ordenados por created_at):
  1. Leer Parquet → DataFrame
  2. Para cada node_id único:
     a. Filtrar filas del nodo, ordenar por time_min
     b. Resamplear a ML_TEMPORAL_RESAMPLE_MIN usando forward-fill
     c. Sliding window con step=ML_TEMPORAL_STEP_MIN:
        - ventana = timesteps[i : i + window_steps]
        - horizonte = timesteps[i + window_steps : i + window_steps + horizon_steps]
        - Si ventana incompleta → descartar
        - Si horizonte incompleto → descartar
        - X_seq = ventana[temporal_cols].values
        - X_static = unir desde network_nodes
        - y_class = 1 si any(horizonte.flooding_lps > 0) else 0
        - y_reg = max(horizonte.flooding_lps)
        - Agregar fila a lista de muestras
3. Convertir lista a arrays numpy
4. Devolver TemporalWindowDataset
```

---

## Normalización

La normalización **no** ocurre en `build_temporal_windows()`. Las features crudas se devuelven sin escalar. El escalado es responsabilidad del pipeline de entrenamiento (SP3/SP4) para mantener train/test separation limpia — exactamente el mismo principio que usa `preprocessing.py` en el pipeline tabular.

---

## Cambios en el código

### `swmm_resilience/ml/temporal/dataset.py`

Firma final (reemplaza el placeholder):

```python
def build_temporal_windows(
    db_path: Path = DEFAULT_DB_FILE,
    networks_dir: Path = NETWORKS_DIR,
    window_spec: TemporalWindowSpec | None = None,
    dataset_spec: TemporalDatasetSpec | None = None,
) -> TemporalWindowDataset:
```

- Reemplaza el placeholder actual que lanza `NotImplementedError`
- Lee `temporal_artifacts` vía SQL para descubrir los Parquet; no importa `register_temporal_artifact`

### `swmm_resilience/ml/temporal/schemas.py`

- Agregar `TemporalWindowDataset` dataclass con los campos descritos arriba
- `TemporalWindowSpec` ya existe; solo agregar `resample_min: int = ML_TEMPORAL_RESAMPLE_MIN`

### `swmm_resilience/config.py`

- No requiere cambios. Los parámetros ya existen.

---

## CLI de diagnóstico

```bash
python -m swmm_resilience.ml.temporal.dataset --db data/training/swmm_resilience.db --summary
```

Imprime:
- Número de Parquets encontrados
- Total de muestras generadas
- Distribución de clases (`failure_within_horizon`)
- Balance positivos / negativos

---

## Precondiciones

- SP1 completado: tabla `temporal_artifacts` existe y tiene al menos una fila con `status='completed'`.
- Los archivos Parquet existen en las rutas registradas.
- `network_nodes` tiene filas para el `network_hash` de las corridas.

---

## Pruebas

- `tests/ml/temporal/test_window_builder.py`
  - `test_build_windows_produces_correct_shape`: con un Parquet sintético de 2 nodos × 20 steps → verifica dimensiones de `X_seq`, `X_static`, `y_class`, `y_reg`.
  - `test_no_leakage_between_runs`: si hay 2 run_ids, `groups` contiene exactamente 2 valores distintos.
  - `test_incomplete_window_discarded`: si el Parquet tiene menos steps que `window_min`, devuelve dataset vacío sin errores.
  - `test_failure_within_horizon_label`: construye caso conocido (flooding en step N+1) y verifica que `y_class == 1`.
  - `test_no_failure_within_horizon_label`: construye caso sin flooding en horizonte y verifica `y_class == 0`.
  - `test_static_features_joined_correctly`: verifica que `X_static` coincide con los valores de `network_nodes` para el `node_id` correcto.

---

## Lo que este sub-proyecto NO hace

- No entrena ningún modelo.
- No normaliza features.
- No implementa CNN ni LSTM.
- No modifica el runner ni el schema (depende de SP1).
