# SP5 — Predictor Temporal + Integración al Desktop

## Objetivo

Implementar `predict_failure_timeline()` en `swmm_resilience/ml/temporal/predict.py` y agregar una nueva pestaña al desktop que muestre, por nodo, la probabilidad de falla, el tiempo estimado a la falla y la severidad esperada — usando el modelo CNN o LSTM entrenado.

---

## Contexto

El archivo `swmm_resilience/ml/temporal/predict.py` existe como placeholder que lanza `NotImplementedError`. El desktop (`swmm_resilience/desktop/app.py`) ya tiene pestañas para el pipeline tabular. Este sub-proyecto cierra el ciclo: conecta los modelos entrenados con el usuario.

---

## Interfaz del predictor

```python
@dataclass
class NodeRiskProfile:
    node_id: str
    failure_probability: float    # [0.0, 1.0] — salida del clasificador
    peak_flooding_lps: float      # salida del regresor (0.0 si sin falla esperada)
    time_to_failure_min: float    # estimado desde el paso actual; NaN si sin falla
    risk_level: str               # 'safe' | 'warning' | 'critical'


def predict_failure_timeline(
    parquet_path: Path | None = None,
    records: list[dict] | None = None,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    model_type: str = "cnn",     # 'cnn' | 'lstm'
    task: str = "both",          # 'classification' | 'regression' | 'both'
    device: str = "cpu",
) -> list[NodeRiskProfile]:
    """
    Infiere riesgo por nodo desde un Parquet o desde registros en vivo.
    Devuelve una lista con un NodeRiskProfile por nodo presente en los datos.
    """
```

### Reglas de `risk_level`

| condición | nivel |
|---|---|
| `failure_probability < 0.3` | `'safe'` |
| `0.3 ≤ failure_probability < 0.7` | `'warning'` |
| `failure_probability ≥ 0.7` | `'critical'` |

---

## Pasos internos del predictor

1. Si se pasa `records` (lista de dicts de timesteps), convertirlos a DataFrame con `REQUIRED_TIMESERIES_COLUMNS`.
2. Si se pasa `parquet_path`, leerlo con `pd.read_parquet(...)`.
3. Construir la ventana más reciente disponible por nodo (los últimos `window_min` minutos de datos).
4. Cargar scalers (`scaler_seq`, `scaler_static`) y pesos del modelo (`.pt`) desde `artifacts_dir`.
5. Instanciar `SWMMTemporalCNN` o `SWMMTemporalLSTM` según `model_type`, cargar `state_dict`.
6. Ejecutar `model.eval()` + `torch.no_grad()` → probabilidad de falla y/o `peak_flooding_lps`.
7. Estimar `time_to_failure_min`: si `failure_probability ≥ 0.3`, usar `horizon_min` como proxy hasta tener una calibración más refinada. Si `< 0.3`, `NaN`.
8. Devolver lista de `NodeRiskProfile`.

---

## Archivos

### `swmm_resilience/ml/temporal/predict.py` (reemplaza placeholder)

- Implementar `predict_failure_timeline(...)` completa.
- Mantener la firma visible actual para no romper importaciones existentes.

### `swmm_resilience/config.py`

`DEFAULT_TEMPORAL_ARTIFACTS_DIR` ya fue agregada en SP3. Verificar que existe; si el proyecto no pasó por SP3, agregarla:

```python
DEFAULT_TEMPORAL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "temporal" / "model_artifacts"
```

### `swmm_resilience/desktop/app.py`

Agregar pestaña `"Alerta Temporal"` al notebook de pestañas existente.

---

## Pestaña "Alerta Temporal" en el desktop

### Layout

```
┌─ Alerta Temporal ──────────────────────────────────────────────────────────┐
│                                                                             │
│  Modelo: [CNN ▼]    [Cargar Parquet...]    [Inferir]                        │
│                                                                             │
│  ┌─ Tabla de riesgo por nodo ─────────────────────────────────────────┐    │
│  │ Nodo      │ P(falla) │ Pico (lps) │ T a falla (min) │ Riesgo      │    │
│  │ J-001     │ 0.87     │ 12.4       │ ~5              │ 🔴 CRÍTICO  │    │
│  │ J-002     │ 0.41     │ 3.1        │ ~5              │ 🟡 ALERTA   │    │
│  │ J-003     │ 0.08     │ 0.0        │ —               │ 🟢 SEGURO   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Nodos críticos: 1 / 3  │  Umbral configurado: 0.70                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interacción

- **Botón "Cargar Parquet..."**: abre diálogo de archivo para seleccionar un `.parquet` de `node_timeseries`.
- **Dropdown "Modelo"**: CNN | LSTM.
- **Botón "Inferir"**: llama `predict_failure_timeline(parquet_path=..., model_type=...)` en un hilo separado (para no bloquear la UI).
- **Tabla**: actualiza con los resultados. Filas con `risk_level='critical'` se colorean en rojo claro.
- Si no hay artefactos entrenados, el botón muestra `"Modelo no entrenado"` y está deshabilitado.

---

## CLI de inferencia

```bash
# Desde un Parquet
python -m swmm_resilience.ml.temporal.predict \
    --parquet data/networks/chico_hydro-qx1/results/temporal/node_timeseries/run_abc123.parquet \
    --model cnn \
    --output results/risk_report.csv
```

Genera un CSV con una fila por nodo y las columnas de `NodeRiskProfile`.

---

## Pruebas

- `tests/ml/temporal/test_predict.py`
  - `test_predict_from_parquet_returns_node_profiles`: con un Parquet sintético y artefactos mock → devuelve una lista de `NodeRiskProfile` con `node_id` correcto.
  - `test_risk_level_thresholds`: probabilidades en 0.1, 0.5, 0.9 → niveles `'safe'`, `'warning'`, `'critical'` respectivamente.
  - `test_predict_from_records_list`: acepta `records=list[dict]` sin necesidad de Parquet.
  - `test_missing_artifacts_raises_clear_error`: si los `.pt` no existen → `FileNotFoundError` con mensaje que indica qué archivo falta.

- `tests/desktop/test_temporal_tab.py`
  - `test_temporal_tab_exists_in_notebook`: verifica que la pestaña aparece en el notebook de `app.py`.
  - `test_infer_button_disabled_without_artifacts`: sin artefactos, el botón está deshabilitado.

---

## Precondiciones

- SP3 completado: artefactos CNN existen en `model_artifacts/`.
- SP4 opcional: si los artefactos LSTM existen, el predictor puede usarlos.
- Al menos un Parquet de `node_timeseries` disponible para inferencia manual.

---

## Lo que este sub-proyecto NO hace

- No re-entrena modelos (solo carga pesos guardados).
- No implementa streaming en tiempo real desde el runner activo (inferencia sobre corrida en vivo). Eso sería una extensión futura.
- No genera mapas de inundación temporal (eso pertenece al módulo de visualización).
- No calibra probabilidades (la salida del `Sigmoid` se usa directamente sin Platt scaling).
