# SP4 — LSTM Benchmark con PyTorch

## Objetivo

Implementar dos modelos LSTM separados en PyTorch (clasificador y regresor) que sirvan como benchmark comparativo frente a la CNN 1D (SP3). Usan exactamente el mismo `TemporalWindowDataset` y el mismo protocolo de evaluación que SP3, de modo que las métricas sean directamente comparables.

---

## Contexto

La LSTM es el modelo de referencia estándar para series temporales en hidráulica. Entrenarla con los mismos datos y splits que la CNN permite responder: ¿la CNN 1D vale la pena frente a una LSTM más clásica para este problema?

El archivo `swmm_resilience/ml/temporal/train_lstm.py` no existe; hay que crearlo.

---

## Arquitectura del modelo

La arquitectura sigue el mismo esquema de dos ramas que la CNN, solo cambia la rama temporal:

### Entrada

- **Rama temporal**: `[batch, timesteps=4, temporal_features=6]`
- **Rama estática**: `[batch, static_features=7]`

### Rama temporal (LSTM)

```
Input: [batch, 4, 6]
↓ LSTM(input_size=6, hidden_size=64, num_layers=2,
        batch_first=True, dropout=0.2)
↓ Tomar el último hidden state h[-1]    # → [batch, 64]
```

`num_layers=2` con `dropout=0.2` entre capas ofrece algo de regularización sin sobre-complicar el modelo para datasets pequeños.

### Rama estática (densa)

Idéntica a SP3:

```
Input: [batch, 7]
↓ Linear(7, 32) + ReLU
↓ Linear(32, 32) + ReLU
```

### Fusión y cabezas

Idéntica a SP3:

```
Concatenar [batch, 64] + [batch, 32] → [batch, 96]
↓ Linear(96, 64) + ReLU + Dropout(0.3)

Cabeza clasificador:   Linear(64, 1) + Sigmoid
Cabeza regresor:       Linear(64, 1)
```

---

## Archivos

### `swmm_resilience/ml/temporal/models/lstm.py` (nuevo)

```python
class SWMMTemporalLSTM(nn.Module):
    """LSTM + rama estática para clasificación o regresión por nodo."""
    def __init__(self, n_temporal_features: int, n_static_features: int, task: str): ...
    def forward(self, x_seq: Tensor, x_static: Tensor) -> Tensor: ...
```

`task`: `'classification'` | `'regression'`

### `swmm_resilience/ml/temporal/train_lstm.py` (nuevo)

```python
def train_lstm(
    db_path: Path,
    networks_dir: Path,
    artifacts_dir: Path,
    task: str,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_cv_folds: int = 5,
    device: str = "cpu",
) -> dict:
    """Entrena LSTM con GroupKFold. Devuelve métricas por fold."""
```

La firma es intencionalmente idéntica a `train_cnn` para que el informe comparativo pueda llamarlos de manera uniforme.

### `swmm_resilience/ml/temporal/models/__init__.py`

Agregar `SWMMTemporalLSTM`:

```python
from .cnn import SWMMTemporalCNN
from .lstm import SWMMTemporalLSTM
```

---

## Artefactos guardados

Ruta base: `data/networks/<net>/results/temporal/model_artifacts/`

| archivo | contenido |
|---|---|
| `lstm_classifier_weights.pt` | `state_dict` del mejor fold |
| `lstm_classifier_scaler_seq.joblib` | `StandardScaler` para `X_seq` |
| `lstm_classifier_scaler_static.joblib` | `StandardScaler` para `X_static` |
| `lstm_classifier_metrics.csv` | métricas por fold (accuracy, F1, AUC-ROC, precision, recall) |
| `lstm_regressor_weights.pt` | `state_dict` del mejor fold |
| `lstm_regressor_scaler_seq.joblib` | `StandardScaler` para `X_seq` |
| `lstm_regressor_scaler_static.joblib` | `StandardScaler` para `X_static` |
| `lstm_regressor_metrics.csv` | métricas por fold (MAE, RMSE, R²) |

El sufijo `.pt` ya estará en `reset_artifacts` tras SP3.

---

## Informe comparativo CNN vs LSTM

Al final de `train_lstm`, si existen los archivos de métricas de CNN (SP3), se genera automáticamente:

`data/networks/<net>/results/temporal/model_artifacts/comparison_report.csv`

Columnas: `model`, `task`, `fold`, `metric`, `value`

Ejemplo:

```
cnn,classification,mean,auc_roc,0.84
lstm,classification,mean,auc_roc,0.81
cnn,regression,mean,r2,0.73
lstm,regression,mean,r2,0.70
```

Si los archivos CNN no existen, se omite silenciosamente el informe (sin error).

---

## CLI de entrenamiento

```bash
# Clasificador LSTM
python -m swmm_resilience.ml.temporal.train_lstm --task classification --epochs 50

# Regresor LSTM
python -m swmm_resilience.ml.temporal.train_lstm --task regression --epochs 50

# Ambos + genera comparison_report.csv
python -m swmm_resilience.ml.temporal.train_lstm --task all --epochs 50 --compare
```

---

## Pruebas

- `tests/ml/temporal/test_lstm_model.py`
  - `test_forward_pass_classification`: tensor sintético → salida en `[0, 1]`, shape `[batch, 1]`.
  - `test_forward_pass_regression`: tensor sintético → salida continua, shape `[batch, 1]`.
  - `test_training_loss_decreases`: 5 epochs con datos sintéticos → `loss[4] < loss[0]`.
  - `test_artifacts_saved_after_training`: `.pt` y `.joblib` existen después de `train_lstm(...)`.
  - `test_comparison_report_generated`: si existen métricas CNN, `comparison_report.csv` se crea con las columnas correctas.
  - `test_comparison_report_skipped_if_no_cnn`: sin archivos CNN no se lanza error.

---

## Precondiciones

- SP2 completado: `build_temporal_windows()` funciona.
- SP3 completado (recomendado): para que el informe comparativo tenga datos. Si no está completo, `train_lstm` funciona igual pero omite el informe.
- PyTorch instalado.

---

## Lo que este sub-proyecto NO hace

- No modifica la CNN (SP3).
- No implementa el predictor en tiempo real (SP5).
- No agrega pestaña al desktop (SP5).
- No implementa atención (attention LSTM) — eso sería una extensión futura.
