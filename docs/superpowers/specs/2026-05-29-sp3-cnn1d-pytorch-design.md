# SP3 — CNN 1D con PyTorch (Clasificador y Regresor)

## Objetivo

Implementar dos modelos CNN 1D separados en PyTorch para el pipeline temporal: un clasificador (`failure_within_horizon`) y un regresor (`peak_flooding_lps`). Ambos comparten la misma arquitectura base de dos ramas (temporal + estática), pero tienen cabezas de salida independientes y se entrenan por separado.

---

## Contexto

El archivo `swmm_resilience/ml/temporal/train_cnn.py` existe como placeholder. Este sub-proyecto lo reemplaza con una implementación completa. Los datos de entrenamiento provienen de `TemporalWindowDataset` (SP2).

---

## Arquitectura del modelo

### Entrada

- **Rama temporal**: `[batch, timesteps=4, temporal_features=6]`
- **Rama estática**: `[batch, static_features=7]`

### Rama temporal (CNN 1D)

```
Input: [batch, 4, 6]
↓ Conv1d(in_channels=6, out_channels=32, kernel_size=2, padding=1)
↓ BatchNorm1d(32) + ReLU
↓ Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=0)
↓ BatchNorm1d(64) + ReLU
↓ AdaptiveAvgPool1d(1)          # → [batch, 64, 1]
↓ Flatten                        # → [batch, 64]
```

Nota sobre `kernel_size`: con ventanas de solo 4 timesteps, se usa `kernel_size=2` para preservar información temporal sin colapsar la secuencia prematuramente.

### Rama estática (densa)

```
Input: [batch, 7]
↓ Linear(7, 32) + ReLU
↓ Linear(32, 32) + ReLU
```

### Fusión y cabezas

```
Concatenar [batch, 64] + [batch, 32] → [batch, 96]
↓ Linear(96, 64) + ReLU + Dropout(0.3)

Cabeza clasificador:   Linear(64, 1) + Sigmoid  → probabilidad [0,1]
Cabeza regresor:       Linear(64, 1)             → valor continuo ≥ 0
```

Los dos modelos **no** se entrenan juntos. El clasificador y el regresor son instancias separadas de `SWMMTemporalCNN` con `task='classification'` o `task='regression'`.

---

## Normalización dentro del pipeline de entrenamiento

El pipeline de entrenamiento aplica `StandardScaler` a `X_seq` (por canal) y a `X_static` antes de pasar al modelo. El scaler se entrena solo con el fold de entrenamiento — nunca con validación ni prueba. Los scalers se guardan como artefactos junto con los pesos del modelo.

---

## Archivos

### `swmm_resilience/ml/temporal/models/cnn.py` (nuevo)

```python
class SWMMTemporalCNN(nn.Module):
    """CNN 1D + rama estática para clasificación o regresión por nodo."""
    def __init__(self, n_temporal_features: int, n_static_features: int, task: str): ...
    def forward(self, x_seq: Tensor, x_static: Tensor) -> Tensor: ...
```

`task`: `'classification'` | `'regression'`

### `swmm_resilience/ml/temporal/train_cnn.py` (reemplaza placeholder)

```python
def train_cnn(
    db_path: Path,
    networks_dir: Path,
    artifacts_dir: Path,
    task: str,                   # 'classification' | 'regression'
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_cv_folds: int = 5,
    device: str = "cpu",
) -> dict:
    """Entrena CNN 1D con GroupKFold. Devuelve métricas por fold."""
```

Pasos internos:
1. Llamar `build_temporal_windows(db_path, networks_dir)` → `TemporalWindowDataset`
2. `GroupKFold(n_splits=n_cv_folds)` con `groups=dataset.groups`
3. Por cada fold:
   - Crear `SWMMTemporalCNN(task=task)`
   - Fit scalers sobre train set
   - Entrenar con `AdamW`, `ReduceLROnPlateau`
   - Clasificador: `BCELoss`; regresor: `MSELoss`
   - Evaluar en val set → guardar métricas
4. Guardar el mejor fold (`val_loss` mínimo) como artefacto principal
5. Devolver métricas de todos los folds

### `swmm_resilience/ml/temporal/models/__init__.py` (nuevo)

```python
from .cnn import SWMMTemporalCNN
```

### `swmm_resilience/config.py`

Agregar la constante de directorio de artefactos temporales:

```python
DEFAULT_TEMPORAL_ARTIFACTS_DIR = DEFAULT_RESULTS_DIR / "temporal" / "model_artifacts"
```

---

## Artefactos guardados

Ruta base: `data/networks/<net>/results/temporal/model_artifacts/`

| archivo | contenido |
|---|---|
| `cnn_classifier_weights.pt` | `state_dict` del mejor fold |
| `cnn_classifier_scaler_seq.joblib` | `StandardScaler` para `X_seq` |
| `cnn_classifier_scaler_static.joblib` | `StandardScaler` para `X_static` |
| `cnn_classifier_metrics.csv` | métricas por fold (accuracy, F1, AUC-ROC, precision, recall) |
| `cnn_regressor_weights.pt` | `state_dict` del mejor fold |
| `cnn_regressor_scaler_seq.joblib` | `StandardScaler` para `X_seq` |
| `cnn_regressor_scaler_static.joblib` | `StandardScaler` para `X_static` |
| `cnn_regressor_metrics.csv` | métricas por fold (MAE, RMSE, R²) |

### `reset.py`

La función `reset_artifacts` ya elimina `*.joblib` y `*.csv`. Agregar `*.pt` al conjunto de sufijos:

```python
suffixes = {".joblib", ".json", ".csv", ".xlsx", ".pt"}
```

---

## CLI de entrenamiento

```bash
# Clasificador
python -m swmm_resilience.ml.temporal.train_cnn --task classification --epochs 50

# Regresor
python -m swmm_resilience.ml.temporal.train_cnn --task regression --epochs 50

# Ambos en secuencia
python -m swmm_resilience.ml.temporal.train_cnn --task all --epochs 50
```

---

## Métricas de éxito

### Clasificador
- AUC-ROC ≥ 0.80 en el fold de prueba
- F1-score ≥ 0.65

### Regresor
- R² ≥ 0.70 en el fold de prueba
- MAE < 10% del rango de `peak_flooding_lps` en el dataset

Estas métricas son orientativas; el modelo "pasa" si el entrenamiento converge sin NaN y las curvas de loss muestran aprendizaje. Las métricas absolutas dependen del tamaño del dataset, que en esta etapa puede ser pequeño.

---

## Pruebas

- `tests/ml/temporal/test_cnn_model.py`
  - `test_forward_pass_classification`: tensor sintético → salida en `[0, 1]`, shape `[batch, 1]`.
  - `test_forward_pass_regression`: tensor sintético → salida sin cota, shape `[batch, 1]`.
  - `test_training_loss_decreases`: entrenamiento de 5 epochs con datos sintéticos → `loss[4] < loss[0]`.
  - `test_artifacts_saved_after_training`: verifica que `.pt` y `.joblib` existen después de `train_cnn(...)`.
  - `test_no_data_leakage_between_folds`: verifica que los `run_id` del fold de val no aparecen en el fold de train.

---

## Precondiciones

- SP2 completado: `build_temporal_windows()` devuelve `TemporalWindowDataset` válido.
- PyTorch instalado: `pip install torch`.
- Al menos 2 corridas de tipo `hydrograph` en `temporal_artifacts` (para que `GroupKFold` tenga al menos 2 grupos).

---

## Lo que este sub-proyecto NO hace

- No implementa LSTM (SP4).
- No implementa el predictor en tiempo real (SP5).
- No agrega pestaña al desktop (SP5).
- No implementa arquitectura multi-task (clasificador y regresor son modelos separados).
