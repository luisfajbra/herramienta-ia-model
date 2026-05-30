# Surrogate CNN for SWMM Flood Prediction — Design Spec

## Context

Both the early-warning CNN (SP3) and this surrogate use SWMM simulation runs to generate training
data — that is an **offline, one-time process**. The difference is what happens **at inference**:

- **Early-warning CNN (SP3):** its temporal inputs are SWMM hydraulic outputs (`depth_m`,
  `flooding_lps`, `total_outflow_lps`). At inference, those values must come from somewhere —
  meaning a SWMM run is still required before the CNN can make a prediction. The CNN is only
  useful as a faster step on top of SWMM, not as a replacement.

- **Surrogate CNN (this spec):** its temporal input is the inflow hydrograph, which can be
  synthesized from the Qx1.00 base run scaled by a new multiplier — no SWMM run needed.
  Training data still comes from SWMM runs (offline), but once trained the model predicts
  flooding for any multiplier on its own.

The comparison goal: evaluate whether neural networks (with and without the inflow timeseries
branch) outperform the existing tabular Random Forest baseline.

---

## Problem Statement

**At inference:** user provides a scalar multiplier (e.g. 3.32). The system:
1. Loads the Qx1.00 base run inflow timeseries from the database (always available).
2. Generates a synthetic inflow hydrograph: `synthetic(t) = qx1_inflow(t) × multiplier`.
3. Feeds the synthetic hydrograph + static node features through the surrogate CNN.
4. Returns per-node predictions: `flooded? (bool)` and `peak_flooding_lps (float)`.

No SWMM execution is required after training.

---

## Training Data

**Source:** 16 Parquet files registered in `temporal_artifacts`, multipliers Qx1.00–Qx4.75
(increments of 0.25). Flooding begins around Qx1.2, so 15 of 16 runs produce flooded nodes.

**Sample unit:** one `(run_id, node_id)` pair — no sliding windows.

**Sequence input:** full simulation timeseries for the node, resampled to 5-min intervals
(forward-fill, same dedup logic as SP2). Shape: `[T_total, 6]` where `T_total` varies by run
but is typically 24–30 steps for a 2-hour simulation at 5-min resolution.

**Temporal features (TEMPORAL_COLS, 6 channels):**
```
total_inflow_lps, lateral_inflow_lps,
depth_m, depth_ratio, flooding_lps, total_outflow_lps
```

**Static features (STATIC_COLS, 7 features):**
```
full_depth_m, in_degree, out_degree,
upstream_diam_avg_m, downstream_diam_avg_m,
upstream_capacity_lps, downstream_capacity_lps
```

**Labels (per full run, per node):**
- `y_class = int((flooding_lps > 0).any())` — did the node flood at all?
- `y_reg   = float(flooding_lps.max())`     — peak flooding in lps

**Dataset size:** ~16 runs × ~200 nodes ≈ 3,200 samples
(versus ~64,000 for the sliding-window early-warning CNN).

---

## Model Architecture: `SWMMSurrogateCNN`

Dual-branch architecture, dual output head.

```
X_seq    [batch, T_total, 6]
    → permute → [batch, 6, T_total]
    → Conv1d(6→32, k=3, pad=1) → BN → ReLU
    → Conv1d(32→64, k=3, pad=1) → BN → ReLU
    → AdaptiveAvgPool1d(1) → squeeze → [batch, 64]   ← handles variable T_total

X_static [batch, 7]
    → Linear(7→32) → ReLU
    → Linear(32→32) → ReLU
                                                      → [batch, 32]

concat([temporal, static]) → [batch, 96]
    → Linear(96→64) → ReLU → Dropout(0.3)            → [batch, 64]

Classification head: Linear(64→1) → Sigmoid           → [batch, 1]  (flood probability)
Regression head:    Linear(64→1)                      → [batch, 1]  (peak_flooding_lps)
```

`AdaptiveAvgPool1d(1)` makes the model sequence-length-agnostic: runs of different durations
require no padding.

**File:** `swmm_resilience/ml/temporal/models/surrogate_cnn.py`
(separate from `cnn.py` which remains unchanged for the early-warning CNN)

---

## Loss Function

Multi-task loss combining both heads:

```
total_loss = α × BCEWithLogitsLoss(pos_weight=w) + β × MSELoss()

α = 1.0  (classification weight)
β = 0.01 (regression weight — peak_lps is ~100× scale of binary labels)
w = (n_negative / n_positive) computed per training fold
```

`BCEWithLogitsLoss` is used instead of `BCELoss` for numerical stability and to support
`pos_weight`. The classification sigmoid is applied only at inference, not during training.

---

## Training Pipeline

**Cross-validation:** GroupKFold (k=5) on `run_id`. Each fold holds out entire runs so the model
is tested on multiplier values it never saw during training. With 16 runs, each val fold contains
~3 unseen multiplier levels.

**Scalers:** `StandardScaler` fit on training fold only.
- Sequence scaler: fitted on `X_seq` reshaped to `[N×T, 6]`, then reshaped back.
- Static scaler: fitted on `X_static [N, 7]`.

**Optimizer:** AdamW, lr=1e-3, with `ReduceLROnPlateau(patience=5)`.

**Epochs:** 100 (more than early-warning CNN due to smaller dataset).

**Best fold selection:** fold with lowest validation BCE loss is persisted.

**Artifacts saved:**
```
surrogate_cnn_weights.pt
surrogate_cnn_scaler_seq.joblib
surrogate_cnn_scaler_static.joblib
surrogate_cnn_metrics.csv
```

All saved to `DEFAULT_TEMPORAL_ARTIFACTS_DIR`.

**File:** `swmm_resilience/ml/temporal/train_surrogate.py`
(separate from `train_cnn.py` which remains unchanged)

---

## Comparison Experiment

The scientific questions are:
1. Do neural networks outperform the existing tabular Random Forest?
2. Does including the inflow timeseries improve predictions over static features alone?

Three models are trained and compared:

| Model | Inputs | File |
|-------|--------|------|
| RF baseline (existing) | multiplier scalar + STATIC_COLS | `train.py` |
| Surrogate CNN (full) | inflow hydrograph + STATIC_COLS | `train_surrogate.py` |
| Surrogate CNN (ablation, no temporal) | `[multiplier]` scalar + STATIC_COLS | `train_surrogate.py --no-temporal` |

The `--no-temporal` flag disables the temporal branch and appends the multiplier scalar as an extra
static feature. Same architecture and training loop, making the comparison fair.

**Metrics reported per fold and mean across folds:**

| Task | Metrics |
|------|---------|
| Classification | AUC-ROC, F1, Precision, Recall, Accuracy |
| Regression | MAE, RMSE, R² |

---

## Inference: `predict_surrogate_from_multiplier`

```python
def predict_surrogate_from_multiplier(
    multiplier: float,
    db_path: Path = DEFAULT_DB_FILE,
    artifacts_dir: Path = DEFAULT_TEMPORAL_ARTIFACTS_DIR,
    network_hash: str | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict flooding for all nodes given an inflow multiplier.

    No SWMM run required. Uses Qx1.00 base hydrograph scaled by `multiplier`.

    Returns DataFrame: node_id, flood_prob, predicted_flooded, peak_flooding_lps_pred
    """
```

**Steps at inference:**
1. Load Qx1.00 Parquet from `temporal_artifacts` (lowest multiplier run).
2. For each node: scale `total_inflow_lps` and `lateral_inflow_lps` by `multiplier`.
   Set SWMM-output features (`depth_m`, `depth_ratio`, `flooding_lps`, `total_outflow_lps`) to 0.
3. Resample to 5-min grid (same logic as training).
4. Apply saved scalers.
5. Forward pass through `SWMMSurrogateCNN`.
6. Return per-node results.

**Added to:** `swmm_resilience/ml/temporal/predict.py`

---

## Files Modified / Created

| File | Change |
|------|--------|
| `swmm_resilience/ml/temporal/models/surrogate_cnn.py` | **New** — `SWMMSurrogateCNN` with dual head |
| `swmm_resilience/ml/temporal/dataset.py` | **Modify** — add `build_surrogate_dataset()` |
| `swmm_resilience/ml/temporal/train_surrogate.py` | **New** — training script with `--no-temporal` flag |
| `swmm_resilience/ml/temporal/predict.py` | **Modify** — add `predict_surrogate_from_multiplier()` |

Existing files (`train_cnn.py`, `models/cnn.py`, `dataset.py` windows logic) are **unchanged** —
the early-warning CNN remains intact alongside the surrogate.

---

## Out of Scope

- LSTM surrogate (SP4 — separate spec)
- Desktop integration (SP5 — separate spec)
- Timestep-by-timestep surrogate output (deferred, noted as future option)
- Training on networks other than `chico_hydro-qx1`
