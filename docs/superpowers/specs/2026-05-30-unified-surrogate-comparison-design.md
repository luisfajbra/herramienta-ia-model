# Unified Surrogate Comparison — Design Spec

## Goal

Train RF/XGBoost and the surrogate CNN on **identical features and identical data splits**
so their metrics are directly comparable. Both models must be able to predict flooding
**without running SWMM at inference time**.

---

## Why SWMM outputs must be excluded

The CSV exported by `export_ml_dataset()` contains aggregated SWMM outputs
(`max_depth_m`, `time_to_peak_min`, etc.). These only exist after a SWMM run.
If the model requires them as inputs, it cannot be used without SWMM — defeating
the purpose of a surrogate. Both models must train on features that are always
available before any simulation.

---

## Feature sets

### Dropped from both models (SWMM outputs — not available at inference)

```
max_depth_m, max_depth_ratio, time_to_peak_min,
depth_rate_m_per_min, max_total_outflow_lps,
time_to_peak_outflow_min, downstream_link_peak_flows_lps_json
```

### Inference-available features used by BOTH models

| Feature | Source | Type |
|---------|--------|------|
| `inflow_multiplier` | `runs` table / CSV | float |
| `invert_elev_m` | `network_nodes` / CSV | float |
| `full_depth_m` | `network_nodes` / CSV | float |
| `base_inflow_lps` | `network_nodes` / CSV | float |
| `node_type` | `network_nodes` / CSV | categorical (one-hot for CNN) |
| `in_degree` | `network_nodes` / CSV | int |
| `out_degree` | `network_nodes` / CSV | int |
| `upstream_pipes_count` | `network_nodes` / CSV | int |
| `upstream_diam_max_m` | `network_nodes` / CSV | float |
| `upstream_diam_min_m` | `network_nodes` / CSV | float |
| `upstream_diam_avg_m` | `network_nodes` / CSV | float |
| `upstream_slope_avg` | `network_nodes` / CSV | float |
| `upstream_slope_max` | `network_nodes` / CSV | float |
| `upstream_capacity_lps` | `network_nodes` / CSV | float |
| `downstream_pipes_count` | `network_nodes` / CSV | int |
| `downstream_diam_max_m` | `network_nodes` / CSV | float |
| `downstream_diam_min_m` | `network_nodes` / CSV | float |
| `downstream_diam_avg_m` | `network_nodes` / CSV | float |
| `downstream_slope_avg` | `network_nodes` / CSV | float |
| `downstream_slope_max` | `network_nodes` / CSV | float |
| `downstream_capacity_lps` | `network_nodes` / CSV | float |

**21 features total** (20 numeric + 1 categorical). `node_type` has exactly 2 values
(`junction`, `outfall`) — one-hot encoding adds 1 binary column, keeping the total at 21.
`upstream_diam_avg_m`, `upstream_capacity_lps` have NaNs for source nodes (no upstream
pipes); `downstream_diam_avg_m`, `downstream_capacity_lps` have NaNs for outfall nodes.
All NaNs are filled with 0.0 before training.

### CNN-only: temporal branch

Raw inflow timeseries for each `(run_id, node_id)` from the Parquet files:

```
total_inflow_lps   [T timesteps]
lateral_inflow_lps [T timesteps]
```

These are **not** in the CSV. The CNN reads them from the Parquet files and
joins them with the CSV static features by `(run_id, node_id)`.

---

## Model inputs summary

| Model | Inputs |
|-------|--------|
| RF / XGBoost | 21 static+scalar features (flat vector) |
| CNN (full) | Inflow timeseries [T, 2] + 21 static features |
| CNN (ablation, no temporal) | 21 static features (multiplier already included) |

---

## Dataset construction: `build_unified_dataset()`

**File:** `swmm_resilience/ml/temporal/dataset.py` (append new function)

**Logic:**
1. Read `DEFAULT_OUTPUT_CSV` — one row per `(run_id, node_id)`
2. Drop SWMM-output columns and metadata columns (`network_hash`, `network_file`,
   `scenario_type`, `spatial_pattern`)
3. One-hot encode `node_type`
4. For each row, load the matching Parquet file (via `temporal_artifacts` DB table)
   and extract `[total_inflow_lps, lateral_inflow_lps]` timeseries for that `node_id`
5. Resample to 5-min grid, forward-fill (same logic as existing surrogate dataset)
6. Zero-pad sequences to T_max across all samples
7. Return `TemporalWindowDataset`:
   - `X_seq`: `[N, T_max, 2]` — inflow timeseries
   - `X_static`: `[N, n_static_features]` — one-hot encoded static features
   - `y_class`: `[N]` — `flooded` column (0/1)
   - `y_reg`: `[N]` — `peak_flooding_lps` column
   - `groups`: `[N]` — `run_id` (for GroupKFold)
   - `meta`: DataFrame with `run_id`, `node_id`
   - `static_feature_names`: list of column names (for RF feature access)

**Signature:**
```python
def build_unified_dataset(
    csv_path: Path = DEFAULT_OUTPUT_CSV,
    db_path: Path = DEFAULT_DB_FILE,
    resample_min: int = 5,
) -> TemporalWindowDataset:
```

The dataset is shared: RF reads `X_static` as a flat tabular matrix; CNN uses
both `X_seq` and `X_static`.

---

## Comparison pipeline: `compare_surrogate.py`

**File:** `swmm_resilience/ml/temporal/compare_surrogate.py` (new file)

**Logic:**
1. Call `build_unified_dataset()` once
2. Run `GroupKFold(n_splits=5)` on `run_id` — same splits for both models
3. For each fold:
   - Train XGBoost classifier on `X_static[train]` → evaluate on `X_static[val]`
   - Train `SWMMSurrogateCNN` on `(X_seq[train], X_static[train])` → evaluate on val
4. Collect per-fold metrics for both models
5. Print and save comparison table

**Models in comparison:**

| Model | Library | Features |
|-------|---------|---------|
| XGBoost | xgboost | X_static (21 features) |
| CNN full | PyTorch | X_seq + X_static |
| CNN ablation | PyTorch | X_static only (use_temporal=False) |

**Metrics reported (per fold + mean ± std):**

| Task | Metrics |
|------|---------|
| Classification | AUC-ROC, F1, Precision, Recall, Accuracy |
| Regression | MAE, RMSE, R² |

**Output:**
- Printed table to stdout
- `comparison_results.csv` saved to `DEFAULT_TEMPORAL_ARTIFACTS_DIR`

**CLI:**
```bash
python -m swmm_resilience.ml.temporal.compare_surrogate
```

---

## Targets

- `y_class`: `flooded` column (already binary 0/1, no NaNs)
- `y_reg`: `peak_flooding_lps` column

Class balance in current dataset: 1712 non-flooded / 864 flooded (~2:1).
XGBoost uses `scale_pos_weight = n_neg / n_pos`. CNN uses `BCEWithLogitsLoss(pos_weight)`.

---

## Files modified / created

| File | Change |
|------|--------|
| `swmm_resilience/ml/temporal/dataset.py` | Append `build_unified_dataset()` |
| `swmm_resilience/ml/temporal/compare_surrogate.py` | New — comparison pipeline |
| `tests/ml/temporal/test_unified_dataset.py` | New — dataset builder tests |
| `tests/ml/temporal/test_compare_surrogate.py` | New — comparison pipeline tests |

Existing files (`train_surrogate.py`, `train.py`, `predict.py`) are **unchanged**.

---

## Out of scope

- Retraining the existing surrogate CNN artifacts (this is a separate comparison experiment)
- LSTM surrogate (SP4)
- Desktop integration (SP5)
- Hyperparameter tuning for XGBoost
