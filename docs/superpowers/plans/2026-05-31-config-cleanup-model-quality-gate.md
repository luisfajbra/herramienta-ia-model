# Config Cleanup Model Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the remaining `swmm_resilience/config.py` edits by keeping validated static hydraulic features while restoring PCA to the known 5-component baseline.

**Architecture:** Add one focused preprocessing test that proves topology/capacity columns are selected as model inputs while hydraulic result columns remain excluded. Then update `config.py` so those static features stay out of `ML_DROP_COLUMNS` and `ML_PCA_COMPONENTS` returns to `5`. Finish with focused preprocessing verification and full-suite verification when practical.

**Tech Stack:** Python, pytest, pandas, git.

---

## File Structure

- Create `tests/ml/test_preprocessing_feature_contract.py`: verifies tabular feature selection keeps static hydraulic topology/capacity inputs and drops result/leakage columns.
- Modify `swmm_resilience/config.py`: restore `ML_PCA_COMPONENTS = 5`; keep `in_degree`, `out_degree`, `upstream_capacity_lps`, and `downstream_capacity_lps` absent from `ML_DROP_COLUMNS`.
- No production code beyond `config.py` should change in this cleanup.
- No files should be deleted.

---

### Task 1: Add Tabular Static Feature Contract Test

**Files:**
- Create: `tests/ml/test_preprocessing_feature_contract.py`
- Read: `swmm_resilience/ml/preprocessing.py`
- Read: `swmm_resilience/config.py`

- [ ] **Step 1: Create the failing/pinning test file**

Create `tests/ml/test_preprocessing_feature_contract.py` with this exact content:

```python
import pandas as pd

from swmm_resilience.ml.preprocessing import get_feature_columns


def test_tabular_feature_selection_keeps_static_hydraulic_features():
    df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "node_id": ["J1"],
            "scenario_type": ["steady"],
            "spatial_pattern": ["uniform"],
            "delta_inflow_lps": [0.0],
            "inflow_multiplier": [2.0],
            "invert_elev_m": [100.0],
            "full_depth_m": [2.0],
            "base_inflow_lps": [1.5],
            "in_degree": [2],
            "out_degree": [1],
            "upstream_capacity_lps": [50.0],
            "downstream_capacity_lps": [40.0],
            "flooded": [1],
            "peak_flooding_lps": [12.0],
            "flooding_duration_min": [5.0],
            "max_depth_m": [1.8],
            "max_depth_ratio": [0.9],
            "time_to_peak_min": [15.0],
            "depth_rate_m_per_min": [0.2],
            "max_total_outflow_lps": [30.0],
            "time_to_peak_outflow_min": [20.0],
        }
    )

    features = get_feature_columns(df, target="peak_flooding_lps")

    assert "inflow_multiplier" in features
    assert "invert_elev_m" in features
    assert "full_depth_m" in features
    assert "base_inflow_lps" in features
    assert "in_degree" in features
    assert "out_degree" in features
    assert "upstream_capacity_lps" in features
    assert "downstream_capacity_lps" in features


def test_tabular_feature_selection_drops_result_and_metadata_columns():
    df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "node_id": ["J1"],
            "scenario_type": ["steady"],
            "spatial_pattern": ["uniform"],
            "delta_inflow_lps": [0.0],
            "inflow_multiplier": [2.0],
            "in_degree": [2],
            "out_degree": [1],
            "upstream_capacity_lps": [50.0],
            "downstream_capacity_lps": [40.0],
            "flooded": [1],
            "peak_flooding_lps": [12.0],
            "flooding_duration_min": [5.0],
            "max_depth_m": [1.8],
            "max_depth_ratio": [0.9],
            "time_to_peak_min": [15.0],
            "depth_rate_m_per_min": [0.2],
            "max_total_outflow_lps": [30.0],
            "time_to_peak_outflow_min": [20.0],
        }
    )

    features = get_feature_columns(df, target="peak_flooding_lps")

    assert "run_id" not in features
    assert "node_id" not in features
    assert "delta_inflow_lps" not in features
    assert "flooded" not in features
    assert "peak_flooding_lps" not in features
    assert "flooding_duration_min" not in features
    assert "max_depth_m" not in features
    assert "max_depth_ratio" not in features
    assert "time_to_peak_min" not in features
    assert "depth_rate_m_per_min" not in features
    assert "max_total_outflow_lps" not in features
    assert "time_to_peak_outflow_min" not in features
```

- [ ] **Step 2: Run the new test before editing config further**

Run:

```bash
pytest tests/ml/test_preprocessing_feature_contract.py -v
```

Expected result with the currently unstaged config state: both tests pass, proving the topology/capacity feature change is already present locally and the result columns are still dropped.

- [ ] **Step 3: Commit the test only after the config cleanup in Task 2**

Do not commit yet. The test should be committed together with the intentional `config.py` cleanup so the repository has one coherent local commit for this gate.

---

### Task 2: Resolve `config.py` Intentionally

**Files:**
- Modify: `swmm_resilience/config.py`
- Test: `tests/ml/test_preprocessing_feature_contract.py`

- [ ] **Step 1: Restore PCA to the baseline value**

In `swmm_resilience/config.py`, change:

```python
ML_PCA_COMPONENTS = 7
```

to:

```python
ML_PCA_COMPONENTS = 5
```

- [ ] **Step 2: Keep static topology/capacity columns out of `ML_DROP_COLUMNS`**

Ensure `ML_DROP_COLUMNS` does not contain these four columns:

```python
"in_degree",
"out_degree",
"upstream_capacity_lps",
"downstream_capacity_lps",
```

The relevant block should remain:

```python
ML_DROP_COLUMNS = [
    "run_id",
    "node_id",
    "scenario_type",
    "spatial_pattern",
    "delta_inflow_lps",
    "upstream_diam_avg_m",
    "downstream_diam_avg_m",
    "flooded",
    "peak_flooding_lps",
    "flooding_duration_min",
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "max_total_outflow_lps",
    "time_to_peak_outflow_min",
]
```

- [ ] **Step 3: Verify the config diff is exactly the intended feature cleanup**

Run:

```bash
git diff -- swmm_resilience/config.py
```

Expected output should show only the four removed `ML_DROP_COLUMNS` entries relative to `HEAD`. It should not show `ML_PCA_COMPONENTS = 7`.

- [ ] **Step 4: Run the focused feature contract test**

Run:

```bash
pytest tests/ml/test_preprocessing_feature_contract.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Run the existing config default guardrail**

Run:

```bash
pytest tests/test_config_defaults.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit the config cleanup**

Run:

```bash
git add swmm_resilience/config.py tests/ml/test_preprocessing_feature_contract.py
git commit -m "fix: keep static hydraulic tabular features"
```

Expected: one local commit. Do not push.

---

### Task 3: Repository Verification And Audit Note

**Files:**
- Modify if verification fails: `docs/model_hydraulic_prediction_audit_2026-05-31.md`
- Read: `docs/superpowers/specs/2026-05-31-config-cleanup-model-quality-gate-design.md`

- [ ] **Step 1: Run the focused stabilization suite**

Run:

```bash
pytest tests/test_config_defaults.py tests/test_run_experiment_input_semantics.py tests/simulation/test_partial_timeseries_scaling.py tests/ml/temporal/test_window_builder.py tests/ml/temporal/test_surrogate_dataset.py tests/ml/temporal/test_train_surrogate.py tests/ml/temporal/test_surrogate_predict.py tests/ml/temporal/test_predict_from_parquet_task_schema.py tests/ml/test_preprocessing_feature_contract.py -v
```

Expected: all selected tests pass. Warnings from dependency deprecations or manifest guardrail tests are acceptable if tests pass.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -v
```

Expected: full suite passes. If it fails, do not hide the failure. Capture:

- failing test path
- failing assertion or exception
- whether the failure is related to this config cleanup

- [ ] **Step 3: If full-suite verification fails, append a verification note**

Only perform this step if `pytest -v` fails.

Append this section to `docs/model_hydraulic_prediction_audit_2026-05-31.md`.
Use the actual failing test path and reason from the pytest output; do not use
generic wording.

The note must have these exact headings:

- `## Config cleanup verification note`
- `Full-suite verification command`
- `Result`
- `Failures`
- `Relation to cleanup`

The `Result` line must say how many tests failed. The `Failures` list must
include each failing pytest node id and the first assertion or exception message
shown by pytest. The `Relation to cleanup` sentence must classify the failure as
`related`, `unrelated`, or `unclear`, followed by one sentence of evidence.

- [ ] **Step 4: Commit the verification note only if one was needed**

Run this only if Step 3 modified the audit file:

```bash
git add docs/model_hydraulic_prediction_audit_2026-05-31.md
git commit -m "docs: record config cleanup verification"
```

Expected: one local docs commit. Do not push.

- [ ] **Step 5: Confirm the final local state**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected:

- No tracked unstaged `swmm_resilience/config.py` cleanup remains.
- Existing unrelated untracked files may still appear and should not be deleted.
- The newest local commit is either `fix: keep static hydraulic tabular features` or `docs: record config cleanup verification` if a verification note was needed.
