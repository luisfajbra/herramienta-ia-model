# Safe Stabilization Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent misleading hydraulic ML predictions by enforcing input semantics, feature contracts, deterministic neural training, artifact manifests, and inference guardrails.

**Architecture:** Add small validation and metadata helpers around the existing SWMM, temporal dataset, training, and inference modules. Keep model architecture mostly unchanged; this pass stabilizes data contracts and saved artifacts before later model-quality work. Tests lead every change and focus on the audit's correctness risks.

**Tech Stack:** Python, pytest, pandas, numpy, scikit-learn, PyTorch, joblib, SQLite, PySWMM/swmm-api.

---

## File Structure

- Modify `swmm_resilience/config.py`: point `DEFAULT_INP_FILE` at an existing network file.
- Modify `swmm_resilience/main.py`: reject the misleading `delta_inflows_lps` alias during this stabilization pass.
- Modify `swmm_resilience/simulation/swmm_api_io.py`: make partial timeseries scaling isolate selected nodes instead of scaling shared series globally.
- Modify `swmm_resilience/ml/temporal/dataset.py`: define temporal feature contracts and make deployable temporal builders use only pre-SWMM inflow columns.
- Modify `swmm_resilience/ml/temporal/schemas.py`: carry optional metadata such as feature names and sequence lengths.
- Modify `swmm_resilience/ml/temporal/train_surrogate.py`: add deterministic seed setup, final full-data retraining after CV, transformed regression target, and a temporal manifest.
- Modify `swmm_resilience/ml/temporal/predict.py`: validate manifests, warn on Qx extrapolation, and separate classification/regression output schemas.
- Create `tests/test_config_defaults.py`: default path validation.
- Create `tests/test_run_experiment_input_semantics.py`: multiplier vs delta guardrail.
- Create `tests/simulation/test_partial_timeseries_scaling.py`: shared-timeseries partial scaling regression test.
- Modify `tests/ml/temporal/test_window_builder.py`: assert temporal predictor feature contracts exclude SWMM outputs.
- Modify `tests/ml/temporal/test_train_surrogate.py`: assert manifest, final full-data training metadata, deterministic seed metadata.
- Modify `tests/ml/temporal/test_surrogate_predict.py`: assert extrapolation warning and manifest mismatch validation.
- Create `tests/ml/temporal/test_predict_from_parquet_task_schema.py`: task-specific inference output schemas.

---

### Task 1: Input Semantics And Default Path Guardrails

**Files:**
- Modify: `swmm_resilience/config.py:29-34`
- Modify: `swmm_resilience/main.py:145-172`
- Create: `tests/test_config_defaults.py`
- Create: `tests/test_run_experiment_input_semantics.py`

- [ ] **Step 1: Write failing tests for default `.inp` path and rejected `delta_inflows_lps`**

Create `tests/test_config_defaults.py`:

```python
from swmm_resilience.config import DEFAULT_INP_FILE


def test_default_inp_file_exists():
    assert DEFAULT_INP_FILE.exists(), f"DEFAULT_INP_FILE does not exist: {DEFAULT_INP_FILE}"
```

Create `tests/test_run_experiment_input_semantics.py`:

```python
import pytest

from swmm_resilience.main import run_experiment


def test_delta_inflows_lps_is_rejected_before_running_simulation(tmp_path):
    inp = tmp_path / "network.inp"
    inp.write_text("[TITLE]\nminimal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="delta_inflows_lps"):
        run_experiment(
            inp_file=inp,
            db_file=tmp_path / "test.db",
            output_csv=tmp_path / "dataset.csv",
            delta_inflows_lps=[10.0],
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_config_defaults.py tests/test_run_experiment_input_semantics.py -v
```

Expected:
- `test_default_inp_file_exists` fails while `DEFAULT_INP_FILE` points at the missing steady file in `chico_hydro-qx1`.
- `test_delta_inflows_lps_is_rejected_before_running_simulation` fails because `run_experiment()` currently treats `delta_inflows_lps` as multipliers.

- [ ] **Step 3: Fix `DEFAULT_INP_FILE`**

In `swmm_resilience/config.py`, replace the default network path block with:

```python
DEFAULT_NETWORK_KEY = "chico_hydro-qx1"
DEFAULT_NETWORK_DIR = NETWORKS_DIR / DEFAULT_NETWORK_KEY
DEFAULT_RESULTS_DIR = DEFAULT_NETWORK_DIR / "results"

DEFAULT_INP_FILE = DEFAULT_NETWORK_DIR / "SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
LEGACY_INP_FILE = BASE_DIR / "SWMM - Chico (PVC) Prueba 1 - Steady.inp"
```

- [ ] **Step 4: Reject `delta_inflows_lps` explicitly**

In `swmm_resilience/main.py`, replace:

```python
    if inflow_multipliers is not None and delta_inflows_lps is not None:
        raise ValueError("Usa inflow_multipliers o delta_inflows_lps, pero no ambos.")
    multipliers = _normalize_inflow_multipliers(
        inflow_multipliers if inflow_multipliers is not None else delta_inflows_lps
    )
```

with:

```python
    if delta_inflows_lps is not None:
        raise ValueError(
            "delta_inflows_lps esta deshabilitado temporalmente porque antes se "
            "interpretaba como multiplicador Qx, no como delta aditivo en L/s. "
            "Usa inflow_multipliers=[...] para escenarios Qx."
        )
    multipliers = _normalize_inflow_multipliers(inflow_multipliers)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_config_defaults.py tests/test_run_experiment_input_semantics.py -v
```

Expected: both tests pass.

Commit:

```bash
git add swmm_resilience/config.py swmm_resilience/main.py tests/test_config_defaults.py tests/test_run_experiment_input_semantics.py
git commit -m "fix: stabilize default input and scenario semantics"
```

---

### Task 2: Temporal Feature Contracts And No-Leakage Builders

**Files:**
- Modify: `swmm_resilience/ml/temporal/dataset.py:34-56`
- Modify: `swmm_resilience/ml/temporal/schemas.py`
- Modify: `tests/ml/temporal/test_window_builder.py`
- Modify: `tests/ml/temporal/test_surrogate_dataset.py`

- [ ] **Step 1: Write failing feature-contract tests**

Append to `tests/ml/temporal/test_window_builder.py`:

```python
from swmm_resilience.ml.temporal.dataset import (
    PRE_SWMM_TEMPORAL_COLS,
    SWMM_OUTPUT_TEMPORAL_COLS,
    TEMPORAL_COLS,
)


def test_deployable_temporal_features_exclude_swmm_outputs():
    assert PRE_SWMM_TEMPORAL_COLS == ["total_inflow_lps", "lateral_inflow_lps"]
    forbidden = set(SWMM_OUTPUT_TEMPORAL_COLS)
    assert forbidden
    assert not (set(PRE_SWMM_TEMPORAL_COLS) & forbidden)


def test_legacy_temporal_cols_are_explicitly_post_swmm():
    assert "flooding_lps" in SWMM_OUTPUT_TEMPORAL_COLS
    assert "depth_m" in SWMM_OUTPUT_TEMPORAL_COLS
    assert "total_outflow_lps" in SWMM_OUTPUT_TEMPORAL_COLS
    assert set(TEMPORAL_COLS) == set(PRE_SWMM_TEMPORAL_COLS + SWMM_OUTPUT_TEMPORAL_COLS)
```

Append to `tests/ml/temporal/test_surrogate_dataset.py`:

```python
from swmm_resilience.ml.temporal.dataset import PRE_SWMM_TEMPORAL_COLS


def test_surrogate_meta_records_temporal_feature_names(tmp_path):
    db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
    ds = build_surrogate_dataset(db_path=db_path)
    assert ds.meta.attrs["temporal_feature_names"] == PRE_SWMM_TEMPORAL_COLS
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/ml/temporal/test_window_builder.py tests/ml/temporal/test_surrogate_dataset.py -v
```

Expected: failure because `PRE_SWMM_TEMPORAL_COLS` and `SWMM_OUTPUT_TEMPORAL_COLS` do not exist and metadata attrs are not set.

- [ ] **Step 3: Add explicit feature groups**

In `swmm_resilience/ml/temporal/dataset.py`, replace the current `TEMPORAL_COLS` and `SURROGATE_TEMPORAL_COLS` constants with:

```python
PRE_SWMM_TEMPORAL_COLS = [
    "total_inflow_lps",
    "lateral_inflow_lps",
]

SWMM_OUTPUT_TEMPORAL_COLS = [
    "depth_m",
    "depth_ratio",
    "flooding_lps",
    "total_outflow_lps",
]

TEMPORAL_COLS = PRE_SWMM_TEMPORAL_COLS + SWMM_OUTPUT_TEMPORAL_COLS

SURROGATE_TEMPORAL_COLS = PRE_SWMM_TEMPORAL_COLS
```

- [ ] **Step 4: Attach feature metadata to datasets**

In `build_temporal_windows()`, after `meta_rows` becomes a DataFrame, set attrs before returning:

```python
    meta = pd.DataFrame(meta_rows)
    meta.attrs["temporal_feature_names"] = TEMPORAL_COLS
    meta.attrs["static_feature_names"] = STATIC_COLS
```

and return `meta=meta`.

In `build_surrogate_dataset()`, replace:

```python
        meta=pd.DataFrame(meta_rows),
```

with:

```python
        meta=_with_dataset_attrs(
            pd.DataFrame(meta_rows),
            temporal_feature_names=SURROGATE_TEMPORAL_COLS,
            static_feature_names=STATIC_COLS + ([] if use_temporal else ["inflow_multiplier"]),
        ),
```

Add helper near the constants:

```python
def _with_dataset_attrs(
    meta: pd.DataFrame,
    *,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
) -> pd.DataFrame:
    meta.attrs["temporal_feature_names"] = list(temporal_feature_names)
    meta.attrs["static_feature_names"] = list(static_feature_names)
    return meta
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/ml/temporal/test_window_builder.py tests/ml/temporal/test_surrogate_dataset.py -v
```

Expected: pass.

Commit:

```bash
git add swmm_resilience/ml/temporal/dataset.py swmm_resilience/ml/temporal/schemas.py tests/ml/temporal/test_window_builder.py tests/ml/temporal/test_surrogate_dataset.py
git commit -m "fix: declare temporal feature contracts"
```

---

### Task 3: Partial Timeseries Scaling Isolation

**Files:**
- Modify: `swmm_resilience/simulation/swmm_api_io.py:150-238`
- Create: `tests/simulation/test_partial_timeseries_scaling.py`

- [ ] **Step 1: Write failing test for shared timeseries partial scaling**

Create `tests/simulation/test_partial_timeseries_scaling.py`:

```python
from pathlib import Path

import pytest

from swmm_resilience.config import SCENARIO_MODE_TIMESERIES
from swmm_resilience.simulation.swmm_api_io import load_inp, write_scaled_inp


pytest.importorskip("swmm_api")


def _write_shared_timeseries_inp(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[TITLE]",
                "shared timeseries",
                "[OPTIONS]",
                "FLOW_UNITS           LPS",
                "INFILTRATION         HORTON",
                "FLOW_ROUTING         KINWAVE",
                "START_DATE           01/01/2020",
                "START_TIME           00:00:00",
                "REPORT_START_DATE    01/01/2020",
                "REPORT_START_TIME    00:00:00",
                "END_DATE             01/01/2020",
                "END_TIME             01:00:00",
                "SWEEP_START          01/01",
                "SWEEP_END            12/31",
                "DRY_DAYS             0",
                "REPORT_STEP          00:05:00",
                "WET_STEP             00:05:00",
                "DRY_STEP             00:05:00",
                "ROUTING_STEP         0:05:00",
                "[JUNCTIONS]",
                "J1 0 1 0 0 0",
                "J2 0 1 0 0 0",
                "[OUTFALLS]",
                "O1 0 FREE NO",
                "[CONDUITS]",
                "C1 J1 O1 100 0.013 0 0 0 0",
                "C2 J2 O1 100 0.013 0 0 0 0",
                "[XSECTIONS]",
                "C1 CIRCULAR 0.3 0 0 0 1",
                "C2 CIRCULAR 0.3 0 0 0 1",
                "[INFLOWS]",
                "J1 FLOW Shared 1.0 1.0 FLOW",
                "J2 FLOW Shared 1.0 1.0 FLOW",
                "[TIMESERIES]",
                "Shared 0:00 10",
                "Shared 1:00 20",
                "[END]",
            ]
        ),
        encoding="utf-8",
    )


def test_partial_scaling_duplicates_shared_timeseries_for_selected_node(tmp_path):
    inp_path = tmp_path / "shared.inp"
    out_path = tmp_path / "scaled.inp"
    _write_shared_timeseries_inp(inp_path)

    write_scaled_inp(
        inp_path,
        multiplier=2.0,
        target_nodes={"J1"},
        output_file=out_path,
        scenario_mode=SCENARIO_MODE_TIMESERIES,
    )

    scaled = load_inp(out_path)
    j1 = scaled["INFLOWS"][("J1", "FLOW")]
    j2 = scaled["INFLOWS"][("J2", "FLOW")]
    assert str(j1.time_series) != str(j2.time_series)

    j1_values = [value for _time, value in scaled["TIMESERIES"][str(j1.time_series)].data]
    j2_values = [value for _time, value in scaled["TIMESERIES"][str(j2.time_series)].data]
    assert j1_values == [20.0, 40.0]
    assert j2_values == [10.0, 20.0]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/simulation/test_partial_timeseries_scaling.py -v
```

Expected: failure because both J1 and J2 still reference the same scaled series.

- [ ] **Step 3: Implement selected-node timeseries duplication**

In `swmm_resilience/simulation/swmm_api_io.py`, replace `_scale_target_timeseries()` with this implementation:

```python
def _clone_timeseries_object(ts_obj, scaled_data):
    cloned = ts_obj.copy() if hasattr(ts_obj, "copy") else type(ts_obj).__new__(type(ts_obj))
    if not hasattr(ts_obj, "copy"):
        cloned.__dict__.update(getattr(ts_obj, "__dict__", {}))
    cloned.data = list(scaled_data)
    return cloned


def _unique_timeseries_name(inp, base_name: str, node_id: str) -> str:
    candidate = f"{base_name}__scaled_{node_id}"
    index = 1
    while candidate in inp["TIMESERIES"]:
        index += 1
        candidate = f"{base_name}__scaled_{node_id}_{index}"
    return candidate


def _scale_target_timeseries(inp, selected_inflows: list[tuple[str, object]], multiplier: float) -> int:
    """Scale timeseries values used by selected nodes without mutating unselected users."""
    if "TIMESERIES" not in inp:
        raise ValueError(
            "El escenario 'timeseries' requiere una seccion [TIMESERIES] en el archivo .inp."
        )

    changed_values = 0
    missing_series: list[str] = []

    for node_id, inflow in selected_inflows:
        ts_name = _normalize_timeseries_name(inflow.time_series)
        if ts_name is None:
            continue
        if ts_name not in inp["TIMESERIES"]:
            missing_series.append(ts_name)
            continue

        ts_obj = inp["TIMESERIES"][ts_name]
        scaled_data = []
        for time_value, flow_value in ts_obj.data:
            scaled_value = flow_value * multiplier
            if scaled_value != flow_value:
                changed_values += 1
            scaled_data.append((time_value, scaled_value))

        new_name = _unique_timeseries_name(inp, str(ts_name), str(node_id))
        inp["TIMESERIES"][new_name] = _clone_timeseries_object(ts_obj, scaled_data)
        inflow.time_series = new_name

    if missing_series:
        missing = ", ".join(sorted(set(missing_series)))
        raise ValueError(
            "El escenario 'timeseries' referencia series que no existen en [TIMESERIES]: "
            f"{missing}"
        )
    if changed_values == 0:
        raise ValueError(
            "El escenario 'timeseries' no encontro valores de serie temporal para escalar. "
            "Usa el modo 'steady' si el caudal esta en Baseline dentro de [INFLOWS]."
        )
    return changed_values
```

- [ ] **Step 4: Run test and commit**

Run:

```bash
pytest tests/simulation/test_partial_timeseries_scaling.py -v
```

Expected: pass.

Commit:

```bash
git add swmm_resilience/simulation/swmm_api_io.py tests/simulation/test_partial_timeseries_scaling.py
git commit -m "fix: isolate partial timeseries scaling"
```

---

### Task 4: Deterministic Surrogate Training, Final Full-Data Fit, And Manifest

**Files:**
- Modify: `swmm_resilience/ml/temporal/train_surrogate.py`
- Modify: `tests/ml/temporal/test_train_surrogate.py`

- [ ] **Step 1: Write failing manifest and final-training tests**

Append to `tests/ml/temporal/test_train_surrogate.py`:

```python
import json


def test_surrogate_manifest_records_training_contract(tmp_path):
    dataset = _synthetic_surrogate_dataset(n_runs=4, n_nodes=3, T=8)
    train_surrogate(
        artifacts_dir=tmp_path / "artifacts",
        n_epochs=1,
        batch_size=8,
        n_cv_folds=2,
        _dataset=dataset,
    )
    manifest = json.loads((tmp_path / "artifacts" / "surrogate_cnn_manifest.json").read_text())
    assert manifest["model_type"] == "cnn"
    assert manifest["seed"] == 42
    assert manifest["trained_run_ids"] == ["run_00", "run_01", "run_02", "run_03"]
    assert manifest["temporal_feature_names"]
    assert manifest["static_feature_names"]
    assert manifest["regression_target_transform"] == "log1p"


def test_train_surrogate_returns_final_training_marker(tmp_path):
    dataset = _synthetic_surrogate_dataset(n_runs=4, n_nodes=3, T=8)
    result = train_surrogate(
        artifacts_dir=tmp_path / "artifacts",
        n_epochs=1,
        batch_size=8,
        n_cv_folds=2,
        _dataset=dataset,
    )
    assert result["final_model_trained_on_all_groups"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/ml/temporal/test_train_surrogate.py -v
```

Expected: new tests fail because no manifest is written and return value has no final-training marker.

- [ ] **Step 3: Add seed and manifest helpers**

In `swmm_resilience/ml/temporal/train_surrogate.py`, add imports:

```python
import json
import random
from datetime import datetime, timezone
from typing import Any
```

Add helpers near `_MODEL_PREFIXES`:

```python
DEFAULT_SURROGATE_SEED = 42


def _set_torch_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _feature_names(dataset: TemporalWindowDataset, fallback_temporal: list[str], fallback_static_count: int) -> tuple[list[str], list[str]]:
    temporal = list(dataset.meta.attrs.get("temporal_feature_names", fallback_temporal))
    static = list(dataset.meta.attrs.get("static_feature_names", [f"static_{i}" for i in range(fallback_static_count)]))
    return temporal, static


def _write_surrogate_manifest(
    artifacts_dir: Path,
    *,
    prefix: str,
    model_type: str,
    seed: int,
    dataset: TemporalWindowDataset,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
    use_temporal: bool,
) -> Path:
    run_ids = sorted({str(group) for group in dataset.groups.tolist()})
    manifest = {
        "model_type": model_type,
        "prefix": prefix,
        "seed": int(seed),
        "use_temporal": bool(use_temporal),
        "trained_run_ids": run_ids,
        "trained_rows": int(dataset.X_seq.shape[0]),
        "temporal_feature_names": temporal_feature_names,
        "static_feature_names": static_feature_names,
        "regression_target": "peak_flooding_lps",
        "regression_target_transform": "log1p",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = artifacts_dir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
```

- [ ] **Step 4: Add final full-data fit helper**

Add helper below `_write_surrogate_manifest()`:

```python
def _fit_surrogate_final_model(
    *,
    dataset: TemporalWindowDataset,
    model_cls: type,
    use_temporal: bool,
    n_epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    beta: float,
    device: str,
) -> tuple[dict[str, torch.Tensor], StandardScaler, StandardScaler]:
    dev = torch.device(device)
    X_seq = dataset.X_seq
    X_static = dataset.X_static
    y_cls = dataset.y_class.astype(np.float32)
    y_reg_log = np.log1p(dataset.y_reg.astype(np.float32))

    N, T, F = X_seq.shape
    scaler_seq = StandardScaler()
    X_seq_sc = scaler_seq.fit_transform(X_seq.reshape(-1, F)).reshape(N, T, F)
    scaler_static = StandardScaler()
    X_static_sc = scaler_static.fit_transform(X_static)

    n_pos = max(float(y_cls.sum()), 1.0)
    n_neg = float(len(y_cls)) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(dev)

    model = model_cls(
        n_temporal_features=F,
        n_static_features=X_static.shape[1],
        use_temporal=use_temporal,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_reg = nn.HuberLoss()

    generator = torch.Generator()
    generator.manual_seed(DEFAULT_SURROGATE_SEED)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_seq_sc, dtype=torch.float32),
            torch.tensor(X_static_sc, dtype=torch.float32),
            torch.tensor(y_cls).unsqueeze(1),
            torch.tensor(y_reg_log).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    for _ in range(n_epochs):
        model.train()
        for x_seq_b, x_static_b, y_cls_b, y_reg_b in loader:
            x_seq_b = x_seq_b.to(dev)
            x_static_b = x_static_b.to(dev)
            y_cls_b = y_cls_b.to(dev)
            y_reg_b = y_reg_b.to(dev)
            optimizer.zero_grad()
            cls_logit, reg_out = model(x_seq_b if use_temporal else None, x_static_b)
            loss = alpha * criterion_cls(cls_logit, y_cls_b) + beta * criterion_reg(reg_out, y_reg_b)
            loss.backward()
            optimizer.step()

    return {key: value.cpu().clone() for key, value in model.state_dict().items()}, scaler_seq, scaler_static
```

- [ ] **Step 5: Wire seed, log target, final fit, and manifest into `train_surrogate()`**

At the top of `train_surrogate()`, before dataset building, add:

```python
    seed = DEFAULT_SURROGATE_SEED
    _set_torch_determinism(seed)
```

In each CV fold, replace:

```python
        y_reg_tr = dataset.y_reg[train_idx]
        y_reg_val = dataset.y_reg[val_idx]
```

with:

```python
        y_reg_tr = np.log1p(dataset.y_reg[train_idx].astype(np.float32))
        y_reg_val_raw = dataset.y_reg[val_idx]
        y_reg_val = np.log1p(y_reg_val_raw.astype(np.float32))
```

When evaluating regression, replace:

```python
            reg_pred = reg_out_v.cpu().numpy().flatten()
```

with:

```python
            reg_pred_log = reg_out_v.cpu().numpy().flatten()
            reg_pred = np.expm1(reg_pred_log).clip(min=0.0)
```

and compute MAE/RMSE/R2 against `y_reg_val_raw`.

Before saving artifacts, replace best-fold persistence with final fit:

```python
    temporal_feature_names, static_feature_names = _feature_names(
        dataset,
        fallback_temporal=[f"temporal_{i}" for i in range(dataset.X_seq.shape[2])],
        fallback_static_count=dataset.X_static.shape[1],
    )
    final_state_dict, final_scaler_seq, final_scaler_static = _fit_surrogate_final_model(
        dataset=dataset,
        model_cls=model_cls,
        use_temporal=use_temporal,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    torch.save(final_state_dict, artifacts_dir / f"{prefix}_weights.pt")
    joblib.dump(final_scaler_seq, artifacts_dir / f"{prefix}_scaler_seq.joblib")
    joblib.dump(final_scaler_static, artifacts_dir / f"{prefix}_scaler_static.joblib")
    _write_surrogate_manifest(
        artifacts_dir,
        prefix=prefix,
        model_type="cnn" if model_cls is SWMMSurrogateCNN else "lstm",
        seed=seed,
        dataset=dataset,
        temporal_feature_names=temporal_feature_names,
        static_feature_names=static_feature_names,
        use_temporal=use_temporal,
    )
```

Return:

```python
    return {
        "n_folds": actual_folds,
        "folds": fold_results,
        "best_fold": best_fold_idx,
        "final_model_trained_on_all_groups": True,
    }
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest tests/ml/temporal/test_train_surrogate.py -v
```

Expected: pass.

Commit:

```bash
git add swmm_resilience/ml/temporal/train_surrogate.py tests/ml/temporal/test_train_surrogate.py
git commit -m "fix: persist final surrogate models with manifests"
```

---

### Task 5: Inference Manifest Guardrails And Task-Specific Output Schemas

**Files:**
- Modify: `swmm_resilience/ml/temporal/predict.py`
- Modify: `tests/ml/temporal/test_surrogate_predict.py`
- Create: `tests/ml/temporal/test_predict_from_parquet_task_schema.py`

- [ ] **Step 1: Write failing tests for extrapolation warning and missing manifest**

Append to `tests/ml/temporal/test_surrogate_predict.py`:

```python
import json
import warnings


def _write_manifest(artifacts_dir: Path, run_ids: list[str] | None = None) -> None:
    manifest = {
        "model_type": "cnn",
        "prefix": "surrogate_cnn",
        "seed": 42,
        "use_temporal": True,
        "trained_run_ids": run_ids or ["run_qx100"],
        "trained_rows": 4,
        "temporal_feature_names": ["total_inflow_lps", "lateral_inflow_lps"],
        "static_feature_names": [
            "full_depth_m",
            "in_degree",
            "out_degree",
            "upstream_diam_avg_m",
            "downstream_diam_avg_m",
            "upstream_capacity_lps",
            "downstream_capacity_lps",
        ],
        "regression_target": "peak_flooding_lps",
        "regression_target_transform": "log1p",
        "min_multiplier": 1.0,
        "max_multiplier": 2.0,
    }
    (artifacts_dir / "surrogate_cnn_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_surrogate_prediction_warns_outside_manifest_multiplier_range(tmp_path):
    db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
    _write_manifest(artifacts_dir)

    with pytest.warns(UserWarning, match="outside the training multiplier range"):
        predict_surrogate_from_multiplier(
            multiplier=5.0,
            db_path=db_path,
            artifacts_dir=artifacts_dir,
        )


def test_surrogate_prediction_rejects_manifest_run_id_mismatch(tmp_path):
    db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
    _write_manifest(artifacts_dir, run_ids=["missing_run"])

    with pytest.raises(ValueError, match="manifest run IDs"):
        predict_surrogate_from_multiplier(
            multiplier=1.5,
            db_path=db_path,
            artifacts_dir=artifacts_dir,
        )
```

- [ ] **Step 2: Write failing tests for `predict_from_parquet()` task-specific schemas**

Create `tests/ml/temporal/test_predict_from_parquet_task_schema.py`:

```python
import pytest

from swmm_resilience.ml.temporal.predict import _output_columns_for_temporal_task


def test_classification_schema_uses_probability_columns():
    assert _output_columns_for_temporal_task("classification") == [
        "node_id",
        "max_flood_prob",
        "mean_flood_prob",
        "windows_total",
        "windows_flood_predicted",
        "actual_flooded",
    ]


def test_regression_schema_uses_peak_lps_columns():
    assert _output_columns_for_temporal_task("regression") == [
        "node_id",
        "max_peak_flooding_lps_pred",
        "mean_peak_flooding_lps_pred",
        "windows_total",
        "actual_peak_flooding_lps",
    ]


def test_unknown_temporal_task_is_rejected():
    with pytest.raises(ValueError, match="classification.*regression"):
        _output_columns_for_temporal_task("ranking")
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
pytest tests/ml/temporal/test_surrogate_predict.py tests/ml/temporal/test_predict_from_parquet_task_schema.py -v
```

Expected: new tests fail because manifest validation and `_output_columns_for_temporal_task()` do not exist.

- [ ] **Step 4: Add manifest load and validation helpers**

In `swmm_resilience/ml/temporal/predict.py`, add imports:

```python
import json
```

Add helpers near `_SURROGATE_PREFIXES`:

```python
def _surrogate_manifest_path(artifacts_dir: Path, prefix: str) -> Path:
    return artifacts_dir / f"{prefix}_manifest.json"


def _load_surrogate_manifest(artifacts_dir: Path, prefix: str) -> dict:
    path = _surrogate_manifest_path(artifacts_dir, prefix)
    if not path.exists():
        warnings.warn(
            f"No surrogate manifest found at {path}; inference will run without artifact contract validation.",
            stacklevel=2,
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _db_run_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT run_id FROM runs").fetchall()
    finally:
        conn.close()
    return {str(row[0]) for row in rows}


def _validate_surrogate_manifest(manifest: dict, *, db_path: Path, multiplier: float) -> None:
    if not manifest:
        return
    manifest_run_ids = {str(run_id) for run_id in manifest.get("trained_run_ids", [])}
    if manifest_run_ids and not manifest_run_ids.issubset(_db_run_ids(db_path)):
        missing = sorted(manifest_run_ids - _db_run_ids(db_path))
        raise ValueError(
            "The surrogate manifest run IDs do not match the current database. "
            f"Missing run IDs: {', '.join(missing[:5])}"
        )

    min_multiplier = manifest.get("min_multiplier")
    max_multiplier = manifest.get("max_multiplier")
    if min_multiplier is not None and max_multiplier is not None:
        if multiplier < float(min_multiplier) or multiplier > float(max_multiplier):
            warnings.warn(
                f"Requested multiplier Qx{multiplier:.2f} is outside the training multiplier range "
                f"Qx{float(min_multiplier):.2f}-Qx{float(max_multiplier):.2f}.",
                stacklevel=2,
            )


def _output_columns_for_temporal_task(task: str) -> list[str]:
    if task == "classification":
        return [
            "node_id",
            "max_flood_prob",
            "mean_flood_prob",
            "windows_total",
            "windows_flood_predicted",
            "actual_flooded",
        ]
    if task == "regression":
        return [
            "node_id",
            "max_peak_flooding_lps_pred",
            "mean_peak_flooding_lps_pred",
            "windows_total",
            "actual_peak_flooding_lps",
        ]
    raise ValueError("task must be 'classification' or 'regression'.")
```

- [ ] **Step 5: Wire manifest validation and log-target inverse transform**

In `predict_surrogate_from_multiplier()`, after `prefix = _SURROGATE_PREFIXES[model_type]`, add:

```python
    manifest = _load_surrogate_manifest(artifacts_dir, prefix)
    _validate_surrogate_manifest(manifest, db_path=Path(db_path), multiplier=float(multiplier))
```

When reading `reg_out`, replace:

```python
            peak_lps = float(reg_out.cpu().item())
```

with:

```python
            raw_peak = float(reg_out.cpu().item())
            if manifest.get("regression_target_transform") == "log1p":
                peak_lps = float(np.expm1(raw_peak))
            else:
                peak_lps = raw_peak
```

In `predict_from_parquet()`, validate `task` at the start:

```python
    _output_columns_for_temporal_task(task)
```

For regression records, use `max_peak_flooding_lps_pred` and `mean_peak_flooding_lps_pred` instead of probability columns. Keep classification output unchanged.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest tests/ml/temporal/test_surrogate_predict.py tests/ml/temporal/test_predict_from_parquet_task_schema.py -v
```

Expected: pass.

Commit:

```bash
git add swmm_resilience/ml/temporal/predict.py tests/ml/temporal/test_surrogate_predict.py tests/ml/temporal/test_predict_from_parquet_task_schema.py
git commit -m "fix: validate temporal inference artifacts"
```

---

### Task 6: Full Stabilization Regression Suite And Audit Traceability

**Files:**
- Modify: `docs/model_hydraulic_prediction_audit_2026-05-31.md`

- [ ] **Step 1: Add implementation status section to the audit**

Append this section to `docs/model_hydraulic_prediction_audit_2026-05-31.md`:

```markdown
## Safe stabilization implementation status

This audit is being addressed by `docs/superpowers/plans/2026-05-31-safe-stabilization-pass.md`.

Covered in the first pass:

- Findings 1 and 11: temporal feature contracts and task-specific temporal inference schemas.
- Finding 2: selected-node timeseries scaling isolation.
- Finding 3: rejection of ambiguous `delta_inflows_lps`.
- Findings 4, 5, 6, 7, 13, and 17: deterministic surrogate training, final full-data fit, manifests, target transform, and inference guardrails.
- Finding 8: default `.inp` path correction.

Deferred:

- Findings 9, 14, 15, 16, 18, 19, and 20 remain planned for later model-quality and documentation passes.
```

- [ ] **Step 2: Run targeted stabilization suite**

Run:

```bash
pytest \
  tests/test_config_defaults.py \
  tests/test_run_experiment_input_semantics.py \
  tests/simulation/test_partial_timeseries_scaling.py \
  tests/ml/temporal/test_window_builder.py \
  tests/ml/temporal/test_surrogate_dataset.py \
  tests/ml/temporal/test_train_surrogate.py \
  tests/ml/temporal/test_surrogate_predict.py \
  tests/ml/temporal/test_predict_from_parquet_task_schema.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Run broader ML temporal suite**

Run:

```bash
pytest tests/ml/temporal -v
```

Expected: all temporal tests pass. If tests fail because existing tests assume old 6-channel surrogate inputs, update those tests to use `len(PRE_SWMM_TEMPORAL_COLS)` for surrogate paths and `len(TEMPORAL_COLS)` for post-SWMM temporal-window paths.

- [ ] **Step 4: Commit audit traceability**

Commit:

```bash
git add docs/model_hydraulic_prediction_audit_2026-05-31.md
git commit -m "docs: track safe stabilization coverage"
```

---

## Self-Review

Spec coverage:

- Critical findings 1, 2, 3 are covered by Tasks 1, 2, and 3.
- High findings 4, 5, 6, 7, 8, and 10 are covered by Tasks 1, 4, and 5. Padding masks are not fully redesigned; the first pass reduces deployable surrogate channels to fixed pre-SWMM inputs and records feature contracts.
- Medium findings 11, 13, and 17 are covered by Tasks 4 and 5.
- Findings 9, 14, 15, 16, 18, 19, and 20 are explicitly deferred.

Placeholder scan:

- No step uses placeholder markers, unspecified validation, or "write tests" without concrete test code.
- Every task has exact files, commands, expected outcomes, and commit boundaries.

Type consistency:

- `PRE_SWMM_TEMPORAL_COLS`, `SWMM_OUTPUT_TEMPORAL_COLS`, and `SURROGATE_TEMPORAL_COLS` are defined in `dataset.py` and imported by tests.
- `surrogate_cnn_manifest.json` naming matches `_surrogate_manifest_path()`.
- `regression_target_transform == "log1p"` is written by training and read by inference.
