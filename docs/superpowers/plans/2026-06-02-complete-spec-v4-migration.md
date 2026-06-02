# Complete Spec V4 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the current spec-style SWMM pipeline so `config.yaml -> simulations -> dataset_final.csv -> trained models -> metrics -> maps -> prediction` works reliably.

**Architecture:** Keep the new spec-aligned modules already present under `swmm_resilience/` and finish them incrementally with focused tests. Preserve the existing `swmm_api_io.py` utility layer, because it already contains the low-risk SWMM file I/O primitives that the new modules depend on.

**Tech Stack:** Python 3.10+, pyswmm, swmm-api, networkx, pandas, numpy, xgboost, scikit-learn, matplotlib, pyyaml, joblib, pytest

---

## Current State

The repository is already mid-migration toward `spec_tecnico_desarrollo.md` v4.0. These spec modules exist and compile:

- `config.yaml`
- `main.py`
- `swmm_resilience/config.py`
- `swmm_resilience/simulation/runner.py`
- `swmm_resilience/simulation/batch.py`
- `swmm_resilience/extraction/static_features.py`
- `swmm_resilience/extraction/topology.py`
- `swmm_resilience/extraction/dynamic_features.py`
- `swmm_resilience/extraction/labels.py`
- `swmm_resilience/dataset/assembler.py`
- `swmm_resilience/dataset/validator.py`
- `swmm_resilience/ml/trainer.py`
- `swmm_resilience/ml/evaluator.py`
- `swmm_resilience/ml/feature_importance.py`
- `swmm_resilience/ml/predict.py`
- `swmm_resilience/visualization/flood_map.py`

The old desktop/SQLite/temporal architecture is currently deleted in the worktree. This plan treats that as intentional for the spec v4 migration and focuses on making the new architecture production-worthy.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `pytest` and align minimum dependencies with the spec. |
| `tests/conftest.py` | Create | Provide shared lightweight config and synthetic ML dataset fixtures. |
| `tests/test_config.py` | Create | Validate YAML loading, path resolution, factor generation, and config errors. |
| `tests/test_dynamic_dataset_validation.py` | Create | Test dynamic feature math, dataset assembly, and validator failures. |
| `tests/test_labels.py` | Create | Test `.rpt` label behavior through monkeypatched `read_node_flooding_summary`. |
| `tests/test_ml_trainer_predict.py` | Create | Test feature columns, model factory behavior, hash validation, and prediction routing with fake models. |
| `tests/test_evaluator.py` | Create | Test LOSO/GroupKFold behavior and assert non-empty finite metrics, including `log_nse`. |
| `tests/test_visualization.py` | Create | Test flood-map output with monkeypatched SWMM input. |
| `swmm_resilience/config.py` | Modify | Harden config validation and output-directory creation. |
| `swmm_resilience/simulation/runner.py` | Modify | Ensure `.rpt` exists after PySWMM and preserve failure context. |
| `swmm_resilience/dataset/assembler.py` | Modify | Fill missing labels after joins and enforce required columns. |
| `swmm_resilience/ml/trainer.py` | Modify | Prevent empty flooded-regressor training, train regressor target with `log1p`, and support optional scaler cleanly. |
| `swmm_resilience/ml/evaluator.py` | Modify | Use DataFrames instead of raw arrays, train folds with `log1p`, invert with `expm1`, add `log_nse`, and make edge cases explicit. |
| `swmm_resilience/ml/predict.py` | Modify | Keep DataFrame feature names and invert regressor predictions with `expm1`. |
| `swmm_resilience/visualization/flood_map.py` | Modify | Return output path and handle empty/zero-volume inputs deterministically. |
| `main.py` | Modify | Add `--skip-simulation`, improve mode semantics, and fail fast when expected files are missing. |

---

## Task 1: Add Test Harness

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_config.py`

- [ ] **Step 1: Add pytest to requirements**

Update `requirements.txt` so it contains this line:

```text
pytest>=8.0
```

- [ ] **Step 2: Write config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from swmm_resilience.config import load_config


def write_config(tmp_path: Path, inp_name: str = "network.inp", factor_step: float = 0.2):
    inp = tmp_path / inp_name
    inp.write_text("[TITLE]\n;; test\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
network:
  inp_path: "{inp_name}"
  name: "Test Network"
simulation:
  factor_min: 0.2
  factor_max: 0.6
  factor_step: {factor_step}
dataset:
  output_path: "data/training/dataset_final.csv"
  flood_threshold_m3: 0.0
ml:
  classifier:
    algorithm: "xgboost"
    n_estimators: 10
    max_depth: 3
    learning_rate: 0.1
    subsample: 1.0
    scale_pos_weight: "auto"
  regressor:
    algorithm: "xgboost"
    n_estimators: 10
    max_depth: 3
    learning_rate: 0.1
    subsample: 1.0
  use_scaler: false
evaluation:
  methods: ["LOSO", "GroupKFold5"]
  stratify_by_factor: true
visualization:
  factors_to_plot: [0.2, 0.4]
  colormap: "RdYlBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
""",
        encoding="utf-8",
    )
    return cfg


def test_load_config_resolves_paths_and_factors(tmp_path):
    cfg = load_config(write_config(tmp_path))

    assert cfg.network.inp_path == tmp_path / "network.inp"
    assert cfg.dataset.output_path == tmp_path / "data" / "training" / "dataset_final.csv"
    assert cfg.visualization.output_path == tmp_path / "outputs" / "maps"
    assert cfg.factors() == [0.2, 0.4, 0.6]


def test_load_config_rejects_missing_inp(tmp_path):
    cfg_path = write_config(tmp_path, inp_name="missing.inp")
    (tmp_path / "missing.inp").unlink()

    with pytest.raises(FileNotFoundError, match="archivo .inp no existe"):
        load_config(cfg_path)


def test_load_config_rejects_non_positive_step(tmp_path):
    cfg_path = write_config(tmp_path, factor_step=0)

    with pytest.raises(ValueError, match="factor_step debe ser mayor que 0"):
        load_config(cfg_path)
```

- [ ] **Step 3: Run config tests**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/test_config.py
git commit -m "test: add config loading coverage"
```

---

## Task 2: Harden Config Validation

**Files:**
- Modify: `swmm_resilience/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add algorithm validation test**

Append to `tests/test_config.py`:

```python
def test_load_config_rejects_unknown_algorithm(tmp_path):
    cfg_path = write_config(tmp_path)
    text = cfg_path.read_text(encoding="utf-8")
    cfg_path.write_text(text.replace('algorithm: "xgboost"', 'algorithm: "svm"', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Algoritmo de clasificador no soportado"):
        load_config(cfg_path)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_config.py::test_load_config_rejects_unknown_algorithm -v
```

Expected: FAIL because `load_config` currently accepts unknown algorithms.

- [ ] **Step 3: Implement algorithm and evaluation validation**

Add helper functions near the top of `swmm_resilience/config.py`:

```python
SUPPORTED_MODEL_ALGORITHMS = {"xgboost", "random_forest"}
SUPPORTED_EVALUATION_METHODS = {"LOSO", "GroupKFold5"}


def _validate_algorithm(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MODEL_ALGORITHMS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_ALGORITHMS))
        raise ValueError(f"Algoritmo de {label} no soportado: {value}. Opciones: {allowed}")
    return normalized


def _validate_methods(methods: list) -> list[str]:
    normalized = [str(method).strip() for method in methods]
    invalid = [method for method in normalized if method not in SUPPORTED_EVALUATION_METHODS]
    if invalid:
        allowed = ", ".join(sorted(SUPPORTED_EVALUATION_METHODS))
        raise ValueError(f"Metodo de evaluacion no soportado: {', '.join(invalid)}. Opciones: {allowed}")
    return normalized
```

Then update the `load_config` return block so classifier/regressor algorithms and methods use those helpers:

```python
algorithm=_validate_algorithm(clf["algorithm"], "clasificador"),
```

```python
algorithm=_validate_algorithm(reg["algorithm"], "regresor"),
```

```python
methods=_validate_methods(ev["methods"]),
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/config.py tests/test_config.py
git commit -m "fix: validate config algorithms and evaluation methods"
```

---

## Task 3: Dataset Assembly and Validation Tests

**Files:**
- Create: `tests/test_dynamic_dataset_validation.py`
- Modify: `swmm_resilience/dataset/assembler.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/test_dynamic_dataset_validation.py`:

```python
import pandas as pd
import pytest

from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.dataset.validator import validate_dataset
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features


def static_topology_df():
    return pd.DataFrame(
        {
            "node_id": ["J1", "J2"],
            "elev_fondo": [10.0, 9.5],
            "prof_max": [1.5, 1.4],
            "n_tuberias_in": [0, 1],
            "n_tuberias_out": [1, 1],
            "diam_max_in": [None, 0.6],
            "diam_max_out": [0.6, 0.7],
            "pendiente_max_in": [None, 0.01],
            "pendiente_out": [0.01, 0.02],
            "base_inflow_lps": [2.0, 3.0],
            "dist_outfall_m": [100.0, 40.0],
            "n_nodos_aguas_arriba": [0, 1],
            "q_pico_acum_base": [2.0, 5.0],
            "upstream_capacity_lps": [None, 25.0],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
        }
    )


def test_dynamic_features_scale_peak_and_accumulated_flow():
    df = compute_dynamic_features(static_topology_df(), 1.5)

    assert df.to_dict("records") == [
        {"node_id": "J1", "factor_mult": 1.5, "q_pico_nodo": 3.0, "q_pico_acum_escalado": 3.0},
        {"node_id": "J2", "factor_mult": 1.5, "q_pico_nodo": 4.5, "q_pico_acum_escalado": 7.5},
    ]


def test_assemble_dataset_fills_missing_labels_as_zero(tmp_path):
    static_df = static_topology_df()
    dynamic_df = compute_dynamic_features(static_df, 1.0)
    labels_df = pd.DataFrame({"node_id": ["J2"], "vol_inundacion_m3": [12.0], "inunda": [1]})

    dataset = assemble_dataset(static_df, [(1.0, dynamic_df, labels_df)], tmp_path / "dataset.csv")

    row_j1 = dataset[dataset["node_id"] == "J1"].iloc[0]
    assert row_j1["vol_inundacion_m3"] == 0.0
    assert row_j1["inunda"] == 0
    assert (tmp_path / "dataset.csv").exists()


def test_validate_dataset_rejects_wrong_row_count():
    df = pd.DataFrame({"inunda": [0], "vol_inundacion_m3": [0.0]})

    with pytest.raises(ValueError, match="Filas esperadas"):
        validate_dataset(df, n_nodes=2, n_factors=1)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_dynamic_dataset_validation.py::test_assemble_dataset_fills_missing_labels_as_zero -v
```

Expected: FAIL because `assemble_dataset` leaves missing labels as NaN.

- [ ] **Step 3: Fill missing labels in assembler**

In `swmm_resilience/dataset/assembler.py`, after the second merge, add:

```python
        if "vol_inundacion_m3" not in merged.columns or "inunda" not in merged.columns:
            raise ValueError("labels_df debe incluir columnas vol_inundacion_m3 e inunda")
        merged["vol_inundacion_m3"] = merged["vol_inundacion_m3"].fillna(0.0)
        merged["inunda"] = merged["inunda"].fillna(0).astype(int)
```

- [ ] **Step 4: Run dataset tests**

Run:

```bash
python -m pytest tests/test_dynamic_dataset_validation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/dataset/assembler.py tests/test_dynamic_dataset_validation.py
git commit -m "fix: make dataset assembly label joins deterministic"
```

---

## Task 4: Label Extraction Tests

**Files:**
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write label tests**

Create `tests/test_labels.py`:

```python
import pandas as pd

from swmm_resilience.extraction import labels


def test_extract_labels_marks_missing_rpt_nodes_as_not_flooded(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J2"], "flooding_volume_m3": [15.0]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    df = labels.extract_labels(tmp_path / "run.rpt", ["J1", "J2"], threshold_m3=0.0)

    assert df.to_dict("records") == [
        {"node_id": "J1", "vol_inundacion_m3": 0.0, "inunda": 0},
        {"node_id": "J2", "vol_inundacion_m3": 15.0, "inunda": 1},
    ]


def test_extract_labels_applies_threshold(monkeypatch, tmp_path):
    def fake_read_node_flooding_summary(path):
        return pd.DataFrame({"node_id": ["J1"], "flooding_volume_m3": [0.5]})

    monkeypatch.setattr(labels, "read_node_flooding_summary", fake_read_node_flooding_summary)

    df = labels.extract_labels(tmp_path / "run.rpt", ["J1"], threshold_m3=1.0)

    assert df.iloc[0]["vol_inundacion_m3"] == 0.5
    assert df.iloc[0]["inunda"] == 0
```

- [ ] **Step 2: Run label tests**

Run:

```bash
python -m pytest tests/test_labels.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_labels.py
git commit -m "test: cover rpt label extraction"
```

---

## Task 5: Trainer Safety and Optional Scaler

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_ml_trainer_predict.py`
- Modify: `swmm_resilience/ml/trainer.py`

- [ ] **Step 1: Write shared ML test fixtures**

Create `tests/conftest.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.ml.trainer import FEATURE_COLS


@pytest.fixture
def tiny_config_factory(tmp_path: Path):
    def _factory(algorithm: str = "random_forest", use_scaler: bool = False):
        inp = tmp_path / "network.inp"
        inp.write_text("network", encoding="utf-8")
        return SimpleNamespace(
            network=SimpleNamespace(inp_path=inp),
            ml=SimpleNamespace(
                classifier=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                    scale_pos_weight="auto",
                ),
                regressor=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                ),
                use_scaler=use_scaler,
            ),
        )

    return _factory


@pytest.fixture
def trainer_training_df():
    rows = []
    for i in range(8):
        row = {col: float(i + 1) for col in FEATURE_COLS}
        row["inunda"] = 1 if i >= 4 else 0
        row["vol_inundacion_m3"] = float(i * 10) if i >= 4 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write trainer tests**

Create `tests/test_ml_trainer_predict.py`:

```python
import pytest

from swmm_resilience.ml import trainer


def test_train_models_rejects_dataset_without_flooded_rows(tmp_path, tiny_config_factory, trainer_training_df):
    df = trainer_training_df.copy()
    df["inunda"] = 0
    df["vol_inundacion_m3"] = 0.0

    with pytest.raises(ValueError, match="No hay filas inundadas"):
        trainer.train_models(df, tiny_config_factory(), tmp_path / "models")


def test_train_models_writes_artifacts(tmp_path, tiny_config_factory, trainer_training_df):
    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    assert (tmp_path / "models" / "classifier.joblib").exists()
    assert (tmp_path / "models" / "regressor.joblib").exists()
    assert (tmp_path / "models" / "training_inp_hash.txt").exists()
```

- [ ] **Step 3: Run failing safety test**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py::test_train_models_rejects_dataset_without_flooded_rows -v
```

Expected: FAIL because current trainer fits a regressor on an empty DataFrame.

- [ ] **Step 4: Add explicit flooded-row guard**

In `swmm_resilience/ml/trainer.py`, before `reg.fit(...)`, replace:

```python
    df_flooded = df[df["inunda"] == 1]
    reg = make_regressor(config)
    reg.fit(df_flooded[FEATURE_COLS], df_flooded["vol_inundacion_m3"])
```

with:

```python
    df_flooded = df[df["inunda"] == 1]
    if df_flooded.empty:
        raise ValueError("No hay filas inundadas para entrenar el regresor.")
    reg = make_regressor(config)
    reg.fit(df_flooded[FEATURE_COLS], df_flooded["vol_inundacion_m3"])
```

- [ ] **Step 5: Run trainer tests**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/trainer.py tests/conftest.py tests/test_ml_trainer_predict.py
git commit -m "fix: validate flooded rows before regressor training"
```

---

## Task 6: Prediction DataFrame Compatibility

**Files:**
- Modify: `tests/test_ml_trainer_predict.py`
- Modify: `swmm_resilience/ml/predict.py`

- [ ] **Step 1: Add prediction test with fake models**

Append to `tests/test_ml_trainer_predict.py`:

```python
import joblib
import pandas as pd

from swmm_resilience.ml import predict


class FakeClassifier:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return [0, 1]


class FakeRegressor:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return [33.0]


def test_predict_network_uses_dataframe_features(monkeypatch, tmp_path, tiny_config_factory):
    cfg = tiny_config_factory()
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(FakeClassifier(), models / "classifier.joblib")
    joblib.dump(FakeRegressor(), models / "regressor.joblib")
    (models / "training_inp_hash.txt").write_text(predict._md5(cfg.network.inp_path), encoding="utf-8")

    static = pd.DataFrame(
        {
            "node_id": ["J1", "J2"],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
            **{col: [1.0, 2.0] for col in trainer.FEATURE_COLS if col not in {"factor_mult", "q_pico_nodo", "q_pico_acum_escalado"}},
        }
    )

    def fake_extract_static_features(path):
        return static[["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max"]].copy()

    def fake_compute_topology_features(static_df, inp_path):
        return static.copy()

    def fake_compute_dynamic_features(full_df, factor):
        return pd.DataFrame(
            {
                "node_id": ["J1", "J2"],
                "factor_mult": [factor, factor],
                "q_pico_nodo": [factor, factor * 2],
                "q_pico_acum_escalado": [factor, factor * 2],
            }
        )

    monkeypatch.setattr(predict, "extract_static_features", fake_extract_static_features)
    monkeypatch.setattr(predict, "compute_topology_features", fake_compute_topology_features)
    monkeypatch.setattr(predict, "compute_dynamic_features", fake_compute_dynamic_features)

    result = predict.predict_network(1.0, cfg, models)

    assert result.loc[result["node_id"] == "J1", "inunda_pred"].iloc[0] == 0
    assert result.loc[result["node_id"] == "J1", "vol_pred_m3"].iloc[0] == 0.0
    assert result.loc[result["node_id"] == "J2", "inunda_pred"].iloc[0] == 1
    assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(33.0)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py::test_predict_network_uses_dataframe_features -v
```

Expected: FAIL because `predict_network` passes `X.values[flood_mask]` to the regressor.

- [ ] **Step 3: Keep regressor input as DataFrame**

In `swmm_resilience/ml/predict.py`, replace:

```python
        vol_pred[flood_mask] = reg.predict(X.values[flood_mask])
```

with:

```python
        vol_pred[flood_mask] = reg.predict(X.loc[flood_mask])
```

- [ ] **Step 4: Run prediction tests**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/predict.py tests/test_ml_trainer_predict.py
git commit -m "fix: preserve feature names during prediction"
```

---

## Task 7: Evaluator Edge Cases

**Files:**
- Create: `tests/test_evaluator.py`
- Modify: `swmm_resilience/ml/evaluator.py`

- [ ] **Step 1: Write evaluator tests**

Create `tests/test_evaluator.py`:

```python
import json
import math

import pandas as pd

from swmm_resilience.ml.evaluator import evaluate_models
from swmm_resilience.ml.trainer import FEATURE_COLS


def evaluation_df():
    rows = []
    for factor in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for node_idx in range(4):
            row = {col: float(node_idx + 1) for col in FEATURE_COLS}
            row["factor_mult"] = factor
            row["q_pico_nodo"] = factor * (node_idx + 1)
            row["q_pico_acum_escalado"] = factor * (node_idx + 2)
            row["inunda"] = 1 if factor >= 2.0 and node_idx >= 2 else 0
            row["vol_inundacion_m3"] = factor * 10.0 if row["inunda"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_evaluate_models_writes_expected_json_files(tmp_path, tiny_config_factory):
    cfg = tiny_config_factory(algorithm="random_forest")
    cfg.evaluation = type("Eval", (), {"methods": ["LOSO", "GroupKFold5"], "stratify_by_factor": True})()

    results = evaluate_models(evaluation_df(), cfg, tmp_path / "metrics")

    assert "LOSO" in results
    assert "GroupKFold5" in results
    assert (tmp_path / "metrics" / "metrics_classifier.json").exists()
    assert (tmp_path / "metrics" / "metrics_regressor.json").exists()
    assert (tmp_path / "metrics" / "metrics_endtoend.json").exists()
    assert (tmp_path / "metrics" / "metrics_by_factor.json").exists()

    reg_metrics = json.loads((tmp_path / "metrics" / "metrics_regressor.json").read_text(encoding="utf-8"))
    assert "nse" in reg_metrics
    assert math.isfinite(reg_metrics["nse"])
```

- [ ] **Step 2: Run evaluator tests**

Run:

```bash
python -m pytest tests/test_evaluator.py -v
```

Expected: PASS or FAIL with a concrete edge case. If it fails because a fold has only one class in training, continue with Step 3.

- [ ] **Step 3: Add fold guard for one-class classifier training**

In `swmm_resilience/ml/evaluator.py`, before `clf.fit(X_tr, yc_tr)`, add:

```python
        if len(np.unique(yc_tr)) < 2:
            continue
```

This skips folds where the classifier cannot be trained because the training side has only one class.

- [ ] **Step 4: Run evaluator tests**

Run:

```bash
python -m pytest tests/test_evaluator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/evaluator.py tests/test_evaluator.py
git commit -m "test: cover grouped model evaluation outputs"
```

---

## Task 8: Simulation Runner Failure Checks

**Files:**
- Modify: `swmm_resilience/simulation/runner.py`

- [ ] **Step 1: Add `.rpt` existence check**

In `swmm_resilience/simulation/runner.py`, replace the final return block:

```python
    with suppress(OSError):
        tmp_inp.unlink()

    return tmp_inp.with_suffix(".rpt")
```

with:

```python
    rpt_path = tmp_inp.with_suffix(".rpt")

    with suppress(OSError):
        tmp_inp.unlink()

    if not rpt_path.exists():
        raise FileNotFoundError(f"SWMM no genero el archivo .rpt esperado: {rpt_path}")

    return rpt_path
```

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m compileall swmm_resilience/simulation/runner.py
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add swmm_resilience/simulation/runner.py
git commit -m "fix: fail when swmm rpt output is missing"
```

---

## Task 9: Visualization Test

**Files:**
- Create: `tests/test_visualization.py`
- Modify: `swmm_resilience/visualization/flood_map.py`

- [ ] **Step 1: Write flood-map test**

Create `tests/test_visualization.py`:

```python
from types import SimpleNamespace

import pandas as pd

from swmm_resilience.visualization import flood_map


def test_generate_flood_map_writes_png(monkeypatch, tmp_path):
    fake_inp = {
        "COORDINATES": {
            "J1": SimpleNamespace(x=0.0, y=0.0),
            "J2": SimpleNamespace(x=10.0, y=0.0),
        },
        "CONDUITS": {
            "C1": SimpleNamespace(from_node="J1", to_node="J2"),
        },
    }
    monkeypatch.setattr(flood_map, "load_inp", lambda path: fake_inp)

    output = flood_map.generate_flood_map(
        tmp_path / "network.inp",
        pd.DataFrame({"node_id": ["J1", "J2"], "vol_inundacion_m3": [0.0, 4.0]}),
        1.0,
        tmp_path / "map.png",
        "Test Network",
    )

    assert output == tmp_path / "map.png"
    assert (tmp_path / "map.png").exists()
    assert (tmp_path / "map.png").stat().st_size > 0
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python -m pytest tests/test_visualization.py -v
```

Expected: FAIL because `generate_flood_map` currently returns `None`.

- [ ] **Step 3: Return output path**

At the end of `swmm_resilience/visualization/flood_map.py`, after the print, add:

```python
    return output_path
```

- [ ] **Step 4: Run visualization test**

Run:

```bash
python -m pytest tests/test_visualization.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/visualization/flood_map.py tests/test_visualization.py
git commit -m "test: cover flood map generation"
```

---

## Task 10: CLI Mode Semantics

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `--skip-simulation` argument**

In `main.py`, add this parser option after `--skip-extraction`:

```python
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Saltar SWMM y reusar .rpt existentes no soportado aun; usa --skip-extraction si ya tienes dataset")
```

- [ ] **Step 2: Fail fast for unsupported `--skip-simulation`**

After `args = parser.parse_args()`, add:

```python
    if args.skip_simulation and not args.skip_extraction:
        parser.error("--skip-simulation requiere --skip-extraction en esta version; el pipeline aun no indexa .rpt persistentes")
```

This makes the CLI honest: the spec mentions `--skip-simulation`, but the current pipeline does not persist/index `.rpt` paths across runs.

- [ ] **Step 3: Avoid extraction in `--only-ml` and `--skip-extraction`**

Replace:

```python
    print("\nExtrayendo features estaticas...")
    static_df = extract_static_features(config.network.inp_path)
    print(f"  {len(static_df)} nodos junction")

    print("Calculando features topologicas...")
    static_topo_df = compute_topology_features(static_df, config.network.inp_path)

    all_node_ids = static_topo_df["node_id"].tolist()
    n_nodes = len(all_node_ids)
    factors = config.factors()
    n_factors = len(factors)
```

with:

```python
    factors = config.factors()
    n_factors = len(factors)
    static_topo_df = None
    all_node_ids = []
    n_nodes = 0

    if not use_existing_dataset:
        print("\nExtrayendo features estaticas...")
        static_df = extract_static_features(config.network.inp_path)
        print(f"  {len(static_df)} nodos junction")

        print("Calculando features topologicas...")
        static_topo_df = compute_topology_features(static_df, config.network.inp_path)

        all_node_ids = static_topo_df["node_id"].tolist()
        n_nodes = len(all_node_ids)
        print(f"  Red: {n_nodes} nodos, {n_factors} factores ({factors[0]:.2f}-{factors[-1]:.2f})")
```

- [ ] **Step 4: Compile main**

Run:

```bash
python -m compileall main.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "fix: clarify cli partial-run modes"
```

---

## Task 11: Logarithmic Regressor Transform

**Files:**
- Modify: `tests/test_ml_trainer_predict.py`
- Modify: `tests/test_evaluator.py`
- Modify: `swmm_resilience/ml/trainer.py`
- Modify: `swmm_resilience/ml/evaluator.py`
- Modify: `swmm_resilience/ml/predict.py`

- [ ] **Step 1: Add trainer test for log-space regressor target**

Append to `tests/test_ml_trainer_predict.py`:

```python
import numpy as np


class RecordingRegressor:
    def __init__(self):
        self.fit_y = None

    def fit(self, X, y):
        self.fit_y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def test_train_models_fits_regressor_in_log_space(monkeypatch, tmp_path, tiny_config_factory, trainer_training_df):
    recorded = RecordingRegressor()
    monkeypatch.setattr(trainer, "make_regressor", lambda config: recorded)

    trainer.train_models(trainer_training_df, tiny_config_factory(), tmp_path / "models")

    flooded = trainer_training_df[trainer_training_df["inunda"] == 1]
    np.testing.assert_allclose(recorded.fit_y, np.log1p(flooded["vol_inundacion_m3"].to_numpy()))
```

- [ ] **Step 2: Add evaluator assertion for `log_nse`**

In `tests/test_evaluator.py`, extend `test_evaluate_models_writes_expected_json_files` after `assert math.isfinite(reg_metrics["nse"])`:

```python
    assert "log_nse" in reg_metrics
    assert math.isfinite(reg_metrics["log_nse"])
    assert reg_metrics["log_nse"] > -10
```

- [ ] **Step 3: Update prediction fake regressor to return log-volume**

In `tests/test_ml_trainer_predict.py`, update `FakeRegressor.predict`:

```python
class FakeRegressor:
    def predict(self, X):
        assert list(X.columns) == trainer.FEATURE_COLS
        return np.log1p([33.0])
```

The existing prediction assertion remains in m3 and uses approximate float comparison:

```python
assert result.loc[result["node_id"] == "J2", "vol_pred_m3"].iloc[0] == pytest.approx(33.0)
```

- [ ] **Step 4: Run log-transform tests and verify failure**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py::test_train_models_fits_regressor_in_log_space tests/test_ml_trainer_predict.py::test_predict_network_uses_dataframe_features tests/test_evaluator.py::test_evaluate_models_writes_expected_json_files -v
```

Expected: FAIL because the regressor still trains/predicts directly in m3 and evaluator does not emit `log_nse`.

- [ ] **Step 5: Train final regressor on `log1p(target)`**

In `swmm_resilience/ml/trainer.py`, add this import:

```python
import numpy as np
```

Then replace:

```python
    reg.fit(df_flooded[FEATURE_COLS], df_flooded["vol_inundacion_m3"])
```

with:

```python
    reg.fit(df_flooded[FEATURE_COLS], np.log1p(df_flooded["vol_inundacion_m3"]))
```

- [ ] **Step 6: Invert evaluator regressor predictions and add `log_nse`**

In `swmm_resilience/ml/evaluator.py`, replace the fold regressor fit:

```python
            reg.fit(X_tr[flooded_tr], yr_tr[flooded_tr])
```

with:

```python
            reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))
```

Replace Level 2 prediction:

```python
            yr_pred_oracle = reg.predict(X_te[flooded_te])
```

with:

```python
            yr_pred_oracle = np.expm1(reg.predict(X_te[flooded_te]))
            yr_pred_oracle = np.clip(yr_pred_oracle, a_min=0.0, a_max=None)
```

Replace the Level 2 metrics block with:

```python
            reg_m.append({
                "nse": _nse(yr_true_oracle, yr_pred_oracle),
                "log_nse": _nse(np.log1p(yr_true_oracle), np.log1p(yr_pred_oracle)),
                "rmse": float(np.sqrt(mean_squared_error(yr_true_oracle, yr_pred_oracle))),
                "mae": float(mean_absolute_error(yr_true_oracle, yr_pred_oracle)),
                "r2": float(r2_score(yr_true_oracle, yr_pred_oracle)),
            })
```

Replace Level 3 prediction:

```python
            yr_pred_e2e[clf_flood_mask] = reg.predict(X_te[clf_flood_mask])
```

with:

```python
            yr_pred_e2e[clf_flood_mask] = np.expm1(reg.predict(X_te[clf_flood_mask]))
            yr_pred_e2e = np.clip(yr_pred_e2e, a_min=0.0, a_max=None)
```

- [ ] **Step 7: Invert prediction outputs with `expm1`**

In `swmm_resilience/ml/predict.py`, replace:

```python
        vol_pred[flood_mask] = reg.predict(X.loc[flood_mask])
```

with:

```python
        vol_pred[flood_mask] = np.expm1(reg.predict(X.loc[flood_mask]))
        vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)
```

- [ ] **Step 8: Run log-transform tests**

Run:

```bash
python -m pytest tests/test_ml_trainer_predict.py tests/test_evaluator.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add swmm_resilience/ml/trainer.py swmm_resilience/ml/evaluator.py swmm_resilience/ml/predict.py tests/test_ml_trainer_predict.py tests/test_evaluator.py
git commit -m "feat: train flooding-volume regressor in log space"
```

---

## Task 12: Full Test and Import Verification

**Files:**
- No code changes expected unless tests reveal failures.

- [ ] **Step 1: Run full unit tests**

Run:

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 2: Run compileall**

Run:

```bash
python -m compileall main.py swmm_resilience
```

Expected: PASS.

- [ ] **Step 3: Run import smoke test**

Run:

```bash
python -c "import main; import swmm_resilience.config; import swmm_resilience.simulation.runner; import swmm_resilience.extraction.static_features; import swmm_resilience.ml.trainer; print('imports_ok')"
```

Expected output:

```text
imports_ok
```

- [ ] **Step 4: Commit test stabilization changes**

If any fixes were needed:

```bash
git add main.py swmm_resilience tests requirements.txt
git commit -m "test: stabilize spec v4 pipeline"
```

If no fixes were needed, skip this commit.

---

## Task 13: Real Pipeline Smoke Run

**Files:**
- No code changes expected unless the real run exposes a defect.

- [ ] **Step 1: Run ML-only mode if dataset exists**

Run:

```bash
python main.py --only-ml
```

Expected:

- Reads `data/training/dataset_final.csv`
- Writes `outputs/models/classifier.joblib`
- Writes `outputs/models/regressor.joblib`
- Writes `outputs/models/training_inp_hash.txt`
- Writes JSON files in `outputs/metrics/`
- `outputs/metrics/metrics_regressor.json` contains finite `log_nse`

- [ ] **Step 1b: Verify log-transform success metric**

Run:

```bash
python -c "import json, math; m=json.load(open('outputs/metrics/metrics_regressor.json', encoding='utf-8')); assert 'log_nse' in m, m; assert math.isfinite(m['log_nse']), m; assert m['log_nse'] > 0, m; print('log_nse_ok', m['log_nse'])"
```

Expected output starts with:

```text
log_nse_ok
```

- [ ] **Step 2: Run maps-only mode**

Run:

```bash
python main.py --only-maps
```

Expected: creates one or more `outputs/maps/flood_map_factor_*.png` files for factors found in the dataset.

- [ ] **Step 3: Run prediction mode**

Run:

```bash
python main.py --predict --factor 3.5
```

Expected:

- Loads existing models.
- Validates the `.inp` hash.
- Prints predicted flooded nodes.
- Writes `outputs/maps/flood_map_pred_3.50.png`.

- [ ] **Step 4: Commit smoke-run fixes**

If any code changes were required:

```bash
git add main.py swmm_resilience tests
git commit -m "fix: pass real pipeline smoke runs"
```

If no code changes were required, skip this commit.

---

## Task 14: Documentation Sync

**Files:**
- Modify: `README.md`
- Modify: `QUICKSTART.md`
- Modify: `DOCUMENTACION_COMPLETA_PROYECTO.md`

- [ ] **Step 1: Update README commands**

Add this section to `README.md`:

```markdown
## Pipeline spec v4

El flujo principal se configura desde `config.yaml`.

```bash
python main.py
python main.py --skip-extraction
python main.py --only-ml
python main.py --only-maps
python main.py --predict --factor 3.5
```

Salidas principales:

- `data/training/dataset_final.csv`
- `outputs/models/classifier.joblib`
- `outputs/models/regressor.joblib`
- `outputs/metrics/*.json`
- `outputs/maps/*.png`
```

- [ ] **Step 2: Update QUICKSTART**

Add this compact run order to `QUICKSTART.md`:

```markdown
## Ejecucion recomendada

1. Revisa `config.yaml`.
2. Ejecuta `python main.py --only-ml` si ya existe `data/training/dataset_final.csv`.
3. Ejecuta `python main.py --only-maps` para regenerar mapas desde el CSV.
4. Ejecuta `python main.py --predict --factor 3.5` para inferencia sin SWMM.
5. Ejecuta `python main.py` solo cuando quieras recalcular simulaciones SWMM.
```

- [ ] **Step 3: Document removed desktop/SQLite architecture**

Add this note to `DOCUMENTACION_COMPLETA_PROYECTO.md` near the architecture section:

```markdown
Nota de migracion: la version spec v4 usa un pipeline directo basado en `config.yaml`, CSV, modelos joblib, metricas JSON y mapas PNG. La arquitectura anterior con SQLite, app Tkinter y modulos temporales fue reemplazada en esta rama de trabajo para alinear el codigo con `spec_tecnico_desarrollo.md`.
```

- [ ] **Step 4: Commit docs**

```bash
git add README.md QUICKSTART.md DOCUMENTACION_COMPLETA_PROYECTO.md
git commit -m "docs: document spec v4 pipeline usage"
```

---

## Self-Review

**Spec coverage:** This plan covers config loading, batch simulation, static/topological/dynamic features, label parsing, dataset assembly/validation, training, evaluation, feature importance, prediction, maps, CLI modes, requirements, docs, and the approved log-transform regressor spec (`log1p` training, `expm1` inference, `log_nse` metric). The only intentionally limited item is `--skip-simulation`: the current architecture does not persist a reusable `.rpt` index, so the plan makes that mode fail honestly unless paired with `--skip-extraction`.

**Placeholder scan:** No task uses placeholder instructions. Every test file contains executable code, and every code modification identifies exact replacement or insertion text.

**Type consistency:** The plan consistently uses `Config`, `FEATURE_COLS`, `factor_mult`, `inunda`, `vol_inundacion_m3`, `classifier.joblib`, `regressor.joblib`, `training_inp_hash.txt`, `tests/conftest.py` fixtures, and log-space regressor predictions inverted to m3 before user-visible outputs.
