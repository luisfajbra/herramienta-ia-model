# Feature Analysis Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--analyze-features` CLI flag that produces correlation heatmaps, an ablation study, and SHAP figures for the 17-feature XGBoost flood-prediction model.

**Architecture:** One new module `swmm_resilience/ml/feature_analysis.py` exposes three functions (`plot_correlation`, `run_ablation`, `plot_shap`). `main.py` wires a new `--analyze-features` flag that loads the dataset + trained models and calls all three, saving to `outputs/feature_analysis/`.

**Tech Stack:** XGBoost, scikit-learn, matplotlib, shap>=0.44.0, pandas, numpy.

## Global Constraints

- All figure-generating functions must call `plt.close()` after `fig.savefig()` — never leave figures open.
- Feature display names always via `feature_display_name()` from `swmm_resilience.visualization.labels` — never raw column names in axis labels or titles.
- `out_dir` always created with `Path(out_dir).mkdir(parents=True, exist_ok=True)`.
- `FEATURE_COLS` (17-element list) is defined in `swmm_resilience/ml/trainer.py` — import it, never redefine.
- `make_classifier` and `make_regressor` are also in `swmm_resilience/ml/trainer.py`.
- Models are sklearn `Pipeline([("imputer", SimpleImputer(...)), ("model", XGB...)])` — access raw model via `pipeline.named_steps["model"]`.
- DPI 150, `bbox_inches="tight"` for all saved figures.
- The ablation local CV loop must NOT modify the global `FEATURE_COLS` — operate on a local list.
- `shap` is imported lazily inside `plot_shap` (not at module level) to avoid import cost for users who never run SHAP.

---

## File Map

| Action | Path |
|--------|------|
| Create | `swmm_resilience/ml/feature_analysis.py` |
| Create | `tests/ml/test_feature_analysis.py` |
| Modify | `requirements.txt` — add `shap>=0.44.0` |
| Modify | `main.py` — add `--analyze-features` flag + handler |

---

## Task 1: Add shap dependency + implement `plot_correlation` + tests

**Files:**
- Create: `swmm_resilience/ml/feature_analysis.py`
- Create: `tests/ml/test_feature_analysis.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `plot_correlation(df: pd.DataFrame, out_dir: Path) -> None`

---

- [ ] **Step 1: Add `shap` to requirements.txt**

Open `requirements.txt` and append one line:

```
shap>=0.44.0
```

The file already ends without a trailing newline — add the line after `pytest>=8.0`.

- [ ] **Step 2: Write the failing test**

Create `tests/ml/test_feature_analysis.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.trainer import FEATURE_COLS


@pytest.fixture
def analysis_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    data = {col: rng.uniform(0.1, 10.0, n) for col in FEATURE_COLS}
    data["inunda"] = np.array([1 if i < 100 else 0 for i in range(n)])
    data["vol_inundacion_m3"] = np.where(
        data["inunda"] == 1, rng.uniform(0.5, 50.0, n), 0.0
    )
    data["factor_mult"] = np.tile([0.5, 1.0, 1.5, 2.0, 2.5], 40)
    return pd.DataFrame(data)


@pytest.fixture
def analysis_config(tmp_path: Path):
    inp = tmp_path / "network.inp"
    inp.write_text("network", encoding="utf-8")
    return SimpleNamespace(
        network=SimpleNamespace(inp_path=inp),
        ml=SimpleNamespace(
            classifier=SimpleNamespace(
                algorithm="random_forest",
                n_estimators=3,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
                scale_pos_weight="auto",
            ),
            regressor=SimpleNamespace(
                algorithm="random_forest",
                n_estimators=3,
                max_depth=2,
                learning_rate=0.1,
                subsample=1.0,
            ),
            use_scaler=False,
        ),
    )


def test_plot_correlation_creates_files(analysis_df, tmp_path):
    from swmm_resilience.ml.feature_analysis import plot_correlation

    plot_correlation(analysis_df, tmp_path)

    assert (tmp_path / "correlation_pearson.png").exists()
    assert (tmp_path / "correlation_spearman.png").exists()
    assert (tmp_path / "feature_target_correlation.png").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/luis/herramienta-ia-model
pytest tests/ml/test_feature_analysis.py::test_plot_correlation_creates_files -v
```

Expected: `FAILED` — `ModuleNotFoundError` or `ImportError` (module doesn't exist yet).

- [ ] **Step 4: Create `swmm_resilience/ml/feature_analysis.py` with `plot_correlation`**

```python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from .trainer import FEATURE_COLS, make_classifier, make_regressor
from ..visualization.labels import feature_display_name


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    """Save Pearson heatmap, Spearman heatmap, and feature-target bar chart."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df = df[FEATURE_COLS]
    display_names = [feature_display_name(f) for f in FEATURE_COLS]
    n = len(FEATURE_COLS)

    def _save_heatmap(mat: np.ndarray, title: str, filename: str) -> None:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        display_mat = np.where(mask, np.nan, mat)
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(display_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                if not mask[i, j] and abs(mat[i, j]) >= 0.3:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    pearson = feat_df.corr(method="pearson").to_numpy()
    spearman = feat_df.corr(method="spearman").to_numpy()

    _save_heatmap(pearson, "Pearson Correlation — Features", "correlation_pearson.png")
    _save_heatmap(spearman, "Spearman Correlation — Features", "correlation_spearman.png")

    rho_inunda = [df[f].corr(df["inunda"], method="spearman") for f in FEATURE_COLS]
    flooded = df[df["inunda"] == 1]
    rho_vol = [
        flooded[f].corr(flooded["vol_inundacion_m3"], method="spearman")
        for f in FEATURE_COLS
    ]

    order = np.argsort(np.abs(rho_inunda))
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(
        y - bar_h / 2,
        [rho_inunda[i] for i in order],
        bar_h,
        label="vs. Flooded (all rows)",
        color="steelblue",
    )
    ax.barh(
        y + bar_h / 2,
        [rho_vol[i] for i in order],
        bar_h,
        label="vs. Flood Volume (flooded only)",
        color="darkorange",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([display_names[i] for i in order], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_title("Feature–Target Spearman Correlation", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

(Leave the file open — `run_ablation` and `plot_shap` will be added in later tasks.)

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/ml/test_feature_analysis.py::test_plot_correlation_creates_files -v
```

Expected: `PASSED`.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
pytest --tb=short -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt swmm_resilience/ml/feature_analysis.py tests/ml/test_feature_analysis.py
git commit -m "feat: add feature_analysis module with plot_correlation + shap dep"
```

---

## Task 2: Implement `run_ablation` + tests

**Files:**
- Modify: `swmm_resilience/ml/feature_analysis.py` — add `run_ablation`
- Modify: `tests/ml/test_feature_analysis.py` — add ablation tests

**Interfaces:**
- Consumes: `make_classifier(config, spw)`, `make_regressor(config)` from `trainer.py`
- Produces: `run_ablation(df: pd.DataFrame, config, out_dir: Path) -> dict`
  - Returns `{"full": {"classifier": {"f1": float, "auc_roc": float}, "regressor_oracle": {"nse": float, "r2": float}}, "reduced": {...}}`

---

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_feature_analysis.py`:

```python
def test_run_ablation_returns_both_runs(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation

    result = run_ablation(analysis_df, analysis_config, tmp_path)

    assert "full" in result
    assert "reduced" in result
    for run_key in ("full", "reduced"):
        assert "classifier" in result[run_key]
        assert "regressor_oracle" in result[run_key]
        assert "f1" in result[run_key]["classifier"]
        assert "auc_roc" in result[run_key]["classifier"]
        assert "nse" in result[run_key]["regressor_oracle"]
        assert "r2" in result[run_key]["regressor_oracle"]


def test_run_ablation_writes_json(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation
    import json

    run_ablation(analysis_df, analysis_config, tmp_path)

    json_path = tmp_path / "ablation_results.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert "full" in data and "reduced" in data


def test_run_ablation_writes_chart(analysis_df, analysis_config, tmp_path):
    from swmm_resilience.ml.feature_analysis import run_ablation

    run_ablation(analysis_df, analysis_config, tmp_path)

    assert (tmp_path / "ablation_comparison.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ml/test_feature_analysis.py::test_run_ablation_returns_both_runs \
       tests/ml/test_feature_analysis.py::test_run_ablation_writes_json \
       tests/ml/test_feature_analysis.py::test_run_ablation_writes_chart -v
```

Expected: `FAILED` — `ImportError: cannot import name 'run_ablation'`.

- [ ] **Step 3: Add `run_ablation` to `swmm_resilience/ml/feature_analysis.py`**

Append this function after `plot_correlation` in the file:

```python
def run_ablation(df: pd.DataFrame, config, out_dir: Path) -> dict:
    """LOSO ablation: full 17 features vs. 15 features (no duracion/tiempo_al_pico).

    Returns the results dict and writes ablation_results.json + ablation_comparison.png.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced_cols = [f for f in FEATURE_COLS if f not in ("duracion_horas", "tiempo_al_pico_h")]

    def _loso_metrics(feature_cols: list) -> dict:
        X = df[feature_cols].values
        y_clf = df["inunda"].values
        y_reg = df["vol_inundacion_m3"].values
        groups = df["factor_mult"].values

        loso = LeaveOneGroupOut()
        f1_scores: list[float] = []
        auc_scores: list[float] = []
        reg_true_parts: list[np.ndarray] = []
        reg_pred_parts: list[np.ndarray] = []

        for train_idx, test_idx in loso.split(X, y_clf, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
            yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]

            n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            clf = make_classifier(config, spw)
            clf.fit(X_tr, yc_tr)

            reg = make_regressor(config)
            flooded_tr = yc_tr == 1
            if flooded_tr.sum() > 0:
                reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))

            yc_pred = clf.predict(X_te)
            yc_prob = clf.predict_proba(X_te)[:, 1]

            f1_scores.append(float(f1_score(yc_te, yc_pred, zero_division=0)))
            has_both = yc_te.sum() > 0 and (1 - yc_te).sum() > 0
            auc_scores.append(
                float(roc_auc_score(yc_te, yc_prob)) if has_both else float("nan")
            )

            flooded_te = yc_te == 1
            if flooded_te.sum() > 0:
                yr_pred = np.expm1(reg.predict(X_te[flooded_te]))
                yr_pred = np.clip(yr_pred, 0.0, None)
                reg_true_parts.append(yr_te[flooded_te])
                reg_pred_parts.append(yr_pred)

        f1_mean = float(np.nanmean(f1_scores))
        auc_mean = float(np.nanmean(auc_scores))

        if reg_true_parts:
            y_true_all = np.concatenate(reg_true_parts)
            y_pred_all = np.concatenate(reg_pred_parts)
            ss_res = float(np.sum((y_true_all - y_pred_all) ** 2))
            ss_tot = float(np.sum((y_true_all - np.mean(y_true_all)) ** 2))
            nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            r2 = float(r2_score(y_true_all, y_pred_all))
        else:
            nse = r2 = float("nan")

        return {
            "classifier": {"f1": f1_mean, "auc_roc": auc_mean},
            "regressor_oracle": {"nse": nse, "r2": r2},
        }

    results = {
        "full": _loso_metrics(FEATURE_COLS),
        "reduced": _loso_metrics(reduced_cols),
    }

    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    metric_labels = ["F1", "AUC-ROC", "NSE", "R²"]
    full_vals = [
        results["full"]["classifier"]["f1"],
        results["full"]["classifier"]["auc_roc"],
        results["full"]["regressor_oracle"]["nse"],
        results["full"]["regressor_oracle"]["r2"],
    ]
    reduced_vals = [
        results["reduced"]["classifier"]["f1"],
        results["reduced"]["classifier"]["auc_roc"],
        results["reduced"]["regressor_oracle"]["nse"],
        results["reduced"]["regressor_oracle"]["r2"],
    ]

    x = np.arange(len(metric_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_full = ax.bar(x - width / 2, full_vals, width, label="Full (17 features)", color="steelblue")
    bars_red = ax.bar(x + width / 2, reduced_vals, width, label="Reduced (15 features)", color="darkorange")
    for bar in list(bars_full) + list(bars_red):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Full vs. Reduced Feature Set (LOSO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ml/test_feature_analysis.py::test_run_ablation_returns_both_runs \
       tests/ml/test_feature_analysis.py::test_run_ablation_writes_json \
       tests/ml/test_feature_analysis.py::test_run_ablation_writes_chart -v
```

Expected: all three `PASSED`.

- [ ] **Step 5: Run full test suite**

```bash
pytest --tb=short -q
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/feature_analysis.py tests/ml/test_feature_analysis.py
git commit -m "feat: add run_ablation to feature_analysis (LOSO full vs reduced)"
```

---

## Task 3: Implement `plot_shap` + tests

**Files:**
- Modify: `swmm_resilience/ml/feature_analysis.py` — add `plot_shap`
- Modify: `tests/ml/test_feature_analysis.py` — add SHAP test

**Interfaces:**
- Consumes: `clf_pipeline` and `reg_pipeline` — sklearn `Pipeline` with `["imputer", "model"]` steps; model is `XGBClassifier` / `XGBRegressor`
- Produces: `plot_shap(clf_pipeline, reg_pipeline, df: pd.DataFrame, out_dir: Path) -> None`

---

- [ ] **Step 1: Install shap in the current environment**

```bash
pip install "shap>=0.44.0"
```

Expected: installs without error.

- [ ] **Step 2: Write the failing test**

Append to `tests/ml/test_feature_analysis.py`:

```python
def test_plot_shap_creates_summary_files(analysis_df, tmp_path):
    import joblib
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier, XGBRegressor
    from swmm_resilience.ml.feature_analysis import plot_shap
    from swmm_resilience.ml.trainer import FEATURE_COLS

    rng = np.random.default_rng(0)
    X = analysis_df[FEATURE_COLS].values
    y_clf = analysis_df["inunda"].values
    y_reg = np.log1p(analysis_df.loc[analysis_df["inunda"] == 1, "vol_inundacion_m3"].values)
    X_reg = analysis_df.loc[analysis_df["inunda"] == 1, FEATURE_COLS].values

    clf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(n_estimators=3, max_depth=2, random_state=42,
                                eval_metric="logloss")),
    ])
    clf_pipeline.fit(X, y_clf)

    reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(n_estimators=3, max_depth=2, random_state=42)),
    ])
    reg_pipeline.fit(X_reg, y_reg)

    plot_shap(clf_pipeline, reg_pipeline, analysis_df, tmp_path)

    assert (tmp_path / "shap_classifier_summary.png").exists()
    assert (tmp_path / "shap_regressor_summary.png").exists()
    assert (tmp_path / "shap_dependence_classifier_duracion_horas.png").exists()
    assert (tmp_path / "shap_dependence_classifier_tiempo_al_pico_h.png").exists()
    assert (tmp_path / "shap_dependence_regressor_duracion_horas.png").exists()
    assert (tmp_path / "shap_dependence_regressor_tiempo_al_pico_h.png").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/ml/test_feature_analysis.py::test_plot_shap_creates_summary_files -v
```

Expected: `FAILED` — `ImportError: cannot import name 'plot_shap'`.

- [ ] **Step 4: Add `plot_shap` to `swmm_resilience/ml/feature_analysis.py`**

Append after `run_ablation`:

```python
def plot_shap(clf_pipeline, reg_pipeline, df: pd.DataFrame, out_dir: Path) -> None:
    """Save SHAP beeswarm summaries and dependence plots for both models.

    Regressor SHAP values are in log1p(vol m³) space — axis labels note this.
    clf_pipeline and reg_pipeline are trusted local artifacts (written by train_models).
    """
    import shap  # lazy import — only needed when this function is called

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    display_names = [feature_display_name(f) for f in FEATURE_COLS]

    # ── Classifier ───────────────────────────────────────────────────────────
    X_clf = clf_pipeline.named_steps["imputer"].transform(df[FEATURE_COLS])
    clf_explainer = shap.TreeExplainer(clf_pipeline.named_steps["model"])
    clf_shap_values = clf_explainer.shap_values(X_clf)
    # XGBoost binary classifier: shap_values is either a 2D array or a list
    # [neg_class, pos_class] depending on shap version — normalise to pos_class
    if isinstance(clf_shap_values, list):
        clf_shap_values = clf_shap_values[1]

    shap.summary_plot(clf_shap_values, X_clf, feature_names=display_names, show=False)
    plt.title("SHAP Summary — Classifier", fontsize=12)
    plt.savefig(out_dir / "shap_classifier_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feat in ("duracion_horas", "tiempo_al_pico_h"):
        feat_idx = FEATURE_COLS.index(feat)
        shap.dependence_plot(
            feat_idx, clf_shap_values, X_clf,
            feature_names=FEATURE_COLS, show=False,
        )
        plt.savefig(
            out_dir / f"shap_dependence_classifier_{feat}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # ── Regressor (flooded rows only — model trained on log1p(vol)) ──────────
    df_flooded = df[df["inunda"] == 1].reset_index(drop=True)
    X_reg = reg_pipeline.named_steps["imputer"].transform(df_flooded[FEATURE_COLS])
    reg_explainer = shap.TreeExplainer(reg_pipeline.named_steps["model"])
    reg_shap_values = reg_explainer.shap_values(X_reg)
    if isinstance(reg_shap_values, list):
        reg_shap_values = reg_shap_values[1]

    shap.summary_plot(reg_shap_values, X_reg, feature_names=display_names, show=False)
    plt.title(
        "SHAP Summary — Regressor (values in log1p(vol m³) space)", fontsize=12
    )
    plt.savefig(out_dir / "shap_regressor_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feat in ("duracion_horas", "tiempo_al_pico_h"):
        feat_idx = FEATURE_COLS.index(feat)
        shap.dependence_plot(
            feat_idx, reg_shap_values, X_reg,
            feature_names=FEATURE_COLS, show=False,
        )
        plt.savefig(
            out_dir / f"shap_dependence_regressor_{feat}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/ml/test_feature_analysis.py::test_plot_shap_creates_summary_files -v
```

Expected: `PASSED`.

- [ ] **Step 6: Run full test suite**

```bash
pytest --tb=short -q
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/ml/feature_analysis.py tests/ml/test_feature_analysis.py
git commit -m "feat: add plot_shap to feature_analysis (beeswarm + dependence plots)"
```

---

## Task 4: Wire `--analyze-features` into `main.py`

**Files:**
- Modify: `main.py` — new CLI flag + handler block

**Interfaces:**
- Consumes: `plot_correlation`, `run_ablation`, `plot_shap` from `swmm_resilience.ml.feature_analysis`

---

- [ ] **Step 1: Add the CLI argument**

In `main.py`, locate the block that adds `--evaluate-shapes` (around line 75). Add the new argument immediately after it:

```python
    parser.add_argument("--analyze-features", action="store_true",
                        help="Correlación, ablación y SHAP para los features del modelo")
```

- [ ] **Step 2: Add the handler block**

In `main.py`, locate the `if args.evaluate_shapes:` block (around line 182). Add the new handler **before** it (so early-exit modes are grouped together):

```python
    # ── Modo: análisis de features ───────────────────────────────────────────
    if args.analyze_features:
        import joblib
        from swmm_resilience.ml.feature_analysis import (
            plot_correlation,
            run_ablation,
            plot_shap,
        )

        dataset_path = Path(config.dataset.output_path)
        if not dataset_path.exists():
            parser.error(
                f"--analyze-features requiere el dataset en {dataset_path}; "
                "ejecuta el pipeline completo primero"
            )

        models_dir = Path("outputs/models")
        clf_path = models_dir / "classifier.joblib"
        reg_path = models_dir / "regressor.joblib"
        if not clf_path.exists() or not reg_path.exists():
            parser.error(
                "--analyze-features requiere modelos entrenados en outputs/models/; "
                "ejecuta python main.py --only-ml primero"
            )

        df_analysis = pd.read_csv(dataset_path)
        # joblib/pickle is safe here: files are written by this pipeline's own
        # train_models() call and never loaded from an external or untrusted source.
        clf_pipeline = joblib.load(clf_path)
        reg_pipeline = joblib.load(reg_path)
        out_dir = Path("outputs/feature_analysis")

        print("\nAnálisis de features...")
        print("  1/3  Correlación...")
        plot_correlation(df_analysis, out_dir)
        print("  2/3  Ablación (LOSO × 2 runs) — puede tardar ~2 min...")
        ablation_result = run_ablation(df_analysis, config, out_dir)
        print(
            f"        Full  → F1={ablation_result['full']['classifier']['f1']:.3f}  "
            f"NSE={ablation_result['full']['regressor_oracle']['nse']:.3f}"
        )
        print(
            f"        Reduced → F1={ablation_result['reduced']['classifier']['f1']:.3f}  "
            f"NSE={ablation_result['reduced']['regressor_oracle']['nse']:.3f}"
        )
        print("  3/3  SHAP...")
        plot_shap(clf_pipeline, reg_pipeline, df_analysis, out_dir)
        print(f"\nResultados guardados en {out_dir}/")
        return
```

- [ ] **Step 3: Verify argument is registered correctly**

```bash
python main.py --help | grep analyze-features
```

Expected output contains: `--analyze-features`

- [ ] **Step 4: Run the full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: add --analyze-features CLI flag (correlation, ablation, SHAP)"
```

---

## Verification

After all tasks complete, run the full analysis end-to-end:

```bash
python main.py --analyze-features
```

Expected console output:
```
Análisis de features...
  1/3  Correlación...
  2/3  Ablación (LOSO × 2 runs) — puede tardar ~2 min...
        Full  → F1=0.7xx  NSE=0.9xx
        Reduced → F1=0.7xx  NSE=0.9xx
  3/3  SHAP...

Resultados guardados en outputs/feature_analysis/
```

Expected files in `outputs/feature_analysis/`:
- `correlation_pearson.png`
- `correlation_spearman.png`
- `feature_target_correlation.png`
- `ablation_results.json`
- `ablation_comparison.png`
- `shap_classifier_summary.png`
- `shap_regressor_summary.png`
- `shap_dependence_classifier_duracion_horas.png`
- `shap_dependence_classifier_tiempo_al_pico_h.png`
- `shap_dependence_regressor_duracion_horas.png`
- `shap_dependence_regressor_tiempo_al_pico_h.png`
