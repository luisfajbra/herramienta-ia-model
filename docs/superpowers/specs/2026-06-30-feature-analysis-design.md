# Feature Analysis Module — Design Spec

## Goal

Add a `--analyze-features` CLI command that runs three analyses (correlation,
ablation, SHAP) on the trained XGBoost models and the 28k-row dataset, saving
thesis-ready figures to `outputs/feature_analysis/`.

## Context

The pipeline trains a classifier (flood/no-flood) and a regressor (flood volume
in log1p space) on 17 features. Two features — `duracion_horas` and
`tiempo_al_pico_h` — were added to capture hydrograph shape characteristics.
This module validates their contribution for the thesis.

Dataset: `data/training/dataset_final.csv` — 28,000 rows × 23 columns  
(7 shapes × 25 factors × 160 nodes).

Trained models: `outputs/models/classifier.joblib`, `outputs/models/regressor.joblib`

---

## Architecture

One new module: `swmm_resilience/ml/feature_analysis.py`  
Three public functions, each independently testable.  
One new CLI flag in `main.py`: `--analyze-features`  
New dependency: `shap` added to `requirements.txt`.

### Output structure

```
outputs/feature_analysis/
├── correlation_pearson.png
├── correlation_spearman.png
├── feature_target_correlation.png
├── ablation_results.json
├── ablation_comparison.png
├── shap_classifier_summary.png
├── shap_regressor_summary.png
├── shap_dependence_classifier_duracion_horas.png
├── shap_dependence_classifier_tiempo_al_pico_h.png
├── shap_dependence_regressor_duracion_horas.png
└── shap_dependence_regressor_tiempo_al_pico_h.png
```

---

## Section 1 — Correlation

**Function:** `plot_correlation(df: pd.DataFrame, out_dir: Path) -> None`

**Inputs:** Full 28k-row dataset (all columns present).

**Outputs:**

- `correlation_pearson.png` — 17×17 Pearson correlation heatmap of feature columns
  only. Upper triangle masked. Color diverges at 0 (e.g., `coolwarm`). Feature
  labels use `feature_display_name()` from `swmm_resilience.visualization.labels`.
- `correlation_spearman.png` — same layout, Spearman ρ.
- `feature_target_correlation.png` — horizontal grouped bar chart. For each of
  the 17 features, show Spearman ρ against:
  - `inunda` (binary, all 28k rows)
  - `vol_inundacion_m3` (continuous, flooded rows only — ~8k rows)
  Two bars per feature, sorted descending by absolute ρ against `inunda`.

**Constraints:**
- Feature order in heatmaps: same as `FEATURE_COLS` in `trainer.py`
- Annotations (numeric ρ values) only shown in cells where |ρ| ≥ 0.3 to avoid clutter
- DPI: 150; `bbox_inches="tight"`; `plt.close()` after each figure

---

## Section 2 — Ablation

**Function:** `run_ablation(df: pd.DataFrame, config: Config, out_dir: Path) -> dict`

**Inputs:** Full dataset + config (for ML hyperparameters and LOSO grouping).

**Two runs:**

| Run | Feature set | N features |
|-----|-------------|-----------|
| Full | All 17 (`FEATURE_COLS`) | 17 |
| Reduced | `FEATURE_COLS` minus `duracion_horas` and `tiempo_al_pico_h` | 15 |

**Cross-validation:** LOSO grouped by `factor_mult` — identical to `evaluator.py`
`_run_cv`. The ablation calls `_run_cv` directly (imported, not duplicated) with a
temporary config-like object or by passing the feature list explicitly.

> Implementation note: `_run_cv` in `evaluator.py` reads `X = df[FEATURE_COLS].values`
> directly. The ablation must NOT call `_run_cv` directly — instead, implement a
> minimal local CV loop inside `feature_analysis.py` that accepts a `feature_cols`
> list parameter. Copy only the metric computation logic (classifier F1/AUC-ROC,
> regressor oracle NSE/R²) — do not duplicate the by-factor stratification or
> end-to-end metrics. This keeps `evaluator.py` unchanged and avoids brittle
> cross-module coupling.

**Metrics compared** (LOSO averages):
- Classifier: F1, AUC-ROC
- Regressor oracle: NSE, R²

**Outputs:**

- `ablation_results.json` — structure:
  ```json
  {
    "full":    {"classifier": {"f1": ..., "auc_roc": ...}, "regressor_oracle": {"nse": ..., "r2": ...}},
    "reduced": {"classifier": {"f1": ..., "auc_roc": ...}, "regressor_oracle": {"nse": ..., "r2": ...}}
  }
  ```
- `ablation_comparison.png` — grouped bar chart. 4 metric groups on x-axis (F1,
  AUC-ROC, NSE, R²). Two bars per group: Full (blue) and Reduced (orange). y-axis
  0–1. Annotate each bar with its numeric value (2 decimal places). Title:
  "Ablation Study: Full vs. Reduced Feature Set (LOSO)".

**Returns:** the dict written to `ablation_results.json`.

---

## Section 3 — SHAP

**Function:** `plot_shap(clf_pipeline, reg_pipeline, df: pd.DataFrame, out_dir: Path) -> None`

**Inputs:**
- `clf_pipeline` — loaded sklearn Pipeline (imputer + XGBClassifier) from
  `outputs/models/classifier.joblib`
- `reg_pipeline` — loaded sklearn Pipeline (imputer + XGBRegressor) from
  `outputs/models/regressor.joblib`
- `df` — full 28k-row dataset

**SHAP computation:**
- Use `shap.TreeExplainer` on `pipeline.named_steps["model"]` (the raw XGBoost model)
- Pass imputed X through the pipeline imputer first:
  `X_imp = pipeline.named_steps["imputer"].transform(df[FEATURE_COLS])`
- Classifier: compute on all 28k rows
- Regressor: compute on flooded rows only (`df[df["inunda"] == 1]`), since the
  regressor was trained only on those rows

**Outputs:**

- `shap_classifier_summary.png` — `shap.plots.beeswarm(shap_values, show=False)` —
  all 17 features, sorted by mean |SHAP|. Feature labels via `feature_display_name()`.
- `shap_regressor_summary.png` — same, on flooded subset. y-axis label notes
  "SHAP value (log1p(vol m³) space)".
- `shap_dependence_classifier_duracion_horas.png` — `shap.dependence_plot(
  "duracion_horas", shap_values, X_imp, feature_names=FEATURE_COLS, show=False)`.
  X-axis: raw feature values; Y-axis: SHAP value; color: auto-selected interaction
  feature.
- `shap_dependence_classifier_tiempo_al_pico_h.png` — same for `tiempo_al_pico_h`.
- `shap_dependence_regressor_duracion_horas.png`
- `shap_dependence_regressor_tiempo_al_pico_h.png`

**All figures:** DPI 150, `bbox_inches="tight"`, `plt.close()` after each.

---

## CLI Integration (`main.py`)

New argument:
```python
parser.add_argument("--analyze-features", action="store_true",
                    help="Correlation, ablation y SHAP para los features del modelo")
```

Handler (after dataset is loaded, models exist):
```python
if args.analyze_features:
    import joblib
    from swmm_resilience.ml.feature_analysis import plot_correlation, run_ablation, plot_shap

    dataset_path = Path(config.dataset.output_path)
    if not dataset_path.exists():
        parser.error(f"--analyze-features requiere el dataset en {dataset_path}")

    models_dir = Path("outputs/models")
    clf_path = models_dir / "classifier.joblib"
    reg_path = models_dir / "regressor.joblib"
    if not clf_path.exists() or not reg_path.exists():
        parser.error("--analyze-features requiere modelos entrenados en outputs/models/")

    df_analysis = pd.read_csv(dataset_path)
    # joblib/pickle is safe here: files are written by this pipeline's own
    # train_models() call and never loaded from an external or untrusted source.
    clf_pipeline = joblib.load(clf_path)
    reg_pipeline = joblib.load(reg_path)
    out_dir = Path("outputs/feature_analysis")

    print("\nAnálisis de features...")
    print("  1/3 Correlación...")
    plot_correlation(df_analysis, out_dir)
    print("  2/3 Ablación (LOSO × 2 runs)...")
    run_ablation(df_analysis, config, out_dir)
    print("  3/3 SHAP...")
    plot_shap(clf_pipeline, reg_pipeline, df_analysis, out_dir)
    print(f"\nResultados guardados en {out_dir}/")
```

---

## Tests

**File:** `tests/ml/test_feature_analysis.py`

Synthetic dataset: 200 rows, 17 feature columns (random floats), `inunda` binary,
`vol_inundacion_m3` positive floats where `inunda=1`, `factor_mult` column with
5 distinct values (for LOSO grouping).

Tests:
1. `test_plot_correlation_creates_files` — call `plot_correlation(df, tmp_path)`,
   assert the three PNG files exist.
2. `test_run_ablation_returns_both_runs` — call `run_ablation(df, config, tmp_path)`,
   assert result dict has keys `"full"` and `"reduced"`, both contain
   `"classifier"` and `"regressor_oracle"` sub-dicts with numeric values.
3. `test_run_ablation_reduced_has_fewer_features` — assert that the JSON file
   is written and `ablation_results.json` exists in `tmp_path`.
4. `test_plot_shap_creates_summary_files` — train minimal XGB pipelines on the
   synthetic dataset, call `plot_shap(clf, reg, df, tmp_path)`, assert
   `shap_classifier_summary.png` and `shap_regressor_summary.png` exist.

---

## Dependencies

Add to `requirements.txt`:
```
shap>=0.44.0
```

---

## Constraints

- All figure functions must call `plt.close()` after saving — no GUI windows opened
- Feature display names always via `feature_display_name()` — never raw column names
- No global state — each function is self-contained and re-entrant
- `out_dir` is always created with `mkdir(parents=True, exist_ok=True)`
- The ablation does NOT modify `FEATURE_COLS` or any global — it operates on a
  local feature list
