# Correction 02 — Traceability of figures and models, and script safety (V-01, V-07, M-07/V-04)

- **Date:** 2026-10-11
- **Branch:** `worktree-optuna-tuning-familia-ganadora`. Commits `93e34b4`, `57dc170`, `8f17147`, `0c8a3f4` and `398d6bc`.
- **Origin of the findings:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md), [`../anexos/05-analisis-visualizacion-soporte.md`](../anexos/05-analisis-visualizacion-soporte.md) (V-01, V-04, V-07) and [`../anexos/02-ml-protocolo-evaluacion.md`](../anexos/02-ml-protocolo-evaluacion.md) (M-07, M-13).
- **Scope:** these changes do not modify any model, data or metric. They make visible whether a figure is an in-sample fit and which model generated it, and they prevent a script or a test from altering reference files.

---

## 1. V-01 (and M-13 in part) — In-sample figures and outputs without labels

### Problem and evidence

The resilience curves, the total flooded volume curves and the per-factor comparison evaluate the model on the same factors and nodes it was trained with: `train_models` fits with the complete dataset and `predict_network` is evaluated on the 25 factors of the dataset (0.2 to 5.0) and the 160 nodes (annex 05, V-01). The titles said "SWMM (Actual)" and "ML Prediction" without indicating that it is an in-sample fit. The same happened with the output of `--evaluate-shapes`, which prints F1 and NSE on training shapes and factors (annex 02, M-13). A reader may take those figures and numbers as validation.

### Cause

The figures were designed to show the behaviour of the final model on the simulation grid; there was no explicit distinction between fit and validation in the visualization routines or in the CLI.

### What changed

Commit `93e34b4`:

| File | Change |
|---|---|
| `swmm_resilience/visualization/labels.py` | Constants `IN_SAMPLE_NOTE` (figure footnote, in English: "In-sample: the model was trained on these scenarios; this is not an independent validation.") and `IN_SAMPLE_BANNER` (console warning, in Spanish: "[EN MUESTRA] Estos escenarios (formas y factores) se usaron para entrenar el modelo; las métricas no son una validación independiente." — i.e. "[IN SAMPLE] These scenarios (shapes and factors) were used to train the model; the metrics are not an independent validation."), and the functions `add_in_sample_note(fig)` and `print_in_sample_banner()` |
| `swmm_resilience/visualization/resilience_curve.py` | Footnote in `resilience_ml.png` (the SWMM-only figure does not change) |
| `swmm_resilience/visualization/flood_volume_curve.py` | Footnote in `flood_volume_ml.png` and `flood_volume_combined.png` |
| `swmm_resilience/visualization/model_comparison.py` | Footnote in the per-factor comparison figures (`volume_by_node_factor_*`, `parity_factor_*`) |
| `main.py` | `--evaluate-shapes` prints the warning before its metrics |

### Verification

| Check | Before | After | Source |
|---|---|---|---|
| ML figure families with an "in-sample" warning | 0 | 5 (`resilience_ml`, `flood_volume_ml`, `flood_volume_combined`, `volume_by_node_factor_*`, `parity_factor_*`) | Commit `93e34b4`; `tests/visualization/test_in_sample_note.py` |
| Warning in the output of `--evaluate-shapes` | No | Yes | `test_the_evaluate_shapes_branch_prints_the_in_sample_banner` |
| New tests | — | 6, all green | `tests/visualization/test_in_sample_note.py` (`test_add_in_sample_note_writes_the_note_into_the_figure`, `test_resilience_ml_figure_carries_the_in_sample_note`, `test_flood_volume_figures_carry_the_in_sample_note`, `test_factor_comparison_figures_carry_the_in_sample_note`, `test_print_in_sample_banner_emits_the_banner`, `test_the_evaluate_shapes_branch_prints_the_in_sample_banner`) |

According to the task review log, the reviewer inspected the rendered figures.

### Remaining limits

- **The figures already existing in `outputs/metrics/` were not regenerated.** They do not carry the warning. Only those generated from now on carry it.
- **A figure footnote was added; titles and legends were not changed.** The spec (§2.6) asked to add "(in sample)" to the title as well; the implementation was limited to the footnote.
- **The footnote is in English**, like the rest of the text of the figures. If a figure goes into the thesis, the "in-sample" character must also be stated in the Spanish-language caption.
- **M-13 remains partially open.** `--evaluate-generalization` (interpolation between seen factors) and `--evaluate-validation-shapes` (which protects the out-of-sample condition only by the name of the shape) were not labelled.
- **No out-of-sample versions of the curves were generated** from the out-of-fold predictions (optional improvement, spec §7).
- Tests: the figure test checks the text of the note, not its position in the image; with more than 30 nodes the note sits close to the axis label (deferred nitpicks, see `04`).

### Text for the thesis

> The resilience curves, the total flooded volume curves and the per-factor comparisons presented for the final model were computed on the same combinations of shape, factor and node with which the model was trained. They therefore describe the fit of the model to the training data and not its predictive capacity; this is indicated in each figure. The out-of-sample evaluation is that of the held-out-shape cross-validation and that of the validation hydrographs not used in training.

---

## 2. V-07 — `halve_timeseries.py` overwrote the base network when imported

### Problem and evidence

`halve_timeseries.py` was a versioned script without an `if __name__ == "__main__"` guard and with fixed relative paths: when imported or executed it read the Qx2 network and **overwrote** the base `.inp` `chico_hydro-qx1/...Qx1.00.inp` (the one used by `config.yaml`), dividing the series by 2 (annex 05, V-07; confirmed by reading in audit §3.1). That would change the MD5 hash of the network, and `predict_network` would reject the models (`training_inp_hash.txt`). The audit also verified that the current Qx1 `.inp` is consistent with Qx2/2 (`max|Qx1·2 − Qx2| = 0.001`), that is, the script or an equivalent had already been applied at some point.

### Cause

Executable code at module level, without arguments or protection against overwriting.

### What changed

Commit `57dc170`, files `halve_timeseries.py` and `tests/scripts/test_halve_timeseries.py`:

- Function `halve_timeseries(src, dst, force=False)` that divides by 2 only the `[TIMESERIES]` section.
- `main()` with explicit arguments `--src`, `--dst` and `--force`, under `if __name__ == "__main__"`. Importing the module has no side effects.
- It refuses to overwrite an existing destination without `force` and to use the source as the destination.
- The file was not deleted.

### Verification

| Check | Before | After | Source |
|---|---|---|---|
| Importing the module writes to `data/` | Yes (audit §3.1) | No | `test_importing_the_module_has_no_side_effects` |
| Overwrites an existing destination without `--force` | Yes (it always wrote) | No | `test_refuses_to_overwrite_an_existing_destination_without_force` |
| Accepts the source as the destination | — | No | `test_refuses_to_use_the_source_as_destination` |
| Changes only `[TIMESERIES]` | — | Yes | `test_halves_only_the_timeseries_section` |
| `git status --short data/` before and after the tests | — | Empty in both cases | Report of task 6 (unversioned working notes) |

### Remaining limits

- The provenance of the Qx1, Qx2 and Qx3 networks (Qx1 as a rounded derivative of Qx2) remains undocumented (recommendation of annex 05).
- Deferred nitpicks: loose assertions (`in`) in the halving test, `main()` without a CLI test, a race window between checking and writing the destination, and no warning if the section has no rows (see `04`).
- The paired finding in the audit plan, T-04 (`python -m swmm_resilience.ml.train --help` trains), was not addressed in this series.

### Text for the thesis

No text is required in the thesis: it is a repository integrity risk, not a risk to the results. If it is to be mentioned in the reproducibility section: *"The auxiliary scripts that derive networks from other networks are executed only explicitly and do not overwrite existing files, so that the base network used to generate the data cannot be altered by accident."*

---

## 3. M-07 / V-04 — Model on disk different from the promoted one; figures without provenance

### Problem and evidence

In the audited checkout (branch `fix/cierre-auditoria-hidrogramas-20260927`, commit `6e8579b`), `outputs/models/classifier.joblib` had SHA-256 `660be918…` while `promotion.json` declared `e2e7925e…`; the regressor, `6b61f627…` versus `40e67e22…` (audit §3.1; annex 05, V-04). The model on disk had been retrained by the default pipeline (`train_models`) on 3 October, with 40 shapes, while the promotion recorded a benchmark candidate with 19 shapes (annex 02, M-07). The figures in `outputs/metrics/` dated from 6 to 15 September, prior to the loaded model. It could not be stated which model had generated each figure nor whether `promotion.json` told the truth.

**In the worktree where these corrections were made the starting state was different:** the models in `outputs/models` are versioned and their SHA-256 values already matched `promotion.json` (spec §1). `promotion.json` itself, however, was ignored by `.gitignore` (`outputs/models/*`) and was not versioned; since commit `25eb169` it is versioned with `git add -f`, and `.gitignore` has the explicit exception `!outputs/models/promotion.json` (commit `e90448d`), so that future promotions are not silently ignored. `verify_promotion` stops the prediction (`ModelProvenanceError`) only when `promotion.json` EXISTS and does not match the models; if it is missing, it warns and continues, and the figures are left with family "desconocida" (unknown). For this document the check was repeated with `verify_promotion("outputs/models")`, which returns `{'status': 'verified', 'ok': True, 'discrepancies': []}`; the `promotion.json` of the worktree declares `family: xgboost`, `classifier_sha256` `a7bb8f8f…`, `source` `scripts/comparar_contratos_xgboost.py` and date 2026-10-06. That is, the discrepancy in the audit was due to the outdated checkout, not to the worktree. What was missing in both was an **automatic verification** when loading the model and **provenance** in the figures.

### Cause

`predict_network` loaded `classifier.joblib` and `regressor.joblib` without comparing their SHA-256 values with `promotion.json`, and the default pipeline could rewrite `outputs/models` without touching `promotion.json` (annex 02, M-07: `main.py:1082`, `ml/trainer.py:78-80` in the audited checkout). In the worktree, `train_models` already refused to overwrite a model with another number of features and wrote its own `promotion.json` (changes prior to this series, not reviewed here).

### What changed

| Commit | File | Change |
|---|---|---|
| `8f17147` | `swmm_resilience/ml/bench/promote.py` | `verify_promotion(models_dir)` compares the SHA-256 of `classifier.joblib` and `regressor.joblib` with `promotion.json` and returns a structured result; `assert_promotion_consistent` raises `ModelProvenanceError` with both SHA values if they do not match, and warns and continues if `promotion.json` does not exist (legacy models); `model_provenance` summarizes the active model |
| `8f17147` | `swmm_resilience/ml/predict.py` | `predict_network` verifies before loading the models (deferred import so as not to load torch) |
| `8f17147` | `swmm_resilience/visualization/runner.py` | `write_figure_provenance` writes `provenance.json` next to the ML figures of `generate_ml_map` and of the `run` batch (in the batch, only if at least one new figure was generated, so as not to attribute old figures to the current model) |
| `0c8a3f4` | `promote.py`, `main.py` | Without `promotion.json`, the provenance records the real SHA-256 of the model files (family "desconocida" (unknown)); `main.py` stamps `provenance.json` after `--resilience-curve`, `--flood-volume-curve` and the per-factor comparison |
| `0c8a3f4` | `tests/test_ml_trainer_predict.py`, `tests/test_predict_shape_features.py` | The existing prediction tests create a `promotion.json` consistent with their test model (removing spurious warnings) |

### Verification

| Check | Before | After | Source |
|---|---|---|---|
| `predict_network` with a model that does not match `promotion.json` | Loads and predicts | Refuses (`ModelProvenanceError` with both SHA values) | `test_predict_network_refuses_models_that_do_not_match_promotion_json`, `test_assert_raises_on_mismatch_naming_the_hashes` |
| Without `promotion.json` | Loads without warning | Warns and continues | `test_predict_network_warns_and_proceeds_without_promotion_json` |
| Curve and comparison figures with `provenance.json` | No | Yes | `test_figure_provenance_records_the_active_model`, `test_figure_provenance_without_a_record_stamps_the_artifact_hashes`, tests in `tests/test_cli_sql_reads.py` |
| Importing `ml/predict.py` loads torch | No | No (kept) | `test_importing_predict_does_not_load_torch` |
| State of `outputs/models` in the worktree | Matching SHA values (spec §1) | `verified`, 0 discrepancies | `verify_promotion("outputs/models")`, run 2026-10-11 |
| New tests | — | 15 in `tests/ml/bench/test_verify_promotion.py` and 2 in `tests/visualization/test_figure_provenance.py` | Cited files |

During all the runs of this series, the SHA-256 of `outputs/models/*`, of `outputs/training_v17.sqlite3` and of `outputs/bench/reports/*` did not change (report of task 8, `sha256sum -c` before and after; unversioned working notes).

### Remaining limits

- **Other model loaders do not go through `verify_promotion`.** They still load without verification (decision log R8; line numbers at the time of the task 7 review):
  - `ScenarioPredictor` in `main.py` (lines 614, 702 and 831 at that time);
  - the SHAP analysis (`main.py`, line 522: direct `joblib.load`);
  - `predict_from_inp` through `ml/train.py` (`load_saved_model_artifact`), which is also used by the ML map of the desktop application (`generate_ml_map`).
  The desktop map does receive `provenance.json`, but the model it draws is not verified against `promotion.json`.
- **A malformed `promotion.json` or `manifest.json`** produces a `JSONDecodeError` without a domain message.
- **Coarse granularity:** `provenance.json` describes the active model at the time of writing, per folder, not per figure.
- **The existing figures were not regenerated** with the declared model (audit recommendation, P0 item 4): they remain without provenance. The final model of the thesis is not yet fixed (Part 2).
- **Without `promotion.json`, the provenance identifies files, not candidates:** the SHA-256 of the artefacts does not say with which `prep_id` or with which benchmark candidate they were trained.

### Text for the thesis

> Each figure generated with the machine-learning model is accompanied by a provenance record with the model family, the SHA-256 fingerprint of the classifier and regressor files, the training dataset and the date. Before predicting, the system checks that the model files match the promotion record and stops if they do not, so that a figure produced with one model cannot be attributed to another.

(If this paragraph is used, it should be limited to the thesis figures that are actually regenerated with the final model, and not applied to the scenario prediction, SHAP or desktop paths, which are not verified.)

---

## 4. Defect found during the work — tests that wrote false provenance to `outputs/metrics`

### Problem and evidence

When provenance was added in `main.py` (commit `0c8a3f4`), three existing tests that run `main.main()` started writing `provenance.json` to the **real** `outputs/metrics` of the repository: `tests/test_cli_sql_reads.py::test_resilience_curve_reads_the_frame_from_sql`, `::test_flood_volume_curve_reads_the_frame_from_sql` and `tests/test_factor_comparison_cli.py::test_factor_comparison_cli_generates_plots_in_metrics_directory`. It was detected during the run of task 8: the full suite overwrote `outputs/metrics/provenance.json` and `outputs/metrics/factor_comparison/provenance.json` (report of task 8). Those files attributed the existing figures to the test model: false provenance.

### Cause

The tests replaced only the computation and the drawing, but left `main.METRICS_DIR` pointing to the real folder, and nothing prevented a test from writing there.

### What changed

Commit `398d6bc` (decision log R10):

- The three tests redirect `main.METRICS_DIR` to a temporary folder; their assertions do not change.
- `tests/conftest.py` adds an automatic guard: before and after each test it takes (size, modification date) of each file under the real `outputs/metrics` and fails if any was created or changed.
- The two spurious `provenance.json` files were deleted (ignored by git; there was no legitimate version to restore).

### Verification

| Check | Result | Source |
|---|---|---|
| Suite with the guard and without the fix | 1,145 pass; 3 teardown errors, exactly the three guilty tests | Report of task 7, second round (unversioned working notes) |
| Suite with the guard and the fix | 1,145 pass, 0 errors; the two files do not reappear | same; decision log R10 |
| Full suite at the close of the series | See `04` §7 | Command of `04` §7 |

### Remaining limits

- The guard watches only `outputs/metrics`, not `outputs/models` or `outputs/bench`.
- The guard has no test of its own demonstrating that it detects a write.
- An empty folder `outputs/metrics/factor_comparison/` was left behind.

### Text for the thesis

No text is required in the thesis. It is a defect of the test environment, corrected before any figure or number of the thesis was generated.
