# Part 2 — approximate SVM and fair XGBoost / MLP / SVM comparison — design

**Date:** 2026-10-12
**Branch:** `fix/cierre-auditoria-hidrogramas-20260927` (worktree `optuna-tuning-familia-ganadora`).
**Precedes:** `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md` (Part 1, closed).
**Author's decisions (2026-10-12):** (1) label: **path A** (no re-simulation; the database is neither touched nor migrated); (2) XGBoost and MLP are always presented; the SVM enters as a third candidate with its result; (3) this week: corrected SVM or its support-vector alternative + runs with real data + update of the 2026-10-13 presentation; (4) do only training and low-harm or slow things.

## 1. Objective
Bring the SVM to a state comparable with XGBoost and MLP and obtain reproducible results for the three on the same data (`tabular_v4_21`, `prep_id ca706e27654e0cf5`, 154,080 rows), with the same tuning budget and the same evaluation protocol (LeaveOneShapeOut as the main figure), documented for the research.

## 2. Starting state
- `svm_family.py` (WIP commit `1ecca50`): Nystroem (RBF) + `SGDClassifier(loss="log_loss")` + `SGDRegressor(loss="epsilon_insensitive")`. The regressor is a linear SVR on the kernel mapping; the classifier is logistic regression, **not** a support vector machine. The 2026-10-05 benchmark left it 3rd: RMSE 155.2 m³, F1 0.843 (`v3_17` contract, untuned).
- `tune.py` already has `SEARCH_SPACES` for `xgboost`, `mlp` and `svm`, and a `run_study` (GroupKFold(5) by internal `shape_id`, objective `end_to_end.rmse_vol_todos_nodos`). It labels its results as `estimacion_exploratoria_condicionada` (conditional exploratory estimate).
- The CLI (`--bench-tune`, `--bench-evaluate`) only prepares/reads the `tabular_v3_17` contract; the current contract `tabular_v4_21` is already prepared in `outputs/bench/prepared_v4_21/ca706e27654e0cf5` and is used from scripts.
- Hardware: 64 CPUs, no GPU, torch CPU (32 threads).

## 3. Design decisions

### 3.1 SVM: "approximate SVM with Nystroem mapping"
- Classifier: `Nystroem(kernel="rbf")` → `CalibratedClassifierCV(SGDClassifier(loss="hinge", class_weight=…), method="sigmoid", cv=3)`. It is a linear margin SVM (*hinge* loss) on the approximate RBF kernel, with Platt calibration to obtain `predict_proba` (required by the evaluator for the AUC). With an approximate kernel, it is equivalent to the `SVC(probability=True)` used by the exact version.
- Regressor: unchanged — `SGDRegressor(loss="epsilon_insensitive")` on the same mapping (linear SVR on Nystroem), with the target in `log1p`.
- The internal family name remains `svm` (compatibility with the registry and the Optuna space); the preprocessing descriptor declares `svm_aproximado_nystroem` and the documentation calls it "approximate SVM (Nystroem + linear SVM)". It is not presented as an exact-kernel SVM.
- Tunable hyperparameters: `gamma`, `n_components`, `alpha` (classifier and regressor) and `epsilon` (regressor) — the existing space; an equivalent `C` is added via `alpha` (SGD). The space is not extended beyond that.
- Discarded alternative: exact `SVC`/`SVR` (O(n²)–O(n³); the 2026-10-05 run accumulated more than 1 day and 15 h of CPU without finishing).

### 3.2 Fair comparison
- Same data (`prep_id ca706e27654e0cf5`), same seed (`ML_RANDOM_STATE`), same objective (`end_to_end.rmse_vol_todos_nodos`), **same number of Optuna trials** for the three families, same internal CV (GroupKFold(5) by shape).
- Budget: equal N trials for the three, fixed after measuring the cost of one trial per family on this machine (see §5). If the MLP does not fit in the available time with 200 epochs, `epochs` is reduced and/or `batch_size` is raised **equally in all the MLP evaluations** (tuning and final evaluation) and this is declared.
- Declared bias: the hyperparameters are chosen with all the shapes (`estimacion_exploratoria_condicionada`), so the subsequent LeaveOneShapeOut figures are optimistic **equally** for the three families; they serve to compare, not as independent confirmation. A nested per-shape tuning (×40) remains as future work.
- XGBoost: besides the tuned result, the already measured result with default parameters is kept (`datos/runs/metrics_xgboost.json`), to show the effect of tuning.

### 3.3 Evaluation and reporting
- Main protocol: LeaveOneShapeOut (40 folds) for the tuned model of each family, plus the SVM with default parameters (cheap). LOSO/GroupKFold5 are not repeated for these runs (they measure interpolation between factors; already documented) unless time remains.
- No-learning references: the ones already computed are reused (same `prep_id`).
- Tables with the same format and provenance as `docs/auditoria/correcciones/datos/` (`informe_evaluacion.build_tables`), in a new directory `docs/auditoria/correcciones/datos_parte2/`: comparison, confusion matrix, 1–2 m³ band and per shape, plus the summary of each study (best parameters, trials, timings, convergence curve) and `provenance.json`.
- Document `docs/auditoria/correcciones/06-seleccion-de-modelos.md` with "Text for the thesis" and the presentation rule: XGBoost and MLP always; the SVM as a third candidate with its result.

### 3.4 Execution
- Resumable script `scripts/ejecutar_parte2_v4_21.py`: stages per family (study → best parameters → LeaveOneShapeOut evaluation → OOF and metrics); each stage is persisted when it finishes and is skipped if it already exists. Order: svm, xgboost, mlp (the MLP is the bottleneck). It does not write to `outputs/models`, `outputs/bench/reports`, `outputs/bench/prepared*`, the SQLite database or `data/`.
- No migrations or re-simulation. No model is promoted.

## 4. Out of scope
Re-simulation/regeneration of the dataset (path B); nested tuning; extending the search space; GroupKFold5/LOSO for the tuned models; promotion of a model to `outputs/models`; temporal LSTM/CNN.

## 5. Execution plan (compact, per task)
1. **Approximate SVM** (code, TDD): `svm_family.py` with the hinge + Platt classifier; contract tests (predict_proba in [0,1], `class_weight`, determinism, unknown parameters fail, descriptor); Optuna space unchanged.
2. **Cost measurement** (spike, read-only): time of one fit and of one trial per family on this machine over `prepared_v4_21`; proposal for N and for the MLP `epochs/batch_size`.
3. **Resumable script** (code, TDD on the pure parts): stages, persistence, provenance, tables.
4. **Real run** (in the background), monitored by stages.
5. **Document 06** and **presentation** (update with what already exists and then with results).
