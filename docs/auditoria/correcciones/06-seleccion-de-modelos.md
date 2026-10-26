# 06 — Model selection: fair XGBoost / MLP / approximate SVM comparison (part 2)

**Date:** 2026-10-13 · **Run:** commit `8a06605` (`git_dirty: false`) · **Data:** `datos_parte2/` · **Design:** `docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md`

Source abbreviations used throughout the document:

| Abbreviation | File |
|---|---|
| CP | `datos_parte2/comparacion_protocolos.csv` |
| MC | `datos_parte2/matriz_confusion.csv` |
| BL | `datos_parte2/banda_limitrofe.csv` |
| PF | `datos_parte2/por_forma.csv` |
| RA | `datos_parte2/resumen_afinacion.csv` |
| CA | `datos_parte2/convergencia_afinacion.csv` |
| TP | `datos_parte2/tables_provenance.json` |
| ST | `outputs/bench/parte2/ca706e27654e0cf5/<familia>/<etapa>/stage.json` |
| BS | per-shape bootstrap resampling of this document (§5.3; disposable script, not versioned) |

Figures marked "derived" are computed arithmetically from the cited column; sections marked **Reading** are interpretation, not measurement.

---

## 1. Summary

1. Three model families were compared —XGBoost, a dense neural network (MLP) and an approximate SVM (Nystroem + linear SVM)— with the same data, the same protocol (LeaveOneShapeOut, 40 shapes) and the same tuning budget (20 Optuna trials per family, same objective).
2. **Tuned XGBoost** obtains the lowest pooled end-to-end volume RMSE: **69.64 m³**, versus 101.12 m³ for the tuned MLP, 214.87 m³ for the tuned approximate SVM and 172.86 m³ for the no-learning reference (CP `rmse_vol_agregado_m3` (pooled volume RMSE)).
3. Tuning reduces the pooled RMSE by 38.3 % in XGBoost, 11.6 % in the MLP and 3.4 % in the approximate SVM (derived from CP).
4. As classifiers, XGBoost and the MLP are practically indistinguishable (AUC 0.9993–0.9994, F1 0.978–0.981); the approximate SVM falls below (AUC 0.963–0.972) and its RMSE is above that of the no-learning reference.
5. The figures are a **conditional exploratory estimate** (hyperparameters chosen with all the shapes), from a single run and one seed; a per-shape bootstrap (§5.3) indicates that the advantage of XGBoost over the MLP is stable when shapes are resampled, but the intervals of each model are wide.

---

## 2. Design of the comparison

| Element | Value | Source |
|---|---|---|
| Feature contract | `tabular_v4_21` | TP `feature_contract_id` |
| Prepared dataset | `prep_id = ca706e27654e0cf5` | TP `prep_id` |
| Rows | 154,080 (node × run) | TP `n_rows` |
| Runs | 963 of 1,000 in the grid; 37 cells are missing after the continuity-error filter ≤ 5 % | TP `n_runs`, `n_runs_full_grid`, `n_missing_cells`, `max_continuity_error_pct` |
| Network | a single drainage network | design |
| Flooding threshold | 1.0 m³ | TP `flood_threshold_m3` |
| Evaluation protocol | LeaveOneShapeOut: 40 folds, in each one a complete hydrograph shape is held out | TP `evaluation_protocol`; PF (40 shapes per model) |
| Main metric | Pooled end-to-end volume RMSE over all nodes | CP `rmse_vol_agregado_m3` |
| Tuning objective | the same end-to-end RMSE, with internal GroupKFold(5) grouped by shape | `scripts/ejecutar_parte2_v4_21.py` (docstring; `INNER_SPLITS = 5`) |
| Budget | **20 trials per family**, identical for the three | RA `n_trials`; `N_TRIALS = 20` |
| Sampler | Optuna TPE (`TPESampler(seed=42)`), with its 10 initial random trials by default: 10 random + 10 guided | `swmm_resilience/ml/bench/tune.py` |
| Seed | 42 | ST `seed` |

**Models.** All are two-stage: a classifier decides whether the node floods and a regressor estimates the volume; the end-to-end prediction is the regressor volume where the classifier predicts flooding and 0 elsewhere (verified: with this rule the OOFs exactly reproduce CP, 69.642 and 101.117 m³; BS).

| Family | Classifier | Regressor | Searched hyperparameters (per stage) | Fixed parameters (not searched) |
|---|---|---|---|---|
| XGBoost | gradient-boosted trees | same | `max_depth` 3–10, `learning_rate` 1e-3–0.3 (log), `n_estimators` 50–500, `subsample` 0.5–1.0 | `n_jobs = 1` |
| MLP (dense network) | dense network | dense network | `hidden_sizes` ∈ {(32), (64, 32), (128, 64), (64, 32, 16)}, `learning_rate` 1e-4–1e-2 (log), `dropout` 0–0.5 | `epochs = 200`, `batch_size = 1024`, 1 thread |
| Approximate SVM | Nystroem (RBF kernel) + linear SVM with *hinge* loss + Platt calibration | Nystroem (RBF) + linear ε-insensitive SVR | `gamma` 1e-3–1 (log), `n_components` 100–400, `alpha` 1e-6–1e-2 (log); the regressor additionally `epsilon` 0.01–1.0 | none |

Sources: `SEARCH_SPACES` in `swmm_resilience/ml/bench/tune.py`; RA `fixed_params`; `swmm_resilience/ml/bench/models/svm_family.py`.

The approximate SVM **is not an exact-kernel SVM**: it approximates the RBF kernel with at most 400 Nystroem components and solves a linear problem on them. An exact SVM is not feasible with 154,080 rows × 40 folds × 20 trials in the available time.

`n_jobs = 1` is fixed in XGBoost because, inside the per-fold processes (with `OMP_NUM_THREADS=1`), XGBoost already runs with one thread even if more are requested; it was measured that `n_jobs=4` in a child process gives exactly the predictions of `n_jobs=1` and not those of `n_jobs=4` in the main process. Fixing it makes tuning and evaluation coincide and be reproducible (docstring of `scripts/ejecutar_parte2_v4_21.py`).

For each family two variants were evaluated in LeaveOneShapeOut: `default` (parameters of `config.yaml` plus the fixed ones in the table) and `tuned` (best parameters of the study). The no-learning reference (`baseline_forma_mas_cercana`) copies the response of the most similar training shape.

---

## 3. Results

### 3.1 Main table (LeaveOneShapeOut, 40 folds)

| Model | Variant | Pooled RMSE (m³) | Per-fold mean RMSE (m³) | F1 | Recall | Precision | AUC | Oracle NSE | Total volume error (%) |
|---|---|---|---|---|---|---|---|---|---|
| Reference: nearest shape | no learning | 172.86 | 98.14 | 0.9764 | 0.9808 | 0.9721 | 0.9844 | 0.7102 | −2.98 |
| XGBoost | default | 112.80 | 58.65 | 0.9815 | 0.9919 | 0.9714 | 0.9993 | 0.8766 | −8.69 |
| XGBoost | **tuned** | **69.64** | **40.19** | 0.9812 | 0.9918 | 0.9707 | 0.9993 | **0.9530** | −2.95 |
| MLP | default | 114.38 | 64.08 | 0.9783 | 0.9889 | 0.9679 | 0.9992 | 0.8778 | −5.79 |
| MLP | tuned | 101.12 | 50.13 | 0.9783 | 0.9898 | 0.9672 | 0.9994 | 0.9018 | −5.67 |
| Approximate SVM | default | 222.46 | 142.64 | 0.8625 | 0.8544 | 0.8707 | 0.9719 | 0.5476 | −15.99 |
| Approximate SVM | tuned | 214.87 | 144.16 | 0.8302 | 0.8037 | 0.8586 | 0.9626 | 0.5831 | −15.81 |

Source: CP, columns `rmse_vol_agregado_m3`, `rmse_vol_media_por_fold_m3`, `f1_agregado`, `recall_agregado`, `precision_agregada`, `auc_roc_agregado`, `nse_regresor_oracle`, `error_vol_total_pct`. In the seven models `folds_f1_indefinido = 0`.

Definitions: the **pooled RMSE** gathers the 154,080 residuals of the 40 folds and computes a single RMSE (main metric); the **per-fold mean RMSE** is the mean of the 40 per-shape RMSEs (each shape weighs the same, and since the error is concentrated in a few shapes, it turns out lower). The **oracle NSE** evaluates only the regressor on the nodes that actually flooded (without classifier errors).

### 3.2 Effect of tuning (default → tuned)

| Family | Pooled RMSE (m³) | Change | Per-fold mean RMSE (m³) | Change | Oracle NSE | F1 |
|---|---|---|---|---|---|---|
| XGBoost | 112.80 → 69.64 | −38.3 % | 58.65 → 40.19 | −31.5 % | 0.8766 → 0.9530 | 0.9815 → 0.9812 |
| MLP | 114.38 → 101.12 | −11.6 % | 64.08 → 50.13 | −21.8 % | 0.8778 → 0.9018 | 0.9783 → 0.9783 |
| Approximate SVM | 222.46 → 214.87 | −3.4 % | 142.64 → 144.16 | +1.1 % | 0.5476 → 0.5831 | 0.8625 → 0.8302 |

Derived from CP. Tuning improves mainly the regressor (oracle NSE) and barely changes the classifier of XGBoost and of the MLP. In the approximate SVM, F1 and AUC get worse on tuning (0.9719 → 0.9626): the tuning objective is the end-to-end RMSE, not F1.

### 3.3 Best hyperparameters per family

| Family | Best internal value (RMSE, GroupKFold 5, m³) | Classifier | Regressor |
|---|---|---|---|
| XGBoost | 65.70 | `max_depth` 7, `learning_rate` 0.0228, `n_estimators` 239, `subsample` 0.686 | `max_depth` 10, `learning_rate` 0.121, `n_estimators` 346, `subsample` 0.951 |
| MLP | 77.20 | layers (128, 64), `learning_rate` 1.79e-3, `dropout` 0.229 | layers (64, 32, 16), `learning_rate` 4.29e-3, `dropout` 0.013 |
| Approximate SVM | 202.40 | `gamma` 1.13e-3, `n_components` 100, `alpha` 4.55e-6 | `gamma` 0.0353, `n_components` 303, `alpha` 1.46e-4, `epsilon` 0.308 |

Source: RA `best_value`, `best_classifier_params`, `best_regressor_params` (the fixed parameters of §2 are omitted). Observation (derived from RA and `SEARCH_SPACES`): the chosen SVM classifier is at the lower edge of the space in `n_components` (100) and almost at that of `gamma` (1.13e-3 versus the minimum 1e-3); the XGBoost regressor is at the upper edge of `max_depth` (10). That the optimum touches the edge suggests that the search space could be narrow in those dimensions; widening it was not tested.

**Convergence (CA).** The best trial was number 10 in XGBoost, 11 in the MLP and 17 in the approximate SVM (column `best_so_far`). In XGBoost the first 5 random trials gave 311–331 m³ and from trial 5 onwards 12 of the 15 trials fell below 120 m³ (column `value`).

### 3.4 Duration

| Family | default (s) | tuning (s) | tuned (s) | Processes per fold |
|---|---|---|---|---|
| XGBoost | 227 | 761 | 146 | `--eval-jobs 16` |
| MLP | 805 | 3,103 | 990 | script default (40) |
| Approximate SVM | 315 | 833 | 91 | `--eval-jobs 16` |

Source: ST `duration_s` and `command`; RA `duration_s` gives 756, 3,101 and 829 s for tuning (study time without the stage overhead). The three families ran **at the same time** on the same 64-CPU machine (ST `started_utc`: 02:55 UTC all three), with a different number of processes; the durations are **not comparable across families** and only indicate the order of magnitude of the cost.

### 3.5 Confusion matrices and false negatives in the borderline band

| Model | TN | FP | FN | TP | FN in 1–2 m³ band | FN outside the band | FN with volume ≥ 100 m³ |
|---|---|---|---|---|---|---|---|
| XGBoost tuned | 106,472 | 1,383 | 378 | 45,847 | 333 | 45 | 0 |
| MLP tuned | 106,302 | 1,553 | 473 | 45,752 | 354 | 119 | 4 |
| (XGBoost default) | 106,503 | 1,352 | 373 | 45,852 | 325 | 48 | 0 |
| (MLP default) | 106,340 | 1,515 | 514 | 45,711 | 368 | 146 | 19 |
| (Approximate SVM tuned) | 101,738 | 6,117 | 9,076 | 37,149 | 1,217 | 7,859 | 1,443 |
| (Reference) | 106,556 | 1,299 | 889 | 45,336 | 611 | 278 | 0 |

Source: MC (`tn`, `fp`, `fn`, `tp`) and BL (`fn_en_banda_1_2m3`, `fn_fuera_de_banda`, `fn_con_vol_ge_100m3`). In tuned XGBoost 88 % of the false negatives (333 of 378) are nodes with the minimum volume that the label considers flooding; in the tuned MLP, 75 % (354 of 473), and 4 of its false negatives exceed 100 m³ (derived from BL).

---

## 4. Reading (interpretation)

### 4.1 Ranking of the candidates

- **Tuned XGBoost is first** on the main metric: 69.64 m³, 103.22 m³ (−59.7 %) below the no-learning reference (derived from CP).
- **The tuned MLP is second**: 101.12 m³, 71.75 m³ (−41.5 %) below the reference and 31.47 m³ (+45.2 %) above tuned XGBoost (derived from CP). As a classifier it is equivalent to XGBoost (AUC 0.9994 versus 0.9993; F1 0.9783 versus 0.9812); the difference is in the volume (oracle NSE 0.9018 versus 0.9530).
- **The approximate SVM is third and falls below the no-learning reference**: its tuned pooled RMSE (214.87 m³) exceeds that of the reference by 42.00 m³ (+24.3 %); the default, by 49.59 m³ (+28.7 %) (derived from CP). It is also worse in the per-fold mean RMSE (144.16 versus 98.14 m³) and in the total volume error (−15.8 % versus −3.0 %).

**Presentation rule (author's decision, 2026-10-12, recorded in the design):** XGBoost and the MLP are always presented; the approximate SVM enters as a third candidate with its result, even if that result falls below the reference.

### 4.2 Why the approximate SVM could perform worse (hypotheses not tested here)

None of these causes was isolated experimentally in this run; they are plausible explanations to be verified.

1. **Approximate, low-dimensional kernel.** Nystroem with ≤ 400 components (the chosen classifier uses 100) may not represent well the boundary or the volume surface, which in XGBoost need trees of depth 7–10.
2. **Narrow search space.** Only 3 hyperparameters in the classifier and 4 in the regressor, and the classifier optimum touches the edge of the space (§3.3); with 20 trials, the SVM improved only 3.4 % on tuning.
3. **Platt calibration versus `class_weight`.** The classifier uses *hinge* loss with `class_weight`, but the Platt calibration is fitted on unweighted internal folds and removes almost all the effect of the weighting on the final probability. Its 0.5 threshold is approximately equivalent to the unweighted Bayes decision, whereas XGBoost and the MLP shift their effective threshold with `scale_pos_weight`. **Therefore the recall and F1 of the SVM are not directly comparable** with those of the other two families («Sesgo conocido» (known bias) docstring of `svm_family.py`). The AUC, which does not depend on the threshold, is the fairest classifier comparison: 0.963–0.972 for the SVM versus 0.999 for XGBoost and the MLP (CP `auc_roc_agregado`). Even with that metric, the SVM falls below, and also below the reference (0.9844).
4. **Linear model in the regressor.** The ε-insensitive SVR is linear on the Nystroem components; with highly asymmetric volumes (0 in most nodes, hundreds of m³ in a few) the ε-insensitive loss does not penalize large errors quadratically, unlike the metric.

### 4.3 Where the error is concentrated

| Model | Worst shape | Share of the squared error of the worst shape | Volume error of that shape | Top 3 shapes | Pooled RMSE without the worst shape (m³) | Range of per-shape volume error |
|---|---|---|---|---|---|---|
| XGBoost tuned | `shape_quasi_uniform_block` | 61.0 % | −31.2 % | 75.4 % | 44.06 | −31.2 % to +33.0 % |
| MLP tuned | `shape_quasi_uniform_block` | 73.0 % | −26.3 % | 84.2 % | 53.28 | −26.3 % to +61.6 % |
| XGBoost default (this run) | `shape_quasi_uniform_block` | 73.6 % | −55.9 % | 82.9 % | 58.74 | −55.9 % to +32.3 % |
| XGBoost (report 04, `datos/`) | `shape_quasi_uniform_block` | 71.6 % | −52.1 % | — | 57.32 | −52.1 % to +33.3 % |

Derived from PF: squared error of a shape = `rmse_vol² × n_rows`; share = that value over the sum of the 40 shapes; range of `error_vol_pct` (volume error). The last row comes from `04-informe-consolidado.md` (`datos/por_forma.csv`).

- The worst shape is the same in the two tuned models (and in the default): `shape_quasi_uniform_block`, block-like almost uniform inflow, with the volume underestimated. The tuning of XGBoost lowered its RMSE on that shape from 600.6 to 337.6 m³ (PF `rmse_vol`) and its share of the error from 73.6 % to 61.0 %, but it still dominates.
- The three worst shapes coincide in both tuned models: `shape_quasi_uniform_block`, `shape_twin_plateau_15h` and `shape_sustained_plateau_14h` (in that order in both). They are shapes with sustained or plateau inflow.
- **Reading:** the weakness lies in extrapolating the volume to unseen shapes with prolonged inflow, and it is shared by the two families, not a defect of only one. Without the worst shape, the distance between XGBoost and the MLP is reduced to 9.2 m³ (44.06 versus 53.28; derived), versus 31.5 m³ with it; a good part of the advantage of XGBoost is decided on that shape.

---

## 5. Declared limits and biases

1. **Conditional exploratory estimate.** The hyperparameters were chosen with GroupKFold(5) over **all** the shapes (RA `status = estimacion_exploratoria_condicionada`), so the subsequent LeaveOneShapeOut evaluation is not independent: the figures of the tuned models are optimistic, **equally for the three families**. They serve to compare families, not as a confirmatory estimate of the error on new shapes. A nested tuning (one per held-out shape, ×40) remains as future work.
2. **Small budget.** 20 trials per family (10 random + 10 guided). With more trials or wider spaces the order could change, especially for the SVM (optima at the edge, §3.3) and the MLP (its best trial appeared at 11 and the study was the most expensive).
3. **One run, one seed.** There are no repetitions with other seeds, so there are no intervals of the training variability. For the variability due to which shapes are in the sample, the bootstrap of §5.3 was computed.
4. **Defaults not identical to those of part 1.** `xgboost_default` uses `n_jobs = 1` (fixed in this comparison), and `mlp_default` uses `batch_size = 1024` instead of 256 from `config.yaml`. Therefore `xgboost_default` (112.80 m³ pooled, 58.65 m³ per-fold mean) **does not match** the XGBoost of report 04 (106.21 and 56.81 m³; `datos/comparacion_protocolos.csv`), trained with the defaults of `config.yaml` and the default `n_jobs`. Both figures are correct for their configuration; they must not be mixed in the same table without this note.
5. **Durations not comparable across families** (§3.4): the three ran at the same time on 64 CPUs with a different number of processes.
6. **Pooled AUC.** The AUC gathers the probabilities of the 40 folds, each with its own calibration (in the SVM, with its own Platt fit); the probability scales are not strictly comparable between folds.
7. **Two definitions of RMSE** (§3.1): the pooled one weights each shape by its error and is the one dominated by the worst shape; the per-fold mean weights the shapes equally. The order XGBoost < MLP < reference < SVM holds with both (CP).
8. **A single drainage network.** Nothing above says how the models generalize to another network.
9. **Internal calibration of the SVM** (`CalibratedClassifierCV(cv=3)`): its internal folds do not shuffle and depend on the row order; if the rows are not ordered by shape, the calibration may be optimistic (docstring of `svm_family.py`).

### 5.3 Per-shape bootstrap (interpretation aid)

**Method.** From the OOFs of `xgboost/tuned/oof.parquet` and `mlp/tuned/oof.parquet` and the true volume of the prepared dataset, the sum of squared errors and the number of rows of each of the 40 shapes (folds) were computed. The 40 shapes were resampled with replacement 2,000 times (seed 42), **pairing** both models in each resample, and in each one the pooled RMSE (√(Σ errors² / Σ rows)) of each model and their difference were recomputed. Percentile intervals 2.5–97.5. The computation reproduces the point figures of CP (69.642 and 101.117 m³).

| Quantity | Point estimate (m³) | 95 % bootstrap interval (m³) |
|---|---|---|
| Pooled RMSE, tuned XGBoost | 69.64 | 35.3 – 106.5 |
| Pooled RMSE, tuned MLP | 101.12 | 40.8 – 161.6 |
| Difference MLP − XGBoost | 31.47 | 4.4 – 56.2 |

Source: BS. In 0.15 % of the resamples (3 of 2,000) the difference was ≤ 0.

**Reading.** The intervals of each model are very wide and overlap, because the pooled RMSE depends on whether the shape `shape_quasi_uniform_block` enters the resample or not. The paired difference, in contrast, stays above 0: in almost all combinations of shapes tuned XGBoost has a lower pooled RMSE than the tuned MLP. This interval only reflects the variability due to the choice of shapes; it does not include the variability by seed or the optimism of §5.1, and the 40 shapes of the grid are not a random sample of a population of hydrographs, so it is an interpretation aid, not a formal test.

---

## 6. Text for the thesis

> **Selection of families and comparison protocol.** Three model families were compared for the two-stage surrogate (classification of node flooding and regression of the volume): gradient-boosted trees (XGBoost), a dense neural network (MLP) and a support vector machine with an RBF kernel approximated by Nystroem (at most 400 components), with a linear *hinge*-loss SVM calibrated with Platt as the classifier and a linear ε-insensitive SVR as the regressor; the latter is not an exact-kernel SVM, which is infeasible with 154,080 rows. The three families used the same dataset (`tabular_v4_21` contract, 963 of the 1,000 grid simulations after discarding those with a continuity error above 5 %, a single network), the same evaluation protocol (leaving out one complete hydrograph shape, 40 splits) and the same tuning budget: 20 configurations per family explored with the Optuna TPE sampler (10 random and 10 guided), minimizing the end-to-end volume RMSE with internal cross-validation grouped by shape (five groups). The main metric is the pooled volume RMSE over all nodes and splits, and a no-learning predictor that copies the response of the most similar training shape is included as a reference.
>
> **Results and their scope.** Tuned XGBoost obtained the lowest pooled RMSE (69.64 m³, with a regressor NSE of 0.953 on the flooded nodes), followed by the tuned MLP (101.12 m³; NSE 0.902); both clearly improve on the no-learning reference (172.86 m³) and are equivalent as classifiers (AUC 0.999; F1 0.978–0.981). The approximate SVM fell below the reference (214.87 m³; AUC 0.963), possibly because of the low-dimensional kernel approximation, a reduced search space and a calibration that cancels the class weighting, causes that were not isolated here. A bootstrap resampling of the 40 shapes places the difference between the MLP and XGBoost between 4.4 and 56.2 m³ (95 %), which suggests that the advantage of XGBoost does not depend on a few shapes, although the error of both models is concentrated in the same sustained-inflow shapes: the worst of them accounts for 61 % (XGBoost) and 73 % (MLP) of the squared error. These figures must be read as an exploratory estimate: the hyperparameters were chosen with all the shapes, which favours the three families equally, they come from a single run and seed, and they were obtained on a single network. For this reason XGBoost and the MLP are presented as the main candidates and the approximate SVM as a third candidate with its result; a nested per-shape tuning and repetition with several seeds remain as future work.

---

## 7. Provenance

- **Run commit:** `8a0660589425e59849813b9ca59c468458f9d287` (`8a06605`), `git_dirty: false` (ST `git`, TP `git_commit`).
- **Versions:** Python 3.12.10, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.6.0, optuna 5.0.0 (TP `versions`).
- **Commands executed** (ST `command`), from the repository root:
  - `python scripts/ejecutar_parte2_v4_21.py --family xgboost --eval-jobs 16`
  - `python scripts/ejecutar_parte2_v4_21.py --family mlp`
  - `python scripts/ejecutar_parte2_v4_21.py --family svm --eval-jobs 16`
  - `python scripts/ejecutar_parte2_v4_21.py --build-tables` (tables generated on 2026-10-13 17:19 UTC; TP `generated_at_utc`)
- **Artefacts per stage** (read-only): `outputs/bench/parte2/ca706e27654e0cf5/{xgboost,mlp,svm}/{default,tune,tuned}/` with `stage.json`, `metrics.json` and `oof.parquet` (default, tuned) and the study `tune/<familia>/<prep_id>_<marca>/study.json`. The SHA-256 fingerprints of the OOFs, the studies and the tables are in TP `inputs_sha256` and `outputs_sha256`.
- **Tables:** `docs/auditoria/correcciones/datos_parte2/{comparacion_protocolos,matriz_confusion,banda_limitrofe,por_forma,resumen_afinacion,convergencia_afinacion}.csv` and `tables_provenance.json`.
- **No-learning reference:** `outputs/bench/oof/ca706e27654e0cf5/baseline_forma_mas_cercana.parquet`.
- **Execution logs:** `svm.log`, `xgboost.log` and `mlp.log` of the session (temporary, outside the repository); their final figures match CP. The MLP log contains only Optuna warnings about categorical options with tuples (`hidden_sizes`), which do not affect the search.
- **Bootstrap (§5.3):** disposable script of the session, not versioned; method described in §5.3 and reproducible with the cited OOFs and `load_prepared()` of `scripts/ejecutar_parte2_v4_21.py`.
- **Reproduce:** `venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family <xgboost|mlp|svm>` (stages marked `done` are skipped; delete or use another output root to recompute) and then `venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --build-tables`.
