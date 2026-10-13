# Annex 04 — Closure of previous audits and documentation consistency

**Date:** 2026-10-10
**Audited branch:** `fix/cierre-auditoria-hidrogramas-20260927`, base commit `6e8579b` (2026-09-27), **plus** the uncommitted changes in the working tree (phase A: 14 `M` files). The code evaluated is that of the current working tree.
**Mode:** read-only. No code, existing documentation, `config.yaml` or `outputs/` was modified. No commits were made.

---

## 1. Method

1. **Inventory of previous findings.** The following were read in full: `AUDITORIA_SURROGATE_HIDROGRAMAS.md` (including the section "Seguimiento 2026-09-27" (Follow-up 2026-09-27)), `AUDITORIA_CODIGO_BASURA_2026-06-15.md`, `AVANCES_RECIENTES_PROYECTO.md`, `docs/superpowers/specs/2026-06-12-validation-harness-inference-fixes-design.md`, `docs/revision_academica_redes_neuronales_2026-09-22.md`, `docs/proximos_estudios.md`, `docs/superpowers/plans/2026-09-27-auditoria-fase-a-integridad-protocolo.md` (and its spec `2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md`), `docs/superpowers/plans/2026-09-29-optuna-tuning-familia-ganadora.md` and its spec. Each finding, recommendation, acceptance criterion or pending task received an ID (section 2).
2. **Cross-check against the current code.** The modules involved were read with line numbers (`scenario_predict.py`, `dynamic_features.py`, `labels.py`, `hydrograph_batch.py`, `timeseries_scenario.py`, `static_features.py`, `topology.py`, `contracts.py`, `swmm_api_io.py`, `ml/bench/*`, `config.py`, `config.yaml`, `main.py`). `git log`, `git show`, `git diff` and `git ls-tree` were used.
3. **Test quality (critical reading + mutation).** For each key behavior a **mutation** was applied on a copy of the repository in the scratchpad (never on the repo) and only the focused tests were run. "DETECTED" = some test fails with the mutation; "SURVIVES" = all pass, i.e. the test does not protect that behavior. Baseline of the copy: 160 and 147 tests green (two batches). 26 mutations: **18 detected, 8 survive**. The results are in section 2 and the script in section 5.
4. **Reachability (junk code).** The import graph was rebuilt by AST (section 5, `reachability.py`) from the production entry points (`main.py`, `swmm_resilience/main.py`, `swmm_resilience/desktop/app.py`), the tests and the root scripts. `venv/`, `build/`, `.claude/` (worktree), `data/`, `outputs/` and `docs/` were excluded.
5. **Documentation.** `README.md`, `QUICKSTART.md`, `COMANDOS.md`, `comandos.txt`, `docs/FLUJO_ACTUAL.md`, `DOCUMENTACION_COMPLETA_PROYECTO.md`, `spec_tecnico_desarrollo.md` and `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md` were compared with the code, the real `--help` and the data (`data/`, `outputs/`).
6. **Test count without running the suite:** `pytest --collect-only` gives **881/884 collected (3 deselected)** and **258** in `tests/ml/bench`. The full suite was not run.
7. **CLIs executed:** `python main.py --help` and `python -m swmm_resilience.ml.temporal.train_cnn --help` (they use argparse; read-only). **Incident to be reported:** `python -m swmm_resilience.ml.train --help` was also executed. That module **has no argparse**, so it ignored `--help` and started training. It aborted in `load_dataset` with `FileNotFoundError` because `data/networks/chico_hydro-qx1/results/dataset_ml.csv` does not exist. It was verified afterwards with `git status` and by listing `data/networks/chico_hydro-qx1/` that it wrote no file. The incident became finding T-04.

**Context that conditions the statuses.** A git worktree exists at `.claude/worktrees/optuna-tuning-familia-ganadora` (branch `worktree-optuna-tuning-familia-ganadora`, HEAD `70e3717`, 2026-10-06) with **69 commits not merged** into this branch: phases B1 (inflow closure), B2 (tributary features `tabular_v4_21`), C1 (drain state), C2 (continuity filter), Optuna (`tune.py`, `--bench-tune`) and v4_21 promotion. **The statuses in this annex refer to the audited branch.** When a pending item appears resolved in that branch it is indicated as "in unmerged branch (commit X)". That implementation was **not verified** here: it was only established that the files exist (`git ls-tree`) and the commit messages were read.

**Status criterion.**
- **CLOSED:** code present + a test that **detects** the regression (or execution evidence when the criterion requires it).
- **PARTIAL:** one of those pieces is missing.
- **OPEN:** not implemented in this branch.
- **NOT APPLICABLE:** out of scope, optional or superseded.
- **NOT VERIFIED:** could not be checked without running something prohibited.

---

## 2. Traceability table

Abbreviations: `hb` = `swmm_resilience/validation/hydrograph_batch.py`; `sp` = `swmm_resilience/ml/scenario_predict.py`; `ts` = `swmm_resilience/simulation/timeseries_scenario.py`; `wt` = uncommitted change in the working tree; "mut:" = mutation result.

### 2.1 `AUDITORIA_SURROGATE_HIDROGRAMAS.md` — findings

| ID | Summary | Status | Evidence (code/commit/test) | What is missing |
|---|---|---|---|---|
| C1 | Features blind to shape (peak only) | PARTIAL | `ml/contracts.py:32-50`: 17 features; the shape only enters via `duracion_horas`/`tiempo_al_pico_h` (per-shape scalars). `extraction/event_features.py` does not exist. The `tabular_v4_21` contract (6 tributary features) is only in the unmerged branch (`587b704`, `2236c9a`, `976f45f`) | Integrate the contract with volume/exceedance, ablation 17 vs new and per-shape sensitivity (spec 09-27 §13) |
| C2 | One degree of freedom; optimistic validation by factor | PARTIAL | wt: `ml/bench/schemas.py:56-60` (`LeaveOneShapeOut→shape_id`), `folds.py:54-98`, `config.yaml:48`, `config.py:247`. Test `tests/ml/bench/test_folds.py:37`; mut `loso_shape_groups_by_factor` DETECTED. The PreparedDataset `outputs/bench/prepared/af3751551ca5df98` exists (160,000 rows, `protocol_definitions`). Dataset: 39 CSV in `data/hydrograph_shapes` + base shape = 40 shapes | No commit. **No per-shape OOF/ranking executed** in this branch (`outputs/bench/reports/ranking.csv` = SVM only, LOSO by factor, 2026-09-07, 76k rows). The legacy evaluator (`config.yaml:35-39`) and `--persist-sql` (`main.py:899`) still group by factor |
| M1 | Inference features inconsistent with training | PARTIAL | Actual peaks + accumulation by ancestors: `extraction/dynamic_features.py:32-64`; test `tests/test_scenario_predict.py:127`; mut `accum_without_ancestors` DETECTED. Average factor removed | `tiempo_al_pico_h` is taken from the **first non-empty series** (`sp:118-119`); mut `tpeak_from_last_series` **SURVIVES**. `duracion_horas = last_time_hours` of the CSV vs the duration of the normalized shape in training (`simulation/hydrograph_shapes.py:35-36`). Inference does **not** go through `TABULAR_V3_17.validate_frame` (`sp:136`, unlike `ml/trainer.py:64`). No training↔inference parity test |
| M2 | Silent extrapolation | PARTIAL | `sp:75-95` (`extrapolated`), test `tests/test_scenario_predict.py:178`; mut `extrapolation_never_flags` DETECTED | Branch for nodes with base 0 without coverage (mut `extrap_base0_never_flags` **SURVIVES**). There is no out-of-training-shape flag in the predictor; only the INTERPOLATION/EXTRAPOLATION label by duration in `--evaluate-validation-shapes` (`main.py:~700-710`) |
| M3 | Truncated simulation (no drain) | OPEN | `ts:120-123` adds **a single** zero point at `last+6h`, which interpolates an artificial linear inflow tail. The training shapes use the same writer (`simulation/batch.py:60-62`). Shapes that do not end at zero: `long_moderate` (final q 0.35), `shape_2h` (0.0747), `shape_5h` (0.0744). mut `drain_zero_point_off`/`drain_end_time_off` DETECTED, but mut `batch_ignores_drain_down_param` (not propagating `drain_down_hours` from the batch) **SURVIVES** | Inflow closure policy, drain verification from `.out` and regeneration of the affected SWMM ground truth. In unmerged branch: `92d2c23`/`53de401` (B1), `6c0ac1a`/`95d888a` (C1) |
| M4 | Only nodes with direct inflow | PARTIAL | Correct logic: `hb:283-284` (`all_node_ids = predictor.node_ids`), `hb:196-209`; helper test `tests/test_hydrograph_batch.py:394` | The wiring is not protected: mut `truth_only_inflow_nodes_M4` (use `expected_nodes`) **SURVIVES** because the fake predictor has the same nodes as the CSV (`tests/test_hydrograph_batch.py:19,51`). The audit asked to "keep an automated test" |
| M5 | `>` vs `>=` and different threshold values | CLOSED | `extraction/labels.py:19-23` (`>=`), `config.yaml:14` (1.0), CLI fallback 1.0 (`main.py:771-778`). wt `ml/bench/quality.py:95` uses `flood_label`. Tests `tests/test_labels.py:33`, `tests/ml/bench/test_quality.py:106`; mut `labels_ge_to_gt`, `quality_revert_gt`, `label_zero_threshold_counts_dry` DETECTED | The classifier/regressor reconciliation is handled in m1 |
| MO1 | `q_pico_acum_base` assumes simultaneous peaks; capacity only immediate | OPEN | No changes: `extraction/topology.py:84-91`. The 09-27 spec §10 accepts as closure **documenting** the limitation, and it is not documented (neither in `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md` nor in `README.md`) | Document the proxy and its limitation; downstream capacity is postponed |
| MO2 | Metrics that mask errors | PARTIAL | Validation harness: CSI, PR-AUC, conditional MAE/RMSE and per-scenario metrics (`hb:360-385, 418-428`); test `tests/test_hydrograph_batch.py:297,312` | The bench (`ml/bench/metrics.py:36-70`, `evaluate.py:85-122`) has no CSI, PR-AUC, per-shape or per-scenario breakdown (only `by_factor`); relative error per flooded node is missing |
| MO3 | No unit validation | PARTIAL | `hb:102-107` aborts if `FLOW_UNITS≠LPS`; test `tests/test_hydrograph_batch.py:339`; mut `flow_units_check_off` DETECTED. The base INP declares `FLOW_UNITS LPS` (line 6) | If `FLOW_UNITS` is **missing**, it passes silently (the SWMM default is CFS). The check only exists in the harness, not in the training extraction. The ×1000 conversion of the `swmm_api` route (`simulation/swmm_api_io.py:303-307`) is not tested (mut `swmm_api_rpt_scale_wrong` **SURVIVES**). Units are not recorded in any manifest |
| MO4 | `base_inflow_lps` is a peak | OPEN | `extraction/static_features.py:7-26,110` takes the `max()`; `contracts.py:41` describes it as "Persisted baseline node inflow"; `DICCIONARIO…:71` says "baseline per node" | Document it as a peak (or rename it) |
| MO5 | Total volume per scenario not reported | CLOSED | `hb:366-385, 459-470` (`scenario_totals.csv`, `totals_comparison.png`); console `main.py:827-837`; test `tests/test_hydrograph_batch.py:253`; mut `error_pct_sign` DETECTED; real outputs in `validation_output/*/scenario_totals.csv` (2026-09-15) | — (the PNG is written at the root of `out_dir`, not in `plots/`: see D-04) |
| MO6 | No SWMM vs ML timings | PARTIAL | `hb:298-312, 387-401`; predictor created only once (`hb:283`); test `tests/test_scenario_predict.py:188` (predictor level) | mut `batch_reloads_predictor_per_scenario` **SURVIVES** (nothing detects that the batch reloads models per scenario). `device` fixed at `"cpu"` (`hb:399`). The measurement must be repeated on the final machine |
| m1 | `inunda_pred=1` with `vol_pred<umbral` (reconciliation) | PARTIAL | Documented rule (spec 06-12 §4e): the classifier decides (`sp:140-146, 176-178`; `README.md:86-87`) | Versioned rule of the 09-27 spec §10 (`inunda_pred = vol_final>=1`) not implemented. mut `regressor_on_all_nodes` **SURVIVES** (`vol=0` is not tested when the class is 0) |
| m2 | Duplicated parsing of the Node Flooding Summary | CLOSED | Single source: `extraction/labels.py:93-139`. Used by `hb:206` and `main.py:298,331,1019,1045`. Legacy residue: `simulation/runner.py:606` (pipeline A/GUI) | Remove the residue together with pipeline A |
| m3 | Parser takes the last token (ponding) | PARTIAL | Text: `labels.py:79` (last token). swmm_api: `swmm_api_io.py:293-296` maps any column with "volume" (ambiguous with ponding). Ponding warning only in the harness (`hb:109-116`, mut DETECTED). Base INP with `ALLOW_PONDING NO` | Header-driven parser that aborts on ambiguity (09-27 spec §10) |
| m4 | Continuity errors not recorded | PARTIAL | `hb:175-193, 314-321, 383` (`continuity_error_pct` column); parsing test `tests/test_hydrograph_batch.py:378` | The warning is not tested (mut `continuity_warning_off` **SURVIVES**). The training extraction neither measures nor excludes by continuity (in unmerged branch: `30b1110`, `2bb77fc`). Evidence: 8/24 ultrafast scenarios >5% in `validation_output/val_extrap_ultrafast_20min/metrics_per_scenario.csv` |
| m5 | INP hash is not verified | CLOSED | `hb:118-137` + `--allow-inp-mismatch`; test `tests/test_hydrograph_batch.py:350`; mut `inp_hash_check_off` DETECTED; `outputs/models/training_inp_hash.txt` = `19a8768c…` = MD5 of the INP of `config.yaml:2` | Nuance: MD5 (not SHA-256) and, if the hash file is missing, it only emits a warning (`hb:119-125`) |

### 2.2 `AUDITORIA_SURROGATE…` — strengths, recommendations and follow-up

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| F1 | The base INP is never modified | CLOSED | `ts:89,134-135`; test `tests/test_timeseries_scenario.py:141` | — |
| F2 | Strict validation of the hydrograph CSV | CLOSED | `validation/hydrograph_csv.py:42-80` (exact columns, numeric, finite, non-negative) | The "9 rules" figure was not recounted (not verified) |
| F3 | `log1p`/`expm1` regressor + clip ≥ 0 | CLOSED | `sp:145-146`; `ml/bench/evaluate.py:169`; test `tests/test_scenario_predict.py:159` | — |
| F4 | 3-level evaluation + OOF `log_nse` | CLOSED | `ml/bench/metrics.py:36-70`; `evaluate.py:118-123` | — |
| F5 | 10⁶ L → m³ conversion documented | CLOSED | `labels.py:30-31,86`; `swmm_api_io.py:304` | See MO3: the swmm_api route has no test |
| R1 | (C1/C2) Event descriptors | PARTIAL | = C1 | = C1 |
| R2 | (C2) Event diversity + LOSO by shape | PARTIAL | 40 shapes × 25 factors × 160 nodes = 160,000 rows (`data/training/dataset_final.csv`); LeaveOneShapeOut in wt | Run and report the per-shape protocol |
| R3 | (M1) Fix the inference | PARTIAL | = M1 | = M1 |
| R4 | (M3) Drain cushion + "ends at ~0" rule | OPEN | = M3 (the warning does exist: `ts:94-109`, mut DETECTED) | = M3 |
| R5 | (M5/MO3) Single threshold and `FLOW_UNITS` | PARTIAL | M5 CLOSED; MO3 PARTIAL | = MO3 |
| R6 | (M2) Report extrapolation | PARTIAL | = M2 | = M2 |
| R7 | (MO2) CSI, PR-AUC, conditionals, per scenario | PARTIAL | = MO2 | = MO2 |
| R8 | (MO5) Totals per scenario | CLOSED | = MO5 | — |
| R9 | (MO6) Timings per scenario, single load | PARTIAL | = MO6 | = MO6 |
| R10 | (Strategic) Temporal CNN/LSTM scaffold | NOT APPLICABLE | The 09-27 spec §4/§12 makes it conditional on a measured residual gap | — |
| S1 | Implement Leave-One-Shape-Out (priority 1) | PARTIAL | = C2 | Per-shape table (mean, dispersion, result per held-out shape) |
| S2 | Event features (priority 2) | OPEN | = C1 in this branch (v4_21 in unmerged branch) | Integrate and retrain under LeaveOneShapeOut |
| S3 | Drain and traceability (priority 3) | OPEN | `outputs/models/promotion.json`: `classifier_sha256 = e2e7925e…`, prep `4a48359e7e5b53c6`, 2026-09-07. Actual SHA-256 of `outputs/models/classifier.joblib` = `660be918…` (matches the `HEAD` tree, commit `6e8579b`). The outputs in `validation_output/` (2026-09-15) do not record model, dataset or protocol hashes | Coherent promotion manifest; hashes in every validation result |
| S4 | Decide neural network (priority 4) | OPEN | Depends on S1/S2/S5 | — |
| S5 | Re-evaluate the 5 families with per-shape folds | OPEN | No execution in this branch (see C2). The unmerged branch declares a "5-family benchmark" (`51c923d`), not verified | Run with identical folds |
| OPT-1 | Optuna not implemented | OPEN | There is no `ml/bench/tune.py`; `requirements.txt` without `optuna`; `--help` without `--bench-tune` | See OP-1…OP-9 |
| OPT-2 | Naive Bayes not integrated | NOT APPLICABLE | Optional and low priority; the registry only has 5 families (`ml/bench/registry.py:36-37`) | — |
| OPT-3 | Tuning only inside the train of each per-shape fold | OPEN | Not implemented; moreover the 09-29 spec §10 deviates from this criterion (T-12) | Nested selection or shape holdout |
| FP-1 | Select the candidate family with per-shape OOF | OPEN | No per-shape OOF | — |
| FP-2 | Run Optuna on the winner | OPEN | = OPT-1 | — |
| FP-3 | Retrain and compare with the same folds | OPEN | — | — |
| FP-4 | Naive Bayes as an optional baseline | NOT APPLICABLE | = OPT-2 | — |

### 2.3 `AUDITORIA_CODIGO_BASURA_2026-06-15.md`

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| B0 | "0 orphan package modules" | CLOSED (still valid) | `reachability.py`: 99 package modules; 92 reachable from production; **0 unreachable** from any entry point. Only via tests: `database.recovery`, `database.upgrade`, `database.workflow_lock` (API documented in `COMANDOS.md:284-288`), `ml.temporal.compare_surrogate`, `train_cnn`, `train_surrogate` (they have their own `__main__`) | — |
| B1 | Delete `pruebas_locales.py` | CLOSED | Deleted in `8195315` (2026-06-24) | — |
| B2 | Delete `verificar.py` | CLOSED | Deleted in `8195315` | — |
| B3 | Delete `halve_timeseries.py` (overwrites the production INP) | OPEN | Still versioned (last commit `46c5152`); `DST` = the same INP as `config.yaml:2`; nobody imports it (`reachability`: imported by []) | `git rm` (the overwrite risk remains) |
| B4 | Delete `outputs/factor_comparisons/` | CLOSED | Does not exist | — |
| B5 | Caches `__pycache__/`, `.pytest_cache/` | NOT APPLICABLE | They were regenerated (132 `__pycache__` outside `venv`, `.pytest_cache/`); they are ignored | — |
| BR1 | Retirement of pipeline A (separate decision) | OPEN | `ml/train.py`, `predict_from_inp.py`, `predict_tabular.py`, `preprocessing.py` and `analysis/eda.py` are only reachable from `desktop/app.py` and `swmm_resilience/main.py`. It does not work without regenerating data: `data/networks/chico_hydro-qx1/results/dataset_ml.csv` (`config.py:43`) and `data/training/swmm_resilience.db` (`config.py:42`) are missing. **Correction:** `ml/scenario_predict.py` is **not** pipeline A; the active validation uses it (`main.py → hb:34`), see T-11 | Run the plan `2026-08-21-sqlite-v17-legacy-retirement` excluding `scenario_predict.py` |
| BR2 | `validation_output_smoke/` (fixtures?) | OPEN | `validation_output_smoke/inp/hydrograph_{2h,5h}.inp` are byte-for-byte identical to `validation_output/inp/…` (`git diff --no-index` empty). No test or module references them. Their source CSVs (`data/hydrographs/`) were deleted in `3e17363` | They can be retired; see T-08 |
| BR3 | Content of `outputs/` | NOT APPLICABLE | Ignored local artifacts; `outputs/new_csv_comparison/` no longer exists | — |
| BR4 | Keep `comandos.txt` | NOT APPLICABLE | It is kept, but it is outdated (D-15) | Update or merge with `COMANDOS.md` |
| BR5 | `.md` docs out of scope | NOT APPLICABLE | They are covered in section 4 of this annex | — |
| BR6 | Intra-file dead code not audited | OPEN | Not audited in this annex (not verified) | Dedicated review with coverage |

### 2.4 `AVANCES_RECIENTES_PROYECTO.md` (pending items §17)

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| AV-1 | Validate temporal Phase 1 (Parquet per run) | NOT VERIFIED | There is no `data/networks/*/results/temporal/` on disk | — |
| AV-2 | `temporal_artifacts` table | CLOSED | `database/schema.py:113`; `tests/database/test_temporal_artifacts.py` (legacy schema, not v17) | — |
| AV-3 | Temporal windows | CLOSED | `ml/temporal/dataset.py:103` (`build_temporal_windows`); `tests/ml/temporal/test_window_builder.py` | — |
| AV-4 | Review the precision/recall trade-off of `svc_rbf` | NOT APPLICABLE | Legacy pipeline metrics, already obsolete | — |

### 2.5 Spec `2026-06-12-validation-harness-inference-fixes-design.md`

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| SP-1 | Single labeling rule ≥ 1 m³ | CLOSED | = M5, m2 | — |
| SP-2 | Contract v2 (15 features, without `factor_mult`) | CLOSED (superseded) | `factor_mult` is not in `contracts.py:32-50`; the current contract is `tabular_v3_17` (17 features, spec 06-26) | Update the docs that say 15 (D-01, D-22) |
| SP-3 | `ScenarioPredictor` (single load, all junctions, prob, extrapolated) | CLOSED | `sp:41-163`; tests `tests/test_scenario_predict.py:144-205` | See M2/MO6 |
| SP-4a | Post-event drain | PARTIAL | Implemented as specified (`ts:120-131`), but the design itself generates an artificial tail (M3) | = M3 |
| SP-4b | Guards (units, ponding, hash, continuity) | PARTIAL | `hb:84-137, 314-321` | `FLOW_UNITS` missing; continuity warning without test |
| SP-4c | Coverage of all junctions | PARTIAL | = M4 | Wiring test |
| SP-4d | New outputs (totals, timings, metrics_per_scenario, PNG, console) | CLOSED | `hb:456-477`; `main.py:804-861` | PNG path ≠ spec (D-04) |
| SP-4e | Reconciliation rule documented and tested | PARTIAL | Documented (`README.md:86-87`) | Not tested (mut `regressor_on_all_nodes` survives) |
| SP-5 | Relabel, retrain and update README/DICCIONARIO | PARTIAL | Active models with 17 features (`feature_names_in_` verified), INP hash matches | `DICCIONARIO…` not updated (D-22…D-24); `README.md` says 15 features |
| AC-1 | `pytest tests` green | NOT VERIFIED | 881 collected; `.superpowers/sdd/…/task-5-report.md` reports 881 passed; here only 160+147 focused tests were run (green) | — |
| AC-2 | `--evaluate-hydrographs` run with the 3 CSVs + PNG + speed-up | CLOSED | `validation_output/val_*/{scenario_totals,timings,metrics_per_scenario}.csv` (2026-09-15). Follow-up figures verified: marathon f2.5 CSI 0.941, 28,227 vs 23,801 m³ (−15.7%); ultrafast f3.7 CSI 0.889, 829 vs 623 m³ (−24.9%); speed-up 29.3–50.8× and 8.7–11.2× | — |
| AC-3 | Model and static feature loading once per batch | CLOSED | `hb:283`, constant `t_model_load_s` in `timings.csv` | Batch-level test (MO6) |
| AC-4 | Same labeling function in training and validation | CLOSED | = m2 | — |
| AC-5 | Dataset regenerated, models retrained, hash verified | CLOSED | `training_inp_hash.txt` = MD5 of the INP; models with 17 features | — |

### 2.6 Spec `2026-09-27-cierre-auditoria-surrogate-hidrogramas-design.md` (phases and criteria not covered above)

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| SC-A | Phase A: inclusive rule, LeaveOneShapeOut, SHA-256 provenance, cache rejection | PARTIAL | wt: `schemas.py:53-170` (semantic validation, `prep_id`), `preprocess.py:137-230` (read-only SQLite backup, `training_frame_sha256`); mut `prepared_semantics_check_off` and `held_out_groups_empty` DETECTED | §5 requires commit + code snapshot + diff hash + config/seed hashes: the `af37…` manifest does **not** have them. Exclusion registry nonexistent. **No commit** in this branch |
| SC-B | Phase B: inflow closure policy and tributary features | OPEN | Absent here; in unmerged branch (B1/B2) | Integrate |
| SC-C | Phase C: `drained/not_drained/unverified` state; continuity > 5% excluded | OPEN | Absent here; in unmerged branch (C1/C2) | Integrate |
| SC-D | Phase D: fair benchmark (CSI, PR-AUC, per-shape macro, persisted OOF, timings) | OPEN | `main.py:194-211` discards the OOF (`_, metrics = …`); `reports.py` only writes metrics and ranking; the ranking uses f1/nse as tiebreaker (`config.yaml:49-54`) | Implement per §8 |
| SC-9 | Explicit, transactional promotion with guardrails | PARTIAL | Not implicit: `main.py:213-223`; mut `bench_promotes_implicitly`/`promote_accepts_none` DETECTED | `ml/bench/promote.py:29-50` copies without rollback, without verifying hashes or guardrails |
| SC-10 | Versioned reconciliation; `temporal_descriptor_out_of_range`; `shape_status` | PARTIAL | Only the magnitude flag (= M2) | The rest absent |
| SC-11 | Staged execution (pilot, checkpoints, SVM timeout) | OPEN | Nothing in `ml/bench/` | — |
| SC-12 | Nested Optuna without outer test | OPEN | = OPT-3 | — |
| SC-2 | Record the hash/date inconsistency without rewriting it | CLOSED | Recorded in `AUDITORIA_SURROGATE_HIDROGRAMAS.md:315`; `promotion.json` untouched | Reconcile (S3) |
| SC-3a | Do not run the 08-04 Optuna spec as is | CLOSED | No trace of execution; it is replaced by the 09-29 spec | — |
| SC-3b | Unmeasured timings → null / `not_measured` | OPEN | `database/csv_backfill.py:291-295` inserts `fit_seconds=0`, `predict_seconds=0` and `hyperparameters_json='{}'`; the schema requires `NOT NULL` (`sql/001_v17_initial.sql:208-209`) | Migration + real values |

### 2.7 Plan `2026-09-27-auditoria-fase-a-integridad-protocolo.md`

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| FA-1 | Protocol catalog and per-shape folds | CLOSED (uncommitted) | wt `schemas.py:53-60,137-143`, `folds.py:54-98`; `test_folds.py:37,136` | Commit |
| FA-2 | Reproducible preparation and rejection of incompatible cache | CLOSED (uncommitted) | wt `preprocess.py:162-282`; `resolve_prepared('…/4a48359e7e5b53c6')` is rejected (task-5 report); mut DETECTED | Commit |
| FA-3 | Inclusive threshold audit in `quality.py` | CLOSED (uncommitted) | wt `quality.py:18,95`; `test_quality.py:106`; mut DETECTED | Commit |
| FA-4 | Promotion disabled; LeaveOneShapeOut main | CLOSED (uncommitted) | wt `config.yaml:48,55`, `config.py:247,343`, `main.py:151,213-223` | Commit. Help/docs outdated (D-08, D-09) |
| FA-5 | Phase verification | CLOSED | 258 tests in bench and 881 in total collected (match the report); `af3751551ca5df98` exists with `protocol_definitions` | The plan's command `--config config.yaml` (line 297) does not exist in the CLI (T-15) |

### 2.8 Plan and spec `2026-09-29-optuna-tuning-familia-ganadora*`

| ID | Summary | Status | Evidence (this branch) | In unmerged branch |
|---|---|---|---|---|
| OP-1 | `optuna` dependency pinned | OPEN | `requirements.txt` without optuna (although `optuna` **is installed** in `venv`: T-10) | `618342b` (`optuna==5.0.0`) |
| OP-2 | Fixture `sql_training_db_six_shapes` | OPEN | `grep six_shapes tests/` empty | `da016a5` |
| OP-3 | Extract `_fit_predict_fold()` | OPEN | Absent in `ml/bench/evaluate.py` | `ee8248a`, `d2ec61a` |
| OP-4 | `build_tuning_folds()` | OPEN | There is no `tune.py` | `5b56a3b` |
| OP-5 | `SEARCH_SPACES`/`_sample_params` | OPEN | — | `e82ca82` |
| OP-6 | `run_study`/`TuneResult`/`save_tune_result` | OPEN | — | `ae37188`, `11ffd13` |
| OP-7 | `BenchTuningConfig` | OPEN | Absent in `config.py` | `08b452a` |
| OP-8 | CLI `--bench-tune` | OPEN | `main.py --help` does not list it | `f4b2fde` |
| OP-9 | Closure verification | OPEN | — | not verified |

### 2.9 `docs/revision_academica_redes_neuronales_2026-09-22.md` and `docs/proximos_estudios.md`

| ID | Summary | Status | Evidence | What is missing |
|---|---|---|---|---|
| RA-1 | MLP: fixed epochs, no weight selection/early stopping | OPEN | `ml/bench/models/mlp_family.py:81` (`for _ in range(self.epochs)`) | Inner validation + early stopping |
| RA-2 | MLP on CPU only (GPU available) | OPEN | No `device`/`cuda` in `mlp_family.py` | — |
| RA-3 | Splits by factor ≠ shape generalization | PARTIAL | = C2 | — |
| RA-4 | MLP search space (1–3 layers, weight decay, batch, Huber) | OPEN | No tuning; the 09-29 spec §6 does not include weight decay, batch, loss or early stopping | — |
| RA-5 | Tuning (including the classifier threshold) with grouped inner validation | OPEN | The threshold is the implicit 0.5 of `predict` | — |
| RA-6 | Extend shape features | OPEN | = C1/S2 | — |
| RA-7 | Conditional regressor on flooded train rows; E2E on all nodes | CLOSED | `ml/bench/evaluate.py:166-169`; `evaluate.py:101-102` | — |
| RA-8 | Metrics: RMSE in m³, CSI, PR-AUC, MAE, bias per event, per family | PARTIAL | The bench has P/R/F1/AUC-ROC/E2E RMSE | CSI, PR-AUC, E2E MAE, bias, per shape |
| RA-9 | Repeat with several seeds | OPEN | Single seed (`mlp_family.py:67-68`) | — |
| RA-10 | Calibration if classes are weighted | OPEN | `scale_pos_weight` is used (`evaluate.py:159`), without calibration | — |
| RA-11 | Learning curves by events/families | OPEN | — | — |
| RA-12 | Persist model + transformers + units + contract; timings | PARTIAL | joblib pipelines with imputer/scaler; `contract_id` in the bench manifest; timings in the harness | Units not persisted |
| RA-13 | Multi-network transfer | NOT APPLICABLE | Later stage; there is a single network | — |
| RA-14 | GNN as a second line | NOT APPLICABLE | Future | — |
| RA-15 | Verify the full text of the ASCE and the Zhang preprint | NOT APPLICABLE | Bibliographic pending item, not code | — |
| PE-1 | GLUE (SWMM parametric uncertainty) | NOT APPLICABLE | The document itself says "does not imply commitment" | — |
| PE-2 | Surrogate uncertainty (bootstrap, conformal, quantile) | NOT APPLICABLE | Future; nothing implemented | — |

**Count (117 items):** CLOSED **32** · PARTIAL **30** · OPEN **40** · NOT APPLICABLE **13** · NOT VERIFIED **2**.
Only the 18 technical findings of the surrogate audit (C1…m5): CLOSED 4 · PARTIAL 11 · OPEN 3.

**Summary of mutations** (those that survive = test gaps): `tpeak_from_last_series` (M1), `extrap_base0_never_flags` (M2), `batch_ignores_drain_down_param` (M3), `truth_only_inflow_nodes_M4` (M4), `batch_reloads_predictor_per_scenario` (MO6), `regressor_on_all_nodes` (m1), `continuity_warning_off` (m4), `swmm_api_rpt_scale_wrong` (MO3/F5).

---

## 3. New findings (documentation, orphan code and governance)

| ID | Sev. | Finding | Location | Recommendation |
|---|---|---|---|---|
| T-01 | 🟠 | Phase A exists **only as uncommitted changes** in this branch (14 `M` files, +1,148/−119). In the unmerged branch it is committed (`2de534d`). There is a risk of loss and divergence | `git status`; `swmm_resilience/ml/bench/{schemas,folds,preprocess,quality}.py`, `config.yaml`, `main.py` | Commit or merge the branch before continuing; then re-run this annex |
| T-02 | 🟠 | There is a worktree with **69 unmerged commits** (B1, B2, C1, C2, Optuna, v4_21, v4_21 promotion). The documentation and the audit closure of this branch do not reflect it. The untracked document `docs/justificacion-features-tributarias-tabular-v4-21.md` describes a contract (`tabular_v4_21`) that **does not exist** in this branch | `.claude/worktrees/optuna-tuning-familia-ganadora` (HEAD `70e3717`) | Decide the merge; meanwhile, mark the v4_21 document as "belongs to branch X" |
| T-03 | 🟠 | Eight behaviors closed according to the audit **have no test that detects their regression** (see summary of mutations) | `tests/test_hydrograph_batch.py:19-51` (the fake predictor has the same nodes as the CSV and a fake `write_scenario_inp` with default 6.0), `tests/test_scenario_predict.py` | Add: `tiempo_al_pico_h` parity, node without inflow in the batch, count of predictor constructions, propagation of `drain_down_hours`, base 0 with peak>0, `vol=0` when the class is 0, continuity warning, ×1000 scale of the swmm_api route |
| T-04 | 🟠 | `python -m swmm_resilience.ml.train --help` **runs the training** (it has no argparse) and would write legacy artifacts if the CSV existed. `README.md:184` and `DOCUMENTACION_COMPLETA_PROYECTO.md:833` recommend it. `swmm_resilience/main.py` (`:318`) does not have argparse either | `swmm_resilience/ml/train.py:1062-1116` | Add argparse or retire it with pipeline A; remove it from the documentation |
| T-05 | 🟠 | If `FLOW_UNITS` is missing from the INP, the guard passes silently (the SWMM default is CFS) | `hb:102-107` | Treat `None` as an error or assume CFS and abort |
| T-06 | 🟠 | Inference does not validate the feature contract: training uses `TABULAR_V3_17.validate_frame` and inference indexes columns directly | `sp:136`; `ml/predict.py` (no validator); compare `ml/trainer.py:64` | Pass `X` through the same validator (the unmerged branch does so in `70e3717`) |
| T-07 | 🟡 | **No main document mentions `--bench*` or LeaveOneShapeOut** (0 matches in README, COMANDOS, FLUJO_ACTUAL, QUICKSTART, DOCUMENTACION_COMPLETA, spec_tecnico, comandos.txt) | see section 4 | Add a "Model bench" section to `COMANDOS.md` and `README.md` |
| T-08 | 🔵 | `validation_output/inp/*.inp` and `validation_output_smoke/inp/*.inp`: versioned, identical to each other, unreferenced and with deleted sources. `validation_output/` is in `.gitignore` but contains versioned files | `.gitignore` (rule "validation_output/"); commits `c703383`, `939c1e9`, `3e17363` | `git rm` both pairs |
| T-09 | 🟡 | `config_7shapes.yaml` is an orphan and obsolete configuration: `promote: "auto"`, protocols without LeaveOneShapeOut, no `db_path`. Its header says "swap this file for config.yaml", which would revert phase A. The unmerged branch removed it (`6f4b202`) | `config_7shapes.yaml:1-4` | Delete it |
| T-10 | 🟡 | Environment drift. venv = Python **3.12.10**; `scripts/reproduce_results.{sh,ps1}` require **3.13** and `.venv`; `pyproject.toml` ≥3.11; spec_tecnico 3.10+; phase A plan "3.11". `optuna` is installed in `venv` but is not in `requirements.txt`; neither is `build` (needed by the packaging tests) | `scripts/reproduce_results.sh:7-17`; `requirements.txt` | Unify the version and pin the real dependencies |
| T-11 | 🟡 | The junk-code audit misclassified `ml/scenario_predict.py` as "Pipeline A". The active validation uses it (`main.py → hb:34`); retiring it together with pipeline A would break `--evaluate-hydrographs/--evaluate-shapes/--evaluate-validation-shapes` | `AUDITORIA_CODIGO_BASURA_2026-06-15.md:52` | Correct the scope of the pipeline A retirement |
| T-12 | 🟡 | The 09-29 Optuna spec §10 adopts **a single GroupKFold(5) study over the whole dataset**. That contradicts the 09-27 spec §12 ("the inner GroupKFold will see exclusively the train of the outer fold"). If the tuned configuration is later compared against the fixed one over the same 40 per-shape folds (09-27 spec §12), the comparison is contaminated. Moreover, the evidence of the 09-29 spec §2 ("optuna not installed") is no longer true | `docs/superpowers/specs/2026-09-29-optuna-tuning-familia-ganadora-design.md:15-27,266-269` | Explicitly forbid that comparison or nest the tuning per outer fold |
| T-13 | 🟡 | 6 `.pptx` presentations (~69 MB) versioned at the root | `Presentacion_MLSMFP_15-09*.pptx` | Move them out of the repo or to Git LFS |
| T-14 | 🔵 | `build/` contains an obsolete copy of the package (`build/lib/swmm_resilience`, with `ml/bench/preprocess.py`) and pytest temporaries from 2026-09-08 (`build/pytest_hydro_*`); it is ignored | `build/` | Clean up |
| T-15 | 🔵 | Execution leftovers: `.pytest-tmp-phase-a/` (9 dirs of `test_bench_config`, 2026-09-27, untracked and **without an ignore rule**) and `tmp20qwka3w/` (empty; the phase A handoff describes it as inaccessible). The phase A plan (line 297) documents an invalid command: `main.py --bench-prepare --config config.yaml` | root; `.superpowers/sdd/…/handoff.md:9` | Delete; add `.pytest-tmp-*/` and `tmp*/` to `.gitignore`; fix the plan |
| T-16 | 🔵 | Inconsistent `.gitignore`: `.claude/` is ignored but `.claude/settings.json` is versioned; whitelist `!data/hydrographs/*.csv` for a nonexistent directory; rule `validation_output/` with versioned files | `.gitignore` | Clean up |
| T-17 | 🔵 | `diagramas_pipeline.html` is versioned, nobody references it and it states "19 shapes" (there are 40 today) | `diagramas_pipeline.html` | Update it or retire it |
| T-18 | 🔵 | `_run_bench` only converts `ValueError` into `SystemExit`. If there is no PreparedDataset, `resolve_prepared` raises `FileNotFoundError` and a traceback is shown | `main.py:175-181`; `ml/bench/preprocess.py:264-280` | Also catch `FileNotFoundError` |
| T-19 | 🔵 | The bench does not persist the OOF predictions or the inference timings (reproducibility requirement of the 09-27 spec §8) | `main.py:198` | Persist OOF per candidate (parquet) together with the `prep_id` |

---

## 4. Documentation ↔ code discrepancies

| ID | Document:line | Says | Reality (code/data) |
|---|---|---|---|
| D-01 | `README.md:77` | "the models use 15 features" | 17 (`ml/contracts.py:32-50`; `feature_names_in_` of the active model = 17) |
| D-02 | `README.md:67-69` | "does not use … SQLite … CSV-based pipeline" | `--only-ml/--skip-extraction/--only-maps` and 6 more flags read `outputs/training_v17.sqlite3` (`main.py:376,394,414,441,482,560,921,1072`); the bench reads SQL |
| D-03 | `README.md:4-5`; `QUICKSTART.md:25`; `docs/FLUJO_ACTUAL.md:145`; `spec_tecnico_desarrollo.md:427,455`; `main.py:9,72` | `--predict` = "inference without SWMM" | It also runs SWMM to verify (`main.py:294-307`) |
| D-04 | `README.md:134`; spec 06-12 §4d | `plots/totals_comparison.png` | `out_dir/totals_comparison.png` (`hb:470`) |
| D-05 | `README.md:111-112` | "aborts if `FLOW_UNITS != LPS`" | It does not abort if the option is missing (T-05) |
| D-06 | `README.md:178`, `:389`, `:412`; `DOCUMENTACION_COMPLETA_PROYECTO.md:106,108`; `AVANCES…:406` | `app.py`, `view_db.py`, `REVISION_DETALLADA_DELTA_Y_ESCENARIOS_PARCIALES.md` | They do not exist |
| D-07 | `README.md:288-291, 362-384` | regression on `peak_flooding_lps`, "replaces flooding_volume_m3" | The active target is `vol_inundacion_m3` (`contracts.py:57-60`); the legacy one uses `total_flood_volume_m3` (`config.py:78`). The README contradicts its own line 73 |
| D-08 | `main.py:125-126` (help) | `--bench`: "Corre las cuatro etapas" (Runs the four stages) | It runs 3; promotion is no longer part of it (`main.py:151,213`) |
| D-09 | `main.py:123-124` (help) | `--bench-promote`: "Copia el candidato ganador" (Copies the winning candidate) | It requires an explicit family in `config.bench.promote` (≠ none/auto); it does not use the ranking (`main.py:214-219`) |
| D-10 | `main.py:107-110` (help) | `--clf-path/--reg-path` "(.pkl)" | The artifacts are `.joblib` |
| D-11 | `main.py:6,8`; help `:64-67`; `QUICKSTART.md:22-24` | `--skip-extraction/--only-ml/--only-maps` read "existing CSV" | They read SQLite (they require a prior `--persist-sql`) |
| D-12 | `COMANDOS.md:30`; `docs/FLUJO_ACTUAL.md:302,373,388` | `pd.read_csv` in `main.py:573` | `main.py:874` |
| D-13 | `COMANDOS.md:187` | shape guard in `main.py:681-687` | `main.py:663-675` |
| D-14 | `COMANDOS.md:86-89`; `docs/FLUJO_ACTUAL.md:152,257,403` | "531 passed" | 881 collected (phase A report: 881 passed) |
| D-15 | `comandos.txt:8-9` | `--skip-extraction` "retrains from the existing CSV" | It reads SQLite |
| D-16 | `COMANDOS.md:237` | `--evaluate-hydrographs data/hydrograph_shapes …` | Those CSVs are `time_h,q_norm`; the loader requires `node_id,time,value_lps` (`validation/hydrograph_csv.py:43`): the command fails |
| D-17 | `COMANDOS.md:333-334` | Optuna "only designed" (spec 08-04) | The 08-04 spec was superseded by the 09-29 one (09-27 spec §3) |
| D-18 | `docs/FLUJO_ACTUAL.md:4` | branch `feature/hydrograph-augmentation-sql-persistence` | Current branch `fix/cierre-auditoria-hidrogramas-20260927` |
| D-19 | `docs/FLUJO_ACTUAL.md:12,45,164`; `config.yaml:60-61` | "≈2 dozen" shapes, "~12 MB", "~25 shapes"; "175 scenarios" | 39 CSV + base = 40 shapes; `dataset_final.csv` = 27.4 MB, 160,000 rows, 24 columns; 1,000 runs |
| D-20 | `docs/FLUJO_ACTUAL.md:300` | `load_training_frame` in `training_queries.py:334` | `:368` |
| D-21 | `docs/FLUJO_ACTUAL.md:95`; `spec_tecnico_desarrollo.md:150` | LOSO evaluation main, by factor | The bench uses LeaveOneShapeOut as the main protocol (`config.yaml:48`, wt) |
| D-22 | `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md:3-7` | "15 features", trains from `dataset_final.csv` | 17 features; reads SQL v17 |
| D-23 | `DICCIONARIO…:13-16, 165-226` | Base `data/training/swmm_resilience.db`, `dataset_ml.csv`; list of legacy features (`inflow_multiplier`, `invert_elev_m`, …) | They do not exist on disk; the active features are those of `contracts.py:32-50` |
| D-24 | `DICCIONARIO…:71` | `base_inflow_lps` = "baseline per node" | It is the **peak** of the base series (`static_features.py:7-26`) |
| D-25 | `DOCUMENTACION_COMPLETA_PROYECTO.md:648` | `ML_TARGET_REGRESSION = "flooding_volume_m3"` | `config.py:78` = `"total_flood_volume_m3"` (legacy); active target `vol_inundacion_m3` |
| D-26 | `DOCUMENTACION_COMPLETA_PROYECTO.md:1-9, 694-740` | Presented as the "master document"; describes PCA as a basis | It describes the legacy pipeline A; PCA only exists in `ml/train.py:243` (GUI) |
| D-27 | `spec_tecnico_desarrollo.md:124, 202, 213` | `flood_threshold_m3: 0.0`; `factor_mult` as a feature; `inunda = vol > threshold` | 1.0; `factor_mult` is metadata; `>=` |
| D-28 | `spec_tecnico_desarrollo.md:34` | Python 3.10+ | `pyproject.toml` ≥3.11; venv 3.12.10 |
| D-29 | `spec_tecnico_desarrollo.md:451` | `--skip-simulation` only "you already have the .rpt" | Rejected without `--skip-extraction/--only-ml` (`main.py:265-266`) |
| D-30 | `spec_tecnico_desarrollo.md:207` | `vol_hidrograma_nodo` removed because of "identical temporal shape" | Today there are 40 distinct shapes: the justification no longer holds (related to C1) |
| D-31 | `README.md:147-158` | `reproduce_results` with Python 3.13 and `.venv` | local venv `venv/` with 3.12.10 (T-10) |
| D-32 | `README.md:41-65` | Architecture: `extraction/assembler.py` | It is in `swmm_resilience/dataset/assembler.py`; `ml/bench/`, `database/`, `validation/` are missing |
| D-33 | `AVANCES_RECIENTES_PROYECTO.md:217-305` | Metrics, PCA, `GroupShuffleSplit` by `run_id` as the current state | They correspond to the legacy pipeline; the active pipeline uses XGBoost without PCA and per-factor/per-shape folds |
| D-34 | `diagramas_pipeline.html` | "19 shapes" | 40 shapes (T-17) |

---

## 5. Reproducible scripts

All are run with `venv\Scripts\python.exe -I` from a directory outside the repo (scratchpad). None writes to the repository.

### 5.1 Reachability by AST — `reachability.py <repo_root>`

```python
import ast, sys
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve()
EXCLUDE = {"venv", ".git", "build", "__pycache__", ".pytest_cache", "swmm_resilience.egg-info",
           ".pytest-tmp-phase-a", "tmp20qwka3w", "data", "outputs", "docs", "obsidian",
           ".superpowers", ".claude"}
def iter_py():
    for p in ROOT.rglob("*.py"):
        if not any(part in EXCLUDE for part in p.relative_to(ROOT).parts):
            yield p
def modname(p):
    parts = list(p.relative_to(ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
files = {modname(p): p for p in iter_py()}
is_pkg = {m: files[m].name == "__init__.py" for m in files}
def resolve(cur, node):
    if isinstance(node, ast.Import):
        return {a.name for a in node.names}
    if node.level:
        base = cur.split(".") if is_pkg.get(cur) else cur.split(".")[:-1]
        if node.level > 1: base = base[: len(base) - (node.level - 1)]
        prefix = ".".join(base); mod = f"{prefix}.{node.module}" if node.module else prefix
    else:
        mod = node.module or ""
    return {mod} | {f"{mod}.{a.name}" for a in node.names}
edges = {}
for m, p in files.items():
    tgt = set()
    for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for cand in resolve(m, n):
                parts = cand.split(".")
                tgt |= {".".join(parts[:i]) for i in range(1, len(parts)+1) if ".".join(parts[:i]) in files}
    edges[m] = tgt
def closure(starts):
    seen, stack = set(), list(starts)
    while stack:
        m = stack.pop()
        if m in seen or m not in files: continue
        seen.add(m); stack.extend(edges.get(m, ()))
    return seen
pkg = sorted(m for m in files if m.startswith("swmm_resilience"))
prod = closure([m for m in ("main", "swmm_resilience.main", "swmm_resilience.desktop.app") if m in files])
tests = closure(list(prod) + [m for m in files if m.startswith("tests") or m == "conftest"])
allr = closure(list(tests) + [m for m in files if not m.startswith(("swmm_resilience", "tests"))])
print("modulos:", len(pkg), "prod:", len([m for m in pkg if m in prod]))
print("solo via tests:", [m for m in pkg if m not in prod and m in tests])
print("inalcanzables:", [m for m in pkg if m not in allr])
```

Result (2026-10-10): 99 modules; 92 reachable from production; only via tests: `database.recovery`, `database.upgrade`, `database.workflow_lock`, `desktop` (artifact: the `__init__` of the package that contains the `desktop.app` entry point), `ml.temporal.compare_surrogate`, `ml.temporal.train_cnn`, `ml.temporal.train_surrogate`; unreachable: none. Root scripts: `generate_obsidian_graph` (imported by `tests/test_obsidian_graph.py`) and `halve_timeseries` (nobody imports it).

### 5.2 Mutations on a copy — `mutation_check.py <repo_root> <work_dir>`

```python
import shutil, subprocess, sys
from pathlib import Path
SRC, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
REPO, PY = WORK / "repo", str(Path(sys.argv[1]) / "venv" / "Scripts" / "python.exe")
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
for d in ("swmm_resilience", "tests"):
    shutil.copytree(SRC / d, REPO / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for f in ("main.py", "conftest.py", "pytest.ini", "config.yaml"):
    shutil.copy2(SRC / f, REPO / f)
MUTATIONS = [  # (id, archivo, original, mutado, tests)
    ("labels_ge_to_gt", "swmm_resilience/extraction/labels.py",
     "return (volumes >= threshold_m3).astype(int)", "return (volumes > threshold_m3).astype(int)",
     ["tests/test_labels.py", "tests/ml/bench/test_quality.py"]),
    # ... (lista completa en la tabla de abajo)
]
def run(tests):
    p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                        "--basetemp", str(WORK / "bt"), *tests], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, ([l for l in p.stdout.splitlines() if l.strip()] or ["-"])[-1]
print("BASELINE", run(sorted({t for m in MUTATIONS for t in m[4]})))
for mid, rel, old, new, tests in MUTATIONS:
    path = REPO / rel; orig = path.read_text(encoding="utf-8")
    assert orig.count(old) == 1, mid
    path.write_text(orig.replace(old, new), encoding="utf-8")
    try: rc, tail = run(tests)
    finally: path.write_text(orig, encoding="utf-8")
    print(mid, "DETECTADA" if rc else "SOBREVIVE", tail)
```

Mutations applied (original → mutated) and result:

| Mutation | File | Change | Tests | Result |
|---|---|---|---|---|
| labels_ge_to_gt | `extraction/labels.py` | `>=` → `>` | test_labels, test_quality | DETECTED |
| quality_revert_gt | `ml/bench/quality.py` | `flood_label(...)` → `(y_reg > t)` | test_quality | DETECTED |
| loso_shape_groups_by_factor | `ml/bench/schemas.py` | `LeaveOneShapeOut: shape_id` → `factor_mult` | test_folds/schemas/preprocess | DETECTED |
| accum_without_ancestors | `extraction/dynamic_features.py` | `ancestors \| {nid}` → `{nid}` | test_scenario_predict | DETECTED |
| extrapolation_never_flags | `ml/scenario_predict.py` | `ratio<fmin or ratio>fmax` → `False` | test_scenario_predict | DETECTED |
| tpeak_from_last_series | `ml/scenario_predict.py` | first series → last series | test_scenario_predict, test_predict_shape_features | **SURVIVES** |
| flow_units_check_off | `hb` | `if flow_units…` → `if False` | test_hydrograph_batch | DETECTED |
| inp_hash_check_off | `hb` | `if stored and current != stored` → `if False` | idem | DETECTED |
| continuity_warning_off | `hb` | `if continuity_pct…` → `if False` | idem | **SURVIVES** |
| truth_only_inflow_nodes_M4 | `hb` | `list(predictor.node_ids)` → `sorted(expected_nodes)` | idem | **SURVIVES** |
| drain_zero_point_off | `ts` | `if drain_down_hours > 0` → `if False` | test_timeseries_scenario | DETECTED |
| drain_end_time_off | `ts` | `+ max(drain_down_hours,0)` → removed | idem | DETECTED |
| bench_promotes_implicitly | `main.py` | `if args.bench_promote` → `… or run_all` | test_bench_cli | DETECTED |
| promote_accepts_none | `main.py` | `if chosen in {"none","auto"}` → `if False` | test_bench_cli | DETECTED |
| batch_reloads_predictor_per_scenario | `hb` | re-create the predictor inside the loop | test_hydrograph_batch | **SURVIVES** |
| batch_ignores_drain_down_param | `hb` | do not pass `drain_down_hours` | idem | **SURVIVES** |
| ponding_warning_off | `hb` | `if ponding…` → `if False` | idem | DETECTED |
| unfinished_hydrograph_warning_off | `ts` | `if peak > 0 and last > 0.01*peak` → `if False` | test_timeseries_scenario | DETECTED |
| extrap_base0_never_flags | `ml/scenario_predict.py` | `flags.append(peak > 0.0)` → `False` | test_scenario_predict | **SURVIVES** |
| regressor_on_all_nodes | `ml/scenario_predict.py` | regressor applied to all nodes | idem | **SURVIVES** |
| prepared_semantics_check_off | `ml/bench/schemas.py` | `if definitions != expected` → `if False` | test_schemas/preprocess/bench_cli | DETECTED |
| held_out_groups_empty | `ml/bench/folds.py` | `held_out_groups` → `[]` | test_folds | DETECTED |
| label_zero_threshold_counts_dry | `extraction/labels.py` | `volumes > 0` → `>= 0` | test_labels | DETECTED |
| text_parser_scale_wrong | `extraction/labels.py` | without `×1000` (text parser) | test_labels | DETECTED |
| swmm_api_rpt_scale_wrong | `simulation/swmm_api_io.py` | `×1000` → `×1` | test_labels, test_flood_volume_extraction, test_hydrograph_batch | **SURVIVES** |
| error_pct_sign | `hb` | inverted sign of `error_pct` | test_hydrograph_batch | DETECTED |

### 5.3 Spot checks (read-only commands)

```bash
# Ítems "seguro de borrar" de la auditoría de basura
git log --oneline --diff-filter=D -- pruebas_locales.py verificar.py   # 8195315
git ls-files halve_timeseries.py validation_output/inp validation_output_smoke/inp
git diff --no-index --stat validation_output/inp/hydrograph_2h.inp validation_output_smoke/inp/hydrograph_2h.inp  # vacío = idénticos
# Rama no integrada
git worktree list; git log --oneline 6e8579b..worktree-optuna-tuning-familia-ganadora | wc -l   # 69
# Trazabilidad de modelos (PowerShell)
Get-FileHash outputs\models\classifier.joblib -Algorithm SHA256   # 660BE918… vs promotion.json e2e7925e…
# Formas que no terminan en cero (>1 % del pico)
for f in data/hydrograph_shapes/*.csv; do tail -1 "$f"; done      # long_moderate 0.350, shape_2h 0.0747, shape_5h 0.0744
# Conteo de tests sin ejecutarlos
venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider tests          # 881/884
venv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider tests/ml/bench # 258
```
