# Decision log and reviews of the audit corrections

Record of the execution process of the plan `docs/superpowers/plans/2026-10-11-correcciones-auditoria.md`
(25+ commits on the base `9ad4df0`). Each task was implemented with a subagent, reviewed (spec compliance and
quality), corrected and reviewed again; at the end there was a review of the whole branch and a final round
of corrections. The decisions taken by the controller on behalf of the author appear as `Ruling: R<n>`
with their reason and what they would cost if they were wrong. Deferred minor findings appear as
`minor (deferred)` or in the review lines of each task.

Closing (verified by the controller): full suite **1,174 passed, 0 failed** (baseline before the series: 1,049),
clean tree, no writes to `outputs/metrics`, `outputs/models` (only `promotion.json` was versioned) or `data/`.
Minors deferred at closing: runner stamping without a test; two tests that promise more than they check;
docstring of `ensure_allowed_output`; `informe_evaluacion --out` without a guard; unvalidated `inf` descriptors;
`que_mide` column of `comparacion_protocolos.csv` with the previous GroupKFold5 text (hash-traced CSV).

Original text of the ledger (unedited):

---

# SDD ledger — plan: docs/superpowers/plans/2026-10-11-correcciones-auditoria.md

Spec: docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md (binding authority)
Worktree: .claude/worktrees/optuna-tuning-familia-ganadora, branch worktree-optuna-tuning-familia-ganadora
Start HEAD: 9ad4df0 (suite baseline 1058 passed, 0 failed @ 1ecca50)

## Pre-flight scan

| Pair / task | Shared file or interface | Produces vs consumes | Finding |
|---|---|---|---|
| T1 -> T2 | metrics.py / evaluate.py | T1 6 pure fns; T2 imports exactly those names | consistent |
| T2 -> T3 | evaluate.py `score_predictions(protocols=)`, tests/ml/bench/prepared_factory.py | T2 creates; T3 consumes | consistent |
| T3 -> T4 | baselines.py (`BASELINES`, `BaselineNotApplicable`, `evaluate_baseline`), schemas.PROTOCOL_INTERPRETATION | T3 defines; T4 imports | consistent |
| T2/T4/T5 | main.py | T4 edits evaluate block of _run_bench; T5 edits start of --evaluate-shapes block | disjoint hunks |
| T4 -> T8 | reports.write_oof layout `<root>/<prep_id>/<name>.parquet` | T8 script reads that layout | consistent |
| T3/T4/T8 | `resolve_prepared(prepared_dir, prep_id, protocols)` | same call as main.py | consistent |
| T7 predict.py | `assert_promotion_consistent` import | promote.py imports `.train` -> registry -> torch at import time | RULING R2 |
| T8 step 4 | outputs/bench/reports, ranking.csv are historical results | plan says run `main.py --bench-evaluate` on real data, which rewrites reports/ranking.csv with xgboost only; and `--bench-prepare` builds contract tabular_v3_17 (17 feat), while promoted model/official comparison use tabular_v4_21 (21 feat, prepared dir with LeaveOneShapeOut only) | RULING R1 |
| T1..T10 each (self-consistency) | tests vs code | spot-checked T1 (volumes/errors hand-recomputed), T2 (fixture folds, FN/FP counts, by_shape 100%, pooled 75%), T3 (interp. extremes, nearest-shape picks) | agree |
| T5 | `plot_flood_volume_curves` name unverified | plan flags it | implementer reads real name |
| T9 | spike | no package changes | consistent with spec 2.9 |

## Rulings

Ruling: R1 — Task 8 does NOT use the CLI on real data. It uses a new script scripts/ejecutar_evaluacion_v4_21.py that (a) prepares tabular_v4_21 with the 3 protocols from config bench.protocols into a NEW dir outputs/bench/prepared_v4_21 (feature_columns=FEATURE_COLUMNS_V21, feature_contract=TABULAR_V4_21, view_name="training_samples_v4_21", node_features_table="node_features_v4_21", max_continuity_error_pct from config.validation), (b) runs evaluate_candidate for xgboost with the config.yaml bench params (untuned defaults) and both baselines, (c) writes OOF to outputs/bench/oof/<prep_id>/ and metrics JSON to docs/auditoria/correcciones/datos/, never touching outputs/bench/reports, ranking.csv or outputs/models. — why: global constraint forbids overwriting historical artifacts; CLI cannot select v4_21 — cost if wrong: numbers describe untuned XGBoost on v4_21, not the Optuna-tuned promoted model; documented as such.
Ruling: R2 — in ml/predict.py import assert_promotion_consistent lazily inside predict_network (not at module top). Functions stay in ml/bench/promote.py per spec 2.8. — why: promote.py imports bench.train -> registry -> torch; a top-level import would load torch for every consumer of predict.py — cost if wrong: first predict call pays torch import time (seconds).
Ruling: R3 — no `git push` during execution. Push to origin/fix/cierre-auditoria-hidrogramas-20260927 is asked of the human at the end. — why: push to a shared branch is outside the worktree and was authorized only for the unification — cost if wrong: remote lags local until the end.

## Tasks
- Task 1: dispatched (BASE 9ad4df0, sonnet implementer)
Ruling: R4 — test baseline corrected to 1049 passed (earlier 1058 counted dots inside warning lines); Task 1 run: 1062 = 1049 + 13 new. Plan/spec text fixed in a docs commit. — why: evidence from progress-line count and implementer's full run — cost if wrong: none (docs only).
- Task 1: implementer DONE (commit b831f0f, 25 passed in test_metrics, suite 1062/0). Pending task review.
Task 1: review: spec OK, quality Approved; 1 Important (plan-mandated): pooled precision/recall/f1 give 0.0 instead of NaN when pool has no (predicted) positives.
Ruling: R5 — fix it (fix round 1): pooled precision/recall/F1 derived from confusion counts with NaN semantics, like fold_classifier_metrics. — why: binding spec says undefined = NaN, never 0 — cost if wrong: none (tiny change).
Task 1: minor (deferred): RMSE computed two ways (np.sqrt(np.mean) vs mean_squared_error); shape_breakdown_entry with empty arrays warns; M-03 contrast not asserted in pooled test (covered later by Task 2 legacy-mean test). nan hoist asked in the fix round.
- Task 1: fix round 1/5 dispatched (implementer resumed)
Task 1: fix round 1/5 (2 addressed, 0 open; commits b831f0f..2871917)
Task 1: complete (commits 9ad4df0..2871917, review clean). Suite after: 1065 passed (short basetemp C:/Users/LF409F~1.FAJ/AppData/Local/Temp/pt1s; long scratchpad basetemp hits WinError 206 on unrelated tests).
- Task 2: implementer DONE (bd0f23f; 20 passed score_predictions, bench 321, suite 1072 + 1 env Tcl failure). Pending task review. Note: spec 2.3 says n_folds_undefined, code emits n_undefined -> fix spec text at end.
Task 2: review: spec OK; 1 Important (plan-mandated): protocols= test vacuous (single-protocol dataset); Minors deferred: routing test does not exercise routing (all clf correct); new parity test name overpromises; duplicated concat/confusion_counts in score_predictions vs pooled_classifier_metrics; `protocols` argument not validated against dataset (typo/bare string); lost docstring rationale in parity test.
Ruling: R6 — fix round 1 covers the Important plus spec wording drift (n_undefined/n_positives) and the E402 import; other Minors deferred to the final review. — why: Task 3 relies on protocols=; spec is the binding authority and must match the code — cost if wrong: none.
- Task 2: fix round 1/5 dispatched (implementer resumed)
Task 2: fix round 1/5 (3 addressed, 0 open; commits bd0f23f..5528479)
Task 2: complete (commits 2871917..5528479, review clean). Suite after: 1073 passed. Consumers of score_predictions checked by reviewer: main.py:207, ranking.py, reports.py, scripts/* — none break.
Task 3: implementer DONE_WITH_CONCERNS (commit 9c8db1b, 11 new tests, suite 1084/0). Concern: FEATURE_DEFINITIONS_V21 (tabular_v4_21) has no duracion_horas/tiempo_al_pico_h -> forma_mas_cercana would KeyError on v4_21 data.
Ruling: R7 — nearest-shape baseline gets optional `shape_descriptors` DataFrame (index shape_id); default derives from X when both columns exist; otherwise BaselineNotApplicable. Task 8 passes descriptors taken from the v3_17 prepared dataset (same 40 shapes). — why: official contract for the promoted model is v4_21, and the baseline must run on it — cost if wrong: baseline compares against shapes by (duration, time-to-peak) only; a different notion of "closest shape" would change that reference number.
- Task 3: addition dispatched before review (implementer resumed), BASE for review stays 5528479
- Task 3: implementer DONE after R7 addition (commits 9c8db1b, 02cdaab; baselines 17 passed, suite 1089 + 1 env Tcl flake). Pending task review.
Task 3: review: spec OK; 1 Important: explicit shape_descriptors accepts NaN/duplicate shapes -> silently wrong neighbours (probe-confirmed). Minors deferred: interpolation tests only exercise midpoints (weight 0.5; no uneven grid/GroupKFold5 test); no test catches standardizing with test-shape statistics (leak guard untested); pivot duplicate error is pandas' raw message; NaN train volume reported as "series without all factors"; weak match="C" assertion.
- Task 3: fix round 1/5 dispatched (implementer resumed)
Task 3: fix round 1/5 (1 addressed, 0 open; commits 02cdaab..6c48c46)
Task 3: complete (commits 5528479..6c48c46, review clean). Suite after: 1093 passed (+ known Tcl flake). Deferred minors: see Task 3 review line; inf in descriptors not checked.
- Task 4: implementer DONE (b95a480; reports 11 passed, bench 346, suite 1098/0). Pending task review.
Task 4: review clean (Approved, 0 Critical/Important). Minors deferred: CLI test does not read back baseline parquet/JSON contents; metrics_<family>/baseline JSON not keyed by prep_id so stale files can sit beside a newer provenance.json (spec fixes file names; note in docs); imports mid-file in test_reports.py; no test for NaN->null in provenance; no test pinning "other ValueError propagates".
Task 4: complete (commits 6c48c46..b95a480, review clean). Suite after: 1098 passed.
- Task 5: implementer DONE (93e34b4; 6 new tests, suite 1103 + 1 env Tcl flake). Pending task review.
Task 5: review clean (Approved; rendered figures checked by reviewer). Minors deferred: spy test checks note text not image placement; source check keyed on first "if args.evaluate_shapes:"; note close to "Node ID" label with 30+ nodes.
Task 5: complete (commits b95a480..93e34b4, review clean). Suite after: 1103 passed (+ known Tcl flake).
- Task 6: implementer DONE (57dc170; tests/scripts 6 passed, suite 1107; data/ clean verified). Pending task review.
Task 6: review clean (Approved). Minors deferred: loose `in` assertions in the halving test; main() CLI not tested; TOCTOU/hardlink on overwrite guard; count 0 gives no warning.
Task 6: complete (commits 93e34b4..57dc170, review clean). Suite after: 1107 passed (+ known Tcl flake). data/ verified clean.
- Task 7: implementer DONE (8f17147; 11 new tests; R2 applied lazily; verify_promotion('outputs/models') = verified). Concern: some existing predict tests now emit UserWarning (fixture models w/o promotion.json). Pending task review.
Task 7: review: spec ❌ (figure provenance); 1 Important (plan defect): runner stamp reads promotion.json from data/networks/<net>/results/model_artifacts where it never exists -> always "desconocida"/null; figures cited by V-04 (main.py curves + factor_comparison) get no stamp. Minors/gaps: other loaders unguarded (ScenarioPredictor main.py:614/702/831, SHAP main.py:522, predict_from_inp via ml/train.py:522 incl. desktop generate_ml_map); UserWarning noise in 4 existing predict tests; third sha256 helper copy; malformed promotion.json -> raw JSONDecodeError; coarse per-directory stamp granularity.
Ruling: R8 — fix round 1: model_provenance falls back to SHA-256 of the real artifact files when no promotion.json; stamp main.py figure outputs; reuse bench.train._sha256; remove warning noise from 4 tests (introduced by this task). Other loaders stay unguarded and are documented as a known gap (spec 2.8 scopes predict.py). — why: spec intent (V-04: attribute figures to a model) is not met by the plan's runner-only stamp — cost if wrong: SHA of manifest+joblibs identifies files but not the training candidate/prep_id for legacy artifacts.
- Task 7: fix round 1/5 dispatched (implementer resumed)
Task 7: fix round 1/5 (all addressed, 0 open; commits 8f17147..0c8a3f4). Suite 1128 passed (implementer's captured count).
Task 7: complete (commits 57dc170..0c8a3f4, review clean). Known gap (documented, out of scope by R8): ScenarioPredictor (main.py:614/702/831), SHAP (main.py:522), predict_from_inp via ml/train.py:522 incl. desktop generate_ml_map load models without verify; malformed promotion.json/manifest.json raise raw JSONDecodeError.
Task 8: implementer NEEDS_CONTEXT (commit a26155e: scripts/informe_evaluacion.py + scripts/ejecutar_evaluacion_v4_21.py + 17 tests; suite 1145/0). Real run: prepare 88s (prep_id ca706e27654e0cf5, 154080 rows, 21 feat, 3 protocols), xgboost 193s OK; baseline_interpolacion_factor FAILED: after the 5% continuity filter only 963/1000 runs remain (37 (shape,factor) cells missing) -> series without all train factors.
Ruling: R9 — baselines must work on the incomplete grid of the evaluated dataset: interpolation per series over that series' own available train factors (one-sided -> nearest available); nearest-shape falls back to the next-nearest train shape that has the needed (node,factor) cell; error only if no train shape has it. (Task 3 follow-up, package code, tests with an incomplete grid.) Rejected alternative: re-running on unfiltered 160k rows (would not be comparable to the 56.8088 official figure). — why: baselines must be scored on exactly the rows XGBoost is scored on — cost if wrong: baseline numbers include a small amount of nearest-neighbour fallback where cells were removed by the continuity filter.
Ruling: R10 — Task 7 post-completion defect: pre-existing tests run main.py branches that now write provenance.json into the REAL outputs/metrics (false provenance for the 2026-10-03 figures). Fix the offending tests' isolation, delete the 2 stray generated files, add a conftest guard. — cost if wrong: none.
- Task 7: follow-up fix dispatched (implementer resumed); then Task 3 follow-up (R9); then resume Task 8 run.
Task 7: follow-up (R10) 398d6bc reviewed: all addressed, no new breakage (3 offending tests isolated, conftest guard, 2 stray files deleted; suite 1145/0). Minor: guard has no meta-test; empty dir outputs/metrics/factor_comparison/ left.
- Task 3: follow-up (R9, incomplete grid) dispatched (implementer resumed). Next: scoped review, then resume Task 8 run.
Ruling: R11 — accepted implementer's deviation: baseline `fallback_rows` counter is defined against the FOLD's train factor grid (not the full grid) so GroupKFold5's deliberately held-out factors do not count as fallback. — cost if wrong: counter semantics differ from the literal R9 text; documented in the module docstring.
- Task 3 follow-up R9: implementer DONE_WITH_CONCERNS (833760d; baselines 30 passed, suite 1154/0). Pending scoped review.
Task 3: follow-up R9 (incomplete grids) commits 398d6bc..5f0586d: reviewed (opus: correct vs naive reference, no leakage, 5.8s on 150k rows) + fix round for test strength; scoped re-review all addressed. Suite 1154/0. Deferred minors: nearest-shape duplicate/NaN messages untested; counters are totals across protocols (documented); assert vs -O.
- Task 8: resuming implementer for the real run (R9 baselines now handle incomplete grid; save per stage; record fallback counters)
Task 8: implementer DONE_WITH_CONCERNS after R9 (commits a26155e, 4a2a87c, 4717885, 395a2ee; suite 1159/0). Real run: XGBoost LOSHO mean RMSE 56.8088 == documented official figure; confusion sums == n_rows; interpolation LOSO RMSE 9.126/F1 0.9718; fallback 1.25% interp, 0.42% nearest-shape; oracle NSE LOSHO 0.8906 (was 0.8141 on v3_17); per-shape vol error -52.1%..+33.3% (was -63..+216); quasi_uniform_block 71.6% of SSE. Pending task review (diff file review-task8.diff).
Task 8: review: all numbers independently recomputed OK; 1 Important: provenance.json (commit 4a2a87c) does not describe the committed CSVs (regenerated later at 4717885; nothing versioned says so). Minors bundled into the fix (CSVs must be regenerated anyway): flood_threshold param unused/dead CLI option, band test vacuous, CRLF bytes, run counts (963/1000) in provenance. Deferred minors: main() orchestration closures untested; ensure_allowed_output docstring overclaims; informe main --out has no guard.
Ruling: R12 — bundle the table minors with the traceability fix. — why: CSVs are regenerated anyway and are the source of all thesis numbers — cost if wrong: none.
- Task 8: fix round 1/5 dispatched (implementer resumed)
Task 8: fix round 1/5 (all addressed, 0 open; commits 395a2ee..284927d). Suite 1165/0. Tables traceable: tables_provenance.json (commit e26fc5a), README, .gitattributes eol=lf.
Task 8: complete (commits 0c8a3f4..284927d for Task 8 files, review clean). Deferred minors: main() closures untested; run-script docstring overclaims ensure_allowed_output; informe main --out unguarded; runs/*.json lack new fields (disclosed in README); git_dirty ignores untracked.
Task 9: spike DONE_WITH_CONCERNS (commit 2035a51, doc 03-resolucion-de-la-etiqueta.md, trailer says Opus not Sonnet — note only). Headline: rpt == round(V_cont) (852/852 nodes <100 m3); label == V_cont>=0.5 (960/960); 21 label changes under rule (ii), 106 under (iii); .out integral at REPORT_STEP 3min worse than rpt (up to 9.3-9.5 m3); API stat is the only better source; 16.8 min SWMM / 1000 runs; path A numbers verified by controller (333/380 FN in band, F1 .9817->.9847). Recommendation A.
Task 9 review (sonnet, independent re-run on 3 new scenarios: 0 discrepancies): Needs fixes. Important: I1 D-04 correction over-claimed (3 of 72 re-simulated; 66I only 0.03), I2 unmeasured sentence, I3 scope of "exactamente/sin ruido", I4 regressor effect not evaluated, I5 incomplete list of earlier docs to correct, I6 H-01 as conditional hypothesis. Plus minors.
Ruling: R13 — spike author does a fix round that also applies dated correction notes to annex 03 / consolidated report / annex 01 / datos README (original text kept, notes appended). — why: the spike invalidates two audit statements (D-04 rounding-noise explanation; read-from-.out recommendation) — cost if wrong: notes can be reverted.
- Task 9: fix round 1/5 dispatched (author resumed, model fable). Note: an Opus reviewer died of an API session limit (429) before; re-run with sonnet.
Task 9: fix round 1/5 (867ce6e): re-review all addressed (I1-I6 + 9 minors); numbers recomputed by reviewer. Extra measured: 12 of 72 transitions re-simulated (7 drop >=0.3, 3 in [0.1,0.3), 2 <0.1); threshold bound (0.4988, 0.5002] over 2400 node-scenario cases (15 scenarios); regressor in [1,5] m3: median abs err 1.63, bias +2.57, 22% within +-0.5. Open minor edits for the docs pass: scenario count 17 not 15 (s06 pair adds 2) in header/notes; "<=0.5 m3 below 100 m3" -> "<=0.51" in §7.6 and annex 01 note (node 8C diff 0.508). Unsupported-but-hedged: §6.4 bias (regression to the mean), §6.2 FP closeness, §7.1 design conclusion, §3 wording.
Task 9: complete (commits 2035a51..867ce6e). Decision A/B pending from the human.
- Task 10: dispatching documentation writer.
- Task 10: docs DONE (0a6d1a6; suite 1164 + 1 Tk flake, re-run green = 1165). Follow-ups not applied: '13 escenarios' notes (auditoria-completa l.275, anexo 03 l.76, datos/README l.22) and '<=0,5' note in anexo 01 l.113. Next: final whole-branch review.
FINAL REVIEW (opus, whole branch code+tests): verdict "con reservas" / Ready to merge: With fixes. 0 Critical. Important 1: outputs/models/promotion.json is gitignored (.gitignore:53 outputs/models/*) while the joblibs are tracked -> in a clone predict_network only warns and figures are stamped "desconocida"; docs 02 §3 line ~104 and 04 line ~214 claim it is versioned/stops. Minors: (2) baseline ValueError aborts before write_reports (main.py:224-245) -> ranking lost after hours of compute; (3) baseline threshold uses config not manifest (main.py:228); (4) PROTOCOL_INTERPRETATION["GroupKFold5"] says "5 bloques" but folds hold out interleaved factors; (5) metrics.py module docstring still says classifier is averaged per fold; (6) spec 2.5/2.6 differ from code (provenance.json separate; footnote instead of title) + metrics_*.json of different runs can mix definitions; (7) test_in_sample_note reads Path("main.py") relative; (8) runner stamping untested; (9) two tests promise more than they check. Docs follow-ups (13 escenarios -> 17 notes; <=0,5 -> <=0,51 annex 01 note). All recomputed doc numbers match.
Ruling: R14 — one final fix dispatch (fresh implementer) for: Important 1 via `git add -f outputs/models/promotion.json` (file unchanged; SHA already verified) + docs wording; minors 2,3,4,5,6(spec text aligned to code, plus metrics_definition marker),7; docs follow-ups. Minors 8,9 deferred (can wait). — why: cheap, and affects what is published — cost if wrong: promotion.json becomes tracked; future promotions need `git add -f`.

---

## Appendix: Part 2 and language pass (2026-10-12 to 2026-10-20)

Continuation of the ledger above (the versioned copy stopped at R14). Rulings are decisions taken on behalf of the author.

## Parte 2 (2026-10-12)
Decisions by the author: path A for the label (no re-simulation/DB changes); XGBoost and MLP always presented, SVM as third candidate; this week: corrected SVM + real runs + deck update for 2026-10-13.
Ruling: R15 — Part 2 uses a compact spec (docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md) and per-task dispatch prompts instead of a full code-level plan document, to meet the deadline; same SDD loop (implement -> review -> fix). — cost if wrong: less pre-written detail; reviewed per task anyway.
Ruling: R16 — SVM classifier = Nystroem RBF + CalibratedClassifierCV(SGDClassifier hinge, sigmoid, cv=3) (an approximate SVM, named so); regressor = linear epsilon-insensitive SVR on the same map; family key stays `svm`. — cost if wrong: it is not an exact-kernel SVM; documented everywhere.
P2-Task1: SVM family implemented (52f3937; tests/ml/bench 392; suite 1187 passed). Real-data smoke LOSHO fold 0 (defaults): 11.8 s fit+predict, F1 0.8808, AUC 0.9789, routed RMSE 57.05. Note: `epsilon` passed to the classifier is dropped silently (as before), not an error. Pending review.
P2-Task1 review (sonnet): Approved conditioned. Important: Platt calibration neutralizes class_weight (probability mean 0.429 at spw=1 vs 0.426 at spw=8) -> SVM's 0.5 threshold ~ unweighted Bayes, XGBoost/MLP shift with scale_pos_weight => recall/F1 not directly comparable; AUC is the fairer comparison. Missing joblib round-trip test. Minors: descriptor loss inferred from presence of `epsilon` (fragile); calibration folds unshuffled over train rows (order-dependent, correlated rows); weak epsilon assertion.
Ruling: R17 — keep the Nystroem+hinge+Platt design and DECLARE the class-weight/threshold asymmetry (docstring + test + doc 06; report AUC alongside F1/recall; maybe PR-AUC). — cost if wrong: the SVM's F1/recall may look lower/higher for a calibration reason, not a modelling one.
- P2-Task1: fix round 1 dispatched (implementer resumed).
P2-Task1: fix round 1 (5a961da): docstring "Sesgo conocido" + calibration-split note, 3 tests (spw 1 vs 8 mean-proba shift <0.05 over 3 seeds, joblib round-trip clf+reg, descriptor limitation pinned). Suite 1201 passed. Complete (docs/tests-only fix; no scoped re-review — recorded).
Deck (a5bdf2c.. agent): scripts/armar_presentacion_avances_13_10.py + test, commit 0c39c80; 43 slides (37 base + 6 new: auditoría, protocolos, matriz de confusión, etiqueta, arquitectura del SVM, comparación justa); results slides are added only when docs/auditoria/correcciones/datos_parte2/comparacion_protocolos.csv exists. Controller viewed slides 39 and 42 rendered: OK.
P2-Task2/3 (99d8cc0 parallel folds opt-in in evaluate_candidate/run_study (protocols=, n_jobs=, fold_n_jobs=, fixed_params=); 8a06605 scripts/ejecutar_parte2_v4_21.py resumable stages + tables + auc_roc_agregado/variante columns): implementer DONE; suite 1247 passed; smoke OK. Deviation: XGBoost pinned n_jobs=1 (with OMP_NUM_THREADS=1 workers, n_jobs=4 in a child == n_jobs=1, differs from n_jobs=4 in parent) — documented in script docstring.
Ruling: R18 — real run launched at 2026-10-12 21:54 (3 processes: mlp --eval-jobs 40, xgboost/svm --eval-jobs 16) BEFORE the code review finished, because compute was idle and the deadline is tomorrow; any result-affecting defect found by the review => rerun the affected stage. N_TRIALS=20 for all three families, MLP epochs 200/batch 1024/1 thread. — cost if wrong: wasted compute.

Ruling: R19 — language pass scope: translate to English the documentation of this series (docs/auditoria, specs and plan), every user-visible message in non-test Python code (≈650 strings; tests that matched Spanish text were updated, no logic changed) and all slide text/notes of the deck; keep identifiers, CSV/JSON column names, file names and machine values (e.g. `estimacion_exploratoria_condicionada`, `referencia`) in Spanish because changing them is a logic change and the committed tables are hash-traced. Pre-existing documentation (≈50 Markdown files, ≈5,000 Spanish lines) was NOT translated. — cost if wrong: mixed-language identifiers in data files; older docs remain Spanish.
Ruling: R20 — the progress deck is regenerated into a NEW file (Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx) instead of the open 13-10 file (PowerPoint lock file present); the new file also replaces, in the output deck only, the Spanish-text pictures of base slides 2, 3, 4, 6, 11 and 15 with the English versions in docs/figures_en/; slides 5 and 34 stay as they were (pictures not reproducible from versioned data). — cost if wrong: two deck files to choose from.
Ruling: R21 — path A for the label (keep threshold/resolution, declare it as ≈0.5 m³ continuous) is recorded as the author's choice of 2026-10-12 ("dale deuna" to the recommendation); report 04 still words it as pending. — cost if wrong: if the author meant B, the dataset would need to be regenerated.
Ruling: R22 — SVM weakness explanations on slides/notes are worded as plausible causes, not attributable causes, because only the Platt/class_weight effect was measured (on a toy problem). — cost if wrong: none.
