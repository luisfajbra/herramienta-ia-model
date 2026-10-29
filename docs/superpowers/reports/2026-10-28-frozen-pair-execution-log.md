# Frozen A/B SWMM rerun execution log

## Frozen run decisions

- Execute the pre-registered 1,000 A and 975 B scenarios with the existing synthetic nodal hydrographs.
- Round source timestamps to the nearest minute, as approved, and preserve nodal peaks.
- Keep B frozen as the transfer test; do not use B targets for fitting or tuning.
- Keep the 5% continuity limit and configured drainage/coverage tolerances provisional. Record ineligible cases and exclude them from training.
- Preserve every run root; do not overwrite partial generations.

## Execution interruptions and recovery

The first execution root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T184001Z`, stopped after 464 A scenarios because Windows denied the atomic checkpoint replacement. Its partial artifacts remain preserved and are not eligible for staging.

The second root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T191042Z`, stopped after the checkpoint JSON had reached 487 completed scenarios while its `.tmp` held 488. No simulator process remained, both files were readable, neither was read-only, and promoting the temporary checkpoint with `os.replace` then succeeded. This confirms a transient Windows file-lock/sharing condition at checkpoint publication rather than a SWMM or hydraulic QA error.

The runner now retries checkpoint replacement after bounded delays when it receives `PermissionError`. A regression test injects two transient permission failures and verifies the third atomic replacement succeeds. The focused runner suite passes (9 tests). The 488-scenario root is preserved as an interrupted generation; a new, cleanly anchored pair run is required for the authoritative A/B manifests.

## Authoritative execution

The current authoritative root is `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z`, anchored before simulation at commit `4efc53a`. A completed all 1,000 scenarios with no technical errors. Recalculated QA currently excludes 134 runs (100 not drained under the frozen retry policy, 35 above the provisional 5% continuity limit, and one with both reasons); all 1,000 have finite coverage. The remaining 866 are provisionally eligible for A training, subject to successful manifest staging.

B remains in progress and sealed from model selection. Its initial checkpoint was 63/975; at 2026-10-29 01:32 UTC it had 268/975 results and 113 QA exclusions. These are operational counts, not final B results.

## Staging corrections found by the real A manifest

The first A staging attempt exposed a representational difference in the INP parser/writer: an empty optional `[TAGS]` section can be present in the source and omitted from the effective INP. Since the parsed section is empty, it carries no hydraulic objects or values. The hydraulic descriptor comparison now ignores only empty sections; a regression test covers this exact equivalence.

A second attempt exposed the distinct frozen event contracts. Synthetic shape events contain every source `FLOW` node with positive base peak. The `base` event preserves every source `FLOW` node, including 13 whose peak is zero. The verifier and feature constructor now accept exactly either set and reject any other node set. A regression fixture exercises a base event with a zero-peak `FLOW` node. The two staging suites pass: 43 tests.

The first two failed staging databases and their diagnostic artifacts are preserved and must not be reused. A third, fresh SQLite staging is underway with unique v3 manifest names. Until it completes and its final manifest is verified, A is not yet a prepared training dataset. No A model fitting or hyperparameter search has started; B targets have not been read.

## Model-selection boundary for the sealed A-to-B test

Because B is the frozen external transfer test, its labels cannot be used in `LeaveOneNetworkOut` tuning. That evaluator would train one direction with B labels when scoring A and would compromise the approved A-only-to-B claim. The approved training budget was estimated for five inner folds, so the A-only studies will use `GroupKFold(5)` grouped by `shape_id`, with 50 trials, `macro_network_rmse`, and the configured 200 MLP epochs. With only network A in these studies, the macro objective equals A's RMSE; the metric name and one-network limitation will remain explicit in metadata. The five-seed MLP ensemble is fitted only after its A-only study selects parameters. B is scored once after prediction hashes are sealed. This estimates transfer to this one held-out network, not performance over a population of unseen networks.

The `--bench-tune` CLI previously omitted both the configured tuning objective and the fold protocol when it called `run_study`, silently selecting the function's default `mean_inner_fold_rmse`. A regression test reproduced the omission. The CLI now passes the configured objective (falling back to the legacy mean-fold objective only when configuration omits one) and explicitly records `GroupKFold`; the focused CLI tuning tests pass.

## A staging and preparation completed

The third fresh A staging completed successfully. It records all 1,000 `COMPLETE` runs; 866 are eligible and 134 are excluded (100 `drainage_not_drained`, 35 `continuity_limit_exceeded`, with one run in both reason groups). The SQLite contains 138,560 eligible feature rows and exactly 138,560 target rows. No target is stored for the excluded runs. The final source manifest SHA-256 is `4bf0c50f61b2287ab8510369e46f26495393104d0fe670ac6ab444df28c055a6`; the verified label-free source manifest is `6b7e7ece058a9e91e708224fe48339afe039a506d84f8327ae53dc4eaa3b42e5`.

PreparedDataset `813323e3ef75b752` contains those 138,560 rows from A and the `tabular_v4_21` contract. Its `GroupKFold5` split integrity was reloaded and checked: all five folds have zero `shape_id` overlap between train and validation. The frozen A protocol has 40 groups (39 synthetic forms plus `base`); QA left no eligible run for `base`, `long_moderate`, `shape_2h`, or `shape_5h`. Ten other groups have between 16 and 24 eligible factors rather than all 25. The training data therefore represent 36/40 groups, with no manual repair or target substitution.

Selection is preregistered in `A/model_selection_Aonly_813323e3ef75b752/selection_preregistration_v2.json` (SHA-256 `6d5b729e33bd11622c2aba70063facdef4e9997c2dffb2aee7dcc8e54b64fd77`). The earlier v1 file is preserved but superseded because its `excluded_run_ids` field held exclusion records rather than integer IDs; the corrected v2 was sealed before either study began. XGBoost completed 50 A-only trials with GroupKFold(5) and `macro_network_rmse`. The best result was 73.78205679320975 at trial 13. Its study JSON is `A/model_selection_Aonly_813323e3ef75b752/studies/xgboost/813323e3ef75b752_20261028T224207Z/study.json` (SHA-256 `5b3ace9ece0223b86c538c63e56ccaff60f6ff37de8e0221021adca57764cd93`). The fitted candidate at `A/candidates/xgboost` records the study and preregistration hashes; model SHA-256 values are classifier `a1a6e504cca9a4553e93edeed3340da48c70b5b74167716c4bc96946f376046f` and regressor `045526a91a4a99a89bde1f473bfc4e67eefa950c5baacf835dcc595c5ecc1b7f`. Candidate provenance reload verification passed. The 50-trial MLP search is running over the same A-only `PreparedDataset`, GroupKFold(5), seed 42, `macro_network_rmse`, 200 epochs, and batch size 256. Trials 0–5 have completed with values 359.0459, 129.1288, 208.1512, 152.8342, 201.8404, and 123.2106; trial 5 is the current best. Trial wall times have ranged from about 22 to 40 minutes. These are inner-search observations, not selected-model estimates. Neither study uses B targets.

## Source-bound A-to-B scoring gate

Task 5 previously exposed only a synthetic scoring helper; the public scorer rejected every real run. `score_transfer` now accepts only a verified B staging and SQLite connection, verifies the frozen source manifest, artifact links, candidate, prediction and input seals, and feature rows, and replays the registered candidate to require identical predictions before reading targets. It then re-extracts labels in memory from each eligible run's exact hash-verified RPT. It checks the report hash against hydraulic QA and the flood threshold, requires exact prediction/label run-node coverage, and writes aggregate overall, by-shape, and by-scenario metrics without saving B label rows or inserting them into SQLite. Callers cannot pass an arbitrary label frame. After adding the replay check, the label-free staging and frozen-transfer suites passed (55 tests).

## Weekly presentation prepared (2026-10-28; initial English version superseded)

Created `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia.pptx` using the 2026-10-20 weekly deck's 16:9 theme and visual style. It reports A's 866 eligible runs and 138,560 rows; the XGBoost A-only search result (73.78205679320975 m³, Optuna trial 13); MLP and B as still running; and the frozen transfer/scoring boundary. The operational B snapshot when generated was 321 QA results, 114 exclusions and 540 pending scenarios. This count is time-dependent, not an evaluation metric.

The deck does not reuse the prior benchmark's parity plots or flood maps as current findings. It explains that current B visuals must be generated only after MLP completion, prediction sealing and verified B scoring. The generated deck passed ZIP integrity and `python-pptx` reopen checks (9 slides). SHA-256: `f357d3e60a34676d4f411c187ea465b4144d6ac4cce7861db26beaa8613129a1`. Rebuild command: `venv/Scripts/python.exe scripts/build_weekly_2026_10_28_deck.py`.

## Spanish style revision (2026-10-28)

The first presentation was open in PowerPoint, so it was left intact and a revised file was generated as `Presentacion_MLSMFP_28-10_avances_auditoria_transferencia_v2.pptx`. The revision translates the deck into Spanish, samples the navy/blue/gold palette from the 20-10 slides, and redistributes the content across a summary, QA, evaluation design, XGBoost result, model/B status, visual-readiness, scope and traceability slides. It passed ZIP integrity and `python-pptx` reopen checks (9 slides). The B progress snapshot in this version is 324 QA results, 114 exclusions and 537 pending scenarios; it is operational and time-dependent. SHA-256: `b72f494918c2b88c8da3a9eaa0a68f437ce29ff78e72a9ac4a77e0b7c4d0e078`.
